"""Fuel-limited detumbling by bounded external thruster torque."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol
import numpy as np

from orbexa.simulation.attached.assembly.model import AttachedAssembly, AssemblyState, positive
from orbexa.simulation.attached.propulsion import ThrusterSystem


class DetumblingPolicy(Protocol):
    def torque_body_nm(self, assembly: AttachedAssembly, state: AssemblyState) -> np.ndarray: ...


@dataclass(frozen=True)
class RateDampingPolicy:
    """Dissipative external torque request, before physical allocation."""
    gain_nm_s: float = 20.0
    compensate_gravity_gradient: bool = False

    def __post_init__(self):
        object.__setattr__(self, "gain_nm_s", positive(self.gain_nm_s, "Rate damping gain"))
        if not isinstance(self.compensate_gravity_gradient, bool):
            raise TypeError("Gravity-gradient compensation must be a boolean")

    def torque_body_nm(self, assembly, state):
        torque = -self.gain_nm_s * state.angular_velocity_body_rad_s
        if self.compensate_gravity_gradient:
            torque -= assembly.gravity_gradient_torque_body(state)
        return torque


@dataclass(frozen=True)
class DetumblingOptions:
    rate_tolerance_rad_s: float = 1e-5
    dwell_s: float = 30.0
    max_duration_s: float = 1200.0
    control_step_s: float = 1.0

    def __post_init__(self):
        for name in vars(self):
            object.__setattr__(self, name, positive(getattr(self, name), name))
        if self.dwell_s > self.max_duration_s:
            raise ValueError("Detumbling dwell must fit the allowed duration")


@dataclass(frozen=True)
class DetumblingResult:
    success: bool
    message: str
    final_state: AssemblyState
    history: list[dict]
    controls: list[dict]
    audit: dict

    def to_dict(self):
        return dict(success=self.success, message=self.message, final_state=self.final_state.to_dict(),
                    history=self.history, controls=self.controls, audit=self.audit)


class DetumblingSimulator:
    def __init__(self, *, options: Optional[DetumblingOptions] = None,
                 policy: Optional[DetumblingPolicy] = None):
        self.options = options or DetumblingOptions()
        self.policy = policy or RateDampingPolicy()

    def run(self, assembly: AttachedAssembly, initial: AssemblyState,
            propulsion: ThrusterSystem, *, observer=None) -> DetumblingResult:
        options, state = self.options, initial
        history, controls, refused_allocations = [assembly.state_record(state)], [], []
        authority = propulsion.authority(assembly, state.fuel_kg)
        impulse, flux = np.zeros(3), np.zeros(3)
        dwell, max_balance_error, saturated_steps = 0., 0., 0
        initial_h = assembly.angular_momentum_eci(initial)
        observations = []
        refused_holds = []
        if initial.enclosure is not None and observer is None:
            from orbexa.simulation.attached.assembly.observation import BoundedAssemblyObserver
            observer = BoundedAssemblyObserver()

        def finish(success, message):
            residual = assembly.angular_momentum_eci(state) - initial_h - impulse - flux
            return DetumblingResult(success, message, state, history, controls, dict(
                elapsed_s=state.time_s - initial.time_s, observed_dwell_s=dwell,
                rate_tolerance_rad_s=options.rate_tolerance_rad_s,
                initial_rate_rad_s=float(np.linalg.norm(initial.angular_velocity_body_rad_s)),
                final_rate_rad_s=float(np.linalg.norm(state.angular_velocity_body_rad_s)),
                fuel_used_kg=float(np.sum(initial.fuel_kg - state.fuel_kg)), authority=authority,
                external_angular_impulse_eci_kg_m2_s=impulse.tolist(),
                fuel_corotation_angular_impulse_eci_kg_m2_s=flux.tolist(),
                angular_momentum_balance_residual_kg_m2_s=float(np.linalg.norm(residual)),
                max_step_angular_balance_residual_kg_m2_s=max_balance_error,
                saturated_steps=saturated_steps, refused_allocations=refused_allocations,
                refused_holds=refused_holds,
                observations=observations, observer=None if observer is None else observer.as_dict(),
                dwell_scope=('complete carried angular-rate swept sets during consecutive held intervals'
                    if state.enclosure is not None else 'consecutive controlled integration endpoints; no continuous-time enclosure')))

        if authority["torque_rank"] < 3:
            return finish(False, "detumbling_insufficient_three_axis_torque_authority")
        while state.time_s - initial.time_s < options.max_duration_s - 1e-10:
            feedback_state = state
            if observer is not None:
                try:
                    state, feedback_state, measurement = observer.update(assembly, state)
                except (ValueError, RuntimeError, ArithmeticError, ImportError) as error:
                    refused_holds.append(dict(time_s=state.time_s, applied=False, phase='observation',
                        exception_type=type(error).__name__, message=str(error)))
                    return finish(False, 'detumbling_observation_refused: '+str(error))
                observations.append(measurement)
            duration = min(options.control_step_s, options.max_duration_s - (state.time_s - initial.time_s))
            quiet_before = np.linalg.norm(state.angular_velocity_body_rad_s) <= options.rate_tolerance_rad_s
            try:
                torque = self.policy.torque_body_nm(assembly, feedback_state)
                allocation = propulsion.allocate(assembly, state, np.zeros(3), torque, duration)
            except (ValueError, RuntimeError, ArithmeticError, ImportError) as error:
                refused_holds.append(dict(time_s=state.time_s, duration_s=duration, applied=False,
                    phase='control_allocation', exception_type=type(error).__name__, message=str(error)))
                return finish(False, 'detumbling_control_refused: '+str(error))
            if not allocation.success:
                refused_allocations.append(dict(time_s=state.time_s, duration_s=duration,
                    requested_torque_body_nm=torque.tolist(), allocation=allocation.to_dict(), applied=False))
                return finish(False, f"detumbling_{allocation.message}")
            try:
                next_state, audit = assembly.advance_with_audit(state, duration, allocation.force_body_n,
                    allocation.torque_body_nm, allocation.fuel_flow_kg_s,
                    propulsion=propulsion, commands_n=allocation.commands_n)
            except (ValueError, RuntimeError, ArithmeticError, ImportError) as error:
                refused_holds.append(dict(time_s=state.time_s, duration_s=duration, applied=False,
                    allocation=allocation.to_dict(), exception_type=type(error).__name__, message=str(error)))
                return finish(False, 'detumbling_held_flow_refused: '+str(error))
            impulse += audit["external_angular_impulse_eci_kg_m2_s"]
            flux += audit["fuel_corotation_angular_impulse_eci_kg_m2_s"]
            max_balance_error = max(max_balance_error, audit["angular_momentum_balance_residual_kg_m2_s"])
            saturated_steps += int(allocation.saturated)
            controls.append(dict(time_s=state.time_s, duration_s=duration, requested_torque_body_nm=torque.tolist(),
                                 allocation=allocation.to_dict(), integration=audit))
            state = next_state
            history.append(assembly.state_record(state))
            if assembly.attachment_limits:
                try:
                    endpoint = assembly.attachment_load_audit(state, propulsion, allocation.commands_n)
                except (ValueError, RuntimeError, ArithmeticError, ImportError) as error:
                    refused_holds.append(dict(time_s=state.time_s, phase='accepted_endpoint_diagnostic',
                        proposed_hold_applied=True, exception_type=type(error).__name__, message=str(error)))
                    return finish(False, 'detumbling_endpoint_audit_refused: '+str(error))
                audit['attachment_load_endpoint'] = endpoint
                if not endpoint['passed']:
                    return finish(False, 'detumbling_attachment_capacity_exceeded')
            quiet = np.linalg.norm(state.angular_velocity_body_rad_s) <= options.rate_tolerance_rad_s
            if state.enclosure is not None:
                from fractions import Fraction
                angular = state.enclosure.last_swept.slice(10, 13)
                quiet = quiet and sum(max(abs(lo), abs(hi))**2 for lo, hi in
                    zip(angular.lower, angular.upper)) <= Fraction(options.rate_tolerance_rad_s)**2
            dwell = dwell + duration if quiet and quiet_before else 0.
            if dwell >= options.dwell_s:
                return finish(True, "detumbling_rate_dwell_completed")
            if np.sum(state.fuel_kg) <= 1e-10:
                return finish(False, "detumbling_fuel_exhausted_before_rate_dwell")
        return finish(False, "detumbling_duration_limit_before_rate_dwell")
