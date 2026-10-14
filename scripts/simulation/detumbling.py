"""Fuel-limited detumbling by bounded external thruster torque."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol
import numpy as np

from orbexa.simulation.assembly import AttachedAssembly, AssemblyState, positive
from orbexa.simulation.propulsion import ThrusterSystem


class DetumblingPolicy(Protocol):
    def torque_body_nm(self, assembly: AttachedAssembly, state: AssemblyState) -> np.ndarray: ...


@dataclass(frozen=True)
class RateDampingPolicy:
    """Dissipative external torque request, before physical allocation."""
    gain_nm_s: float = 20.0

    def __post_init__(self):
        object.__setattr__(self, "gain_nm_s", positive(self.gain_nm_s, "Rate damping gain"))

    def torque_body_nm(self, assembly, state):
        return -self.gain_nm_s * state.angular_velocity_body_rad_s


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
            propulsion: ThrusterSystem) -> DetumblingResult:
        options, state = self.options, initial
        history, controls = [assembly.state_record(state)], []
        authority = propulsion.authority(assembly, state.fuel_kg)
        impulse, flux = np.zeros(3), np.zeros(3)
        dwell, max_balance_error, saturated_steps = 0., 0., 0
        initial_h = assembly.angular_momentum_eci(initial)

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
                saturated_steps=saturated_steps, dwell_scope="consecutive controlled integration endpoints; no continuous-time enclosure"))

        if authority["torque_rank"] < 3:
            return finish(False, "detumbling_insufficient_three_axis_torque_authority")
        while state.time_s - initial.time_s < options.max_duration_s - 1e-10:
            duration = min(options.control_step_s, options.max_duration_s - (state.time_s - initial.time_s))
            quiet_before = np.linalg.norm(state.angular_velocity_body_rad_s) <= options.rate_tolerance_rad_s
            torque = self.policy.torque_body_nm(assembly, state)
            allocation = propulsion.allocate(assembly, state, np.zeros(3), torque, duration)
            if not allocation.success:
                return finish(False, f"detumbling_{allocation.message}")
            next_state, audit = assembly.advance_with_audit(state, duration, allocation.force_body_n,
                                                          allocation.torque_body_nm, allocation.fuel_flow_kg_s)
            impulse += audit["external_angular_impulse_eci_kg_m2_s"]
            flux += audit["fuel_corotation_angular_impulse_eci_kg_m2_s"]
            max_balance_error = max(max_balance_error, audit["angular_momentum_balance_residual_kg_m2_s"])
            saturated_steps += int(allocation.saturated)
            controls.append(dict(time_s=state.time_s, duration_s=duration, requested_torque_body_nm=torque.tolist(),
                                 allocation=allocation.to_dict(), integration=audit))
            state = next_state
            history.append(assembly.state_record(state))
            quiet = np.linalg.norm(state.angular_velocity_body_rad_s) <= options.rate_tolerance_rad_s
            dwell = dwell + duration if quiet and quiet_before else 0.
            if dwell >= options.dwell_s:
                return finish(True, "detumbling_rate_dwell_completed")
            if np.sum(state.fuel_kg) <= 1e-10:
                return finish(False, "detumbling_fuel_exhausted_before_rate_dwell")
        return finish(False, "detumbling_duration_limit_before_rate_dwell")
