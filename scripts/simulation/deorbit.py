"""Apoapsis finite-burn deorbit, terminating at a declared entry interface.

This propagates central-gravity ECI motion, the assembly's configured extra
environment, and the same fueled rigid body used for detumbling.
Entry-interface crossing is a handoff to an atmospheric entry model;
breakup, demise, thermal protection and ground impact are outside this model.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Optional
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation

from orbexa.core.attitude import finite_vector
from orbexa.simulation.assembly import AttachedAssembly, AssemblyState, STANDARD_GRAVITY, positive
from orbexa.simulation.propulsion import ThrusterSystem
from orbexa.simulation.deorbit_entry import EntryInterfaceMonitor, radial_status
from orbexa.simulation.burn_schedule import ApoapsisBurnWindow, supported_planning_state, bounded_hold_duration
from orbexa.simulation.stage_recording import StageRecordRetainer


def _rate_within(state, limit):
    if state.enclosure is None:
        return bool(np.linalg.norm(state.angular_velocity_body_rad_s) <= limit)
    angular = state.enclosure.box.slice(10, 13)
    return sum(max(abs(lo), abs(hi))**2 for lo, hi in zip(angular.lower, angular.upper)) <= Fraction(float(limit))**2


def orbital_elements(position_eci_m, velocity_eci_m_s, mu_m3_s2=3.986004418e14):
    """Osculating two-body invariants from an actual Cartesian state."""
    r = finite_vector(position_eci_m, "Orbital position", 3)
    v = finite_vector(velocity_eci_m_s, "Orbital velocity", 3)
    mu = positive(mu_m3_s2, "Gravitational parameter")
    radius = np.linalg.norm(r)
    if radius == 0:
        raise ValueError("Orbital radius cannot be zero")
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)
    if h_norm == 0:
        raise ValueError("Radial degenerate orbits are unsupported")
    eccentricity_vector = np.cross(v, h) / mu - r / radius
    e = float(np.linalg.norm(eccentricity_vector))
    energy = float(np.dot(v, v) / 2 - mu / radius)
    axis = None if energy == 0 else -mu / (2 * energy)
    p = h_norm**2 / mu
    apogee = p / (1 - e) if energy < 0 and e < 1 else None
    return dict(specific_energy_j_kg=energy, specific_angular_momentum_m2_s=h_norm,
                angular_momentum_direction_eci=(h / h_norm).tolist(), eccentricity=e,
                eccentricity_vector_eci=eccentricity_vector.tolist(), semi_major_axis_m=axis,
                perigee_radius_m=float(p / (1 + e)), apogee_radius_m=apogee,
                radius_m=float(radius), radial_velocity_m_s=float(np.dot(r, v) / radius))


def time_to_apoapsis_s(position, velocity, mu):
    elements = orbital_elements(position, velocity, mu)
    a, e = elements["semi_major_axis_m"], elements["eccentricity"]
    if a is None or a <= 0 or e >= 1:
        raise ValueError("Apoapsis maneuver requires a bound elliptic orbit")
    if e < 1e-8:
        return 0.
    radius = np.linalg.norm(position)
    cosine = np.clip((1 - radius / a) / e, -1, 1)
    sine = np.dot(position, velocity) / (e * np.sqrt(mu * a))
    eccentric_anomaly = np.arctan2(sine, cosine)
    mean_anomaly = eccentric_anomaly - e * np.sin(eccentric_anomaly)
    delta = float((np.pi - mean_anomaly) % (2 * np.pi))
    if 2 * np.pi - delta < 1e-9:
        delta = 0.
    return delta / np.sqrt(mu / a**3)


@dataclass(frozen=True)
class ThrustPointingPolicy:
    attitude_gain_nm: float = .2
    rate_gain_nm_s: float = 20.0
    compensate_gravity_gradient: bool = False
    thrust_axis_body: tuple[float, float, float] = (1., 0., 0.)

    def __post_init__(self):
        object.__setattr__(self, "attitude_gain_nm", positive(self.attitude_gain_nm, "Pointing attitude gain"))
        object.__setattr__(self, "rate_gain_nm_s", positive(self.rate_gain_nm_s, "Pointing rate gain"))
        if not isinstance(self.compensate_gravity_gradient, bool):
            raise TypeError("Gravity-gradient compensation must be a boolean")
        axis = finite_vector(self.thrust_axis_body, 'Body thrust axis', 3)
        scale = np.max(np.abs(axis))
        if scale == 0:
            raise ValueError('Body thrust axis must be nonzero')
        scaled_axis = axis / scale
        normalized = scaled_axis / np.linalg.norm(scaled_axis)
        if not np.all(np.isfinite(normalized)):
            raise ValueError('Body thrust axis must normalize to a finite unit vector')
        object.__setattr__(self, 'thrust_axis_body', tuple(map(float, normalized)))

    def with_thrust_axis(self, axis):
        return replace(self, thrust_axis_body=tuple(axis))

    def command(self, assembly, state):
        r, v = state.position_eci_m, state.velocity_eci_m_s
        radial = r / np.linalg.norm(r)
        normal = np.cross(r, v)
        normal /= np.linalg.norm(normal)
        retrograde = -(v - np.dot(v, radial) * radial)
        retrograde /= np.linalg.norm(retrograde)
        desired = np.column_stack((retrograde, np.cross(normal, retrograde), normal))
        axis = np.asarray(self.thrust_axis_body)
        if self.thrust_axis_body != (1., 0., 0.):
            auxiliary = np.array([0., 0., 1.]) if abs(axis[2]) < .9 else np.array([0., 1., 0.])
            tangent = np.cross(auxiliary, axis)
            tangent /= np.linalg.norm(tangent)
            body_frame = np.column_stack((axis, tangent, np.cross(axis, tangent)))
            desired = desired @ body_frame.T
        current = Rotation.from_quat(state.quaternion_body_to_eci).as_matrix()
        error = Rotation.from_matrix(current.T @ desired).as_rotvec()
        desired_omega_eci = np.cross(r, v) / np.dot(r, r)
        desired_omega_body = current.T @ desired_omega_eci
        omega = state.angular_velocity_body_rad_s
        inertia = assembly.mass_properties(state.fuel_kg)[2]
        feedforward = inertia @ (current.T @ (-2 * np.dot(r, v) / np.dot(r, r) * desired_omega_eci))
        torque = (self.attitude_gain_nm * error - self.rate_gain_nm_s * (omega - desired_omega_body)
                  + np.cross(omega, inertia @ omega) + feedforward)
        if self.compensate_gravity_gradient:
            torque -= assembly.gravity_gradient_torque_body(state)
        pointed_axis = current[:, 0] if self.thrust_axis_body == (1., 0., 0.) else current @ axis
        pointing = float(np.arccos(np.clip(np.dot(pointed_axis, retrograde), -1, 1)))
        return torque, pointing, float(np.linalg.norm(omega - desired_omega_body))


@dataclass(frozen=True)
class DeorbitOptions:
    target_perigee_altitude_m: float = 80000.0
    entry_interface_altitude_m: float = 120000.0
    control_step_s: float = 2.0
    coast_sample_step_s: float = 30.0
    alignment_lead_s: float = 600.0
    max_alignment_s: float = 1800.0
    max_burn_s: float = 7200.0
    max_coast_orbits: float = 2.0
    pointing_tolerance_rad: float = .02
    pointing_rate_tolerance_rad_s: float = .0005
    max_initial_rate_rad_s: float = .00005
    max_body_rate_rad_s: float = .1
    burn_force_n: Optional[float] = None
    coast_rtol: float = 1e-10
    burn_window_half_angle_rad: Optional[float] = None

    def __post_init__(self):
        ApoapsisBurnWindow(self.burn_window_half_angle_rad)
        for name, value in vars(self).items():
            if value is not None:
                object.__setattr__(self, name, positive(value, name, allow_zero=name == "target_perigee_altitude_m"))
        if self.target_perigee_altitude_m >= self.entry_interface_altitude_m:
            raise ValueError("Target perigee must lie below the declared entry interface")


@dataclass(frozen=True)
class DeorbitResult:
    success: bool
    message: str
    final_state: AssemblyState
    history: list[dict]
    controls: list[dict]
    audit: dict

    def to_dict(self):
        return dict(success=self.success, message=self.message, final_state=self.final_state.to_dict(),
                    history=self.history, controls=self.controls, audit=self.audit)


class DeorbitSimulator:
    def __init__(self, *, options: Optional[DeorbitOptions] = None,
                 pointing_policy: Optional[ThrustPointingPolicy] = None, burn_axis_planner=None,
                 burn_throttle_planner=None):
        self.options = options or DeorbitOptions()
        self.pointing_policy = pointing_policy or ThrustPointingPolicy()
        self.burn_axis_planner = burn_axis_planner
        self.burn_throttle_planner = burn_throttle_planner

    def _record(self, assembly, state, phase):
        record = assembly.state_record(state)
        elements = orbital_elements(state.position_eci_m, state.velocity_eci_m_s, assembly.mu_m3_s2)
        record.update(phase=phase, orbit=elements,
                      altitude_m=elements["radius_m"] - assembly.earth_radius_m,
                      perigee_altitude_m=elements["perigee_radius_m"] - assembly.earth_radius_m)
        return record

    def _coast(self, assembly, initial, duration, history, phase, *, stop_at_entry=True, propulsion=None, observer=None,
               entry_monitor=None, record_retainer=None):
        if duration <= 1e-9:
            return initial, False, dict(relative_energy_drift=0., relative_angular_momentum_drift=0.)
        if initial.enclosure is not None:
            return self._carried_coast(assembly, initial, duration, history, phase,
                stop_at_entry=stop_at_entry, propulsion=propulsion, observer=observer, entry_monitor=entry_monitor, record_retainer=record_retainer)
        mass, _, inertia = assembly.mass_properties(initial.fuel_kg)
        values = np.r_[initial.position_eci_m, initial.velocity_eci_m_s, initial.quaternion_body_to_eci,
                       initial.angular_velocity_body_rad_s]

        def rhs(elapsed, state):
            return assembly.derivative(state, mass, inertia, np.zeros(3), np.zeros(3),
                time_s=initial.time_s+elapsed, fuel_kg=initial.fuel_kg)[0]

        def entry(_, state):
            return np.linalg.norm(state[:3]) - assembly.earth_radius_m - self.options.entry_interface_altitude_m

        entry.terminal, entry.direction = True, -1
        solution = solve_ivp(rhs, (0., duration), values, method="DOP853", rtol=self.options.coast_rtol,
            atol=np.r_[np.full(3, 1e-4), np.full(3, 1e-7), np.full(4, 1e-12), np.full(3, 1e-13)],
            max_step=self.options.coast_sample_step_s, dense_output=True, events=entry if stop_at_entry else None)
        if not solution.success or solution.sol is None or not np.all(np.isfinite(solution.y)):
            raise RuntimeError(f"Deorbit coast integration failed: {solution.message}")
        end = float(solution.t[-1])
        samples = np.r_[np.arange(self.options.coast_sample_step_s, end, self.options.coast_sample_step_s), end]
        for sample in samples:
            value = solution.sol(sample)
            state = AssemblyState(initial.time_s + sample, value[:3], value[3:6], value[6:10], value[10:13], initial.fuel_kg)
            history.append(self._record(assembly, state, phase))
        before = orbital_elements(initial.position_eci_m, initial.velocity_eci_m_s, assembly.mu_m3_s2)
        after = orbital_elements(state.position_eci_m, state.velocity_eci_m_s, assembly.mu_m3_s2)
        drift = dict(relative_energy_drift=abs(after["specific_energy_j_kg"] / before["specific_energy_j_kg"] - 1),
                     relative_angular_momentum_drift=abs(after["specific_angular_momentum_m2_s"] / before["specific_angular_momentum_m2_s"] - 1),
                     scope=('Physical osculating-element change includes the injected environment; not an isolated numerical conservation error'
                            if assembly.environment is not None else 'Two-body coast conservation diagnostic'))
        crossed = bool(stop_at_entry and len(solution.t_events[0]) > 0)
        return state, crossed, drift

    def _carried_coast(self, assembly, initial, duration, history, phase, *, stop_at_entry, propulsion, observer=None,
                       entry_monitor=None, record_retainer=None):
        """Zero-nozzle coasts retain the same admitted set and joint-load gates."""
        if propulsion is None:
            raise ValueError('A carried deorbit coast requires the actual installed propulsion declaration')
        records = record_retainer or StageRecordRetainer()
        commands = np.zeros(len(propulsion.thrusters))
        zero, flow = np.zeros(3), np.zeros(len(assembly.elements))
        state, elapsed, crossed = initial, 0., False
        entry_radius = assembly.earth_radius_m+self.options.entry_interface_altitude_m
        monitor = entry_monitor or EntryInterfaceMonitor(entry_radius)
        def conclude(refusal=None):
            before = orbital_elements(initial.position_eci_m, initial.velocity_eci_m_s, assembly.mu_m3_s2)
            after = orbital_elements(state.position_eci_m, state.velocity_eci_m_s, assembly.mu_m3_s2)
            result = dict(relative_energy_drift=abs(after['specific_energy_j_kg']/before['specific_energy_j_kg']-1),
                relative_angular_momentum_drift=abs(after['specific_angular_momentum_m2_s']/before['specific_angular_momentum_m2_s']-1),
                carried_state_enclosure=state.enclosure.as_dict(),
                entry_interface_certificate=monitor.certificate,
                scope='Carried whole-state and whole-hold zero-command load inclusion; successful entry requires a continuous whole-set descending above-to-below bracket')
            if refusal is not None:
                result['refusal'] = dict(time_s=state.time_s, exception_type=type(refusal).__name__,
                    message=str(refusal), proposed_hold_applied=False, accepted_elapsed_s=elapsed)
            return state, crossed, result
        while elapsed < duration-1e-9:
            measurement = None
            try:
                if observer is not None:
                    state, _, measurement = observer.update(assembly, state)
                hold = min(assembly.max_integration_step_s, self.options.coast_sample_step_s, duration-elapsed)
                loads = assembly.attachment_load_enclosure(state, propulsion, commands, hold,
                    capacity_fraction=propulsion.attachment_capacity_fraction)
                if not loads.passed:
                    raise ValueError('Carried zero-thrust coast loads not proven: '+loads.reason)
                next_state, integration = assembly.advance_with_audit(state, hold, zero, zero, flow,
                    propulsion=propulsion, commands_n=commands, max_angular_rate_rad_s=self.options.max_body_rate_rad_s)
            except (ValueError, RuntimeError, ArithmeticError, ImportError) as error:
                return conclude(error)
            interface = monitor.interval(state, next_state, fuel_flow_kg_s=flow)
            crossed = bool(stop_at_entry and interface is not None)
            row = self._record(assembly, next_state, phase)
            summary = dict(row)
            row['integration'] = integration
            row['whole_hold_attachment_loads'] = loads.as_dict()
            if measurement is not None:
                row['observation_conditioning'] = measurement
            if interface is not None:
                row['entry_interface_certificate'] = interface
            history.append(records.keep('coast_history', row, summary))
            state, elapsed = next_state, elapsed+hold
            if crossed:
                break
        return conclude()

    def run(self, assembly: AttachedAssembly, initial: AssemblyState, propulsion: ThrusterSystem, *, observer=None,
            record_sink=None) -> DeorbitResult:
        options, state = self.options, initial
        records = StageRecordRetainer(record_sink)
        burn_window = ApoapsisBurnWindow(options.burn_window_half_angle_rad)
        observations = []
        feedback_state = state
        if initial.enclosure is not None:
            if observer is None:
                from orbexa.simulation.assembly_observation import BoundedAssemblyObserver
                observer = BoundedAssemblyObserver()
            state, feedback_state, measurement = observer.update(assembly, state)
            observations.append(measurement)
        history, controls, coasts = [self._record(assembly, initial, "deorbit_planning")], [], []
        entry_monitor = (EntryInterfaceMonitor(assembly.earth_radius_m+options.entry_interface_altitude_m)
                         if initial.enclosure is not None else None)
        assembly_model = assembly.to_dict()
        audit = dict(entry_interface_altitude_m=options.entry_interface_altitude_m,
                     target_perigee_altitude_m=options.target_perigee_altitude_m,
                     terminal_scope="descending entry-interface handoff; no below-interface entry, heating, breakup, demise or impact propagation",
                     dynamics=("central-gravity Cartesian ECI and controlled rigid-body attitude; "
                         + ("configured central gravity-gradient torque; " if assembly.gravity_gradient else "gravity-gradient torque disabled; ")
                         + ("supplied per-component extra acceleration and intrinsic torque" if assembly.environment is not None else "no extra environment callback")),
                     gravity_gradient=assembly_model['gravity_gradient'],
                     environment=assembly_model['environment'],
                     environment_contract=assembly_model['environment_contract'],
                     initial_orbit=history[0]["orbit"])
        audit['burn_window'] = burn_window.as_dict()
        thrust_delta_v, saturated_steps, max_balance = 0., 0, 0.
        force_residual, torque_residual, burn_time = 0., 0., 0.

        def finish(success, message):
            if entry_monitor is not None:
                audit['entry_interface_certificate'] = entry_monitor.certificate
                if success and entry_monitor.certificate is None:
                    success, message = False, 'deorbit_whole_set_entry_interface_not_proven'
            final = orbital_elements(state.position_eci_m, state.velocity_eci_m_s, assembly.mu_m3_s2)
            audit.update(elapsed_s=state.time_s - initial.time_s, fuel_used_kg=float(np.sum(initial.fuel_kg - state.fuel_kg)),
                final_orbit=final, final_altitude_m=final["radius_m"] - assembly.earth_radius_m,
                target_perigee_achieved=bool(final["perigee_radius_m"] <= assembly.earth_radius_m + options.target_perigee_altitude_m + .01),
                final_radial_velocity_m_s=final["radial_velocity_m_s"], burn_time_s=burn_time,
                integrated_thrust_delta_v_m_s=thrust_delta_v, saturated_steps=saturated_steps,
                maximum_force_allocation_residual_n=force_residual, maximum_torque_allocation_residual_nm=torque_residual,
                max_step_angular_balance_residual_kg_m2_s=max_balance, coast_invariant_audits=coasts)
            audit.update(observer=None if observer is None else observer.as_dict(), observations=observations,
                accepted_record_storage=records.as_dict())
            return DeorbitResult(success, message, state, history, controls, audit)

        if not _rate_within(state, options.max_initial_rate_rad_s):
            return finish(False, "deorbit_requires_detumbled_initial_state")
        initial_elements = orbital_elements(feedback_state.position_eci_m, feedback_state.velocity_eci_m_s,
                                            assembly.mu_m3_s2)
        audit['sensed_planning_orbit'] = initial_elements
        if state.enclosure is not None and not radial_status(state.enclosure.box,
                assembly.earth_radius_m+options.entry_interface_altitude_m)['entirely_above']:
            return finish(False, 'deorbit_initial_whole_state_not_above_entry_interface')
        if initial_elements["radius_m"] <= assembly.earth_radius_m + options.entry_interface_altitude_m:
            return finish(False, "deorbit_initial_state_already_below_entry_interface")
        if initial_elements["apogee_radius_m"] is None:
            return finish(False, "deorbit_requires_bound_elliptical_orbit")
        entry_radius = assembly.earth_radius_m + options.entry_interface_altitude_m
        audit["initial_orbit_entry_intersecting"] = bool(initial_elements["perigee_radius_m"] < entry_radius)
        if audit["initial_orbit_entry_intersecting"]:
            # Interface delivery is already reachable by passive orbital motion.
            # The optional lower perigee target must not impose a fictitious
            # propellant or actuator requirement on this zero-thrust path.
            period = 2 * np.pi * np.sqrt(initial_elements["semi_major_axis_m"]**3 / assembly.mu_m3_s2)
            audit["burn_cutoff_reason"] = "initial_orbit_intersects_entry_interface"
            state, crossed, drift = self._coast(assembly, state, options.max_coast_orbits * period,
                history, "coast_to_entry_interface", propulsion=propulsion, observer=observer, entry_monitor=entry_monitor, record_retainer=records)
            coasts.append(drift)
            if drift.get('refusal'):
                return finish(False, 'deorbit_coast_not_proven: '+drift['refusal']['message'])
            if crossed and np.dot(state.position_eci_m, state.velocity_eci_m_s) < 0:
                final_perigee = orbital_elements(state.position_eci_m, state.velocity_eci_m_s,
                                                assembly.mu_m3_s2)["perigee_radius_m"]
                if final_perigee > assembly.earth_radius_m + options.target_perigee_altitude_m + .01:
                    return finish(True, "deorbit_entry_interface_reached_perigee_target_not_achieved")
                return finish(True, "deorbit_descending_entry_interface_handoff_completed")
            return finish(False, "deorbit_entry_interface_not_reached_within_coast_limit")
        ra = initial_elements["apogee_radius_m"]
        rp = assembly.earth_radius_m + options.target_perigee_altitude_m
        current_apo_speed = initial_elements["specific_angular_momentum_m2_s"] / ra
        transfer_axis = (ra + rp) / 2
        desired_apo_speed = np.sqrt(assembly.mu_m3_s2 * (2 / ra - 1 / transfer_axis))
        impulsive = max(0., float(current_apo_speed - desired_apo_speed))
        mass = assembly.mass_properties(state.fuel_kg)[0]
        best_isp = max(thruster.specific_impulse_s for thruster in propulsion.thrusters)
        fuel_lower_bound = mass * (1 - np.exp(-impulsive / (best_isp * STANDARD_GRAVITY)))
        audit.update(apoapsis_impulsive_reference_delta_v_m_s=impulsive,
                     ideal_fuel_lower_bound_for_planned_strategy_kg=fuel_lower_bound,
                     available_fuel_kg=float(np.sum(state.fuel_kg)),
                     propulsion_authority=propulsion.authority(assembly, state.fuel_kg))
        if np.sum(state.fuel_kg) < fuel_lower_bound:
            return finish(False, "deorbit_propellant_budget_insufficient_for_planned_apoapsis_strategy")
        if audit["propulsion_authority"]["wrench_rank"] < 6:
            return finish(False, "deorbit_insufficient_six_axis_wrench_authority")
        apo_time = state.time_s + time_to_apoapsis_s(feedback_state.position_eci_m, feedback_state.velocity_eci_m_s, assembly.mu_m3_s2)
        coast_duration = max(0., apo_time - state.time_s - options.alignment_lead_s)
        state, crossed, drift = self._coast(assembly, state, coast_duration, history, "coast_to_apoapsis",
            propulsion=propulsion, observer=observer, entry_monitor=entry_monitor, record_retainer=records)
        coasts.append(drift)
        if drift.get('refusal'):
            return finish(False, 'deorbit_coast_not_proven: '+drift['refusal']['message'])
        if crossed:
            return finish(True, "descending_entry_interface_reached_during_coast")
        alignment_start, burn_start, burn_start_clock = state.time_s, None, None
        feedback_state = state
        if observer is not None:
            try:
                state, feedback_state, measurement = observer.update(assembly, state)
            except (ValueError, RuntimeError, ArithmeticError, ImportError) as error:
                audit['refused_observation'] = dict(time_s=state.time_s, applied=False,
                    exception_type=type(error).__name__, message=str(error))
                return finish(False, 'deorbit_observation_refused: '+str(error))
            observations.append(measurement)
        try:
            planning_state = supported_planning_state(state, feedback_state)
        except (ValueError, RuntimeError, ArithmeticError, ImportError) as error:
            audit['refused_planning_support'] = dict(time_s=state.time_s, applied=False,
                exception_type=type(error).__name__, message=str(error))
            return finish(False, 'deorbit_planning_support_refused: '+str(error))
        audit['burn_planning_support'] = dict(
            method='sensed_center_with_unchanged_carried_physical_posterior',
            physical_state_not_substituted=True, time_s=planning_state.time_s,
            carried_state_enclosure=None if planning_state.enclosure is None else planning_state.enclosure.as_dict(),
            observation_identity=None if not observations else observations[-1]['identity'])
        pointing_policy = self.pointing_policy
        axis = np.asarray(getattr(pointing_policy, 'thrust_axis_body', (1., 0., 0.)))
        plan = None
        if self.burn_axis_planner is not None:
            if not callable(getattr(pointing_policy, 'with_thrust_axis', None)):
                return finish(False, 'deorbit_pointing_policy_cannot_use_selected_axis')
            try:
                plan = self.burn_axis_planner.select(assembly, planning_state, propulsion, options.control_step_s)
            except (ValueError, RuntimeError, ArithmeticError, ImportError) as error:
                audit['refused_axis_plan'] = dict(time_s=state.time_s, applied=False,
                    exception_type=type(error).__name__, message=str(error))
                return finish(False, 'deorbit_axis_plan_refused: '+str(error))
            pointing_policy = pointing_policy.with_thrust_axis(plan.direction_body)
            axis = np.asarray(plan.direction_body)
            default_force = plan.commanded_force_n
            audit['burn_axis_plan'] = plan.to_dict()
        else:
            default_force = .65 * sum(thruster.max_force_n * abs(float(np.dot(thruster.direction_body, axis)))
                                      for thruster in propulsion.thrusters)
        burn_force = default_force if options.burn_force_n is None else options.burn_force_n
        if self.burn_throttle_planner is not None:
            if plan is None or options.burn_force_n is not None:
                return finish(False, 'deorbit_throttle_requires_coupled_axis_plan_without_force_override')
            try:
                throttle = self.burn_throttle_planner.select(assembly, planning_state, propulsion, plan, options)
            except (ValueError, RuntimeError, ArithmeticError, ImportError) as error:
                audit['refused_throttle_plan'] = dict(time_s=state.time_s, applied=False,
                    exception_type=type(error).__name__, message=str(error))
                return finish(False, 'deorbit_throttle_plan_refused: '+str(error))
            audit['finite_burn_plan'] = throttle
            if throttle['status'] != 'FINITE_BURN_PREDICTED':
                return finish(False, 'deorbit_no_admitted_finite_burn_preview')
            burn_force = throttle['commanded_force_n']
        audit.update(commanded_burn_force_n=burn_force, planned_apoapsis_time_s=apo_time,
                     thrust_axis_body=axis.tolist())
        while True:
            feedback_state = state
            measurement = None
            if observer is not None:
                try:
                    state, feedback_state, measurement = observer.update(assembly, state)
                except (ValueError, RuntimeError, ArithmeticError, ImportError) as error:
                    audit['refused_observation'] = dict(time_s=state.time_s, applied=False,
                        exception_type=type(error).__name__, message=str(error))
                    return finish(False, 'deorbit_observation_refused: '+str(error))
                observations.append(measurement)
            try:
                torque, pointing, tracking_rate = pointing_policy.command(assembly, feedback_state)
                window_request = burn_window.evaluate(feedback_state.position_eci_m,
                    feedback_state.velocity_eci_m_s, assembly.mu_m3_s2)
            except (ValueError, RuntimeError, ArithmeticError, ImportError) as error:
                audit['refused_control'] = dict(time_s=state.time_s, applied=False,
                    exception_type=type(error).__name__, message=str(error))
                return finish(False, 'deorbit_control_refused: '+str(error))
            aligned = pointing <= options.pointing_tolerance_rad and tracking_rate <= options.pointing_rate_tolerance_rad_s
            if burn_start is None and state.time_s >= apo_time and aligned:
                burn_start = state.time_s
                burn_start_clock = (state.enclosure.exact_time_s if state.enclosure is not None else Fraction(state.time_s))
                audit["burn_start_time_s"] = burn_start
                audit["burn_start_exact_time_s"] = str(burn_start_clock)
                audit["burn_start_orbit"] = orbital_elements(state.position_eci_m, state.velocity_eci_m_s, assembly.mu_m3_s2)
            if burn_start is None and state.time_s - alignment_start > options.max_alignment_s + max(0., apo_time - alignment_start):
                return finish(False, "deorbit_pointing_alignment_timeout")
            if not _rate_within(state, options.max_body_rate_rad_s):
                return finish(False, "deorbit_body_rate_limit_exceeded")
            duration = options.control_step_s
            if burn_start_clock is not None:
                current_clock = (state.enclosure.exact_time_s if state.enclosure is not None else Fraction(state.time_s))
                remaining = Fraction(options.max_burn_s)-(current_clock-burn_start_clock)
                duration = (bounded_hold_duration(duration, remaining) if state.enclosure is not None else
                    bounded_hold_duration(duration, remaining, current_time_s=state.time_s,
                        deadline_s=burn_start_clock+Fraction(options.max_burn_s)))
                if duration <= 0:
                    audit['burn_wall_budget'] = dict(exact_start_time_s=str(burn_start_clock),
                        exact_current_time_s=str(current_clock), exact_remaining_s=str(remaining),
                        proposed_hold_applied=False)
                    return finish(False, 'deorbit_finite_burn_duration_limit')
            fire = burn_start is not None and aligned and window_request['active']
            force = axis * (burn_force if fire else 0.)
            try:
                allocation = propulsion.allocate(assembly, state, force, torque, duration)
            except (ValueError, RuntimeError, ArithmeticError, ImportError) as error:
                audit.setdefault('refused_holds', []).append(dict(time_s=state.time_s, duration_s=duration,
                    applied=False, phase='allocation', exception_type=type(error).__name__, message=str(error)))
                return finish(False, 'deorbit_allocation_refused: '+str(error))
            if not allocation.success:
                audit.setdefault('refused_allocations', []).append(dict(time_s=state.time_s,
                    duration_s=duration, requested_force_body_n=force.tolist(),
                    requested_torque_body_nm=torque.tolist(), allocation=allocation.to_dict(), applied=False))
                return finish(False, f"deorbit_{allocation.message}")

            def propagate(dt):
                if dt <= 0:
                    return state
                return assembly.advance(state, dt, allocation.force_body_n, allocation.torque_body_nm,
                    allocation.fuel_flow_kg_s, propulsion=propulsion, commands_n=allocation.commands_n,
                    max_angular_rate_rad_s=options.max_body_rate_rad_s)

            def predict(dt):
                if dt <= 0:
                    return feedback_state
                estimate = replace(feedback_state, enclosure=None)
                return assembly.advance(estimate, dt, allocation.force_body_n, allocation.torque_body_nm,
                    allocation.fuel_flow_kg_s)

            try:
                next_state = propagate(duration)
                predicted = predict(duration)
            except (ValueError, RuntimeError, ArithmeticError, ImportError) as error:
                audit.setdefault('refused_holds', []).append(dict(time_s=state.time_s, duration_s=duration,
                    applied=False, allocation=allocation.to_dict(), exception_type=type(error).__name__, message=str(error)))
                return finish(False, 'deorbit_held_preview_refused: '+str(error))
            next_orbit = orbital_elements(next_state.position_eci_m, next_state.velocity_eci_m_s, assembly.mu_m3_s2)
            planned_orbit = orbital_elements(predicted.position_eci_m, predicted.velocity_eci_m_s, assembly.mu_m3_s2)
            perigee_reached = burn_start is not None and planned_orbit["perigee_radius_m"] <= rp
            entry_reached = (np.linalg.norm(next_state.position_eci_m) <= assembly.earth_radius_m + options.entry_interface_altitude_m
                             and np.dot(next_state.position_eci_m, next_state.velocity_eci_m_s) < 0)
            perigee_time, entry_time = None, None
            if perigee_reached:
                previous_rp = orbital_elements(feedback_state.position_eci_m, feedback_state.velocity_eci_m_s,
                                               assembly.mu_m3_s2)["perigee_radius_m"]
                if previous_rp > rp:
                    def perigee_event(dt):
                        candidate = predict(dt)
                        return orbital_elements(candidate.position_eci_m, candidate.velocity_eci_m_s,
                                                assembly.mu_m3_s2)["perigee_radius_m"] - rp
                    try:
                        perigee_time = brentq(perigee_event, 0, duration, xtol=1e-7)
                    except (ValueError, RuntimeError, ArithmeticError, ImportError) as error:
                        audit['refused_perigee_preview'] = dict(time_s=state.time_s, applied=False,
                            exception_type=type(error).__name__, message=str(error))
                        return finish(False, 'deorbit_perigee_preview_refused: '+str(error))
                else:
                    perigee_time = 0.
            if entry_reached and entry_monitor is None:
                try:
                    entry_time = brentq(lambda dt: np.linalg.norm(propagate(dt).position_eci_m)
                        - assembly.earth_radius_m - options.entry_interface_altitude_m, 0, duration, xtol=1e-7)
                except (ValueError, RuntimeError, ArithmeticError, ImportError) as error:
                    audit['refused_entry_preview'] = dict(time_s=state.time_s, applied=False,
                        exception_type=type(error).__name__, message=str(error))
                    return finish(False, 'deorbit_entry_preview_refused: '+str(error))
            # Both roots are bracketed against the original complete hold.
            # A later perigee cutoff must never hide an earlier entry handoff.
            # Entry takes precedence when the located times agree to root accuracy.
            if entry_time is not None and (perigee_time is None or entry_time <= perigee_time + 1e-7):
                duration, entry_reached, perigee_reached = entry_time, True, False
            elif perigee_time is not None:
                duration, entry_reached, perigee_reached = perigee_time, False, True
            if duration == 0.:
                audit["burn_cutoff_orbit"] = history[-1]["orbit"]
                audit["burn_cutoff_reason"] = "target_perigee_event"
                break
            try:
                next_state, angular_audit = assembly.advance_with_audit(state, duration, allocation.force_body_n,
                    allocation.torque_body_nm, allocation.fuel_flow_kg_s,
                    propulsion=propulsion, commands_n=allocation.commands_n,
                    max_angular_rate_rad_s=options.max_body_rate_rad_s)
            except (ValueError, RuntimeError, ArithmeticError, ImportError) as error:
                audit.setdefault('refused_holds', []).append(dict(time_s=state.time_s, duration_s=duration,
                    applied=False, allocation=allocation.to_dict(), exception_type=type(error).__name__, message=str(error)))
                return finish(False, 'deorbit_held_flow_refused: '+str(error))
            interface = None
            if entry_monitor is not None:
                if np.any(allocation.fuel_flow_kg_s != 0):
                    entry_monitor.reset()
                else:
                    interface = entry_monitor.interval(state, next_state, fuel_flow_kg_s=allocation.fuel_flow_kg_s)
            mass_before = assembly.mass_properties(state.fuel_kg)[0]
            mass_after = assembly.mass_properties(next_state.fuel_kg)[0]
            thrust_delta_v += np.linalg.norm(allocation.force_body_n) * duration / ((mass_before + mass_after) / 2)
            burn_time += duration if fire else 0.
            saturated_steps += int(allocation.saturated)
            force_residual = max(force_residual, allocation.residual_force_n)
            torque_residual = max(torque_residual, allocation.residual_torque_nm)
            max_balance = max(max_balance, angular_audit["angular_momentum_balance_residual_kg_m2_s"])
            control_record = dict(time_s=state.time_s, duration_s=duration, phase="finite_burn" if fire else "thrust_alignment",
                pointing_error_rad=pointing, pointing_rate_error_rad_s=tracking_rate,
                burn_window=window_request, aligned=aligned,
                requested_force_body_n=force.tolist(), requested_torque_body_nm=torque.tolist(),
                allocation=allocation.to_dict(), integration=angular_audit)
            control_summary = {key:control_record[key] for key in ('time_s','duration_s','phase')}
            controls.append(control_record)
            state = next_state
            history.append(self._record(assembly, state, "finite_burn" if fire else "thrust_alignment"))
            if assembly.attachment_limits:
                try:
                    endpoint = assembly.attachment_load_audit(state, propulsion, allocation.commands_n)
                except (ValueError, RuntimeError, ArithmeticError, ImportError) as error:
                    audit.setdefault('refused_holds', []).append(dict(time_s=state.time_s,
                        phase='accepted_endpoint_diagnostic', proposed_hold_applied=True,
                        exception_type=type(error).__name__, message=str(error)))
                    controls[-1] = records.keep('controls', control_record, control_summary)
                    return finish(False, 'deorbit_endpoint_audit_refused: '+str(error))
                angular_audit['attachment_load_endpoint'] = endpoint
                if not endpoint['passed']:
                    controls[-1] = records.keep('controls', control_record, control_summary)
                    return finish(False, 'deorbit_attachment_capacity_exceeded')
            controls[-1] = records.keep('controls', control_record, control_summary)
            if (entry_reached and entry_monitor is None) or interface is not None:
                audit["burn_cutoff_reason"] = "entry_interface_event_during_powered_phase"
                audit["burn_cutoff_orbit"] = history[-1]["orbit"]
                return finish(True, "deorbit_descending_entry_interface_handoff_completed_during_powered_phase")
            if entry_reached and entry_monitor is not None:
                audit['burn_cutoff_reason'] = 'numerical_entry_neighborhood_requires_whole_set_coast_bracket'
                audit['burn_cutoff_orbit'] = history[-1]['orbit']
                break
            if perigee_reached:
                audit["burn_cutoff_orbit"] = history[-1]["orbit"]
                audit["burn_cutoff_reason"] = "target_perigee_event"
                break
            if np.sum(state.fuel_kg) <= 1e-10:
                if history[-1]["orbit"]["perigee_radius_m"] < assembly.earth_radius_m + options.entry_interface_altitude_m:
                    audit["burn_cutoff_reason"] = "fuel_exhausted_with_entry_intersecting_orbit"
                    audit["burn_cutoff_orbit"] = history[-1]["orbit"]
                    break
                return finish(False, "deorbit_fuel_exhausted_before_entry_intersecting_orbit")
        axis = history[-1]["orbit"]["semi_major_axis_m"]
        period = 2 * np.pi * np.sqrt(axis**3 / assembly.mu_m3_s2)
        state, crossed, drift = self._coast(assembly, state, options.max_coast_orbits * period,
            history, "coast_to_entry_interface", propulsion=propulsion, observer=observer, entry_monitor=entry_monitor, record_retainer=records)
        coasts.append(drift)
        if drift.get('refusal'):
            return finish(False, 'deorbit_coast_not_proven: '+drift['refusal']['message'])
        if crossed and np.dot(state.position_eci_m, state.velocity_eci_m_s) < 0:
            if orbital_elements(state.position_eci_m, state.velocity_eci_m_s, assembly.mu_m3_s2)["perigee_radius_m"] > rp + .01:
                return finish(True, "deorbit_entry_interface_reached_perigee_target_not_achieved")
            return finish(True, "deorbit_descending_entry_interface_handoff_completed")
        return finish(False, "deorbit_entry_interface_not_reached_within_coast_limit")
