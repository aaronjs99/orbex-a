"""Apoapsis finite-burn deorbit, terminating at a declared entry interface.

This propagates a two-body ECI orbit and the same fueled rigid assembly used
for detumbling. Entry-interface crossing is a handoff to an atmospheric model;
breakup, demise, thermal protection and ground impact are outside this model.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation

from orbexa.core.attitude import finite_vector
from orbexa.simulation.assembly import AttachedAssembly, AssemblyState, STANDARD_GRAVITY, positive
from orbexa.simulation.propulsion import ThrusterSystem


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

    def __post_init__(self):
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
                 pointing_policy: Optional[ThrustPointingPolicy] = None, burn_axis_planner=None):
        self.options = options or DeorbitOptions()
        self.pointing_policy = pointing_policy or ThrustPointingPolicy()
        self.burn_axis_planner = burn_axis_planner

    def _record(self, assembly, state, phase):
        record = assembly.state_record(state)
        elements = orbital_elements(state.position_eci_m, state.velocity_eci_m_s, assembly.mu_m3_s2)
        record.update(phase=phase, orbit=elements,
                      altitude_m=elements["radius_m"] - assembly.earth_radius_m,
                      perigee_altitude_m=elements["perigee_radius_m"] - assembly.earth_radius_m)
        return record

    def _coast(self, assembly, initial, duration, history, phase, *, stop_at_entry=True):
        if duration <= 1e-9:
            return initial, False, dict(relative_energy_drift=0., relative_angular_momentum_drift=0.)
        mass, _, inertia = assembly.mass_properties(initial.fuel_kg)
        values = np.r_[initial.position_eci_m, initial.velocity_eci_m_s, initial.quaternion_body_to_eci,
                       initial.angular_velocity_body_rad_s]

        def rhs(_, state):
            return assembly.derivative(state, mass, inertia, np.zeros(3), np.zeros(3))[0]

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
                     relative_angular_momentum_drift=abs(after["specific_angular_momentum_m2_s"] / before["specific_angular_momentum_m2_s"] - 1))
        crossed = bool(stop_at_entry and len(solution.t_events[0]) > 0)
        return state, crossed, drift

    def run(self, assembly: AttachedAssembly, initial: AssemblyState, propulsion: ThrusterSystem) -> DeorbitResult:
        options, state = self.options, initial
        history, controls, coasts = [self._record(assembly, initial, "deorbit_planning")], [], []
        audit = dict(entry_interface_altitude_m=options.entry_interface_altitude_m,
                     target_perigee_altitude_m=options.target_perigee_altitude_m,
                     terminal_scope="descending entry-interface handoff; no atmosphere, heating, breakup, demise or impact",
                     dynamics="two-body Cartesian ECI plus controlled rigid-body attitude; no J2 or atmospheric drag",
                     initial_orbit=history[0]["orbit"])
        thrust_delta_v, saturated_steps, max_balance = 0., 0, 0.
        force_residual, torque_residual, burn_time = 0., 0., 0.

        def finish(success, message):
            final = orbital_elements(state.position_eci_m, state.velocity_eci_m_s, assembly.mu_m3_s2)
            audit.update(elapsed_s=state.time_s - initial.time_s, fuel_used_kg=float(np.sum(initial.fuel_kg - state.fuel_kg)),
                final_orbit=final, final_altitude_m=final["radius_m"] - assembly.earth_radius_m,
                target_perigee_achieved=bool(final["perigee_radius_m"] <= assembly.earth_radius_m + options.target_perigee_altitude_m + .01),
                final_radial_velocity_m_s=final["radial_velocity_m_s"], burn_time_s=burn_time,
                integrated_thrust_delta_v_m_s=thrust_delta_v, saturated_steps=saturated_steps,
                maximum_force_allocation_residual_n=force_residual, maximum_torque_allocation_residual_nm=torque_residual,
                max_step_angular_balance_residual_kg_m2_s=max_balance, coast_invariant_audits=coasts)
            return DeorbitResult(success, message, state, history, controls, audit)

        if np.linalg.norm(initial.angular_velocity_body_rad_s) > options.max_initial_rate_rad_s:
            return finish(False, "deorbit_requires_detumbled_initial_state")
        initial_elements = history[0]["orbit"]
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
                                               history, "coast_to_entry_interface")
            coasts.append(drift)
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
        apo_time = state.time_s + time_to_apoapsis_s(state.position_eci_m, state.velocity_eci_m_s, assembly.mu_m3_s2)
        coast_duration = max(0., apo_time - state.time_s - options.alignment_lead_s)
        state, crossed, drift = self._coast(assembly, state, coast_duration, history, "coast_to_apoapsis")
        coasts.append(drift)
        if crossed:
            return finish(True, "descending_entry_interface_reached_during_coast")
        alignment_start, burn_start = state.time_s, None
        pointing_policy = self.pointing_policy
        axis = np.asarray(getattr(pointing_policy, 'thrust_axis_body', (1., 0., 0.)))
        if self.burn_axis_planner is not None:
            if not callable(getattr(pointing_policy, 'with_thrust_axis', None)):
                raise ValueError('A selected burn axis requires a pointing policy with with_thrust_axis')
            plan = self.burn_axis_planner.select(assembly, state, propulsion, options.control_step_s)
            pointing_policy = pointing_policy.with_thrust_axis(plan.direction_body)
            axis = np.asarray(plan.direction_body)
            default_force = plan.commanded_force_n
            audit['burn_axis_plan'] = plan.to_dict()
        else:
            default_force = .65 * sum(thruster.max_force_n * abs(float(np.dot(thruster.direction_body, axis)))
                                      for thruster in propulsion.thrusters)
        burn_force = default_force if options.burn_force_n is None else options.burn_force_n
        audit.update(commanded_burn_force_n=burn_force, planned_apoapsis_time_s=apo_time,
                     thrust_axis_body=axis.tolist())
        while True:
            torque, pointing, tracking_rate = pointing_policy.command(assembly, state)
            aligned = pointing <= options.pointing_tolerance_rad and tracking_rate <= options.pointing_rate_tolerance_rad_s
            if burn_start is None and state.time_s >= apo_time and aligned:
                burn_start = state.time_s
                audit["burn_start_time_s"] = burn_start
                audit["burn_start_orbit"] = orbital_elements(state.position_eci_m, state.velocity_eci_m_s, assembly.mu_m3_s2)
            if burn_start is None and state.time_s - alignment_start > options.max_alignment_s + max(0., apo_time - alignment_start):
                return finish(False, "deorbit_pointing_alignment_timeout")
            if burn_start is not None and state.time_s - burn_start > options.max_burn_s:
                return finish(False, "deorbit_finite_burn_duration_limit")
            if np.linalg.norm(state.angular_velocity_body_rad_s) > options.max_body_rate_rad_s:
                return finish(False, "deorbit_body_rate_limit_exceeded")
            duration = options.control_step_s
            fire = burn_start is not None and aligned
            force = axis * (burn_force if fire else 0.)
            allocation = propulsion.allocate(assembly, state, force, torque, duration)
            if not allocation.success:
                return finish(False, f"deorbit_{allocation.message}")

            def propagate(dt):
                if dt <= 0:
                    return state
                return assembly.advance(state, dt, allocation.force_body_n, allocation.torque_body_nm, allocation.fuel_flow_kg_s)

            next_state = propagate(duration)
            next_orbit = orbital_elements(next_state.position_eci_m, next_state.velocity_eci_m_s, assembly.mu_m3_s2)
            perigee_reached = burn_start is not None and next_orbit["perigee_radius_m"] <= rp
            entry_reached = (np.linalg.norm(next_state.position_eci_m) <= assembly.earth_radius_m + options.entry_interface_altitude_m
                             and np.dot(next_state.position_eci_m, next_state.velocity_eci_m_s) < 0)
            perigee_time, entry_time = None, None
            if perigee_reached:
                previous_rp = orbital_elements(state.position_eci_m, state.velocity_eci_m_s, assembly.mu_m3_s2)["perigee_radius_m"]
                if previous_rp > rp:
                    def perigee_event(dt):
                        candidate = propagate(dt)
                        return orbital_elements(candidate.position_eci_m, candidate.velocity_eci_m_s,
                                                assembly.mu_m3_s2)["perigee_radius_m"] - rp
                    perigee_time = brentq(perigee_event, 0, duration, xtol=1e-7)
                else:
                    perigee_time = 0.
            if entry_reached:
                entry_time = brentq(lambda dt: np.linalg.norm(propagate(dt).position_eci_m)
                    - assembly.earth_radius_m - options.entry_interface_altitude_m, 0, duration, xtol=1e-7)
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
            next_state, angular_audit = assembly.advance_with_audit(state, duration, allocation.force_body_n,
                allocation.torque_body_nm, allocation.fuel_flow_kg_s)
            mass_before = assembly.mass_properties(state.fuel_kg)[0]
            mass_after = assembly.mass_properties(next_state.fuel_kg)[0]
            thrust_delta_v += np.linalg.norm(allocation.force_body_n) * duration / ((mass_before + mass_after) / 2)
            burn_time += duration if fire else 0.
            saturated_steps += int(allocation.saturated)
            force_residual = max(force_residual, allocation.residual_force_n)
            torque_residual = max(torque_residual, allocation.residual_torque_nm)
            max_balance = max(max_balance, angular_audit["angular_momentum_balance_residual_kg_m2_s"])
            controls.append(dict(time_s=state.time_s, duration_s=duration, phase="finite_burn" if fire else "thrust_alignment",
                pointing_error_rad=pointing, pointing_rate_error_rad_s=tracking_rate,
                requested_force_body_n=force.tolist(), requested_torque_body_nm=torque.tolist(),
                allocation=allocation.to_dict(), integration=angular_audit))
            state = next_state
            history.append(self._record(assembly, state, "finite_burn" if fire else "thrust_alignment"))
            if entry_reached:
                audit["burn_cutoff_reason"] = "entry_interface_event_during_powered_phase"
                audit["burn_cutoff_orbit"] = history[-1]["orbit"]
                return finish(True, "deorbit_descending_entry_interface_handoff_completed_during_powered_phase")
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
                                           history, "coast_to_entry_interface")
        coasts.append(drift)
        if crossed and np.dot(state.position_eci_m, state.velocity_eci_m_s) < 0:
            if orbital_elements(state.position_eci_m, state.velocity_eci_m_s, assembly.mu_m3_s2)["perigee_radius_m"] > rp + .01:
                return finish(True, "deorbit_entry_interface_reached_perigee_target_not_achieved")
            return finish(True, "deorbit_descending_entry_interface_handoff_completed")
        return finish(False, "deorbit_entry_interface_not_reached_within_coast_limit")
