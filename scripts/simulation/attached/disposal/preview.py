"""Finite Cartesian burn previews for numerical throttle selection."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp

from orbexa.simulation.attached.assembly.model import positive
from orbexa.simulation.attached.disposal.simulator import orbital_elements
from orbexa.simulation.attached.disposal.schedule import ApoapsisBurnWindow
from orbexa.core.dynamics.attitude import finite_vector


@dataclass(frozen=True)
class FiniteBurnThrottlePlanner:
    """Compare declared throttles before committing an actual burn.

    This preview tracks two-body translation and total mass, using the
    operational allocator's static per-owner fuel rates. Attitude, changing
    allocation and inertia are omitted; actual mission gates remain binding.
    """
    capacity_fractions: tuple[float, ...] = (.65, .8, .9, .95)
    fuel_reserve_fraction: float = .1
    residual_tolerance: float = 1e-7

    def __post_init__(self):
        values = tuple(positive(v, 'Preview capacity fraction') for v in self.capacity_fractions)
        if not values or len(set(values)) != len(values) or any(v >= 1 for v in values):
            raise ValueError('Preview fractions must be distinct and strictly between zero and one')
        reserve = positive(self.fuel_reserve_fraction, 'Preview fuel reserve', allow_zero=True)
        if reserve >= 1:
            raise ValueError('Preview fuel reserve must be below one')
        object.__setattr__(self, 'capacity_fractions', values)
        object.__setattr__(self, 'fuel_reserve_fraction', reserve)
        object.__setattr__(self, 'residual_tolerance', positive(self.residual_tolerance, 'Preview allocation tolerance'))

    def select(self, assembly, state, propulsion, axis_plan, options):
        orbit = orbital_elements(state.position_eci_m, state.velocity_eci_m_s, assembly.mu_m3_s2)
        if orbit['apogee_radius_m'] is None:
            raise ValueError('Finite burn preview requires a bound orbit')
        radial = (-np.asarray(orbit['eccentricity_vector_eci']) / orbit['eccentricity']
                  if orbit['eccentricity'] >= 1e-8 else state.position_eci_m / np.linalg.norm(state.position_eci_m))
        normal = np.asarray(orbit['angular_momentum_direction_eci'])
        radius = orbit['apogee_radius_m']
        position = radius * radial
        velocity = orbit['specific_angular_momentum_m2_s'] / radius * np.cross(normal, radial)
        mass = assembly.mass_properties(state.fuel_kg)[0]
        initial = np.r_[position, velocity, mass, 0.]
        burn_window = ApoapsisBurnWindow(options.burn_window_half_angle_rad)
        direction = finite_vector(axis_plan.direction_body, 'Preview body axis', 3)
        if abs(np.linalg.norm(direction)-1.) > 1e-10:
            raise ValueError('Preview body axis must be a unit vector')
        installed_capacity = positive(axis_plan.capacity_n, 'Installed static coupled force capacity')
        capacity = positive(axis_plan.commanded_force_n, 'Operational admitted burn force reference')
        if capacity > installed_capacity:
            raise ValueError('Operational burn reference exceeds the installed static capacity witness')
        target_radius = assembly.earth_radius_m + options.target_perigee_altitude_m
        entry_radius = assembly.earth_radius_m + options.entry_interface_altitude_m
        rows, accepted = [], []
        for fraction in self.capacity_fractions:
            force = fraction * capacity
            row = dict(capacity_fraction=fraction, commanded_force_n=force,
                       status='STATIC_ALLOCATION_REFUSED')
            rows.append(row)
            try:
                allocation = propulsion.allocate(assembly, state, force * direction, np.zeros(3), options.control_step_s)
                row['static_allocation'] = allocation.to_dict()
                residuals = np.asarray([allocation.residual_force_n, allocation.residual_torque_nm],dtype=float)
                flow = np.asarray(allocation.fuel_flow_kg_s,dtype=float)
            except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as error:
                row.update(status='PREVIEW_UNRESOLVED',error_type=type(error).__name__,reason=str(error))
                continue
            if (not allocation.success or not np.all(np.isfinite(residuals))
                    or np.any(residuals < 0) or max(residuals) > self.residual_tolerance
                    or flow.shape != state.fuel_kg.shape
                    or not np.all(np.isfinite(flow)) or np.any(flow < 0) or np.sum(flow) <= 0):
                continue
            owner_times = [(1-self.fuel_reserve_fraction)*fuel/rate
                           for fuel, rate in zip(state.fuel_kg, flow) if rate > 0]
            fuel_duration = min(owner_times)
            duration = options.max_burn_s
            row.update(owner_reserve_times_s=owner_times, maximum_preview_duration_s=duration)
            if duration <= 0:
                continue
            total_flow = float(np.sum(flow))

            def derivative(time, values):
                r, v, m = values[:3], values[3:6], values[6]
                norm = np.linalg.norm(r)
                if norm <= 0 or m <= 0:
                    raise ValueError('Nonphysical finite preview state')
                unit = r / norm
                transverse = v - np.dot(v, unit)*unit
                speed = np.linalg.norm(transverse)
                if speed <= 0:
                    raise ValueError('Degenerate transverse direction in finite preview')
                active = burn_window.evaluate(r,v,assembly.mu_m3_s2)['active']
                return np.r_[v, -assembly.mu_m3_s2*r/norm**3 - (force/m*transverse/speed if active else np.zeros(3)),
                    -total_flow if active else 0., 1. if active else 0.]

            def perigee_event(time, values):
                return orbital_elements(values[:3],values[3:6],assembly.mu_m3_s2)['perigee_radius_m']-target_radius

            def entry_event(time, values):
                return np.linalg.norm(values[:3])-entry_radius

            def reserve_event(time, values):
                return fuel_duration-values[7]
            reserve_event.terminal, reserve_event.direction = True, -1
            perigee_event.terminal, perigee_event.direction = True, -1
            entry_event.terminal, entry_event.direction = True, -1
            try:
                solution = solve_ivp(derivative, (0., duration), initial, method='DOP853',
                    rtol=1e-9, atol=np.r_[np.full(3,1e-3),np.full(3,1e-6),1e-7,1e-7],
                    max_step=10., events=(perigee_event,entry_event,reserve_event))
                values = np.asarray(solution.y,dtype=float)
                if (not solution.success or values.ndim != 2 or values.shape[0] != 8
                        or values.shape[1] == 0 or not np.all(np.isfinite(values))):
                    row.update(status='PREVIEW_UNRESOLVED',reason=str(solution.message))
                    continue
                final = values[:,-1]
                final_orbit = orbital_elements(final[:3],final[3:6],assembly.mu_m3_s2)
            except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as error:
                row.update(status='PREVIEW_UNRESOLVED',error_type=type(error).__name__,reason=str(error))
                continue
            elapsed = float(solution.t[-1])
            active_time = float(final[7])
            remaining = state.fuel_kg-flow*active_time
            required_reserve = self.fuel_reserve_fraction*state.fuel_kg
            if (not 0 <= active_time <= elapsed or active_time > fuel_duration
                    or not np.all(np.isfinite(remaining)) or np.any(remaining < required_reserve)):
                row.update(status='PREVIEW_UNRESOLVED',
                    reason='Numerical active-time or owner-reserve postcondition not established',
                    predicted_duration_s=elapsed, predicted_active_burn_s=active_time,
                    predicted_remaining_owner_fuel_kg=remaining.tolist())
                continue
            predicted_goal = len(solution.t_events[0]) > 0
            predicted_entry = len(solution.t_events[1]) > 0 and final_orbit['radial_velocity_m_s'] < 0
            row.update(status='PERIGEE_EVENT_PREDICTED' if predicted_goal else 'ENTRY_EVENT_PREDICTED' if predicted_entry else 'PREVIEW_FUEL_OR_CAP_BEFORE_GOAL',
                predicted_duration_s=elapsed, predicted_active_burn_s=float(final[7]), predicted_fuel_used_kg=total_flow*float(final[7]),
                predicted_remaining_owner_fuel_kg=(state.fuel_kg-flow*float(final[7])).tolist(),
                predicted_final_orbit=final_orbit, predicted_final_state=final.tolist())
            if predicted_goal:
                accepted.append((total_flow*float(final[7]), elapsed, len(rows)-1))
        index = min(accepted)[2] if accepted else None
        return dict(burn_window=burn_window.as_dict(), installed_static_capacity_n=installed_capacity, operational_force_reference_n=capacity,
                    fraction_reference='Each prospective throttle is a fraction of the operational requested wrench within the declared residual tolerance admitted at the planning state; installed LP capacity remains separate.',
                    status='FINITE_BURN_PREDICTED' if accepted else 'NO_ADMITTED_FINITE_BURN_PREVIEW',
                    selected_candidate=index,
                    commanded_force_n=None if index is None else rows[index]['commanded_force_n'],
                    capacity_fraction=None if index is None else rows[index]['capacity_fraction'], candidates=rows,
                    prediction_start='Osculating apoapsis; this is a forecast, never an actual state reset',
                    predicted_initial_state=initial.tolist(),
                    scope='Numerical two-body retrograde-transverse preview with an optional sensed-osculating-apogee request window, changing total mass and static owner flows. Reserve is a planning margin, not a bound on omitted attitude, allocation or tank dynamics; actual mission gates decide success.')
