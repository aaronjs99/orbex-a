"""Coupled nominal/tube planning with persistent, sampled ancillary feedback.

The optimizer and plant share held-input RK4 dynamics in physical LVLH meters.
Continuous-time tube reasoning motivates the law; sampled containment is checked
explicitly and is not promoted to a continuous-time or recursive-feasibility proof.
"""
from collections import deque
import numpy as np

from orbexa.control.mpc_controller import MPCController
from orbexa.control.dynamic_tube_model import (
    TubeProfile, advance_tube, boundary_layer_feedback,
)
from orbexa.control.uncertainty import (
    uncertainty_envelope, nominal_acceleration_bound, coefficient_intervals,
)
from orbexa.control.terminal_cost import local_terminal_cost
from orbexa.core.dynamics import orbital_ellp_drag
from orbexa.utils.anomaly import orbital_anomaly_rate


def rk4_step(state, control, anomaly, step, matrices, periapsis=0.0):
    """One held-input transition, shared by nominal and plant propagation."""
    a, b, _, _, d = matrices
    input_term = np.asarray(b(), dtype=float) @ control
    def rhs(q, x):
        return np.asarray(a(q, periapsis), dtype=float) @ x + input_term + np.asarray(d(q, periapsis), dtype=float)
    k1 = rhs(anomaly, state)
    k2 = rhs(anomaly + step / 2, state + step * k1 / 2)
    k3 = rhs(anomaly + step / 2, state + step * k2 / 2)
    k4 = rhs(anomaly + step, state + step * k3)
    return state + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6


class CoupledTubeMission:
    """Mission policy around the solver-independent dynamic-tube problem data."""

    def __init__(self, runner):
        self.runner = runner
        self.config = runner.config
        counts = ([runner.actuation_steps] if runner.actuation_steps is not None else
                  list(self.config.mpc.actuation_steps.values()))
        if any(count != 1 for count in counts):
            raise ValueError("Coupled ADTMPC replans every outer interval; actuation_steps must be one")
        if runner.target.docking_standoff <= runner.target.radius * runner.target.tolerance + runner.target_barrier_clearance:
            raise ValueError("The docking standoff leaves no free-flight tube clearance; contact requires an explicit contact model")

    def _spec(self, belief, params, q, count, phi, omega, bandwidth):
        tube, orbit = self.config.tube, self.config.orbit
        delta_a, delta_d, a_abs, d_abs = [], [], [], []
        for stage in range(count - 1):
            lo, hi = q + stage * self.config.anom_step, q + (stage + 1) * self.config.anom_step
            da, dd = uncertainty_envelope(lo, hi, params,
                belief.feasible_sets['eccentricity'], belief.feasible_sets['alpha'],
                belief.feasible_sets['beta'], orbit.semi_major_axis, orbit.mu)
            delta_a.append(da)
            delta_d.append(dd)
            a_abs.append(nominal_acceleration_bound(lo, hi, params, orbit.semi_major_axis))
            _, _, dl, dh = coefficient_intervals(lo, hi, params, orbit.semi_major_axis)
            d_abs.append(np.maximum(np.abs(dl), np.abs(dh)))
        return dict(phi_initial=phi, omega_initial=omega, lambda_gain=tube.sliding_gains,
            alpha_initial=bandwidth, alpha_bounds=(tube.bandwidth_range['lower'], tube.bandwidth_range['upper']),
            alpha_rate_limit=tube.bandwidth_rate_limit, delta_A=np.asarray(delta_a), delta_d=np.asarray(delta_d),
            A_abs=np.asarray(a_abs), d_abs=np.asarray(d_abs), eta=np.full(3, tube.numerical_reserve),
            bandwidth_weight=tube.bandwidth_weight, tube_weight=tube.tube_weight,
            terminal_position_tolerance=self.runner.docking_position_tolerance,
            terminal_velocity_tolerance=self.runner.docking_velocity_tolerance, enforce_terminal=False)

    def run(self, *, mission, steps):
        from orbexa.simulation.adtmpc_mission import (
            ADTMPCMissionResult, MissionPhase, ParameterBelief, _serial_ranges, _serial_estimates,
        )
        if mission not in ('single', 'multi'):
            raise ValueError(f'Unknown ADTMPC mission: {mission}')
        r, cfg = self.runner, self.config
        chasers = r._single_chaser() if mission == 'single' else r._multi_chasers()
        ids = [c.chaser_id for c in chasers]
        controller = MPCController(solver_backend='gekko', solver_config=r._solver_config())
        belief = ParameterBelief.from_config_prior(cfg)
        initial_belief = belief.copy()
        states = {c.chaser_id: np.array(c.initial_state, dtype=float) for c in chasers}
        centers = {key: value.copy() for key, value in states.items()}
        phi = {key: np.array(cfg.tube.boundary_layer_0, dtype=float) for key in ids}
        omega = {key: np.zeros(3) for key in ids}
        bandwidth = {key: np.array(cfg.tube.bandwidth_0, dtype=float) for key in ids}
        actual, nominal, controls, stats, costs, profiles = ({key: [] for key in ids} for _ in range(6))
        diagnostics = {key: dict(nominal_center=[], phi=[], omega=[], bandwidth=[],
            position_error=[], sliding_error=[], containment_margin=[], terminal_position_bound=[],
            terminal_velocity_bound=[], max_dense_input_reserve_ratio=[], nominal_control=[],
            ancillary_control_rms=[], applied_control_rms=[], applied_control_effort=[],
            dense_containment_margin=[], dense_physical_clearance=[]) for key in ids}
        for key in ids:
            actual[key].append(states[key].tolist())
            self._record(diagnostics[key], states[key], centers[key], phi[key], omega[key], bandwidth[key])
        q = 0.0
        q_history, elapsed = [q], [0.0]
        phase = MissionPhase('rendezvous', 'rendezvous')
        phases, sample_phases = [], [phase.name]
        radius_history = [0.0]
        attitudes = [r.target.orientation_at(q).tolist()]
        rates = [r.target.angular_velocity_at(q).tolist()]
        feasible = [_serial_ranges(belief.feasible_sets)]
        estimates = [_serial_estimates(belief.estimates)]
        smid_records, membership_records, messages = [], [], []
        dense_states = deque([states[ids[0]].copy()], maxlen=9)
        dense_controls = deque(maxlen=8)
        dense_q = deque([q], maxlen=9)
        previous_plans = {}
        truth_matrices = orbital_ellp_drag(anom_step=cfg.anom_step, **r.truth.dynamics_params())[0]
        substeps = cfg.tube.feedback_substeps
        h = cfg.anom_step / substeps
        lam = np.asarray(cfg.tube.sliding_gains)
        completed, valid, dwell, contained = False, True, 0, True
        dense_min_clearance, dense_min_spacing, dense_min_containment = np.inf, np.inf, np.inf
        max_nominal_discretization_residual = 0.0
        for update in range(int(steps)):
            if phase.operation == 'rendezvous' and r._rendezvous_goal_satisfied(states, chasers):
                phase = MissionPhase('docking', 'docking')
                sample_phases[-1] = phase.name
            phases.append(phase.name)
            count = r._horizon_steps_for_phase(phase)
            params = belief.dynamics_params(r.truth)
            matrices = orbital_ellp_drag(anom_step=cfg.anom_step, **params)[0]
            solutions = {}
            for chaser in chasers:
                key = chaser.chaser_id
                spec = self._spec(belief, params, q, count, phi[key], omega[key], bandwidth[key])
                goal = r._phase_reference_state(phase, chaser, centers[key], q)
                references, feedforward = r._phase_reference_preview(phase, chaser, goal, q, count, matrices, params['time_periapsis'])
                target = r.target.collision_params(phase.operation, q)
                target.update(orientations=[r.target.orientation_at(q + k * cfg.anom_step).tolist() for k in range(count)],
                    robust_side_corridor=phase.operation == 'docking', docking_normal_body=chaser.docking_normal_body,
                    safety_reserve=r.target_barrier_clearance)
                # Arm an explicit terminal set only near the final approach.
                # Completion always checks that set, including the current tube.
                spec['enforce_terminal'] = phase.operation == 'docking' and np.linalg.norm(centers[key][:3] - goal[:3]) < 0.035
                pairwise = []
                for other in ids:
                    if other == key:
                        continue
                    prior = solutions.get(other, previous_plans.get(other))
                    if prior is not None:
                        path = np.asarray(prior['result'].state_trajectory).T
                        rad = np.asarray(prior['profile'].position_radius)
                        shift = 0 if other in solutions else 1
                        indices = np.minimum(np.arange(count) + shift, len(path) - 1)
                        path, rad = path[indices, :3], rad[indices]
                    else:
                        path = np.array([centers[other][:3] + k * cfg.anom_step * centers[other][3:] for k in range(count)])
                        rad = np.full(count, np.linalg.norm(omega[other]))
                    pairwise.append(dict(other_chaser=other, reference_positions=path, tube_radius=rad,
                        min_separation=r.min_chaser_separation))
                kwargs = dict(mean_motion=params['mean_motion'], state_cost_matrix=r._phase_state_weights(phase),
                    input_cost_matrix=cfg.mpc.R, dynamic_tube=spec, target_params=target, pairwise_constraints=pairwise)
                prior = previous_plans.get(key)
                if prior is not None:
                    path = np.asarray(prior['result'].state_trajectory)
                    commands = np.asarray(prior['result'].control_trajectory)
                    kwargs['state_initial_guess'] = path[:, np.minimum(np.arange(count) + 1, path.shape[1] - 1)]
                    kwargs['control_initial_guess'] = commands[:, np.minimum(np.arange(count - 1) + 1, commands.shape[1] - 1)]
                if cfg.mpc.reference_preview_enabled or spec['enforce_terminal']:
                    kwargs.update(state_reference=references, input_reference=feedforward)
                if cfg.mpc.terminal_cost_enabled:
                    kwargs['terminal_cost_matrix'] = local_terminal_cost(matrices[0](q, params['time_periapsis']),
                        matrices[1](), kwargs['state_cost_matrix'], cfg.mpc.R, cfg.anom_step)
                result = controller.solve_step(initial_state=centers[key], final_state=goal, control_input_0=feedforward[:, 0],
                    start_anom=q, anom_step=cfg.anom_step, num_steps=count, dynamics=matrices, bounds=cfg.mpc.bounds(),
                    time_periapsis=params['time_periapsis'], eccentricity=params['eccentricity'], use_anomaly_scaling=False, **kwargs)
                stats[key].append(dict(success=bool(result.success), solve_time=float(result.solve_time), message=result.message,
                    solver_info={name:value for name,value in result.solver_info.items() if name != 'dynamic_tube'}))
                costs[key].append(result.cost)
                if not result.success:
                    valid = False
                    messages.append(f'{key}: {result.message}')
                    break
                tube = result.solver_info['dynamic_tube']
                profile = TubeProfile(phi=np.asarray(tube['phi']), omega=np.asarray(tube['omega']),
                    time_grid=q + cfg.anom_step * np.arange(count), lambda_gain=lam,
                    alpha=np.asarray(tube['alpha']), input_reserve=np.asarray(tube['input_reserve']))
                nominal[key].append(np.asarray(result.state_trajectory).T.tolist())
                profiles[key].append(r._tube_metadata(q, profile))
                solutions[key] = dict(result=result, profile=profile, spec=spec, tube=tube)
            if not valid:
                break
            # Verify simultaneous newly planned two-tube separation before any
            # command is applied. Stale neighbor plans alone are insufficient.
            for i, key in enumerate(ids):
                for other in ids[i+1:]:
                    first, second = solutions[key], solutions[other]
                    distance = np.linalg.norm(np.asarray(first['result'].state_trajectory)[:3].T -
                                              np.asarray(second['result'].state_trajectory)[:3].T, axis=1)
                    required = r.min_chaser_separation + first['profile'].position_radius + second['profile'].position_radius
                    if np.min(distance - required) < -1e-7:
                        valid = False
                        messages.append(f'{key}/{other}: simultaneous planned tubes conflict')
            if not valid:
                break
            reserve_ratios = {key: 0.0 for key in ids}
            applied = {key: [] for key in ids}
            ancillary = {key: [] for key in ids}
            interval_containment = {key: np.inf for key in ids}
            interval_clearance = {key: np.inf for key in ids}
            committed_q = q
            for inner in range(substeps):
                current = q + inner * h
                proposed = {}
                proposed_centers, proposed_phi, proposed_omega, inputs, feedbacks = {}, {}, {}, {}, {}
                for key in ids:
                    sol = solutions[key]
                    u_nom = np.asarray(sol['result'].control_trajectory)[:, 0]
                    alpha = np.asarray(sol['tube']['alpha'])[0]
                    a = np.asarray(matrices[0](current, params['time_periapsis']), dtype=float)[3:]
                    feedback = boundary_layer_feedback(a, sol['spec']['delta_A'][0], states[key], centers[key], phi[key], alpha, lam)
                    u = u_nom + feedback
                    reserve = np.asarray(sol['tube']['input_reserve'])[0]
                    reserve_ratios[key] = max(reserve_ratios[key], float(np.max(np.abs(feedback) / np.maximum(reserve, 1e-12))))
                    if np.max(np.abs(feedback) - reserve) > 1e-5:
                        valid = False
                        messages.append(f'{key}: ancillary feedback exceeded the planned input reserve')
                        break
                    clipped = r._clip_control(u)
                    if np.max(np.abs(clipped - u)) > 1e-8:
                        valid = False
                        messages.append(f'{key}: ancillary input would saturate; tube policy interrupted')
                        break
                    inputs[key], feedbacks[key] = u, feedback
                if not valid:
                    break
                for key in ids:
                    sol = solutions[key]
                    u_nom = np.asarray(sol['result'].control_trajectory)[:, 0]
                    alpha = np.asarray(sol['tube']['alpha'])[0]
                    u = inputs[key]
                    proposed[key] = rk4_step(states[key], u, current, h, truth_matrices, r.truth.time_periapsis)
                    proposed_centers[key] = rk4_step(centers[key], u_nom, current, h, matrices, params['time_periapsis'])
                    proposed_phi[key], proposed_omega[key] = advance_tube(phi[key], omega[key], alpha, lam,
                        np.asarray(sol['tube']['disturbance_bound'])[0], h)
                    bandwidth[key] = alpha
                    if key == ids[0]:
                        dense_controls.append(u.copy())
                        dense_states.append(proposed[key].copy())
                        dense_q.append(current + h)
                states = proposed
                centers, phi, omega = proposed_centers, proposed_phi, proposed_omega
                committed_q = current + h
                for key in ids:
                    applied[key].append(inputs[key])
                    ancillary[key].append(feedbacks[key])
                    error = states[key] - centers[key]
                    margin = min(float(np.min(omega[key] - np.abs(error[:3]))),
                                 float(np.min(phi[key] - np.abs(error[3:] + lam * error[:3]))))
                    clearance = r._active_margin_for_state(states[key], current + h, phase)
                    dense_min_containment = min(dense_min_containment, margin)
                    dense_min_clearance = min(dense_min_clearance, clearance)
                    interval_containment[key] = min(interval_containment[key], margin)
                    interval_clearance[key] = min(interval_clearance[key], clearance)
                    contained &= margin >= -1e-8
                    if margin < -1e-8 or clearance < -r.target_safety_tolerance:
                        valid = False
                        messages.append(f'{key}: dense containment or clearance violated at q={current+h:.9g}')
                for i, key in enumerate(ids):
                    for other in ids[i+1:]:
                        distance = float(np.linalg.norm(states[key][:3] - states[other][:3]))
                        dense_min_spacing = min(dense_min_spacing, distance)
                        if distance < r.min_chaser_separation - r.target_safety_tolerance:
                            valid = False
                            messages.append('dense pairwise spacing violated')
                if not valid:
                    break
            next_q = committed_q
            if next_q == q:
                break
            for key in ids:
                sol = solutions[key]
                if valid:
                    residual = np.linalg.norm(centers[key] - np.asarray(sol['result'].state_trajectory)[:, 1])
                    max_nominal_discretization_residual = max(max_nominal_discretization_residual, float(residual))
                if valid:
                    phi[key] = np.maximum(phi[key], sol['profile'].phi[1])
                    omega[key] = np.maximum(omega[key], sol['profile'].omega[1])
                actual[key].append(states[key].tolist())
                # Display averages; estimation receives exact dense held inputs.
                controls[key].append(np.mean(applied[key], axis=0).tolist())
                self._record(diagnostics[key], states[key], centers[key], phi[key], omega[key], bandwidth[key])
                diagnostics[key]['max_dense_input_reserve_ratio'].append(reserve_ratios[key])
                diagnostics[key]['nominal_control'].append(np.asarray(sol['result'].control_trajectory)[:, 0].tolist())
                diagnostics[key]['ancillary_control_rms'].append(np.sqrt(np.mean(np.square(ancillary[key]), axis=0)).tolist())
                diagnostics[key]['applied_control_rms'].append(np.sqrt(np.mean(np.square(applied[key]), axis=0)).tolist())
                diagnostics[key]['applied_control_effort'].append(float(h * np.sum(np.square(applied[key]))))
                diagnostics[key]['dense_containment_margin'].append(float(interval_containment[key]))
                diagnostics[key]['dense_physical_clearance'].append(float(interval_clearance[key]))
            rate0 = orbital_anomaly_rate(q, **{k:v for k,v in r.truth.dynamics_params().items()
                if k in ('eccentricity','alpha','mu','semi_major_axis','mean_motion','specific_angular_momentum')})
            rate1 = orbital_anomaly_rate(next_q, eccentricity=r.truth.eccentricity, alpha=r.truth.alpha,
                mu=r.truth.mu, semi_major_axis=r.truth.semi_major_axis)
            elapsed.append(elapsed[-1] + (next_q - q) * (1 / rate0 + 1 / rate1) / 2)
            q = next_q
            q_history.append(float(q))
            sample_phases.append(phase.name)
            attitudes.append(r.target.orientation_at(q).tolist())
            rates.append(r.target.angular_velocity_at(q).tolist())
            radius_history.append(max(float(np.linalg.norm(omega[key])) for key in ids))
            if valid and len(dense_states) == 9:
                # Local nonlinear optimizers supply a point estimate. Only the
                # interval contractor is permitted to remove parameter values.
                if update % cfg.tube.membership_update_period == 0:
                    from orbexa.estimation.interval_membership import contract_parameter_box
                    ranges, record = contract_parameter_box(belief.feasible_sets, np.asarray(dense_states),
                        np.asarray(dense_controls), dense_q[0], h, r._smid_context(),
                        error_bound=cfg.tube.membership_error_bound, max_boxes=cfg.tube.membership_box_budget)
                    belief.feasible_sets = ranges
                    for name, interval in ranges.items():
                        belief.estimates[name] = float(np.clip(belief.estimates[name], *interval))
                    membership_records.append(record)
                    if record.get('status') == 'inconsistent':
                        valid = False
                        messages.append('Recorded dynamics are inconsistent with the parameter box and measurement bound')
                _, point, record = r.smid.update(feasible_sets=belief.feasible_sets, estimates=belief.estimates,
                    states=np.asarray(dense_states), controls=np.asarray(dense_controls), start_anom=dense_q[0],
                    anom_step=h, dynamics_context=r._smid_context())
                belief.estimates = point
                smid_records.append(record.to_dict())
            feasible.append(_serial_ranges(belief.feasible_sets))
            estimates.append(_serial_estimates(belief.estimates))
            admissible = phase.operation == 'docking'
            for chaser in chasers:
                key = chaser.chaser_id
                goal = r.target.docking_state(chaser.docking_point_body, q)
                pb = np.linalg.norm(centers[key][:3] - goal[:3]) + np.linalg.norm(omega[key])
                vb = np.linalg.norm(centers[key][3:] - goal[3:]) + np.linalg.norm(phi[key] + lam * omega[key])
                diagnostics[key]['terminal_position_bound'].append(float(pb))
                diagnostics[key]['terminal_velocity_bound'].append(float(vb))
                admissible &= pb <= r.docking_position_tolerance and vb <= r.docking_velocity_tolerance
            dwell = dwell + 1 if admissible and valid else 0
            completed = dwell >= cfg.mpc.terminal_dwell_updates
            previous_plans = solutions
            if completed or not valid:
                break
        pr, pd, tr, td, active = r._safety_margins(actual, q_history, sample_phases, radius_history)
        # Robust margins concern a nominal center and its own matched tube.
        for key in ids:
            tr[key], td[key] = [], []
            for qi, center, bound in zip(q_history, diagnostics[key]['nominal_center'], diagnostics[key]['omega']):
                radius = np.linalg.norm(bound)
                tr[key].append(float(np.linalg.norm(np.asarray(center)[:3]) - r.target.rendezvous_radius - radius))
                radial, axial = r.target.cylinder_constraint(qi, tube_radius=radius).margins(np.asarray(center)[:3])
                td[key].append(float(max(radial, axial)))
        spacing = r._spacing_history(actual, q_history)
        safety = r._active_target_margins_safe(active) and dense_min_clearance >= -r.target_safety_tolerance
        spacing_ok = r._pairwise_spacing_safe(spacing) and dense_min_spacing >= r.min_chaser_separation - r.target_safety_tolerance
        if valid and not completed:
            messages.append('max_mpc_updates_reached_before_terminal_tube_dwell')
        return ADTMPCMissionResult(mission=mission, approximation=r.approximation, solver_backend='gekko',
            success=bool(valid and completed and safety and spacing_ok), message='; '.join(messages),
            anom_history=q_history, phase_history=phases, sample_phase_history=sample_phases,
            actual_trajectories=actual, nominal_trajectories=nominal, controls=controls,
            solve_stats=stats, solver_costs=costs, truth=r.truth.as_dict(), target=r.target.as_dict(),
            target_attitude_history=attitudes, target_angular_velocity_history=rates,
            docking_points={c.chaser_id:c.docking_point_body.tolist() for c in chasers},
            initial_belief=initial_belief.as_metadata(), final_belief=belief.as_metadata(),
            feasible_set_history=feasible, parameter_estimate_history=estimates, smid_records=smid_records,
            tube_profiles=profiles, tube_radius_history=radius_history, rendezvous_margins=pr,
            docking_cylinder_margins=pd, active_target_margins=active, pairwise_spacing=spacing,
            physical_rendezvous_margins=pr, physical_docking_cylinder_margins=pd,
            tightened_rendezvous_margins=tr, tightened_docking_cylinder_margins=td,
            tube_diagnostics=diagnostics,
            metadata=dict(model_version='physical-lvlh-v1', controller='coupled-adaptive-dynamic-tube',
                horizon_steps=r._configured_horizon_metadata(), actuation_steps=1, feedback_substeps=substeps,
                anom_step=cfg.anom_step, dense_anom_step=h, max_mpc_updates=int(steps), mission_complete=completed,
                lambda_gain=lam.tolist(), numerical_reserve=cfg.tube.numerical_reserve,
                membership_error_bound=cfg.tube.membership_error_bound,
                reference_preview_enabled=cfg.mpc.reference_preview_enabled, terminal_cost_enabled=cfg.mpc.terminal_cost_enabled,
                docking_position_tolerance=r.docking_position_tolerance, docking_velocity_tolerance=r.docking_velocity_tolerance,
                terminal_dwell_updates=cfg.mpc.terminal_dwell_updates, terminal_dwell_observed=dwell,
                active_target_margins_safe=bool(safety), pairwise_spacing_safe=bool(spacing_ok),
                min_chaser_separation=r.min_chaser_separation, sampled_tube_containment=bool(contained),
                dense_min_containment_margin=float(dense_min_containment) if np.isfinite(dense_min_containment) else None,
                dense_min_physical_clearance=float(dense_min_clearance) if np.isfinite(dense_min_clearance) else None,
                dense_min_spacing=float(dense_min_spacing) if np.isfinite(dense_min_spacing) else None,
                max_nominal_discretization_residual=max_nominal_discretization_residual,
                elapsed_time_seconds=elapsed, membership_records=membership_records,
                units=dict(position='m', velocity='m/rad anomaly', input='m/rad^2 anomaly',
                    physical_acceleration='qdot^2 times input', target_rates='Euler rad/rad anomaly'),
                target_attitude_model='prescribed Euler angles in true anomaly; not torque-free rigid-body simulation',
                guarantee_scope='Conditional model-box enclosure and measured sampled containment; no continuous-time collision or recursive-feasibility certificate',
                controls_semantics='Mean applied outer inputs; exact held ancillary substep inputs passed to SMID, squared effort from dense inputs',
                tightened_margin_semantics='Persistent nominal center with own matched-node position tube',
                terminal_dwell_semantics='Consecutive outer endpoint membership, with dense containment and physical safety throughout',
                chaser_assignments=[c.assignment_metadata() for c in chasers], primary_smid_chaser=ids[0]))

    def _record(self, data, state, center, phi, omega, bandwidth):
        error = state - center
        sliding = error[3:] + np.asarray(self.config.tube.sliding_gains) * error[:3]
        for key, value in dict(nominal_center=center, phi=phi, omega=omega, bandwidth=bandwidth,
                position_error=error[:3], sliding_error=sliding).items():
            data[key].append(np.asarray(value).tolist())
        data['containment_margin'].append(float(min(np.min(omega - np.abs(error[:3])), np.min(phi - np.abs(sliding)))))
