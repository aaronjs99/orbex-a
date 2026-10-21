# /***********************************************************
# *                                                         *
# * Copyright (c) 2026                                      *
# *                                                         *
# * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
# * University of California, Los Angeles                   *
# *                                                         *
# * Authors: Aaron John Sabu, Brett T. Lopez                *
# * Contact: {aaronjs, btlopez}@ucla.edu                    *
# *                                                         *
# ***********************************************************/

"""Coupled planning of all chasers with the existing dynamic-tube backend."""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Sequence, TYPE_CHECKING
import numpy as np

from orbexa.contracts import SolverResult
from orbexa.control.tubes.model import TubeProfile
from orbexa.control.tubes.specification import DynamicTubeSpec
from orbexa.control.tubes.sample_hold import TUBE_MODEL_VERSION, TUBE_ASSUMPTIONS
from orbexa.control.mpc.cost import local_terminal_cost
from orbexa.control.tubes.uncertainty import uncertainty_envelope, coefficient_intervals
from orbexa.core.dynamics.orbital import orbital_ellp_drag

if TYPE_CHECKING:
    from orbexa.estimation.adaptation.belief import ParameterBelief
    from orbexa.simulation.approach.state import CoupledMissionState
    from orbexa.simulation.approach.models import ChaserConfig, MissionPhase
    from orbexa.simulation.approach.scenario import CoupledController, MissionScenario


@dataclass
class ChaserPlan:
    result: SolverResult
    profile: TubeProfile
    spec: DynamicTubeSpec
    tube: Dict[str, Any]
    arrival_context: Dict[str, Any] | None = None


@dataclass
class PlanningStep:
    matrices: tuple
    parameters: Dict[str, float]
    plans: Dict[str, ChaserPlan] = field(default_factory=dict)
    attempts: Dict[str, SolverResult] = field(default_factory=dict)
    success: bool = True
    messages: List[str] = field(default_factory=list)


class CoupledMissionPlanner:
    """Form one shared update; preserve sequential warm starts and joint checks."""

    def __init__(self, scenario: MissionScenario, controller: CoupledController):
        self.scenario = scenario
        self.config = scenario.config
        self.controller = controller

    def tube_spec(self, belief: ParameterBelief, params: Dict[str, float], anomaly: float,
                  count: int, phi: np.ndarray, omega: np.ndarray, bandwidth: np.ndarray, *,
                  chaser_id: str | None = None, center: np.ndarray | None = None) -> DynamicTubeSpec:
        tube = self.config.tube
        semi_major_axis, mu = params['semi_major_axis'], params['mu']
        delta_a, delta_d, a_abs, a_variation, d_abs = [], [], [], [], []
        for stage in range(count - 1):
            lo, hi = anomaly + stage * self.config.anom_step, anomaly + (stage + 1) * self.config.anom_step
            da, dd = uncertainty_envelope(lo, hi, params,
                belief.feasible_sets['eccentricity'], belief.feasible_sets['alpha'],
                belief.feasible_sets['beta'], semi_major_axis, mu)
            delta_a.append(da)
            delta_d.append(dd)
            al, ah, dl, dh = coefficient_intervals(lo, hi, params, semi_major_axis)
            a_abs.append(np.maximum(np.abs(al), np.abs(ah)))
            # The outer-interval width encloses any change during one inner
            # hold. It is a coefficient change, so is not multiplied by tau.
            a_variation.append(np.nextafter(ah - al, np.inf))
            d_abs.append(np.maximum(np.abs(dl), np.abs(dh)))
        return DynamicTubeSpec(
            phi_initial=phi, omega_initial=omega, lambda_gain=tube.sliding_gains,
            alpha_initial=bandwidth, alpha_bounds=(tube.bandwidth_range['lower'], tube.bandwidth_range['upper']),
            alpha_rate_limit=tube.bandwidth_rate_limit, delta_A=np.asarray(delta_a), delta_d=np.asarray(delta_d),
            A_abs=np.asarray(a_abs), A_variation=np.asarray(a_variation), d_abs=np.asarray(d_abs),
            hold_width=self.config.anom_step / tube.feedback_substeps, eta=np.full(3, tube.numerical_reserve),
            bandwidth_weight=tube.bandwidth_weight, tube_weight=tube.tube_weight,
            terminal_position_tolerance=self.config.mpc.docking_position_tolerance,
            terminal_velocity_tolerance=self.config.mpc.docking_velocity_tolerance, enforce_terminal=False)

    def problem_bounds(self, chaser, state, spec):
        """Allow an explicit plant adapter to restrict its regional model domain."""
        return self.config.mpc.bounds()

    def geometry_reserve(self, anomaly: float, count: int) -> float:
        """Additional length reserve for a separately bounded target preview."""
        return 0.

    def tube_profile_assumptions(self, spec):
        return TUBE_ASSUMPTIONS

    def configure_terminal(self, spec, phase, state, goal, key):
        """Default local activation; explicit deadline adapters override this."""
        spec.enforce_terminal = bool(phase.operation == 'docking' and
            np.linalg.norm(state.centers[key][:3] - goal[:3]) <
            self.config.mission.terminal_activation_distance)

    def planning_peer_ids(self, chasers, state):
        return [chaser.chaser_id for chaser in chasers]

    def plan(self, state: CoupledMissionState, belief: ParameterBelief, phase: MissionPhase,
             chasers: Sequence[ChaserConfig], previous_plans: Dict[str, ChaserPlan], *, arrival_step=None) -> PlanningStep:
        scenario, cfg = self.scenario, self.config
        q = state.anomaly
        count = scenario.reference.horizon_steps_for_phase(phase)
        arrival_policy = scenario.arrival_policy
        if arrival_policy is not None and phase.operation == 'docking':
            from orbexa.simulation.approach.arrival.policy import ArrivalPlanningStep
            if (not isinstance(arrival_step, ArrivalPlanningStep)
                    or arrival_step.policy.fingerprint != arrival_policy.fingerprint):
                raise ValueError('Docking with an arrival policy requires its inherited exact deadline context')
            region = arrival_policy.required_terminal_region
            if region is not None and any(not region.supports_query(
                    arrival_step.exact_anomaly + k * arrival_step.exact_outer_step,
                    q + k * cfg.anom_step) for k in range(arrival_step.terminal_start_index, count)):
                raise ValueError('Arrival suffix queries exceed the finite reference-error clock/domain declaration')
        elif arrival_step is not None:
            raise ValueError('An arrival deadline context applies only to its enabled docking policy')
        params = belief.dynamics_params(scenario.truth)
        matrices = orbital_ellp_drag(anom_step=cfg.anom_step, **params)[0]
        outcome = PlanningStep(matrices=matrices, parameters=params)
        ids = self.planning_peer_ids(chasers,state)
        lam = np.asarray(cfg.tube.sliding_gains)
        for chaser in chasers:
            key = chaser.chaser_id
            spec = self.tube_spec(belief, params, q, count, state.phi[key], state.omega[key], state.bandwidth[key],
                                  chaser_id=key, center=state.centers[key])
            problem_bounds = self.problem_bounds(chaser, state, spec)
            goal = scenario.reference.phase_reference_state(phase, chaser, state.centers[key], q)
            references, feedforward = scenario.reference.phase_reference_preview(
                phase, chaser, goal, q, count, matrices, params['time_periapsis'])
            target = scenario.target.collision_params(phase.operation, q)
            rotation_at = getattr(scenario.target, 'rotation_matrix_at', None)
            if callable(rotation_at):
                target['rotation_matrices'] = [rotation_at(q + k * cfg.anom_step).tolist() for k in range(count)]
            else:
                target['orientations'] = [scenario.target.orientation_at(q + k * cfg.anom_step).tolist() for k in range(count)]
            target.update(robust_side_corridor=phase.operation == 'docking' and target.get('shape') == 'cylinder',
                docking_normal_body=chaser.docking_normal_body,
                safety_reserve=cfg.mission.target_barrier_clearance + scenario.safety.chaser_body_radius
                    + self.geometry_reserve(q, count))
            if target.get('shape') == 'composite':
                target.update(attachment_patch_id=chaser.attachment_patch_id,
                    attachment_component_id=chaser.attachment_component_id,
                    robust_attachment_corridor=phase.operation == 'docking')
            if phase.operation == 'rendezvous' and target.get('shape') != 'composite':
                target['rendezvous_radius'] += scenario.safety.chaser_body_radius
            # Arm an explicit terminal set only near the final approach.
            # Completion always checks that set, including the current tube.
            self.configure_terminal(spec, phase, state, goal, key)
            if arrival_step is not None:
                # The absolute deadline does not recede when a new horizon is
                # solved. A stricter planning subset leaves the outer gate intact.
                spec.enforce_terminal = True
                spec.terminal_start_index = arrival_step.terminal_start_index
                spec.terminal_planning_fraction = arrival_policy.planning_fraction
                if arrival_policy.required_terminal_region is not None:
                    spec.terminal_region = arrival_policy.required_terminal_region
            pairwise = []
            for other in ids:
                if other == key:
                    continue
                prior = outcome.plans.get(other, previous_plans.get(other))
                if prior is not None:
                    path = np.asarray(prior.result.state_trajectory).T
                    rad = np.asarray(prior.profile.position_radius)
                    shift = 0 if other in outcome.plans else 1
                    indices = np.minimum(np.arange(count) + shift, len(path) - 1)
                    path, rad = path[indices, :3], rad[indices]
                else:
                    path = np.array([state.centers[other][:3] + k * cfg.anom_step * state.centers[other][3:] for k in range(count)])
                    rad = np.full(count, np.linalg.norm(state.omega[other]))
                pairwise.append(dict(other_chaser=other, reference_positions=path, tube_radius=rad,
                                     min_separation=scenario.safety.min_chaser_separation))
            kwargs = dict(mean_motion=params['mean_motion'], state_cost_matrix=scenario.reference.phase_state_weights(phase),
                input_cost_matrix=cfg.mpc.R, dynamic_tube=spec.to_extra_params(), target_params=target,
                pairwise_constraints=pairwise)
            prior = previous_plans.get(key)
            if prior is not None:
                path = np.asarray(prior.result.state_trajectory)
                commands = np.asarray(prior.result.control_trajectory)
                kwargs['state_initial_guess'] = path[:, np.minimum(np.arange(count) + 1, path.shape[1] - 1)]
                kwargs['control_initial_guess'] = commands[:, np.minimum(np.arange(count - 1) + 1, commands.shape[1] - 1)]
            if cfg.mpc.reference_preview_enabled or spec.enforce_terminal:
                kwargs.update(state_reference=references, input_reference=feedforward)
            if cfg.mpc.terminal_cost_enabled:
                kwargs['terminal_cost_matrix'] = local_terminal_cost(matrices[0](q, params['time_periapsis']),
                    matrices[1](), kwargs['state_cost_matrix'], cfg.mpc.R, cfg.anom_step)
            solve_started = time.perf_counter()
            try:
                result = self.controller.solve_step(
                    initial_state=state.centers[key], final_state=goal, control_input_0=feedforward[:, 0],
                    start_anom=q, anom_step=cfg.anom_step, num_steps=count, dynamics=matrices, bounds=problem_bounds,
                    time_periapsis=params['time_periapsis'], eccentricity=params['eccentricity'],
                    use_anomaly_scaling=False, **kwargs)
            except (ValueError, FloatingPointError, OverflowError) as error:
                # Unsupported numeric setup or a non-finite initial iterate is
                # a recorded failed attempt, never an infeasibility theorem or
                # an unrecorded mission exit. Programming errors still surface.
                result = SolverResult(success=False, solve_time=time.perf_counter()-solve_started,
                    message=f'Coupled MPC numeric setup failed: {error}',
                    solver_info=dict(failure_stage='numeric_problem_setup', exception_type=type(error).__name__))
            outcome.attempts[key] = result
            if not result.success:
                outcome.success = False
                outcome.messages.append(f'{key}: {result.message}')
                break
            tube = result.solver_info['dynamic_tube']
            profile = TubeProfile(phi=np.asarray(tube['phi']), omega=np.asarray(tube['omega']),
                time_grid=q + cfg.anom_step * np.arange(count), lambda_gain=lam,
                alpha=np.asarray(tube['alpha']), input_reserve=np.asarray(tube['input_reserve']),
                assumptions=self.tube_profile_assumptions(spec), tube_model_version=tube.get('tube_model_version', TUBE_MODEL_VERSION),
                hold_width=spec.hold_width,
                forcing={name: np.asarray(tube[name]) for name in (
                    'base_forcing', 'hold_forcing', 'disturbance_bound', 'nominal_enclosure',
                    'phi_peak', 'omega_peak', 'velocity_peak', 'error_peak',
                    'velocity_derivative_bound', 'error_derivative_bound', 'sliding_derivative_bound',
                    'A_variation','nominal_reset_forcing_phi','nominal_reset_forcing_omega') if name in tube} | ({'measurement_forcing': np.asarray(tube['measurement_forcing'])}
                        if 'measurement_forcing' in tube else {}))
            arrival_context = None
            if arrival_step is not None:
                arrival_context = dict(reference_states=references, reference_inputs=feedforward,
                    target_params=target, pairwise_constraints=pairwise, bounds=problem_bounds,
                    state_cost_matrix=kwargs['state_cost_matrix'], input_cost_matrix=cfg.mpc.R,
                    terminal_cost_matrix=kwargs.get('terminal_cost_matrix'),
                    solver_options=cfg.solver.options(), start_anomaly=q, anomaly_step=cfg.anom_step)
            outcome.plans[key] = ChaserPlan(result=result, profile=profile, spec=spec, tube=tube,
                                           arrival_context=arrival_context)
        if outcome.success:
            outcome.messages.extend(scenario.safety.planned_separation_messages(outcome.plans))
            outcome.success = not outcome.messages
        return outcome
