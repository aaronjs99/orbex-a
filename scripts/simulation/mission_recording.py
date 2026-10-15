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

"""Mission history and diagnostic recording, separate from solver execution."""
from __future__ import annotations

from typing import Any, Dict, Sequence, TYPE_CHECKING
import copy
import numpy as np

from orbexa.contracts import COUPLED_CONTROLLER_ID, PHYSICAL_MODEL_ID
from orbexa.control.dynamic_tube_model import input_tightening_from_profile
from orbexa.control.sample_hold import TUBE_MODEL_VERSION, TUBE_ASSUMPTIONS
from orbexa.estimation.belief import ParameterBelief, serial_estimates, serial_ranges
from orbexa.simulation.mission_result import ADTMPCMissionResult
from orbexa.utils.anomaly import orbital_anomaly_rate

if TYPE_CHECKING:
    from orbexa.estimation.belief import ParameterBelief
    from orbexa.estimation.mission_adaptation import MissionAdaptation
    from orbexa.simulation.coupled_execution import ExecutionStep
    from orbexa.simulation.coupled_planning import PlanningStep
    from orbexa.simulation.coupled_state import CoupledMissionState
    from orbexa.simulation.mission_models import ChaserConfig, MissionPhase
    from orbexa.simulation.mission_scenario import MissionScenario


class MissionRecorder:
    """Retain every solve attempt and every committed dense interval endpoint."""

    def __init__(self, scenario: MissionScenario, chasers: Sequence[ChaserConfig],
                 state: CoupledMissionState, initial_belief: ParameterBelief):
        self.scenario, self.config, self.chasers = scenario, scenario.config, chasers
        self.initial_belief = initial_belief.copy()
        self.ids = [c.chaser_id for c in chasers]
        self.actual, self.nominal, self.controls, self.stats, self.costs, self.profiles = (
            {key: [] for key in self.ids} for _ in range(6)
        )
        self.diagnostics = {key: dict(nominal_center=[], phi=[], omega=[], bandwidth=[],
            position_error=[], sliding_error=[], containment_margin=[], terminal_position_bound=[],
            terminal_velocity_bound=[], max_dense_input_reserve_ratio=[], nominal_control=[],
            ancillary_control_rms=[], applied_control_rms=[], applied_control_effort=[],
            dense_containment_margin=[], dense_physical_clearance=[], nominal_center_defect=[]) for key in self.ids}
        for key in self.ids:
            self.actual[key].append(state.states[key].tolist())
            self.record_tube_sample(self.diagnostics[key], state.states[key], state.centers[key],
                                    state.phi[key], state.omega[key], state.bandwidth[key])
        q = state.anomaly
        self.q_history, self.elapsed = [q], [0.0]
        self.phases, self.sample_phases = [], ['rendezvous']
        self.radius_history = [0.0]
        self.attitudes = [scenario.target.orientation_at(q).tolist()]
        # A proper-rotation target need not expose a nonsingular Euler-rate
        # chart. Its authoritative matrices/spins are recorded at result time.
        self.records_kinematics = callable(getattr(scenario.target, 'recorded_kinematics', None))
        self.rates = [] if self.records_kinematics else [scenario.target.angular_velocity_at(q).tolist()]
        self.feasible = [serial_ranges(initial_belief.feasible_sets)]
        self.estimates = [serial_estimates(initial_belief.estimates)]
        self.contained = True
        self.dense_min_clearance, self.dense_min_spacing, self.dense_min_containment = np.inf, np.inf, np.inf
        self.max_nominal_discretization_residual = 0.0
        self.numerical_restart_history = []
        self.source_uncommitted_planning = {}
        self.source_kinematics = None
        self.source_kinematic_nodes = 0
        self.arrival_runtime = None
        self.policy_transition_history = []

    @classmethod
    def from_checkpoint(cls, scenario, chasers, state, belief, checkpoint, *, preserve_source_stats=False):
        """Retain the complete committed prefix and every original solve attempt."""
        source = checkpoint.source_record
        metadata, count = source['metadata'], checkpoint.committed_updates
        recorder = cls(scenario, chasers, state, belief)
        recorder.initial_belief = ParameterBelief(**copy.deepcopy(source['initial_belief']))
        recorder.actual = copy.deepcopy(source['actual_trajectories'])
        recorder.controls = copy.deepcopy(source['controls'])
        recorder.stats = copy.deepcopy(source['solve_stats'])
        recorder.costs = copy.deepcopy(source['solver_costs'])
        recorder.nominal, recorder.profiles = {}, {}
        recorder.source_uncommitted_planning = copy.deepcopy(metadata.get('source_uncommitted_planning', {}))
        archived = {}
        for key in recorder.ids:
            # Unexecuted proposals from a failed fleet cycle are archived, not
            # inserted as if they drove the following committed trajectory.
            recorder.nominal[key] = copy.deepcopy(source['nominal_trajectories'][key][:count])
            recorder.profiles[key] = copy.deepcopy(source['tube_profiles'][key][:count])
            archived[key] = dict(
                nominal_trajectories=copy.deepcopy(source['nominal_trajectories'][key][count:]),
                tube_profiles=copy.deepcopy(source['tube_profiles'][key][count:]))
            if not preserve_source_stats:
                for index, attempt in enumerate(recorder.stats[key]):
                    attempt.setdefault('anomaly', source['anom_history'][min(index, count)])
                    attempt.setdefault('source_checkpoint_identity', checkpoint.identity)
        recorder.source_uncommitted_planning[checkpoint.identity] = archived
        recorder.diagnostics = copy.deepcopy(source['tube_diagnostics'])
        recorder.q_history = list(source['anom_history'])
        recorder.phases = list(source['phase_history'][:count])
        recorder.sample_phases = list(source['sample_phase_history'])
        recorder.elapsed = list(metadata['elapsed_time_seconds'])
        recorder.source_kinematics = copy.deepcopy(metadata.get('target_kinematics'))
        recorder.source_kinematic_nodes = len(source['anom_history'])
        recorder.radius_history = list(source['tube_radius_history'])
        recorder.attitudes = copy.deepcopy(source['target_attitude_history'])
        recorder.rates = copy.deepcopy(source['target_angular_velocity_history'])
        recorder.feasible = copy.deepcopy(source['feasible_set_history'])
        recorder.estimates = copy.deepcopy(source['parameter_estimate_history'])
        recorder.contained = bool(metadata['sampled_tube_containment'])
        for attr, key in (('dense_min_clearance', 'dense_min_physical_clearance'),
                          ('dense_min_spacing', 'dense_min_spacing'),
                          ('dense_min_containment', 'dense_min_containment_margin')):
            value = metadata.get(key)
            setattr(recorder, attr, np.inf if value is None else float(value))
        recorder.max_nominal_discretization_residual = float(metadata.get('max_nominal_discretization_residual', 0.))
        recorder.numerical_restart_history = copy.deepcopy(metadata.get('numerical_restart_history', []))
        recorder.policy_transition_history = copy.deepcopy(metadata.get('policy_transition_history', []))
        recorder.numerical_restart_history.append(checkpoint.restart_record(scenario))
        recorder.numerical_restart_history[-1]['uncommitted_phase_attempts'] = list(source['phase_history'][count:])
        return recorder

    def tube_metadata(self, anom: float, profile) -> Dict[str, Any]:
        return {
            'start_anom': float(anom), 'time_grid': profile.time_grid.tolist(),
            'phi': profile.phi.tolist(), 'omega': profile.omega.tolist(),
            'position_radius': profile.position_radius.tolist(),
            'max_position_radius': profile.max_position_radius,
            'max_state_error': profile.max_state_error.tolist(),
            'lambda_gain': None if profile.lambda_gain is None else np.asarray(profile.lambda_gain).tolist(),
            'bandwidth': None if profile.alpha is None else np.asarray(profile.alpha).tolist(),
            'input_reserve': None if profile.input_reserve is None else np.asarray(profile.input_reserve).tolist(),
            'assumptions': list(profile.assumptions),
            'tube_model_version': profile.tube_model_version,
            'hold_width': profile.hold_width,
            'stage_forcing': {} if profile.forcing is None else
                {name: np.asarray(value).tolist() for name,value in profile.forcing.items()},
            'input_tightening': input_tightening_from_profile(profile, self.config.tube.sliding_gains).tolist(),
        }

    def record_tube_sample(self, data, state, center, phi, omega, bandwidth) -> None:
        error = state - center
        sliding = error[3:] + np.asarray(self.config.tube.sliding_gains) * error[:3]
        for key, value in dict(nominal_center=center, phi=phi, omega=omega, bandwidth=bandwidth,
                position_error=error[:3], sliding_error=sliding).items():
            data[key].append(np.asarray(value).tolist())
        data['containment_margin'].append(float(min(np.min(omega - np.abs(error[:3])), np.min(phi - np.abs(sliding)))))

    def record_planning(self, planning: PlanningStep, anomaly: float) -> None:
        for key, result in planning.attempts.items():
            self.stats[key].append(dict(anomaly=float(anomaly), success=bool(result.success), solve_time=float(result.solve_time), message=result.message,
                solver_info={name: value for name, value in result.solver_info.items() if name != 'dynamic_tube'}))
            if not result.success and 'dynamic_tube' in result.solver_info:
                self.stats[key][-1]['failed_dynamic_tube'] = {
                    name: value.tolist() if isinstance(value, np.ndarray) else value
                    for name,value in result.solver_info['dynamic_tube'].items()}
            self.costs[key].append(result.cost)
            if key in planning.plans:
                self.nominal[key].append(np.asarray(result.state_trajectory).T.tolist())
                self.profiles[key].append(self.tube_metadata(anomaly, planning.plans[key].profile))

    def record_execution(self, state: CoupledMissionState, step: ExecutionStep,
                         planning: PlanningStep, phase: MissionPhase, dense_anom_step: float) -> None:
        if not step.committed:
            return
        self.contained &= step.sampled_tube_contained
        self.dense_min_containment = min(self.dense_min_containment, min(step.containment.values()))
        self.dense_min_clearance = min(self.dense_min_clearance, min(step.clearance.values()))
        self.dense_min_spacing = min(self.dense_min_spacing, step.min_spacing)
        self.max_nominal_discretization_residual = max(self.max_nominal_discretization_residual,
                                                      step.max_nominal_discretization_residual)
        for key in self.ids:
            sol = planning.plans[key]
            self.actual[key].append(state.states[key].tolist())
            # Display averages; adaptation receives the exact dense held inputs.
            self.controls[key].append(np.mean(step.applied[key], axis=0).tolist())
            self.record_tube_sample(self.diagnostics[key], state.states[key], state.centers[key],
                                    state.phi[key], state.omega[key], state.bandwidth[key])
            data = self.diagnostics[key]
            data['max_dense_input_reserve_ratio'].append(step.reserve_ratios[key])
            data['nominal_control'].append(np.asarray(sol.result.control_trajectory)[:, 0].tolist())
            data['ancillary_control_rms'].append(np.sqrt(np.mean(np.square(step.ancillary[key]), axis=0)).tolist())
            data['applied_control_rms'].append(np.sqrt(np.mean(np.square(step.applied[key]), axis=0)).tolist())
            data['applied_control_effort'].append(float(dense_anom_step * np.sum(np.square(step.applied[key]))))
            data['dense_containment_margin'].append(float(step.containment[key]))
            data['dense_physical_clearance'].append(float(step.clearance[key]))
            data['nominal_center_defect'].append(None if key not in step.nominal_center_defect
                else step.nominal_center_defect[key].tolist())
        truth = self.scenario.truth
        rate0 = orbital_anomaly_rate(step.start_anomaly, **{k: v for k, v in truth.dynamics_params().items()
            if k in ('eccentricity', 'alpha', 'mu', 'semi_major_axis', 'mean_motion', 'specific_angular_momentum')})
        rate1 = orbital_anomaly_rate(step.end_anomaly, eccentricity=truth.eccentricity, alpha=truth.alpha,
                                    mu=truth.mu, semi_major_axis=truth.semi_major_axis)
        self.elapsed.append(self.elapsed[-1] + (step.end_anomaly - step.start_anomaly) * (1 / rate0 + 1 / rate1) / 2)
        q = state.anomaly
        self.q_history.append(float(q))
        self.sample_phases.append(phase.name)
        self.attitudes.append(self.scenario.target.orientation_at(q).tolist())
        if not self.records_kinematics:
            self.rates.append(self.scenario.target.angular_velocity_at(q).tolist())
        self.radius_history.append(max(float(np.linalg.norm(state.omega[key])) for key in self.ids))

    def record_belief(self, belief: ParameterBelief) -> None:
        self.feasible.append(serial_ranges(belief.feasible_sets))
        self.estimates.append(serial_estimates(belief.estimates))

    def recorded_target_kinematics(self):
        """Retain source-prefix facts when projecting a resumed recording."""
        target = self.scenario.target
        if self.source_kinematics is None:
            return target.recorded_kinematics(self.q_history) if self.records_kinematics else None
        kinematics = copy.deepcopy(self.source_kinematics)
        new_clock = self.q_history[self.source_kinematic_nodes:]
        if new_clock:
            additions = target.recorded_kinematics(new_clock)
            kinematics = {name: np.concatenate((np.asarray(values), np.asarray(additions[name])), axis=0)
                          for name, values in kinematics.items()}
        return kinematics

    def emit_progress(self, kind: str, *, result=None) -> None:
        """Publish a detached display snapshot, never an unexecuted proposal.

        A blocking sink may pause execution here, between fleet updates. No
        plotting or projection work runs when the caller supplied no sink.
        """
        sink = self.scenario.progress_sink
        if sink is None:
            return
        if kind not in ('start', 'committed', 'complete'):
            raise ValueError('Unknown committed mission progress event')
        from orbexa.visualization.studio_data import StudioDataset
        if result is None:
            kinematics = self.recorded_target_kinematics()
            elapsed = (self.elapsed if kinematics is None else
                       np.asarray(kinematics['physical_time'], dtype=float).tolist())
            cfg = self.config
            record = dict(anom_history=self.q_history, actual_trajectories=self.actual,
                sample_phase_history=self.sample_phases, target=self.scenario.target.as_dict(),
                target_attitude_history=self.attitudes, target_angular_velocity_history=self.rates,
                docking_points={c.chaser_id: c.docking_point_body.tolist() for c in self.chasers},
                tube_diagnostics=self.diagnostics, success=False, message='approach_in_progress',
                metadata=dict(model_version=PHYSICAL_MODEL_ID, controller=COUPLED_CONTROLLER_ID,
                    scenario=self.scenario.scenario_metadata(), target_kinematics=kinematics,
                    elapsed_time_seconds=elapsed, mission_complete=False,
                    lambda_gain=np.asarray(cfg.tube.sliding_gains).tolist(),
                    docking_position_tolerance=cfg.mpc.docking_position_tolerance,
                    min_chaser_separation=self.scenario.safety.min_chaser_separation,
                    chaser_assignments=[c.assignment_metadata() for c in self.chasers]))
            projection = StudioDataset(record).to_dict()
            projection['outcome']['label'] = 'Computing approach'
        else:
            projection = StudioDataset(result).to_dict()
        sink(copy.deepcopy(dict(kind=kind, update=len(self.q_history) - 1,
            anomaly=float(self.q_history[-1]), projection=projection)))

    def record_terminal_bounds(self, state: CoupledMissionState, phase: MissionPhase) -> bool:
        admissible = phase.operation == 'docking'
        for chaser in self.chasers:
            key = chaser.chaser_id
            pb, vb = self.scenario.reference.terminal_bounds(chaser, state.anomaly, state.centers[key],
                                                            state.phi[key], state.omega[key])
            self.diagnostics[key]['terminal_position_bound'].append(float(pb))
            self.diagnostics[key]['terminal_velocity_bound'].append(float(vb))
            admissible &= self.scenario.reference.terminal_bounds_admissible(pb, vb)
        if self.arrival_runtime is not None and phase.operation == 'docking':
            region = self.arrival_runtime.policy.required_terminal_region
            if region is not None:
                from orbexa.simulation.arrival_policy import exact_mission_clock
                exact_q = exact_mission_clock(self.scenario, state)
                guard = self.scenario.execution_guard
                entry = {chaser.chaser_id: (chaser.chaser_id in guard.endpoints and
                    region.contains_endpoint(guard.endpoints[chaser.chaser_id],
                        self.scenario.target.docking_state(chaser.docking_point_body, state.anomaly), exact_q,
                        query_anomaly=state.anomaly))
                    for chaser in self.chasers}
                admissible &= all(entry.values())
                self.arrival_runtime.events.append(dict(status='REQUIRED_ENDPOINT_REGION_SCREEN',
                    exact_anomaly=str(exact_q), contained=entry,
                    scope='Carried endpoint membership only; physical terminal certificate and fleet admission remain separate'))
        return bool(admissible)

    def result(self, *, mission: str, steps: int, belief: ParameterBelief, adaptation: MissionAdaptation,
               valid: bool, completed: bool, dwell: int, messages: list) -> ADTMPCMissionResult:
        scenario, cfg, target, safety = self.scenario, self.config, self.scenario.target, self.scenario.safety
        pr, pd, tr, td, active = safety.safety_margins(self.actual, self.q_history, self.sample_phases, self.radius_history)
        # Robust margins concern each nominal center and its own matched tube.
        for key in self.ids:
            tr[key], td[key] = [], []
            for qi, center, bound in zip(self.q_history, self.diagnostics[key]['nominal_center'], self.diagnostics[key]['omega']):
                radius = np.linalg.norm(bound) + safety.chaser_body_radius
                if getattr(target, 'geometry', None) is not None:
                    margin = safety.composite_margin(np.asarray(center)[:3], qi, radius)
                    tr[key].append(margin); td[key].append(margin)
                    continue
                tr[key].append(float(np.linalg.norm(np.asarray(center)[:3]) - target.rendezvous_radius - radius))
                radial, axial = target.cylinder_constraint(qi, tube_radius=radius).margins(np.asarray(center)[:3])
                td[key].append(float(max(radial, axial)))
        spacing = safety.spacing_history(self.actual, self.q_history)
        target_safe = safety.active_target_margins_safe(active) and self.dense_min_clearance >= -safety.target_safety_tolerance
        spacing_ok = safety.pairwise_spacing_safe(spacing) and self.dense_min_spacing >= safety.min_chaser_separation - safety.target_safety_tolerance
        if valid and not completed:
            messages.append('max_mpc_updates_reached_before_terminal_tube_dwell')
        scenario_metadata = scenario.scenario_metadata()
        target_metadata = target.as_dict()
        kinematics = self.recorded_target_kinematics()
        if kinematics is not None:
            self.elapsed = np.asarray(kinematics['physical_time'], dtype=float).tolist()
        # Assignment belongs to the scenario/fleet, while the target geometry
        # may retain its default assignment description for other scenarios.
        deployment = scenario_metadata.get('deployment', {})
        if deployment.get('assignment_strategy'):
            target_metadata['assignment_strategy'] = deployment['assignment_strategy']
        from orbexa.simulation.mission_checkpoint import scenario_checkpoint_context
        return ADTMPCMissionResult(mission=mission, approximation=scenario.approximation, solver_backend='gekko',
            success=bool(valid and completed and target_safe and spacing_ok), message='; '.join(messages),
            anom_history=self.q_history, phase_history=self.phases, sample_phase_history=self.sample_phases,
            actual_trajectories=self.actual, nominal_trajectories=self.nominal, controls=self.controls,
            solve_stats=self.stats, solver_costs=self.costs, truth=scenario.truth.as_dict(), target=target_metadata,
            target_attitude_history=self.attitudes, target_angular_velocity_history=self.rates,
            docking_points={c.chaser_id: c.docking_point_body.tolist() for c in self.chasers},
            initial_belief=self.initial_belief.as_metadata(), final_belief=belief.as_metadata(),
            feasible_set_history=self.feasible, parameter_estimate_history=self.estimates,
            smid_records=adaptation.smid_records, tube_profiles=self.profiles, tube_radius_history=self.radius_history,
            rendezvous_margins=pr, docking_cylinder_margins=pd, active_target_margins=active, pairwise_spacing=spacing,
            physical_rendezvous_margins=pr, physical_docking_cylinder_margins=pd,
            tightened_rendezvous_margins=tr, tightened_docking_cylinder_margins=td, tube_diagnostics=self.diagnostics,
            metadata=dict(model_version=PHYSICAL_MODEL_ID, controller=COUPLED_CONTROLLER_ID,
                checkpoint_context=scenario_checkpoint_context(scenario),
                numerical_restart_history=self.numerical_restart_history,
                numerical_restart=bool(self.numerical_restart_history),
                source_uncommitted_planning=self.source_uncommitted_planning,
                tube_model_version=TUBE_MODEL_VERSION, tube_assumptions=list(TUBE_ASSUMPTIONS),
                scenario=scenario_metadata,
                target_kinematics=kinematics,
                continuous_execution_certificate=(None if scenario.execution_guard is None else scenario.execution_guard.as_metadata()),
                horizon_steps=scenario.reference.configured_horizon_metadata(), actuation_steps=1,
                feedback_substeps=cfg.tube.feedback_substeps, anom_step=cfg.anom_step,
                input_bounds=[dict(lower=float(bound['lower']), upper=float(bound['upper']))
                              for bound in cfg.mpc.input_bounds],
                dense_anom_step=cfg.anom_step / cfg.tube.feedback_substeps,
                max_mpc_updates=int(steps), mission_complete=completed,
                lambda_gain=np.asarray(cfg.tube.sliding_gains).tolist(), numerical_reserve=cfg.tube.numerical_reserve,
                membership_error_bound=cfg.tube.membership_error_bound,
                reference_preview_enabled=cfg.mpc.reference_preview_enabled, terminal_cost_enabled=cfg.mpc.terminal_cost_enabled,
                terminal_cost_stage_weights='State/input stage costs at interval starts, DARE with hQ and hR, multiplier 1; tube regularization also at terminal node',
                docking_position_tolerance=cfg.mpc.docking_position_tolerance, docking_velocity_tolerance=cfg.mpc.docking_velocity_tolerance,
                terminal_dwell_updates=cfg.mpc.terminal_dwell_updates, terminal_dwell_observed=dwell,
                active_target_margins_safe=bool(target_safe), pairwise_spacing_safe=bool(spacing_ok),
                min_chaser_separation=safety.min_chaser_separation, sampled_tube_containment=bool(self.contained),
                dense_min_containment_margin=float(self.dense_min_containment) if np.isfinite(self.dense_min_containment) else None,
                dense_min_physical_clearance=float(self.dense_min_clearance) if np.isfinite(self.dense_min_clearance) else None,
                dense_min_spacing=float(self.dense_min_spacing) if np.isfinite(self.dense_min_spacing) else None,
                max_nominal_discretization_residual=self.max_nominal_discretization_residual,
                nominal_center_defect_semantics='Dense RK4 center minus outer shooting center, components [m,m,m,m/rad,m/rad,m/rad]; reported without an enclosure transfer',
                elapsed_time_seconds=self.elapsed, membership_records=adaptation.membership_records,
                units=dict(position='m', velocity='m/rad anomaly', input='m/rad^2 anomaly',
                    physical_acceleration='qdot^2 times input', target_rates='Euler rad/rad anomaly',
                    target_clearance='m; maximum of radial and axial length margins'),
                target_attitude_model=target_metadata.get('attitude_model', 'prescribed Euler angles in true anomaly'),
                guarantee_scope=('Held-tube comparison plus continuously validated orbital-flow and swept finite-body admission under the declared sensing/model/input/clock/target bounds; ordinary approach has no stored continuation on solver failure'
                    if scenario.execution_guard is not None else
                    'Derived held-input continuous-model comparison conditional on stage bounds and assumed acceleration disturbance; measured sampled containment; no RK4 remainder, continuous-time collision or recursive-feasibility certificate'),
                controls_semantics='Mean applied outer inputs; exact held ancillary substep inputs passed to SMID, squared effort from dense inputs',
                tightened_margin_semantics='Persistent nominal center with own matched-node position tube',
                terminal_dwell_semantics='Consecutive outer endpoint membership, with dense containment and physical safety throughout',
                chaser_assignments=[c.assignment_metadata() for c in self.chasers], primary_smid_chaser=self.ids[0],
                **({'arrival_state': self.arrival_runtime.as_metadata(),
                    'arrival_activation': self.arrival_runtime.activation_marker()} if self.arrival_runtime is not None else {}),
                **({'policy_transition_history': copy.deepcopy(self.policy_transition_history)} if self.policy_transition_history else {})))
