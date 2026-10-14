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

"""Coupled mission orchestration over explicit planning and execution services.

Held-input RK4, the persistent nominal center/tube, membership contraction and
whole-tube terminal dwell retain their qualified update order. Sampled safety
checks remain conditional model evidence, not a continuous-time certificate.
"""
from __future__ import annotations

from typing import Optional
import numpy as np

from orbexa.core.dynamics import orbital_ellp_drag
from orbexa.estimation.belief import ParameterBelief
from orbexa.estimation.mission_adaptation import MissionAdaptation, PointEstimator
from orbexa.simulation.coupled_execution import DenseFeedbackExecutor
from orbexa.simulation.coupled_planning import CoupledMissionPlanner
from orbexa.simulation.coupled_state import CoupledMissionState
from orbexa.simulation.integration import rk4_step  # Compatibility import.
from orbexa.simulation.mission_recording import MissionRecorder
from orbexa.simulation.mission_result import ADTMPCMissionResult
from orbexa.simulation.mission_scenario import ControllerFactory, MissionScenario


class CoupledTubeMission:
    """Compose mission services; ``CoupledTubeMission(runner)`` remains usable."""

    def __init__(self, scenario=None, *, config=None, target=None,
                 point_estimator: Optional[PointEstimator] = None,
                 controller_factory: Optional[ControllerFactory] = None):
        if scenario is not None and not isinstance(scenario, MissionScenario):
            # Compatibility adaptation does not retain or call the facade.
            scenario = getattr(scenario, 'scenario', None)
            if not isinstance(scenario, MissionScenario):
                raise ValueError('Provide a MissionScenario or a runner with a MissionScenario')
        if scenario is not None and any(value is not None for value in (config, target, point_estimator, controller_factory)):
            raise ValueError('Inject dependencies into the supplied MissionScenario')
        self.scenario = scenario if scenario is not None else MissionScenario(
            config=config, target=target, point_estimator=point_estimator, controller_factory=controller_factory)
        self.config = self.scenario.config
        self.scenario.require_coupled_execution()

    def run(self, *, mission: str, steps: int, checkpoint=None) -> ADTMPCMissionResult:
        if mission not in ('single', 'multi'):
            raise ValueError(f'Unknown ADTMPC mission: {mission}')
        if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 1:
            raise ValueError('steps must be a positive integer number of MPC updates')
        scenario, cfg = self.scenario, self.config
        if checkpoint is not None:
            from orbexa.simulation.mission_checkpoint import MissionCheckpoint
            if not isinstance(checkpoint, MissionCheckpoint):
                raise TypeError('Provide a source-qualified MissionCheckpoint')
            checkpoint.validate_scenario(scenario, mission)
            chasers, state, belief, phase, dwell = checkpoint.restore(scenario)
            recorder = MissionRecorder.from_checkpoint(scenario, chasers, state, belief, checkpoint)
            update_offset = checkpoint.committed_updates
        else:
            chasers = scenario.fleet.chasers(mission)
            state = CoupledMissionState.initialize(chasers, cfg)
            belief = (ParameterBelief.from_config_prior(cfg) if scenario.initial_belief is None
                      else scenario.initial_belief.copy())
            recorder = MissionRecorder(scenario, chasers, state, belief)
            phase, dwell, update_offset = scenario.reference.initial_phase(), 0, 0
            recorder.sample_phases[0] = phase.name
        controller = scenario.create_controller()
        planner = CoupledMissionPlanner(scenario, controller)
        adaptation = MissionAdaptation(cfg, scenario.truth, scenario.point_estimator,
                                       state.states[chasers[0].chaser_id], state.anomaly,
                                       mode=scenario.adaptation_mode)
        if checkpoint is not None:
            import copy
            adaptation.smid_records = copy.deepcopy(checkpoint.source_record['smid_records'])
            adaptation.membership_records = copy.deepcopy(checkpoint.source_record['metadata'].get('membership_records', []))
        executor = DenseFeedbackExecutor(scenario, adaptation)
        truth_matrices = orbital_ellp_drag(anom_step=cfg.anom_step, **scenario.truth.dynamics_params())[0]
        # Complete old control horizons and exact dense adaptation windows are
        # unavailable in recorded results. The numerical restart declares a
        # cold planner start and an explicitly empty dense measurement window.
        previous_plans = {}
        completed, valid = False, True
        messages = []
        for update in range(int(steps)):
            selected_phase = scenario.reference.select_phase(phase, state, chasers)
            if selected_phase != phase:
                phase = selected_phase
                recorder.sample_phases[-1] = phase.name
            recorder.phases.append(phase.name)
            # Plan the whole fleet and check simultaneous two-tube separation.
            planning = planner.plan(state, belief, phase, chasers, previous_plans)
            recorder.record_planning(planning, state.anomaly)
            messages.extend(planning.messages)
            valid = planning.success
            if not valid:
                break
            # Every dense transition commits all chasers on one shared clock.
            execution = executor.advance(state, planning, phase, truth_matrices)
            messages.extend(execution.messages)
            valid = execution.success
            if not execution.committed:
                break
            recorder.record_execution(state, execution, planning, phase, executor.dense_anom_step)
            # Membership contracts first. Point estimation cannot replace its box.
            updated = adaptation.update(belief, update_offset + update, executor.dense_anom_step, eligible=valid)
            valid = valid and updated.consistent
            messages.extend(updated.messages)
            recorder.record_belief(belief)
            # Completion concerns the full terminal tube over consecutive updates.
            admissible = recorder.record_terminal_bounds(state, phase)
            dwell = dwell + 1 if admissible and valid else 0
            completed = dwell >= cfg.mpc.terminal_dwell_updates
            previous_plans = planning.plans
            if completed or not valid:
                break
        return recorder.result(mission=mission, steps=update_offset + int(steps), belief=belief, adaptation=adaptation,
                               valid=valid, completed=completed, dwell=dwell, messages=messages)
