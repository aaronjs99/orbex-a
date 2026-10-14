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

"""Compatibility facade for configured adaptive dynamic-tube missions.

Plain models and result I/O live in their own modules. Mission execution is
composed through MissionScenario rather than through runner implementation
helpers. Historical model and saved-result imports remain available here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from orbexa.core.config import SimulationConfig
from orbexa.core.orbital_parameters import (
    ADTMPC_PARAMETER_KEYS, OrbitalPlantTruth,
    specific_angular_momentum as _specific_angular_momentum,
)
from orbexa.core.target_geometry import TargetGeometry, TumblingCylinderTarget
from orbexa.estimation.belief import (
    ParameterBelief, serial_estimates as _serial_estimates, serial_ranges as _serial_ranges,
)
from orbexa.estimation.mission_adaptation import PointEstimator
from orbexa.simulation.mission_models import ChaserConfig, MissionPhase
from orbexa.simulation.mission_result import (
    ADTMPCMissionResult, MissionResultStore, load_adtmpc_mission_result, result_chaser_ids,
)
from orbexa.simulation.mission_scenario import ControllerFactory, MissionScenario


class ADTMPCMissionRunner:
    """Small entrypoint with explicit scenario and dependency injection."""

    def __init__(self, *, solver_backend: str = 'gekko', approximation: str = 'nonlinear',
                 config: Optional[SimulationConfig] = None, horizon_steps: Optional[int] = None,
                 actuation_steps: Optional[int] = None, scenario: Optional[MissionScenario] = None,
                 target: Optional[TargetGeometry] = None, point_estimator: Optional[PointEstimator] = None,
                 controller_factory: Optional[ControllerFactory] = None):
        if scenario is not None:
            if any(value is not None for value in (config, horizon_steps, actuation_steps, target, point_estimator, controller_factory)):
                raise ValueError('Inject dependencies into the supplied MissionScenario')
            if solver_backend != 'gekko' and solver_backend != scenario.solver_backend:
                raise ValueError('solver_backend conflicts with the supplied MissionScenario')
            if approximation != 'nonlinear' and approximation != scenario.approximation:
                raise ValueError('approximation conflicts with the supplied MissionScenario')
            self.scenario = scenario
        else:
            self.scenario = MissionScenario(
                config=config, target=target, point_estimator=point_estimator,
                controller_factory=controller_factory, horizon_steps=horizon_steps,
                actuation_steps=actuation_steps, solver_backend=solver_backend, approximation=approximation)

    def run(self, *, mission: str, steps: int, checkpoint=None) -> ADTMPCMissionResult:
        from orbexa.simulation.coupled_mission import CoupledTubeMission
        return CoupledTubeMission(self.scenario).run(mission=mission, steps=steps, checkpoint=checkpoint)

    def resume(self, checkpoint, *, steps: int) -> ADTMPCMissionResult:
        """Continue an unfinished saved mission for additional numerical updates."""
        from orbexa.simulation.mission_checkpoint import MissionCheckpoint
        if not isinstance(checkpoint, MissionCheckpoint):
            raise TypeError('Provide a source-qualified MissionCheckpoint')
        return self.run(mission=checkpoint.mission, steps=steps, checkpoint=checkpoint)

    def save_result(self, result: ADTMPCMissionResult, output_dir: Path) -> Path:
        return MissionResultStore.save(result, output_dir)

    @staticmethod
    def load_result(path: Path) -> ADTMPCMissionResult:
        return MissionResultStore.load(path)

    def __getattr__(self, name):
        # Keep previously exposed runner data and helper calls usable; the engine
        # itself depends only on the public scenario services.
        scenario = self.__dict__.get('scenario')
        if scenario is None:
            raise AttributeError(name)
        if name in {'config', 'truth', 'target', 'solver_backend', 'approximation', 'actuation_steps'}:
            return getattr(scenario, name)
        if name == 'horizon_steps':
            return scenario.reference.horizon_steps
        if name == 'smid':
            return scenario.point_estimator
        mission_aliases = {'rendezvous_standoff', 'rendezvous_capture_radius', 'rendezvous_position_tolerance',
                           'target_safety_tolerance', 'target_barrier_buffer', 'target_barrier_clearance'}
        if name in mission_aliases:
            return getattr(scenario.config.mission, name)
        if name in {'docking_position_tolerance', 'docking_velocity_tolerance', 'min_chaser_separation'}:
            return getattr(scenario.config.mpc, name)
        fleet_methods = {'_single_chaser', '_multi_chasers', '_multi_initial_states', '_build_chaser_configs'}
        reference_methods = {'_phase_reference_state', '_phase_reference_preview', '_rendezvous_goal_satisfied',
                             '_phase_state_weights', '_horizon_steps_for_phase', '_configured_horizon_metadata'}
        safety_methods = {'_clip_control', '_active_margin_for_state', '_safety_margins',
                          '_active_target_margins_safe', '_pairwise_spacing_safe', '_spacing_history'}
        for service, names in ((scenario.fleet, fleet_methods), (scenario.reference, reference_methods),
                               (scenario.safety, safety_methods)):
            if name in names:
                return getattr(service, name[1:])
        if name == '_solver_config':
            return scenario.config.solver.options
        if name == '_smid_context':
            return lambda: {'mean_motion': scenario.truth.mean_motion, 'mu': scenario.truth.mu,
                            'semi_major_axis': scenario.truth.semi_major_axis,
                            'time_periapsis': scenario.truth.time_periapsis}
        raise AttributeError(name)


__all__ = ['ADTMPCMissionRunner', 'ADTMPCMissionResult', 'MissionScenario', 'ChaserConfig', 'MissionPhase',
           'OrbitalPlantTruth', 'ParameterBelief', 'TumblingCylinderTarget',
           'load_adtmpc_mission_result', 'result_chaser_ids']
