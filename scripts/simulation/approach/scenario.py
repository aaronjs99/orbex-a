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

"""Explicit dependencies and policy services for a coupled tube mission."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Protocol
import copy
import json
import numpy as np

from orbexa.core.configuration.model import SimulationConfig
from orbexa.core.dynamics.parameters import OrbitalPlantTruth
from orbexa.core.targets.prescribed import TargetGeometry, TumblingCylinderTarget
from orbexa.estimation.adaptation.belief import ParameterBelief
from orbexa.estimation.adaptation.mission import PointEstimator, default_point_estimator
from orbexa.simulation.approach.fleet import FleetInitializer
from orbexa.simulation.approach.references.policy import MissionReferencePolicy
from orbexa.simulation.approach.safety import MissionSafety


class CoupledController(Protocol):
    solver_backend: str

    def solve_step(self, **kwargs: Any) -> Any: ...


class ControllerFactory(Protocol):
    def __call__(self, *, solver_backend: str, solver_config: Dict[str, Any]) -> CoupledController: ...


def default_controller_factory(*, solver_backend: str, solver_config: Dict[str, Any]) -> CoupledController:
    from orbexa.control.mpc.controller import MPCController
    return MPCController(solver_backend=solver_backend, solver_config=solver_config)


class MissionScenario:
    """Compose supported truth, target, fleet, reference, safety and estimator dependencies.

    Controller construction is deferred until execution. An injected factory
    receives the explicit coupled GEKKO backend and cannot silently select an
    ordinary MPC backend. Custom targets retain responsibility for supplying
    geometry and collision data supported by that formulation.
    """

    def __init__(self, *, config: Optional[SimulationConfig] = None,
                 truth: Optional[OrbitalPlantTruth] = None,
                 target: Optional[TargetGeometry] = None,
                 fleet: Optional[FleetInitializer] = None,
                 reference: Optional[MissionReferencePolicy] = None,
                 safety: Optional[MissionSafety] = None,
                 point_estimator: Optional[PointEstimator] = None,
                 controller_factory: Optional[ControllerFactory] = None,
                 initial_belief: Optional[ParameterBelief] = None,
                 execution_guard=None,
                 arrival_policy=None,
                 progress_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
                 vehicle_design: Optional[Dict[str, Any]] = None,
                 adaptation_mode: str = 'adaptive',
                 horizon_steps: Optional[int] = None,
                 actuation_steps: Optional[int] = None,
                 approximation: str = "nonlinear", solver_backend: str = "gekko"):
        self.config = config if config is not None else SimulationConfig.load()
        if reference is not None and horizon_steps is not None:
            raise ValueError("Inject horizon_steps into the supplied MissionReferencePolicy")
        self.truth = truth if truth is not None else OrbitalPlantTruth.from_config(self.config)
        if initial_belief is not None and not isinstance(initial_belief, ParameterBelief):
            raise TypeError("initial_belief must be a ParameterBelief")
        # Copying also validates a caller-mutated belief. No plant value is used
        # to recenter or enlarge an explicitly supplied controller prior.
        self.initial_belief = None if initial_belief is None else initial_belief.copy()
        if adaptation_mode not in ('adaptive', 'frozen'):
            raise ValueError("adaptation_mode must be 'adaptive' or 'frozen'")
        self.adaptation_mode = adaptation_mode
        self.execution_guard = execution_guard
        if progress_sink is not None and not callable(progress_sink):
            raise TypeError('progress_sink must be callable')
        # Observation happens only at shared committed endpoints. It cannot
        # replace a controller, alter a proposed input or admit an interval.
        self.progress_sink = progress_sink
        if arrival_policy is not None:
            from orbexa.simulation.approach.arrival.policy import ArrivalPolicyConfig
            if not isinstance(arrival_policy, ArrivalPolicyConfig):
                raise TypeError('Inject a typed ArrivalPolicyConfig')
            if execution_guard is None:
                raise ValueError('Arrival policy requires a continuous execution guard and its exact inherited clock')
        self.arrival_policy = arrival_policy
        self.target = target if target is not None else TumblingCylinderTarget.from_config(self.config)
        if getattr(self.target, 'reference_frame_id', None) == 'reference-orbit-frame-v1':
            clock = getattr(self.target, 'reference_clock', None)
            if clock is None or clock.as_dict().get('frame') != self.target.reference_frame_id:
                raise ValueError('A nonlinear target preview requires its immutable scheduled reference clock')
        elif callable(getattr(self.target,'anomaly_clock',None)):
            for name in ('eccentricity','alpha','mu','semi_major_axis'):
                if not hasattr(self.target,name) or float(getattr(self.target,name))!=float(getattr(self.truth,name)):
                    raise ValueError(f'Physical target clock {name} must match the translational plant truth')
            if self.truth.time_periapsis!=0.:
                raise ValueError('The physical target clock currently requires the declared zero anomaly origin')
        self.vehicle_design = copy.deepcopy(vehicle_design)
        if self.vehicle_design is not None:
            if not isinstance(self.vehicle_design,dict):
                raise TypeError('vehicle_design must be explicit JSON-compatible metadata')
            json.dumps(self.vehicle_design,allow_nan=False)
        self.fleet = fleet if fleet is not None else FleetInitializer(self.config, self.target)
        self.reference = reference if reference is not None else MissionReferencePolicy(self.config, self.target, horizon_steps)
        self.safety = safety if safety is not None else MissionSafety(self.config, self.target)
        self.point_estimator = point_estimator if point_estimator is not None else default_point_estimator(self.config)
        self.controller_factory = controller_factory if controller_factory is not None else default_controller_factory
        if actuation_steps is not None and (isinstance(actuation_steps, bool) or
                not isinstance(actuation_steps, (int, np.integer)) or actuation_steps != 1):
            raise ValueError("Coupled ADTMPC actuation_steps must be the integer one")
        self.actuation_steps = None if actuation_steps is None else int(actuation_steps)
        self.approximation = approximation
        self.solver_backend = solver_backend

    def scenario_metadata(self) -> Dict[str, Any]:
        """Record optional scenario policy without coupling the recorder to it."""
        metadata = {**self.reference.scenario_metadata(),
                'tracked_reference_point': 'chaser_center_of_mass',
                'belief_update_mode': self.adaptation_mode,
                'chaser_body_bounding_radius_m': self.safety.chaser_body_radius}
        if self.vehicle_design is not None:
            metadata['vehicle_design']=copy.deepcopy(self.vehicle_design)
        if self.arrival_policy is not None:
            metadata['arrival_policy'] = self.arrival_policy.as_dict()
        return metadata

    def require_coupled_execution(self) -> None:
        if self.approximation != "nonlinear" or self.solver_backend != "gekko":
            raise ValueError("ADTMPC missions require the nonlinear GEKKO coupled dynamic-tube controller")
        if not self.config.tube.run_tube:
            raise ValueError("ADTMPC missions require tube.run_tube=True; use MPCController for ordinary MPC")
        counts = ([self.actuation_steps] if self.actuation_steps is not None else
                  list(self.config.mpc.actuation_steps.values()))
        if any(count != 1 for count in counts):
            raise ValueError("Coupled ADTMPC replans every outer interval; actuation_steps must be one")
        if getattr(self.target, "geometry", None) is not None:
            for chaser in self.fleet.chasers("single" if self.config.num_chasers == 1 else "multi"):
                margin = self.target.geometry.signed_distance(chaser.docking_point_body)
                if margin <= self.config.mission.target_barrier_clearance + self.safety.chaser_body_radius:
                    raise ValueError("Assigned spacecraft COM goals leave no finite-body component clearance")
            return
        if self.target.docking_standoff <= (self.target.radius * self.target.tolerance +
                self.config.mission.target_barrier_clearance + self.safety.chaser_body_radius):
            raise ValueError("The docking standoff leaves no free-flight tube clearance; contact requires an explicit contact model")

    def create_controller(self) -> CoupledController:
        self.require_coupled_execution()
        controller = self.controller_factory(solver_backend="gekko", solver_config=self.config.solver.options())
        if getattr(controller, "solver_backend", None) != "gekko":
            raise ValueError("The coupled controller factory must explicitly retain solver_backend='gekko'")
        return controller
