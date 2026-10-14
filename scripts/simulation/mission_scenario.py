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

from typing import Any, Dict, Optional, Protocol
import numpy as np

from orbexa.core.config import SimulationConfig
from orbexa.core.orbital_parameters import OrbitalPlantTruth
from orbexa.core.target_geometry import TargetGeometry, TumblingCylinderTarget
from orbexa.estimation.mission_adaptation import PointEstimator, default_point_estimator
from orbexa.simulation.fleet import FleetInitializer
from orbexa.simulation.mission_policy import MissionReferencePolicy
from orbexa.simulation.mission_safety import MissionSafety


class CoupledController(Protocol):
    solver_backend: str

    def solve_step(self, **kwargs: Any) -> Any: ...


class ControllerFactory(Protocol):
    def __call__(self, *, solver_backend: str, solver_config: Dict[str, Any]) -> CoupledController: ...


def default_controller_factory(*, solver_backend: str, solver_config: Dict[str, Any]) -> CoupledController:
    from orbexa.control.mpc_controller import MPCController
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
                 horizon_steps: Optional[int] = None,
                 actuation_steps: Optional[int] = None,
                 approximation: str = "nonlinear", solver_backend: str = "gekko"):
        self.config = config if config is not None else SimulationConfig.load()
        if reference is not None and horizon_steps is not None:
            raise ValueError("Inject horizon_steps into the supplied MissionReferencePolicy")
        self.truth = truth if truth is not None else OrbitalPlantTruth.from_config(self.config)
        self.target = target if target is not None else TumblingCylinderTarget.from_config(self.config)
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

    def require_coupled_execution(self) -> None:
        if self.approximation != "nonlinear" or self.solver_backend != "gekko":
            raise ValueError("ADTMPC missions require the nonlinear GEKKO coupled dynamic-tube controller")
        if not self.config.tube.run_tube:
            raise ValueError("ADTMPC missions require tube.run_tube=True; use MPCController for ordinary MPC")
        counts = ([self.actuation_steps] if self.actuation_steps is not None else
                  list(self.config.mpc.actuation_steps.values()))
        if any(count != 1 for count in counts):
            raise ValueError("Coupled ADTMPC replans every outer interval; actuation_steps must be one")
        if self.target.docking_standoff <= self.target.radius * self.target.tolerance + self.config.mission.target_barrier_clearance:
            raise ValueError("The docking standoff leaves no free-flight tube clearance; contact requires an explicit contact model")

    def create_controller(self) -> CoupledController:
        self.require_coupled_execution()
        controller = self.controller_factory(solver_backend="gekko", solver_config=self.config.solver.options())
        if getattr(controller, "solver_backend", None) != "gekko":
            raise ValueError("The coupled controller factory must explicitly retain solver_backend='gekko'")
        return controller
