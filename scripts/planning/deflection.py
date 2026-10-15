# /***********************************************************
# * Copyright (c) 2026                                      *
# * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
# * University of California, Los Angeles                   *
# * Authors: Aaron John Sabu                                *
# * Contact: aaronjs@ucla.edu                               *
# ***********************************************************/

"""Standalone target torque-allocation optimization with explicit solver ownership.

Euler/body-rate kinematics use the shared Rx Ry Rz orientation convention. This
planner stays within one nonsingular Euler chart and does not model contact or
replace the ADTMPC mission's prescribed target motion.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from orbexa.core.attitude import body_rates_to_euler_rates, finite_vector, inertia_matrix
from orbexa.utils.solver_options import gekko_memory_exponent

if TYPE_CHECKING:
    from orbexa.core.target_spacecraft import Target


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")


@dataclass(frozen=True)
class DeflectionSolverOptions:
    remote: bool = False
    disp: bool = False
    solver_type: int = 3
    max_iter: int = 3000
    max_memory: int = 4
    nodes: int = 3
    chart_margin: float = 1e-3

    def __post_init__(self):
        if not isinstance(self.remote, bool) or not isinstance(self.disp, bool):
            raise ValueError("remote and disp must be booleans")
        _integer(self.solver_type, "solver", 1)
        if self.solver_type not in (1, 2, 3):
            raise ValueError("GEKKO solver must be 1, 2 or 3")
        _integer(self.max_iter, "maxIter")
        gekko_memory_exponent(self.max_memory)
        _integer(self.nodes, "nodes", 2)
        if self.nodes > 6:
            raise ValueError("GEKKO nodes must be between 2 and 6")
        if not np.isfinite(self.chart_margin) or not 0 < self.chart_margin < np.pi / 2:
            raise ValueError("chart_margin must be finite and between zero and pi/2")


@dataclass(frozen=True)
class DeflectionProblem:
    initial_state: np.ndarray
    final_state: np.ndarray
    inertia: np.ndarray
    dt: float
    samples: int
    num_chasers: int
    force_bounds: tuple
    pitch_bounds: tuple
    docking_points: np.ndarray | None
    surface: tuple | None

    @classmethod
    def from_inputs(cls, target, discretize_dockers, ellipsoid_points, cylinder_points, options, arguments):
        required = {"dt", "x_f", "numSteps", "num_chasers", "shapeParams"}
        if required - arguments.keys():
            raise ValueError(f"Missing deflection inputs: {sorted(required - arguments.keys())}")
        if not isinstance(discretize_dockers, bool):
            raise ValueError("discretize_dockers must be a boolean")
        dt = float(arguments["dt"])
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be a finite positive sample period in seconds")
        samples, num_chasers = arguments["numSteps"], arguments["num_chasers"]
        _integer(samples, "numSteps", 2)
        _integer(num_chasers, "num_chasers")
        if arguments.get("rLen", 3) != 3 or arguments.get("fLen", 3) != 3:
            raise ValueError("Target torque allocation requires three position and force components")
        initial = np.concatenate((finite_vector(target.curr_state, "Target Euler angles", 3),
                                  finite_vector(target.angular_velocity, "Target body rates", 3)))
        final = finite_vector(arguments["x_f"], "Deflection final state", 6)
        inertia = inertia_matrix(target.get_mom_inertia())
        chart_center = np.floor(initial[1] / np.pi + 0.5) * np.pi
        half_width = np.pi / 2 - options.chart_margin
        pitch_bounds = (float(chart_center - half_width), float(chart_center + half_width))
        if not pitch_bounds[0] <= initial[1] <= pitch_bounds[1]:
            raise ValueError("Initial Euler pitch is too close to a singular chart; use a different chart")
        if not pitch_bounds[0] <= final[1] <= pitch_bounds[1]:
            raise ValueError("Final Euler pitch must stay in the initial nonsingular chart")

        raw_bounds = arguments.get("bounds")
        if raw_bounds is None or len(raw_bounds) == 0:
            raw_bounds = [{"lower": None, "upper": None}] * 3
        if len(raw_bounds) not in (3, 3 * num_chasers):
            raise ValueError("bounds must contain three force bounds or three per chaser")
        bounds = []
        for bound in raw_bounds:
            if not isinstance(bound, Mapping):
                raise ValueError("Force bounds must be mappings with lower/upper values")
            low = -np.inf if bound.get("lower") is None else float(bound["lower"])
            high = np.inf if bound.get("upper") is None else float(bound["upper"])
            if np.isnan(low) or np.isnan(high) or np.isposinf(low) or np.isneginf(high) or low > high:
                raise ValueError("Force bounds must be ordered numeric limits")
            bounds.append((low, high))

        shape = arguments["shapeParams"]
        if not isinstance(shape, Mapping):
            raise ValueError("shapeParams must be a geometry mapping")
        if "ellRadX" in shape and "cylHeight" not in shape:
            shape_name, available_points = "ellipsoid", ellipsoid_points
        elif "cylHeight" in shape and "ellRadX" not in shape:
            shape_name, available_points = "cylinder", cylinder_points
        else:
            raise ValueError("shapeParams must select ellRadX or cylHeight geometry")
        points, surface = None, None
        if discretize_dockers:
            points = np.asarray(available_points, dtype=float)
            if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or not np.all(np.isfinite(points)):
                raise ValueError("Discrete dockers require a nonempty sequence of finite body-frame three-points")
            points = points.copy()
        elif shape_name == "ellipsoid":
            if not {"ellRadX", "ellRadY", "ellRadZ"}.issubset(shape):
                raise ValueError("Continuous ellipsoid points require ellRadX, ellRadY and ellRadZ")
            radii = finite_vector([shape["ellRadX"], shape["ellRadY"], shape["ellRadZ"]], "Ellipsoid radii", 3)
            if np.any(radii <= 0):
                raise ValueError("Ellipsoid radii must be positive")
            surface = (shape_name, tuple(radii))
        else:
            radius = float(shape.get("cylRadius", target.config.target.radius))
            height = float(shape["cylHeight"])
            if any(not np.isfinite(value) or value <= 0 for value in (radius, height)):
                raise ValueError("Cylinder radius and full height must be finite and positive")
            surface = (shape_name, (radius, height / 2))
        return cls(initial, final, inertia, dt, int(samples), int(num_chasers), tuple(bounds), pitch_bounds, points, surface)


def _docking_positions(model, problem):
    positions = []
    for agent in range(problem.num_chasers):
        if problem.docking_points is not None:
            points = problem.docking_points
            if len(points) == 1:
                choice = [model.Param(value=1.0)]
            else:
                choice = [model.FV(value=float(index == agent % len(points)), lb=0, ub=1) for index in range(len(points))]
                for value in choice:
                    value.STATUS = 1
                    model.Equation(value * (1 - value) <= 1e-7)
                model.Equation(sum(choice) == 1.0)
            position = [model.Intermediate(sum(point[dimension] * weight for point, weight in zip(points, choice)))
                        for dimension in range(3)]
        else:
            shape, dimensions = problem.surface
            angle = 2 * np.pi * agent / problem.num_chasers
            limits = np.asarray(dimensions if shape == "ellipsoid" else (dimensions[0], dimensions[0], dimensions[1]))
            seed = limits * [np.cos(angle), np.sin(angle), 0]
            position = [model.FV(value=float(seed[index]), lb=float(-limits[index]), ub=float(limits[index])) for index in range(3)]
            for value in position:
                value.STATUS = 1
            if shape == "ellipsoid":
                model.Equation(sum((position[index] / limits[index])**2 for index in range(3)) == 1.0)
            else:
                # This explicit continuous option allocates points on the cylinder side.
                model.Equation(position[0]**2 + position[1]**2 == limits[0]**2)
        positions.append(position)
    return positions


def _formulate(model, problem):
    model.time = problem.dt * np.arange(problem.samples)
    state = [model.Var(value=float(value), fixed_initial=True) for value in problem.initial_state]
    state[1].lower, state[1].upper = problem.pitch_bounds
    forces = []
    for index in range(3 * problem.num_chasers):
        low, high = problem.force_bounds[index % len(problem.force_bounds)]
        value = model.Var(value=float(np.clip(0, low, high)), fixed_initial=False)
        if np.isfinite(low):
            value.lower = float(low)
        if np.isfinite(high):
            value.upper = float(high)
        forces.append(value)
    positions = _docking_positions(model, problem)
    torque = [0, 0, 0]
    for agent, position in enumerate(positions):
        rx, ry, rz = position
        fx, fy, fz = forces[3 * agent:3 * (agent + 1)]
        torque[0] += ry * fz - rz * fy
        torque[1] += rz * fx - rx * fz
        torque[2] += rx * fy - ry * fx
    omega = state[3:]
    momentum = [sum(problem.inertia[row, column] * omega[column] for column in range(3)) for row in range(3)]
    gyro = [omega[1] * momentum[2] - omega[2] * momentum[1],
            omega[2] * momentum[0] - omega[0] * momentum[2],
            omega[0] * momentum[1] - omega[1] * momentum[0]]
    inverse = np.linalg.inv(problem.inertia)
    acceleration = [sum(inverse[row, column] * (torque[column] - gyro[column]) for column in range(3)) for row in range(3)]
    euler_rates = body_rates_to_euler_rates(state[:3], omega, solver=model)
    model.Equations([state[index].dt() == euler_rates[index] for index in range(3)]
                    + [omega[index].dt() == acceleration[index] for index in range(3)])
    terminal = np.zeros(problem.samples)
    terminal[-1] = 1
    terminal_weight = model.Param(value=terminal)
    terminal_error = sum((state[index] - problem.final_state[index])**2 for index in range(6))
    model.Minimize(terminal_weight * terminal_error + 1e-4 * sum(force**2 for force in forces))
    return state, [value for position in positions for value in position], forces


class TargetDeflectionPlanner:
    """Own one new GEKKO-like model per solve, including exceptional cleanup.

    The optional model_factory accepts ``remote=bool`` and returns a new model.
    The returned model is owned by this planner and always cleaned up. Remote
    execution occurs only when selected in DeflectionSolverOptions.
    """

    def __init__(self, options=None, model_factory=None):
        self.options = options or DeflectionSolverOptions()
        if not isinstance(self.options, DeflectionSolverOptions):
            raise TypeError("options must be DeflectionSolverOptions")
        if model_factory is not None and not callable(model_factory):
            raise TypeError("model_factory must be callable")
        self.model_factory = model_factory

    def solve(self, problem):
        if not isinstance(problem, DeflectionProblem):
            raise TypeError("problem must be DeflectionProblem")
        factory = self.model_factory
        if factory is None:
            from gekko import GEKKO
            factory = GEKKO
        model = factory(remote=self.options.remote)
        try:
            state, positions, forces = _formulate(model, problem)
            model.options.IMODE = 6
            model.options.SOLVER = self.options.solver_type
            model.options.MAX_ITER = self.options.max_iter
            model.options.MAX_MEMORY = self.options.max_memory
            model.options.NODES = self.options.nodes
            try:
                model.solve(disp=self.options.disp)
            except Exception as error:
                return {"success": False, "message": str(error), "target_state": None,
                        "chaser_positions": None, "chaser_forces": None}
            if model.options.APPSTATUS != 1:
                return {"success": False, "message": "Deflection solver did not report success", "target_state": None,
                        "chaser_positions": None, "chaser_forces": None}
            arrays = [np.asarray([list(value.value) for value in variables], dtype=float) for variables in (state, positions, forces)]
            if any(not np.all(np.isfinite(array)) for array in arrays):
                raise RuntimeError("Deflection solver returned nonfinite values")
            return {"success": True, "objective": float(model.options.OBJFCNVAL),
                    "target_state": arrays[0], "chaser_positions": arrays[1], "chaser_forces": arrays[2]}
        finally:
            model.cleanup()


def target_deflect(target: Target, discretize_dockers=True, ellipsoid_points=None, cylinder_points=None, *args, **kwargs):
    """Optimize standalone body-frame torque allocation through the existing API.

    ``numSteps`` is the number of returned state samples, including time zero;
    adjacent samples are ``dt`` seconds apart. States contain Euler angles then
    body rates. Returned forces contain all node samples; node k+1 supplies the
    held command for interval k, while node zero has no preceding interval.
    ``shapeParams`` chooses an ellipsoid (ellRadX/Y/Z) or cylinder
    (cylHeight is full height; cylRadius defaults to target config). Discrete
    points are supplied explicitly. Continuous cylinder points use its side.
    This Euler optimizer rejects a singular chart and constrains pitch throughout
    the prediction. It models ideal point forces, without contact dynamics.
    """
    allowed = {"dt", "x_f", "bounds", "numSteps", "num_chasers", "rLen", "fLen", "shapeParams",
               "remote", "disp", "maxIter", "solver", "nodes", "max_memory", "chart_margin", "model_factory", "backend"}
    if args or kwargs.keys() - allowed:
        raise TypeError(f"Unsupported deflection arguments: {sorted(kwargs.keys() - allowed)}")
    if kwargs.get("backend", "gekko") != "gekko":
        raise ValueError("Target deflection supports backend='gekko'")
    options = DeflectionSolverOptions(remote=kwargs.get("remote", False), disp=kwargs.get("disp", False),
                                      solver_type=kwargs.get("solver", 3), max_iter=kwargs.get("maxIter", 3000),
                                      max_memory=kwargs.get("max_memory", 4), nodes=kwargs.get("nodes", 3),
                                      chart_margin=kwargs.get("chart_margin", 1e-3))
    problem = DeflectionProblem.from_inputs(target, discretize_dockers, ellipsoid_points, cylinder_points, options, kwargs)
    return TargetDeflectionPlanner(options, kwargs.get("model_factory")).solve(problem)
