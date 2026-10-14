"""Bounded external-thruster wrench allocation shared by mission stages.

Each signed actuator represents two opposed nozzles at one location; fuel
flow uses the absolute command. No reaction-wheel momentum sink is assumed.
Nozzle directions and locations stay fixed to the captured structure.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
import numpy as np
from scipy.optimize import lsq_linear

from orbexa.core.attitude import finite_vector
from orbexa.simulation.assembly import AttachedAssembly, STANDARD_GRAVITY, positive


@dataclass(frozen=True)
class Thruster:
    name: str
    owner_index: int
    position_body_m: np.ndarray
    direction_body: np.ndarray
    max_force_n: float
    specific_impulse_s: float

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Thruster name must be nonempty")
        if isinstance(self.owner_index, (bool, np.bool_)) or not isinstance(self.owner_index, Integral) or self.owner_index < 0:
            raise ValueError("Thruster owner_index must be nonnegative")
        object.__setattr__(self, "owner_index", int(self.owner_index))
        position = finite_vector(self.position_body_m, "Thruster position", 3)
        direction = finite_vector(self.direction_body, "Thruster direction", 3)
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError("Thruster direction cannot be zero")
        direction /= norm
        for name, value in (("position_body_m", position), ("direction_body", direction)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "max_force_n", positive(self.max_force_n, "Thruster force limit"))
        object.__setattr__(self, "specific_impulse_s", positive(self.specific_impulse_s, "Specific impulse"))

    def to_dict(self):
        return dict(name=self.name, owner_index=self.owner_index, position_body_m=self.position_body_m.tolist(),
                    direction_body=self.direction_body.tolist(), max_force_n=float(self.max_force_n),
                    specific_impulse_s=float(self.specific_impulse_s))


@dataclass(frozen=True)
class WrenchAllocation:
    success: bool
    message: str
    commands_n: np.ndarray
    force_body_n: np.ndarray
    torque_body_nm: np.ndarray
    fuel_flow_kg_s: np.ndarray
    residual_force_n: float
    residual_torque_nm: float
    saturated: bool

    def to_dict(self):
        return {name: value.tolist() if isinstance(value, np.ndarray) else value for name, value in vars(self).items()}


@dataclass(frozen=True)
class ThrusterSystem:
    thrusters: tuple[Thruster, ...]
    torque_weight_per_m: float = 20.0

    def __post_init__(self):
        thrusters = tuple(self.thrusters)
        if not thrusters or len({thruster.name for thruster in thrusters}) != len(thrusters):
            raise ValueError("Propulsion needs uniquely named thrusters")
        object.__setattr__(self, "thrusters", thrusters)
        object.__setattr__(self, "torque_weight_per_m", positive(self.torque_weight_per_m, "Torque allocation weight"))

    @classmethod
    def for_assembly(cls, assembly: AttachedAssembly, *, translation_force_n: float = 10.0,
                     attitude_force_n: float = .2, moment_arm_m: float = .1,
                     specific_impulse_s: float = 300.0):
        """Representative multi-nozzle bus on every fueled chaser.

        The three translation channels and three opposed torque couples are
        an explicit hardware assumption. The +/- signs denote opposed nozzle
        pairs; continuous throttling and unobstructed plumes are assumed.
        """
        translation_force_n = positive(translation_force_n, "Translation force")
        attitude_force_n = positive(attitude_force_n, "Attitude force")
        moment_arm_m = positive(moment_arm_m, "Moment arm")
        specific_impulse_s = positive(specific_impulse_s, "Specific impulse")
        thrusters = []
        for index, element in enumerate(assembly.elements):
            if element.initial_fuel_kg <= 0:
                continue
            axes = element.rotation_element_to_assembly.T
            for axis in range(3):
                thrusters.append(Thruster(f"{element.name}_translation_{axis}", index, element.position_body_m,
                                          axes[axis], translation_force_n, specific_impulse_s))
                lever, direction = axes[(axis + 1) % 3], axes[(axis + 2) % 3]
                for sign in (-1, 1):
                    thrusters.append(Thruster(f"{element.name}_couple_{axis}_{sign}", index,
                        element.position_body_m + sign * moment_arm_m * lever,
                        sign * direction, attitude_force_n, specific_impulse_s))
        if not thrusters:
            raise ValueError("No fueled chasers are available for installed thrusters")
        return cls(tuple(thrusters))

    def matrix(self, assembly, fuel_kg):
        center = assembly.mass_properties(fuel_kg)[1]
        if any(thruster.owner_index >= len(assembly.elements) for thruster in self.thrusters):
            raise ValueError("Thruster owner does not belong to the assembly")
        return np.column_stack([np.r_[thruster.direction_body,
                                      np.cross(thruster.position_body_m - center, thruster.direction_body)]
                                for thruster in self.thrusters])

    def authority(self, assembly, fuel_kg):
        matrix = self.matrix(assembly, fuel_kg)
        active = np.array([fuel_kg[thruster.owner_index] > 1e-10 for thruster in self.thrusters])
        usable = matrix[:, active]
        return dict(wrench_rank=int(np.linalg.matrix_rank(usable)) if usable.size else 0,
                    torque_rank=int(np.linalg.matrix_rank(usable[3:])) if usable.size else 0,
                    active_actuators=int(np.sum(active)),
                    component_force_limits_n=[thruster.max_force_n for thruster in self.thrusters])

    def allocate(self, assembly, state, force_body_n, torque_body_nm, duration_s):
        force = finite_vector(force_body_n, "Requested force", 3)
        torque = finite_vector(torque_body_nm, "Requested torque", 3)
        duration = positive(duration_s, "Allocation duration")
        matrix = self.matrix(assembly, state.fuel_kg)
        active = np.array([state.fuel_kg[thruster.owner_index] > 1e-10 for thruster in self.thrusters])
        commands = np.zeros(len(self.thrusters))
        message, success = "allocated", True
        if np.any(active):
            weights = np.r_[np.ones(3), np.full(3, self.torque_weight_per_m)]
            limits = np.array([thruster.max_force_n for thruster in self.thrusters])[active]
            solution = lsq_linear(weights[:, None] * matrix[:, active], weights * np.r_[force, torque],
                                  bounds=(-limits, limits), method="bvls", tol=1e-10, max_iter=100)
            if solution.success and np.all(np.isfinite(solution.x)):
                commands[active] = solution.x
            else:
                success, message = False, f"wrench_allocation_failed: {solution.message}"
        elif np.linalg.norm(force) + np.linalg.norm(torque) > 1e-12:
            success, message = False, "fuel_exhausted"
        flow = np.zeros(len(assembly.elements))
        for index, thruster in enumerate(self.thrusters):
            flow[thruster.owner_index] += abs(commands[index]) / (thruster.specific_impulse_s * STANDARD_GRAVITY)
        # Scale all commands together when one tank reaches its physical limit.
        # This preserves the allocated wrench direction and reports saturation.
        scale = min([1.0] + [state.fuel_kg[index] / (rate * duration)
                             for index, rate in enumerate(flow) if rate > 0])
        commands *= scale
        flow *= scale
        actual = matrix @ commands
        limits = np.array([thruster.max_force_n for thruster in self.thrusters])
        saturated = bool(scale < 1 - 1e-12 or np.any(np.abs(commands) >= limits * (1 - 1e-9)))
        return WrenchAllocation(success, message, commands, actual[:3], actual[3:], flow,
                                float(np.linalg.norm(actual[:3] - force)),
                                float(np.linalg.norm(actual[3:] - torque)), saturated)

    def to_dict(self):
        return dict(thrusters=[thruster.to_dict() for thruster in self.thrusters],
                    torque_weight_per_m=self.torque_weight_per_m,
                    assumptions=["signed channels are opposed nozzle pairs", "continuous throttling",
                                 "unobstructed exhaust; no plume-structure interaction", "no power or thermal model"])
