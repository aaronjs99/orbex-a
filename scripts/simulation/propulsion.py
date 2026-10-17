"""Bounded external-thruster wrench allocation shared by mission stages.

Each signed actuator represents two opposed nozzles at one location; fuel
flow uses the absolute command. No reaction-wheel momentum sink is assumed.
Nozzle directions and locations stay fixed to the captured structure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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
    attachment_preflight: dict = field(default_factory=dict)

    def to_dict(self):
        return {name: value.tolist() if isinstance(value, np.ndarray) else value for name, value in vars(self).items()
                if name != 'attachment_preflight' or value}


@dataclass(frozen=True)
class ThrusterSystem:
    thrusters: tuple[Thruster, ...]
    torque_weight_per_m: float = 20.0
    attachment_capacity_fraction: float = .98

    def __post_init__(self):
        thrusters = tuple(self.thrusters)
        if not thrusters or len({thruster.name for thruster in thrusters}) != len(thrusters):
            raise ValueError("Propulsion needs uniquely named thrusters")
        object.__setattr__(self, "thrusters", thrusters)
        object.__setattr__(self, "torque_weight_per_m", positive(self.torque_weight_per_m, "Torque allocation weight"))
        fraction = positive(self.attachment_capacity_fraction, 'Attachment capacity fraction')
        if fraction >= 1:
            raise ValueError('Attachment capacity fraction must be strictly below one')
        object.__setattr__(self, 'attachment_capacity_fraction', fraction)

    @classmethod
    def for_assembly(cls, assembly: AttachedAssembly, *, translation_force_n: float = 10.0,
                     attitude_force_n: float = .2, moment_arm_m: float = .1,
                     specific_impulse_s: float = 300.0, attachment_capacity_fraction: float = .98):
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
        return cls(tuple(thrusters), attachment_capacity_fraction=attachment_capacity_fraction)

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

    def _attachment_preflight(self, assembly, state, commands):
        """Dilate a bounded candidate before motion using actual joint loads.

        At a fixed state, the Newton-Euler force/couple vectors are affine in
        commands, including the environmental and inertial bias at zero input.
        Normal intervals and shear/moment balls are convex. A feasible zero and
        candidate therefore define a one-dimensional convex feasible interval.
        The final actual load calculation remains authoritative; this numerical
        point check does not enclose loads during the subsequent held step.
        """
        zero = np.zeros_like(commands)
        baseline = assembly.attachment_load_audit(state, self, zero)
        candidate = assembly.attachment_load_audit(state, self, commands)
        fraction = self.attachment_capacity_fraction
        record = dict(method='affine-joint-wrench-common-command-dilation-v1',
            time_s=state.time_s, capacity_fraction=fraction,
            capacities_unchanged=True, candidate_commands_n=commands.tolist(),
            zero_command_loads=baseline, candidate_loads=candidate,
            scope='Pre-motion numerical loads at the current state; actual endpoint/history audits remain required')

        def finite(loads):
            return (np.isfinite(loads['maximum_capacity_ratio'])
                and all(np.all(np.isfinite(member['force_body_n']))
                    and np.all(np.isfinite(member['moment_at_pad_body_nm']))
                    and all(np.isfinite(value) for value in member['capacity_ratios'].values())
                    for member in loads['members'].values()))

        if not finite(baseline) or not finite(candidate):
            record.update(passed=False, command_scale=0., final_loads=baseline,
                          reason='attachment_load_nonfinite')
            return zero, False, record['reason'], record
        if baseline['maximum_capacity_ratio'] > fraction:
            reason = ('attachment_load_zero_command_infeasible' if baseline['maximum_capacity_ratio'] > 1.
                      else 'attachment_load_zero_command_reserve_exceeded')
            record.update(passed=False, command_scale=0., final_loads=baseline, reason=reason)
            return zero, False, reason, record
        scale, iterations = 1., 0
        if candidate['maximum_capacity_ratio'] > fraction:
            limits = {value['name']: value for value in assembly.attachment_limits}

            def ratio(value):
                maximum = 0.
                for name, first in baseline['members'].items():
                    last, limit = candidate['members'][name], limits[name]
                    force = np.asarray(first['force_body_n']) + value*(np.asarray(last['force_body_n'])-first['force_body_n'])
                    moment = np.asarray(first['moment_at_pad_body_nm']) + value*(np.asarray(last['moment_at_pad_body_nm'])-first['moment_at_pad_body_nm'])
                    normal = np.asarray(limit['normal_body'])
                    fn = float(force@normal)
                    maximum = max(maximum, max(0.,-fn)/limit['tensile_limit_n'],
                        max(0.,fn)/limit['compression_limit_n'],
                        float(np.linalg.norm(force-fn*normal))/limit['shear_limit_n'],
                        float(np.linalg.norm(moment))/limit['moment_limit_nm'])
                return maximum

            lower, upper = 0., 1.
            for iterations in range(1, 41):
                midpoint = (lower+upper)/2
                if ratio(midpoint) <= fraction:
                    lower = midpoint
                else:
                    upper = midpoint
            # Stay inside the numerical bisection boundary before independent
            # full recomputation. The 2% utilization reserve is separate from
            # this floating-point inward move and never raises a capacity.
            scale = lower*(1.-1e-10)
            commands = commands*scale
            final = assembly.attachment_load_audit(state, self, commands)
        else:
            final = candidate
        passed = finite(final) and final['maximum_capacity_ratio'] <= fraction
        reason = ('attachment_load_command_dilated' if scale < 1. else 'attachment_load_candidate_admitted') if passed else 'attachment_load_final_recheck_failed'
        record.update(passed=passed, command_scale=scale, bisection_iterations=iterations,
                      final_loads=final, reason=reason)
        return commands if passed else zero, passed, reason, record

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
        preflight = {}
        load_scale = 1.
        if success and assembly.attachment_limits:
            commands, success, load_message, preflight = self._attachment_preflight(assembly, state, commands)
            load_scale = preflight['command_scale']
            preflight['fuel_command_scale'] = scale
            if not success or load_scale < 1.:
                message = load_message
            # Every reported wrench, residual and tank flow uses the final
            # commands, including a refused zero command before any propagation.
            flow = np.zeros(len(assembly.elements))
            for command, thruster in zip(commands, self.thrusters):
                flow[thruster.owner_index] += abs(command)/(thruster.specific_impulse_s*STANDARD_GRAVITY)
        actual = matrix @ commands
        limits = np.array([thruster.max_force_n for thruster in self.thrusters])
        saturated = bool(scale < 1 - 1e-12 or load_scale < 1 - 1e-12 or np.any(np.abs(commands) >= limits * (1 - 1e-9)))
        return WrenchAllocation(success, message, commands, actual[:3], actual[3:], flow,
                                float(np.linalg.norm(actual[:3] - force)),
                                float(np.linalg.norm(actual[3:] - torque)), saturated, preflight)

    def to_dict(self):
        return dict(thrusters=[thruster.to_dict() for thruster in self.thrusters],
                    torque_weight_per_m=self.torque_weight_per_m,
                    attachment_capacity_fraction=self.attachment_capacity_fraction,
                    assumptions=["signed channels are opposed nozzle pairs", "continuous throttling",
                                 "unobstructed exhaust; no plume-structure interaction", "no power or thermal model"])
