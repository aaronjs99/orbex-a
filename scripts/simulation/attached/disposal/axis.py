"""Numerical coupled-wrench planning for a fixed installed thrust direction."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
import numpy as np
from scipy.optimize import linprog

from orbexa.simulation.attached.assembly.model import STANDARD_GRAVITY, positive


@dataclass(frozen=True)
class AxialBurnPlan:
    direction_body: tuple[float, float, float]
    capacity_n: float
    commanded_force_n: float
    minimum_planned_fuel_flow_kg_s: float
    capacity_commands_n: tuple[float, ...]
    minimum_flow_commands_n: tuple[float, ...]
    candidates: tuple[dict, ...]

    def to_dict(self):
        return dict(direction_body=list(self.direction_body), capacity_n=self.capacity_n,
                    commanded_force_n=self.commanded_force_n,
                    minimum_planned_fuel_flow_kg_s=self.minimum_planned_fuel_flow_kg_s,
                    capacity_commands_n=list(self.capacity_commands_n),
                    minimum_flow_commands_n=list(self.minimum_flow_commands_n), candidates=list(self.candidates),
                    scope='Installed static LP capacity retained separately from the operational requested wrench within the declared residual tolerance admitted by the allocator at the declared planning state. Finite force candidates and their full held proofs are recorded; later actual allocation, fuel and orbital events remain authoritative.')


@dataclass(frozen=True)
class AxialBurnPlanner:
    """Select a low-flow direction while accounting for its force lever.

    Candidate axes come from the body basis and installed nozzle directions.
    Every accepted direction has a zero-torque capacity witness and an
    independently checked minimum-L1 fuel-flow witness at the commanded force.
    Selection scores the existing allocator's actual flow at that force. This is
    a numerical plan, with no claim of future or validated feasibility.
    """
    command_fraction: float = .65
    residual_tolerance: float = 1e-7
    force_backoff_scales: tuple[float, ...] = (1., .95, .9, .85, .8, .75, .5, .25, .125)

    def __post_init__(self):
        fraction = positive(self.command_fraction, 'Burn command fraction')
        if fraction >= 1:
            raise ValueError('Burn command fraction must leave actuator reserve')
        object.__setattr__(self, 'command_fraction', fraction)
        object.__setattr__(self, 'residual_tolerance', positive(self.residual_tolerance, 'Wrench witness residual tolerance'))
        raw = tuple(self.force_backoff_scales)
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) for value in raw):
            raise ValueError('Burn force backoff scales must be finite real fractions')
        scales = tuple(positive(value, 'Burn force backoff scale') for value in raw)
        if (not 1 <= len(scales) <= 16 or scales[0] != 1.
                or any(a <= b for a, b in zip(scales, scales[1:]))):
            raise ValueError('Declare one to sixteen strictly decreasing positive burn force scales starting at one')
        object.__setattr__(self, 'force_backoff_scales', scales)

    def select(self, assembly, state, propulsion, duration_s):
        duration = positive(duration_s, 'Planned allocation hold')
        matrix = propulsion.matrix(assembly, state.fuel_kg)
        limits = np.array([t.max_force_n if state.fuel_kg[t.owner_index] > 1e-10 else 0.
                           for t in propulsion.thrusters])
        if not np.any(limits > 0) or np.linalg.matrix_rank(matrix[:, limits > 0]) < 6:
            raise ValueError('Axial burn planning requires active six-axis wrench authority')
        n = len(limits)
        directions = []
        for vector in [*np.eye(3), *(t.direction_body for t in propulsion.thrusters)]:
            direction = np.asarray(vector, dtype=float)
            direction = direction / np.linalg.norm(direction)
            if any(abs(float(np.dot(direction, previous))) > 1 - 1e-10 for previous in directions):
                continue
            directions.append(direction)
        candidates, accepted = [], []
        weights = np.array([1 / t.specific_impulse_s for t in propulsion.thrusters])
        objective = np.r_[np.zeros(n), weights / np.max(weights)]
        absolute_rows = np.block([[np.eye(n), -np.eye(n)], [-np.eye(n), -np.eye(n)]])
        bounds = [(-v, v) for v in limits]
        for direction in directions:
            wrench = np.r_[direction, np.zeros(3)]
            capacity = linprog(np.r_[np.zeros(n), -1.],
                A_eq=np.column_stack((matrix, -wrench)), b_eq=np.zeros(6),
                bounds=[*bounds, (0., None)], method='highs')
            row = dict(direction_body=direction.tolist(), capacity_status=int(capacity.status), accepted=False)
            candidates.append(row)
            if not capacity.success or not np.all(np.isfinite(capacity.x)):
                continue
            force = float(capacity.x[-1])
            capacity_residual = float(np.max(np.abs(matrix @ capacity.x[:-1] - force * wrench)))
            row.update(capacity_n=force, capacity_residual=capacity_residual,
                       capacity_commands_n=capacity.x[:-1].tolist())
            if force <= self.residual_tolerance or capacity_residual > self.residual_tolerance:
                continue
            # Installed nozzle authority is not a load-admitted burn request.
            # Retain every finite proposal, then require the operational
            # allocator to achieve one requested six-wrench within the declared residual tolerance while
            # preserving its original whole-hold admission gates.
            requested = self.command_fraction * force
            attempts, allocation, commanded = [], None, None
            for force_scale in self.force_backoff_scales:
                proposed = requested*force_scale
                candidate = propulsion.allocate(assembly, state, proposed*direction, np.zeros(3), duration)
                attempts.append(dict(scale=force_scale, requested_force_n=proposed,
                    allocation=candidate.to_dict(), physical_motion_committed=False))
                actual = matrix@candidate.commands_n
                residuals = (candidate.residual_force_n, candidate.residual_torque_nm,
                    float(np.linalg.norm(actual[:3]-proposed*direction)), float(np.linalg.norm(actual[3:])))
                if (not candidate.success or not np.all(np.isfinite(residuals))
                        or min(residuals) < 0 or max(residuals) > self.residual_tolerance):
                    continue
                allocation, commanded = candidate, proposed
                break
            row.update(declared_force_backoff_scales=list(self.force_backoff_scales),
                installed_fraction_requested_force_n=requested, force_candidates=attempts)
            if allocation is None:
                continue
            minimum = linprog(objective, A_ub=absolute_rows, b_ub=np.zeros(2*n),
                A_eq=np.column_stack((matrix, np.zeros_like(matrix))), b_eq=commanded * wrench,
                bounds=[*bounds, *((0., v) for v in limits)], method='highs')
            row['minimum_flow_status'] = int(minimum.status)
            if not minimum.success or not np.all(np.isfinite(minimum.x)):
                continue
            commands = minimum.x[:n]
            residual = float(np.max(np.abs(matrix @ commands - commanded * wrench)))
            command_excess = float(np.max(np.abs(commands) - limits))
            capacity_excess = float(np.max(np.abs(capacity.x[:-1]) - limits))
            flow = float(np.dot(np.abs(commands), weights) / STANDARD_GRAVITY)
            row.update(commanded_force_n=commanded, minimum_flow_residual=residual, command_excess_n=command_excess,
                       capacity_command_excess_n=capacity_excess, minimum_fuel_flow_kg_s=flow,
                       minimum_flow_commands_n=commands.tolist())
            if max(residual, command_excess, capacity_excess) > self.residual_tolerance or flow <= 0:
                continue
            actual_flow = float(np.sum(allocation.fuel_flow_kg_s))
            row.update(actual_allocator=allocation.to_dict(), actual_fuel_flow_kg_s=actual_flow,
                       actual_flow_per_force_kg_s_n=actual_flow / commanded)
            if not np.isfinite(actual_flow) or actual_flow <= 0:
                continue
            row['accepted'] = True
            accepted.append((actual_flow / commanded, -force, len(candidates)-1, capacity.x[:-1], commands, flow))
        if not accepted:
            raise ValueError('No installed burn direction has an admitted coupled-wrench witness')
        _, negative_capacity, index, capacity_commands, minimum_commands, flow = min(accepted, key=lambda item:item[:3])
        row = candidates[index]
        return AxialBurnPlan(tuple(row['direction_body']), -negative_capacity,
            row['commanded_force_n'], flow,
            tuple(map(float, capacity_commands)), tuple(map(float, minimum_commands)), tuple(candidates))
