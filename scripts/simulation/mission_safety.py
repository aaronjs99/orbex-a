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

"""Measured physical safety and matched-tube planning margins."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple, TYPE_CHECKING
import numpy as np
from orbexa.control.constraints import rendezvous_margin
from orbexa.simulation.mission_models import MissionPhase

if TYPE_CHECKING:
    from orbexa.core.config import SimulationConfig
    from orbexa.core.target_geometry import TargetGeometry


class MissionSafety:
    """Physical target margins, pairwise spacing and actuator feasibility."""

    def __init__(self, config: SimulationConfig, target: TargetGeometry):
        self.config = config
        self.target = target
        self.min_chaser_separation = float(config.mpc.min_chaser_separation)
        self.target_safety_tolerance = float(config.mission.target_safety_tolerance)

    def clip_control(self, control: np.ndarray) -> np.ndarray:
        clipped = np.asarray(control, dtype=float).copy()
        for idx, bound in enumerate(self.config.mpc.input_bounds[: len(clipped)]):
            lower = bound.get("lower", -np.inf)
            upper = bound.get("upper", np.inf)
            if lower != "-Inf":
                clipped[idx] = max(float(lower), clipped[idx])
            if upper != "+Inf":
                clipped[idx] = min(float(upper), clipped[idx])
        return clipped


    def active_margin_for_state(
        self, state: np.ndarray, anom: float, phase: MissionPhase
    ) -> float:
        position = np.asarray(state[:3], dtype=float)
        if phase.operation == "docking":
            cylinder = self.target.cylinder_constraint(anom, tube_radius=0.0)
            radial, axial = cylinder.margins(position)
            return float(max(radial, axial))
        return rendezvous_margin(
            position,
            target_radius=self.target.rendezvous_radius,
            tube_radius=0.0,
        )


    def safety_margins(
        self,
        actual: Dict[str, List[List[float]]],
        anom_history: Sequence[float],
        sample_phase_history: Sequence[str],
        tube_radius_history: Sequence[float],
    ) -> Tuple[
        Dict[str, List[float]],
        Dict[str, List[float]],
        Dict[str, List[float]],
        Dict[str, List[float]],
        Dict[str, List[float]],
    ]:
        """Compute physical safety margins separately from tube-tightened margins.

        Physical margins answer whether the actual chaser is outside the solid
        target.  Tightened margins include the dynamic tube radius and are a
        robust-planning diagnostic, so they can be negative while the physical
        active margin remains safe.
        """
        physical_rendezvous = {}
        physical_docking = {}
        tightened_rendezvous = {}
        tightened_docking = {}
        active = {}
        for chaser_id, states in actual.items():
            physical_rv_values = []
            physical_dock_values = []
            tightened_rv_values = []
            tightened_dock_values = []
            active_values = []
            for idx, state in enumerate(states):
                anom = anom_history[min(idx, len(anom_history) - 1)]
                tube_radius = tube_radius_history[min(idx, len(tube_radius_history) - 1)]
                phase = sample_phase_history[min(idx, len(sample_phase_history) - 1)]
                position = np.asarray(state[:3], dtype=float)
                physical_rv_values.append(
                    rendezvous_margin(
                        position,
                        target_radius=self.target.rendezvous_radius,
                        tube_radius=0.0,
                    )
                )
                tightened_rv_values.append(
                    rendezvous_margin(
                        position,
                        target_radius=self.target.rendezvous_radius,
                        tube_radius=tube_radius,
                    )
                )
                physical_cylinder = self.target.cylinder_constraint(anom, tube_radius=0.0)
                physical_radial, physical_axial = physical_cylinder.margins(position)
                physical_dock_values.append(float(max(physical_radial, physical_axial)))
                tightened_cylinder = self.target.cylinder_constraint(
                    anom, tube_radius=tube_radius
                )
                tightened_radial, tightened_axial = tightened_cylinder.margins(position)
                tightened_dock_values.append(float(max(tightened_radial, tightened_axial)))
                if phase == "docking":
                    active_values.append(float(max(physical_radial, physical_axial)))
                else:
                    active_values.append(physical_rv_values[-1])
            physical_rendezvous[chaser_id] = physical_rv_values
            physical_docking[chaser_id] = physical_dock_values
            tightened_rendezvous[chaser_id] = tightened_rv_values
            tightened_docking[chaser_id] = tightened_dock_values
            active[chaser_id] = active_values
        return (
            physical_rendezvous,
            physical_docking,
            tightened_rendezvous,
            tightened_docking,
            active,
        )


    def active_target_margins_safe(
        self, active_margins: Dict[str, Sequence[float]]
    ) -> bool:
        for values in active_margins.values():
            arr = np.asarray(values, dtype=float)
            if arr.size and float(np.nanmin(arr)) < -self.target_safety_tolerance:
                return False
        return True


    def pairwise_spacing_safe(self, pairwise_spacing: Sequence[Dict[str, float]]) -> bool:
        values = [
            float(value)
            for entry in pairwise_spacing
            for value in entry.values()
            if np.isfinite(value)
        ]
        if not values:
            return True
        return min(values) >= self.min_chaser_separation - self.target_safety_tolerance


    def spacing_history(
        self,
        actual: Dict[str, List[List[float]]],
        anom_history: Sequence[float],
    ) -> List[Dict[str, float]]:
        chaser_ids = list(actual.keys())
        spacing = []
        for idx in range(len(anom_history)):
            entry: Dict[str, float] = {}
            for left_idx, left_id in enumerate(chaser_ids):
                for right_id in chaser_ids[left_idx + 1:]:
                    left = np.asarray(actual[left_id][min(idx, len(actual[left_id]) - 1)][:3])
                    right = np.asarray(actual[right_id][min(idx, len(actual[right_id]) - 1)][:3])
                    entry[f"{left_id}:{right_id}"] = float(np.linalg.norm(left - right))
            spacing.append(entry)
        return spacing


    def planned_separation_messages(self, plans: Dict[str, Any]) -> List[str]:
        """Check both newly planned tubes before any fleet command is applied."""
        messages = []
        ids = list(plans)
        for i, key in enumerate(ids):
            for other in ids[i + 1:]:
                first, second = plans[key], plans[other]
                distance = np.linalg.norm(np.asarray(first.result.state_trajectory)[:3].T -
                                          np.asarray(second.result.state_trajectory)[:3].T, axis=1)
                required = self.min_chaser_separation + first.profile.position_radius + second.profile.position_radius
                if np.min(distance - required) < -self.target_safety_tolerance:
                    messages.append(f"{key}/{other}: simultaneous planned tubes conflict")
        return messages
