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

"""Fleet initialization, separate from mission policy and numerical execution."""
from __future__ import annotations

from typing import List, Sequence, TYPE_CHECKING
import numpy as np
from orbexa.simulation.approach.models import ChaserConfig

if TYPE_CHECKING:
    from orbexa.core.configuration.model import SimulationConfig
    from orbexa.core.targets.prescribed import TargetGeometry


class FleetInitializer:
    """Deterministic initial states and explicit docking assignments."""

    def __init__(self, config: SimulationConfig, target: TargetGeometry):
        self.config = config
        self.target = target

    def chasers(self, mission: str) -> List[ChaserConfig]:
        if mission == "single":
            return self.single_chaser()
        if mission == "multi":
            return self.multi_chasers()
        raise ValueError(f"Unknown ADTMPC mission: {mission}")

    def single_chaser(self) -> List[ChaserConfig]:
        initial_states = [
            np.asarray(self.config.orbit.initial_conditions, dtype=float),
        ]
        return self.build_chaser_configs(initial_states)


    def multi_chasers(self) -> List[ChaserConfig]:
        return self.build_chaser_configs(
            self.multi_initial_states(int(self.config.num_chasers))
        )


    def multi_initial_states(self, count: int) -> List[np.ndarray]:
        """Create deterministic far-field starts for a configurable chaser fleet."""
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count <= 0:
            raise ValueError("num_chasers must be a positive integer")

        base_state = np.asarray(self.config.orbit.initial_conditions, dtype=float)
        base_radius = max(float(np.linalg.norm(base_state[:2])), 5.5)
        base_angle = float(np.arctan2(base_state[1], base_state[0]))
        axial_span = max(abs(float(base_state[2])), 0.7)
        inward_speed = max(float(np.linalg.norm(base_state[3:5])), 0.018)
        axial_speed = max(abs(float(base_state[5])), 0.004)

        states: List[np.ndarray] = []
        golden_offset = np.pi * (3.0 - np.sqrt(5.0))
        for idx in range(count):
            angle = base_angle + 2.0 * np.pi * idx / count
            radial = base_radius * (1.0 + 0.04 * np.sin(idx * golden_offset))
            vertical_phase = angle + 0.5 * idx * golden_offset
            position = np.array(
                [
                    radial * np.cos(angle),
                    radial * np.sin(angle),
                    axial_span * np.sin(vertical_phase),
                ],
                dtype=float,
            )
            radial_unit = position[:2] / max(float(np.linalg.norm(position[:2])), 1.0e-9)
            tangential_unit = np.array([-radial_unit[1], radial_unit[0]], dtype=float)
            velocity_xy = -inward_speed * radial_unit + 0.25 * inward_speed * tangential_unit
            velocity = np.array(
                [
                    velocity_xy[0],
                    velocity_xy[1],
                    axial_speed * np.cos(vertical_phase),
                ],
                dtype=float,
            )
            states.append(np.concatenate([position, velocity]))
        return states


    def build_chaser_configs(
        self, initial_states: Sequence[np.ndarray]
    ) -> List[ChaserConfig]:
        initial_states = [np.asarray(value, dtype=float) for value in initial_states]
        if not initial_states or any(value.shape != (6,) or not np.all(np.isfinite(value)) for value in initial_states):
            raise ValueError("A mission fleet requires finite six-state initial conditions")
        assignments = self.target.assign_docking_targets(initial_states)
        if len(assignments) != len(initial_states):
            raise ValueError("The target must assign exactly one docking point to each chaser")
        return [
            ChaserConfig(
                chaser_id=f"chaser_{idx + 1}",
                initial_state=np.asarray(initial_states[idx], dtype=float),
                docking_point_body=np.asarray(assignment["body_position"], dtype=float),
                docking_normal_body=np.asarray(assignment["body_normal"], dtype=float),
                docking_azimuth=float(assignment["azimuth"]),
                docking_surface=str(assignment["surface"]),
                docking_standoff=float(assignment["standoff"]),
                docking_candidate_index=int(assignment["candidate_index"]),
                attachment_patch_id=assignment.get("attachment_patch_id"),
                attachment_component_id=assignment.get("attachment_component_id"),
            )
            for idx, assignment in enumerate(assignments)
        ]
