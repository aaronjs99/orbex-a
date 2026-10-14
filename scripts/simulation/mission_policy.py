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

"""Configurable phase and tracking policy for the qualified mission model."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
from orbexa.control.tracking_reference import sample_tracking_reference
from orbexa.simulation.mission_models import ChaserConfig, MissionPhase

if TYPE_CHECKING:
    from orbexa.core.config import SimulationConfig
    from orbexa.core.target_geometry import TargetGeometry
    from orbexa.simulation.coupled_state import CoupledMissionState


class MissionReferencePolicy:
    """Rendezvous transition, reference preview, horizons and phase costs."""

    def __init__(self, config: SimulationConfig, target: TargetGeometry, horizon_steps: Optional[int] = None):
        if horizon_steps is not None and (isinstance(horizon_steps, bool) or not isinstance(horizon_steps, (int, np.integer)) or horizon_steps < 2):
            raise ValueError("horizon_steps must be an integer of at least two nodes")
        self.config = config
        self.target = target
        self.horizon_steps = None if horizon_steps is None else int(horizon_steps)

    def initial_phase(self) -> MissionPhase:
        """Select the first phase without constructing or mutating the controller."""
        return MissionPhase("rendezvous", "rendezvous")

    def select_phase(self, phase: MissionPhase, state: CoupledMissionState,
                     chasers: Sequence[ChaserConfig]) -> MissionPhase:
        """Choose the next operation from the current committed fleet state."""
        if phase.operation == "rendezvous" and self.rendezvous_goal_satisfied(state.states, chasers):
            return MissionPhase("docking", "docking")
        return phase

    def scenario_metadata(self) -> Dict[str, Any]:
        """Extra scenario provenance supplied by an injected policy."""
        return {}

    def phase_reference_state(
        self,
        phase: MissionPhase,
        chaser: ChaserConfig,
        state: np.ndarray,
        anom: float,
        tube_radius: float = 0.0,
    ) -> np.ndarray:
        if phase.operation == "docking":
            return self.target.docking_state(chaser.docking_point_body, anom)
        direction = np.asarray(state[:3], dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-9:
            direction = np.array([1.0, 0.0, 0.0])
            norm = 1.0
        physical_radius = self.target.rendezvous_radius + self.config.mission.rendezvous_standoff
        expanded_radius = self.target.rendezvous_radius + float(tube_radius) + 0.2
        approach_radius = max(physical_radius, expanded_radius)
        return np.concatenate((direction / norm * approach_radius, np.zeros(3)))


    def phase_reference_preview(
        self,
        phase: MissionPhase,
        chaser: ChaserConfig,
        current_reference: np.ndarray,
        anom: float,
        horizon_steps: int,
        matrices: Tuple,
        time_periapsis: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Preview target motion and the nominal inputs needed to track it."""
        if phase.operation == "docking":
            def state_at(value):
                return self.target.docking_state(chaser.docking_point_body, value)

            def derivative_at(value):
                state = state_at(value)
                acceleration = self.target.docking_acceleration(chaser.docking_point_body, value)
                return np.concatenate((state[3:], acceleration))
        else:
            def state_at(value):
                return current_reference

            def derivative_at(value):
                return np.zeros_like(current_reference)

        anomalies = anom + np.arange(horizon_steps) * self.config.anom_step
        return sample_tracking_reference(
            anomalies,
            state_at,
            derivative_at,
            lambda value: matrices[0](value, time_periapsis),
            matrices[1](),
            lambda value: matrices[4](value, time_periapsis),
        )


    def rendezvous_goal_satisfied(
        self,
        states: Dict[str, np.ndarray],
        chasers: Sequence[ChaserConfig],
    ) -> bool:
        desired_radius = max(
            self.target.rendezvous_radius + self.config.mission.rendezvous_standoff,
            self.config.mission.rendezvous_capture_radius,
        )
        for chaser in chasers:
            state = np.asarray(states[chaser.chaser_id], dtype=float)
            distance = float(np.linalg.norm(state[:3]))
            velocity = float(np.linalg.norm(state[3:]))
            outside_target = distance >= self.target.rendezvous_radius
            near_shell = distance <= desired_radius + self.config.mission.rendezvous_position_tolerance
            if not (outside_target and near_shell and velocity <= self.config.mission.rendezvous_velocity_tolerance):
                return False
        return True


    def phase_state_weights(self, phase: MissionPhase) -> np.ndarray:
        weights = np.asarray(self.config.mpc.Q, dtype=float).copy()
        if phase.operation == "docking":
            position_factor = self.config.mission.docking_position_weight_factor
            velocity_factor = self.config.mission.docking_velocity_weight_factor
            weights[:3, :3] *= position_factor
            weights[3:, 3:] *= velocity_factor
            cross_factor = np.sqrt(position_factor * velocity_factor)
            weights[:3, 3:] *= cross_factor
            weights[3:, :3] *= cross_factor
        return weights


    def horizon_steps_for_phase(self, phase: MissionPhase) -> int:
        if self.horizon_steps is not None:
            return int(self.horizon_steps)
        return int(self.config.mpc.horizon_steps.get(phase.operation, 10))


    def configured_horizon_metadata(self) -> Dict[str, int]:
        if self.horizon_steps is not None:
            return {"all": int(self.horizon_steps)}
        return {
            key: int(value)
            for key, value in self.config.mpc.horizon_steps.items()
        }


    def terminal_bounds(self, chaser: ChaserConfig, anomaly: float, center: np.ndarray,
                        phi: np.ndarray, omega: np.ndarray) -> Tuple[float, float]:
        goal = self.target.docking_state(chaser.docking_point_body, anomaly)
        position_bound = np.linalg.norm(np.abs(center[:3] - goal[:3]) + omega)
        velocity_bound = np.linalg.norm(np.abs(center[3:] - goal[3:]) +
            phi + np.asarray(self.config.tube.sliding_gains) * omega)
        return float(position_bound), float(velocity_bound)

    def terminal_bounds_admissible(self, position_bound: float, velocity_bound: float) -> bool:
        return (position_bound <= self.config.mpc.docking_position_tolerance and
                velocity_bound <= self.config.mpc.docking_velocity_tolerance)
