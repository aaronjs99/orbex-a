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

"""policy.py: Configurable phase and tracking policy for the qualified mission model.

Copyright: Aaron John Sabu
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from orbexa.control.mpc.references import sample_tracking_reference
from orbexa.mission.approach.models import ChaserConfig, MissionPhase

if TYPE_CHECKING:
    from orbexa.mission.approach.state import CoupledMissionState
    from orbexa.models.configuration.model import SimulationConfig
    from orbexa.models.targets.prescribed import TargetGeometry


class MissionReferencePolicy:
    """Rendezvous transition, reference preview, horizons and phase costs."""

    def __init__(
        self,
        config: SimulationConfig,
        target: TargetGeometry,
        horizon_steps: int | None = None,
    ):
        """Store the config and target and the optional fixed horizon.

        Raises ValueError if horizon_steps is given and is not an integer of at least
        two.
        """
        if horizon_steps is not None and (
            isinstance(horizon_steps, bool)
            or not isinstance(horizon_steps, (int, np.integer))
            or horizon_steps < 2
        ):
            raise ValueError("horizon_steps must be an integer of at least two nodes")
        self.config = config
        self.target = target
        self.horizon_steps = None if horizon_steps is None else int(horizon_steps)

    def initial_phase(self) -> MissionPhase:
        """Select the first phase without constructing or mutating the controller."""
        return MissionPhase("rendezvous", "rendezvous")

    def select_phase(
        self,
        phase: MissionPhase,
        state: CoupledMissionState,
        chasers: Sequence[ChaserConfig],
    ) -> MissionPhase:
        """Choose the next operation from the current committed fleet state."""
        if phase.operation == "rendezvous" and self.rendezvous_goal_satisfied(
            state.states, chasers
        ):
            return MissionPhase("docking", "docking")
        return phase

    def scenario_metadata(self) -> dict[str, Any]:
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
        """Return the 6-vector goal state for a chaser at anomaly anom (rad).

        Docking: the target's docking state at the chaser's docking point.
        Otherwise: a point at zero velocity on the sphere around the target in the
        direction of the chaser position (+x if it is within 1e-9 m of the origin),
        with radius the larger of the rendezvous radius plus standoff and the
        rendezvous radius plus tube_radius plus 0.2 m.
        """
        if phase.operation == "docking":
            return self.target.docking_state(chaser.docking_point_body, anom)
        direction = np.asarray(state[:3], dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-9:
            direction = np.array([1.0, 0.0, 0.0])
            norm = 1.0
        physical_radius = (
            self.target.rendezvous_radius + self.config.mission.rendezvous_standoff
        )
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
        matrices: tuple,
        time_periapsis: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Preview target motion and the nominal inputs needed to track it."""
        if phase.operation == "docking":

            def state_at(value):
                """Return the docking-point state at anomaly value."""
                return self.target.docking_state(chaser.docking_point_body, value)

            def derivative_at(value):
                """Return the anomaly derivative of the docking-point state."""
                state = state_at(value)
                acceleration = self.target.docking_acceleration(
                    chaser.docking_point_body, value
                )
                return np.concatenate((state[3:], acceleration))
        else:

            def state_at(value):
                """Return the fixed current reference, independent of value."""
                return current_reference

            def derivative_at(value):
                """Return zeros: the reference is held constant."""
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
        states: dict[str, np.ndarray],
        chasers: Sequence[ChaserConfig],
    ) -> bool:
        """Return True if every chaser meets the rendezvous hand-over conditions.

        Each chaser must be at least the target rendezvous radius from the origin,
        within the position tolerance of the larger of the standoff shell and the
        capture radius, and moving no faster than the velocity tolerance.
        """
        desired_radius = max(
            self.target.rendezvous_radius + self.config.mission.rendezvous_standoff,
            self.config.mission.rendezvous_capture_radius,
        )
        for chaser in chasers:
            state = np.asarray(states[chaser.chaser_id], dtype=float)
            distance = float(np.linalg.norm(state[:3]))
            velocity = float(np.linalg.norm(state[3:]))
            outside_target = distance >= self.target.rendezvous_radius
            near_shell = (
                distance
                <= desired_radius + self.config.mission.rendezvous_position_tolerance
            )
            if not (
                outside_target
                and near_shell
                and velocity <= self.config.mission.rendezvous_velocity_tolerance
            ):
                return False
        return True

    def phase_state_weights(self, phase: MissionPhase) -> np.ndarray:
        """Return a copy of the MPC state weight matrix Q for the phase.

        For docking, the position, velocity and cross blocks are scaled by the
        configured docking weight factors (cross blocks by their geometric mean).
        """
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
        """Return the horizon length in nodes for the phase.

        The injected horizon_steps if set; otherwise config.mpc.horizon_steps for the
        phase operation, defaulting to 10.
        """
        if self.horizon_steps is not None:
            return int(self.horizon_steps)
        return int(self.config.mpc.horizon_steps.get(phase.operation, 10))

    def configured_horizon_metadata(self) -> dict[str, int]:
        """Return the horizon lengths for the record.

        {"all": n} for an injected horizon, otherwise the per-operation configured
        values.
        """
        if self.horizon_steps is not None:
            return {"all": int(self.horizon_steps)}
        return {key: int(value) for key, value in self.config.mpc.horizon_steps.items()}
