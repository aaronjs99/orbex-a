"""Moving arc references and current-state gating for compact free-flight staging."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np

from orbexa.control.tracking_reference import sample_tracking_reference
from orbexa.simulation.mission_models import ChaserConfig, MissionPhase
from orbexa.simulation.mission_policy import MissionReferencePolicy

if TYPE_CHECKING:
    from orbexa.simulation.coupled_state import CoupledMissionState
    from orbexa.simulation.deployment import CompactArcDeployment


class CompactDeploymentPolicy(MissionReferencePolicy):
    """Fan out, contract to a rotating shell, and arm only ready side corridors."""

    def __init__(self, deployment: CompactArcDeployment, horizon_steps: Optional[int] = None):
        super().__init__(deployment.config, deployment.target, horizon_steps)
        self.deployment = deployment

    def initial_phase(self) -> MissionPhase:
        return MissionPhase("deployment", "rendezvous")

    def select_phase(self, phase: MissionPhase, state: CoupledMissionState,
                     chasers: Sequence[ChaserConfig]) -> MissionPhase:
        if phase.operation == "docking":
            return phase
        if state.anomaly < self.deployment.options.fanout_duration:
            return MissionPhase("deployment", "rendezvous")
        if self.staging_ready(state, chasers):
            return MissionPhase("docking", "docking")
        return MissionPhase("staging", "rendezvous")

    def staging_margins(self, state: CoupledMissionState, chasers: Sequence[ChaserConfig]) -> Dict[str, float]:
        """Nonnegative margins certify this gate only, not the next nonlinear plan.

        Corridor support uses the solver's Euclidean tube inflation, which also
        encloses the current component box. Both nominal boxes and actual states
        must track the co-rotating shell and preserve target/fleet exclusion.
        """
        options, cfg = self.deployment.options, self.config
        margins = {"reference_staged": state.anomaly - self.deployment.staging_anomaly}
        if len(chasers) != self.deployment.count or len({c.chaser_id for c in chasers}) != len(chasers):
            raise ValueError("Staging requires the complete deployment fleet")
        rotation, _, _ = self.deployment.rotation_kinematics(state.anomaly)
        radii = {}
        for chaser in chasers:
            key = chaser.chaser_id
            center, actual, phi, omega = (np.asarray(values[key], dtype=float)
                                          for values in (state.centers, state.states, state.phi, state.omega))
            if (center.shape != (6,) or actual.shape != (6,) or phi.shape != (3,) or omega.shape != (3,) or
                    any(not np.all(np.isfinite(value)) for value in (center, actual, phi, omega)) or
                    np.any(phi < 0) or np.any(omega < 0)):
                raise ValueError("Staging requires finite current states and nonnegative component tubes")
            reference, _ = self.deployment.kinematics(chaser.docking_candidate_index, state.anomaly)
            radius = radii[key] = float(np.linalg.norm(omega))
            velocity_box = phi + np.asarray(cfg.tube.sliding_gains) * omega
            normal_body = np.asarray(chaser.docking_normal_body, dtype=float).copy()
            normal_body[2] = 0.0
            normal_norm = np.linalg.norm(normal_body)
            if normal_norm <= 0:
                raise ValueError("Staging requires a nonzero assigned cylinder-side normal")
            normal = rotation @ (normal_body / normal_norm)
            margins.update({
                f"{key}:position": options.staging_position_tolerance - np.linalg.norm(np.abs(center[:3] - reference[:3]) + omega),
                f"{key}:velocity": options.staging_velocity_tolerance - np.linalg.norm(np.abs(center[3:] - reference[3:]) + velocity_box),
                f"{key}:actual_position": options.staging_position_tolerance - np.linalg.norm(actual[:3] - reference[:3]),
                f"{key}:actual_velocity": options.staging_velocity_tolerance - np.linalg.norm(actual[3:] - reference[3:]),
                f"{key}:corridor": float(normal @ center[:3]) - radius - self.target.radius * (1.0 + self.target.tolerance) - cfg.mission.target_barrier_clearance,
                f"{key}:nominal_target": np.linalg.norm(center[:3]) - radius - self.target.rendezvous_radius,
                f"{key}:actual_target": np.linalg.norm(actual[:3]) - self.target.rendezvous_radius,
            })
        for index, chaser in enumerate(chasers):
            key = chaser.chaser_id
            for other in chasers[index + 1:]:
                other_key = other.chaser_id
                margins[f"{key}/{other_key}:nominal_spacing"] = (
                    np.linalg.norm(state.centers[key][:3] - state.centers[other_key][:3]) -
                    cfg.mpc.min_chaser_separation - radii[key] - radii[other_key])
                margins[f"{key}/{other_key}:actual_spacing"] = (
                    np.linalg.norm(state.states[key][:3] - state.states[other_key][:3]) - cfg.mpc.min_chaser_separation)
        return {key: float(value) for key, value in margins.items()}

    def staging_ready(self, state: CoupledMissionState, chasers: Sequence[ChaserConfig]) -> bool:
        return all(np.isfinite(value) and value >= 0.0 for value in self.staging_margins(state, chasers).values())

    def phase_reference_state(self, phase: MissionPhase, chaser: ChaserConfig,
                              state: np.ndarray, anom: float, tube_radius: float = 0.0) -> np.ndarray:
        if phase.operation == "docking":
            return super().phase_reference_state(phase, chaser, state, anom, tube_radius)
        return self.deployment.kinematics(chaser.docking_candidate_index, anom)[0]

    def phase_reference_preview(self, phase: MissionPhase, chaser: ChaserConfig,
                                current_reference: np.ndarray, anom: float, horizon_steps: int,
                                matrices: Tuple, time_periapsis: float) -> Tuple[np.ndarray, np.ndarray]:
        if phase.operation == "docking":
            return super().phase_reference_preview(phase, chaser, current_reference, anom,
                                                   horizon_steps, matrices, time_periapsis)
        index = chaser.docking_candidate_index
        return sample_tracking_reference(anom + np.arange(horizon_steps) * self.config.anom_step,
            lambda q: self.deployment.kinematics(index, q)[0],
            lambda q: self.deployment.kinematics(index, q)[1],
            lambda q: matrices[0](q, time_periapsis), matrices[1](), lambda q: matrices[4](q, time_periapsis))

    def scenario_metadata(self) -> Dict[str, Any]:
        return self.deployment.scenario_metadata()
