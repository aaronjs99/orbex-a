"""component_policy.py: Readiness gates and reference previews for explicit spacecraft components.

Copyright: Aaron John Sabu
"""

from __future__ import annotations

import numpy as np
from orbexa.control.mpc.references import sample_tracking_reference
from orbexa.mission.approach.models import MissionPhase
from orbexa.mission.approach.references.policy import MissionReferencePolicy


class CompositeDeploymentPolicy(MissionReferencePolicy):
    """Reference policy that stages a composite deployment, then hands to docking."""

    def __init__(self, deployment, horizon_steps=None):
        """Store the deployment and initialise the base policy from its inputs."""
        super().__init__(deployment.config, deployment.target, horizon_steps)
        self.deployment = deployment

    def initial_phase(self):
        """Return the rendezvous phase named carrier_departure or deployment."""
        return MissionPhase(
            "carrier_departure"
            if self.deployment.carrier_transfer is not None
            else "deployment",
            "rendezvous",
        )

    def select_phase(self, phase, state, chasers):
        """Return the phase for the current fleet state.

        A docking phase is kept. Otherwise: carrier_departure at anomaly 0.0 or
        transfer before the transfer anomaly, deployment before the staging anomaly,
        docking once every staging margin is nonnegative, and staging if not.
        """
        if phase.operation == "docking":
            return phase
        if state.anomaly < self.deployment.transfer_anomaly:
            return MissionPhase(
                "carrier_departure" if state.anomaly == 0.0 else "transfer",
                "rendezvous",
            )
        if state.anomaly < self.deployment.staging_anomaly:
            return MissionPhase("deployment", "rendezvous")
        if all(value >= 0 for value in self.staging_margins(state, chasers).values()):
            return MissionPhase("docking", "docking")
        return MissionPhase("staging", "rendezvous")

    def staging_margins(self, state, chasers):
        """Return named staging margins; a nonnegative value means satisfied.

        Includes reference_staged (anomaly minus staging anomaly) and, per chaser,
        nominal and actual position and velocity tolerance margins, signed distance
        to the target components for the nominal centre (less tube radius) and the
        actual state, and pair spacing for nominal and actual positions. Lengths in
        m, velocities in m per rad of anomaly.
        Raises ValueError if chasers are not the complete assigned fleet with unique
        IDs, if state values have the wrong shape, are not finite or have negative
        tube components, or if any margin is not finite.
        """
        if len(chasers) != self.deployment.count or len(
            {c.chaser_id for c in chasers}
        ) != len(chasers):
            raise ValueError("Composite staging requires its complete assigned fleet")
        options, config = self.deployment.options, self.config
        rotation = self.deployment.rotation_kinematics(state.anomaly)[0]
        margins, radii = (
            {
                "reference_staged": float(
                    state.anomaly - self.deployment.staging_anomaly
                )
            },
            {},
        )
        for chaser in chasers:
            key = chaser.chaser_id
            center, actual, phi, omega = (
                np.asarray(mapping[key], dtype=float)
                for mapping in (state.centers, state.states, state.phi, state.omega)
            )
            if (
                center.shape != (6,)
                or actual.shape != (6,)
                or phi.shape != (3,)
                or omega.shape != (3,)
                or any(not np.all(np.isfinite(v)) for v in (center, actual, phi, omega))
                or np.any(phi < 0)
                or np.any(omega < 0)
            ):
                raise ValueError(
                    "Composite staging needs finite current states and nonnegative tube components"
                )
            reference = self.deployment.kinematics(
                chaser.docking_candidate_index, state.anomaly
            )[0]
            velocity_radius = phi + np.asarray(config.tube.sliding_gains) * omega
            radius = radii[key] = float(np.linalg.norm(omega))
            margins.update(
                {
                    key + ":position": options.staging_position_tolerance_m
                    - np.linalg.norm(np.abs(center[:3] - reference[:3]) + omega),
                    key + ":velocity": options.staging_velocity_tolerance_m_per_rad
                    - np.linalg.norm(
                        np.abs(center[3:] - reference[3:]) + velocity_radius
                    ),
                    key + ":actual_position": options.staging_position_tolerance_m
                    - np.linalg.norm(actual[:3] - reference[:3]),
                    key
                    + ":actual_velocity": options.staging_velocity_tolerance_m_per_rad
                    - np.linalg.norm(actual[3:] - reference[3:]),
                    key + ":nominal_components": self.target.geometry.signed_distance(
                        rotation.T @ center[:3]
                    )
                    - radius
                    - self.deployment.body_radius
                    - config.mission.target_barrier_clearance,
                    key + ":actual_components": self.target.geometry.signed_distance(
                        rotation.T @ actual[:3]
                    )
                    - self.deployment.body_radius
                    - config.mission.target_barrier_clearance,
                }
            )
        for i, chaser in enumerate(chasers):
            for other in chasers[i + 1 :]:
                first, second = chaser.chaser_id, other.chaser_id
                margins[first + "/" + second + ":nominal_spacing"] = (
                    np.linalg.norm(state.centers[first][:3] - state.centers[second][:3])
                    - config.mpc.min_chaser_separation
                    - radii[first]
                    - radii[second]
                )
                margins[first + "/" + second + ":actual_spacing"] = (
                    np.linalg.norm(state.states[first][:3] - state.states[second][:3])
                    - config.mpc.min_chaser_separation
                )
        if not all(np.isfinite(value) for value in margins.values()):
            raise ValueError("Composite staging margin is not finite")
        return {key: float(value) for key, value in margins.items()}

    def phase_reference_state(self, phase, chaser, state, anom, tube_radius=0.0):
        """Return the reference state for a chaser at anomaly anom (rad).

        Docking uses the target's docking state at the chaser's docking point;
        otherwise the deployment's LVLH reference state is used. The state and
        tube_radius arguments are not used.
        """
        if phase.operation == "docking":
            return self.target.docking_state(chaser.docking_point_body, anom)
        return self.deployment.kinematics(chaser.docking_candidate_index, anom)[0]

    def phase_reference_preview(
        self,
        phase,
        chaser,
        current_reference,
        anom,
        horizon_steps,
        matrices,
        time_periapsis,
    ):
        """Return the reference states and feedforward inputs over the horizon.

        Docking delegates to the base policy. Otherwise the deployment reference is
        sampled at horizon_steps anomalies from anom spaced by the anomaly step.
        """
        if phase.operation == "docking":
            return super().phase_reference_preview(
                phase,
                chaser,
                current_reference,
                anom,
                horizon_steps,
                matrices,
                time_periapsis,
            )
        index = chaser.docking_candidate_index
        return sample_tracking_reference(
            anom + np.arange(horizon_steps) * self.config.anom_step,
            lambda q: self.deployment.kinematics(index, q)[0],
            lambda q: self.deployment.kinematics(index, q)[1],
            lambda q: matrices[0](q, time_periapsis),
            matrices[1](),
            lambda q: matrices[4](q, time_periapsis),
        )

    def scenario_metadata(self):
        """Return the deployment's scenario metadata."""
        return self.deployment.scenario_metadata()
