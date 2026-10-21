"""Finite post-release arc deployment in the supported free-flight model.

Geometry and timing live here; the phase/readiness policy lives in
``deployment_policy``. No carrier, release impulse, contact or actuator model
is introduced. All trajectory derivatives use true anomaly, not seconds.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np

from orbexa.simulation.approach.fleet import FleetInitializer
from orbexa.simulation.approach.models import ChaserConfig

if TYPE_CHECKING:
    from orbexa.core.configuration.model import SimulationConfig
    from orbexa.core.targets.prescribed import TargetGeometry
    from orbexa.simulation.approach.scenario import MissionScenario


@dataclass(frozen=True)
class CompactDeploymentOptions:
    """Prospective geometry, anomaly durations and staging acceptance bounds."""

    launch_radius: Optional[float] = None
    launch_azimuth: Optional[float] = None
    separation_buffer: float = 0.10
    fanout_duration: float = 3.0
    contraction_duration: float = 1.5
    staging_radius: float = 3.0
    staging_position_tolerance: float = 0.15
    staging_velocity_tolerance: float = 0.20

    def __post_init__(self) -> None:
        if self.launch_radius is not None and (isinstance(self.launch_radius, bool) or
                not np.isfinite(self.launch_radius) or self.launch_radius <= 0):
            raise ValueError("Deployment launch_radius must be finite and positive")
        for name in ("fanout_duration", "contraction_duration", "staging_radius",
                     "staging_position_tolerance", "staging_velocity_tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"Deployment {name} must be finite and positive")
        if self.launch_azimuth is not None and not np.isfinite(self.launch_azimuth):
            raise ValueError("Deployment launch_azimuth must be finite")
        if not np.isfinite(self.separation_buffer) or self.separation_buffer < 0:
            raise ValueError("Deployment separation_buffer must be finite and nonnegative")


def _quintic(value: float, duration: float) -> Tuple[float, float, float]:
    """Clamped C2 progress and its first/second anomaly derivatives."""
    if value <= 0.0:
        return 0.0, 0.0, 0.0
    if value >= duration:
        return 1.0, 0.0, 0.0
    t = value / duration
    return (t**3 * (10.0 - 15.0 * t + 6.0 * t**2),
            30.0 * t**2 * (1.0 - t)**2 / duration,
            60.0 * t * (1.0 - t) * (1.0 - 2.0 * t) / duration**2)


class CompactArcDeployment:
    """One compact arc, cyclic assignments and a smooth rotating staging ring.

    ``config.num_chasers`` fixes the fleet size. The default common direction
    is the configured start projected into the initial target equatorial plane.
    Both deployment and radial staging follow the target's prescribed rotation.
    """

    def __init__(self, config: SimulationConfig, target: Optional[TargetGeometry] = None,
                 options: Optional[CompactDeploymentOptions] = None):
        if target is None:
            from orbexa.core.targets.prescribed import TumblingCylinderTarget
            target = TumblingCylinderTarget.from_config(config)
        self.config, self.target = config, target
        self.options = options if options is not None else CompactDeploymentOptions()
        if not isinstance(self.options, CompactDeploymentOptions):
            raise TypeError("options must be CompactDeploymentOptions")
        count = config.num_chasers
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
            raise ValueError("A deployment needs a positive integer num_chasers")
        self.count = int(count)
        collision = target.collision_params("rendezvous", 0.0)
        center = np.asarray(collision.get("center"), dtype=float)
        if collision.get("shape") != "cylinder" or center.shape != (3,) or np.any(center != 0):
            raise ValueError("Compact arc deployment requires the supported target-centered cylinder")
        start = np.asarray(config.orbit.initial_conditions, dtype=float)
        self.launch_radius = (float(np.linalg.norm(start[:3])) if self.options.launch_radius is None
                              else float(self.options.launch_radius))
        rotation, _, _ = self.rotation_kinematics(0.0)
        body_direction = rotation.T @ start[:3]
        if self.options.launch_azimuth is None and np.linalg.norm(body_direction[:2]) <= 1.0e-12:
            raise ValueError("Supply launch_azimuth when the configured start has no equatorial direction")
        self.launch_azimuth = (float(np.arctan2(body_direction[1], body_direction[0]))
                               if self.options.launch_azimuth is None else float(self.options.launch_azimuth))
        self.pitch = float(config.mpc.min_chaser_separation) + self.options.separation_buffer
        if self.launch_radius <= 0 or self.pitch <= 0 or self.pitch > 2.0 * self.launch_radius:
            raise ValueError("The finite initial arc pitch must fit the positive launch radius")
        self.initial_spacing = float(2.0 * np.arcsin(self.pitch / (2.0 * self.launch_radius)))
        self.final_spacing = 2.0 * np.pi / self.count
        if self.count > 1 and self.initial_spacing > self.final_spacing:
            raise ValueError("The separated fleet cannot fit this launch arc without overlapping its cyclic gap")
        if not target.rendezvous_radius < self.options.staging_radius <= self.launch_radius:
            raise ValueError("Deployment requires target exclusion radius < staging_radius <= launch_radius")
        if self.count > 1:
            for radius, label in ((self.options.staging_radius, "staging"),
                                  (target.radius + target.docking_standoff, "assigned docking")):
                if 2.0 * radius * np.sin(np.pi / self.count) < config.mpc.min_chaser_separation:
                    raise ValueError(f"The {label} ring cannot satisfy the unchanged minimum chaser separation")

    @property
    def staging_anomaly(self) -> float:
        """End of reference motion relative to the rotating target."""
        return self.options.fanout_duration + self.options.contraction_duration

    def rotation_kinematics(self, anomaly: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Obtain exact Q, Q' and Q'' from the supported rigid target-point API."""
        states = [np.asarray(self.target.docking_state(axis, anomaly), dtype=float) for axis in np.eye(3)]
        accelerations = [np.asarray(self.target.docking_acceleration(axis, anomaly), dtype=float) for axis in np.eye(3)]
        if (any(value.shape != (6,) or not np.all(np.isfinite(value)) for value in states) or
                any(value.shape != (3,) or not np.all(np.isfinite(value)) for value in accelerations)):
            raise ValueError("Deployment target kinematics must be finite rigid-point states and accelerations")
        rotation = np.column_stack([value[:3] for value in states])
        if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1.0e-10) or np.linalg.det(rotation) <= 0:
            raise ValueError("Deployment target kinematics must supply a proper common rotation")
        return rotation, np.column_stack([value[3:] for value in states]), np.column_stack(accelerations)

    def kinematics(self, index: int, anomaly: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return [p, p'] and [p', p''] in LVLH at any preview anomaly."""
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or not 0 <= index < self.count:
            raise ValueError("Deployment index is outside the configured fleet")
        if not np.isfinite(anomaly):
            raise ValueError("Deployment anomaly must be finite")
        spread, spread_rate, spread_accel = _quintic(float(anomaly), self.options.fanout_duration)
        contract, contract_rate, contract_accel = _quintic(
            float(anomaly) - self.options.fanout_duration, self.options.contraction_duration)
        spacing_change = self.final_spacing - self.initial_spacing
        order = index - (self.count - 1) / 2.0
        angle = self.launch_azimuth + order * (self.initial_spacing + spacing_change * spread)
        angle_rate, angle_accel = order * spacing_change * spread_rate, order * spacing_change * spread_accel
        radial_change = self.options.staging_radius - self.launch_radius
        radius = self.launch_radius + radial_change * contract
        radius_rate, radius_accel = radial_change * contract_rate, radial_change * contract_accel
        radial = np.array([np.cos(angle), np.sin(angle), 0.0])
        tangent = np.array([-np.sin(angle), np.cos(angle), 0.0])
        body_position = radius * radial
        body_velocity = radius_rate * radial + radius * angle_rate * tangent
        body_accel = ((radius_accel - radius * angle_rate**2) * radial +
                      (2.0 * radius_rate * angle_rate + radius * angle_accel) * tangent)
        rotation, rotation_rate, rotation_accel = self.rotation_kinematics(float(anomaly))
        position = rotation @ body_position
        velocity = rotation_rate @ body_position + rotation @ body_velocity
        acceleration = rotation_accel @ body_position + 2.0 * rotation_rate @ body_velocity + rotation @ body_accel
        return np.concatenate((position, velocity)), np.concatenate((velocity, acceleration))

    def assignment(self, index: int, state: np.ndarray) -> ChaserConfig:
        """Assign the final cyclic side point corresponding to the arc order."""
        self.kinematics(index, 0.0)  # Validate the public index before constructing metadata.
        angle = self.launch_azimuth + (index - (self.count - 1) / 2.0) * self.final_spacing
        normal = np.array([np.cos(angle), np.sin(angle), 0.0])
        return ChaserConfig(chaser_id=f"chaser_{index + 1}", initial_state=state,
            docking_point_body=(self.target.radius + self.target.docking_standoff) * normal,
            docking_normal_body=normal, docking_azimuth=float(np.mod(angle, 2.0 * np.pi)),
            docking_surface="cylinder_side", docking_standoff=self.target.docking_standoff,
            docking_candidate_index=index)

    def scenario_metadata(self) -> Dict[str, Any]:
        rotation, _, _ = self.rotation_kinematics(0.0)
        center = rotation @ np.array([self.launch_radius * np.cos(self.launch_azimuth),
                                      self.launch_radius * np.sin(self.launch_azimuth), 0.0])
        return {"deployment": dict(kind="compact_target_centered_arc", options=asdict(self.options),
            count=self.count, initial_anomaly=0.0, launch_radius=self.launch_radius,
            launch_azimuth=self.launch_azimuth, common_direction_position=center.tolist(),
            initial_pitch=self.pitch, initial_spacing=self.initial_spacing, final_spacing=self.final_spacing,
            arc_span=(self.count - 1) * self.initial_spacing,
            assignment_strategy="explicit_cyclic_arc_order", rotation="supported_prescribed_target",
            scope="post_release_free_flight; no carrier, ejection, contact, finite body or thruster model",
            reference_geometry_only=True)}

    def create_scenario(self, *, horizon_steps: Optional[int] = None, **dependencies: Any) -> MissionScenario:
        """Compose existing services; other explicit MissionScenario dependencies remain injectable."""
        from orbexa.simulation.approach.deployment.arc_policy import CompactDeploymentPolicy
        from orbexa.simulation.approach.scenario import MissionScenario
        if set(dependencies).intersection({"config", "target", "fleet", "reference"}):
            raise ValueError("The deployment owns config, target, fleet and reference; inject other scenario services")
        return MissionScenario(config=self.config, target=self.target, fleet=CompactDeploymentFleet(self),
            reference=CompactDeploymentPolicy(self, horizon_steps=horizon_steps), **dependencies)


class CompactDeploymentFleet(FleetInitializer):
    """Separated, co-rotating post-release states with explicit cyclic assignments."""

    def __init__(self, deployment: CompactArcDeployment):
        super().__init__(deployment.config, deployment.target)
        self.deployment = deployment

    def single_chaser(self) -> List[ChaserConfig]:
        if self.deployment.count != 1:
            raise ValueError("Use mission='multi' for a deployment configured with more than one chaser")
        return self.multi_chasers()

    def multi_chasers(self) -> List[ChaserConfig]:
        return self.build_chaser_configs(self.multi_initial_states(self.deployment.count))

    def multi_initial_states(self, count: int) -> List[np.ndarray]:
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count != self.deployment.count:
            raise ValueError("Deployment fleet size must match its trajectory and cyclic assignments")
        return [self.deployment.kinematics(index, 0.0)[0] for index in range(count)]

    def build_chaser_configs(self, initial_states: Sequence[np.ndarray]) -> List[ChaserConfig]:
        states = [np.asarray(value, dtype=float) for value in initial_states]
        if len(states) != self.deployment.count or any(value.shape != (6,) or not np.all(np.isfinite(value)) for value in states):
            raise ValueError("Deployment requires one finite six-state release condition per chaser")
        for index, state in enumerate(states):
            if np.linalg.norm(state[:3]) < self.target.rendezvous_radius:
                raise ValueError("A post-release chaser starts inside the target exclusion sphere")
            for other in states[index + 1:]:
                distance = float(np.linalg.norm(state[:3] - other[:3]))
                if distance <= 0 or distance < self.config.mpc.min_chaser_separation:
                    raise ValueError("Post-release positions must be distinct and satisfy minimum chaser separation")
        return [self.deployment.assignment(index, state) for index, state in enumerate(states)]
