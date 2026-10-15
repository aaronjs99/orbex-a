"""Compact release and staged, component-aware spacecraft approach references."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np

from orbexa.planning.attachment_routes import CompositeAttachmentPlanner, _point, _positive
from orbexa.simulation.deployment import _quintic
from orbexa.simulation.fleet import FleetInitializer
from orbexa.simulation.mission_models import ChaserConfig


@dataclass(frozen=True)
class CompositeDeploymentOptions:
    """Explicit reference geometry and anomaly-domain speed/acceleration limits.

    These limits schedule a geometric reference; they do not prove bounded
    control feasibility. A common carrier/release mechanism is not simulated.
    """

    release_direction_body: tuple[float, float, float] = (0., -1., 0.)
    release_center_body_m: tuple[float, float, float] | None = None
    release_clearance_m: float = 4.
    separation_buffer_m: float = .10
    staging_offset_m: float = 2.
    route_buffer_m: float = .05
    reference_tube_radius_m: float = .002
    reference_speed_m_per_rad: float = 20.
    reference_acceleration_m_per_rad2: float = 50.
    minimum_segment_duration_rad: float = .10
    staging_position_tolerance_m: float = .15
    staging_velocity_tolerance_m_per_rad: float = .20
    attachment_patch_ids: tuple[str, ...] | None = None

    def __post_init__(self):
        direction = _point(self.release_direction_body, "Release direction")
        if np.linalg.norm(direction) == 0:
            raise ValueError("Release direction must be nonzero")
        object.__setattr__(self, "release_direction_body", tuple(direction / np.linalg.norm(direction)))
        if self.release_center_body_m is not None:
            object.__setattr__(self, "release_center_body_m", tuple(_point(self.release_center_body_m, "Release center")))
        for name in ("release_clearance_m", "separation_buffer_m", "staging_offset_m", "route_buffer_m",
                     "reference_speed_m_per_rad", "reference_acceleration_m_per_rad2", "minimum_segment_duration_rad",
                     "staging_position_tolerance_m", "staging_velocity_tolerance_m_per_rad"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        object.__setattr__(self, "reference_tube_radius_m", _positive(
            self.reference_tube_radius_m, "Reference tube radius", zero=True))
        if self.attachment_patch_ids is not None:
            ids = tuple(self.attachment_patch_ids)
            if not ids or len(set(ids)) != len(ids) or any(not isinstance(key, str) or not key for key in ids):
                raise ValueError("Attachment patch IDs must be nonempty unique strings")
            object.__setattr__(self, "attachment_patch_ids", ids)


class CompositePatchDeployment:
    """One finite cluster, explicit patch matching and sequential safe routes.

    References pass around every declared appendage and stop at patch-normal
    staging points. The docking phase subsequently tracks the final COM pose.
    Sequential movement keeps all other reference points fixed in target-body
    coordinates; rotations preserve their pair distances. Actual trajectories
    still require the nonlinear controller and whole-fleet swept admission.
    """

    def __init__(self, config, target, *, body_radius_m, options=None):
        self.config, self.target = config, target
        self.options = CompositeDeploymentOptions() if options is None else options
        if not isinstance(self.options, CompositeDeploymentOptions):
            raise TypeError("options must be CompositeDeploymentOptions")
        self.body_radius = _positive(body_radius_m, "Deployment body radius")
        count = config.num_chasers
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
            raise ValueError("Composite deployment needs a positive integer fleet size")
        self.count = int(count)
        if not hasattr(target, "geometry") or target.collision_params("rendezvous", 0.).get("shape") != "composite":
            raise ValueError("Composite deployment requires the complete explicit spacecraft geometry")
        self.planner = CompositeAttachmentPlanner(target.geometry, body_radius_m=self.body_radius,
            reference_tube_radius_m=self.options.reference_tube_radius_m,
            clearance_m=config.mission.target_barrier_clearance,
            minimum_pair_separation_m=config.mpc.min_chaser_separation, route_buffer_m=self.options.route_buffer_m)
        direction = np.asarray(self.options.release_direction_body)
        if self.options.release_center_body_m is None:
            vertices = np.concatenate([np.asarray(c.nominal_vertices_body_m, dtype=float) for c in target.geometry.components])
            support = float(np.max(vertices @ direction)) + max(c.uncertainty_m for c in target.geometry.components)
            self.release_center = direction * (support + self.planner.padding + self.options.release_clearance_m)
        else:
            self.release_center = np.asarray(self.options.release_center_body_m)
        tangent = np.cross(direction, np.eye(3)[int(np.argmin(np.abs(direction)))])
        tangent /= np.linalg.norm(tangent)
        second = np.cross(direction, tangent)
        width = int(np.ceil(np.sqrt(self.count)))
        offsets = np.array([(i % width) * tangent + (i // width) * second for i in range(self.count)])
        offsets -= np.mean(offsets, axis=0)
        self.release_pitch = (config.mpc.min_chaser_separation + 2 * self.options.reference_tube_radius_m
                              + self.options.separation_buffer_m)
        self.initial_body_positions = self.release_center + self.release_pitch * offsets
        self.routes = self.planner.plan(self.initial_body_positions, com_standoff_m=target.docking_standoff,
            staging_offset_m=self.options.staging_offset_m, patch_ids=self.options.attachment_patch_ids)
        self.schedule, clock = {}, 0.
        for route in sorted(self.routes, key=lambda value:value.release_order):
            segments = []
            for first, last in zip(route.waypoints_body_m, route.waypoints_body_m[1:]):
                length = float(np.linalg.norm(last - first))
                duration = max(self.options.minimum_segment_duration_rad,
                    1.875 * length / self.options.reference_speed_m_per_rad,
                    np.sqrt((10 * np.sqrt(3) / 3) * length / self.options.reference_acceleration_m_per_rad2))
                segments.append((clock, clock + duration, first.copy(), last.copy()))
                clock += duration
            self.schedule[route.chaser_index] = tuple(segments)
        self._staging_anomaly = float(clock)
        domain = getattr(target, "anomaly_domain", None)
        if domain is not None and not domain[0] <= 0 < self.staging_anomaly < domain[1]:
            raise ValueError(f"Target domain must extend beyond the complete staged reference ({self.staging_anomaly:.9g} rad)")

    @property
    def staging_anomaly(self):
        return self._staging_anomaly

    def rotation_kinematics(self, anomaly):
        states = np.array([self.target.docking_state(axis, anomaly) for axis in np.eye(3)], dtype=float)
        acceleration = np.array([self.target.docking_acceleration(axis, anomaly) for axis in np.eye(3)], dtype=float)
        if states.shape != (3, 6) or acceleration.shape != (3, 3) or not np.all(np.isfinite(states)) or not np.all(np.isfinite(acceleration)):
            raise ValueError("Composite reference needs finite rigid target-point kinematics")
        return states[:, :3].T, states[:, 3:].T, acceleration.T

    def body_kinematics(self, index, anomaly):
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or not 0 <= index < self.count:
            raise ValueError("Composite deployment index is outside its fleet")
        if not np.isfinite(anomaly):
            raise ValueError("Composite reference anomaly must be finite")
        route, segments = self.routes[index], self.schedule[index]
        if anomaly <= segments[0][0]:
            return self.initial_body_positions[index].copy(), np.zeros(3), np.zeros(3)
        for start, end, first, last in segments:
            if anomaly < end:
                progress, rate, acceleration = _quintic(anomaly - start, end - start)
                delta = last - first
                return first + progress * delta, rate * delta, acceleration * delta
        return route.staging_point_body_m.copy(), np.zeros(3), np.zeros(3)

    def kinematics(self, index, anomaly):
        point, velocity, acceleration = self.body_kinematics(index, anomaly)
        rotation, derivative, second = self.rotation_kinematics(float(anomaly))
        position = rotation @ point
        rate = derivative @ point + rotation @ velocity
        force = second @ point + 2 * derivative @ velocity + rotation @ acceleration
        return np.r_[position, rate], np.r_[rate, force]

    def assignment(self, index, state):
        self.body_kinematics(index, 0.)
        route = self.routes[index]
        return ChaserConfig(chaser_id=f"chaser_{index + 1}", initial_state=np.asarray(state, dtype=float),
            docking_point_body=route.final_com_point_body_m, docking_normal_body=route.normal_body,
            docking_azimuth=float(np.arctan2(route.normal_body[1], route.normal_body[0])),
            docking_surface="attachment_patch", docking_standoff=self.target.docking_standoff,
            docking_candidate_index=index, attachment_patch_id=route.patch_id,
            attachment_component_id=route.component_id)

    def scenario_metadata(self):
        return dict(deployment=dict(kind="compact_component_aware_patch_routes", options=asdict(self.options),
            geometry_id=self.target.geometry.geometry_id, geometry_fingerprint=self.target.geometry.fingerprint,
            count=self.count, initial_anomaly=0., release_center_body_m=self.release_center.tolist(),
            common_direction_position=(self.rotation_kinematics(0.)[0] @ self.release_center).tolist(),
            initial_pitch_m=float(self.release_pitch), staging_anomaly=self.staging_anomaly,
            assignment_strategy="component_visibility_route_length_matching_then_sequential_fleet_screening",
            route_order=[route.chaser_index for route in sorted(self.routes, key=lambda value:value.release_order)],
            routes=[dict(chaser_index=route.chaser_index, patch_id=route.patch_id, component_id=route.component_id,
                final_com_point_body_m=route.final_com_point_body_m.tolist(), staging_point_body_m=route.staging_point_body_m.tolist(),
                waypoints_body_m=route.waypoints_body_m.tolist(), length_m=route.length_m,
                schedule_anomaly=[[row[0], row[1]] for row in self.schedule[route.chaser_index]]) for route in self.routes],
            reference_geometry_only=True,
            scope="Numerically screened finite-bus reference routes around all declared components; no carrier/ejection model, actuator feasibility or swept-flow proof is implied"))

    def create_scenario(self, *, horizon_steps=None, **dependencies):
        from orbexa.simulation.composite_deployment_policy import CompositeDeploymentPolicy
        from orbexa.simulation.mission_scenario import MissionScenario
        if set(dependencies).intersection({"config", "target", "fleet", "reference"}):
            raise ValueError("Composite deployment owns config, target, fleet and reference")
        return MissionScenario(config=self.config, target=self.target, fleet=CompositeDeploymentFleet(self),
            reference=CompositeDeploymentPolicy(self, horizon_steps=horizon_steps), **dependencies)


class CompositeDeploymentFleet(FleetInitializer):
    def __init__(self, deployment):
        super().__init__(deployment.config, deployment.target)
        self.deployment = deployment

    def single_chaser(self):
        if self.deployment.count != 1:
            raise ValueError("Use mission='multi' for a multi-agent composite deployment")
        return self.multi_chasers()

    def multi_chasers(self):
        return [self.deployment.assignment(index, self.deployment.kinematics(index, 0.)[0])
                for index in range(self.deployment.count)]

    def multi_initial_states(self, count):
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count != self.deployment.count:
            raise ValueError("Composite fleet count must match its predeclared release routes")
        return [self.deployment.kinematics(index, 0.)[0] for index in range(count)]

    def build_chaser_configs(self, initial_states):
        states = np.asarray(initial_states, dtype=float)
        expected = np.asarray(self.multi_initial_states(self.deployment.count))
        if states.shape != expected.shape or not np.array_equal(states, expected):
            raise ValueError("Changed release states require explicit composite route replanning")
        return [self.deployment.assignment(index, state) for index, state in enumerate(states)]
