"""Measured-state stationkeeping and finite-bandwidth positioner commands."""
from __future__ import annotations

from fractions import Fraction
import numpy as np
from scipy.spatial.transform import Rotation

from orbexa.core.dynamics.attitude import finite_vector


def _limit(vector, maximum):
    norm = float(np.linalg.norm(vector))
    result = np.asarray(vector) * min(1., maximum / max(norm, 1e-30))
    squared_limit = Fraction(float(maximum))**2
    # The rounded normalized vector can lie just outside its true Euclidean
    # hardware ball. Move inward, never enlarge the declared authority.
    for _ in range(8):
        if sum(Fraction(float(value))**2 for value in result) <= squared_limit:
            return result
        result = np.nextafter(result, 0.)
    raise ValueError('Positioner command cannot be rounded inside its declared authority')


class ContactController:
    """Command policy with explicit observation, bond and actuator dependencies."""

    def __init__(self, *, fleet, options, bonds, actuation, initial_values, feedback=None):
        self.fleet = fleet
        self.options = options
        self.bonds = bonds
        self.actuation = actuation
        self.feedback = feedback
        self.saturation_count = 0
        started, members = fleet.started, fleet.members
        snapshot, check_observation = fleet.snapshot, fleet.check_observation
        y = initial_values
        first_observation = snapshot(0., y)
        if feedback is not None:
            first_observation = feedback(started, first_observation)
        check_observation(first_observation, started)
        rt_est = Rotation.from_quat(first_observation['target'].quaternion_body_to_eci).as_matrix()
        reference_offsets, reference_rotations = [], []
        for member in members:
            state = first_observation['chasers'][member.id]
            reference_offsets.append(rt_est.T @ (state.position_eci_m-first_observation['target'].position_eci_m))
            reference_rotations.append(rt_est.T @ Rotation.from_quat(state.quaternion_body_to_eci).as_matrix())
        self.reference_offsets = reference_offsets
        self.reference_rotations = reference_rotations

    def command(self, t, values, diag):
        fleet, o = self.fleet, self.options
        target_properties, members = fleet.target_properties, fleet.members
        mounts, anchors, normals = fleet.mounts, fleet.anchors, fleet.normals
        mu, gravity_gradient, started = fleet.mu, fleet.gravity_gradient, fleet.started
        unpack, snapshot, gravity = fleet.unpack, fleet.snapshot, fleet.gravity
        check_observation, feedback = fleet.check_observation, self.feedback
        controls, extension_acceleration = self.actuation.controls, self.actuation.extension_acceleration
        sensed_normal_force, sensed_gap = self.actuation.sensed_normal_force, self.actuation.sensed_gap
        frozen_extension, engagement = self.actuation.frozen_extension, self.bonds.engagement
        reference_offsets, reference_rotations = self.reference_offsets, self.reference_rotations
        b, tips = unpack(values)
        observation = snapshot(t, values, diag)
        if feedback is not None:
            observation = feedback(started+t, observation)
        check_observation(observation, started+t)
        sensed_tips = tips.copy()
        if observation.get('positioner_encoders') is not None:
            encoders = observation['positioner_encoders']
            if set(encoders) != {member.id for member in members}:
                raise ValueError('Contact positioner encoders must name the exact fleet')
            for k, member in enumerate(members):
                sensed_tips[k, :3] = finite_vector(encoders[member.id]['extension_body_m'], 'Positioner extension encoder', 3)
                sensed_tips[k, 3:] = finite_vector(encoders[member.id]['extension_velocity_body_m_s'], 'Positioner velocity encoder', 3)
        # Positioner commands use their proprioceptive encoder products.
        # The plant's packed tip states remain separate.
        tips = sensed_tips
        target = observation['target']
        rt = Rotation.from_quat(target.quaternion_body_to_eci).as_matrix()
        wt = rt@target.angular_velocity_body_rad_s
        measured = observation.get('contact_wrenches_eci', {})
        measured_forces, measured_couples = [], []
        target_torque = np.zeros(3)
        for k, member in enumerate(members):
            channel = measured.get(member.id, {})
            measured_forces.append(finite_vector(channel.get('force_n', np.zeros(3)), 'Measured contact force', 3))
            measured_couples.append(finite_vector(channel.get('couple_nm', np.zeros(3)), 'Measured pad couple', 3))
            estimate = observation['chasers'][member.id]
            ri = Rotation.from_quat(estimate.quaternion_body_to_eci).as_matrix()
            tip = estimate.position_eci_m+ri@(mounts[k]+tips[k, :3])
            target_torque -= np.cross(tip-target.position_eci_m, measured_forces[-1])+measured_couples[-1]
        target_inertia = target_properties.inertia_kg_m2
        target_torque_body = rt.T@target_torque
        if gravity_gradient:
            radial = rt.T@(target.position_eci_m/np.linalg.norm(target.position_eci_m))
            target_torque_body += 3*mu/np.linalg.norm(target.position_eci_m)**3*np.cross(radial, target_inertia@radial)
        target_alpha = rt@np.linalg.solve(target_inertia, target_torque_body-
            np.cross(target.angular_velocity_body_rad_s, target_inertia@target.angular_velocity_body_rad_s))
        target_reaction_acceleration = -np.sum(measured_forces, axis=0)/target_properties.mass_kg
        for k, member in enumerate(members):
            j, state = k+1, observation['chasers'][member.id]
            ri = Rotation.from_quat(state.quaternion_body_to_eci).as_matrix()
            wi = ri@state.angular_velocity_body_rad_s
            offset = rt@reference_offsets[k]
            position_error = target.position_eci_m+offset-state.position_eci_m
            velocity_error = target.velocity_eci_m_s+np.cross(wt, offset)-state.velocity_eci_m_s
            a = gravity(target.position_eci_m)-gravity(state.position_eci_m)+target_reaction_acceleration
            a += np.cross(target_alpha, offset)+np.cross(wt, np.cross(wt, offset))
            a += o.stationkeeping_frequency_rad_s**2*position_error+2*o.stationkeeping_frequency_rad_s*velocity_error
            force_requested = ri.T@(diag['masses'][j]*a-measured_forces[k])
            expected = rt@reference_rotations[k]
            angle = Rotation.from_matrix(ri.T@expected).as_rotvec()
            inertia = diag['inertias'][j]
            torque_requested = inertia@(ri.T@target_alpha+o.attitude_frequency_rad_s**2*angle+
                2*o.attitude_frequency_rad_s*(ri.T@(wt-wi)))+np.cross(state.angular_velocity_body_rad_s, inertia@state.angular_velocity_body_rad_s)
            lever = ri@(mounts[k]+tips[k, :3])
            torque_requested -= ri.T@(np.cross(lever, measured_forces[k])+measured_couples[k])
            if gravity_gradient:
                radial = ri.T@(state.position_eci_m/np.linalg.norm(state.position_eci_m))
                torque_requested -= 3*mu/np.linalg.norm(state.position_eci_m)**3*np.cross(radial, inertia@radial)
            # Once the compliant bond can carry load, release free-flight
            # stationkeeping continuously. Cancelling measured bond damping
            # indefinitely would create a delayed positive feedback loop.
            free_flight_fraction = 1.-diag['rows'][k]['activation']
            force_requested *= free_flight_fraction
            torque_requested *= free_flight_fraction
            force = np.clip(force_requested, -o.translation_force_n, o.translation_force_n)
            torque_limit = 2*o.attitude_moment_arm_m*o.attitude_nozzle_force_n
            torque = np.clip(torque_requested, -torque_limit, torque_limit)
            self.saturation_count += int(np.any(force != force_requested) or np.any(torque != torque_requested))
            controls[j] = np.r_[force, torque]
            if b[j, 13] <= 1e-12:
                controls[j] = 0.
            patch = target.position_eci_m+rt@anchors[k]
            normal = rt@normals[k]
            cmd = ri.T@(patch-state.position_eci_m)-mounts[k]-ri.T@normal*(
                o.surface_search_depth_m+o.preload_command_factor*o.preload_n/o.compression_stiffness_n_m)
            if np.isfinite(engagement[k]) and not np.all(np.isfinite(frozen_extension[k])):
                frozen_extension[k] = tips[k, :3]
            if np.all(np.isfinite(frozen_extension[k])):
                cmd = frozen_extension[k]
            observed_tip = state.position_eci_m+ri@(mounts[k]+tips[k, :3])
            gap_est = float((observed_tip-patch)@normal)
            ranges = observation.get('contact_gaps_m', {})
            if member.id in ranges:
                gap_est = float(ranges[member.id])
                if not np.isfinite(gap_est):
                    raise ValueError('Contact gap measurement must be finite')
                slow_distance = o.slow_touch_distance_m+o.range_error_bound_m
            else:
                # Without a local range channel, reserve the entire
                # declared surface-search uncertainty before fast motion.
                slow_distance = o.slow_touch_distance_m+o.surface_search_depth_m
            sensed_gap[k] = gap_est
            sensed_normal_force[k] = float(measured_forces[k]@normal)
            speed = min(o.extension_speed_m_s, o.touch_speed_m_s+np.sqrt(
                2*o.extension_acceleration_m_s2*max(0., gap_est-slow_distance)))
            desired_velocity = _limit((cmd-tips[k, :3])/o.positioner_time_constant_s, speed)
            acceleration = _limit((desired_velocity-tips[k, 3:])/o.positioner_time_constant_s,
                                  o.extension_acceleration_m_s2)
            observed_tip_velocity = state.velocity_eci_m_s+np.cross(wi, ri@(mounts[k]+tips[k, :3]))+ri@tips[k, 3:]
            closing = -float((observed_tip_velocity-target.velocity_eci_m_s-
                np.cross(wt, observed_tip-target.position_eci_m))@normal)
            braking_distance = max(0., closing**2-o.touch_speed_m_s**2)/(2*o.extension_acceleration_m_s2)
            braking_distance += max(0., closing-o.touch_speed_m_s)*o.positioner_time_constant_s
            if closing > o.touch_speed_m_s and gap_est-slow_distance <= braking_distance:
                outward = ri.T@normal
                acceleration += outward*(o.extension_acceleration_m_s2-acceleration@outward)
            extension_acceleration[k] = _limit(acceleration, o.extension_acceleration_m_s2)
        return observation
