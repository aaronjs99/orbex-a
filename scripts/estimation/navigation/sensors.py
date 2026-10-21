"""Truth-side bounded sensor sampling with versioned draw ordering."""
from __future__ import annotations

import math
import numpy as np

from orbexa.core.dynamics.reference_frame import exact
from orbexa.control.certification.reachability import IntervalBox
from orbexa.control.certification.nonlinear import FRAME
from orbexa.estimation.navigation.noise import PhysicalSlotNoise
from orbexa.estimation.navigation.records import StudySensorProfile, SensorPacket


def _reported_additive_support(reported,declared_error):
    """Outward additive support using reported binary64 values only.

    Hardware noise/bias declarations and draws are unchanged. Exact rational
    addition retains two reported-value ULPs for numerical measurement
    evaluation, including endpoint addition rounding.
    """
    return tuple(exact(error)+2*abs(exact(np.spacing(float(value))))
        for value,error in zip(reported,declared_error))

class SensorSuite:
    """Truth-side packet generator; never inject this object into a planner."""
    def __init__(self, profile=None):
        self.profile = profile or StudySensorProfile()
        self.rng = np.random.default_rng(self.profile.seed)
        self.noise = (PhysicalSlotNoise(self.profile.seed, self.profile.acquisition_period_s,
            self.profile.acquisition_epoch_s, self.profile.noise_time_quantum_s)
            if self.profile.randomness_policy == 'physical-slot-keyed-v1' else None)
        self.sequence, self.biases, self.last_acquisition_s = 0, {}, None

    def _uniform(self, slot, agent, channel, low=0., high=1., size=None):
        if self.noise is None:
            return self.rng.uniform(low, high, size)
        return self.noise.uniform(slot, agent, channel, low, high, size)

    def _random(self, slot, agent, channel):
        return self.rng.random() if self.noise is None else self.noise.uniform(slot, agent, channel)

    def sample(self, *, q, physical_time_s, relative_states, target_rotation, target_angular_velocity,
               chaser_inertial_states=None):
        profile, time = self.profile, float(physical_time_s)
        if not math.isfinite(time):
            raise ValueError('Acquisition time must be finite')
        if self.last_acquisition_s is not None and time < self.last_acquisition_s:
            raise ValueError('Sensor sampling clock cannot run backward')
        if self.last_acquisition_s is not None and time-self.last_acquisition_s < profile.acquisition_period_s:
            return ()
        self.last_acquisition_s = time
        if any(float(lo) <= time <= float(hi) for lo, hi in profile.dropout_bursts_s):
            return ()
        rotation, omega = np.asarray(target_rotation, dtype=float), np.asarray(target_angular_velocity, dtype=float)
        if (rotation.shape != (3, 3) or omega.shape != (3,) or not np.all(np.isfinite(rotation))
                or not np.all(np.isfinite(omega))):
            raise ValueError('Truth pose/rate must have finite declared shapes')
        slot = None if self.noise is None else self.noise.slot_at(time)
        packets = []
        def emit(agent, kind, value, radius, orientation_error=0, frame=FRAME):
            self.sequence += 1
            latency = float(self._uniform(slot, agent, kind+'.latency', 0, profile.max_latency_s))
            packets.append(SensorPacket(self.sequence, str(agent), 'synthetic-'+kind, (q, q),
                IntervalBox.point((time,)), IntervalBox.point((time+latency,)), kind, tuple(value),
                IntervalBox.from_center_radius((0,)*len(value), radius), frame,
                profile.fingerprint, orientation_operator_error=orientation_error))
        for agent, relative in relative_states.items():
            state = np.asarray(relative, dtype=float)
            if state.shape != (6,) or not np.all(np.isfinite(state)):
                raise ValueError('Truth relative states require six finite components')
            if chaser_inertial_states is not None and agent in chaser_inertial_states:
                inertial = np.asarray(chaser_inertial_states[agent], dtype=float)
                if inertial.shape != (6,) or not np.all(np.isfinite(inertial)):
                    raise ValueError('Chaser GNSS truth has six finite inertial SI components')
                gnss_error = np.array((profile.gnss_position_error_m,)*3+(profile.gnss_velocity_error_m_s,)*3)
                measured_gnss=inertial+self._uniform(slot, agent, 'chaser_gnss.noise', -gnss_error, gnss_error)
                emit(agent, 'chaser_gnss', measured_gnss,
                     _reported_additive_support(measured_gnss,gnss_error),frame='inertial-cartesian-si-v1')
            distance = float(np.linalg.norm(state[:3]))
            if (distance < profile.minimum_range_m or distance > profile.maximum_range_m
                    or self._random(slot, agent, 'range_bearing.dropout') < profile.dropout_probability):
                continue
            bias = self.biases.setdefault(str(agent), float(self._uniform(None, agent,
                'range_bearing.bias', -profile.range_bias_m, profile.range_bias_m)))
            range_error = profile.range_noise_m+profile.range_noise_relative*distance
            value = [distance+bias+self._uniform(slot, agent, 'range_bearing.range_noise', -range_error, range_error)]
            value.extend(state[:3]/distance+self._uniform(slot, agent, 'range_bearing.bearing_noise', -profile.bearing_component_noise,
                                                           profile.bearing_component_noise, 3))
            # Do not normalize noisy bearings: the component support is exact.
            # Calibrate the support from the reported range, not the hidden
            # truth range (which would leak truth through packet metadata).
            rounding = exact(1e-12+8*abs(np.spacing(value[0])))
            floor, bias_bound, proportional = map(exact, (profile.range_noise_m,
                profile.range_bias_m, profile.range_noise_relative))
            radius_upper = (exact(value[0])+floor+bias_bound+rounding)/(1-proportional)
            declared_error = floor+bias_bound+proportional*radius_upper+rounding
            emit(agent, 'range_bearing', value,
                (declared_error,)+(profile.bearing_component_noise+1e-12,)*3)
        if self._random(slot, 'target', 'target_pose.dropout') >= profile.dropout_probability:
            angle = self._uniform(slot, 'target', 'target_pose.angle', -profile.target_rotation_operator_error, profile.target_rotation_operator_error)
            c, s = math.cos(angle), math.sin(angle)
            perturbation = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
            measured_rotation = perturbation@rotation
            measured_omega = omega+self._uniform(slot, 'target', 'target_pose.rate_noise', -profile.target_rate_error_rad_s, profile.target_rate_error_rad_s, 3)
            rate_support=_reported_additive_support(measured_omega,(profile.target_rate_error_rad_s,)*3)
            # Float matrix evaluation needs its own small, explicit study allowance.
            error = profile.target_rotation_operator_error+1e-12
            emit('target', 'target_pose', tuple(measured_rotation.ravel())+tuple(measured_omega),
                 (error,)*9+rate_support,error)
        return tuple(packets)
