"""Immutable processed sensor declarations, packets and navigation products."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
import math

from orbexa.core.dynamics.reference_frame import exact, bounds, fingerprint
from orbexa.control.certification.reachability import IntervalBox
from orbexa.control.certification.nonlinear import FRAME
from orbexa.estimation.navigation.noise import PhysicalSlotNoise
from orbexa.estimation.navigation.boxes import midpoint


@dataclass(frozen=True)
class StudySensorProfile:
    range_noise_m: float = 0.0005
    range_noise_relative: float = 20e-6
    bearing_component_noise: float = 30e-6
    range_bias_m: float = 0.0005
    target_rotation_operator_error: float = 1e-3
    target_rate_error_rad_s: float = 1e-4
    max_latency_s: float = 0.05
    dropout_probability: float = 0.0
    acquisition_period_s: float = 0.0
    minimum_range_m: float = 0.01
    maximum_range_m: float = 1000.0
    gnss_position_error_m: float = 1.0
    gnss_velocity_error_m_s: float = 0.005
    dropout_bursts_s: tuple = ()
    seed: int = 0
    declaration: str = 'Illustrative processed COM range/bearing and target-pose study products; no device qualification'
    # Sequential is the original replay policy. The keyed policy defines a
    # piecewise-constant common random field, not an observation schedule or
    # iid white noise; each packet still records its actual acquisition time.
    randomness_policy: str = 'sequential-v1'
    acquisition_epoch_s: float = 0.
    noise_time_quantum_s: float = 1e-9

    def __post_init__(self):
        for name in ('range_noise_m', 'range_noise_relative', 'bearing_component_noise', 'range_bias_m',
            'target_rotation_operator_error', 'target_rate_error_rad_s', 'max_latency_s',
            'dropout_probability', 'acquisition_period_s', 'minimum_range_m', 'maximum_range_m',
            'gnss_position_error_m', 'gnss_velocity_error_m_s'):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
            object.__setattr__(self, name, value)
        if (self.minimum_range_m <= 0 or self.maximum_range_m <= self.minimum_range_m
                or self.dropout_probability > 1 or self.range_noise_relative >= 1 or isinstance(self.seed, bool)
                or not isinstance(self.seed, int) or self.seed < 0 or not self.declaration.strip()):
            raise ValueError('Invalid bounded synthetic sensor profile')
        if self.randomness_policy not in ('sequential-v1', 'physical-slot-keyed-v1'):
            raise ValueError('Select a versioned sequential or physical-slot-keyed sensor randomness policy')
        for name in ('acquisition_epoch_s', 'noise_time_quantum_s'):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(name+' must be finite')
            object.__setattr__(self, name, value)
        if self.randomness_policy == 'physical-slot-keyed-v1':
            PhysicalSlotNoise(self.seed, self.acquisition_period_s, self.acquisition_epoch_s,
                              self.noise_time_quantum_s)
        elif self.acquisition_epoch_s != 0. or self.noise_time_quantum_s != 1e-9:
            raise ValueError('A noise epoch/quantum requires the physical-slot-keyed policy')
        object.__setattr__(self, 'dropout_bursts_s', tuple(bounds(pair) for pair in self.dropout_bursts_s))

    def as_dict(self):
        record = dict(self.__dict__, scope='Processed synthetic products; no raw image/point-cloud or feature-association claim')
        if self.randomness_policy == 'sequential-v1':
            # Old profiles, frozen packet identities and cohort fingerprints
            # retain exactly their original serialization.
            for name in ('randomness_policy', 'acquisition_epoch_s', 'noise_time_quantum_s'):
                record.pop(name)
        record['dropout_bursts_s'] = tuple(tuple(map(str, pair)) for pair in self.dropout_bursts_s)
        return record

    @property
    def fingerprint(self):
        return fingerprint(self.as_dict())

@dataclass(frozen=True)
class SensorPacket:
    sequence: int
    agent_id: str
    sensor_id: str
    acquired_q: tuple
    acquisition_time_s: IntervalBox
    delivery_time_s: IntervalBox
    measurement_kind: str
    value: tuple
    deterministic_error_box: IntervalBox
    frame_id: str
    sensor_model_fingerprint: str
    quality_status: str = 'VALID'
    orientation_operator_error: object = 0

    def __post_init__(self):
        object.__setattr__(self, 'acquired_q', bounds(self.acquired_q))
        object.__setattr__(self, 'value', tuple(map(exact, self.value)))
        object.__setattr__(self, 'orientation_operator_error', exact(self.orientation_operator_error))
        if (self.acquisition_time_s.dimension != 1 or self.delivery_time_s.dimension != 1
                or self.delivery_time_s.lower[0] < self.acquisition_time_s.lower[0]
                or self.deterministic_error_box.dimension != len(self.value)
                or self.orientation_operator_error < 0 or not self.sensor_model_fingerprint):
            raise ValueError('Invalid sensor packet support or event ordering')

    @property
    def packet_id(self):
        return fingerprint(self.as_dict())

    def as_dict(self):
        return dict(sequence=self.sequence, agent_id=self.agent_id, sensor_id=self.sensor_id,
            acquired_q=tuple(map(str, self.acquired_q)), acquisition_time_s=self.acquisition_time_s.as_dict(),
            delivery_time_s=self.delivery_time_s.as_dict(), measurement_kind=self.measurement_kind,
            value=tuple(map(str, self.value)), deterministic_error_box=self.deterministic_error_box.as_dict(),
            frame_id=self.frame_id, sensor_model_fingerprint=self.sensor_model_fingerprint,
            quality_status=self.quality_status, orientation_operator_error=str(self.orientation_operator_error))

@dataclass(frozen=True)
class NavigationEstimate:
    exact_event_q: object
    physical_time_box: IntervalBox
    relative_state_box_by_agent: object
    target_ephemeris_box: IntervalBox
    target_orientation_matrix: tuple
    target_orientation_operator_error: object
    target_angular_velocity_box: IntervalBox
    packet_ids: tuple
    clock_fingerprint: str
    model_fingerprint: str
    sensor_fingerprints: tuple
    status: str
    reason: str
    frame_id: str = FRAME

    def __post_init__(self):
        object.__setattr__(self, 'relative_state_box_by_agent',
                           MappingProxyType(dict(self.relative_state_box_by_agent)))

    @property
    def point_relative_by_agent(self):
        return MappingProxyType({key: midpoint(box) for key, box in self.relative_state_box_by_agent.items()})
