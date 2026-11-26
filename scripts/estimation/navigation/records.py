"""records.py: Immutable processed sensor declarations, packets and navigation products.

Copyright: Aaron John Sabu
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType

from orbexa.common.intervals import IntervalBox
from orbexa.estimation.navigation.boxes import midpoint
from orbexa.estimation.navigation.noise import PhysicalSlotNoise
from orbexa.models.dynamics.reference_frame import bounds, exact, fingerprint
from orbexa.proofs.nonlinear import FRAME


@dataclass(frozen=True)
class StudySensorProfile:
    """Frozen declaration of the synthetic processed-sensor study bounds.

    Names carry units: _m (m), _s (s), _rad_s (rad/s), _m_s (m/s). Bearing and
    rotation errors are dimensionless components or operator-norm bounds.
    """

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
    declaration: str = "Illustrative processed COM range/bearing and target-pose study products; no device qualification"
    # Sequential is the original replay policy. The keyed policy defines a
    # piecewise-constant common random field, not an observation schedule or
    # iid white noise; each packet still records its actual acquisition time.
    randomness_policy: str = "sequential-v1"
    acquisition_epoch_s: float = 0.0
    noise_time_quantum_s: float = 1e-9

    def __post_init__(self):
        """Coerce numeric fields to float, validate them and normalize dropout bursts.

        Raises ValueError when a bound is not finite and nonnegative, the range limits
        are not 0 < minimum < maximum, dropout_probability exceeds 1,
        range_noise_relative is at least 1, seed is not a nonnegative int,
        declaration is blank, randomness_policy is unknown, or a noise epoch/quantum
        is set without the physical-slot-keyed policy. The keyed policy also builds a
        PhysicalSlotNoise, which raises for an invalid period, epoch or quantum.
        """
        for name in (
            "range_noise_m",
            "range_noise_relative",
            "bearing_component_noise",
            "range_bias_m",
            "target_rotation_operator_error",
            "target_rate_error_rad_s",
            "max_latency_s",
            "dropout_probability",
            "acquisition_period_s",
            "minimum_range_m",
            "maximum_range_m",
            "gnss_position_error_m",
            "gnss_velocity_error_m_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        if (
            self.minimum_range_m <= 0
            or self.maximum_range_m <= self.minimum_range_m
            or self.dropout_probability > 1
            or self.range_noise_relative >= 1
            or isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
            or not self.declaration.strip()
        ):
            raise ValueError("Invalid bounded synthetic sensor profile")
        if self.randomness_policy not in ("sequential-v1", "physical-slot-keyed-v1"):
            raise ValueError(
                "Select a versioned sequential or physical-slot-keyed sensor randomness policy"
            )
        for name in ("acquisition_epoch_s", "noise_time_quantum_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(name + " must be finite")
            object.__setattr__(self, name, value)
        if self.randomness_policy == "physical-slot-keyed-v1":
            PhysicalSlotNoise(
                self.seed,
                self.acquisition_period_s,
                self.acquisition_epoch_s,
                self.noise_time_quantum_s,
            )
        elif self.acquisition_epoch_s != 0.0 or self.noise_time_quantum_s != 1e-9:
            raise ValueError(
                "A noise epoch/quantum requires the physical-slot-keyed policy"
            )
        object.__setattr__(
            self,
            "dropout_bursts_s",
            tuple(bounds(pair) for pair in self.dropout_bursts_s),
        )

    def as_dict(self):
        """Return the profile fields plus a scope statement as a dict.

        For the sequential-v1 policy the randomness_policy, acquisition_epoch_s and
        noise_time_quantum_s keys are removed. dropout_bursts_s is returned as nested
        tuples of strings.
        """
        record = dict(
            self.__dict__,
            scope="Processed synthetic products; no raw image/point-cloud or feature-association claim",
        )
        if self.randomness_policy == "sequential-v1":
            # Old profiles, frozen packet identities and cohort fingerprints
            # retain exactly their original serialization.
            for name in (
                "randomness_policy",
                "acquisition_epoch_s",
                "noise_time_quantum_s",
            ):
                record.pop(name)
        record["dropout_bursts_s"] = tuple(
            tuple(map(str, pair)) for pair in self.dropout_bursts_s
        )
        return record

    @property
    def fingerprint(self):
        """Return the fingerprint of as_dict()."""
        return fingerprint(self.as_dict())


@dataclass(frozen=True)
class SensorPacket:
    """Immutable processed measurement with its deterministic error support.

    acquisition_time_s and delivery_time_s are one-dimensional boxes in physical
    seconds. acquired_q is the unwrapped-anomaly interval of the acquisition event.
    deterministic_error_box bounds measurement minus true value componentwise.
    """

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
    quality_status: str = "VALID"
    orientation_operator_error: object = 0

    def __post_init__(self):
        """Normalize acquired_q, value and orientation_operator_error and validate them.

        Raises ValueError when a time box is not one-dimensional, the delivery lower
        bound precedes the acquisition lower bound, the error box dimension differs
        from len(value), orientation_operator_error is negative, or
        sensor_model_fingerprint is empty.
        """
        object.__setattr__(self, "acquired_q", bounds(self.acquired_q))
        object.__setattr__(self, "value", tuple(map(exact, self.value)))
        object.__setattr__(
            self, "orientation_operator_error", exact(self.orientation_operator_error)
        )
        if (
            self.acquisition_time_s.dimension != 1
            or self.delivery_time_s.dimension != 1
            or self.delivery_time_s.lower[0] < self.acquisition_time_s.lower[0]
            or self.deterministic_error_box.dimension != len(self.value)
            or self.orientation_operator_error < 0
            or not self.sensor_model_fingerprint
        ):
            raise ValueError("Invalid sensor packet support or event ordering")

    @property
    def packet_id(self):
        """Return the fingerprint of as_dict(), identifying the packet content."""
        return fingerprint(self.as_dict())

    def as_dict(self):
        """Return the packet fields as a dict; exact values and bounds are strings."""
        return {
            "sequence": self.sequence,
            "agent_id": self.agent_id,
            "sensor_id": self.sensor_id,
            "acquired_q": tuple(map(str, self.acquired_q)),
            "acquisition_time_s": self.acquisition_time_s.as_dict(),
            "delivery_time_s": self.delivery_time_s.as_dict(),
            "measurement_kind": self.measurement_kind,
            "value": tuple(map(str, self.value)),
            "deterministic_error_box": self.deterministic_error_box.as_dict(),
            "frame_id": self.frame_id,
            "sensor_model_fingerprint": self.sensor_model_fingerprint,
            "quality_status": self.quality_status,
            "orientation_operator_error": str(self.orientation_operator_error),
        }


@dataclass(frozen=True)
class NavigationEstimate:
    """Immutable navigation product: state boxes, target pose bounds and provenance.

    exact_event_q is the unwrapped-anomaly event and physical_time_box its
    physical-time enclosure (s). status and reason report the filter state.
    """

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
        """Replace relative_state_box_by_agent with a read-only copy of the mapping."""
        object.__setattr__(
            self,
            "relative_state_box_by_agent",
            MappingProxyType(dict(self.relative_state_box_by_agent)),
        )

    @property
    def point_relative_by_agent(self):
        """Return a read-only mapping of agent to its relative-state box midpoint.

        The midpoints are numerical representatives, not enclosures.
        """
        return MappingProxyType(
            {
                key: midpoint(box)
                for key, box in self.relative_state_box_by_agent.items()
            }
        )
