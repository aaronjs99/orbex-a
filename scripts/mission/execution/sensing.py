"""sensing.py: Acquisition and delayed packet delivery for nonlinear execution.

Queues belong here; navigation owns its estimator and the numerical plant owns
truth. Their dependencies enter explicitly at acquisition/delivery boundaries.

Copyright: Aaron John Sabu
"""

from __future__ import annotations

from orbexa.models.dynamics.reference_frame import exact


class SensorAcquisitionArchiveError(RuntimeError):
    """A measured batch exists but its audit write is not acknowledged."""

    def __init__(self, event, reason):
        """Build the message from `reason` and keep the unconfirmed audit event."""
        super().__init__("Sensor audit publication failed: " + str(reason))
        self.event = event


class SensorAcquisitionAudit:
    """Stream measured products before consumption; retain only one batch.

    A missing sink is explicitly unarchived. A failed supplied sink blocks
    further sampling and consumption, keeping the unconfirmed batch available
    to the existing execution-failure recorder. No truth state is published.
    """

    def __init__(
        self,
        sensors,
        *,
        clock_fingerprint,
        model_fingerprint,
        truth_model_fingerprint,
        sink=None,
    ):
        """Store the sensors, the sink and the clock and model fingerprints.

        Inputs:
        - sink: callable receiving each audit event, or None for an unarchived audit.

        Outputs:
        - Raises TypeError if the sink is neither callable nor None.
        """
        if sink is not None and not callable(sink):
            raise TypeError("Sensor audit sink must be callable or None")
        self.sensors, self.sink = sensors, sink
        self.context = {
            "clock_fingerprint": str(clock_fingerprint),
            "model_fingerprint": str(model_fingerprint),
            "truth_model_fingerprint": str(truth_model_fingerprint),
        }
        self.last_event = self.unconfirmed_event = None
        self.unarchived_events = 0

    def require_confirmed(self):
        """Raise SensorAcquisitionArchiveError if the last write is unconfirmed."""
        if self.unconfirmed_event is not None:
            raise SensorAcquisitionArchiveError(
                self.unconfirmed_event, "previous write remains unconfirmed"
            )

    def _publish(self, event):
        """Record an audit event and write it to the sink.

        It first requires the previous write to be confirmed. Without a sink the event
        is
        counted as unarchived. With a sink the event stays unconfirmed until the sink
        returns; a sink exception raises SensorAcquisitionArchiveError.
        """
        self.require_confirmed()
        self.last_event = event
        if self.sink is None:
            self.unarchived_events += 1
            return
        self.unconfirmed_event = event
        try:
            self.sink(event)
        except Exception as error:
            raise SensorAcquisitionArchiveError(event, error) from error
        self.unconfirmed_event = None

    def acquired(self, packets, *, q=None, time_s=None, origin="GENERATED"):
        """Publish a SENSOR_ACQUISITION audit event for a batch of measured packets.

        The event lists each packet with its noise slot, the sensor profile and the
        noise
        field declaration.

        Inputs:
        - packets: measured packets of the batch.
        - q, time_s: exact anomaly and physical time (s) of the acquisition; unused for
          an INITIAL_SUPPLIED origin.
        - origin: label of where the batch came from.
        """
        profile = self.sensors.profile
        field = {
            "policy": profile.randomness_policy,
            "profile": profile.as_dict(),
            "physical_slot": None,
            "scope": "Noise field declaration, not an enforced observation schedule",
        }
        if self.sensors.noise is not None and time_s is not None:
            field["physical_slot"] = self.sensors.noise.slot_at(time_s)
        rows = []
        for packet in packets:
            low, high = (
                packet.acquisition_time_s.lower[0],
                packet.acquisition_time_s.upper[0],
            )
            slot = (
                self.sensors.noise.slot_at(float(low))
                if self.sensors.noise is not None and low == high
                else None
            )
            rows.append(
                {
                    "packet_id": packet.packet_id,
                    "packet": packet.as_dict(),
                    "physical_slot": slot,
                }
            )
        self._publish(
            dict(
                archive_event_kind="SENSOR_ACQUISITION",
                schema="orbexa.sensor-acquisition-audit.v1",
                origin=origin,
                acquisition_event=(
                    None
                    if origin == "INITIAL_SUPPLIED"
                    else {
                        "exact_q": str(exact(q)),
                        "physical_time_s": str(exact(time_s)),
                    }
                ),
                packet_count=len(rows),
                packets=rows,
                noise_field=field,
                sensor_profile_id=profile.fingerprint,
                **self.context,
            )
        )

    def delivered(self, packets, navigation, *, estimate=None):
        """Publish a delivery audit event for packets offered to navigation.

        Without `estimate` the event kind is SENSOR_DELIVERY_ATTEMPT; with it the event
        is SENSOR_DELIVERY_RESULT and adds the estimate status, the accepted packet ids
        within the offered batch, and the reason.
        """
        record = dict(
            archive_event_kind=(
                "SENSOR_DELIVERY_ATTEMPT"
                if estimate is None
                else "SENSOR_DELIVERY_RESULT"
            ),
            schema="orbexa.sensor-delivery-audit.v1",
            packet_ids=[packet.packet_id for packet in packets],
            exact_event_q=(
                None if not hasattr(navigation, "q") else str(exact(navigation.q))
            ),
            event_time_s=navigation.time.as_dict(),
            **self.context,
        )
        if estimate is not None:
            offered = {packet.packet_id for packet in packets}
            record.update(
                status=getattr(estimate, "status", "UNAVAILABLE"),
                accepted_source_packet_ids=[
                    identity
                    for identity in getattr(estimate, "packet_ids", ())
                    if identity in offered
                ],
                accepted_scope="OFFERED_BATCH_ONLY",
                reason=getattr(estimate, "reason", None),
            )
        self._publish(record)


class MissionSensorDelivery:
    """Hold pending and delivered sensor packets and drive their audited delivery.

    Packets are sampled from truth into `pending` and handed to navigation once their
    delivery time has passed.
    """

    def __init__(self, initial_packets=(), *, acquisition_audit=None):
        """Store the initial packets as pending, with an optional acquisition audit."""
        self.pending = list(initial_packets)
        self.delivered = []
        self.acquisition_audit = acquisition_audit
        self._initial_recorded = not bool(self.pending)

    def record_initial(self, q=None):
        """Audit the initially supplied packets once, if there were any.

        The `q` argument is not used.
        """
        if not self._initial_recorded:
            if self.acquisition_audit is not None:
                self.acquisition_audit.acquired(self.pending, origin="INITIAL_SUPPLIED")
            self._initial_recorded = True

    def sample(self, sensors, target, clock, relative_states):
        """Sample the sensors at the target's current anomaly and queue the packets.

        It requires the previous audit write to be confirmed and records the initial
        packets first. The chaser inertial states are derived from the relative states
        through the clock. New packets are appended to `pending`; an acquisition event
        is
        audited only if the sensors' last acquisition time changed.
        """
        if self.acquisition_audit is not None:
            self.acquisition_audit.require_confirmed()
        self.record_initial()
        q = target.current_q
        previous = sensors.last_acquisition_s
        states = {
            key: clock.to_inertial(value, q, target.inertial_state_at(float(q)))[0]
            for key, value in relative_states.items()
        }
        packets = sensors.sample(
            q=q,
            physical_time_s=target.physical_time_at(float(q)),
            relative_states=relative_states,
            target_rotation=target.rotation_matrix_at(float(q)),
            target_angular_velocity=target.body_angular_velocity_at(float(q)),
            chaser_inertial_states=states,
        )
        self.pending.extend(packets)
        if (
            self.acquisition_audit is not None
            and sensors.last_acquisition_s != previous
        ):
            self.acquisition_audit.acquired(
                packets, q=q, time_s=sensors.last_acquisition_s
            )

    def consume_ready(self, navigation):
        """Deliver the packets whose delivery time has passed to navigation.

        A packet is ready if its delivery time upper bound is at most the lower bound of
        the navigation time. Ready packets are audited as an attempt, consumed, and
        audited again with the estimate. They are then removed from `pending` and
        appended to `delivered`. Returns the navigation estimate.
        """
        if self.acquisition_audit is not None:
            self.acquisition_audit.require_confirmed()
        self.record_initial()
        ready = [
            packet
            for packet in self.pending
            if packet.delivery_time_s.upper[0] <= navigation.time.lower[0]
        ]
        if ready and self.acquisition_audit is not None:
            self.acquisition_audit.delivered(ready, navigation)
        estimate = navigation.consume(ready)
        if ready and self.acquisition_audit is not None:
            self.acquisition_audit.delivered(ready, navigation, estimate=estimate)
        identifiers = {packet.packet_id for packet in ready}
        self.pending = [
            packet for packet in self.pending if packet.packet_id not in identifiers
        ]
        self.delivered.extend(ready)
        return estimate

    def drain(self):
        """Return the delivered packets as a tuple and clear the delivered list."""
        packets = tuple(self.delivered)
        self.delivered = []
        return packets
