"""Acquisition and delayed packet delivery for nonlinear execution.

Queues belong here; navigation owns its estimator and the numerical plant owns
truth. Their dependencies enter explicitly at acquisition/delivery boundaries.
"""
from __future__ import annotations

from orbexa.core.reference_frame import exact


class SensorAcquisitionArchiveError(RuntimeError):
    """A measured batch exists but its audit write is not acknowledged."""

    def __init__(self, event, reason):
        super().__init__('Sensor audit publication failed: '+str(reason))
        self.event = event


class SensorAcquisitionAudit:
    """Stream measured products before consumption; retain only one batch.

    A missing sink is explicitly unarchived. A failed supplied sink blocks
    further sampling and consumption, keeping the unconfirmed batch available
    to the existing execution-failure recorder. No truth state is published.
    """

    def __init__(self, sensors, *, clock_fingerprint, model_fingerprint,
                 truth_model_fingerprint, sink=None):
        if sink is not None and not callable(sink):
            raise TypeError('Sensor audit sink must be callable or None')
        self.sensors, self.sink = sensors, sink
        self.context = dict(clock_fingerprint=str(clock_fingerprint),
            model_fingerprint=str(model_fingerprint), truth_model_fingerprint=str(truth_model_fingerprint))
        self.last_event = self.unconfirmed_event = None
        self.unarchived_events = 0

    def require_confirmed(self):
        if self.unconfirmed_event is not None:
            raise SensorAcquisitionArchiveError(self.unconfirmed_event, 'previous write remains unconfirmed')

    def _publish(self, event):
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

    def acquired(self, packets, *, q=None, time_s=None, origin='GENERATED'):
        profile = self.sensors.profile
        field = dict(policy=profile.randomness_policy, profile=profile.as_dict(),
            physical_slot=None, scope='Noise field declaration, not an enforced observation schedule')
        if self.sensors.noise is not None and time_s is not None:
            field['physical_slot'] = self.sensors.noise.slot_at(time_s)
        rows = []
        for packet in packets:
            low, high = packet.acquisition_time_s.lower[0], packet.acquisition_time_s.upper[0]
            slot = (self.sensors.noise.slot_at(float(low))
                    if self.sensors.noise is not None and low == high else None)
            rows.append(dict(packet_id=packet.packet_id, packet=packet.as_dict(), physical_slot=slot))
        self._publish(dict(archive_event_kind='SENSOR_ACQUISITION',
            schema='orbexa.sensor-acquisition-audit.v1', origin=origin,
            acquisition_event=(None if origin == 'INITIAL_SUPPLIED' else
                dict(exact_q=str(exact(q)), physical_time_s=str(exact(time_s)))),
            packet_count=len(rows), packets=rows, noise_field=field,
            sensor_profile_id=profile.fingerprint, **self.context))

    def delivered(self, packets, navigation, *, estimate=None):
        record = dict(archive_event_kind=('SENSOR_DELIVERY_ATTEMPT' if estimate is None else 'SENSOR_DELIVERY_RESULT'),
            schema='orbexa.sensor-delivery-audit.v1', packet_ids=[packet.packet_id for packet in packets],
            exact_event_q=(None if not hasattr(navigation, 'q') else str(exact(navigation.q))),
            event_time_s=navigation.time.as_dict(), **self.context)
        if estimate is not None:
            offered={packet.packet_id for packet in packets}
            record.update(status=getattr(estimate, 'status', 'UNAVAILABLE'),
                accepted_source_packet_ids=[identity for identity in getattr(estimate,'packet_ids',())
                    if identity in offered],accepted_scope='OFFERED_BATCH_ONLY',
                reason=getattr(estimate, 'reason', None))
        self._publish(record)


class MissionSensorDelivery:
    def __init__(self,initial_packets=(),*,acquisition_audit=None):
        self.pending=list(initial_packets)
        self.delivered=[]
        self.acquisition_audit=acquisition_audit
        self._initial_recorded=not bool(self.pending)

    def record_initial(self,q=None):
        if not self._initial_recorded:
            if self.acquisition_audit is not None:
                self.acquisition_audit.acquired(self.pending,origin='INITIAL_SUPPLIED')
            self._initial_recorded=True

    def sample(self,sensors,target,clock,relative_states):
        if self.acquisition_audit is not None:self.acquisition_audit.require_confirmed()
        self.record_initial()
        q=target.current_q
        previous=sensors.last_acquisition_s
        states={key:clock.to_inertial(value,q,target.inertial_state_at(float(q)))[0]
                for key,value in relative_states.items()}
        packets=sensors.sample(q=q,physical_time_s=target.physical_time_at(float(q)),
            relative_states=relative_states,target_rotation=target.rotation_matrix_at(float(q)),
            target_angular_velocity=target.body_angular_velocity_at(float(q)),chaser_inertial_states=states)
        self.pending.extend(packets)
        if self.acquisition_audit is not None and sensors.last_acquisition_s!=previous:
            self.acquisition_audit.acquired(packets,q=q,time_s=sensors.last_acquisition_s)

    def consume_ready(self,navigation):
        if self.acquisition_audit is not None:self.acquisition_audit.require_confirmed()
        self.record_initial()
        ready=[packet for packet in self.pending
               if packet.delivery_time_s.upper[0]<=navigation.time.lower[0]]
        if ready and self.acquisition_audit is not None:self.acquisition_audit.delivered(ready,navigation)
        estimate=navigation.consume(ready)
        if ready and self.acquisition_audit is not None:self.acquisition_audit.delivered(ready,navigation,estimate=estimate)
        identifiers={packet.packet_id for packet in ready}
        self.pending=[packet for packet in self.pending if packet.packet_id not in identifiers]
        self.delivered.extend(ready)
        return estimate

    def drain(self):
        packets=tuple(self.delivered)
        self.delivered=[]
        return packets
