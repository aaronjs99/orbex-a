"""Acquisition and delayed packet delivery for nonlinear execution.

Queues belong here; navigation owns its estimator and the numerical plant owns
truth. Their dependencies enter explicitly at acquisition/delivery boundaries.
"""
from __future__ import annotations


class MissionSensorDelivery:
    def __init__(self,initial_packets=()):
        self.pending=list(initial_packets)
        self.delivered=[]

    def sample(self,sensors,target,clock,relative_states):
        q=target.current_q
        states={key:clock.to_inertial(value,q,target.inertial_state_at(float(q)))[0]
                for key,value in relative_states.items()}
        packets=sensors.sample(q=q,physical_time_s=target.physical_time_at(float(q)),
            relative_states=relative_states,target_rotation=target.rotation_matrix_at(float(q)),
            target_angular_velocity=target.body_angular_velocity_at(float(q)),chaser_inertial_states=states)
        self.pending.extend(packets)

    def consume_ready(self,navigation):
        ready=[packet for packet in self.pending
               if packet.delivery_time_s.upper[0]<=navigation.time.lower[0]]
        estimate=navigation.consume(ready)
        identifiers={packet.packet_id for packet in ready}
        self.pending=[packet for packet in self.pending if packet.packet_id not in identifiers]
        self.delivered.extend(ready)
        return estimate

    def drain(self):
        packets=tuple(self.delivered)
        self.delivered=[]
        return packets
