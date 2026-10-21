"""Bounded delayed sensing using the installed feedback proof only.

This observer belongs to the independent executor.  It never calls the
nonlinear integrator or the planner.  A held input and the installed proof's
uniform acceleration enclosure give an elementary integral prediction.
Processed relative range/bearing packets constrain their actual acquisition
event; delayed constraints are carried through those same held integrals.
The last accepted acquisition, separately for every member, owns sensing age.

The packet producer is a trusted sensing boundary.  Its declared deterministic
support and anomaly/physical-time stamps must enclose the physical measurement.
No image processing, packet authentication or physical timing qualification is
inferred by accepting these processed products.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from fractions import Fraction
import uuid

from orbexa.control.certification.reachability import IntervalBox, _exact
from orbexa.control.fallback.refuge import RefugeObservation
from orbexa.control.fallback.records import HeldFleetDecision
from orbexa.control.fallback.plan import FleetFallbackPlan
from orbexa.estimation.navigation.records import SensorPacket
from orbexa.estimation.navigation.boxes import intersect
from orbexa.simulation.nonlinear.safety.drift import ZERO_NOMINAL_ID


def _bounded_box(box):
    """Outward dyadic enclosure bounds arithmetic size, without narrowing.

    Repeated position-difference divisions otherwise accumulate unrelated
    rational denominators.  Each exported endpoint is an exact outward bound;
    the installed measurement-error gate still checks its complete width.
    """
    scale=1 << 80
    lower,upper=[],[]
    for lo,hi in zip(box.lower,box.upper):
        a,b=lo*scale,hi*scale
        lower.append(Fraction(a.numerator//a.denominator,scale))
        upper.append(Fraction(-((-b.numerator)//b.denominator),scale))
    return IntervalBox(tuple(lower),tuple(upper))


def _propagate(box, acceleration, width):
    h = _exact(width)
    return _bounded_box(IntervalBox(
        tuple(p+h*v+h*h*a/2 for p,v,a in zip(box.lower[:3],box.lower[3:],acceleration.lower))+
            tuple(v+h*a for v,a in zip(box.lower[3:],acceleration.lower)),
        tuple(p+h*v+h*h*a/2 for p,v,a in zip(box.upper[:3],box.upper[3:],acceleration.upper))+
            tuple(v+h*a for v,a in zip(box.upper[3:],acceleration.upper))))


def _position(packet):
    if len(packet.value) != 4 or packet.deterministic_error_box.dimension != 4:
        raise ValueError('A processed relative range/bearing packet needs four components')
    support = packet.deterministic_error_box
    true = tuple((v-hi,v-lo) for v,lo,hi in zip(packet.value,support.lower,support.upper))
    if true[0][0] <= 0:
        raise ValueError('Relative range support must be proved positive')
    products = [tuple(r*b for r in true[0] for b in direction) for direction in true[1:]]
    return _bounded_box(IntervalBox(tuple(map(min,products)),tuple(map(max,products))))


@dataclass(frozen=True)
class SampledHoldPrediction:
    """Pre-dispatch integral prediction; commit only after actual actuation."""
    owner_id: str
    generation: int
    decision_id: str
    fallback_id: str
    start: Fraction
    end: Fraction
    initial_boxes: tuple
    endpoint_boxes: tuple
    accelerations: tuple
    initial_time: IntervalBox
    endpoint_time: IntervalBox


class OnlineRelativeObserver:
    """Finite memory, delayed packet replay and per-agent sensing deadlines.

    ``prepare_hold`` performs every prospective age/error-budget check before
    dispatch.  ``commit_hold`` advances the observation clock after the adapter
    has applied that exact hold.  Inconsistent returned sensing is an actual
    assumption violation; it cannot silently revert a physical clock.
    """
    def __init__(self, observations, *, clock, physical_time_box,
                 maximum_blackout_q, sensor_fingerprints, history_length=512,
                 event_sink=None):
        if (not observations or any(not isinstance(v,RefugeObservation) for v in observations.values())
                or any(v.q != next(iter(observations.values())).q for v in observations.values())):
            raise ValueError('Declare a complete simultaneous initial relative observation fleet')
        if (not isinstance(physical_time_box,IntervalBox) or physical_time_box.dimension != 1
                or isinstance(history_length,bool) or not isinstance(history_length,int) or history_length < 2):
            raise ValueError('Declare a physical-time support and at least two bounded history holds')
        allowed = frozenset(sensor_fingerprints)
        if not allowed or any(not isinstance(value,str) or not value for value in allowed):
            raise ValueError('Bind the observer to explicit processed sensor model identities')
        if event_sink is not None and not callable(event_sink):
            raise TypeError('event_sink must be callable or None')
        self.clock, self.time = clock, physical_time_box
        self.q = next(iter(observations.values())).q
        self.maximum_blackout_q = _exact(maximum_blackout_q)
        if self.maximum_blackout_q <= 0:
            raise ValueError('The original sensing horizon must be finite and positive')
        self.ids = tuple(observations)
        self.context = {key:(value.sensor_id,value.frame,value.clock_id) for key,value in observations.items()}
        if any(value[2] != clock.fingerprint for value in self.context.values()):
            raise ValueError('Observation and commanded reference clock identities differ')
        self.boxes = {key:value.state_enclosure for key,value in observations.items()}
        self.last_sensor_q = {key:self.q for key in self.ids}
        self.allowed_sensor_ids, self.event_sink = allowed,event_sink
        self.history = deque(maxlen=history_length)
        self._holds_by_start = {}
        zero = ((Fraction(0),)*3,)*4
        self._moments = {self.q:{key:zero for key in self.ids}}
        self.positions = {key:deque(((self.q,self.boxes[key].slice(0,3)),),maxlen=history_length)
                          for key in self.ids}
        self._accepted = deque(maxlen=history_length*len(self.ids)*3)
        self._accepted_set = set()
        self._owner_id,self._generation = uuid.uuid4().hex,0
        self.failed = None

    def _emit(self, record):
        if self.event_sink is not None:
            self.event_sink(record)

    def observations(self):
        if self.failed is not None:
            raise ValueError('An inconsistent online observation boundary needs explicit recovery')
        return {key:RefugeObservation(self.q,tuple((lo+hi)/2 for lo,hi in zip(box.lower,box.upper)),
                    box,*self.context[key]) for key,box in self.boxes.items()}

    def prepare_hold(self, decision, plan):
        """Check the complete next hold and produce an immutable prediction."""
        if self.failed is not None:
            raise ValueError('An inconsistent online observation boundary needs explicit recovery')
        if (not isinstance(decision,HeldFleetDecision) or not isinstance(plan,FleetFallbackPlan)
                or decision.source not in ('FALLBACK_TRACK','FALLBACK_HOLD')
                or decision.start != self.q or decision.active_fallback_id != plan.fingerprint
                or tuple(plan.ids) != self.ids or set(dict(decision.applied_inputs)) != set(self.ids)):
            raise ValueError('Integral sensing prediction requires the exact installed whole-fleet feedback hold')
        end = decision.start+decision.hold_width
        expired = [key for key in self.ids if end-self.last_sensor_q[key] > self.maximum_blackout_q]
        if expired:
            raise ValueError('Unchanged sensing acquisition-age limit cannot cover the next hold: '+', '.join(expired))
        controllers, applied = dict(plan.segment_at(self.q,decision.hold_width).controllers),dict(decision.applied_inputs)
        acceleration, endpoints = {},{}
        for key in self.ids:
            controller = controllers[key]
            spec = controller.certificate.spec
            if spec.context.nominal_model_id != ZERO_NOMINAL_ID:
                raise ValueError('Integral sensing requires the certified actual zero-nominal drift envelope')
            drift = controller.certificate.drift_bounds.mismatch_abs
            acceleration[key] = IntervalBox(tuple(v-w for v,w in zip(applied[key].lower,drift)),
                tuple(v+w for v,w in zip(applied[key].upper,drift)))
            endpoints[key] = _propagate(self.boxes[key],acceleration[key],decision.hold_width)
            sensor,frame,clock = self.context[key]
            if (spec.context.sensor_id,spec.context.frame,spec.context.clock_id) != (sensor,frame,clock):
                raise ValueError('The installed feedback changed the physical observation context')
            next_controller = dict(plan.segment_at(end).controllers)[key]
            box = endpoints[key]
            next_controller.observation_box(RefugeObservation(end,
                tuple((lo+hi)/2 for lo,hi in zip(box.lower,box.upper)),box,sensor,frame,clock),end)
        rate = self.clock.interval_bounds((self.q,end))['rate_s_inv']
        if rate.lower[0] <= 0:
            raise ValueError('Physical-time prediction needs a proved positive reference clock')
        h = decision.hold_width
        endpoint_time = _bounded_box(IntervalBox((self.time.lower[0]+h/rate.upper[0],),
                                   (self.time.upper[0]+h/rate.lower[0],)))
        return SampledHoldPrediction(self._owner_id,self._generation,decision.fingerprint,plan.fingerprint,
            self.q,end,tuple(self.boxes.items()),tuple(endpoints.items()),tuple(acceleration.items()),
            self.time,endpoint_time)

    def _at(self, key, q):
        if q == self.q:
            return self.boxes[key]
        node = self._holds_by_start.get(q)
        if node is not None:
            return dict(node.initial_boxes)[key]
        for hold in self.history:
            if hold.start <= q < hold.end:
                return _propagate(dict(hold.initial_boxes)[key],dict(hold.accelerations)[key],q-hold.start)
        return None

    def _moment_at(self,key,q):
        node=self._moments.get(q)
        if node is not None:
            return node[key]
        for hold in self.history:
            if hold.start < q < hold.end:
                previous=self._moments[hold.start][key]
                acceleration=dict(hold.accelerations)[key]
                h,weight=q-hold.start,(q*q-hold.start*hold.start)/2
                return tuple(tuple(p+weight*v for p,v in zip(row,coefficient))
                    for row,coefficient,weight in ((previous[0],acceleration.lower,h),
                        (previous[1],acceleration.lower,weight),(previous[2],acceleration.upper,h),
                        (previous[3],acceleration.upper,weight)))
        return None

    def _velocity(self, key, acquired, position):
        earliest = self.history[0].start if self.history else self.q
        samples=list(self.positions[key])
        indices={0}
        span=1
        while span <= len(samples):
            indices.add(len(samples)-span)
            span*=2
        result=None
        last=self._moment_at(key,acquired)
        for index in sorted(indices):
            q0,p0=samples[index]
            if not earliest <= q0 < acquired:
                continue
            first=self._moment_at(key,q0)
            if first is None or last is None:
                continue
            # Separate exact scalar moments of lower/upper coefficients
            # reproduce integral((s-q0)*a(s)) with nonnegative weights.
            lower=tuple(b-a-q0*(d-c) for a,b,c,d in zip(first[1],last[1],first[0],last[0]))
            upper=tuple(b-a-q0*(d-c) for a,b,c,d in zip(first[3],last[3],first[2],last[2]))
            h=acquired-q0
            bound=_bounded_box(IntervalBox(tuple((p-a+i)/h for p,a,i in zip(position.lower,p0.upper,lower)),
                tuple((p-a+i)/h for p,a,i in zip(position.upper,p0.lower,upper))))
            result=bound if result is None else intersect(result,bound)
            if result is None:
                raise ValueError('Linked position history contradicts the installed acceleration envelope')
        # A logarithmic set of genuine lookbacks balances position noise
        # against accumulated acceleration uncertainty.  Keeping only the
        # oldest observation can make an otherwise useful velocity bound
        # grow forever. Every intersected constraint separately contains truth.
        return result

    def _carry(self, key, acquired, box):
        cursor = acquired
        while cursor < self.q:
            hold=self._holds_by_start.get(cursor)
            if hold is None:
                hold=next((item for item in self.history if item.start < cursor < item.end),None)
            if hold is None:
                return None
            box=_propagate(box,dict(hold.accelerations)[key],hold.end-cursor)
            cursor=hold.end
        return box if cursor == self.q else None

    def consume(self, packets):
        """Narrow current support using actual delayed processed products only."""
        if self.failed is not None:
            raise ValueError('An inconsistent online observation boundary needs explicit recovery')
        events=[]
        for packet in packets:
            if not isinstance(packet,SensorPacket):
                raise TypeError('The sensing adapter must return typed processed SensorPacket products')
            event = dict(schema='orbexa.online-relative-sensing.v1',packet_id=packet.packet_id,
                agent_id=packet.agent_id,exact_event_q=str(self.q),acquired_q=tuple(map(str,packet.acquired_q)))
            if packet.packet_id in self._accepted_set:
                event['status']='DUPLICATE'
            elif packet.sensor_model_fingerprint not in self.allowed_sensor_ids:
                self.failed=dict(event,status='INCONSISTENT',reason='Processed sensor model identity changed')
                self._emit(self.failed)
                raise ValueError(self.failed['reason'])
            elif (packet.quality_status != 'VALID' or packet.acquired_q[0] != packet.acquired_q[1]
                    or packet.acquired_q[1] > self.q or packet.delivery_time_s.upper[0] > self.time.lower[0]):
                event['status']='NOT_READY_OR_UNSUPPORTED'
            elif (packet.measurement_kind != 'range_bearing' or packet.agent_id not in self.boxes
                    or packet.frame_id != self.context[packet.agent_id][1]):
                event['status']='PLANNING_ONLY_PRODUCT'
            else:
                key,acquired = packet.agent_id,packet.acquired_q[0]
                prior = self._at(key,acquired)
                if prior is None:
                    event['status']='OUTSIDE_FINITE_HISTORY'
                else:
                    try:
                        position = _position(packet)
                        velocity = self._velocity(key,acquired,position)
                    except (ValueError,TypeError) as error:
                        self.failed=dict(event,status='INCONSISTENT',reason=type(error).__name__+': '+str(error))
                        self._emit(self.failed)
                        raise ValueError(self.failed['reason']) from error
                    candidate = intersect(prior,IntervalBox(position.lower+prior.lower[3:],position.upper+prior.upper[3:]))
                    if candidate is not None and velocity is not None:
                        candidate = intersect(candidate,IntervalBox(candidate.lower[:3]+velocity.lower,
                            candidate.upper[:3]+velocity.upper))
                    acquired_position=None if candidate is None else candidate.slice(0,3)
                    if candidate is not None:
                        candidate = self._carry(key,acquired,candidate)
                    narrowed = None if candidate is None else intersect(self.boxes[key],candidate)
                    if narrowed is None:
                        self.failed=dict(event,status='INCONSISTENT',reason='Relative measurement contradicts retained integral prediction')
                        self._emit(self.failed)
                        raise ValueError(self.failed['reason'])
                    self.boxes[key]=narrowed
                    self.positions[key].append((acquired,acquired_position))
                    self.last_sensor_q[key]=max(self.last_sensor_q[key],acquired)
                    if len(self._accepted)==self._accepted.maxlen:
                        self._accepted_set.discard(self._accepted[0])
                    self._accepted.append(packet.packet_id)
                    self._accepted_set.add(packet.packet_id)
                    event.update(status='ACCEPTED',velocity_from_linked_positions=velocity is not None,
                        acquisition_age_q=str(self.q-acquired),last_sensor_q=str(self.last_sensor_q[key]))
            events.append(event)
            self._emit(event)
        return tuple(events)

    def commit_hold(self, prediction, *, packets=()):
        if (not isinstance(prediction,SampledHoldPrediction) or prediction.owner_id != self._owner_id
                or prediction.generation != self._generation or prediction.start != self.q
                or prediction.initial_boxes != tuple(self.boxes.items()) or prediction.initial_time != self.time):
            raise ValueError('Observation prediction is stale or belongs to another sensing boundary')
        previous=self._moments[prediction.start]
        h,weight=prediction.end-prediction.start,(prediction.end**2-prediction.start**2)/2
        accelerations=dict(prediction.accelerations)
        moments={key:tuple(tuple(p+weight*v for p,v in zip(row,coefficient))
            for row,coefficient,weight in ((previous[key][0],accelerations[key].lower,h),
                (previous[key][1],accelerations[key].lower,weight),(previous[key][2],accelerations[key].upper,h),
                (previous[key][3],accelerations[key].upper,weight))) for key in self.ids}
        if len(self.history)==self.history.maxlen:
            expired=self.history[0].start
            self._holds_by_start.pop(expired,None)
            self._moments.pop(expired,None)
        self.history.append(prediction)
        self._holds_by_start[prediction.start]=prediction
        self._moments[prediction.end]=moments
        self.boxes,self.q,self.time = dict(prediction.endpoint_boxes),prediction.end,prediction.endpoint_time
        self._generation += 1
        self.consume(packets)
        return self.observations()
