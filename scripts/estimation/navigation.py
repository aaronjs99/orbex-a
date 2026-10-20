"""Synthetic bounded sensor packets and a separate set-navigation consumer.

Defaults are explicitly illustrative study parameters, not device performance.
The sampler alone accepts truth. The filter accepts priors, proved predictions,
and packets; velocity is inferred from process/observation history.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
import math
import numpy as np

from orbexa.core.reference_frame import exact, bounds, fingerprint
from orbexa.control.reachability import IntervalBox, HeldInputRequest, FlowCertificate, SweptSegment
from orbexa.control.nonlinear_flow import FRAME, MODEL
from orbexa.estimation.sensor_noise import PhysicalSlotNoise


def intersect(left, right):
    if left.dimension != right.dimension:
        raise ValueError('Cannot intersect unlike state dimensions')
    lo = tuple(max(a, b) for a, b in zip(left.lower, right.lower))
    hi = tuple(min(a, b) for a, b in zip(left.upper, right.upper))
    return None if any(a > b for a, b in zip(lo, hi)) else IntervalBox(lo, hi)


def midpoint(box):
    return tuple(float((a+b)/2) for a, b in zip(box.lower, box.upper))


class _AccelerationMomentLedger:
    """Bounded exact acceleration moments guarded by current source scalars.

    Lower and upper acceleration supports have separate scalar moments M0 and
    M1. Their query integral is delta(M1)-q0*delta(M0), exactly the old
    nonnegative segment weights. No interval subtraction or proof identity is
    cached. Unsupported types or incomplete topology defer to the old scan.
    """
    def __init__(self):
        self.owner, self.keys, self.sources, self.valid = None, (), (), False
        self._clear()

    def _clear(self):
        self.starts, self.ends, self.coefficients = [], [], []
        self.prefix = [((Fraction(0),)*3,)*4]
        self.counts, self.head, self.dead_holds = [], 0, 0

    @staticmethod
    def _key(hold, agent):
        if type(hold) is not dict or type(hold.get('flows')) is not dict:
            return None
        start, end = hold.get('start'), hold.get('end')
        if type(start) is not Fraction or type(end) is not Fraction:
            return None
        flow = hold['flows'].get(agent)
        if type(flow) is not FlowCertificate or type(flow.segments) is not tuple:
            return None
        segments = []
        for segment in flow.segments:
            if (type(segment) is not SweptSegment or type(segment.offset) is not tuple or len(segment.offset) != 2
                    or any(type(value) is not Fraction for value in segment.offset)
                    or type(segment.derivative) is not IntervalBox
                    or len(segment.derivative.lower) != 6 or len(segment.derivative.upper) != 6):
                return None
            lo, hi = segment.offset
            lower, upper = segment.derivative.lower[3:], segment.derivative.upper[3:]
            if any(type(value) is not Fraction for value in lower+upper):
                return None
            # Copy primitive integers: tuple aliases or object identities alone
            # miss forced same-object IntervalBox/Fraction mutations.
            scalars = tuple(part for value in lower+upper for part in (value.numerator, value.denominator))
            segments.append((id(segment), id(segment.derivative), lo.numerator, lo.denominator,
                             hi.numerator, hi.denominator, scalars))
        return (id(hold), start.numerator, start.denominator, end.numerator, end.denominator,
                id(flow), id(flow.segments), tuple(segments))

    def _append(self, key):
        scalars = key[1:5]+tuple(value for segment in key[7] for value in segment[2:6]+segment[6])
        # Fraction's usual normalization can itself be broken by forced
        # mutation. Such data keeps the original scan's arithmetic semantics.
        if (any(type(value) is not int for value in scalars)
                or any(d <= 0 or math.gcd(n, d) != 1 for n, d in zip(scalars[::2], scalars[1::2]))):
            return False
        start, end = Fraction(key[1], key[2]), Fraction(key[3], key[4])
        if end <= start or self.ends and start != self.ends[-1]:
            return False
        cursor = Fraction(0)
        for segment in key[7]:
            lo, hi = Fraction(segment[2], segment[3]), Fraction(segment[4], segment[5])
            if lo != cursor or hi <= lo:
                return False
            a, b = start+lo, start+hi
            values = segment[6]
            lower = tuple(Fraction(values[i], values[i+1]) for i in (0, 2, 4))
            upper = tuple(Fraction(values[i], values[i+1]) for i in (6, 8, 10))
            if any(lo > hi for lo, hi in zip(lower, upper)):
                return False
            dt, moment, previous = b-a, (b*b-a*a)/2, self.prefix[-1]
            self.prefix.append(tuple(tuple(p+weight*v for p, v in zip(row, bound))
                for row, bound, weight in ((previous[0], lower, dt), (previous[1], lower, moment),
                                          (previous[2], upper, dt), (previous[3], upper, moment))))
            self.starts.append(a)
            self.ends.append(b)
            self.coefficients.append((lower, upper))
            cursor = hi
        if start+cursor != end:
            return False
        self.counts.append(len(key[7]))
        return True

    def _rebuild(self, keys):
        self._clear()
        return all(self._append(key) for key in keys)

    def _prepare(self, history, agent):
        if type(history) is not deque or history.maxlen is None or not history:
            return False
        sources = tuple(history)
        try:
            keys = tuple(self._key(hold, agent) for hold in sources)
        except (KeyError, IndexError, AttributeError, TypeError, ValueError):
            return False
        if any(key is None for key in keys):
            return False
        if self.owner is history and keys == self.keys:
            return self.valid
        incremental = False
        if self.owner is history and self.valid:
            index = next((i for i, key in enumerate(self.keys) if key[0] == keys[0][0]), None)
            if index is not None:
                overlap = len(self.keys)-index
                if len(keys) >= overlap and self.keys[index:] == keys[:overlap]:
                    self.head += sum(self.counts[:index])
                    self.dead_holds += index
                    self.counts = self.counts[index:]
                    incremental = all(self._append(key) for key in keys[overlap:])
        self.valid = incremental or self._rebuild(keys)
        self.owner, self.keys, self.sources = history, keys, sources
        live = len(self.starts)-self.head
        if self.valid and (self.head >= max(1, live) or self.dead_holds >= history.maxlen):
            self.valid = self._rebuild(keys)
        return self.valid

    def _at(self, q):
        index = bisect_left(self.ends, q, self.head)
        if index >= len(self.ends) or q < self.starts[index]:
            return None
        if q == self.starts[index]:
            return self.prefix[index]
        if q == self.ends[index]:
            return self.prefix[index+1]
        dt, moment = q-self.starts[index], (q*q-self.starts[index]*self.starts[index])/2
        previous, (lower, upper) = self.prefix[index], self.coefficients[index]
        return tuple(tuple(p+weight*v for p, v in zip(row, bound))
            for row, bound, weight in ((previous[0], lower, dt), (previous[1], lower, moment),
                                      (previous[2], upper, dt), (previous[3], upper, moment)))

    def integral(self, history, agent, q0, q1):
        result = self.integrals(history, agent, ((q0, q1),))
        return result if result is NotImplemented else result[0]

    def integrals(self, history, agent, queries):
        """Check the actual source once for a finite synchronous query batch."""
        queries = tuple(queries)
        if (any(type(q) not in (Fraction, int) or type(q) is Fraction and
                    (type(q.numerator) is not int or type(q.denominator) is not int
                     or q.denominator <= 0 or math.gcd(q.numerator, q.denominator) != 1)
                for query in queries for q in query)
                or not self._prepare(history, agent)):
            return NotImplemented
        return tuple(self._integral_prepared(q0, q1) for q0, q1 in queries)

    def _integral_prepared(self, q0, q1):
        if not self.starts[self.head] <= q0 < q1 <= self.ends[-1]:
            return None
        first, last = self._at(q0), self._at(q1)
        if first is None or last is None:
            return None
        lower = tuple(b-a-q0*(d-c) for a, b, c, d in zip(first[1], last[1], first[0], last[0]))
        upper = tuple(b-a-q0*(d-c) for a, b, c, d in zip(first[3], last[3], first[2], last[2]))
        return lower, upper


def _product(a, b):
    values = (a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1])
    return min(values), max(values)


def _reported_additive_support(reported,declared_error):
    """Outward additive support using reported binary64 values only.

    Hardware noise/bias declarations and draws are unchanged. Exact rational
    addition retains two reported-value ULPs for numerical measurement
    evaluation, including endpoint addition rounding.
    """
    return tuple(exact(error)+2*abs(exact(np.spacing(float(value))))
        for value,error in zip(reported,declared_error))


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


class BoundedNavigationFilter:
    """Set prediction/intersection with no truth center or covariance shortcut."""
    def __init__(self, initial_relative_boxes, target_ephemeris_box, clock, model,
                 initial_q=0, initial_time_s=(0, 0), *, history_length=512,
                 initial_target_rotation=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                 initial_target_rotation_error=2, target_rate_bound_rad_s='0.1',
                 target_acceleration_bound_rad_s2='0.001', encloser=None, sensor_fingerprints=(),
                 initial_target_rate_box=None):
        from orbexa.control.nonlinear_flow import NonlinearFlowEncloser
        self.boxes, self.target = dict(initial_relative_boxes), target_ephemeris_box
        if (not self.boxes or any(box.dimension != 6 for box in self.boxes.values())
                or self.target.dimension != 6 or isinstance(history_length, bool) or history_length < 1):
            raise ValueError('Navigation requires a finite nonempty six-state prior and history')
        self.clock, self.model, self.q = clock, model, exact(initial_q)
        self.time = initial_time_s if isinstance(initial_time_s, IntervalBox) else IntervalBox(*zip(bounds(initial_time_s)))
        self.rotation = tuple(tuple(map(exact, row)) for row in initial_target_rotation)
        self.rotation_error = exact(initial_target_rotation_error)
        self.rate_bound = exact(target_rate_bound_rad_s)
        self.acceleration_bound = exact(target_acceleration_bound_rad_s2)
        if self.rotation_error < 0 or self.rate_bound < 0 or self.acceleration_bound < 0:
            raise ValueError('Pose/rate error bounds must be nonnegative')
        self.omega = (IntervalBox.from_center_radius((0, 0, 0), (self.rate_bound,)*3)
                      if initial_target_rate_box is None else initial_target_rate_box)
        if not isinstance(self.omega,IntervalBox) or self.omega.dimension!=3:
            raise ValueError('Initial target angular velocity requires a three-state body-rate box')
        if initial_target_rate_box is not None:
            from orbexa.control.attitude_flow import _sqrt_upper
            if _sqrt_upper(sum(max(abs(lo),abs(hi))**2 for lo,hi in zip(self.omega.lower,self.omega.upper)))>self.rate_bound:
                raise ValueError('Initial target body-rate box exceeds the declared norm bound')
        self.history = deque(maxlen=int(history_length))
        self._acceleration_moments = {}
        self.position_observations = {key: deque(maxlen=int(history_length)) for key in self.boxes}
        self.encloser = encloser or NonlinearFlowEncloser()
        self.packet_ids, self.sensor_ids = [], set()
        self.allowed_sensor_ids = set(sensor_fingerprints)
        self.status, self.reason, self.events = 'VALID', 'Initial declared navigation prior', []

    def predict(self, flows_by_agent, target_flow, commands=None):
        if self.status == 'INCONSISTENT':
            raise ValueError('An inconsistent navigation filter cannot resume silently')
        if set(flows_by_agent) != set(self.boxes) or not target_flow.proven:
            raise ValueError('Complete proved simultaneous fleet prediction is required')
        if (not target_flow.initial_state.contains(self.target) or
                target_flow.model_bounds.get('target_model_fingerprint') != self.model.target_fingerprint
                or target_flow.model_bounds.get('clock_fingerprint') != self.clock.fingerprint):
            raise ValueError('Target prediction does not enclose its retained prior or identity')
        for key, flow in flows_by_agent.items():
            if (not flow.proven or flow.model != MODEL or flow.frame != FRAME
                    or flow.start_anomaly != (self.q, self.q)
                    or flow.hold_width[0] != flow.hold_width[1]
                    or not flow.initial_state.contains(self.boxes[key])
                    or flow.model_bounds.get('plant_fingerprint') != self.model.fingerprint
                    or flow.model_bounds.get('clock_fingerprint') != self.clock.fingerprint
                    or flow.model_bounds.get('target_flow_fingerprint') != target_flow.fingerprint
                    or flow.start_anomaly != target_flow.start_anomaly or flow.hold_width != target_flow.hold_width):
                raise ValueError('Relative prediction changed a prior, clock, model or target identity')
        first = next(iter(flows_by_agent.values()))
        end = self.q+first.hold_width[1]
        if any(flow.hold_width != first.hold_width for flow in flows_by_agent.values()):
            raise ValueError('Navigation fleet clocks must be simultaneous')
        old_time,old_q = self.time,self.q
        self.history.append(dict(start=self.q, end=end, initial=dict(self.boxes),
            flows=dict(flows_by_agent), target_flow=target_flow))
        self.boxes = {key: flow.endpoint for key, flow in flows_by_agent.items()}
        new_time = target_flow.physical_time_endpoint
        self.rotation,self.rotation_error = propagate_pose_estimate(self.rotation,self.rotation_error,
            self.omega,old_q,end,old_time,new_time,self.clock,self.acceleration_bound)
        self.target, self.q, self.time = target_flow.endpoint, end, new_time
        elapsed = max(exact(0), self.time.upper[0]-old_time.lower[0])
        padding = self.acceleration_bound*elapsed
        self.omega = IntervalBox(tuple(v-padding for v in self.omega.lower), tuple(v+padding for v in self.omega.upper))
        self.reason = 'Validated prediction; no observation narrowing'
        return self.estimate()

    def _position_box(self, packet):
        if len(packet.value) != 4:
            raise ValueError('Range/bearing packet has four components')
        # The packet support is measurement-minus-true, so invert by subtraction.
        support = packet.deterministic_error_box
        true = tuple((v-hi, v-lo) for v, lo, hi in zip(packet.value, support.lower, support.upper))
        if true[0][0] <= 0:
            raise ValueError('Range support is not proved positive')
        components = [_product(true[0], direction) for direction in true[1:]]
        return IntervalBox(tuple(pair[0] for pair in components), tuple(pair[1] for pair in components))

    def _constrain(self, box, position):
        return intersect(box, IntervalBox(position.lower+box.lower[3:], position.upper+box.upper[3:]))

    def _velocity_constraint(self, agent, acquired, position):
        """Intersect genuine short and long position-baseline constraints.

        The longest baseline alone eventually magnifies acceleration
        uncertainty. A logarithmic set retains the short-baseline information
        without assuming an independent sensor draw or a measured velocity.
        """
        earliest = self.history[0]['start'] if self.history else self.q
        previous = [(q, p) for q, p in self.position_observations[agent] if earliest <= q < acquired]
        if not previous:
            return None
        original_first_q = previous[0][0]
        previous.sort(key=lambda item: item[0])
        indexes, lookback = {0, next(i for i, item in enumerate(previous) if item[0] == original_first_q)}, 1
        while lookback <= len(previous):
            indexes.add(len(previous)-lookback)
            lookback *= 2
        selected = tuple(previous[index] for index in sorted(indexes))
        if not hasattr(self, '_acceleration_moments'):
            self._acceleration_moments = {}
        ledger = self._acceleration_moments.get(agent)
        if ledger is None:
            ledger = _AccelerationMomentLedger()
            self._acceleration_moments[agent] = ledger
        integrals = ledger.integrals(self.history, agent, tuple((q0, acquired) for q0, _ in selected))
        result = None
        for index, (q0, p0) in enumerate(selected):
            if integrals is NotImplemented:
                candidate = BoundedNavigationFilter._velocity_constraint_scan(
                    self, agent, acquired, position, previous=(q0, p0))
            else:
                integral = integrals[index]
                if integral is None:
                    continue
                lower, upper = integral
                duration = acquired-q0
                candidate = IntervalBox(
                    tuple((a-b+value)/duration for a, b, value in zip(position.lower, p0.upper, lower)),
                    tuple((a-b+value)/duration for a, b, value in zip(position.upper, p0.lower, upper)))
            if candidate is None:
                continue
            result = candidate if result is None else intersect(result, candidate)
            if result is None:
                self.status, self.reason = 'INCONSISTENT', 'Linked position and acceleration supports imply disjoint velocities'
                raise ValueError(self.reason)
        return result

    def _velocity_constraint_scan(self, agent, acquired, position, *, previous=None):
        """Bound endpoint velocity from two positions and certified acceleration.

        w(q1)=(p1-p0)/dq + integral((s-q0)*w'(s),q0,q1)/dq.
        Segment weights are nonnegative exact integrals, so interval sums are
        rigorous. No measured or numerical truth velocity is used.
        """
        earliest = self.history[0]['start'] if self.history else self.q
        if previous is None:
            previous = next(((q, p) for q, p in self.position_observations[agent]
                             if earliest <= q < acquired), None)
        if previous is None:
            return None
        q0, p0 = previous
        duration, cursor = acquired-q0, q0
        lower, upper = [exact(0)]*3, [exact(0)]*3
        for hold in self.history:
            for segment in hold['flows'][agent].segments:
                lo = max(q0, hold['start']+segment.offset[0])
                hi = min(acquired, hold['start']+segment.offset[1])
                if hi <= lo:
                    continue
                if lo != cursor:
                    return None
                weight = ((hi-q0)**2-(lo-q0)**2)/2
                for i in range(3):
                    lower[i] += weight*segment.derivative.lower[i+3]
                    upper[i] += weight*segment.derivative.upper[i+3]
                cursor = hi
        if cursor != acquired:
            return None
        return IntervalBox(tuple((a-b+integral)/duration for a,b,integral in
            zip(position.lower, p0.upper, lower)), tuple((a-b+integral)/duration for a,b,integral in
            zip(position.upper, p0.lower, upper)))

    def _observation_state(self, box, position, velocity=None):
        constrained = self._constrain(box, position)
        if constrained is None or velocity is None:
            return constrained
        return intersect(constrained, IntervalBox(constrained.lower[:3]+velocity.lower,
                                                 constrained.upper[:3]+velocity.upper))

    def _delayed(self, packet, position, velocity=None):
        if packet.acquired_q[0] != packet.acquired_q[1]:
            return None, 'Uncertain acquisition event retained without narrowing'
        acquired = packet.acquired_q[0]
        history = list(self.history)
        index = next((i for i, hold in enumerate(history) if hold['start'] == acquired), None)
        if index is None:
            return None, 'Acquisition event outside finite prediction history; retained without narrowing'
        candidate = self._observation_state(history[index]['initial'][packet.agent_id], position, velocity)
        if candidate is None:
            return False, 'Delayed observation contradicts its acquisition prior'
        for hold in history[index:]:
            source = hold['flows'][packet.agent_id]
            request_record = source.request
            request = HeldInputRequest(candidate,
                IntervalBox(request_record['held_input']['exact_lower'], request_record['held_input']['exact_upper']),
                source.start_anomaly, source.hold_width,
                IntervalBox(request_record['acceleration_disturbance']['exact_lower'],
                            request_record['acceleration_disturbance']['exact_upper']))
            flow = self.encloser.enclose_relative(request, hold['target_flow'], self.model, self.clock)
            if not flow.proven:
                return None, 'Delayed-observation replay unresolved; retained without narrowing'
            candidate = flow.endpoint
        return intersect(self.boxes[packet.agent_id], candidate), 'Delayed observation replayed through certified holds'

    def _gnss(self, packet):
        if packet.acquired_q == (self.q, self.q):
            return infer_target_ephemeris(packet, self.boxes[packet.agent_id], self.clock)
        if packet.acquired_q[0] != packet.acquired_q[1]:
            return None
        history = list(self.history)
        index = next((i for i, hold in enumerate(history) if hold['start'] == packet.acquired_q[0]), None)
        if index is None:
            return None
        candidate = infer_target_ephemeris(packet, history[index]['initial'][packet.agent_id], self.clock)
        candidate = intersect(candidate, history[index]['target_flow'].initial_state)
        if candidate is None:
            return False
        for hold in history[index:]:
            source = hold['target_flow']
            flow = self.encloser.enclose_target(candidate, source.start_anomaly, source.hold_width,
                self.model, self.clock, source.physical_time_initial)
            if not flow.proven:
                return None
            candidate = flow.endpoint
        return candidate

    def consume(self, packets, event_q=None, physical_time_box=None):
        if self.status == 'INCONSISTENT':
            raise ValueError('An inconsistent navigation filter cannot resume silently')
        if event_q is not None and exact(event_q) != self.q:
            raise ValueError('Measurements cannot relabel the navigation event')
        if physical_time_box is not None and physical_time_box != self.time:
            raise ValueError('Measurement consumer must use the retained physical-time enclosure')
        accepted = set(self.packet_ids)
        for packet in packets:
            if packet.packet_id in accepted:
                continue
            if (packet.frame_id not in (FRAME, 'inertial-cartesian-si-v1') or packet.acquired_q[1] > self.q or
                    packet.delivery_time_s.upper[0] > self.time.lower[0] or packet.quality_status != 'VALID'):
                self.events.append(dict(packet_id=packet.packet_id, status='NOT_READY_OR_UNSUPPORTED'))
                continue
            if not self.allowed_sensor_ids:
                self.allowed_sensor_ids.add(packet.sensor_model_fingerprint)
            elif packet.sensor_model_fingerprint not in self.allowed_sensor_ids:
                self.status, self.reason = 'INCONSISTENT', 'Sensor model identity changed without a declared transition'
                return self.estimate()
            if packet.measurement_kind == 'range_bearing' and packet.frame_id == FRAME and packet.agent_id in self.boxes:
                position = self._position_box(packet)
                velocity = (self._velocity_constraint(packet.agent_id, packet.acquired_q[0], position)
                            if packet.acquired_q[0] == packet.acquired_q[1] else None)
                if packet.acquired_q == (self.q, self.q):
                    narrowed = self._observation_state(self.boxes[packet.agent_id], position, velocity)
                    reason = 'Current processed range/bearing support intersection'
                else:
                    narrowed, reason = self._delayed(packet, position, velocity)
                if narrowed is False or narrowed is None and 'contradict' in reason:
                    self.status, self.reason = 'INCONSISTENT', reason
                    return self.estimate()
                if narrowed is None:
                    # Distinguish an empty replay intersection from unavailable replay.
                    if reason.startswith('Delayed observation replayed') or reason.startswith('Current'):
                        self.status, self.reason = 'INCONSISTENT', 'Observation contradicts validated state enclosure'
                        return self.estimate()
                    self.events.append(dict(packet_id=packet.packet_id, status='UNRESOLVED', reason=reason))
                    continue
                self.boxes[packet.agent_id] = narrowed
                self.position_observations[packet.agent_id].append((packet.acquired_q[0], position))
                if velocity is not None:
                    reason += '; velocity constrained by certified acceleration integral'
            elif packet.measurement_kind == 'chaser_gnss' and packet.agent_id in self.boxes:
                inferred = self._gnss(packet)
                if inferred is None:
                    self.events.append(dict(packet_id=packet.packet_id, status='UNRESOLVED_GNSS_REPLAY'))
                    continue
                narrowed = None if inferred is False else intersect(self.target, inferred)
                if narrowed is None:
                    self.status, self.reason = 'INCONSISTENT', 'GNSS/relative geometry contradicts target ephemeris prior'
                    return self.estimate()
                self.target = narrowed
                reason = 'Target ephemeris inferred from chaser GNSS plus relative geometry'
            elif packet.measurement_kind == 'target_pose':
                if len(packet.value) != 12:
                    raise ValueError('Processed target pose packet requires matrix and angular rate')
                age = max(exact(0), self.time.upper[0]-packet.acquisition_time_s.lower[0])
                support = packet.deterministic_error_box
                acquisition_rate = IntervalBox(tuple(v-hi for v,hi in zip(packet.value[9:],support.upper[9:])),
                                               tuple(v-lo for v,lo in zip(packet.value[9:],support.lower[9:])))
                self.rotation,self.rotation_error = propagate_pose_estimate(
                    tuple(tuple(packet.value[3*i:3*i+3]) for i in range(3)),packet.orientation_operator_error,
                    acquisition_rate,packet.acquired_q,self.q,packet.acquisition_time_s,self.time,
                    self.clock,self.acceleration_bound)
                rate = IntervalBox(tuple(v-hi-self.acceleration_bound*age for v, hi in zip(packet.value[9:], support.upper[9:])),
                                   tuple(v-lo+self.acceleration_bound*age for v, lo in zip(packet.value[9:], support.lower[9:])))
                narrowed = intersect(self.omega, rate)
                if narrowed is None:
                    self.status, self.reason = 'INCONSISTENT', 'Target rate observation contradicts retained rate bounds'
                    return self.estimate()
                self.omega = narrowed
                reason = 'Processed pose/rate support with acquisition-age padding'
            else:
                continue
            self.packet_ids.append(packet.packet_id)
            accepted.add(packet.packet_id)
            self.sensor_ids.add(packet.sensor_model_fingerprint)
            self.reason = reason
            self.events.append(dict(packet_id=packet.packet_id, status='ACCEPTED', reason=reason))
        return self.estimate()

    def estimate(self):
        return NavigationEstimate(self.q, self.time, dict(self.boxes), self.target,
            self.rotation, self.rotation_error, self.omega, tuple(self.packet_ids),
            self.clock.fingerprint, self.model.fingerprint, tuple(sorted(self.sensor_ids)), self.status, self.reason)

    def condition_on_admitted_fallback(self,plan,observations):
        """Retain the relational invariant already owned by admission runtime.

        A rectangular observation may protrude from that invariant. Its
        established physical membership justifies this intersection; it is
        never a replacement by a numerical point or a newly claimed invariant.
        """
        from orbexa.control.fleet_fallback import FleetFallbackPlan
        if not isinstance(plan,FleetFallbackPlan) or not plan.certificate.admissible:
            raise ValueError('A complete admitted fallback certificate is required')
        plan.verify_integrity()
        if set(observations)!=set(self.boxes):
            raise ValueError('Fallback conditioning fleet changed')
        updated = {}
        for key,controller in plan.segment_at(self.q).controllers:
            candidate = controller.observation_box(observations[key],self.q)
            updated[key] = intersect(self.boxes[key],candidate)
            if updated[key] is None:
                raise ValueError('Admitted invariant contradicts navigation')
        self.boxes = updated
        self.events.append(dict(status='CONDITIONED_ON_ADMITTED_INVARIANT',q=str(self.q),fallback_id=plan.fingerprint))
        return self.estimate()

    def verify_angular_acceleration_bound(self,attitude_certificate):
        """Gate the pose-prediction hypothesis on an actual attitude inclusion."""
        from orbexa.control.attitude_flow import AttitudeFlowCertificate,_sqrt_upper
        def refuse(reason):
            self.status,self.reason='INCONSISTENT',reason
            raise ValueError(reason)
        if (not isinstance(attitude_certificate,AttitudeFlowCertificate) or not attitude_certificate.proven
                or attitude_certificate.context.get('clock_fingerprint')!=self.clock.fingerprint
                or attitude_certificate.context.get('plant_fingerprint')!=self.model.fingerprint
                or not attitude_certificate.initial_state.slice(4,7).contains(self.omega)):
            refuse('Angular-acceleration gate requires a same-model attitude inclusion containing the retained rate prior')
        observed=attitude_certificate.context.get('observation_reference_rotation')
        error=attitude_certificate.context.get('observation_operator_error')
        if (observed is None or tuple(tuple(map(exact,row)) for row in observed)!=self.rotation
                or exact(error)!=self.rotation_error):
            refuse('Attitude inclusion does not bind the retained observed rotation support')
        maximum=exact(0)
        for segment in attitude_certificate.segments:
            rate=self.clock.interval_bounds(segment.anomaly)['rate_s_inv'].upper[0]
            acceleration=_sqrt_upper(sum((max(abs(lo),abs(hi))*rate)**2
                for lo,hi in zip(segment.derivative.lower[4:],segment.derivative.upper[4:])))
            maximum=max(maximum,acceleration)
        if maximum>self.acceleration_bound:
            refuse('Declared target angular-acceleration bound is smaller than the admitted physical attitude inclusion')
        record=dict(status='ANGULAR_ACCELERATION_BOUND_VERIFIED',attitude_flow_id=attitude_certificate.fingerprint,
                    physical_acceleration_upper_rad_s2=str(maximum),declared_bound_rad_s2=str(self.acceleration_bound))
        self.events.append(record)
        return record


def infer_target_ephemeris(gnss_packet, relative_box, clock, precision_bits=128):
    """Infer target state from a chaser receiver and relative navigation only.

    The caller must respect delivery time before using this initial prior.
    No cooperative target receiver or truth ephemeris is assumed.
    """
    if (gnss_packet.measurement_kind != 'chaser_gnss' or
            gnss_packet.frame_id != 'inertial-cartesian-si-v1' or len(gnss_packet.value) != 6):
        raise ValueError('A chaser inertial GNSS packet is required')
    from orbexa.control.nonlinear_flow import _Arithmetic
    from orbexa.control.reachability import _ARB_LOCK
    from orbexa.core.nonlinear_relative import matvec, z
    import flint
    with _ARB_LOCK, flint.ctx.workprec(precision_bits):
        ar = _Arithmetic(flint)
        support = gnss_packet.deterministic_error_box
        absolute = ar.vector(IntervalBox(tuple(v-hi for v, hi in zip(gnss_packet.value, support.upper)),
                                        tuple(v-lo for v, lo in zip(gnss_packet.value, support.lower))))
        relative = ar.vector(relative_box)
        rotation, rate, _ = clock.evaluate(ar.interval(gnss_packet.acquired_q), ar)
        delta_r = matvec(rotation, relative[:3])
        delta_v = matvec(rotation, tuple(rate*(a+b) for a,b in zip(relative[3:], z(relative[:3]))))
        return ar.box(tuple(a-b for a,b in zip(absolute[:3], delta_r))+
                      tuple(a-b for a,b in zip(absolute[3:], delta_v)))


def propagate_pose_estimate(rotation_reference,operator_error,body_rate_box,q_start,q_end,
                            time_start,time_end,clock,angular_acceleration_bound,precision_bits=128):
    """Propagate an observed pose center, with rigorous group-error padding.

    The center uses constant midpoint body rate. The deterministic error adds
    rate uncertainty, bounded angular acceleration, timing and an Arb bound on
    all floating matrix/rotation evaluation error. This prevents latency from
    becoming an unmodeled pose lag while retaining the full error support.
    """
    from scipy.spatial.transform import Rotation
    from orbexa.control.nonlinear_flow import _Arithmetic
    from orbexa.control.attitude_flow import _sqrt_upper
    from orbexa.control.reachability import _ARB_LOCK
    import flint
    q0,q1 = bounds(q_start),bounds(q_end)
    delta_lo = max(exact(0),time_end.lower[0]-time_start.upper[0])
    delta_hi = max(exact(0),time_end.upper[0]-time_start.lower[0])
    tau = (delta_lo+delta_hi)/2
    omega = tuple((lo+hi)/2 for lo,hi in zip(body_rate_box.lower,body_rate_box.upper))
    radii = tuple((hi-lo)/2 for lo,hi in zip(body_rate_box.lower,body_rate_box.upper))
    omega_error,omega_norm = _sqrt_upper(sum(v*v for v in radii)),_sqrt_upper(sum(v*v for v in omega))
    error = exact(operator_error)+omega_error*delta_hi+exact(angular_acceleration_bound)*delta_hi**2/2
    error += omega_norm*max(tau-delta_lo,delta_hi-tau)
    c0 = np.asarray(clock.point(float(sum(q0)/2)).rotation)
    c1 = np.asarray(clock.point(float(sum(q1)/2)).rotation)
    observed = tuple(tuple(map(exact,row)) for row in rotation_reference)
    center = c1.T@c0@np.asarray(observed,dtype=float)@Rotation.from_rotvec(np.array(list(map(float,omega)))*float(tau)).as_matrix()
    center = tuple(tuple(map(exact,row)) for row in center)
    with _ARB_LOCK,flint.ctx.workprec(precision_bits):
        ar = _Arithmetic(flint)
        first,last = clock.evaluate(ar.interval(q0),ar)[0],clock.evaluate(ar.interval(q1),ar)[0]
        ww = tuple(map(ar.scalar,omega))
        norm2 = sum(ar.square(v) for v in ww)
        skew = ((ar.scalar(0),-ww[2],ww[1]),(ww[2],ar.scalar(0),-ww[0]),(-ww[1],ww[0],ar.scalar(0)))
        if omega_norm==0:
            exp_rotation = tuple(tuple(ar.scalar(int(i==j)) for j in range(3)) for i in range(3))
        else:
            norm = ar.sqrt(norm2)
            theta = norm*ar.scalar(tau)
            k2 = tuple(tuple(sum(skew[i][k]*skew[k][j] for k in range(3)) for j in range(3)) for i in range(3))
            exp_rotation = tuple(tuple(ar.scalar(int(i==j))+ar.sin(theta)/norm*skew[i][j]
                +(1-ar.cos(theta))/norm2*k2[i][j] for j in range(3)) for i in range(3))
        obs = tuple(tuple(ar.scalar(v) for v in row) for row in observed)
        fixed = tuple(tuple(sum(first[i][k]*obs[k][j] for k in range(3)) for j in range(3)) for i in range(3))
        inertial = tuple(tuple(sum(fixed[i][k]*exp_rotation[k][j] for k in range(3)) for j in range(3)) for i in range(3))
        propagated = tuple(tuple(sum(last[k][i]*inertial[k][j] for k in range(3)) for j in range(3)) for i in range(3))
        residual = ar.box(tuple(propagated[i][j]-ar.scalar(center[i][j]) for i in range(3) for j in range(3)))
        error += _sqrt_upper(sum(max(abs(lo),abs(hi))**2 for lo,hi in zip(residual.lower,residual.upper)))
    return center,error
