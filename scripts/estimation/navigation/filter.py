"""Bounded navigation consuming declared priors, proved flow and packets."""
from __future__ import annotations

from collections import deque

from orbexa.core.dynamics.reference_frame import exact, bounds
from orbexa.control.certification.reachability import IntervalBox, HeldInputRequest
from orbexa.control.certification.nonlinear import FRAME, MODEL
from orbexa.estimation.navigation.boxes import intersect, _product
from orbexa.estimation.navigation.records import NavigationEstimate
from orbexa.estimation.navigation.moments import _AccelerationMomentLedger
from orbexa.estimation.navigation.target_pose import infer_target_ephemeris, propagate_pose_estimate


class BoundedNavigationFilter:
    """Set prediction/intersection with no truth center or covariance shortcut."""
    def __init__(self, initial_relative_boxes, target_ephemeris_box, clock, model,
                 initial_q=0, initial_time_s=(0, 0), *, history_length=512,
                 initial_target_rotation=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                 initial_target_rotation_error=2, target_rate_bound_rad_s='0.1',
                 target_acceleration_bound_rad_s2='0.001', encloser=None, sensor_fingerprints=(),
                 initial_target_rate_box=None):
        from orbexa.control.certification.nonlinear import NonlinearFlowEncloser
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
            from orbexa.control.certification.attitude import _sqrt_upper
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
        from orbexa.control.fallback.plan import FleetFallbackPlan
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
        from orbexa.control.certification.attitude import AttitudeFlowCertificate, _sqrt_upper
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
