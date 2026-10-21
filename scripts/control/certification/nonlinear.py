"""Finite-budget Arb inclusions for the explicitly declared nonlinear plant.

No numeric callback, sampled maximum, or numerical ephemeris reset is accepted
as an interval model. Coordinates use the immutable commanded reference clock.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
from pathlib import Path

from orbexa.core.dynamics.reference_frame import bounds, fingerprint
from orbexa.core.dynamics.relative import parameters, target_field, relative_field, differential_acceleration, matvec, transpose, z
from orbexa.control.certification.reachability import IntervalBox, HeldInputRequest, OrbitalModelBounds, FlowCertificate, SweptSegment, EnclosureStatus, ReachabilityConfig, _ARB_LOCK, _ArbArithmetic, _ProofUnresolved, _orbital_coefficients

FRAME = 'reference-orbit-frame-v1'
MODEL = 'nonlinear-reference-relative-v1'
_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class _Arithmetic(_ArbArithmetic):
    sqrt = staticmethod(lambda value: value.sqrt())
    exp = staticmethod(lambda value: value.exp())
    expm1 = staticmethod(lambda value: value.expm1())
    sin = staticmethod(lambda value: value.sin())
    cos = staticmethod(lambda value: value.cos())

    def square(self, value):
        # flint 0.8's generic power uses log for negative balls, even **2.
        box = self.box((value,))
        lo, hi = box.lower[0], box.upper[0]
        lower = 0 if lo <= 0 <= hi else min(lo*lo, hi*hi)
        return self.interval((lower, max(lo*lo, hi*hi)))


@dataclass(frozen=True)
class NominalCoefficientBounds:
    """Finite Arb CH surrogate coefficients in the declared q coordinates."""
    q_interval: tuple
    nominal_model: OrbitalModelBounds
    acceleration_matrix: IntervalBox
    acceleration_drift: IntervalBox
    precision_bits: int

    @property
    def matrix_rows(self):return tuple(self.acceleration_matrix.slice(6*i,6*(i+1)) for i in range(3))

    @property
    def absolute_upper(self):
        return tuple(tuple(max(abs(lo),abs(hi)) for lo,hi in zip(row.lower,row.upper)) for row in self.matrix_rows)

    @property
    def variation_upper(self):
        return tuple(tuple(hi-lo for lo,hi in zip(row.lower,row.upper)) for row in self.matrix_rows)

    @property
    def drift_upper(self):
        return tuple(max(abs(lo),abs(hi)) for lo,hi in zip(self.acceleration_drift.lower,self.acceleration_drift.upper))

    def as_dict(self):
        return dict(method='arb-commanded-CH-nominal-coefficients-v1',backend_source_sha256=_SOURCE_SHA256,
            q_interval=tuple(map(str,self.q_interval)),nominal_model=self.nominal_model.as_dict(),
            acceleration_matrix=self.acceleration_matrix.as_dict(),acceleration_drift=self.acceleration_drift.as_dict(),
            precision_bits=self.precision_bits,role='Nominal CH surrogate coefficients; no nonlinear physical-frame admission')

    @property
    def fingerprint(self):return fingerprint(self.as_dict())


def bound_nominal_coefficients(q_interval,nominal_parameters,precision_bits=128):
    """Enclose all nominal acceleration entries/drift over an exact q interval.

    Uses the same fixed Arb expression as the continuous nominal service and
    nonlinear-minus-CH residual. Callers outwardly map exact component bounds;
    sampled coefficients or an arbitrary callback cannot set this envelope.
    """
    if isinstance(precision_bits,bool) or not isinstance(precision_bits,int) or not 64<=precision_bits<=4096:
        raise ValueError('Declare a finite Arb coefficient precision between64 and4096bits')
    interval=bounds(q_interval)
    nominal=OrbitalModelBounds(eccentricity=bounds(nominal_parameters['eccentricity']),
        alpha=bounds(nominal_parameters['alpha']),beta=bounds(nominal_parameters['beta']),
        semi_major_axis=nominal_parameters['semi_major_axis'],mu=nominal_parameters['mu'],
        theta0=bounds(nominal_parameters.get('theta0',0)))
    import flint
    with _ARB_LOCK,flint.ctx.workprec(precision_bits):
        ar=_Arithmetic(flint)
        matrix,drift,_=_orbital_coefficients(ar.interval(interval),nominal,ar)
        acceleration=ar.box(tuple(value for row in matrix[3:] for value in row))
        disturbance=ar.box(tuple(drift[3:]))
    return NominalCoefficientBounds(interval,nominal,acceleration,disturbance,precision_bits)


@dataclass(frozen=True)
class TargetEphemerisCertificate(FlowCertificate):
    physical_time_initial: IntervalBox | None = None
    physical_time_endpoint: IntervalBox | None = None
    physical_time_swept: IntervalBox | None = None
    segment_time_boxes: tuple = ()

    def as_dict(self, *, include_segments=True):
        result = super().as_dict(include_segments=include_segments)
        result.update(physical_time_initial=None if self.physical_time_initial is None else self.physical_time_initial.as_dict(),
            physical_time_endpoint=None if self.physical_time_endpoint is None else self.physical_time_endpoint.as_dict(),
            physical_time_swept=None if self.physical_time_swept is None else self.physical_time_swept.as_dict())
        if include_segments:
            result['segment_time_boxes'] = tuple(box.as_dict() for box in self.segment_time_boxes)
        return result


@dataclass(frozen=True)
class RegionalResidualCertificate:
    status: EnclosureStatus
    reason: str
    component_bounds: IntervalBox | None
    context: dict

    @property
    def proven(self):
        return self.status == EnclosureStatus.PROVEN

    @property
    def eta_component_box(self):
        return self.component_bounds

    @property
    def upper_bounds(self):
        if not self.proven:
            raise ValueError('Unresolved residual has no usable forcing allowance')
        return tuple(max(abs(lo), abs(hi)) for lo, hi in zip(self.component_bounds.lower, self.component_bounds.upper))

    @property
    def fingerprint(self):
        return fingerprint(self.as_dict())

    def as_dict(self):
        return dict(status=self.status.value, reason=self.reason,
            component_bounds=None if self.component_bounds is None else self.component_bounds.as_dict(), context=self.context)


def _region(state_box, model):
    if state_box.dimension != 6:
        raise ValueError('Relative regional domain requires six states')
    limits = (model.relative_position_limit_m,)*3+(model.relative_velocity_limit_m_per_rad,)*3
    if any(lo < -limit or hi > limit for lo, hi, limit in zip(state_box.lower, state_box.upper, limits)):
        raise ValueError('Complete relative region escapes its declared componentwise domain')


def _target_box(flow, interval):
    if not flow.proven or not isinstance(flow, TargetEphemerisCertificate):
        raise ValueError('A proven nonlinear target ephemeris certificate is required')
    lo, hi = interval
    if lo < flow.start_anomaly[0] or hi > flow.start_anomaly[1]+flow.hold_width[1]:
        raise ValueError('Relative query escapes target ephemeris clock coverage')
    selected = [segment.swept_state for segment in flow.segments
                if segment.anomaly[0] <= hi and segment.anomaly[1] >= lo]
    if not selected:
        if flow.hold_width[1] == 0:
            return flow.initial_state
        raise ValueError('Target ephemeris has no covering continuous segment')
    result = selected[0]
    for box in selected[1:]:
        result = result.hull(box)
    return result


class NonlinearFlowEncloser:
    def __init__(self, config=None):
        self.config = config or ReachabilityConfig()

    @staticmethod
    def identity(flow):
        return {name: flow.model_bounds.get(name) for name in
            ('plant_fingerprint', 'target_model_fingerprint', 'clock_fingerprint', 'target_flow_fingerprint')}

    def enclose_target(self, initial_state, q_start, q_width, model, clock, initial_time_s=(0, 0)):
        start, width = bounds(q_start), bounds(q_width)
        time = initial_time_s if isinstance(initial_time_s, IntervalBox) else IntervalBox(*zip(bounds(initial_time_s)))
        if initial_state.dimension != 6 or time.dimension != 1 or width[0] < 0:
            raise ValueError('Target ephemeris needs six states, one time and nonnegative width')
        context = dict(plant=model.as_dict(), target_model_fingerprint=model.target_fingerprint,
                       clock=clock.as_dict(), clock_fingerprint=clock.fingerprint, backend_source_sha256=_SOURCE_SHA256)
        request = dict(initial_state=initial_state.as_dict(), start_anomaly=tuple(map(str, start)),
            hold_width=tuple(map(str, width)), physical_time_initial=time.as_dict())
        common = dict(start_anomaly=start, hold_width=width, initial_state=initial_state,
            frame='inertial-cartesian-si-v1', model='nonlinear-target-ephemeris-v1',
            model_bounds=context, request=request, precision_bits=self.config.precision_bits)
        def field(q_bounds, state, ar):
            q = ar.interval(q_bounds)
            return target_field(q, state[:6], model, clock, ar)+(1/clock.evaluate(q, ar)[1],)
        initial = IntervalBox(initial_state.lower+time.lower, initial_state.upper+time.upper)
        return self._run(initial, start, width, field, model, clock, common, target=True, time_initial=time)

    def enclose_relative(self, request, target_flow, model, clock):
        if not isinstance(request, HeldInputRequest):
            raise TypeError('Use a HeldInputRequest for the six-state relative core')
        context = dict(plant=model.as_dict(), plant_fingerprint=model.fingerprint,
            backend_source_sha256=_SOURCE_SHA256,
            target_model_fingerprint=model.target_fingerprint, clock=clock.as_dict(),
            clock_fingerprint=clock.fingerprint, target_flow_fingerprint=target_flow.fingerprint,
            actuation='held-q-command; inertial acceleration=nu_ref^2*C_ref*u; additive error in q acceleration')
        common = dict(start_anomaly=request.start_anomaly, hold_width=request.hold_width,
            initial_state=request.initial_state, frame=FRAME, model=MODEL, model_bounds=context,
            request=request.as_dict(), precision_bits=self.config.precision_bits)
        def field(q_bounds, state, ar):
            _region(ar.box(state), model)
            if (target_flow.model_bounds.get('target_model_fingerprint') != model.target_fingerprint or
                    target_flow.model_bounds.get('clock_fingerprint') != clock.fingerprint):
                raise ValueError('Target ephemeris model or clock identity changed')
            target = ar.vector(_target_box(target_flow, q_bounds))
            rhs = relative_field(ar.interval(q_bounds), state, target, ar.vector(request.held_input), model, clock, ar)
            disturbance = ar.vector(request.acceleration_disturbance)
            return rhs[:3]+tuple(a+b for a, b in zip(rhs[3:], disturbance))
        return self._run(request.initial_state, request.start_anomaly, request.hold_width,
                         field, model, clock, common)

    def _run(self, initial, start, width, field, model, clock, common, target=False, time_initial=None):
        certificate = TargetEphemerisCertificate if target else FlowCertificate
        extra = dict(physical_time_initial=time_initial) if target else {}
        try:
            import flint
        except ImportError as error:
            return certificate(EnclosureStatus.UNAVAILABLE, str(error), **common, **extra)
        common['backend_version'] = getattr(flint, '__version__', 'unknown')
        try:
            clock.check_domain((start[0], start[1]+width[1]))
            with _ARB_LOCK, flint.ctx.workprec(self.config.precision_bits):
                return self._integrate(initial, start, width, field, clock, _Arithmetic(flint),
                                       common, target, time_initial)
        except (ValueError, ArithmeticError, _ProofUnresolved) as error:
            return certificate(EnclosureStatus.UNRESOLVED, str(error), **common, **extra)

    def _integrate(self, initial, start, hold, field, clock, ar, common, target, time_initial):
        cfg, final = self.config, hold[1]
        if final == 0:
            # Even a zero-width request must have a valid physical domain.
            field(start, ar.vector(initial), ar)
            extra = dict(physical_time_initial=time_initial, physical_time_endpoint=time_initial,
                         physical_time_swept=time_initial) if target else {}
            cls = TargetEphemerisCertificate if target else FlowCertificate
            return cls(EnclosureStatus.PROVEN, 'Zero-width exact initial enclosure',
                endpoint=common['initial_state'], swept_state=common['initial_state'],
                elapsed_seconds=IntervalBox.point((0,)), **common, **extra)
        current, offset = ar.vector(initial), Fraction(0)
        proposed = min(cfg.max_step, final)
        segments, time_segments, swept, terminal = [], [], None, None
        physical_velocity, elapsed_lo, elapsed_hi = None, Fraction(0), Fraction(0)
        terminal_time = None
        while offset < final:
            if len(segments) >= cfg.max_substeps:
                raise _ProofUnresolved('Nonlinear inclusion exhausted max_substeps')
            remaining = final-offset
            ratio = remaining/proposed
            width = remaining/((ratio.numerator+ratio.denominator-1)//ratio.denominator)
            witness = None
            while width >= cfg.min_step:
                q_bounds = (start[0]+offset, start[1]+offset+width)
                try:
                    witness = self._picard(current, q_bounds, width, field, ar)
                except (ValueError, ArithmeticError):
                    witness = None
                if witness is not None:
                    break
                width /= 2
            if witness is None:
                raise _ProofUnresolved('Nonlinear Picard self-inclusion or physical domain unresolved at min_step')
            candidate, image, derivative, endpoint, iterations = witness
            boxes = tuple(ar.box(value[:6]) for value in (current, candidate, endpoint, image, derivative))
            rate = clock.evaluate(ar.interval(q_bounds), ar)[1]
            segment = SweptSegment((offset, offset+width), q_bounds, *boxes, iterations, ar.box((rate,)))
            segments.append(segment)
            swept = boxes[1] if swept is None else swept.hull(boxes[1])
            dt = ar.box((ar.scalar(width)/rate,))
            elapsed_lo += dt.lower[0]
            elapsed_hi += dt.upper[0]
            if target:
                time_segments.append(ar.box(candidate[6:7]))
            if hold[0] != hold[1] and offset+width >= hold[0]:
                stop = ar.interval((max(Fraction(0), hold[0]-offset), width))
                stop_state = ar.box(tuple(x+stop*f for x, f in zip(current[:6], derivative[:6])))
                terminal = stop_state if terminal is None else terminal.hull(stop_state)
                if target:
                    stopped_time = ar.box((current[6]+stop*derivative[6],))
                    terminal_time = stopped_time if terminal_time is None else terminal_time.hull(stopped_time)
            if not target:
                velocity = ar.box(tuple(v*rate for v in candidate[3:6]))
                physical_velocity = velocity if physical_velocity is None else physical_velocity.hull(velocity)
            current, offset = endpoint, offset+width
            proposed = min(cfg.max_step, 2*width)
        if hold[0] == hold[1]:
            terminal = ar.box(current[:6])
            if target:
                terminal_time = ar.box(current[6:7])
        cls = TargetEphemerisCertificate if target else FlowCertificate
        extras = dict(physical_time_initial=time_initial, physical_time_endpoint=terminal_time,
            physical_time_swept=IntervalBox((min(time_initial.lower[0], min(b.lower[0] for b in time_segments)),),
                (max(b.upper[0] for b in time_segments),)), segment_time_boxes=tuple(time_segments)) if target else {}
        return cls(EnclosureStatus.PROVEN, 'All nonlinear substeps satisfy strict Arb Picard self-inclusion',
            endpoint=terminal, swept_state=swept, segments=tuple(segments),
            swept_physical_velocity=physical_velocity,
            elapsed_seconds=IntervalBox((elapsed_lo if hold[0] == hold[1] else 0,), (elapsed_hi,)),
            **common, **extras)

    def _picard(self, initial, q_bounds, width, field, ar):
        offsets, h = ar.interval((0, width)), ar.scalar(width)
        first = field(q_bounds, initial, ar)
        if not all(v.is_finite() for v in first):
            return None
        candidate = ar.inflate(tuple(x.union(x+offsets*f) for x, f in zip(initial, first)), self.config)
        for iteration in range(1, self.config.max_picard_iterations+1):
            derivative = field(q_bounds, candidate, ar)
            image = tuple(x+offsets*f for x, f in zip(initial, derivative))
            if not all(v.is_finite() for v in image):
                return None
            if all(y.contains_interior(value) for y, value in zip(candidate, image)):
                endpoint = tuple(x+h*f for x, f in zip(initial, derivative))
                return candidate, image, derivative, endpoint, iteration
            candidate = ar.inflate(tuple(y.union(value) for y, value in zip(candidate, image)), self.config)
        return None

    def bound_residual(self, q_interval, target_flow, regional_state_box, input_box,
                       nominal_parameters, model, clock):
        interval = bounds(q_interval)
        context = dict(q_interval=tuple(map(str, interval)), regional_state_box=regional_state_box.as_dict(),
            backend_source_sha256=_SOURCE_SHA256,
            input_box=input_box.as_dict(), nominal_parameters={k: str(v) for k, v in nominal_parameters.items()},
            plant_fingerprint=model.fingerprint, clock_fingerprint=clock.fingerprint,
            target_flow_fingerprint=target_flow.fingerprint)
        try:
            import flint
        except ImportError as error:
            return RegionalResidualCertificate(EnclosureStatus.UNAVAILABLE, str(error), None, context)
        try:
            _region(regional_state_box, model)
            clock.check_domain(interval)
            if input_box.dimension != 3:
                raise ValueError('Residual input has three components')
            if (target_flow.model_bounds.get('target_model_fingerprint') != model.target_fingerprint or
                    target_flow.model_bounds.get('clock_fingerprint') != clock.fingerprint):
                raise ValueError('Residual target model/clock identity mismatch')
            nominal = OrbitalModelBounds(eccentricity=bounds(nominal_parameters['eccentricity']),
                alpha=bounds(nominal_parameters['alpha']), beta=bounds(nominal_parameters['beta']),
                semi_major_axis=nominal_parameters.get('semi_major_axis', clock.semi_major_axis_m),
                mu=nominal_parameters.get('mu', clock.mu_m3_s2), theta0=bounds(nominal_parameters.get('theta0', 0)))
            with _ARB_LOCK, flint.ctx.workprec(self.config.precision_bits):
                ar = _Arithmetic(flint)
                state, q = ar.vector(regional_state_box), ar.interval(interval)
                # Cancel common commanded input, Coriolis and centrifugal
                # terms algebraically before interval evaluation.
                rotation, rate, kappa = clock.evaluate(q, ar)
                p, w = state[:3], state[3:]
                zp = z(p)
                delta_r = matvec(rotation, p)
                delta_v = matvec(rotation, tuple(rate*(a+b) for a,b in zip(w,zp)))
                delta_a = differential_acceleration(ar.vector(_target_box(target_flow, interval)),
                    delta_r, delta_v, model, parameters(model,ar), ar)
                normalized = matvec(transpose(rotation), delta_a)
                matrix, drift, _ = _orbital_coefficients(q, nominal, ar)
                curvature, damping = -matrix[5][2], matrix[5][5]
                gravity_linear = (-curvature*p[0], 2*curvature*p[1], -curvature*p[2])
                difference = tuple(a/rate**2-g+(-kappa-damping)*(v+b)-d for a,g,v,b,d
                    in zip(normalized,gravity_linear,w,zp,drift[3:]))
                box = ar.box(difference)
            return RegionalResidualCertificate(EnclosureStatus.PROVEN,
                'Full regional nonlinear-minus-CH expression enclosed with Arb; command cancels exactly', box, context)
        except (ValueError, ArithmeticError, _ProofUnresolved) as error:
            return RegionalResidualCertificate(EnclosureStatus.UNRESOLVED, str(error), None, context)
