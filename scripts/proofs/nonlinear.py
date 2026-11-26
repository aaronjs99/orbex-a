"""nonlinear.py: Finite-budget Arb inclusions for the explicitly declared nonlinear plant.

No numeric callback, sampled maximum, or numerical ephemeris reset is accepted
as an interval model. Coordinates use the immutable commanded reference clock.

Copyright: Aaron John Sabu
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from orbexa.common.intervals import (
    _ARB_LOCK,
    IntervalBox,
    _Arithmetic,
    _ProofUnresolved,
)
from orbexa.models.dynamics.reference_frame import bounds, fingerprint
from orbexa.models.dynamics.relative import (
    differential_acceleration,
    matvec,
    parameters,
    relative_field,
    target_field,
    transpose,
    z,
)
from orbexa.proofs.reachability import (
    EnclosureStatus,
    FlowCertificate,
    HeldInputRequest,
    OrbitalModelBounds,
    ReachabilityConfig,
    SweptSegment,
    _orbital_coefficients,
)

FRAME = "reference-orbit-frame-v1"
MODEL = "nonlinear-reference-relative-v1"
_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class NominalCoefficientBounds:
    """Finite Arb CH surrogate coefficients in the declared q coordinates."""

    q_interval: tuple
    nominal_model: OrbitalModelBounds
    acceleration_matrix: IntervalBox
    acceleration_drift: IntervalBox
    precision_bits: int

    @property
    def matrix_rows(self):
        """Return the three acceleration rows as six-entry slices of the matrix box."""
        return tuple(
            self.acceleration_matrix.slice(6 * i, 6 * (i + 1)) for i in range(3)
        )

    @property
    def absolute_upper(self):
        """Return max(|lower|, |upper|) per entry for each acceleration row."""
        return tuple(
            tuple(max(abs(lo), abs(hi)) for lo, hi in zip(row.lower, row.upper))
            for row in self.matrix_rows
        )

    @property
    def variation_upper(self):
        """Return upper minus lower per entry for each acceleration row."""
        return tuple(
            tuple(hi - lo for lo, hi in zip(row.lower, row.upper))
            for row in self.matrix_rows
        )

    @property
    def drift_upper(self):
        """Return max(|lower|, |upper|) per component of the acceleration drift."""
        return tuple(
            max(abs(lo), abs(hi))
            for lo, hi in zip(
                self.acceleration_drift.lower, self.acceleration_drift.upper
            )
        )

    def as_dict(self):
        """Return a serializable record of the bounds, backend source hash and role."""
        return {
            "method": "arb-commanded-CH-nominal-coefficients-v1",
            "backend_source_sha256": _SOURCE_SHA256,
            "q_interval": tuple(map(str, self.q_interval)),
            "nominal_model": self.nominal_model.as_dict(),
            "acceleration_matrix": self.acceleration_matrix.as_dict(),
            "acceleration_drift": self.acceleration_drift.as_dict(),
            "precision_bits": self.precision_bits,
            "role": "Nominal CH surrogate coefficients; no nonlinear physical-frame admission",
        }

    @property
    def fingerprint(self):
        """Return the fingerprint of the as_dict() record."""
        return fingerprint(self.as_dict())


def bound_nominal_coefficients(q_interval, nominal_parameters, precision_bits=128):
    """Enclose all nominal acceleration entries/drift over an exact q interval.

    Uses the same fixed Arb expression as the continuous nominal service and
    nonlinear-minus-CH residual. Callers outwardly map exact component bounds;
    sampled coefficients or an arbitrary callback cannot set this envelope.
    """
    if (
        isinstance(precision_bits, bool)
        or not isinstance(precision_bits, int)
        or not 64 <= precision_bits <= 4096
    ):
        raise ValueError(
            "Declare a finite Arb coefficient precision between64 and4096bits"
        )
    interval = bounds(q_interval)
    nominal = OrbitalModelBounds(
        eccentricity=bounds(nominal_parameters["eccentricity"]),
        alpha=bounds(nominal_parameters["alpha"]),
        beta=bounds(nominal_parameters["beta"]),
        semi_major_axis=nominal_parameters["semi_major_axis"],
        mu=nominal_parameters["mu"],
        theta0=bounds(nominal_parameters.get("theta0", 0)),
    )
    import flint

    with _ARB_LOCK, flint.ctx.workprec(precision_bits):
        ar = _Arithmetic(flint)
        matrix, drift, _ = _orbital_coefficients(ar.interval(interval), nominal, ar)
        acceleration = ar.box(tuple(value for row in matrix[3:] for value in row))
        disturbance = ar.box(tuple(drift[3:]))
    return NominalCoefficientBounds(
        interval, nominal, acceleration, disturbance, precision_bits
    )


@dataclass(frozen=True)
class TargetEphemerisCertificate(FlowCertificate):
    """FlowCertificate for the target ephemeris, with physical-time [s] boxes."""

    physical_time_initial: IntervalBox | None = None
    physical_time_endpoint: IntervalBox | None = None
    physical_time_swept: IntervalBox | None = None
    segment_time_boxes: tuple = ()

    def as_dict(self, *, include_segments=True):
        """Return the FlowCertificate record plus the three physical-time boxes.

        Inputs: include_segments adds segment_time_boxes to the record when true.
        Unset time boxes are exported as None.
        """
        result = super().as_dict(include_segments=include_segments)
        result.update(
            physical_time_initial=None
            if self.physical_time_initial is None
            else self.physical_time_initial.as_dict(),
            physical_time_endpoint=None
            if self.physical_time_endpoint is None
            else self.physical_time_endpoint.as_dict(),
            physical_time_swept=None
            if self.physical_time_swept is None
            else self.physical_time_swept.as_dict(),
        )
        if include_segments:
            result["segment_time_boxes"] = tuple(
                box.as_dict() for box in self.segment_time_boxes
            )
        return result


@dataclass(frozen=True)
class RegionalResidualCertificate:
    """Status and Arb component box of the regional nonlinear-minus-CH residual."""

    status: EnclosureStatus
    reason: str
    component_bounds: IntervalBox | None
    context: dict

    @property
    def proven(self):
        """Return True when the status is EnclosureStatus.PROVEN."""
        return self.status == EnclosureStatus.PROVEN

    @property
    def eta_component_box(self):
        """Return component_bounds, which is None when no box was enclosed."""
        return self.component_bounds

    @property
    def upper_bounds(self):
        """Return max(|lower|, |upper|) per component of the enclosed residual box.

        Outputs: raises ValueError when the certificate is not proven.
        """
        if not self.proven:
            raise ValueError("Unresolved residual has no usable forcing allowance")
        return tuple(
            max(abs(lo), abs(hi))
            for lo, hi in zip(self.component_bounds.lower, self.component_bounds.upper)
        )

    @property
    def fingerprint(self):
        """Return the fingerprint of the as_dict() record."""
        return fingerprint(self.as_dict())

    def as_dict(self):
        """Return a serializable record of status, reason, bounds and context."""
        return {
            "status": self.status.value,
            "reason": self.reason,
            "component_bounds": None
            if self.component_bounds is None
            else self.component_bounds.as_dict(),
            "context": self.context,
        }


def _region(state_box, model):
    """Require a six-state box to lie inside the model's componentwise domain.

    Inputs: state_box holds position [m] then velocity [m/rad]; the limits are
    model.relative_position_limit_m and model.relative_velocity_limit_m_per_rad.
    Outputs: raises ValueError if the dimension is not 6 or any bound exceeds
    its symmetric limit.
    """
    if state_box.dimension != 6:
        raise ValueError("Relative regional domain requires six states")
    limits = (model.relative_position_limit_m,) * 3 + (
        model.relative_velocity_limit_m_per_rad,
    ) * 3
    if any(
        lo < -limit or hi > limit
        for lo, hi, limit in zip(state_box.lower, state_box.upper, limits)
    ):
        raise ValueError(
            "Complete relative region escapes its declared componentwise domain"
        )


def _target_box(flow, interval):
    """Return the hull of target swept states whose segments meet an anomaly interval.

    Inputs: flow is a proven TargetEphemerisCertificate; interval is the
    (lower, upper) anomaly range queried.
    Outputs: an IntervalBox. Raises ValueError if flow is not a proven target
    certificate, the interval leaves [start_anomaly[0], start_anomaly[1] +
    hold_width[1]], or no segment covers it while hold_width[1] is nonzero. With
    no covering segment and zero hold width, flow.initial_state is returned.
    """
    if not flow.proven or not isinstance(flow, TargetEphemerisCertificate):
        raise ValueError("A proven nonlinear target ephemeris certificate is required")
    lo, hi = interval
    if lo < flow.start_anomaly[0] or hi > flow.start_anomaly[1] + flow.hold_width[1]:
        raise ValueError("Relative query escapes target ephemeris clock coverage")
    selected = [
        segment.swept_state
        for segment in flow.segments
        if segment.anomaly[0] <= hi and segment.anomaly[1] >= lo
    ]
    if not selected:
        if flow.hold_width[1] == 0:
            return flow.initial_state
        raise ValueError("Target ephemeris has no covering continuous segment")
    result = selected[0]
    for box in selected[1:]:
        result = result.hull(box)
    return result


class NonlinearFlowEncloser:
    """Compute Arb Picard inclusions for the target ephemeris and the relative plant."""

    def __init__(self, config=None):
        """Store the reachability config, defaulting to ReachabilityConfig()."""
        self.config = config or ReachabilityConfig()

    @staticmethod
    def identity(flow):
        """Return the plant, target model, clock and target flow fingerprints of a flow.

        Each value is read from flow.model_bounds and is None when absent.
        """
        return {
            name: flow.model_bounds.get(name)
            for name in (
                "plant_fingerprint",
                "target_model_fingerprint",
                "clock_fingerprint",
                "target_flow_fingerprint",
            )
        }

    def enclose_target(
        self, initial_state, q_start, q_width, model, clock, initial_time_s=(0, 0)
    ):
        """Enclose the target state and physical time over a held anomaly span.

        Inputs: initial_state is a six-state IntervalBox; q_start and q_width give
        the start anomaly and hold width; initial_time_s is the physical time [s]
        as an IntervalBox or bounds pair.
        Outputs: a TargetEphemerisCertificate whose status is PROVEN, UNRESOLVED or
        UNAVAILABLE (see _run). Raises ValueError if the state is not six-dimensional,
        the time is not one-dimensional or the width lower bound is negative.
        """
        start, width = bounds(q_start), bounds(q_width)
        time = (
            initial_time_s
            if isinstance(initial_time_s, IntervalBox)
            else IntervalBox(*zip(bounds(initial_time_s)))
        )
        if initial_state.dimension != 6 or time.dimension != 1 or width[0] < 0:
            raise ValueError(
                "Target ephemeris needs six states, one time and nonnegative width"
            )
        context = {
            "plant": model.as_dict(),
            "target_model_fingerprint": model.target_fingerprint,
            "clock": clock.as_dict(),
            "clock_fingerprint": clock.fingerprint,
            "backend_source_sha256": _SOURCE_SHA256,
        }
        request = {
            "initial_state": initial_state.as_dict(),
            "start_anomaly": tuple(map(str, start)),
            "hold_width": tuple(map(str, width)),
            "physical_time_initial": time.as_dict(),
        }
        common = {
            "start_anomaly": start,
            "hold_width": width,
            "initial_state": initial_state,
            "frame": "inertial-cartesian-si-v1",
            "model": "nonlinear-target-ephemeris-v1",
            "model_bounds": context,
            "request": request,
            "precision_bits": self.config.precision_bits,
        }

        def field(q_bounds, state, ar):
            """Return target state derivatives in q plus dt/dq = 1/clock rate."""
            q = ar.interval(q_bounds)
            return target_field(q, state[:6], model, clock, ar) + (
                1 / clock.evaluate(q, ar)[1],
            )

        initial = IntervalBox(
            initial_state.lower + time.lower, initial_state.upper + time.upper
        )
        return self._run(
            initial,
            start,
            width,
            field,
            model,
            clock,
            common,
            target=True,
            time_initial=time,
        )

    def enclose_relative(self, request, target_flow, model, clock):
        """Enclose the six-state relative flow for a held input and anomaly span.

        Inputs: request is a HeldInputRequest; target_flow supplies the target
        ephemeris; model is the plant; clock is the reference clock.
        Outputs: a FlowCertificate whose status is PROVEN, UNRESOLVED or UNAVAILABLE.
        Raises TypeError if request is not a HeldInputRequest.
        """
        if not isinstance(request, HeldInputRequest):
            raise TypeError("Use a HeldInputRequest for the six-state relative core")
        context = {
            "plant": model.as_dict(),
            "plant_fingerprint": model.fingerprint,
            "backend_source_sha256": _SOURCE_SHA256,
            "target_model_fingerprint": model.target_fingerprint,
            "clock": clock.as_dict(),
            "clock_fingerprint": clock.fingerprint,
            "target_flow_fingerprint": target_flow.fingerprint,
            "actuation": "held-q-command; inertial acceleration=nu_ref^2*C_ref*u; additive error in q acceleration",
        }
        common = {
            "start_anomaly": request.start_anomaly,
            "hold_width": request.hold_width,
            "initial_state": request.initial_state,
            "frame": FRAME,
            "model": MODEL,
            "model_bounds": context,
            "request": request.as_dict(),
            "precision_bits": self.config.precision_bits,
        }

        def field(q_bounds, state, ar):
            """Return relative derivatives in q, adding the acceleration disturbance.

            Raises ValueError if the state box leaves the model domain or the target
            flow's model or clock fingerprint differs from the model's or clock's.
            """
            _region(ar.box(state), model)
            if (
                target_flow.model_bounds.get("target_model_fingerprint")
                != model.target_fingerprint
                or target_flow.model_bounds.get("clock_fingerprint")
                != clock.fingerprint
            ):
                raise ValueError("Target ephemeris model or clock identity changed")
            target = ar.vector(_target_box(target_flow, q_bounds))
            rhs = relative_field(
                ar.interval(q_bounds),
                state,
                target,
                ar.vector(request.held_input),
                model,
                clock,
                ar,
            )
            disturbance = ar.vector(request.acceleration_disturbance)
            return rhs[:3] + tuple(a + b for a, b in zip(rhs[3:], disturbance))

        return self._run(
            request.initial_state,
            request.start_anomaly,
            request.hold_width,
            field,
            model,
            clock,
            common,
        )

    def _run(
        self,
        initial,
        start,
        width,
        field,
        model,
        clock,
        common,
        target=False,
        time_initial=None,
    ):
        """Run the Picard integration and return a certificate instead of raising.

        Returns UNAVAILABLE if flint cannot be imported, and UNRESOLVED if the clock
        domain check or integration raises ValueError, ArithmeticError or
        _ProofUnresolved. Integration runs under the Arb lock at the configured
        precision. When target is true the certificate is a TargetEphemerisCertificate.
        """
        certificate = TargetEphemerisCertificate if target else FlowCertificate
        extra = {"physical_time_initial": time_initial} if target else {}
        try:
            import flint
        except ImportError as error:
            return certificate(
                EnclosureStatus.UNAVAILABLE, str(error), **common, **extra
            )
        common["backend_version"] = getattr(flint, "__version__", "unknown")
        try:
            clock.check_domain((start[0], start[1] + width[1]))
            with _ARB_LOCK, flint.ctx.workprec(self.config.precision_bits):
                return self._integrate(
                    initial,
                    start,
                    width,
                    field,
                    clock,
                    _Arithmetic(flint),
                    common,
                    target,
                    time_initial,
                )
        except (ValueError, ArithmeticError, _ProofUnresolved) as error:
            return certificate(
                EnclosureStatus.UNRESOLVED, str(error), **common, **extra
            )

    def _integrate(
        self, initial, start, hold, field, clock, ar, common, target, time_initial
    ):
        """Advance the state enclosure over the hold with Picard-validated substeps.

        Each substep starts at max_step, halves down to min_step until Picard
        self-inclusion holds, and doubles after success. Raises _ProofUnresolved if
        max_substeps is exhausted or no step width at least min_step validates. A
        zero-width hold only evaluates the field at the start and returns the initial
        state. For a nonzero hold width, the endpoint hulls the states reached at
        hold_width[0] through hold_width[1].
        """
        cfg, final = self.config, hold[1]
        if final == 0:
            # Even a zero-width request must have a valid physical domain.
            field(start, ar.vector(initial), ar)
            extra = (
                {
                    "physical_time_initial": time_initial,
                    "physical_time_endpoint": time_initial,
                    "physical_time_swept": time_initial,
                }
                if target
                else {}
            )
            cls = TargetEphemerisCertificate if target else FlowCertificate
            return cls(
                EnclosureStatus.PROVEN,
                "Zero-width exact initial enclosure",
                endpoint=common["initial_state"],
                swept_state=common["initial_state"],
                elapsed_seconds=IntervalBox.point((0,)),
                **common,
                **extra,
            )
        current, offset = ar.vector(initial), Fraction(0)
        proposed = min(cfg.max_step, final)
        segments, time_segments, swept, terminal = [], [], None, None
        physical_velocity, elapsed_lo, elapsed_hi = None, Fraction(0), Fraction(0)
        terminal_time = None
        while offset < final:
            if len(segments) >= cfg.max_substeps:
                raise _ProofUnresolved("Nonlinear inclusion exhausted max_substeps")
            remaining = final - offset
            ratio = remaining / proposed
            width = remaining / (
                (ratio.numerator + ratio.denominator - 1) // ratio.denominator
            )
            witness = None
            while width >= cfg.min_step:
                q_bounds = (start[0] + offset, start[1] + offset + width)
                try:
                    witness = self._picard(current, q_bounds, width, field, ar)
                except (ValueError, ArithmeticError):
                    witness = None
                if witness is not None:
                    break
                width /= 2
            if witness is None:
                raise _ProofUnresolved(
                    "Nonlinear Picard self-inclusion or physical domain unresolved at min_step"
                )
            candidate, image, derivative, endpoint, iterations = witness
            boxes = tuple(
                ar.box(value[:6])
                for value in (current, candidate, endpoint, image, derivative)
            )
            rate = clock.evaluate(ar.interval(q_bounds), ar)[1]
            segment = SweptSegment(
                (offset, offset + width), q_bounds, *boxes, iterations, ar.box((rate,))
            )
            segments.append(segment)
            swept = boxes[1] if swept is None else swept.hull(boxes[1])
            dt = ar.box((ar.scalar(width) / rate,))
            elapsed_lo += dt.lower[0]
            elapsed_hi += dt.upper[0]
            if target:
                time_segments.append(ar.box(candidate[6:7]))
            if hold[0] != hold[1] and offset + width >= hold[0]:
                stop = ar.interval((max(Fraction(0), hold[0] - offset), width))
                stop_state = ar.box(
                    tuple(x + stop * f for x, f in zip(current[:6], derivative[:6]))
                )
                terminal = stop_state if terminal is None else terminal.hull(stop_state)
                if target:
                    stopped_time = ar.box((current[6] + stop * derivative[6],))
                    terminal_time = (
                        stopped_time
                        if terminal_time is None
                        else terminal_time.hull(stopped_time)
                    )
            if not target:
                velocity = ar.box(tuple(v * rate for v in candidate[3:6]))
                physical_velocity = (
                    velocity
                    if physical_velocity is None
                    else physical_velocity.hull(velocity)
                )
            current, offset = endpoint, offset + width
            proposed = min(cfg.max_step, 2 * width)
        if hold[0] == hold[1]:
            terminal = ar.box(current[:6])
            if target:
                terminal_time = ar.box(current[6:7])
        cls = TargetEphemerisCertificate if target else FlowCertificate
        extras = (
            {
                "physical_time_initial": time_initial,
                "physical_time_endpoint": terminal_time,
                "physical_time_swept": IntervalBox(
                    (
                        min(
                            time_initial.lower[0],
                            min(b.lower[0] for b in time_segments),
                        ),
                    ),
                    (max(b.upper[0] for b in time_segments),),
                ),
                "segment_time_boxes": tuple(time_segments),
            }
            if target
            else {}
        )
        return cls(
            EnclosureStatus.PROVEN,
            "All nonlinear substeps satisfy strict Arb Picard self-inclusion",
            endpoint=terminal,
            swept_state=swept,
            segments=tuple(segments),
            swept_physical_velocity=physical_velocity,
            elapsed_seconds=IntervalBox(
                (elapsed_lo if hold[0] == hold[1] else 0,), (elapsed_hi,)
            ),
            **common,
            **extras,
        )

    def _picard(self, initial, q_bounds, width, field, ar):
        """Return a strict Picard self-inclusion witness for one substep, or None.

        Inputs: initial is the state box at the substep start; q_bounds is the
        anomaly interval; width is the substep length in anomaly.
        Outputs: (candidate, image, derivative, endpoint, iterations) when every
        image component lies strictly inside the candidate within
        max_picard_iterations; None if any value is non-finite or no inclusion is
        reached. ValueError and ArithmeticError from field() propagate.
        """
        offsets, h = ar.interval((0, width)), ar.scalar(width)
        first = field(q_bounds, initial, ar)
        if not all(v.is_finite() for v in first):
            return None
        candidate = ar.inflate(
            tuple(x.union(x + offsets * f) for x, f in zip(initial, first)), self.config
        )
        for iteration in range(1, self.config.max_picard_iterations + 1):
            derivative = field(q_bounds, candidate, ar)
            image = tuple(x + offsets * f for x, f in zip(initial, derivative))
            if not all(v.is_finite() for v in image):
                return None
            if all(y.contains_interior(value) for y, value in zip(candidate, image)):
                endpoint = tuple(x + h * f for x, f in zip(initial, derivative))
                return candidate, image, derivative, endpoint, iteration
            candidate = ar.inflate(
                tuple(y.union(value) for y, value in zip(candidate, image)), self.config
            )
        return None

    def bound_residual(
        self,
        q_interval,
        target_flow,
        regional_state_box,
        input_box,
        nominal_parameters,
        model,
        clock,
    ):
        """Enclose the nonlinear-minus-CH acceleration residual over a state box.

        Inputs: q_interval is the anomaly interval; target_flow is a proven target
        ephemeris; regional_state_box is the six-state region; input_box is the
        three-component input (checked for dimension only); nominal_parameters
        defines the CH model.
        Outputs: a RegionalResidualCertificate, UNAVAILABLE without flint and
        UNRESOLVED on a failed check or enclosure.
        """
        interval = bounds(q_interval)
        context = {
            "q_interval": tuple(map(str, interval)),
            "regional_state_box": regional_state_box.as_dict(),
            "backend_source_sha256": _SOURCE_SHA256,
            "input_box": input_box.as_dict(),
            "nominal_parameters": {k: str(v) for k, v in nominal_parameters.items()},
            "plant_fingerprint": model.fingerprint,
            "clock_fingerprint": clock.fingerprint,
            "target_flow_fingerprint": target_flow.fingerprint,
        }
        try:
            import flint
        except ImportError as error:
            return RegionalResidualCertificate(
                EnclosureStatus.UNAVAILABLE, str(error), None, context
            )
        try:
            _region(regional_state_box, model)
            clock.check_domain(interval)
            if input_box.dimension != 3:
                raise ValueError("Residual input has three components")
            if (
                target_flow.model_bounds.get("target_model_fingerprint")
                != model.target_fingerprint
                or target_flow.model_bounds.get("clock_fingerprint")
                != clock.fingerprint
            ):
                raise ValueError("Residual target model/clock identity mismatch")
            nominal = OrbitalModelBounds(
                eccentricity=bounds(nominal_parameters["eccentricity"]),
                alpha=bounds(nominal_parameters["alpha"]),
                beta=bounds(nominal_parameters["beta"]),
                semi_major_axis=nominal_parameters.get(
                    "semi_major_axis", clock.semi_major_axis_m
                ),
                mu=nominal_parameters.get("mu", clock.mu_m3_s2),
                theta0=bounds(nominal_parameters.get("theta0", 0)),
            )
            with _ARB_LOCK, flint.ctx.workprec(self.config.precision_bits):
                ar = _Arithmetic(flint)
                state, q = ar.vector(regional_state_box), ar.interval(interval)
                # Cancel common commanded input, Coriolis and centrifugal
                # terms algebraically before interval evaluation.
                rotation, rate, kappa = clock.evaluate(q, ar)
                p, w = state[:3], state[3:]
                zp = z(p)
                delta_r = matvec(rotation, p)
                delta_v = matvec(rotation, tuple(rate * (a + b) for a, b in zip(w, zp)))
                delta_a = differential_acceleration(
                    ar.vector(_target_box(target_flow, interval)),
                    delta_r,
                    delta_v,
                    model,
                    parameters(model, ar),
                    ar,
                )
                normalized = matvec(transpose(rotation), delta_a)
                matrix, drift, _ = _orbital_coefficients(q, nominal, ar)
                curvature, damping = -matrix[5][2], matrix[5][5]
                gravity_linear = (
                    -curvature * p[0],
                    2 * curvature * p[1],
                    -curvature * p[2],
                )
                difference = tuple(
                    a / rate**2 - g + (-kappa - damping) * (v + b) - d
                    for a, g, v, b, d in zip(
                        normalized, gravity_linear, w, zp, drift[3:]
                    )
                )
                box = ar.box(difference)
            return RegionalResidualCertificate(
                EnclosureStatus.PROVEN,
                "Full regional nonlinear-minus-CH expression enclosed with Arb; command cancels exactly",
                box,
                context,
            )
        except (ValueError, ArithmeticError, _ProofUnresolved) as error:
            return RegionalResidualCertificate(
                EnclosureStatus.UNRESOLVED, str(error), None, context
            )
