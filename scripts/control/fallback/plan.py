"""Complete finite feedback plans, integrity and remaining resources."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
from typing import Any, Sequence
from orbexa.control.certification.reachability import _exact
from orbexa.control.fallback.refuge import RefugeController, exact_record, fingerprint
from orbexa.control.fallback.common import _pairs, _same_plant
from orbexa.control.fallback.polynomial import PolynomialReferenceSource, _reference_methods
from orbexa.control.fallback.geometry import FleetGeometryCertificate, FleetGeometryOracle


FLEET_FALLBACK_METHOD = "exact-finite-fleet-feedback-fallback-v1"


@dataclass(frozen=True)
class FleetFeedbackSegment:
    controllers: Any

    def __post_init__(self):
        values = _pairs(self.controllers, "feedback controllers")
        if any(not isinstance(c, RefugeController) or not c.certificate.admissible for _, c in values):
            raise TypeError("Every fleet segment needs executable recertified feedback controllers")
        object.__setattr__(self, "controllers", values)
        first = values[0][1].certificate.spec
        if any(c.certificate.spec.start != first.start or c.certificate.spec.end != first.end for _, c in values):
            raise ValueError("Fleet controllers must share this segment's exact scope")

    @property
    def start(self):
        return self.controllers[0][1].certificate.spec.start

    @property
    def end(self):
        return self.controllers[0][1].certificate.spec.end

    @property
    def ids(self):
        return tuple(key for key, _ in self.controllers)

    @property
    def certificate_ids(self):
        return tuple(c.certificate.fingerprint for _, c in self.controllers)


@dataclass(frozen=True)
class FleetFallbackCertificate:
    start: Fraction
    end: Fraction
    controller_ids: tuple[tuple[str, ...], ...]
    geometry: tuple[FleetGeometryCertificate, ...]
    required_fuel: tuple[tuple[str, Fraction], ...]
    terminal_holding_start: Fraction
    terminal_reference_ids: tuple[tuple[str, str], ...]
    reasons: tuple[str, ...]
    source_plan_id: str | None = None
    method: str = FLEET_FALLBACK_METHOD

    @property
    def admissible(self):
        return self.method == FLEET_FALLBACK_METHOD and not self.reasons and all(p.admissible for p in self.geometry)

    @cached_property
    def fingerprint(self):
        # Composition binds every complete immutable geometry proof without
        # serializing thousands of exact swept records again for each suffix.
        # The original proofs remain available in as_dict for full evidence.
        return fingerprint(dict(start=self.start,end=self.end,controller_ids=self.controller_ids,
            geometry_ids=tuple(proof.fingerprint for proof in self.geometry),
            required_fuel=self.required_fuel,terminal_holding_start=self.terminal_holding_start,
            terminal_reference_ids=self.terminal_reference_ids,reasons=self.reasons,
            source_plan_id=self.source_plan_id,method=self.method))

    def as_dict(self):
        return dict(status="CONDITIONAL_PROVEN" if self.admissible else "REFUSED",
                    identity=self.fingerprint, **exact_record(self))


class FleetFallbackPlan:
    """Complete finite feedback route followed by a proved holding interval.

    Segment references describe one continuous C2 route per member. Boundary
    changes in tube/gains need exact inclusion of the previous (e,s) set. A
    different route needs a replacement plan and whole-endpoint admission.
    """
    def __init__(self, segments: Sequence[FleetFeedbackSegment], geometry: FleetGeometryOracle):
        self.segments = tuple(segments)
        if not self.segments or not isinstance(geometry, FleetGeometryOracle):
            raise TypeError("A nonempty finite feedback route and shared geometry oracle are required")
        if any(not isinstance(s, FleetFeedbackSegment) for s in self.segments):
            raise TypeError("Every route segment must be typed")
        self.geometry_oracle = geometry
        ids, reasons, proofs = self.segments[0].ids, [], []
        if geometry.ids != ids:
            raise ValueError("Feedback and geometry fleet order must agree")
        previous = None
        source_guards=[]
        for segment in self.segments:
            if segment.ids != ids:
                raise ValueError("Feedback fleet identity/order changed")
            for _, controller in segment.controllers:
                ctx = controller.certificate.spec.context
                if (ctx.geometry_id != geometry.geometry_id or ctx.target_model_id != geometry.target_model_id
                        or ctx.frame != geometry.frame or ctx.clock_id != geometry.clock_id):
                    raise ValueError("Feedback and whole-fleet geometry context differ")
            if previous is not None:
                if previous.end != segment.start:
                    reasons.append("Fallback route has a clock gap or overlap")
                for (key, old), (_, new) in zip(previous.controllers, segment.controllers):
                    a, b = old.certificate.spec, new.certificate.spec
                    if not _same_plant(a.context, b.context) or a.context.reference_id != b.context.reference_id:
                        reasons.append(f"{key}: route model/sensor/reference identity changed")
                    if (any(o > n for o, n in zip(a.omega, b.omega)) or
                            any(p+abs(ln-lo)*o > pn for p, ln, lo, o, pn in
                                zip(a.phi, b.lambda_gain, a.lambda_gain, a.omega, b.phi))):
                        reasons.append(f"{key}: next feedback region does not contain the previous invariant tube")
                    if old.reference_at(segment.start) != new.reference_at(segment.start):
                        reasons.append(f"{key}: reference jets differ at a feedback boundary")
            proof = geometry.feedback_regions(dict(segment.controllers),segment.start,segment.end)
            for key,source in proof.reference_sources:
                controller=dict(segment.controllers)[key]
                provider=controller.builder.reference_provider
                source_guards.append((controller,provider,source.fingerprint,_reference_methods(provider)))
            proofs.append(proof)
            reasons.extend(proof.reasons)
            previous = segment
        last = self.segments[-1]
        for key, c in last.controllers:
            if not c.certificate.reference_bounds.is_constant:
                reasons.append(f"{key}: fallback has no constant-reference terminal holding region")
            if last.end-last.start < c.certificate.spec.maximum_hold:
                reasons.append(f"{key}: terminal holding scope does not cover one complete declared hold")
        required = self.remaining_fuel(self.segments[0].start)
        for key, value in required.items():
            reserves = [dict(s.controllers)[key].certificate.spec.available_fuel for s in self.segments]
            if value > min(reserves):
                reasons.append(f"{key}: route and terminal holding exceed the fuel reserve")
        self.certificate = FleetFallbackCertificate(self.segments[0].start, last.end,
            tuple(s.certificate_ids for s in self.segments), tuple(proofs), tuple(required.items()), last.start,
            tuple((key, c.certificate.spec.context.reference_id) for key, c in last.controllers), tuple(reasons))
        self._identity = self.certificate.fingerprint
        self._geometry_identity = geometry.fingerprint
        self._polynomial_source_guards=tuple(source_guards)

    @property
    def ids(self):
        return self.segments[0].ids

    @property
    def fingerprint(self):
        self.verify_integrity()
        return self._identity

    def verify_integrity(self):
        if (self.certificate.fingerprint != self._identity or tuple(s.certificate_ids for s in self.segments)
                != self.certificate.controller_ids or self.geometry_oracle.fingerprint != self._geometry_identity):
            raise ValueError("Installed feedback bounds were altered after admission")
        checked={}
        for controller,provider,identity,methods in self._polynomial_source_guards:
            if (controller.builder.reference_provider is not provider
                    or _reference_methods(provider)!=methods):
                raise ValueError("Installed polynomial reference provider changed after admission")
            key=id(provider)
            if key not in checked:
                source=provider.polynomial_source()
                checked[key]=source.fingerprint if type(source) is PolynomialReferenceSource else None
            if checked[key]!=identity:
                raise ValueError("Installed polynomial reference coefficients/identities changed after admission")

    def segment_at(self, q, hold_width=0):
        q, h = _exact(q), _exact(hold_width)
        if h < 0:
            raise ValueError("Negative fallback hold")
        selected = next((s for s in self.segments if s.start <= q < s.end), None)
        if selected is None and h == 0 and q == self.segments[-1].end:
            selected = self.segments[-1]
        if selected is None or q+h > selected.end or any(h > c.certificate.spec.maximum_hold for _, c in selected.controllers):
            raise ValueError("Complete hold must fit one installed finite feedback segment")
        return selected

    def contains(self, state_boxes, q, hold_width=0):
        if not self.certificate.admissible or set(state_boxes) != set(self.ids):
            return False
        self.verify_integrity()
        segment = self.segment_at(q, hold_width)
        return all(c.contains(state_boxes[key], q, hold_width) for key, c in segment.controllers)

    def remaining_fuel(self, q):
        q = _exact(q)
        if not getattr(self, "_view_start", self.segments[0].start) <= q <= self.segments[-1].end:
            raise ValueError("Fuel query is outside the finite fallback route")
        return {key: sum(max(Fraction(0), s.end-max(q, s.start)) *
            dict(s.controllers)[key].certificate.spec.fuel_rate_upper_per_q for s in self.segments)
            for key in self.ids}

    def suffix(self, start):
        """Restrict a proved route's clock domain without rebuilding bounds.

        Retains the original global reference/drift/geometry proofs on a
        superset of the new scope. No reference, tube or plant changes. Fresh
        whole-endpoint admission is still mandatory before switching to it.
        """
        self.verify_integrity()
        start = _exact(start)
        if not self.certificate.admissible or not self.certificate.start <= start < self.certificate.end:
            raise ValueError("A suffix needs an admitted nonempty remaining finite domain")
        selected = tuple((s,p) for s,p in zip(self.segments,self.certificate.geometry) if s.end > start)
        value = object.__new__(FleetFallbackPlan)
        value.segments = tuple(s for s,_ in selected)
        value.geometry_oracle = self.geometry_oracle
        value._view_start = start
        value.certificate = FleetFallbackCertificate(start=start,end=self.certificate.end,
            controller_ids=tuple(s.certificate_ids for s,_ in selected),
            geometry=tuple(p for _,p in selected),
            required_fuel=tuple(self.remaining_fuel(start).items()),
            terminal_holding_start=max(start,self.certificate.terminal_holding_start),
            terminal_reference_ids=self.certificate.terminal_reference_ids,reasons=(),source_plan_id=self.fingerprint)
        value._identity = value.certificate.fingerprint
        value._geometry_identity = self._geometry_identity
        value._polynomial_source_guards=self._polynomial_source_guards
        return value
