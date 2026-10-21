"""Finite certified fleet feedback and transactional performance admission.

The active fallback precedes motion and ends in a finite invariant holding
segment. A failed performance proposal retains that feedback law; selecting it
needs neither a new NLP nor a new flow proof. See ``docs/fleet-fallback.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
from itertools import combinations
from math import comb
from typing import Any, Callable, Mapping, Sequence

from orbexa.control.certification.reachability import FlowCertificate, HeldInputRequest, IntervalBox, SweptCollisionGuard, _exact
from orbexa.control.fallback.refuge import ControlContext, RefugeController, RefugeObservation, _identity, _proof_source, _scope, exact_record, fingerprint

FLEET_FALLBACK_METHOD = "exact-finite-fleet-feedback-fallback-v1"
FLEET_SWEEP_METHOD = "exact-component-box-fleet-sweep-v1"
POLYNOMIAL_PEER_METHOD = "exact-same-time-polynomial-reference-peer-v1"


def _pairs(value, name):
    items = tuple(value.items()) if isinstance(value, Mapping) else tuple(value)
    if not items or any(not isinstance(key, str) or not key.strip() for key, _ in items):
        raise ValueError(f"{name} requires nonempty named fleet members")
    if len({key for key, _ in items}) != len(items):
        raise ValueError(f"{name} contains duplicate identities")
    return items


def _box_distance_squared(left, right):
    return sum(max(Fraction(0), lo2-hi1, lo1-hi2)**2
               for lo1, hi1, lo2, hi2 in zip(left.lower, left.upper, right.lower, right.upper))


def _polynomial_shift(coefficients, offset):
    if offset==0:return coefficients
    return tuple(sum(coefficients[i]*comb(i,j)*offset**(i-j)
                     for i in range(j,len(coefficients))) for j in range(len(coefficients)))


def _polynomial_range(coefficients, lower, upper):
    """Exact Bernstein convex hull, after retaining the common time variable."""
    # Shared fleet motion can cancel every nonconstant coefficient exactly.
    # Its interval is then a point; no basis shift or degree elevation remains.
    if coefficients and not any(coefficients[1:]):
        return coefficients[0], coefficients[0]
    degree=len(coefficients)-1
    shifted=coefficients if lower==0 else _polynomial_shift(coefficients,lower)
    power=tuple(value*(upper-lower)**j for j,value in enumerate(shifted))
    values=tuple(sum(power[j]*Fraction(comb(k,j),comb(degree,j)) for j in range(k+1))
                 for k in range(degree+1))
    return min(values),max(values)


@dataclass(frozen=True)
class PolynomialReferenceSpan:
    start: Any
    end: Any
    coefficients: Any

    def __post_init__(self):
        start,end=_scope(self.start,self.end)
        rows=tuple(tuple(map(_exact,row)) for row in self.coefficients)
        if len(rows)!=3 or any(not row or len(row)>6 for row in rows):
            raise ValueError("Three exact polynomials of degree at most five are required")
        object.__setattr__(self,"start",start);object.__setattr__(self,"end",end)
        object.__setattr__(self,"coefficients",rows)

    def jet(self,q,order):
        value=_exact(q)-self.start
        result=[]
        for row in self.coefficients:
            derivative=row
            for _ in range(order):
                derivative=tuple(i*c for i,c in enumerate(derivative) if i) or (Fraction(0),)
            shifted=_polynomial_shift(derivative,value)
            result.append(shifted[0])
        return tuple(result)


@dataclass(frozen=True)
class PolynomialReferenceSource:
    """Full immutable analytic source for one existing C2 reference provider.

    The coefficient digest must also match the controller's original analytic
    reference-bound proof. This source never asserts a relation between errors.
    """
    pieces: Any
    reference_id: str
    frame: str
    clock_id: str

    def __post_init__(self):
        pieces=tuple(self.pieces)
        if not pieces or any(type(piece) is not PolynomialReferenceSpan for piece in pieces):
            raise TypeError("Complete typed exact polynomial reference pieces are required")
        for old,new in zip(pieces,pieces[1:]):
            if old.end!=new.start or any(old.jet(old.end,k)!=new.jet(new.start,k) for k in range(3)):
                raise ValueError("Polynomial peer sources must cover one gap-free C2 reference")
        object.__setattr__(self,"pieces",pieces)
        for name in ("reference_id","frame","clock_id"):_identity(getattr(self,name),name)

    @cached_property
    def coefficient_fingerprint(self):return fingerprint(self.pieces)

    @cached_property
    def fingerprint(self):return fingerprint(self)

    @property
    def proof_reference(self):return "exact-quintic-Bernstein:"+self.coefficient_fingerprint

    def difference_bounds(self,other,start,end):
        start,end=_scope(start,end)
        if (type(other) is not PolynomialReferenceSource or self.frame!=other.frame
                or self.clock_id!=other.clock_id or any(source.pieces[0].start>start
                or source.pieces[-1].end<end for source in (self,other))):
            raise ValueError("Same-frame/clock polynomial sources must cover the entire peer window")
        cuts=sorted({start,end,*(v for source in (self,other) for piece in source.pieces
                               for v in (piece.start,piece.end) if start<v<end)})
        result=None
        for lower,upper in zip(cuts,cuts[1:]):
            left=next((piece for piece in self.pieces if piece.start<=lower<upper<=piece.end),None)
            right=next((piece for piece in other.pieces if piece.start<=lower<upper<=piece.end),None)
            if left is None or right is None:raise ValueError("Incomplete polynomial peer source window")
            ranges=[]
            for first,second in zip(left.coefficients,right.coefficients):
                if left.start==right.start:
                    # Cancel shared translation in the original exact basis
                    # before shifting once. The resulting Bernstein numbers
                    # are identical to shifting both polynomials separately.
                    a,b=first,second
                    local_lower,local_upper=lower-left.start,upper-left.start
                else:
                    a,b=_polynomial_shift(first,lower-left.start),_polynomial_shift(second,lower-right.start)
                    local_lower,local_upper=Fraction(0),upper-lower
                coefficients=tuple((a[i] if i<len(a) else 0)-(b[i] if i<len(b) else 0)
                                   for i in range(max(len(a),len(b))))
                ranges.append(_polynomial_range(coefficients,local_lower,local_upper))
            box=IntervalBox(tuple(v[0] for v in ranges),tuple(v[1] for v in ranges))
            result=box if result is None else result.hull(box)
        return result


@dataclass(frozen=True)
class PolynomialPeerBounds:
    first_id: str
    second_id: str
    start: Fraction
    end: Fraction
    first_controller_id: str
    second_controller_id: str
    first_reference_id: str
    second_reference_id: str
    first_source_id: str
    second_source_id: str
    first_omega: tuple
    second_omega: tuple
    reference_difference: IntervalBox
    relative_position: IntervalBox
    frame: str
    clock_id: str
    method: str = POLYNOMIAL_PEER_METHOD

    def __post_init__(self):
        start,end=_scope(self.start,self.end)
        object.__setattr__(self,"start",start);object.__setattr__(self,"end",end)
        for name in ("first_id","second_id","first_controller_id","second_controller_id",
                     "first_reference_id","second_reference_id","first_source_id","second_source_id","frame","clock_id"):
            _identity(getattr(self,name),name)
        first,second=tuple(map(_exact,self.first_omega)),tuple(map(_exact,self.second_omega))
        if (self.first_id==self.second_id or len(first)!=3 or len(second)!=3
                or any(value<0 for value in first+second)
                or any(not isinstance(box,IntervalBox) or box.dimension!=3
                       for box in (self.reference_difference,self.relative_position))
                or self.method!=POLYNOMIAL_PEER_METHOD):
            raise ValueError("Complete exact independent peer error/reference bounds are required")
        radius=tuple(a+b for a,b in zip(first,second))
        expected=IntervalBox(tuple(v-r for v,r in zip(self.reference_difference.lower,radius)),
                             tuple(v+r for v,r in zip(self.reference_difference.upper,radius)))
        if self.relative_position!=expected:raise ValueError("Peer error expansion differs from the independent Omega sum")
        object.__setattr__(self,"first_omega",first);object.__setattr__(self,"second_omega",second)


def _reference_methods(provider):
    return tuple(getattr(value,"__func__",value) for value in
                 (provider.sample,provider.bounds,provider.polynomial_source))


def _intersect(left, right):
    lower = tuple(max(a, c) for a, c in zip(left.lower, right.lower))
    upper = tuple(min(a, c) for a, c in zip(left.upper, right.upper))
    if any(a > c for a, c in zip(lower, upper)):
        raise ValueError("Observation contradicts the proved continuous endpoint")
    return IntervalBox(lower, upper)


def _freeze(value):
    if isinstance(value, Mapping):
        return tuple((str(key), _freeze(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (float, Fraction)):
        return _exact(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise TypeError("Attempt details must be immutable JSON-like evidence")


@dataclass(frozen=True)
class PerformanceAttempt:
    status: str
    message: str
    solver_id: str = "unspecified-performance-planner"
    details: Any = ()

    def __post_init__(self):
        _identity(self.status, "performance status")
        _identity(self.solver_id, "performance solver identity")
        if not isinstance(self.message, str):
            raise TypeError("Performance message must be text")
        object.__setattr__(self, "details", _freeze(self.details))


class PerformanceAdmissionRefusal(ValueError):
    """An admission failure with exact evidence independent of solver outcome."""
    def __init__(self, message, *, details=()):
        super().__init__(message)
        self.admission_details = _freeze(details)


@dataclass(frozen=True)
class OccupiedGeometryBounds:
    """Proved occupied component AABBs throughout [start,end].

    Boxes include physical dimensions, placement and motion uncertainty.
    Immutable component IDs establish complete coverage. Sampling is insufficient.
    """
    start: Any
    end: Any
    components: Any
    geometry_id: str
    target_model_id: str
    frame: str
    clock_id: str
    proof_reference: str
    assumptions: tuple[str, ...]
    support_projections: Any = ()
    proof_kind: str = "validated-interval-global"

    def __post_init__(self):
        start, end = _scope(self.start, self.end)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        components = _pairs(self.components, "occupied components")
        if any(not isinstance(box, IntervalBox) or box.dimension != 3 for _, box in components):
            raise ValueError("Every occupied component needs a complete 3D box")
        object.__setattr__(self, "components", components)
        for name in ("geometry_id", "target_model_id", "frame", "clock_id", "proof_reference"):
            _identity(getattr(self, name), name)
        _proof_source(self.proof_kind, self.assumptions)
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        projections = tuple(self.support_projections)
        for item in projections:
            if (not isinstance(item,OccupiedProjection) or item.component_id not in dict(components)):
                raise ValueError("Support projections must name complete declared occupied components")
        object.__setattr__(self,"support_projections",projections)


@dataclass(frozen=True)
class OccupiedProjection:
    component_id: str
    direction: Any
    projected_interval: IntervalBox

    def __post_init__(self):
        _identity(self.component_id,"projected component identity")
        direction = tuple(map(_exact,self.direction))
        if len(direction) != 3 or not any(direction):
            raise ValueError("A fixed nonzero exact projection direction is required")
        if not isinstance(self.projected_interval,IntervalBox) or self.projected_interval.dimension != 1:
            raise ValueError("A proved one-dimensional occupied projection is required")
        object.__setattr__(self,"direction",direction)


@dataclass(frozen=True)
class SeparationRecord:
    kind: str
    first_id: str
    second_id: str
    start: Fraction
    end: Fraction
    distance_squared_lower: Fraction
    required_squared: Fraction
    occupied_bounds_id: str


@dataclass(frozen=True)
class FleetGeometryCertificate:
    start: Fraction
    end: Fraction
    records: tuple[SeparationRecord, ...]
    object_ids: tuple[str, ...]
    component_ids: tuple[str, ...]
    geometry_id: str
    target_model_id: str
    frame: str
    clock_id: str
    body_radii: tuple[tuple[str, Fraction], ...]
    clearance: Fraction
    minimum_pair_center_distance: Fraction
    flow_fingerprints: tuple[str, ...] = ()
    region_fingerprints: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    method: str = FLEET_SWEEP_METHOD
    reference_sources: tuple[tuple[str, PolynomialReferenceSource], ...] = ()
    peer_relative_bounds: tuple[PolynomialPeerBounds, ...] = ()

    @property
    def admissible(self):
        return (self.method == FLEET_SWEEP_METHOD and not self.reasons and bool(self.records)
                and all(item.distance_squared_lower > item.required_squared for item in self.records))

    @cached_property
    def fingerprint(self):
        return fingerprint(self)

    def as_dict(self):
        return dict(status="CONDITIONAL_PROVEN" if self.admissible else "REFUSED",
                    identity=self.fingerprint, **exact_record(self))


class FleetGeometryOracle:
    """Exact conservative component/finite-body separation checker.

    occupied_provider(start,end), or its bounds method, supplies actual global
    OccupiedGeometryBounds. These proof inputs cannot be numerical predictions.
    """
    def __init__(self, occupied_provider, *, component_ids, body_radii,
                 geometry_id, target_model_id, frame, clock_id, clearance=0,
                 minimum_pair_center_distance=0):
        self.occupied_provider = occupied_provider
        self.component_ids = tuple(component_ids)
        if (not self.component_ids or len(set(self.component_ids)) != len(self.component_ids)
                or any(not isinstance(key, str) or not key.strip() for key in self.component_ids)):
            raise ValueError("Declare every occupied target component identity")
        self.body_radii = tuple((key, _exact(value)) for key, value in _pairs(body_radii, "finite bodies"))
        if any(value <= 0 for _, value in self.body_radii):
            raise ValueError("Every complete spacecraft needs a positive enclosing radius")
        self.clearance = _exact(clearance)
        self.minimum_pair_center_distance = _exact(minimum_pair_center_distance)
        if self.clearance < 0 or self.minimum_pair_center_distance < 0:
            raise ValueError("Geometry clearances must be nonnegative")
        self.geometry_id, self.target_model_id = geometry_id, target_model_id
        self.frame, self.clock_id = frame, clock_id
        for name in ("geometry_id", "target_model_id", "frame", "clock_id"):
            _identity(getattr(self, name), name)

    @property
    def ids(self):
        return tuple(key for key, _ in self.body_radii)

    @property
    def fingerprint(self):
        return fingerprint(dict(component_ids=self.component_ids, body_radii=self.body_radii,
            geometry_id=self.geometry_id, target_model_id=self.target_model_id, frame=self.frame,
            clock_id=self.clock_id, clearance=self.clearance,
            minimum_pair_center_distance=self.minimum_pair_center_distance))

    def _occupied(self, start, end):
        provider = self.occupied_provider
        value = provider(start, end) if callable(provider) else provider.bounds(start, end)
        if (not isinstance(value, OccupiedGeometryBounds) or value.start > start or value.end < end
                or value.geometry_id != self.geometry_id or value.target_model_id != self.target_model_id
                or value.frame != self.frame or value.clock_id != self.clock_id
                or set(dict(value.components)) != set(self.component_ids)):
            raise ValueError("Occupied proof has incomplete coverage or stale frame/clock/model identities")
        return value

    def _records(self, boxes, start, end, peer_relative_bounds=()):
        if set(boxes) != set(self.ids):
            raise ValueError("Swept geometry fleet changed")
        target, radii = self._occupied(start, end), dict(self.body_radii)
        bounds_id, records, reasons = fingerprint(target), [], []
        peers={(value.first_id,value.second_id):value for value in peer_relative_bounds}
        if peer_relative_bounds and (len(peers)!=len(peer_relative_bounds)
                or set(peers)!=set(combinations(self.ids,2))
                or any(type(value) is not PolynomialPeerBounds or value.start!=start or value.end!=end
                       or value.frame!=self.frame or value.clock_id!=self.clock_id
                       or value.method!=POLYNOMIAL_PEER_METHOD for value in peer_relative_bounds)):
            raise ValueError("Polynomial peer bounds must cover this complete fleet/window/frame/clock")
        for key in self.ids:
            box = boxes[key]
            if not isinstance(box, IntervalBox) or box.dimension != 3:
                raise ValueError("A complete swept position box is required per spacecraft")
            needed = (radii[key]+self.clearance)**2
            for part, occupied in target.components:
                distance = _box_distance_squared(box, occupied)
                # Every bound is independently sufficient. Once one proves
                # strict separation, further axes cannot change admission.
                # Unseparated components still exhaust all declared axes.
                if distance <= needed:
                    for projection in target.support_projections:
                        if projection.component_id == part:
                            direction = projection.direction
                            products = tuple((d*lo,d*hi) for d,lo,hi in zip(direction,box.lower,box.upper))
                            lower,upper = sum(min(a,b) for a,b in products),sum(max(a,b) for a,b in products)
                            occupied_interval = projection.projected_interval
                            gap = max(Fraction(0),lower-occupied_interval.upper[0],occupied_interval.lower[0]-upper)
                            distance = max(distance,gap*gap/sum(d*d for d in direction))
                            if distance > needed:
                                break
                records.append(SeparationRecord("target_component", key, part, start, end, distance, needed, bounds_id))
                if distance <= needed:
                    reasons.append(f"No strict whole-set separation for {key} and component {part}")
        for first, second in combinations(self.ids, 2):
            required = max(radii[first]+radii[second]+self.clearance, self.minimum_pair_center_distance)
            distance = _box_distance_squared(boxes[first], boxes[second])
            if (first,second) in peers:
                distance=max(distance,_box_distance_squared(peers[first,second].relative_position,
                                                           IntervalBox.point((0,0,0))))
            records.append(SeparationRecord("peer", first, second, start, end, distance, required**2, bounds_id))
            if distance <= required**2:
                reasons.append(f"No strict whole-set peer separation for {first}/{second}")
        return tuple(records), tuple(reasons)

    def regions(self, boxes, start, end, *, region_fingerprints):
        start, end = _scope(start, end)
        records, reasons = self._records(boxes, start, end)
        return FleetGeometryCertificate(start, end, records, self.ids, self.component_ids,
            self.geometry_id, self.target_model_id, self.frame, self.clock_id,
            self.body_radii, self.clearance, self.minimum_pair_center_distance,
            region_fingerprints=tuple(region_fingerprints), reasons=reasons)

    def feedback_regions(self, controllers, start, end):
        """Retain same-time polynomial correlation solely for feedback plans.

        Each position error remains independently inside its original Omega
        box. Actual performance FlowCertificates continue through flows().
        """
        start,end=_scope(start,end)
        if tuple(controllers)!=self.ids:raise ValueError("Feedback geometry fleet/order changed")
        boxes={key:c.certificate.state_box.slice(0,3) for key,c in controllers.items()}
        advertised={key:callable(getattr(c.builder.reference_provider,"polynomial_source",None))
                    for key,c in controllers.items()}
        if not any(advertised.values()):
            return self.regions(boxes,start,end,region_fingerprints=tuple(c.certificate.fingerprint for c in controllers.values()))
        if not all(advertised.values()):raise ValueError("Partial polynomial reference-source fleet")
        sources={}
        for key,c in controllers.items():
            spec,bounds=c.certificate.spec,c.certificate.reference_bounds
            source=c.builder.reference_provider.polynomial_source()
            if (type(source) is not PolynomialReferenceSource or source.reference_id!=spec.context.reference_id
                    or source.frame!=self.frame or source.clock_id!=self.clock_id
                    or source.proof_reference!=bounds.proof_reference
                    or not spec.start<=start<end<=spec.end
                    or source.pieces[0].start>start or source.pieces[-1].end<end):
                raise ValueError("Polynomial reference source differs from the actual controller proof/window/context")
            sources[key]=source
        peers=[]
        for first,second in combinations(self.ids,2):
            a,b=controllers[first].certificate,controllers[second].certificate
            left,right=sources[first],sources[second]
            difference=left.difference_bounds(right,start,end)
            radius=tuple(x+y for x,y in zip(a.spec.omega,b.spec.omega))
            relative=IntervalBox(tuple(v-r for v,r in zip(difference.lower,radius)),
                                 tuple(v+r for v,r in zip(difference.upper,radius)))
            peers.append(PolynomialPeerBounds(first,second,start,end,a.fingerprint,b.fingerprint,
                left.reference_id,right.reference_id,left.fingerprint,right.fingerprint,
                a.spec.omega,b.spec.omega,difference,relative,self.frame,self.clock_id))
        records,reasons=self._records(boxes,start,end,tuple(peers))
        return FleetGeometryCertificate(start,end,records,self.ids,self.component_ids,
            self.geometry_id,self.target_model_id,self.frame,self.clock_id,self.body_radii,
            self.clearance,self.minimum_pair_center_distance,
            region_fingerprints=tuple(c.certificate.fingerprint for c in controllers.values()),reasons=reasons,
            reference_sources=tuple(sources.items()),peer_relative_bounds=tuple(peers))

    def flows(self, flows):
        if set(flows) != set(self.ids):
            raise ValueError("Performance flow fleet changed")
        ordered = tuple(flows[key] for key in self.ids)
        for flow in ordered:
            if not isinstance(flow, FlowCertificate) or not flow.proven:
                raise ValueError("Every performance flow must be PROVEN")
            SweptCollisionGuard._check_flow_structure(flow)
        first = ordered[0]
        if (first.start_anomaly[0] != first.start_anomaly[1] or first.hold_width[0] != first.hold_width[1]
                or first.hold_width[0] <= 0 or any(flow.frame != self.frame or
                flow.start_anomaly != first.start_anomaly or flow.hold_width != first.hold_width for flow in ordered)):
            raise ValueError("Performance flows must share one exact frame and positive hold clock")
        start, width = first.start_anomaly[0], first.hold_width[0]
        cuts = sorted({Fraction(0), width, *(v for f in ordered for s in f.segments for v in s.offset)})
        records, reasons = [], []
        for lower, upper in zip(cuts, cuts[1:]):
            boxes = {}
            for key, flow in zip(self.ids, ordered):
                segment = next((s for s in flow.segments if s.offset[0] <= lower and upper <= s.offset[1]), None)
                if segment is None:
                    raise ValueError("Flow segments have missing common-clock coverage")
                boxes[key] = segment.swept_state.slice(0, 3)
            local_records, local_reasons = self._records(boxes, start+lower, start+upper)
            records.extend(local_records)
            reasons.extend(local_reasons)
        return FleetGeometryCertificate(start, start+width, tuple(records), self.ids, self.component_ids,
            self.geometry_id, self.target_model_id, self.frame, self.clock_id,
            self.body_radii, self.clearance, self.minimum_pair_center_distance,
            flow_fingerprints=tuple(flow.fingerprint for flow in ordered), reasons=tuple(reasons))


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


def _same_plant(left: ControlContext, right: ControlContext):
    return all(getattr(left, name) == getattr(right, name) for name in
        ("model_id", "parameter_set_id", "target_model_id", "frame", "clock_id", "sensor_id",
         "actuator_id", "geometry_id", "fuel_domain_id"))


def _require_policy_supports(current, candidate, start, acceleration_disturbance):
    """Context names alone cannot authenticate tighter physical assumptions.

    A candidate may not silently increase hardware authority or shrink the
    original sensor/actuator support. A separately authenticated calibration
    interface would be needed to allow a justified reduction in these budgets.
    Every future segment, not only its entry controller, is checked.
    """
    original = dict(current.segment_at(start).controllers)
    disturbance = tuple(max(abs(lo), abs(hi)) for lo, hi in zip(
        acceleration_disturbance.lower, acceleration_disturbance.upper))
    for segment in candidate.segments:
        for key, controller in segment.controllers:
            a, b = original[key].certificate.spec, controller.certificate.spec
            if not _same_plant(a.context, b.context):
                raise ValueError('Successor changed actual model, sensing, actuator, geometry or fuel context')
            if not a.input_limits.contains(b.input_limits):
                raise ValueError('Successor increased the original physical hardware input authority')
            if any(new < old for new, old in zip(b.actuator_error_abs, a.actuator_error_abs)):
                raise ValueError('Successor reduced the original physical actuator-error support')
            if any(new < old for new, old in zip(b.sensing_error_abs, a.sensing_error_abs)):
                raise ValueError('Successor reduced the original physical sensing-error support')
            if any(disturbance):
                verifier = getattr(controller.builder.drift_bounds_provider, 'verify_certificate_source', None)
                if not callable(verifier) or verifier(controller.certificate, disturbance) is not True:
                    raise ValueError('Successor lacks a source-authenticated nonzero acceleration-disturbance bound')


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


@dataclass(frozen=True)
class PerformanceProposal:
    start: Any
    hold_width: Any
    applied_inputs: Any
    successor: FleetFallbackPlan
    observation_ids: Any
    context_ids: Any
    attempt: PerformanceAttempt

    def __post_init__(self):
        q, h = _exact(self.start), _exact(self.hold_width)
        if h <= 0:
            raise ValueError("A positive performance hold is required")
        object.__setattr__(self, "start", q)
        object.__setattr__(self, "hold_width", h)
        inputs = _pairs(self.applied_inputs, "performance held inputs")
        if any(not isinstance(box, IntervalBox) or box.dimension != 3 for _, box in inputs):
            raise ValueError("Performance inputs need applied three-input uncertainty boxes")
        object.__setattr__(self, "applied_inputs", inputs)
        object.__setattr__(self, "observation_ids", _pairs(self.observation_ids, "proposal observations"))
        object.__setattr__(self, "context_ids", _pairs(self.context_ids, "proposal contexts"))
        if not isinstance(self.successor, FleetFallbackPlan) or not isinstance(self.attempt, PerformanceAttempt):
            raise TypeError("Proposals require a prepared successor and retained attempt")

    @property
    def fingerprint(self):
        return fingerprint(dict(start=self.start, hold_width=self.hold_width, applied_inputs=self.applied_inputs,
            successor=self.successor.fingerprint, observation_ids=self.observation_ids,
            context_ids=self.context_ids, attempt=self.attempt))


@dataclass(frozen=True)
class HeldFleetDecision:
    start: Fraction
    hold_width: Fraction
    source: str
    reason: str
    commands: tuple[tuple[str, tuple[Fraction, ...]], ...]
    applied_inputs: tuple[tuple[str, IntervalBox], ...]
    attempt: PerformanceAttempt
    active_fallback_id: str
    successor_fallback_id: str
    context_ids: tuple[tuple[str, str], ...]
    observation_ids: tuple[tuple[str, str], ...]
    charged_fuel: tuple[tuple[str, Fraction], ...]
    proposal_id: str | None = None
    flow_fingerprints: tuple[str, ...] = ()
    geometry: FleetGeometryCertificate | None = None

    @property
    def fingerprint(self):
        return fingerprint(self)

    def float_commands(self):
        return {key: tuple(float(value) for value in command) for key, command in self.commands}

    def as_dict(self):
        return dict(identity=self.fingerprint, **exact_record(self))


@dataclass(frozen=True)
class CommandResourceBound:
    start: Any
    hold_width: Any
    applied_input: IntervalBox
    fuel_upper_kg: Any
    force_margin_n: Any
    fuel_domain_id: str
    proof_reference: str

    def __post_init__(self):
        for name in ("start", "hold_width", "fuel_upper_kg", "force_margin_n"):
            object.__setattr__(self, name, _exact(getattr(self, name)))
        if self.hold_width <= 0 or self.fuel_upper_kg < 0 or not isinstance(self.applied_input, IntervalBox):
            raise ValueError("A finite, nonnegative command resource proof is required")
        _identity(self.fuel_domain_id, "fuel domain identity")
        _identity(self.proof_reference, "command resource proof reference")


class FleetAdmissionRuntime:
    """Own the admitted fallback, exact clock, fuel and pending fleet decision.

    flow_provider(chaser_id, HeldInputRequest) returns a genuine FlowCertificate.
    flow_identity(flow) supplies plant_fingerprint, target_model_fingerprint,
    clock_fingerprint and target_flow_fingerprint. Its default reads model_bounds.
    A scheduler must invoke the ready decision independently of a late planner;
    these classes do not implement physical actuator deadlines.
    """
    def __init__(self, active_fallback: FleetFallbackPlan, initial_observations,
                 *, flow_provider: Callable, remaining_fuel=None, flow_identity: Callable | None = None,
                 acceleration_disturbance: IntervalBox | None = None, performance_resource_bound: Callable | None = None):
        if not isinstance(active_fallback, FleetFallbackPlan) or not active_fallback.certificate.admissible:
            raise ValueError("A complete proved fleet fallback is required before motion")
        if not callable(flow_provider):
            raise TypeError("An actual nonlinear flow proof provider is required")
        active_fallback.verify_integrity()
        q = active_fallback.certificate.start
        if set(initial_observations) != set(active_fallback.ids):
            raise ValueError("Initial observed fleet differs from the complete fallback")
        boxes = {key: value.state_enclosure for key, value in initial_observations.items()}
        if not active_fallback.contains(boxes, q):
            raise ValueError("Complete initial state enclosure is outside the fallback entry region")
        for key, controller in active_fallback.segment_at(q).controllers:
            controller.observation_box(initial_observations[key], q)
        self.active_fallback, self.q = active_fallback, q
        self.flow_provider = flow_provider
        self.performance_resource_bound = performance_resource_bound
        self.flow_identity = (lambda flow: flow.model_bounds) if flow_identity is None else flow_identity
        self.acceleration_disturbance = acceleration_disturbance or IntervalBox.point((0, 0, 0))
        if self.acceleration_disturbance.dimension != 3:
            raise ValueError("Additional acceleration disturbance needs three components")
        first = active_fallback.segments[0]
        fuel = ({key: c.certificate.spec.available_fuel for key, c in first.controllers}
                if remaining_fuel is None else dict(remaining_fuel))
        if set(fuel) != set(active_fallback.ids):
            raise ValueError("Fuel fleet differs from fallback fleet")
        self.remaining_fuel = {key: _exact(value) for key, value in fuel.items()}
        needed = active_fallback.remaining_fuel(q)
        if any(self.remaining_fuel[key] < needed[key] for key in needed):
            raise ValueError("Reserved fuel does not cover the complete finite fallback")
        self._pending = None
        self.events = []

    def _fallback(self, observations, q, h, attempt, reason, *, refine_resources=True):
        plan = self.active_fallback
        plan.verify_integrity()
        if set(observations) != set(plan.ids):
            raise ValueError("Observed fleet changed")
        segment = plan.segment_at(q, h)
        commands, applied, charge, resources = [], [], [], []
        for key, controller in segment.controllers:
            value = controller.command(observations[key], q=q, hold_width=h, admitted_invariant=True)
            commands.append((key, tuple(_exact(x) for x in value.float_command())))
            applied.append((key, value.applied_input))
            b = controller.certificate.spec
            uniform = cost = h*b.fuel_rate_upper_per_q
            # The installed invariant already covers this complete command.
            # A refused optional refinement cannot interrupt its execution.
            if refine_resources and self.performance_resource_bound is not None:
                try:
                    resource = self.performance_resource_bound(key, value.applied_input, q, h)
                    if (isinstance(resource, CommandResourceBound) and resource.start == q
                            and resource.hold_width == h and resource.applied_input == value.applied_input
                            and resource.fuel_domain_id == b.context.fuel_domain_id
                            and resource.force_margin_n >= 0 and resource.fuel_upper_kg >= 0):
                        cost = min(uniform, resource.fuel_upper_kg)
                        if cost < uniform:
                            resources.append(dict(agent_id=key, uniform_fuel_upper_kg=uniform,
                                charged_fuel_kg=cost, command_resource_bound=exact_record(resource)))
                except Exception:
                    cost = uniform
            charge.append((key, cost))
        charge = tuple(charge)
        needed = plan.remaining_fuel(q+h)
        if any(dict(charge)[key]+needed[key] > self.remaining_fuel[key] for key in plan.ids):
            raise ValueError("Remaining finite fallback exceeds reserved fuel")
        if resources:
            attempt = PerformanceAttempt(attempt.status, attempt.message, attempt.solver_id,
                dict(original_attempt=exact_record(attempt), fallback_resources=resources))
        geometry = next(p for s,p in zip(plan.segments,plan.certificate.geometry) if s is segment)
        return HeldFleetDecision(q, h,
            "FALLBACK_HOLD" if q >= plan.certificate.terminal_holding_start else "FALLBACK_TRACK",
            reason, tuple(commands), tuple(applied), attempt, plan.fingerprint, plan.fingerprint,
            tuple((key, c.certificate.spec.context.fingerprint) for key, c in segment.controllers),
            tuple((key, observations[key].fingerprint) for key in plan.ids), charge,geometry=geometry)

    def decide(self, observations, *, q, hold_width,
               proposal: PerformanceProposal | None = None, failed_attempt: PerformanceAttempt | None = None,
               target_flow_fingerprint: str | None = None, refine_fallback_resources=True):
        q, h = _exact(q), _exact(hold_width)
        if not isinstance(refine_fallback_resources, bool):
            raise TypeError('Fallback resource refinement must be an explicit boolean')
        if self._pending is not None or q != self.q:
            raise ValueError("A decision is pending or the execution clock changed")
        attempt = (proposal.attempt if isinstance(proposal, PerformanceProposal) else failed_attempt
                   if failed_attempt is not None else PerformanceAttempt("NO_PROPOSAL", "No ready performance proposal"))
        if not isinstance(attempt, PerformanceAttempt):
            raise TypeError("Retain the actual typed performance attempt")
        # Existing feedback precedes consideration of a performance proof. No
        # proposal error can replace this queued command by zero or a pause.
        fallback = self._fallback(observations, q, h, attempt,
            "Stored finite feedback policy selected: " + attempt.status,
            refine_resources=refine_fallback_resources)
        if proposal is None:
            self._pending = (fallback, self.active_fallback, None)
            return fallback
        try:
            decision, endpoints = self._admit(proposal, observations, q, h, target_flow_fingerprint)
            self._pending = (decision, proposal.successor, endpoints)
            return decision
        except Exception as error:
            refused_attempt = PerformanceAttempt("ADMISSION_REFUSED", str(error), attempt.solver_id,
                dict(original_attempt=exact_record(attempt),
                     proposal_id=proposal.fingerprint if isinstance(proposal, PerformanceProposal) else None,
                     successor_id=proposal.successor.fingerprint if isinstance(proposal, PerformanceProposal) else None,
                     exception_type=type(error).__name__, q=q, hold_width=h,
                     admission_details=getattr(error, "admission_details", ())))
            fallback = self._fallback(observations, q, h, refused_attempt,
                f"Performance admission refused ({type(error).__name__}: {error}); stored finite feedback selected",
                refine_resources=refine_fallback_resources)
            self._pending = (fallback, self.active_fallback, None)
            return fallback

    def performance_fuel_charge(self,key,box,q,h,controller):
        """Same proved held debit for prospective search and final admission."""
        b=controller.certificate.spec
        if self.performance_resource_bound is None:
            if not controller.certificate.input_box.contains(box):
                raise ValueError("A wider performance command needs its own physical force/fuel proof")
            return h*b.fuel_rate_upper_per_q
        resource=self.performance_resource_bound(key,box,q,h)
        if (not isinstance(resource,CommandResourceBound) or resource.start!=q or resource.hold_width!=h
                or resource.applied_input!=box or resource.fuel_domain_id!=b.context.fuel_domain_id
                or resource.force_margin_n<0 or resource.fuel_upper_kg<0):
            raise ValueError("Performance command resource proof is stale or exceeds installed force authority")
        return resource.fuel_upper_kg

    def _admit(self, proposal, observations, q, h, target_flow_id):
        if not isinstance(proposal, PerformanceProposal) or proposal.start != q or proposal.hold_width != h:
            raise ValueError("Performance proposal is stale or has a different hold schedule")
        _identity(target_flow_id, "current trusted target-flow fingerprint")
        current, successor = self.active_fallback, proposal.successor
        successor.verify_integrity()
        if (not successor.certificate.admissible or successor.ids != current.ids
                or successor.certificate.start != q+h):
            raise ValueError("No complete admitted successor at the exact performance endpoint")
        if (successor.geometry_oracle.body_radii != current.geometry_oracle.body_radii
                or successor.geometry_oracle.component_ids != current.geometry_oracle.component_ids
                or successor.geometry_oracle.clearance < current.geometry_oracle.clearance
                or successor.geometry_oracle.minimum_pair_center_distance < current.geometry_oracle.minimum_pair_center_distance):
            raise ValueError("Successor changed physical bodies/components or weakened the safety clearance")
        _require_policy_supports(current, successor, q, self.acceleration_disturbance)
        segment, next_segment = current.segment_at(q, h), successor.segment_at(q+h)
        inputs = dict(proposal.applied_inputs)
        observed, contexts = dict(proposal.observation_ids), dict(proposal.context_ids)
        if any(set(value) != set(current.ids) for value in (inputs, observed, contexts)):
            raise ValueError("Proposal omitted a fleet input/observation/context")
        flows, commands, charge = {}, [], []
        for key, controller in segment.controllers:
            b, next_b = controller.certificate.spec, dict(next_segment.controllers)[key].certificate.spec
            if not _same_plant(b.context, next_b.context):
                raise ValueError("Successor changed actual model, geometry, sensing, actuator or fuel context")
            if observed[key] != observations[key].fingerprint or contexts[key] != b.context.fingerprint:
                raise ValueError("Proposal used a stale observation or controller context")
            box = inputs[key]
            if not b.input_limits.contains(box):
                raise ValueError("Performance input exceeds the hardware box")
            midpoint = tuple((lo+hi)/2 for lo, hi in zip(box.lower, box.upper))
            executable = tuple(_exact(float(value)) for value in midpoint)
            if any(abs(value-exact) > limit for value, exact, limit in zip(executable, midpoint, b.command_rounding_abs)):
                raise ValueError("Performance conversion exceeds its rounding budget")
            if not box.contains(IntervalBox.from_center_radius(executable, b.actuator_error_abs)):
                raise ValueError("Performance input box omits physical actuator error")
            initial = controller.observation_box(observations[key], q)
            request = HeldInputRequest(initial, box, (q, q), (h, h), self.acceleration_disturbance)
            flow = self.flow_provider(key, request)
            if (not isinstance(flow, FlowCertificate) or not flow.proven or flow.request != request.as_dict()
                    or flow.initial_state != initial or flow.start_anomaly != (q, q) or flow.hold_width != (h, h)
                    or flow.frame != b.context.frame or flow.model != b.context.model_id):
                raise ValueError("Performance flow is unresolved/incomplete or bound to another request/model/frame")
            SweptCollisionGuard._check_flow_structure(flow)
            identity = self.flow_identity(flow)
            if (identity.get("plant_fingerprint") != b.context.parameter_set_id
                    or identity.get("target_model_fingerprint") != b.context.target_model_id
                    or identity.get("clock_fingerprint") != b.context.clock_id
                    or identity.get("target_flow_fingerprint") != target_flow_id):
                raise ValueError("Flow has stale physical model, target ephemeris or clock identities")
            flows[key] = flow
            commands.append((key, executable))
            cost=self.performance_fuel_charge(key,box,q,h,controller)
            charge.append((key, cost))
        geometry = current.geometry_oracle.flows(flows)
        if not geometry.admissible:
            raise ValueError("Complete component/peer swept separation not proved: " + "; ".join(geometry.reasons))
        endpoints = {key: flows[key].endpoint for key in current.ids}
        if not successor.contains(endpoints, q+h):
            entry, summary = {}, []
            for key, controller in next_segment.controllers:
                b, box = controller.certificate.spec, endpoints[key]
                reference = controller.reference_at(q+h)
                position_required, sliding_required = [], []
                for i in range(3):
                    elo = box.lower[i]-reference.position[i]
                    ehi = box.upper[i]-reference.position[i]
                    position_required.append(max(abs(elo), abs(ehi))+b.reference_error_abs[i])
                    slo = box.lower[i+3]-reference.velocity[i]+b.lambda_gain[i]*elo
                    shi = box.upper[i+3]-reference.velocity[i]+b.lambda_gain[i]*ehi
                    sliding_required.append(max(abs(slo), abs(shi))+b.reference_error_abs[i+3]
                                            +b.lambda_gain[i]*b.reference_error_abs[i])
                position_margin = tuple(a-r for a, r in zip(b.omega, position_required))
                sliding_margin = tuple(a-r for a, r in zip(b.phi, sliding_required))
                entry[key] = dict(endpoint_box=exact_record(box), reference=exact_record(reference),
                    omega=b.omega, phi=b.phi, lambda_gain=b.lambda_gain,
                    reference_error_abs=b.reference_error_abs,
                    position_required=tuple(position_required), sliding_required=tuple(sliding_required),
                    position_margin=position_margin, sliding_margin=sliding_margin)
                summary.append(f"{key} p_margin={tuple(float(v) for v in position_margin)} "
                               f"s_margin={tuple(float(v) for v in sliding_margin)}")
            raise PerformanceAdmissionRefusal(
                "Complete performance endpoint is outside successor fallback entry; " + "; ".join(summary),
                details=dict(endpoint_q=q+h, per_agent=entry,
                             display_scope="Float margins are display only; exact rational evidence is retained"))
        needed = successor.remaining_fuel(q+h)
        if any(dict(charge)[key]+needed[key] > self.remaining_fuel[key] for key in current.ids):
            raise ValueError("Performance leaves insufficient fuel for its complete successor")
        return HeldFleetDecision(q, h, "PERFORMANCE", "Joint swept safety and complete finite feedback successors proved",
            tuple(commands), tuple((key, inputs[key]) for key in current.ids), proposal.attempt,
            current.fingerprint, successor.fingerprint,
            tuple((key, c.certificate.spec.context.fingerprint) for key, c in segment.controllers),
            tuple((key, observations[key].fingerprint) for key in current.ids), tuple(charge),
            proposal.fingerprint, tuple(flows[key].fingerprint for key in current.ids), geometry), endpoints

    def commit(self, decision: HeldFleetDecision, observations, *, q):
        """Commit after applying the exact simultaneous whole-fleet hold.

        Impossible observations are assumption violations. They cannot grant a
        safety claim or a physical pause. No proof clock changes on rejection.
        """
        if self._pending is None or not isinstance(decision, HeldFleetDecision):
            raise ValueError("No admitted fleet decision is pending")
        expected, successor, endpoints = self._pending
        q = _exact(q)
        if decision != expected or q != decision.start+decision.hold_width:
            raise ValueError("Execution decision/clock differs from the pending proof")
        successor.verify_integrity()
        if set(observations) != set(successor.ids):
            raise ValueError("Endpoint observation fleet changed")
        contracted = {}
        for key, controller in successor.segment_at(q).controllers:
            box = controller.observation_box(observations[key], q)
            contracted[key] = _intersect(box, endpoints[key]) if endpoints is not None else box
        remaining = {key: self.remaining_fuel[key]-value for key, value in decision.charged_fuel}
        if any(value < 0 for value in remaining.values()):
            raise ValueError("Committed command exceeds reserved fuel")
        self.active_fallback, self.q, self.remaining_fuel = successor, q, remaining
        self._pending = None
        record = dict(decision=decision.as_dict(), exact_end=str(q),
            endpoint_observations={key: exact_record(value) for key, value in observations.items()},
            contracted_beliefs={key: exact_record(value) for key, value in contracted.items()},
            remaining_fuel=exact_record(remaining), remaining_safe_q=str(successor.certificate.end-q),
            status="FINITE_SAFE_WINDOW_ENDED" if q == successor.certificate.end else "COMMITTED")
        self.events.append(record)
        return record
