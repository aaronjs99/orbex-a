"""Whole-fleet occupied-component and peer separation certificates."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
from itertools import combinations
from typing import Any
from orbexa.control.certification.reachability import FlowCertificate, IntervalBox, SweptCollisionGuard, _exact
from orbexa.control.fallback.refuge import _identity, _proof_source, _scope, exact_record, fingerprint
from orbexa.control.fallback.common import _pairs
from orbexa.control.fallback.polynomial import POLYNOMIAL_PEER_METHOD, PolynomialReferenceSource, PolynomialPeerBounds


FLEET_SWEEP_METHOD = "exact-component-box-fleet-sweep-v1"


def _box_distance_squared(left, right):
    return sum(max(Fraction(0), lo2-hi1, lo1-hi2)**2
               for lo1, hi1, lo2, hi2 in zip(left.lower, left.upper, right.lower, right.upper))


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
