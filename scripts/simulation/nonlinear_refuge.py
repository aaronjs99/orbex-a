"""Constructive finite C2 refuges for the declared nonlinear study plant.

Reference bounds use exact polynomial Bernstein enclosures. Actual drift uses
the shared nonlinear force algebra with Arb and a validated target ephemeris.
Occupied geometry remains an injected complete pose-enclosure proof source.
"""
from __future__ import annotations

from dataclasses import dataclass, replace, fields, is_dataclass
from collections import OrderedDict
from collections.abc import Mapping
from fractions import Fraction
from math import comb, factorial, gcd, perm
from typing import Any

from orbexa.control.reachability import IntervalBox, _exact, _ARB_LOCK
from orbexa.control.refuge import (ControlContext, ReferenceJet, ReferenceBounds, NonlinearDriftBounds,
    RefugeTrackingSpec, RefugeBoundsBuilder, RefugeController, RefugeObservation,
    _vec, _absbox, _identity, _scope, exact_record, fingerprint)
from orbexa.control.fleet_fallback import (CommandResourceBound, FleetFallbackPlan,
    FleetFeedbackSegment, FleetGeometryOracle, OccupiedGeometryBounds,OccupiedProjection,
    PolynomialReferenceSource,PolynomialReferenceSpan)
from orbexa.control.fuel_budget import FleetFuelBudget
from orbexa.control.attitude_flow import _sqrt_upper
from orbexa.control.rotation_support import all_rotation_component_bounds
from orbexa.control.nonlinear_flow import _Arithmetic, _region, _target_box, TargetEphemerisCertificate
from orbexa.core.nonlinear_relative import relative_field, parameters

ZERO_NOMINAL_ID = "exact-zero-drift-refuge-v1"


class _FrozenProofIdentityCache:
    """Bounded identities for frozen proof payloads with mutable-context guards.

    Flow/attitude dataclass fields, segment tuples and interval boxes are frozen.
    Their mapping/list contexts are not: every lookup deeply hashes those small
    contexts before reusing the expensive full continuous-segment identity.
    """
    def __init__(self,maximum_objects=4):
        self.maximum_objects,self.entries = maximum_objects,OrderedDict()

    def identity(self,value):
        if not is_dataclass(value) or not value.__dataclass_params__.frozen:
            raise TypeError("Only frozen proof dataclass payloads can reuse a cached identity")
        contexts = {f.name:getattr(value,f.name) for f in fields(value)
                    if isinstance(getattr(value,f.name),(Mapping,list))}
        guard = fingerprint(contexts)
        key = id(value)
        entry = self.entries.get(key)
        if entry is not None and entry[0] is value and entry[1] == guard:
            self.entries.move_to_end(key)
            return entry[2]
        identity = value.fingerprint
        self.entries[key] = (value,guard,identity)
        self.entries.move_to_end(key)
        while len(self.entries) > self.maximum_objects:
            self.entries.popitem(last=False)
        return identity


def _derivative(coefficients, order):
    if type(order) is int and order == 0:
        return coefficients
    if type(order) is int and order > 0:
        return tuple(value*perm(i, order) for i, value in enumerate(coefficients)
                     if i >= order) or (Fraction(0),)
    return tuple(value*Fraction(factorial(i), factorial(i-order))
                 for i, value in enumerate(coefficients) if i >= order) or (Fraction(0),)


def _polynomial(coefficients, value):
    iterator = reversed(coefficients)
    total = next(iterator, Fraction(0))
    if len(coefficients) == 1 and type(total) is Fraction:
        # Keep a returned constant independent of the source Fraction object.
        return Fraction(total.numerator, total.denominator)
    for coefficient in iterator:
        total = total*value+coefficient
    return total


def _fraction_scalars(value):
    if type(value) is not Fraction:
        raise ValueError("Polynomial source scalars must remain exact Fractions")
    numerator, denominator = value.numerator, value.denominator
    if (type(numerator) is not int or type(denominator) is not int
            or denominator <= 0 or gcd(numerator, denominator) != 1):
        raise ValueError("Polynomial source rationals must remain normalized")
    return numerator, denominator


def _coefficient_scalars(rows):
    if type(rows) is not tuple or len(rows) != 3 or any(
            type(row) is not tuple or not 1 <= len(row) <= 6 for row in rows):
        raise ValueError("Three immutable finite polynomial rows are required")
    return tuple(tuple(_fraction_scalars(value) for value in row) for row in rows)


def _bernstein(coefficients, lower, upper):
    """Exact range enclosure on [lower,upper] by Bernstein convex hull."""
    degree = len(coefficients)-1
    if degree == 0:
        value = coefficients[0]
        if type(value) is Fraction:
            value = Fraction(value.numerator, value.denominator)
        return value, value
    power = tuple(sum(coefficients[i]*comb(i,j)*lower**(i-j)*(upper-lower)**j
                      for i in range(j, degree+1)) for j in range(degree+1))
    bernstein = tuple(sum(power[j]*Fraction(comb(k,j), comb(degree,j))
                          for j in range(k+1)) for k in range(degree+1))
    return min(bernstein), max(bernstein)


@dataclass(frozen=True)
class PolynomialReferencePiece:
    start: Any
    end: Any
    coefficients: Any

    def __post_init__(self):
        start, end = _scope(self.start, self.end)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        value = tuple(tuple(map(_exact, row)) for row in self.coefficients)
        if len(value) != 3 or any(not row or len(row) > 6 for row in value):
            raise ValueError("Three polynomials of degree at most five are required")
        object.__setattr__(self, "coefficients", value)

    def jet(self, q, order):
        q = _exact(q)
        if not self.start <= q <= self.end:
            raise ValueError("Polynomial query exceeds its finite piece")
        offset = q-self.start
        return tuple(_polynomial(_derivative(row, order), offset) for row in self.coefficients)

    def jets(self, q):
        """Evaluate all three exact jets with one shared rational clock offset."""
        q = _exact(q)
        if not self.start <= q <= self.end:
            raise ValueError("Polynomial query exceeds its finite piece")
        offset = q-self.start
        return tuple(tuple(_polynomial(_derivative(row, order), offset)
                           for row in self.coefficients) for order in range(3))

    def enclose(self, start, end, order):
        if not self.start <= start <= end <= self.end:
            raise ValueError("Polynomial range exceeds its finite piece")
        pairs = tuple(_bernstein(_derivative(row, order), start-self.start, end-self.start)
                      for row in self.coefficients)
        return IntervalBox(tuple(lo for lo, _ in pairs), tuple(hi for _, hi in pairs))


_PIECE_JET_METHOD = PolynomialReferencePiece.jet
_PIECE_JETS_METHOD = PolynomialReferencePiece.jets
_DERIVATIVE_METHOD = _derivative
_POLYNOMIAL_METHOD = _polynomial


class C2RefugeReference:
    """Exact continuous polynomial route with a final constant holding piece."""
    def __init__(self, pieces, *, frame, clock_id, reference_id=None):
        self.pieces = tuple(pieces)
        if not self.pieces or any(not isinstance(piece, PolynomialReferencePiece) for piece in self.pieces):
            raise ValueError("A finite polynomial route is required")
        for old, new in zip(self.pieces, self.pieces[1:]):
            if old.end != new.start or any(old.jet(old.end, order) != new.jet(new.start, order) for order in range(3)):
                raise ValueError("Reference pieces must join at exact common C2 jets")
        self.frame, self.clock_id = _identity(frame, "reference frame"), _identity(clock_id, "clock identity")
        self.reference_id = reference_id or fingerprint(dict(pieces=exact_record(self.pieces), frame=frame, clock_id=clock_id))
        _identity(self.reference_id, "reference identity")

    @classmethod
    def retreat(cls, *, start, retreat_end, hold_end, initial_state, refuge_position,
                initial_acceleration=(0, 0, 0), frame, clock_id, reference_id=None):
        start, retreat_end = _scope(start, retreat_end)
        _, hold_end = _scope(retreat_end, hold_end)
        x, goal = _vec(initial_state, 6, "reference initial state"), _vec(refuge_position, 3, "refuge position")
        acceleration = _vec(initial_acceleration, 3, "initial reference acceleration")
        duration, rows = retreat_end-start, []
        for p, v, a, g in zip(x[:3], x[3:], acceleration, goal):
            c0, c1, c2 = p, v, a/2
            dp, dv, da = g-c0-c1*duration-c2*duration**2, -c1-2*c2*duration, -2*c2
            rows.append((c0,c1,c2,10*dp/duration**3-4*dv/duration**2+da/(2*duration),
                -15*dp/duration**4+7*dv/duration**3-da/duration**2,
                6*dp/duration**5-3*dv/duration**4+da/(2*duration**3)))
        return cls((PolynomialReferencePiece(start,retreat_end,rows),
                    PolynomialReferencePiece(retreat_end,hold_end,tuple((g,) for g in goal))),
                   frame=frame,clock_id=clock_id,reference_id=reference_id)

    def sample(self, q):
        q = _exact(q)
        piece = next((p for p in self.pieces if p.start <= q < p.end), None)
        if piece is None and q == self.pieces[-1].end:
            piece = self.pieces[-1]
        if piece is None:
            raise ValueError("Reference sample exceeds its finite route")
        known = (type(piece) is PolynomialReferencePiece
                 and getattr(piece.jet, "__func__", piece.jet) is _PIECE_JET_METHOD
                 and getattr(piece.jets, "__func__", piece.jets) is _PIECE_JETS_METHOD
                 and _derivative is _DERIVATIVE_METHOD and _polynomial is _POLYNOMIAL_METHOD)
        jets = piece.jets(q) if known else tuple(piece.jet(q, order) for order in range(3))
        return ReferenceJet(q, *jets,
                            self.reference_id,self.frame,self.clock_id)

    def polynomial_source(self):
        """Reuse an immutable source only while every actual exact field agrees.

        The key copies field values rather than retaining Piece identities, so
        even replacing fields on a frozen piece cannot alias the old guard.
        Mutable containers never qualify for this cache.
        """
        if type(self.pieces) is not tuple or not self.pieces or any(type(piece) is not PolynomialReferencePiece for piece in self.pieces):
            raise ValueError("Polynomial source caching requires immutable typed reference pieces")
        pieces=tuple((_fraction_scalars(piece.start),_fraction_scalars(piece.end),
                      _coefficient_scalars(piece.coefficients)) for piece in self.pieces)
        key=(self.reference_id,self.frame,self.clock_id,pieces)
        cached=getattr(self,"_polynomial_source_cache",None)
        if type(cached) is tuple and len(cached)==2 and cached[0]==key and type(cached[1]) is PolynomialReferenceSource:
            source=cached[1]
            try:
                unchanged=(type(source.pieces) is tuple
                    and all(type(piece) is PolynomialReferenceSpan for piece in source.pieces)
                    and (source.reference_id,source.frame,source.clock_id,
                        tuple((_fraction_scalars(piece.start),_fraction_scalars(piece.end),
                               _coefficient_scalars(piece.coefficients)) for piece in source.pieces))==key)
            except (AttributeError,TypeError,ValueError):
                unchanged=False
            if unchanged:
                return source
        # Copy every rational from native integers; the reusable source must
        # not alias a source/cache Fraction forced to change after admission.
        source=PolynomialReferenceSource(tuple(PolynomialReferenceSpan(Fraction(*start),Fraction(*end),
            tuple(tuple(Fraction(*value) for value in row) for row in rows)) for start,end,rows in pieces),
            self.reference_id,self.frame,self.clock_id)
        self._polynomial_source_cache=(key,source)
        return source

    def bounds(self, start, end):
        start, end = _scope(start, end)
        if start < self.pieces[0].start or end > self.pieces[-1].end:
            raise ValueError("Reference bound query exceeds its finite route")
        ranges = []
        for order in range(4):
            value = None
            for piece in self.pieces:
                lower, upper = max(start,piece.start), min(end,piece.end)
                if lower < upper:
                    local = piece.enclose(lower, upper, order)
                    value = local if value is None else value.hull(local)
            ranges.append(value)
        return ReferenceBounds(start,end,*ranges[:3],_absbox(ranges[3]),self.reference_id,
            self.frame,self.clock_id,"exact-quintic-Bernstein:"+fingerprint(self.pieces),
            ("Exact rational polynomial coefficients and exact C2 boundary identities",
             "Bernstein convex hull encloses every position, velocity, acceleration and one-sided jerk value"),
            proof_kind="analytic-global")

    def windows(self, maximum_width, maximum_windows=10000):
        width = _exact(maximum_width)
        if width <= 0:
            raise ValueError("Positive finite reference bound width is required")
        output = []
        for piece in self.pieces:
            ratio = (piece.end-piece.start)/width
            count = max(1,(ratio.numerator+ratio.denominator-1)//ratio.denominator)
            if len(output)+count > maximum_windows:
                raise ValueError("Finite reference-window budget exhausted")
            step = (piece.end-piece.start)/count
            output.extend((piece.start+i*step,piece.start+(i+1)*step) for i in range(count))
        return tuple(output)


class NonlinearZeroDriftBounds:
    """Conservative actual nonlinear acceleration envelope, with F_nom=0.

    Zero nominal drift has exact zero Jacobians/evaluation error. Actual force,
    J2, vector drag, frame rotation and clock terms are enclosed with Arb.
    target_provider(start,end) must return a validated full target ephemeris,
    never a reset numerical pose. Limits and source identities are checked.
    """
    def __init__(self, *, model, clock, target_provider, maximum_width="0.0001",
                 maximum_pieces=10000, precision_bits=128, extra_acceleration_abs=(0,0,0)):
        self.model,self.clock,self.target_provider = model,clock,target_provider
        self.maximum_width = _exact(maximum_width)
        self.maximum_pieces,self.precision_bits = maximum_pieces,precision_bits
        self.extra_acceleration_abs = _vec(extra_acceleration_abs,3,"extra acceleration",nonnegative=True)
        self._identities = _FrozenProofIdentityCache()
        self._target_boxes = OrderedDict()
        if self.maximum_width <= 0 or maximum_pieces < 1 or precision_bits < 64:
            raise ValueError("Finite nonlinear bound budgets are required")

    @staticmethod
    def nominal_drift(q, measurement):
        return (Fraction(0),)*3

    def verify_certificate_source(self, certificate, required_extra):
        """Authenticate existing drift/support provenance without a new flow.

        A mutable provider tag cannot enlarge a previously proved disturbance
        budget. Recompute the original target/model/clock/region/scope/extra/
        precision digest and require its exact retained proof reference.
        """
        from orbexa.control.refuge import RefugeCertificate
        if type(self) is not NonlinearZeroDriftBounds or not isinstance(certificate, RefugeCertificate) or not certificate.admissible:
            raise ValueError('A source-backed nonlinear refuge certificate is required')
        required = _vec(required_extra, 3, 'required extra acceleration', nonnegative=True)
        extra = _vec(self.extra_acceleration_abs, 3, 'declared extra acceleration', nonnegative=True)
        if any(need > bound for need, bound in zip(required, extra)):
            raise ValueError('Existing nonlinear source does not cover the required extra acceleration')
        spec, ctx = certificate.spec, certificate.spec.context
        if (ctx.model_id != 'nonlinear-reference-relative-v1' or ctx.nominal_model_id != ZERO_NOMINAL_ID
                or ctx.parameter_set_id != self.model.fingerprint
                or ctx.target_model_id != self.model.target_fingerprint
                or ctx.clock_id != self.clock.fingerprint or ctx.frame != 'reference-orbit-frame-v1'):
            raise ValueError('Existing nonlinear drift certificate changed model/frame/clock context')
        target = self.target_provider(spec.start, spec.end)
        if (not isinstance(target, TargetEphemerisCertificate) or not target.proven
                or target.model_bounds.get('target_model_fingerprint') != self.model.target_fingerprint
                or target.model_bounds.get('clock_fingerprint') != self.clock.fingerprint):
            raise ValueError('Existing nonlinear drift source lacks a complete same-model target proof')
        source = fingerprint(dict(target=self._identities.identity(target), model=self.model.fingerprint,
            clock=self.clock.fingerprint, region=exact_record(certificate.state_box),
            scope=(str(spec.start), str(spec.end)), extra=exact_record(extra), precision=self.precision_bits))
        if certificate.drift_bounds.proof_reference != 'Arb-actual-nonlinear-drift:'+source:
            raise ValueError('Existing nonlinear drift/source declaration changed after proof')
        return True

    def __call__(self, spec, reference, state_box, sensing_box):
        import flint
        ctx,model,clock = spec.context,self.model,self.clock
        if (ctx.model_id != "nonlinear-reference-relative-v1" or ctx.nominal_model_id != ZERO_NOMINAL_ID
                or ctx.parameter_set_id != model.fingerprint or ctx.target_model_id != model.target_fingerprint
                or ctx.frame != "reference-orbit-frame-v1" or ctx.clock_id != clock.fingerprint):
            raise ValueError("Nonlinear refuge model/reference-clock identities differ")
        _region(state_box,model)
        clock.check_domain((spec.start,spec.end))
        target = self.target_provider(spec.start,spec.end)
        if (not isinstance(target,TargetEphemerisCertificate) or not target.proven
                or target.model_bounds.get("target_model_fingerprint") != model.target_fingerprint
                or target.model_bounds.get("clock_fingerprint") != clock.fingerprint):
            raise ValueError("A proved same-model/clock target ephemeris is required")
        target_id = self._identities.identity(target)
        ratio = (spec.end-spec.start)/self.maximum_width
        count = max(1,(ratio.numerator+ratio.denominator-1)//ratio.denominator)
        if count > self.maximum_pieces:
            raise ValueError("Finite nonlinear drift-bound budget exhausted")
        step,result = (spec.end-spec.start)/count,None
        with _ARB_LOCK, flint.ctx.workprec(self.precision_bits):
            ar = _Arithmetic(flint)
            par,state,zero = parameters(model,ar),ar.vector(state_box),(ar.scalar(0),)*3
            for index in range(count):
                interval = (spec.start+index*step,spec.start+(index+1)*step)
                cache_key = (target_id,interval)
                target_box = self._target_boxes.get(cache_key)
                if target_box is None:
                    target_box = _target_box(target,interval)
                    self._target_boxes[cache_key] = target_box
                    while len(self._target_boxes) > 1024:
                        self._target_boxes.popitem(last=False)
                value = relative_field(ar.interval(interval),state,ar.vector(target_box),
                    zero,model,clock,ar,par)[3:]
                local = ar.box(value)
                result = local if result is None else result.hull(local)
        mismatch = tuple(v+e for v,e in zip(_absbox(result),self.extra_acceleration_abs))
        source = fingerprint(dict(target=target_id, model=model.fingerprint, clock=clock.fingerprint,
            region=exact_record(state_box), scope=(str(spec.start),str(spec.end)),
            extra=exact_record(self.extra_acceleration_abs), precision=self.precision_bits))
        return NonlinearDriftBounds(spec.start,spec.end,state_box,sensing_box,spec.input_limits,
            IntervalBox.point((0,0,0)),mismatch,((0,0,0),)*3,((0,0,0),)*3,(0,0,0),(0,0,0),
            ctx.model_id,ctx.nominal_model_id,ctx.parameter_set_id,ctx.target_model_id,ctx.fuel_domain_id,
            ctx.frame,ctx.clock_id,"Arb-actual-nonlinear-drift:"+source,
            ("Declared full nonlinear gravity/J2/vector drag and fixed commanded reference clock",
             "Validated target ephemeris continuous coverage; no numerical target resets",
             "Exact zero nominal drift and global regional actual-drift forcing enclosure",
             "Explicit additional bounded anomaly-acceleration disturbance"))


@dataclass(frozen=True)
class TargetPoseEnclosure:
    start: Any
    end: Any
    rotation_entries: IntervalBox
    center: IntervalBox
    geometry_id: str
    target_model_id: str
    frame: str
    clock_id: str
    proof_reference: str
    assumptions: tuple[str, ...]

    def __post_init__(self):
        start,end = _scope(self.start,self.end)
        object.__setattr__(self,"start",start)
        object.__setattr__(self,"end",end)
        if self.rotation_entries.dimension != 9 or self.center.dimension != 3:
            raise ValueError("Pose bounds require complete 3 by 3 rotation and 3-center enclosures")
        for name in ("geometry_id","target_model_id","frame","clock_id","proof_reference"):
            _identity(getattr(self,name),name)
        if not self.assumptions or any(not isinstance(item,str) or not item.strip() for item in self.assumptions):
            raise ValueError("Whole-window pose assumptions are required")
        object.__setattr__(self,"assumptions",tuple(self.assumptions))


class AnyProperTargetPose:
    """Analytical all-attitude enclosure, useful for separated far-field refuges.

    Every entry of a proper rotation lies in [-1,1]. This exact but conservative
    bound covers arbitrary target tumbling; no sampled pose/rate is consulted.
    Target-centered coordinates place the actual target COM at zero by definition.
    """
    def __init__(self, *, geometry_id,target_model_id,frame,clock_id):
        self.geometry_id,self.target_model_id = geometry_id,target_model_id
        self.frame,self.clock_id = frame,clock_id

    def __call__(self,start,end):
        return TargetPoseEnclosure(start,end,IntervalBox((-1,)*9,(1,)*9),IntervalBox.point((0,0,0)),
            self.geometry_id,self.target_model_id,self.frame,self.clock_id,
            "analytic-all-proper-rotation-entry-bounds",
            ("Actual target attitude remains a proper rotation",
             "Actual target-centered reference coordinates place its COM at zero"))


_ANY_PROPER_POSE_METHOD = AnyProperTargetPose.__call__


class CompositeOccupiedBoundsProvider:
    """Transform every exact expanded component vertex by a proved pose box."""
    def __init__(self,geometry,pose_provider):
        self.geometry,self.pose_provider = geometry,pose_provider
        self._numeric_payload = None

    @staticmethod
    def _box_payload(box):
        """Copy immutable primitive values, including forced-mutation guards."""
        rows=[]
        for row in (box.lower,box.upper):
            if type(row) is not tuple:return None
            values=[]
            for value in row:
                if (type(value) is not Fraction or type(value.numerator) is not int
                        or type(value.denominator) is not int or value.denominator<=0
                        or gcd(value.numerator,value.denominator)!=1):return None
                values.append((value.numerator,value.denominator))
            rows.append(tuple(values))
        return tuple(rows)

    @staticmethod
    def _payload_box(payload):
        # Fresh boxes/Fractions prevent a caller's forced output mutation from
        # changing the privately retained numeric payload.
        return IntervalBox(*(tuple(Fraction(n,d) for n,d in row) for row in payload))

    @staticmethod
    def _bounds(start,end,pose,components,projections):
        return OccupiedGeometryBounds(start,end,tuple(components),pose.geometry_id,pose.target_model_id,
            pose.frame,pose.clock_id,"exact-expanded-vertices-pose-product:"+fingerprint(pose),
            pose.assumptions+("Every declared solid component is enclosed by its exact uncertainty-expanded vertices",
                             "Fixed nonzero support directions enclose n^T R(q) of every expanded vertex"),
            support_projections=tuple(projections))

    def __call__(self,start,end):
        pose = self.pose_provider(start,end)
        geometry_id = self.geometry.fingerprint
        if (not isinstance(pose,TargetPoseEnclosure) or pose.start > start or pose.end < end
                or pose.geometry_id != geometry_id):
            raise ValueError("Complete same-geometry whole-window pose enclosure is required")
        rotation,center = self._box_payload(pose.rotation_entries),self._box_payload(pose.center)
        proper = (type(self.pose_provider) is AnyProperTargetPose
            and type(self.pose_provider).__call__ is _ANY_PROPER_POSE_METHOD)
        key = None if rotation is None or center is None else (geometry_id,rotation,center,proper)
        if key is not None and self._numeric_payload is not None and self._numeric_payload[0] == key:
            component_payload,projection_payload = self._numeric_payload[1]
            components = tuple((name,self._payload_box(box)) for name,box in component_payload)
            projections = tuple(OccupiedProjection(name,tuple(Fraction(n,d) for n,d in direction),self._payload_box(box))
                for name,direction,box in projection_payload)
            return self._bounds(start,end,pose,components,projections)
        components,projections = [],[]
        midpoint = tuple((a+b)/2 for a,b in zip(pose.rotation_entries.lower,pose.rotation_entries.upper))
        for component in self.geometry.components:
            if proper:
                box,supports=all_rotation_component_bounds(component.vertices_body_m,pose.center)
                components.append((component.id,box))
                projections.extend(OccupiedProjection(component.id,direction,projected)
                    for direction,projected in supports)
                continue
            points = []
            for vertex in component.vertices_body_m:
                lower,upper = [],[]
                for i in range(3):
                    lo,hi = pose.center.lower[i],pose.center.upper[i]
                    for j,value in enumerate(vertex):
                        a,b = pose.rotation_entries.lower[3*i+j]*value,pose.rotation_entries.upper[3*i+j]*value
                        lo,hi = lo+min(a,b),hi+max(a,b)
                    lower.append(lo); upper.append(hi)
                points.append(IntervalBox(tuple(lower),tuple(upper)))
            box = points[0]
            for point in points[1:]: box = box.hull(point)
            components.append((component.id,box))
            # Fixed exact directions need not be exact unit normals. Query-local
            # projected OBB axes are useful candidates; validity comes entirely
            # from enclosing n^T R(q) vertices, not from the direction's name.
            component_rotation = component.rotation_component_to_body
            directions = [(Fraction(1),Fraction(0),Fraction(0)),(Fraction(0),Fraction(1),Fraction(0)),
                          (Fraction(0),Fraction(0),Fraction(1))]
            directions.extend(tuple(sum(midpoint[3*i+k]*_exact(component_rotation[k][j]) for k in range(3))
                                    for i in range(3)) for j in range(3))
            for direction in dict.fromkeys(directions):
                if not any(direction):
                    continue
                coeff_lo,coeff_hi = [],[]
                for j in range(3):
                    products = tuple((direction[i]*pose.rotation_entries.lower[3*i+j],
                                      direction[i]*pose.rotation_entries.upper[3*i+j]) for i in range(3))
                    coeff_lo.append(sum(min(a,b) for a,b in products))
                    coeff_hi.append(sum(max(a,b) for a,b in products))
                center_products = tuple((d*lo,d*hi) for d,lo,hi in zip(direction,pose.center.lower,pose.center.upper))
                center_lo,center_hi = sum(min(a,b) for a,b in center_products),sum(max(a,b) for a,b in center_products)
                vertex_ranges = []
                for vertex in component.vertices_body_m:
                    products = tuple((a*v,b*v) for a,b,v in zip(coeff_lo,coeff_hi,vertex))
                    vertex_ranges.append((center_lo+sum(min(a,b) for a,b in products),
                                          center_hi+sum(max(a,b) for a,b in products)))
                projected = IntervalBox((min(lo for lo,_ in vertex_ranges),),(max(hi for _,hi in vertex_ranges),))
                projections.append(OccupiedProjection(component.id,direction,projected))
        if key is not None:
            payload = (tuple((name,self._box_payload(box)) for name,box in components),
                tuple((value.component_id,tuple((v.numerator,v.denominator) for v in value.direction),
                       self._box_payload(value.projected_interval)) for value in projections))
            self._numeric_payload = (key,payload)
        return self._bounds(start,end,pose,components,projections)


class AttitudePoseBoundsProvider:
    """Adapt a proved quaternion/Euler flow to complete rotation-entry bounds.

    attitude_provider(start,end) returns a same-context AttitudeFlowCertificate
    whose initial pose observation and full relative angular-rate bound cover
    the requested window. It is never reconstructed from sampled rotations.
    """
    def __init__(self,attitude_provider,*,geometry_id,target_model_id,frame,clock_id,clock,precision_bits=128):
        self.attitude_provider,self.geometry_id = attitude_provider,geometry_id
        self.target_model_id,self.frame,self.clock_id = target_model_id,frame,clock_id
        self.clock,self.precision_bits = clock,precision_bits
        self._identities = _FrozenProofIdentityCache()
        self._verified_attitudes = OrderedDict()
        self._pose_boxes = OrderedDict()
        if clock.fingerprint != clock_id or precision_bits < 64:
            raise ValueError("A same-identity actual reference clock and bounded Arb precision are required")

    def __call__(self,start,end):
        from orbexa.control.attitude_flow import (AttitudeFlowCertificate,AttitudeSegment,_matrix,
            _SOURCE_SHA256 as attitude_source)
        value = self.attitude_provider(start,end)
        if (not isinstance(value,AttitudeFlowCertificate) or not value.proven
                or value.context.get("target_model_fingerprint") != self.target_model_id
                or value.context.get("clock_fingerprint") != self.clock_id
                or value.context.get("frame") != self.frame):
            raise ValueError("A same-model/clock proved attitude flow is required")
        attitude_id = self._identities.identity(value)
        q = tuple(map(_exact,value.context["start_q"]))
        width = tuple(map(_exact,value.context["width_q"]))
        if q[0] != q[1] or width[0] != width[1] or q[0] > start or q[0]+width[0] < end:
            raise ValueError("Attitude flow does not cover the complete occupied window")
        cache_key = (attitude_id,start,end,self.geometry_id,self.target_model_id,self.frame,self.clock_id)
        if cache_key in self._pose_boxes:
            self._pose_boxes.move_to_end(cache_key)
            return self._pose_boxes[cache_key]
        offset,previous = Fraction(0),value.initial_state
        for segment in (() if attitude_id in self._verified_attitudes else value.segments):
            if (not isinstance(segment,AttitudeSegment) or segment.offset[0] != offset
                    or segment.offset[1] <= offset or segment.anomaly[0] > q[0]+segment.offset[0]
                    or segment.anomaly[1] < q[0]+segment.offset[1]
                    or any(not isinstance(box,IntervalBox) or box.dimension != 7 for box in
                           (segment.initial_state,segment.swept_state,segment.endpoint,segment.picard_image,segment.derivative))
                    or not segment.initial_state.contains(previous)
                    or not segment.swept_state.contains(segment.initial_state)
                    or not segment.swept_state.contains(segment.endpoint)):
                raise ValueError("Incomplete continuous attitude segment coverage or state chain")
            offset,previous = segment.offset[1],segment.endpoint
        if attitude_id not in self._verified_attitudes and (offset != width[0] or value.endpoint != previous):
            raise ValueError("Attitude certificate omits part of its full finite clock domain")
        # The maintained quaternion ODE preserves unit norm of the physical
        # state. Its Picard boxes contain additional off-sphere points, which
        # only widen the polynomial image. Interval-normalizing that ambient
        # box reintroduces dependencies without covering any extra physical
        # attitude. Keep the generic normalization for other declared flows.
        physical_unit=(value.context.get('model')=='nonlinear-ephemeris-quaternion-euler-v1'
            and value.context.get('source_sha256')==attitude_source
            and value.context.get('quaternion_order')=='xyzw'
            and 'Initial box contains a unit physical quaternion' in value.context.get('assumptions',()))
        if physical_unit:
            initial=value.initial_state.slice(0,4)
            norm_lower=sum(0 if lo<=0<=hi else min(lo*lo,hi*hi)
                for lo,hi in zip(initial.lower,initial.upper))
            norm_upper=sum(max(lo*lo,hi*hi) for lo,hi in zip(initial.lower,initial.upper))
            if not norm_lower<=1<=norm_upper:
                raise ValueError('Maintained attitude flow has no admissible unit initial quaternion')
        self._verified_attitudes[attitude_id] = True
        self._verified_attitudes.move_to_end(attitude_id)
        while len(self._verified_attitudes) > 4:
            self._verified_attitudes.popitem(last=False)
        import flint
        rotation = None
        with _ARB_LOCK,flint.ctx.workprec(self.precision_bits):
            ar = _Arithmetic(flint)
            for segment in value.segments:
                lower,upper = max(start,segment.anomaly[0]),min(end,segment.anomaly[1])
                if lower > upper:
                    continue
                quat = ar.vector(segment.swept_state.slice(0,4))
                norm = None if physical_unit else ar.sqrt(sum(ar.square(x) for x in quat))
                if physical_unit or norm > 0:
                    body = _matrix(quat if physical_unit else tuple(x/norm for x in quat),ar)
                    reference = self.clock.evaluate(ar.interval((lower,upper)),ar)[0]
                    local = ar.box(tuple(sum(reference[k][i]*body[k][j] for k in range(3))
                                         for i in range(3) for j in range(3)))
                    local = IntervalBox(tuple(max(Fraction(-1),x) for x in local.lower),
                                        tuple(min(Fraction(1),x) for x in local.upper))
                else:
                    # A wide interval may contain the zero vector even though
                    # the certified physical quaternion is unit. The exact
                    # all-proper-rotation envelope remains valid and may refuse.
                    local = IntervalBox((-1,)*9,(1,)*9)
                rotation = local if rotation is None else rotation.hull(local)
        if rotation is None:
            raise ValueError("No continuous attitude segment covers the requested occupied window")
        result = TargetPoseEnclosure(start,end,rotation,IntervalBox.point((0,0,0)),self.geometry_id,
            self.target_model_id,self.frame,self.clock_id,("unit-quaternion-polynomial-sweeps:" if physical_unit else
                "local-quaternion-Euler-sweeps-Arb-reference-rotation:")+attitude_id,
            ("Proved quaternion/Euler continuous flow on the nonlinear ephemeris",
             ("Unit physical quaternion polynomial image of each complete swept box; no ambient normalization" if physical_unit else
              "Every queried swept quaternion box is normalized with Arb"),
             "Every image is transformed by the complete interval C_ref(q)^T",
             "Unit true quaternion invariance; all proper rotations retained if a wide interval includes zero"))
        self._pose_boxes[cache_key] = result
        while len(self._pose_boxes) > 4096:
            self._pose_boxes.popitem(last=False)
        return result


@dataclass(frozen=True)
class StudyPropulsionBounds:
    mass_upper_kg: Any = 130
    installed_force_n: Any = 100
    specific_impulse_lower_s: Any = 300
    available_fuel_kg: Any = 10
    attitude_force_allocation_n: Any = 0
    g0_m_s2: Any = "9.80665"
    declaration: str = "Prospective study hardware; no physical thruster qualification"

    def __post_init__(self):
        for name in ("mass_upper_kg","installed_force_n","specific_impulse_lower_s","available_fuel_kg",
                     "attitude_force_allocation_n","g0_m_s2"):
            value = _exact(getattr(self,name))
            if value < 0 or (name not in ("available_fuel_kg","attitude_force_allocation_n") and value == 0):
                raise ValueError("Physical resource bounds need positive mass/force/Isp/gravity and nonnegative fuel")
            object.__setattr__(self,name,value)
        if self.attitude_force_allocation_n >= self.installed_force_n:
            raise ValueError("Attitude allocation exhausts the installed study authority")
        _identity(self.declaration,"study hardware declaration")

    @property
    def fingerprint(self):
        return fingerprint(self)

    def input_resource(self,key,input_box,q,h,clock):
        q,h = _exact(q),_exact(h)
        rate = clock.interval_bounds((q,q+h))["rate_s_inv"]
        lower,upper = rate.lower[0],rate.upper[0]
        if lower <= 0:
            raise ValueError("Resource proof requires a positive reference-clock rate")
        # For every proper rotation C and every member of the applied box,
        # ||C*u||1 <= sqrt(3)||u||2 <= sqrt(3 sum_i max|u_i|^2).
        # Keep the old valid bound when outward sqrt rounding is larger.
        # This tightens an upper debit; it does not predict actual fuel use.
        magnitudes = _absbox(input_box)
        input_upper = min(3*max(magnitudes),_sqrt_upper(3*sum(value*value for value in magnitudes)))
        force_upper = self.mass_upper_kg*upper**2*input_upper+self.attitude_force_allocation_n
        fuel_per_q = (self.mass_upper_kg*upper*input_upper+self.attitude_force_allocation_n/lower)/(
            self.specific_impulse_lower_s*self.g0_m_s2)
        # Sum a common-denominator upper bound. Repeated exact reciprocals of
        # independent Arb rate endpoints otherwise accumulate huge coprime
        # denominators in the finite tank ledger. The ceiling increases every
        # debit, preserving its conservative physical meaning.
        quantum=1<<80
        scaled=fuel_per_q*quantum
        fuel_per_q=Fraction((scaled.numerator+scaled.denominator-1)//scaled.denominator,quantum)
        return CommandResourceBound(q,h,input_box,h*fuel_per_q,self.installed_force_n-force_upper,
            self.fingerprint,"analytic-nu-squared-sharp-SO3-body-L1-force-and-Isp-dyadic-upper80:"+clock.fingerprint)

    def callback(self,clock):
        return lambda key,input_box,q,h: self.input_resource(key,input_box,q,h,clock)


def construct_refuge_plan(*, references, contexts, drift_adapter: NonlinearZeroDriftBounds,
                          geometry_oracle: FleetGeometryOracle, propulsion: StudyPropulsionBounds,
                          omega,phi,lambda_gain,beta,acceleration_bound,maximum_hold,input_limits,
                          sensing_error_abs,actuator_error_abs,command_rounding_abs,
                          maximum_window="0.002",maximum_windows=10000):
    """Build a real finite feedback route; mathematical/geometry refusal survives.

    All members share piece boundaries. A new route/reference is never admitted
    by a numerical predicted endpoint; caller must perform whole-box entry or
    the runtime's joint nonlinear successor transaction.
    """
    ids = tuple(references)
    if ids != geometry_oracle.ids or set(contexts) != set(ids):
        raise ValueError("Reference, geometry and model-context fleets differ")
    windows = references[ids[0]].windows(maximum_window,maximum_windows)
    if any(references[key].windows(maximum_window,maximum_windows) != windows for key in ids):
        raise ValueError("Constructive fleet routes require common finite reference windows")
    segments = []
    omega,phi = _vec(omega,3,"omega",positive=True),_vec(phi,3,"phi",positive=True)
    lam,force = _vec(lambda_gain,3,"lambda",positive=True),_vec(beta,3,"beta",positive=True)
    errors = _vec(sensing_error_abs,6,"sensor envelope",nonnegative=True)
    actuator = _vec(actuator_error_abs,3,"actuator error",nonnegative=True)
    rounding = _vec(command_rounding_abs,3,"command rounding",nonnegative=True)
    for start,end in windows:
        controllers = {}
        for key in ids:
            reference,ctx = references[key],contexts[key]
            if (ctx.reference_id != reference.reference_id or ctx.fuel_domain_id != propulsion.fingerprint):
                raise ValueError("Constructive refuge reference/fuel identities differ from the requested context")
            bounds = reference.bounds(start,end)
            ve = tuple(p+l*o for p,l,o in zip(phi,lam,omega))
            pad = tuple(l*(v+e)+b+a+r for l,v,e,b,a,r in zip(lam,ve,errors[3:],force,actuator,rounding))
            command_range = IntervalBox(tuple(a-p for a,p in zip(bounds.acceleration.lower,pad)),
                                       tuple(a+p for a,p in zip(bounds.acceleration.upper,pad)))
            resources = propulsion.input_resource(key,command_range,start,end-start,drift_adapter.clock)
            if resources.force_margin_n < 0:
                raise ValueError(f"{key}: finite feedback/reference range exceeds installed physical study force")
            spec = RefugeTrackingSpec(start,end,ctx,omega,phi,lam,force,maximum_hold,acceleration_bound,
                input_limits,errors,actuator,rounding,(0,)*9,resources.fuel_upper_kg/(end-start),
                propulsion.available_fuel_kg,resources.proof_reference,
                (propulsion.declaration,"Mass upper bound and Isp lower bound remain valid throughout the finite route",
                 "Unknown proper body rotation uses the outward sqrt(3 sum max|u_i|^2) bound, capped by 3 max|u|, with explicit attitude allocation"))
            builder = RefugeBoundsBuilder(spec,reference,drift_adapter,drift_adapter.nominal_drift,
                                           nominal_model_id=ZERO_NOMINAL_ID)
            controllers[key] = RefugeController(builder)
        segments.append(FleetFeedbackSegment(controllers))
    return FleetFallbackPlan(tuple(segments),geometry_oracle)


@dataclass(frozen=True)
class PatchRefugeCandidateAttempt:
    retreat_duration: Fraction
    holding_duration: Fraction
    tangent_offset: tuple[Fraction,Fraction]
    status: str
    reasons: tuple[str,...]
    plan_id: str | None = None


@dataclass(frozen=True)
class PatchRefugeSearchResult:
    plan: FleetFallbackPlan | None
    references: tuple[tuple[str,C2RefugeReference],...]
    contexts: tuple[tuple[str,ControlContext],...]
    attempts: tuple[PatchRefugeCandidateAttempt,...]

    @property
    def admissible(self):
        return self.plan is not None and self.plan.certificate.admissible

    @property
    def reference_map(self):
        return dict(self.references)

    @property
    def context_map(self):
        return dict(self.contexts)


def construct_patch_refuge_candidates(*, initial_states, initial_enclosures,
        outward_directions, tangent_u, tangent_v, start, contexts,
        drift_adapter: NonlinearZeroDriftBounds, geometry_oracle: FleetGeometryOracle,
        propulsion: StudyPropulsionBounds, retreat_durations=("0.015","0.02"),
        holding_duration="0.003", outward_distance=2,
        tangent_offsets=(("0.5",0),(0,0),("-0.5",0),(0,"0.5"),(0,"-0.5"),
                         ("0.5","0.5"),("0.5","-0.5"),("-0.5","0.5"),("-0.5","-0.5")),
        maximum_candidates=18, fuel_budget=None, **tracking_options):
    """Prepare a finite jointly proved patch-to-refuge route, or retain refusal.

    All agents receive the same fixed reference-frame tangent displacement.
    Supplied outward vectors, initial jets and intervals are exact inputs;
    their interpretation as measured patch normals is a caller assumption.
    No numerical optimizer or truth pose is used. Every attempted complete
    route still needs the nonlinear invariant inequalities, all occupied
    components, peer separation, terminal holding, fuel and initial-box entry.
    This construction runs before installation, never at an execution failure.
    """
    ids = geometry_oracle.ids
    if (set(initial_states) != set(ids) or set(initial_enclosures) != set(ids)
            or set(outward_directions) != set(ids) or set(contexts) != set(ids)):
        raise ValueError("Patch candidate states, enclosures, directions, contexts and geometry fleets differ")
    if not isinstance(maximum_candidates,int) or isinstance(maximum_candidates,bool) or maximum_candidates < 1:
        raise ValueError("Declare a positive finite patch candidate budget")
    q,hold,distance = map(_exact,(start,holding_duration,outward_distance))
    if fuel_budget is not None:
        if type(fuel_budget) is not FleetFuelBudget:
            raise TypeError('Patch continuation search needs an exact FleetFuelBudget')
        fuel_budget.check_context(ids,q,propulsion.fingerprint)
    durations = tuple(_exact(value) for value in retreat_durations)
    offsets = tuple(_vec(value,2,"common tangent offset") for value in tangent_offsets)
    if not durations or not offsets or hold <= 0 or distance <= 0 or any(value <= 0 for value in durations):
        raise ValueError("Positive finite retreat, holding and outward-distance domains are required")
    u,v = _vec(tangent_u,3,"common tangent u"),_vec(tangent_v,3,"common tangent v")
    if not any(u) or not any(v) or not any(u[i]*v[j]-u[j]*v[i] for i in range(3) for j in range(i)):
        raise ValueError("Common patch tangent vectors must be nonzero and independent")
    states,directions = {},{}
    for key in ids:
        states[key] = _vec(initial_states[key],6,key+" reference initial jet")
        directions[key] = _vec(outward_directions[key],3,key+" fixed outward vector")
        box = initial_enclosures[key]
        if (not any(directions[key]) or not isinstance(box,IntervalBox) or box.dimension != 6
                or not box.contains(IntervalBox.point(states[key])) or not isinstance(contexts[key],ControlContext)):
            raise ValueError(key+": a nonzero outward vector and complete initial state enclosure are required")
        ctx = contexts[key]
        if ctx.frame != geometry_oracle.frame or ctx.clock_id != geometry_oracle.clock_id:
            raise ValueError(key+": candidate reference frame and clock differ from the fleet geometry")
    attempts = []
    for duration in durations:
        for offset in offsets:
            if len(attempts) >= maximum_candidates:
                return PatchRefugeSearchResult(None,(),(),tuple(attempts))
            references,rebound = {},{}
            try:
                displacement = tuple(offset[0]*a+offset[1]*b for a,b in zip(u,v))
                for key in ids:
                    ctx,state,direction = contexts[key],states[key],directions[key]
                    goal = tuple(p+distance*n+d for p,n,d in zip(state[:3],direction,displacement))
                    reference = C2RefugeReference.retreat(start=q,retreat_end=q+duration,hold_end=q+duration+hold,
                        initial_state=state,refuge_position=goal,frame=ctx.frame,clock_id=ctx.clock_id)
                    references[key] = reference
                    rebound[key] = replace(ctx,reference_id=reference.reference_id)
                plan = construct_refuge_plan(references=references,contexts=rebound,drift_adapter=drift_adapter,
                    geometry_oracle=geometry_oracle,propulsion=propulsion,**tracking_options)
                reasons = tuple(dict.fromkeys(plan.certificate.reasons))
                if not reasons and not plan.contains(initial_enclosures,q):
                    reasons = ("Complete supplied initial state boxes do not enter every candidate feedback tube",)
                if not reasons and fuel_budget is not None:
                    try:fuel_budget.require(plan.remaining_fuel(q))
                    except ValueError as error:reasons=(str(error),)
                if reasons:
                    attempts.append(PatchRefugeCandidateAttempt(duration,hold,offset,"REFUSED",reasons))
                    continue
                attempts.append(PatchRefugeCandidateAttempt(duration,hold,offset,"CONDITIONAL_PROVEN",(),plan.fingerprint))
                return PatchRefugeSearchResult(plan,tuple(references.items()),tuple(rebound.items()),tuple(attempts))
            except Exception as error:
                attempts.append(PatchRefugeCandidateAttempt(duration,hold,offset,"REFUSED",
                    (type(error).__name__+": "+str(error),)))
    return PatchRefugeSearchResult(None,(),(),tuple(attempts))


class BlackoutObservationPropagator:
    """Finite sensor-blackout interval propagation, without a solver/flow service.

    Known applied controls plus globally proved actual-drift boxes propagate a
    valid state enclosure. Its midpoint becomes a virtual measurement only
    while the installed controller's sensing budget and blackout horizon hold.
    A horizon violation is an assumption failure, not an infinite-safe pause.
    """
    def __init__(self, observations, *, maximum_blackout_q):
        self.boxes = {key:value.state_enclosure for key,value in observations.items()}
        self.q = next(iter(observations.values())).q
        if any(value.q != self.q for value in observations.values()):
            raise ValueError("Blackout observer fleet must share one sample clock")
        self.last_sensor_q = self.q
        self.maximum_blackout_q = _exact(maximum_blackout_q)
        if self.maximum_blackout_q <= 0:
            raise ValueError("Declare a positive finite blackout horizon")

    def advance(self,decision,plan):
        if decision.source not in ("FALLBACK_TRACK","FALLBACK_HOLD"):
            raise ValueError("Performance navigation requires its actual flow proof; this propagator retains fallback invariance only")
        if decision.start != self.q:
            raise ValueError("Blackout propagation clock is noncontiguous")
        h = decision.hold_width
        if self.q+h-self.last_sensor_q > self.maximum_blackout_q:
            raise ValueError("Declared sensor-blackout horizon exhausted")
        segment,inputs = plan.segment_at(self.q,h),dict(decision.applied_inputs)
        updated,observations = {},{}
        for key,controller in segment.controllers:
            b,box = controller.certificate.spec,self.boxes[key]
            if b.context.nominal_model_id != ZERO_NOMINAL_ID:
                raise ValueError("Blackout acceleration envelope requires the actual-zero-nominal drift adapter")
            drift = controller.certificate.drift_bounds.mismatch_abs
            applied = inputs[key]
            al = tuple(u-w for u,w in zip(applied.lower,drift))
            ah = tuple(u+w for u,w in zip(applied.upper,drift))
            lo = tuple(p+h*v+h*h*a/2 for p,v,a in zip(box.lower[:3],box.lower[3:],al))+tuple(v+h*a for v,a in zip(box.lower[3:],al))
            hi = tuple(p+h*v+h*h*a/2 for p,v,a in zip(box.upper[:3],box.upper[3:],ah))+tuple(v+h*a for v,a in zip(box.upper[3:],ah))
            result = IntervalBox(lo,hi)
            measured = tuple((a+c)/2 for a,c in zip(lo,hi))
            observation = RefugeObservation(self.q+h,measured,result,b.context.sensor_id,b.context.frame,b.context.clock_id)
            # The next segment may change gains/tube, while its exact region
            # inclusion was already proved by the complete plan constructor.
            next_controller = dict(plan.segment_at(self.q+h).controllers)[key]
            next_controller.observation_box(observation,self.q+h)
            updated[key],observations[key] = result,observation
        self.boxes,self.q = updated,self.q+h
        return observations

    def correct(self,observations):
        if set(observations) != set(self.boxes) or any(value.q != self.q for value in observations.values()):
            raise ValueError("Recovered sensor data have a different fleet/clock")
        updated = {}
        for key,value in observations.items():
            old,new = self.boxes[key],value.state_enclosure
            lo,hi = tuple(max(a,b) for a,b in zip(old.lower,new.lower)),tuple(min(a,b) for a,b in zip(old.upper,new.upper))
            if any(a>b for a,b in zip(lo,hi)):
                raise ValueError("Recovered sensor contradicts propagated blackout enclosure")
            updated[key] = IntervalBox(lo,hi)
        self.boxes,self.last_sensor_q = updated,self.q
