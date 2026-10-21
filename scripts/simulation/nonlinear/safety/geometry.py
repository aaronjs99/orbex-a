"""Complete occupied geometry from declared target pose enclosures."""
from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
from fractions import Fraction
from math import gcd
from typing import Any
from orbexa.control.certification.reachability import IntervalBox, _exact, _ARB_LOCK
from orbexa.control.fallback.refuge import _identity, _scope, fingerprint
from orbexa.control.fallback.geometry import OccupiedGeometryBounds, OccupiedProjection
from orbexa.control.certification.rotation import all_rotation_component_bounds
from orbexa.control.certification.nonlinear import _Arithmetic

from orbexa.simulation.nonlinear.safety.cache import _FrozenProofIdentityCache


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
        from orbexa.control.certification.attitude import AttitudeFlowCertificate, AttitudeSegment, _matrix, _SOURCE_SHA256 as attitude_source
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
