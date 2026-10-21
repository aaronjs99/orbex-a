"""Conditional quaternion/Euler inclusions driven by nonlinear ephemeris.

The initial quaternion box must contain a unit physical quaternion. Unit norm
is preserved by the exact ODE; ambient interval candidates need not be unit.
Only central spherical gravity-gradient torque (or zero torque) is declared.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import math
from pathlib import Path

from orbexa.core.dynamics.reference_frame import exact, fingerprint
from orbexa.core.dynamics.relative import cross, matvec, transpose, radial
from orbexa.control.certification.reachability import IntervalBox, EnclosureStatus, ReachabilityConfig, _ARB_LOCK, _ProofUnresolved
from orbexa.control.certification.nonlinear import _Arithmetic, _target_box, TargetEphemerisCertificate, FRAME

_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _matrix(quaternion, arithmetic):
    x,y,z,w = quaternion
    square = arithmetic.square
    return ((1-2*(square(y)+square(z)),2*(x*y-z*w),2*(x*z+y*w)),
            (2*(x*y+z*w),1-2*(square(x)+square(z)),2*(y*z-x*w)),
            (2*(x*z-y*w),2*(y*z+x*w),1-2*(square(x)+square(y))))


def _inertia(value):
    matrix = tuple(tuple(map(exact,row)) for row in value)
    if len(matrix)!=3 or any(len(row)!=3 for row in matrix):
        raise ValueError('Inertia must be a three by three exact declaration')
    if any(matrix[i][j]!=matrix[j][i] for i in range(3) for j in range(3)):
        raise ValueError('Inertia must be symmetric')
    a,b,c = matrix[0];_,d,e = matrix[1];_,_,f = matrix[2]
    determinant = a*(d*f-e*e)-b*(b*f-c*e)+c*(b*e-c*d)
    if a<=0 or a*d-b*b<=0 or determinant<=0:
        raise ValueError('Inertia must be exactly positive definite')
    inverse = ((d*f-e*e,c*e-b*f,b*e-c*d),
               (c*e-b*f,a*f-c*c,b*c-a*e),
               (b*e-c*d,b*c-a*e,a*d-b*b))
    return matrix,tuple(tuple(x/determinant for x in row) for row in inverse)


def _sqrt_upper(value):
    value = exact(value)
    if value < 0:
        raise ValueError('Nonnegative square root required')
    scale = 10**15
    root = math.isqrt(value.numerator*scale*scale//value.denominator)
    if root*root*value.denominator < value.numerator*scale*scale:
        root += 1
    return Fraction(root,scale)


@dataclass(frozen=True)
class AttitudeSegment:
    offset: tuple
    anomaly: tuple
    initial_state: IntervalBox
    swept_state: IntervalBox
    endpoint: IntervalBox
    picard_image: IntervalBox
    derivative: IntervalBox
    iterations: int

    def as_dict(self):
        return dict(offset=tuple(map(str,self.offset)),anomaly=tuple(map(str,self.anomaly)),
            **{key:getattr(self,key).as_dict() for key in
               ('initial_state','swept_state','endpoint','picard_image','derivative')},iterations=self.iterations)


@dataclass(frozen=True)
class AttitudeFlowCertificate:
    status: EnclosureStatus
    reason: str
    initial_state: IntervalBox
    endpoint: IntervalBox | None
    swept_state: IntervalBox | None
    segments: tuple
    reference_rotation_at_start: tuple | None
    initial_rotation_operator_error: object | None
    relative_angular_rate_per_anomaly: object | None
    context: dict

    @property
    def proven(self):
        return self.status == EnclosureStatus.PROVEN

    @property
    def fingerprint(self):
        return fingerprint(self.as_dict())

    def state_at(self,query):
        """Point inclusion from the certified segment integral, not its sweep.

        Each exact trajectory satisfies ``x(q)=x(a)+(q-a) f(Y)`` inside a
        proved Picard segment. Intersecting this image with its swept box
        retains that trajectory. Exact endpoints use the endpoint witness.
        This does not narrow any whole-window occupied-geometry enclosure.
        """
        if not self.proven:
            raise ValueError('An unresolved attitude has no point inclusion')
        q=exact(query)
        start=tuple(map(exact,self.context['start_q']))
        width=tuple(map(exact,self.context['width_q']))
        if start[0]!=start[1] or width[0]!=width[1] or not start[0]<=q<=start[0]+width[0]:
            raise ValueError('Attitude point query requires exact covered clock scope')
        previous=self.initial_state;cursor=start[0];result=self.initial_state if q==start[0] else None
        for segment in self.segments:
            lo,hi=segment.anomaly
            if (lo!=cursor or hi<=lo or not segment.initial_state.contains(previous)
                    or segment.initial_state.dimension!=7 or segment.derivative.dimension!=7):
                raise ValueError('Attitude point query found an incomplete segment chain')
            if lo<=q<=hi:
                if q==lo:box=segment.initial_state
                elif q==hi:box=segment.endpoint
                else:
                    h=q-lo
                    image=IntervalBox(tuple(x+h*f for x,f in zip(segment.initial_state.lower,segment.derivative.lower)),
                                      tuple(x+h*f for x,f in zip(segment.initial_state.upper,segment.derivative.upper)))
                    box=IntervalBox(tuple(max(a,b) for a,b in zip(image.lower,segment.swept_state.lower)),
                                    tuple(min(a,b) for a,b in zip(image.upper,segment.swept_state.upper)))
                result=box if result is None else IntervalBox(
                    tuple(max(a,b) for a,b in zip(result.lower,box.lower)),
                    tuple(min(a,b) for a,b in zip(result.upper,box.upper)))
            previous,cursor=segment.endpoint,hi
        if cursor!=start[0]+width[0] or (width[0] and previous!=self.endpoint):
            raise ValueError('Attitude point query found incomplete endpoint coverage')
        if not width[0]:result=self.initial_state
        if result is None:raise ValueError('No certified attitude segment covers the point')
        return result

    def as_dict(self):
        return dict(status=self.status.value,reason=self.reason,initial_state=self.initial_state.as_dict(),
            endpoint=None if self.endpoint is None else self.endpoint.as_dict(),
            swept_state=None if self.swept_state is None else self.swept_state.as_dict(),
            segments=tuple(segment.as_dict() for segment in self.segments),
            reference_rotation_at_start=None if self.reference_rotation_at_start is None else
                tuple(tuple(map(str,row)) for row in self.reference_rotation_at_start),
            initial_rotation_operator_error=None if self.initial_rotation_operator_error is None else str(self.initial_rotation_operator_error),
            relative_angular_rate_per_anomaly=None if self.relative_angular_rate_per_anomaly is None else str(self.relative_angular_rate_per_anomaly),
            context=self.context)

    def occupied_bounds(self,geometry):
        from orbexa.control.certification.geometry import RotatingCompositeBounds
        if not self.proven:
            raise ValueError('Unresolved attitude does not supply an occupied-pose bound')
        return RotatingCompositeBounds(geometry=geometry,
            orientation_matrix_at_start=self.reference_rotation_at_start,
            orientation_matrix_error=self.initial_rotation_operator_error,
            angular_rate_per_anomaly=self.relative_angular_rate_per_anomaly,
            frame=FRAME,scope='Quaternion/Euler Arb inclusion on nonlinear target ephemeris; unit initial attitude assumption; '+self.fingerprint)


class AttitudeFlowEncloser:
    def __init__(self,config=None):
        self.config = config or ReachabilityConfig()

    def enclose(self,initial_quaternion_box,initial_body_rate_box,target_flow,inertia,model,clock,gravity_gradient=True):
        if initial_quaternion_box.dimension!=4 or initial_body_rate_box.dimension!=3:
            raise ValueError('Attitude uses four scalar-last quaternion and three body-rate states')
        matrix,inverse = _inertia(inertia)
        initial = IntervalBox(initial_quaternion_box.lower+initial_body_rate_box.lower,
                              initial_quaternion_box.upper+initial_body_rate_box.upper)
        context = dict(model='nonlinear-ephemeris-quaternion-euler-v1',frame=FRAME,
            source_sha256=_SOURCE_SHA256,plant_fingerprint=model.fingerprint,
            target_model_fingerprint=model.target_fingerprint,clock_fingerprint=clock.fingerprint,
            target_flow_fingerprint=target_flow.fingerprint,inertia=tuple(tuple(map(str,row)) for row in matrix),
            gravity_gradient=bool(gravity_gradient),quaternion_order='xyzw',
            start_q=tuple(map(str,target_flow.start_anomaly)),width_q=tuple(map(str,target_flow.hold_width)),
            assumptions=['Initial box contains a unit physical quaternion',
                         'Constant inertia and central spherical gravity-gradient torque only'])
        def failure(status,reason):
            return AttitudeFlowCertificate(status,reason,initial,None,None,(),None,None,None,context)
        try:
            import flint
        except ImportError as error:
            return failure(EnclosureStatus.UNAVAILABLE,str(error))
        try:
            if (not target_flow.proven or not isinstance(target_flow,TargetEphemerisCertificate)
                    or target_flow.model_bounds.get('target_model_fingerprint')!=model.target_fingerprint
                    or target_flow.model_bounds.get('clock_fingerprint')!=clock.fingerprint
                    or target_flow.hold_width[0]!=target_flow.hold_width[1]):
                raise ValueError('Attitude requires a same-model exact-width nonlinear target flow')
            with _ARB_LOCK,flint.ctx.workprec(self.config.precision_bits):
                ar = _Arithmetic(flint)
                qnorm = sum(ar.square(v) for v in ar.vector(initial_quaternion_box))
                if not qnorm.contains(ar.scalar(1)):
                    raise ValueError('Initial quaternion domain contains no unit norm')
                iballs = tuple(tuple(ar.scalar(v) for v in row) for row in matrix)
                invballs = tuple(tuple(ar.scalar(v) for v in row) for row in inverse)
                # Collect equal quadratic monomials before evaluating intervals.
                # For principal inertia this yields (Iy-Iz)*wy*wz exactly.
                quadratic=[]
                for component in range(3):
                    coefficients={}
                    j,k=(component+1)%3,(component+2)%3
                    for axis,sign,row in ((j,1,matrix[k]),(k,-1,matrix[j])):
                        for other,coefficient in enumerate(row):
                            monomial=tuple(sorted((axis,other)))
                            coefficients[monomial]=coefficients.get(monomial,exact(0))+sign*coefficient
                    quadratic.append({pair:value for pair,value in coefficients.items() if value})
                def field(qb,state):
                    q,omega = state[:4],state[4:]
                    _,rate,_ = clock.evaluate(ar.interval(qb),ar)
                    tau = (ar.scalar(0),)*3
                    if gravity_gradient:
                        target = ar.vector(_target_box(target_flow,qb))
                        radius = radial(target[:3],model,ar)
                        radial_body = matvec(transpose(_matrix(q,ar)),tuple(v/radius for v in target[:3]))
                        # Continuous clipped off-manifold extension. Every unit
                        # physical quaternion already has each radial component
                        # in [-1,1], so this changes no physical trajectory.
                        rb=ar.box(radial_body)
                        radial_body=ar.vector(IntervalBox(
                            tuple(max(exact(-1),min(exact(1),v)) for v in rb.lower),
                            tuple(max(exact(-1),min(exact(1),v)) for v in rb.upper)))
                        tau = tuple(3*ar.interval(model.mu_m3_s2)*v/radius**3
                                    for v in cross(radial_body,matvec(iballs,radial_body)))
                    gyroscopic=tuple(sum(ar.scalar(coefficient)*(ar.square(omega[i]) if i==j else omega[i]*omega[j])
                        for (i,j),coefficient in coefficients.items()) for coefficients in quadratic)
                    euler = matvec(invballs,tuple(a-b for a,b in zip(tau,gyroscopic)))
                    vector = tuple((q[3]*w+c)/2/rate for w,c in zip(omega,cross(q[:3],omega)))
                    scalar = -sum(v*w for v,w in zip(q[:3],omega))/2/rate
                    return vector+(scalar,)+tuple(v/rate for v in euler)
                current,offset,final = ar.vector(initial),exact(0),target_flow.hold_width[1]
                proposed = min(self.config.max_step,final) if final else exact(0)
                segments,swept,rate_upper = [],initial,exact(0)
                while offset<final:
                    if len(segments)>=self.config.max_substeps:
                        raise _ProofUnresolved('Attitude inclusion exhausted max_substeps')
                    remaining = final-offset;ratio = remaining/proposed
                    width = remaining/((ratio.numerator+ratio.denominator-1)//ratio.denominator)
                    witness = None
                    while width>=self.config.min_step:
                        qb = (target_flow.start_anomaly[0]+offset,target_flow.start_anomaly[1]+offset+width)
                        try:
                            witness = self._picard(current,qb,width,field,ar)
                        except (ValueError,ArithmeticError):
                            witness = None
                        if witness is not None:break
                        width /= 2
                    if witness is None:
                        raise _ProofUnresolved('Attitude Picard inclusion unresolved at min_step')
                    candidate,image,derivative,endpoint,iterations = witness
                    boxes = tuple(ar.box(value) for value in (current,candidate,endpoint,image,derivative))
                    segments.append(AttitudeSegment((offset,offset+width),qb,*boxes,iterations))
                    swept = swept.hull(boxes[1])
                    omega_norm = _sqrt_upper(sum(max(abs(lo),abs(hi))**2 for lo,hi in zip(boxes[1].lower[4:],boxes[1].upper[4:])))
                    nu = ar.box((clock.evaluate(ar.interval(qb),ar)[1],)).lower[0]
                    rate_upper = max(rate_upper,1+omega_norm/nu)
                    current,offset,proposed = endpoint,offset+width,min(self.config.max_step,2*width)
                if not final:
                    field(target_flow.start_anomaly,current)
                    omega_norm = _sqrt_upper(sum(max(abs(lo),abs(hi))**2 for lo,hi in zip(initial.lower[4:],initial.upper[4:])))
                    nu = ar.box((clock.evaluate(ar.interval(target_flow.start_anomaly),ar)[1],)).lower[0]
                    rate_upper = 1+omega_norm/nu
                # A proper numerical center with a fully outward Frobenius
                # error bound against all compatible physical start rotations.
                import numpy as np
                from scipy.spatial.transform import Rotation
                midpoint = np.array([float((a+b)/2) for a,b in zip(initial.lower[:4],initial.upper[:4])])
                if np.linalg.norm(midpoint)==0:
                    midpoint = np.array([0.,0.,0.,1.])
                inertial_center = Rotation.from_quat(midpoint).as_matrix()
                ref_center = np.asarray(clock.point(float(target_flow.start_anomaly[0])).rotation).T@inertial_center
                reference = clock.evaluate(ar.interval(target_flow.start_anomaly),ar)[0]
                actual = _matrix(ar.vector(initial)[:4],ar)
                actual_ref = tuple(tuple(sum(reference[k][i]*actual[k][j] for k in range(3)) for j in range(3)) for i in range(3))
                rb = ar.box(tuple(v for row in actual_ref for v in row))
                center = tuple(tuple(exact(v) for v in row) for row in ref_center)
                error2 = sum(max(abs(lo-center[i//3][i%3]),abs(hi-center[i//3][i%3]))**2
                             for i,(lo,hi) in enumerate(zip(rb.lower,rb.upper)))
                return AttitudeFlowCertificate(EnclosureStatus.PROVEN,
                    'Every quaternion/Euler substep satisfies strict Arb Picard self-inclusion',initial,
                    ar.box(current),swept,tuple(segments),center,_sqrt_upper(error2),rate_upper,context)
        except (ValueError,ArithmeticError,_ProofUnresolved) as error:
            return failure(EnclosureStatus.UNRESOLVED,str(error))

    def _picard(self,initial,qb,width,field,ar):
        offset,h = ar.interval((0,width)),ar.scalar(width)
        first = field(qb,initial)
        candidate = ar.inflate(tuple(x.union(x+offset*f) for x,f in zip(initial,first)),self.config)
        for iteration in range(1,self.config.max_picard_iterations+1):
            derivative = field(qb,candidate)
            image = tuple(x+offset*f for x,f in zip(initial,derivative))
            if not all(value.is_finite() for value in image):return None
            if all(y.contains_interior(value) for y,value in zip(candidate,image)):
                return candidate,image,derivative,tuple(x+h*f for x,f in zip(initial,derivative)),iteration
            candidate = ar.inflate(tuple(y.union(value) for y,value in zip(candidate,image)),self.config)
        return None

    def enclose_observation(self,rotation_reference,operator_error,body_rate_box,target_flow,inertia,model,clock,gravity_gradient=True):
        """Lift a processed reference-frame rotation support into quaternion space."""
        import numpy as np
        from scipy.spatial.transform import Rotation
        error = exact(operator_error)
        if error<0:
            raise ValueError('Observation operator error must be nonnegative')
        q0 = target_flow.start_anomaly[0]
        observed = tuple(tuple(map(exact,row)) for row in rotation_reference)
        matrix = np.asarray(observed,dtype=float)
        center = Rotation.from_matrix(np.asarray(clock.point(float(q0)).rotation)@matrix).as_quat()
        # Certify the projection/lift error; do not assume a fixed floating
        # allowance for a possibly nonorthogonal processed observation.
        import flint
        with _ARB_LOCK,flint.ctx.workprec(self.config.precision_bits):
            ar = _Arithmetic(flint)
            raw = tuple(ar.scalar(exact(v)) for v in center)
            norm = ar.sqrt(sum(ar.square(v) for v in raw))
            unit = tuple(v/norm for v in raw)
            inertial = _matrix(unit,ar)
            reference = clock.evaluate(ar.interval(target_flow.start_anomaly),ar)[0]
            projected = tuple(tuple(sum(reference[k][i]*inertial[k][j] for k in range(3))
                                    for j in range(3)) for i in range(3))
            difference = ar.box(tuple(projected[i][j]-ar.scalar(observed[i][j])
                                      for i in range(3) for j in range(3)))
            projection_error = _sqrt_upper(sum(max(abs(lo),abs(hi))**2
                                               for lo,hi in zip(difference.lower,difference.upper)))
            radius = error+projection_error
            if radius>=2:
                quaternion = IntervalBox((-1,)*4,(1,)*4)
            else:
                ub = ar.box(unit)
                quaternion = IntervalBox(tuple(max(exact(-1),v-radius) for v in ub.lower),
                                         tuple(min(exact(1),v+radius) for v in ub.upper))
        result = self.enclose(quaternion,body_rate_box,target_flow,inertia,model,clock,gravity_gradient)
        if result.proven:
            from dataclasses import replace
            # Preserve the tighter directly supplied pose support at start;
            # its physical existence remains the explicit sensor assumption.
            result = replace(result,reference_rotation_at_start=observed,
                initial_rotation_operator_error=error,context=dict(result.context,
                    observation_reference_rotation=tuple(tuple(map(str,row)) for row in observed),
                    observation_operator_error=str(error),quaternion_projection_operator_error=str(projection_error)))
        return result
