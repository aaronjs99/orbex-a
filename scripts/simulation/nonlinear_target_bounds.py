"""Validated moving-goal errors about an observation-driven numerical preview."""
from __future__ import annotations

from fractions import Fraction
import hashlib
from pathlib import Path
from orbexa.control.reachability import IntervalBox,_ARB_LOCK
from orbexa.control.nonlinear_flow import _Arithmetic,TargetEphemerisCertificate
from orbexa.control.attitude_flow import _matrix,_sqrt_upper,_SOURCE_SHA256 as _ATTITUDE_SOURCE_SHA256
from orbexa.core.reference_frame import exact,fingerprint
from orbexa.core.nonlinear_relative import matvec,transpose,cross,z

_SOURCE_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def exact_norm_upper(values):
    """Outward Euclidean norm of a vector of exact declarations."""
    return _sqrt_upper(sum(exact(value)**2 for value in values))


def _norm_box(box):
    return exact_norm_upper(max(abs(lo),abs(hi)) for lo,hi in zip(box.lower,box.upper))


def _multiply(left,right):
    return tuple(tuple(sum(left[i][k]*right[k][j] for k in range(3)) for j in range(3)) for i in range(3))


def _matrix_difference_norm(left,right,ar):
    return _norm_box(ar.box(tuple(left[i][j]-right[i][j] for i in range(3) for j in range(3))))


def _proper_rotation_comparison(attitude,preview,clock,queries,ar):
    """An isometric SO(3) comparison, independent of preview accuracy.

    Let G(t)=G0 exp([w0]t), with G0 exactly proper by normalized declared
    quaternion. For a physical proper R, differentiation of R exp(-[w0]t)
    gives an operator norm bound integral(||w-w0||dt). The actual body-rate
    supports come from the unchanged nonlinear Euler/Picard forecast.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation
    if attitude.context.get('source_sha256')!=_ATTITUDE_SOURCE_SHA256:
        raise ValueError('Proper-rotation comparison requires the current declared attitude backend')
    observed=tuple(tuple(map(exact,row)) for row in attitude.context['observation_reference_rotation'])
    error=exact(attitude.context['observation_operator_error'])
    if (len(observed)!=3 or any(len(row)!=3 for row in observed) or error<0
            or observed!=attitude.reference_rotation_at_start
            or error!=exact(attitude.initial_rotation_operator_error)):
        raise ValueError('Proper-rotation comparison found inconsistent initial pose support')
    q0=exact(attitude.context['start_q'][0])
    if (getattr(preview,'role',None)!='observation_driven_numerical_preview'
            or exact(preview.current_q)!=q0 or preview.reference_clock.fingerprint!=clock.fingerprint
            or preview.physical_model_bounds.fingerprint!=attitude.context['plant_fingerprint']):
        raise ValueError('Proper-rotation comparison requires the current observation-driven preview')
    # The float projection chooses an arbitrary reference. Its exact unit
    # normalization and all defects against the observed support are enclosed.
    raw=Rotation.from_matrix(np.asarray(clock.point(q0).rotation)@np.asarray(observed,dtype=float)).as_quat()
    quaternion=tuple(ar.scalar(value) for value in raw)
    norm=ar.sqrt(sum(ar.square(value) for value in quaternion))
    proper_initial=_matrix(tuple(value/norm for value in quaternion),ar)
    initial_reference=_multiply(transpose(clock.evaluate(ar.scalar(q0),ar)[0]),proper_initial)
    observed_matrix=tuple(tuple(ar.scalar(value) for value in row) for row in observed)
    projection=_matrix_difference_norm(initial_reference,observed_matrix,ar)
    gamma0=error+projection
    omega0=tuple(map(exact,preview.body_angular_velocity_at(float(q0))))
    if len(omega0)!=3:raise ValueError('Comparison body rate has three finite components')
    w=tuple(ar.scalar(value) for value in omega0);x,y,zz=w
    skew=((0,-zz,y),(zz,0,-x),(-y,x,0));skew2=_multiply(skew,skew)
    norm2=sum(value*value for value in omega0)
    rate_norm=ar.sqrt(ar.scalar(norm2)) if norm2 else None
    wanted=sorted(set(queries));maximum=wanted[-1]
    elapsed_lower=elapsed_upper=variation=Fraction(0);cursor=q0;pieces=[]
    prefix={q0:(elapsed_lower,elapsed_upper,variation,0)}
    for index,segment in enumerate(attitude.segments):
        lo,hi=segment.anomaly
        if lo>=maximum:break
        if lo!=cursor:raise ValueError('Proper-rotation comparison found a missing attitude time interval')
        end=min(hi,maximum)
        cuts=[lo]+[q for q in wanted if lo<q<end]+[end]
        delta=IntervalBox(tuple(a-b for a,b in zip(segment.swept_state.lower[4:],omega0)),
                          tuple(a-b for a,b in zip(segment.swept_state.upper[4:],omega0)))
        deviation=_norm_box(delta)
        for start,stop in zip(cuts,cuts[1:]):
            nu=clock.evaluate(ar.interval((start,stop)),ar)[1]
            clock_box=ar.box((nu,))
            if clock_box.lower[0]<=0:raise ValueError('Comparison clock is not proved positive')
            duration=ar.box((ar.scalar(stop-start)/nu,))
            contribution=duration.upper[0]*deviation
            elapsed_lower+=duration.lower[0];elapsed_upper+=duration.upper[0];variation+=contribution
            pieces.append(dict(segment_index=index,q_interval=tuple(map(str,(start,stop))),
                body_rate_deviation_box=delta.as_dict(),body_rate_deviation_norm_upper=str(deviation),
                clock_rate_box=clock_box.as_dict(),duration_s=duration.as_dict(),
                rotation_variation_upper=str(contribution)))
            if stop in wanted:prefix[stop]=(elapsed_lower,elapsed_upper,variation,len(pieces))
        cursor=end
        if end==maximum:break
    if any(q not in prefix for q in wanted):
        raise ValueError('Proper-rotation comparison does not cover every requested event')
    records={}
    for q in wanted:
        lower,upper,integral,count=prefix[q]
        if norm2:
            angle=rate_norm*ar.interval((lower,upper))
            sine=ar.sin(angle)/rate_norm;cosine=(1-ar.cos(angle))/ar.scalar(norm2)
            exponential=tuple(tuple(int(i==j)+sine*skew[i][j]+cosine*skew2[i][j]
                for j in range(3)) for i in range(3))
        else:exponential=tuple(tuple(ar.scalar(int(i==j)) for j in range(3)) for i in range(3))
        comparison=_multiply(transpose(clock.evaluate(ar.scalar(q),ar)[0]),_multiply(proper_initial,exponential))
        nominal=tuple(tuple(ar.scalar(value) for value in row) for row in preview.rotation_matrix_at(float(q)))
        reference_defect=_matrix_difference_norm(comparison,nominal,ar)
        gamma=gamma0+integral+reference_defect
        records[q]=dict(rotation_operator_error_upper=gamma,
            initial_operator_error_upper=gamma0,integrated_rate_variation_upper=integral,
            reference_minus_preview_operator_upper=reference_defect,
            elapsed_time_s=(lower,upper),rate_piece_count=count)
    witness=dict(method='proper-rotation-constant-spin-comparison-v1',backend_source_sha256=_SOURCE_SHA256,
        attitude_flow_fingerprint=attitude.fingerprint,clock_fingerprint=clock.fingerprint,
        plant_fingerprint=attitude.context['plant_fingerprint'],
        target_flow_fingerprint=attitude.context['target_flow_fingerprint'],exact_start_q=str(q0),
        observed_rotation=tuple(tuple(map(str,row)) for row in observed),observation_operator_error=str(error),
        proper_reference_quaternion=tuple(str(exact(value)) for value in raw),
        initial_projection_operator_error=str(projection),constant_body_rate_rad_s=tuple(map(str,omega0)),
        rate_variation_pieces=pieces,
        records=[dict(exact_q=str(q),**{key:(tuple(map(str,value)) if isinstance(value,tuple) else str(value))
            for key,value in values.items()}) for q,values in records.items()],
        assumptions=['Physical initial attitude is proper and lies in the declared observation support',
            'Body rates follow the linked nonlinear Euler/Picard forecast',
            'The defined numerical preview is a reference center; its integration error is explicitly enclosed'])
    witness['identity']=fingerprint(witness)
    return records,witness


def observed_point_goal_bounds(estimate,preview,clock,chasers,*,precision_bits=128):
    """Bind a same-event pose/rate support to the defined preview goal.

    The physical rotation is proper. Its operator error is bounded about the
    raw navigation matrix; the preview may project that matrix to SO(3).
    Center projection and floating point evaluation defects are retained.
    This pointwise result does not replace a swept attitude certificate.
    """
    import flint
    if (estimate.status!='VALID' or estimate.clock_fingerprint!=clock.fingerprint
            or preview.current_q!=exact(estimate.exact_event_q)):
        raise ValueError('Terminal goal bound requires a same-event observed preview')
    q=exact(estimate.exact_event_q)
    nominal=tuple(tuple(map(exact,row)) for row in preview.rotation_matrix_at(float(q)))
    observed=tuple(tuple(map(exact,row)) for row in estimate.target_orientation_matrix)
    gamma=exact(estimate.target_orientation_operator_error)+exact_norm_upper(
        observed[i][j]-nominal[i][j] for i in range(3) for j in range(3))
    omega=tuple(map(exact,preview.body_angular_velocity_at(float(q))))
    rate_error=exact_norm_upper(max(abs(lo-center),abs(hi-center)) for lo,hi,center in
        zip(estimate.target_angular_velocity_box.lower,estimate.target_angular_velocity_box.upper,omega))
    nu_lower=clock.interval_bounds((q,q),precision_bits)['rate_s_inv'].lower[0]
    if nu_lower<=0:raise ValueError('Terminal clock rate is not proved positive')
    goals={}
    with _ARB_LOCK,flint.ctx.workprec(precision_bits):
        ar=_Arithmetic(flint)
        _,nu,_=clock.evaluate(ar.scalar(q),ar)
        rotation=tuple(tuple(ar.scalar(v) for v in row) for row in nominal)
        rates=tuple(ar.scalar(v) for v in omega)
        for chaser in chasers:
            point=tuple(map(exact,chaser.docking_point_body));length=exact_norm_upper(point)
            p=matvec(rotation,tuple(ar.scalar(v) for v in point))
            tangential=matvec(rotation,cross(rates,tuple(ar.scalar(v) for v in point)))
            velocity=tuple(v/nu-a for v,a in zip(tangential,z(p)))
            center=preview.docking_state(chaser.docking_point_body,float(q))
            defect=ar.box(tuple(value-ar.scalar(c) for value,c in zip(p+velocity,center)))
            pe=exact_norm_upper(max(abs(lo),abs(hi)) for lo,hi in zip(defect.lower[:3],defect.upper[:3]))
            ve=exact_norm_upper(max(abs(lo),abs(hi)) for lo,hi in zip(defect.lower[3:],defect.upper[3:]))
            position=gamma*length+pe
            speed=gamma*length+(gamma*exact_norm_upper(omega)+rate_error)*length/nu_lower+ve
            goals[chaser.chaser_id]=dict(position_error_norm_upper=position,
                velocity_error_norm_upper=speed,numerical_position_defect=pe,numerical_velocity_defect=ve)
    return dict(exact_q=str(q),rotation_operator_error_upper=gamma,rate_error_norm_upper=rate_error,
        clock_rate_lower=nu_lower,goals=goals,clock_fingerprint=clock.fingerprint,
        method='observed-proper-rotation-point-goal-bound-v1')


def preview_error_bounds(attitude,preview,clock,chasers,query_anomalies,*,precision_bits=128,
                         proper_rotation_comparison=True,target_flow=None):
    """Enclose actual goal minus the defined numerical reference at checked nodes.

    Independent quaternion images and a proper-rotation comparison both bind
    the same physical attitude to the defined numerical preview. Their minimum
    is valid. Whole-window geometry still uses the original swept witnesses.
    Preview integration accuracy is never assumed to establish either bound.
    """
    import flint
    if (not attitude.proven or attitude.context['clock_fingerprint']!=clock.fingerprint
            or not isinstance(proper_rotation_comparison,bool)):
        raise ValueError('A proved same-clock physical attitude forecast is required')
    if target_flow is not None:
        if (not isinstance(target_flow,TargetEphemerisCertificate) or not target_flow.proven
                or target_flow.fingerprint!=attitude.context['target_flow_fingerprint']
                or target_flow.start_anomaly!=tuple(map(exact,attitude.context['start_q']))
                or target_flow.hold_width!=tuple(map(exact,attitude.context['width_q']))
                or fingerprint(target_flow.model_bounds.get('plant'))!=attitude.context['plant_fingerprint']
                or any(target_flow.model_bounds.get(key)!=attitude.context[key] for key in
                    ('target_model_fingerprint','clock_fingerprint'))):
            raise ValueError('Moving-goal forecast requires the exact linked full target certificate')
    elif proper_rotation_comparison and 'observation_reference_rotation' in attitude.context:
        raise ValueError('Proper-rotation comparison requires an explicit full target certificate')
    queries=tuple(map(exact,query_anomalies))
    if not queries:raise ValueError('Moving-goal prediction requires at least one checked event')
    source_identity=attitude.fingerprint;clock_identity=clock.fingerprint
    target_identity=None if target_flow is None else target_flow.fingerprint
    states={q:attitude.state_at(q) for q in set(queries)}
    records=[];maximum_rotation=Fraction(0);maximum_position=Fraction(0);maximum_velocity=Fraction(0)
    comparison_records={};comparison_witness=None
    with _ARB_LOCK,flint.ctx.workprec(precision_bits):
        ar=_Arithmetic(flint)
        if proper_rotation_comparison and 'observation_reference_rotation' in attitude.context:
            comparison_records,comparison_witness=_proper_rotation_comparison(attitude,preview,clock,queries,ar)
        for q in queries:
            state=states[q]
            values=ar.vector(state);quaternion,omega=values[:4],values[4:]
            # The exact physical quaternion is unit; the polynomial matrix
            # already maps it correctly. Ambient nonunit interval points only
            # widen the result, so no point normalization is inferred.
            inertial=_matrix(quaternion,ar)
            reference,nu,_=clock.evaluate(ar.scalar(q),ar)
            rotation=tuple(tuple(sum(reference[k][i]*inertial[k][j] for k in range(3))
                for j in range(3)) for i in range(3))
            nominal=tuple(tuple(map(exact,row)) for row in preview.rotation_matrix_at(float(q)))
            rotation_box=ar.box(tuple(v-ar.scalar(nominal[i][j]) for i,row in enumerate(rotation) for j,v in enumerate(row)))
            raw_gamma=_norm_box(rotation_box)
            comparison=comparison_records.get(q)
            gamma=raw_gamma if comparison is None else min(raw_gamma,comparison['rotation_operator_error_upper'])
            maximum_rotation=max(maximum_rotation,gamma)
            nominal_rotation=tuple(tuple(ar.scalar(value) for value in row) for row in nominal)
            nominal_rate=tuple(map(exact,preview.body_angular_velocity_at(float(q))))
            rate_difference=_norm_box(IntervalBox(tuple(a-b for a,b in zip(state.lower[4:],nominal_rate)),
                tuple(a-b for a,b in zip(state.upper[4:],nominal_rate))))
            rate_norm=exact_norm_upper(nominal_rate);nu_lower=ar.box((nu,)).lower[0]
            if nu_lower<=0:raise ValueError('Moving-goal clock is not proved positive')
            goal_errors={}
            for chaser in chasers:
                point=tuple(ar.scalar(v) for v in chaser.docking_point_body)
                position=matvec(rotation,point)
                tangent=matvec(rotation,cross(omega,point))
                zp=z(position)
                velocity=tuple(a/nu-b for a,b in zip(tangent,zp))
                center=preview.docking_state(chaser.docking_point_body,float(q))
                difference=ar.box(tuple(v-ar.scalar(c) for v,c in zip(position+velocity,center)))
                raw_pnorm=_norm_box(difference.slice(0,3));raw_vnorm=_norm_box(difference.slice(3,6))
                # R_actual is proper. In the rate term use
                # (R_actual-R_bar) w_bar + R_actual (w_actual-w_bar),
                # so no unphysical Cartesian matrix corners multiply rate error.
                pbar=matvec(nominal_rotation,point)
                vbar=tuple(a/nu-b for a,b in zip(matvec(nominal_rotation,cross(
                    tuple(ar.scalar(value) for value in nominal_rate),point)),z(pbar)))
                defect=ar.box(tuple(value-ar.scalar(c) for value,c in zip(pbar+vbar,center)))
                pe=_norm_box(defect.slice(0,3));ve=_norm_box(defect.slice(3,6))
                length=exact_norm_upper(chaser.docking_point_body)
                norm_position=gamma*length+pe
                norm_velocity=(gamma+(gamma*rate_norm+rate_difference)/nu_lower)*length+ve
                pnorm=min(raw_pnorm,norm_position);vnorm=min(raw_vnorm,norm_velocity)
                maximum_position=max(maximum_position,pnorm);maximum_velocity=max(maximum_velocity,vnorm)
                goal_errors[chaser.chaser_id]=dict(component_error=difference.as_dict(),position_norm_upper=str(pnorm),
                    velocity_norm_upper=str(vnorm),raw_quaternion_position_norm_upper=str(raw_pnorm),
                    raw_quaternion_velocity_norm_upper=str(raw_vnorm),proper_rotation_position_norm_upper=str(norm_position),
                    proper_rotation_velocity_norm_upper=str(norm_velocity),numerical_position_defect=str(pe),
                    numerical_velocity_defect=str(ve))
            records.append(dict(exact_q=str(q),rotation_operator_error_upper=str(gamma),
                raw_quaternion_rotation_operator_upper=str(raw_gamma),point_state=state.as_dict(),
                rate_error_norm_upper=str(rate_difference),clock_rate_lower=str(nu_lower),
                numerical_preview_rotation=tuple(tuple(map(str,row)) for row in nominal),
                numerical_preview_body_rate=tuple(map(str,nominal_rate)),goals=goal_errors))
    if (attitude.fingerprint!=source_identity or clock.fingerprint!=clock_identity
            or (target_flow is not None and target_flow.fingerprint!=target_identity)):
        raise ValueError('Moving-goal proof sources changed during evaluation')
    return dict(rotation_operator_error_upper=maximum_rotation,position_error_norm_upper=maximum_position,
        velocity_error_norm_upper=maximum_velocity,records=records,
        attitude_flow_fingerprint=source_identity,clock_fingerprint=clock_identity,
        target_flow_fingerprint=attitude.context['target_flow_fingerprint'],
        backend_source_sha256=_SOURCE_SHA256,proper_rotation_comparison=comparison_witness,
        method='minimum-independent-Arb-point-quaternion-and-proper-rotation-reference-bounds-v2')
