"""Sensed sliding-tube algebra, plan checks and certified endpoint transfer.

These operations take declared model/measurement bounds. They never sample truth.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
import numpy as np

from orbexa.core.reference_frame import exact, fingerprint
from orbexa.control.reachability import IntervalBox
from orbexa.control.sample_hold import sample_hold_bounds, SENSED_TUBE_MODEL_VERSION

_TUBE_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

def _upper(value):
    result = float(value)
    return np.nextafter(result,np.inf) if Fraction(result)<value else result


def _exact_sensed_stage(spec,stage):
    """Reevaluate the existing nonnegative v2 algebra over exact rationals."""
    vector=lambda value:tuple(map(exact,value))
    matrix=lambda value:tuple(vector(row) for row in value)
    dot=lambda rows,values:tuple(sum(a*b for a,b in zip(row,values)) for row in rows)
    a,delta,variation=map(matrix,(spec.A_abs[0],spec.delta_A[0],spec.A_variation[0]))
    lam,phi,omega,alpha=map(vector,(spec.lambda_gain,stage['phi_peak'],stage['omega_peak'],stage['alpha']))
    observation=vector(spec.measurement_error_abs)
    base=tuple(value+d+eta for value,d,eta in zip(dot(delta,vector(stage['nominal_enclosure'])),
        vector(spec.delta_d[0]),vector(spec.eta)))
    velocity=tuple(p+l*o for p,l,o in zip(phi,lam,omega));error=omega+velocity
    combined=tuple(tuple(x+y for x,y in zip(row,drow)) for row,drow in zip(a,delta))
    drift,uncertain=dot(combined,error),dot(delta,error)
    observed=tuple(value+l*n for value,l,n in zip(dot(combined,observation),lam,observation[3:]))
    reserve=tuple(d+l*v+k*p+n for d,l,v,k,p,n in zip(drift,lam,velocity,alpha,phi,observed))
    sensed=tuple(n+2*u+k*(nv+l*np_) for n,u,k,nv,l,np_ in
        zip(observed,uncertain,alpha,observation[3:],lam,observation[:3]))
    if not any(observation):sensed=(Fraction(0),)*3
    dv=tuple(d+b+r for d,b,r in zip(drift,base,reserve));de=velocity+dv
    ds=tuple(d+l*v for d,l,v in zip(dv,lam,velocity))
    tau=exact(spec.hold_width)
    rho=tuple(2*u+c+tau*(change+l*d+k*s)+n for u,c,change,l,d,k,s,n in
        zip(uncertain,dot(variation,error),dot(a,de),lam,dv,alpha,ds,sensed))
    total=tuple(b+r for b,r in zip(base,rho))
    terms=dict(base_forcing=base,hold_forcing=rho,disturbance_bound=total,velocity_peak=velocity,
        error_peak=error,input_reserve=reserve,velocity_derivative_bound=dv,error_derivative_bound=de,
        sliding_derivative_bound=ds,measurement_forcing=sensed)
    source=dict(A_abs=a,delta_A=delta,A_variation=variation,lambda_gain=lam,phi_peak=phi,
        omega_peak=omega,alpha=alpha,measurement_error_abs=observation,
        nominal_enclosure=vector(stage['nominal_enclosure']),delta_d=vector(spec.delta_d[0]),
        eta=vector(spec.eta),hold_width=tau)
    from orbexa.control.refuge import exact_record
    record=dict(method='exact-rational-sensed-v2-stage-v1',source_sha256=_TUBE_SOURCE_SHA256,
        inputs=exact_record(source),exact_terms=exact_record(terms))
    record['identity']=fingerprint(record)
    return terms,record


def _outward_tube_endpoint(phi,omega,alpha,lam,forcing,h,phi_peak,precision_bits):
    """Enclose the same positive comparison solutions with Arb exponentials."""
    import flint
    from orbexa.control.nonlinear_flow import _Arithmetic
    from orbexa.control.reachability import _ARB_LOCK
    with _ARB_LOCK,flint.ctx.workprec(precision_bits):
        ar=_Arithmetic(flint);time=ar.scalar(h);pn=[];on=[]
        for p,o,a,l,d,peak in zip(phi,omega,alpha,lam,forcing,phi_peak):
            aa,ll=ar.scalar(exact(a)),ar.scalar(exact(l))
            aw=time if exact(a)==0 else -(-aa*time).expm1()/aa
            lw=time if exact(l)==0 else -(-ll*time).expm1()/ll
            pn.append((-aa*time).exp()*ar.scalar(exact(p))+aw*ar.scalar(exact(d)))
            on.append((-ll*time).exp()*ar.scalar(exact(o))+lw*ar.scalar(exact(peak)))
        phi_box,omega_box=ar.box(tuple(pn)),ar.box(tuple(on))
    return (np.array([_upper(v) for v in phi_box.upper]),
        np.array([_upper(v) for v in omega_box.upper]),phi_box,omega_box)


@dataclass(frozen=True)
class PhysicalTubeWitness:
    """An executor-owned conditional first-exit induction, not a sensor box."""
    agent: str
    exact_q: object
    center: tuple
    phi: tuple
    omega: tuple
    lambda_gain: tuple
    model_fingerprint: str
    clock_fingerprint: str
    source: tuple

    def as_dict(self):
        return dict(method='conditional-sensed-physical-tube-induction-v1',agent=self.agent,
            exact_q=str(self.exact_q),center=tuple(map(str,self.center)),phi=tuple(map(str,self.phi)),
            omega=tuple(map(str,self.omega)),lambda_gain=tuple(map(str,self.lambda_gain)),
            model_fingerprint=self.model_fingerprint,clock_fingerprint=self.clock_fingerprint,
            source=dict(self.source),scope='Physical relational tube carried by admitted v2 first-exit and continuous nominal endpoint transfer')

    @cached_property
    def fingerprint(self):return fingerprint(self.as_dict())


def stage_from_plan(state,plan,key,h,*,sliding_gains):
    tube,spec=plan.tube,plan.spec
    if tube.get('tube_model_version')!=SENSED_TUBE_MODEL_VERSION or h>exact(spec.hold_width):
        raise ValueError('Incompatible sensed tube model or feedback hold')
    if not np.array_equal(spec.lambda_gain,np.asarray(sliding_gains)):
        raise ValueError('Planned and executed sliding gains differ')
    data={name:np.asarray(tube[name],dtype=float)[0].copy() for name in
        ('alpha','phi_peak','omega_peak','input_reserve','nominal_enclosure','disturbance_bound','A_variation')}
    if any(np.any(~np.isfinite(value)) or np.any(value<0) for value in data.values()):
        raise ValueError('Invalid sensed tube stage values')
    for name,value in data.items():
        shape=(6,) if name=='nominal_enclosure' else (3,6) if name=='A_variation' else (3,)
        if value.shape!=shape:raise ValueError('Invalid sensed tube shape: '+name)
    if (np.any(data['alpha']<spec.alpha_bounds[0]) or np.any(data['alpha']>spec.alpha_bounds[1])
            or not np.array_equal(data['A_variation'],spec.A_variation[0])):
        raise ValueError('Sensed stage bandwidth or coefficient variation changed')
    if (not np.allclose(spec.phi_initial,state.phi[key],atol=1e-8,rtol=0)
            or not np.allclose(spec.omega_initial,state.omega[key],atol=1e-8,rtol=0)
            or not np.allclose(np.asarray(plan.result.state_trajectory)[:,0],state.centers[key],atol=1e-8,rtol=0)):
        raise ValueError('Plan starts from a different nominal/tube state')
    base=spec.delta_A[0]@data['nominal_enclosure']+spec.delta_d[0]+spec.eta
    expected=sample_hold_bounds(spec.A_abs[0],spec.delta_A[0],spec.A_variation[0],base,
        data['phi_peak'],data['omega_peak'],data['alpha'],spec.lambda_gain,spec.hold_width,
        measurement_error_abs=spec.measurement_error_abs)
    for name,value in expected.items():
        if name not in tube or not np.allclose(np.asarray(tube[name])[0],value,atol=1e-7,rtol=1e-10):
            raise ValueError('Solved sensed tube is inconsistent: '+name)
    exact_terms,record=_exact_sensed_stage(spec,data)
    for name in ('input_reserve','disturbance_bound'):
        data[name]=np.maximum(data[name],np.array([_upper(value) for value in exact_terms[name]]))
    record['effective_input_reserve']=tuple(map(str,map(exact,data['input_reserve'])))
    record['effective_disturbance_bound']=tuple(map(str,map(exact,data['disturbance_bound'])))
    record['identity']=fingerprint({key:value for key,value in record.items() if key!='identity'})
    declared_budget=getattr(spec,'nominal_endpoint_error_abs',None)
    budget=np.asarray(np.zeros(6) if declared_budget is None else declared_budget,dtype=float)
    if budget.shape!=(6,) or np.any(~np.isfinite(budget)) or np.any(budget<0):
        raise ValueError('Nominal endpoint error budget must contain six finite nonnegative bounds')
    if np.any(budget):
        data['nominal_endpoint_error_abs']=tuple(map(exact,budget))
        data['nominal_endpoint_budget_hold']=exact(spec.hold_width)
        record['nominal_endpoint_error_abs']=tuple(map(str,data['nominal_endpoint_error_abs']))
        record['nominal_endpoint_budget_hold']=str(data['nominal_endpoint_budget_hold'])
        record['identity']=fingerprint({key:value for key,value in record.items() if key!='identity'})
    data['proof_record']=record
    return data


def feedback_evaluation(coefficients,measurement,center,phi,alpha,lam,u_nom,command,*,numerical_reserve):
    """Bound the actual held command's evaluation error from known data."""
    error=tuple(exact(m)-exact(c) for m,c in zip(measurement,center))
    lower,upper=[],[]
    for i,row in enumerate(coefficients.matrix_rows):
        products=tuple((a*e,b*e) for a,b,e in zip(row.lower,row.upper,error))
        adot_lower=sum(min(a,b) for a,b in products);adot_upper=sum(max(a,b) for a,b in products)
        gain,width,band=exact(lam[i]),exact(phi[i]),exact(alpha[i])
        sliding=error[i+3]+gain*error[i]
        sat=(min(Fraction(1),max(Fraction(-1),sliding/width)) if width>0 else
            Fraction(1 if sliding>0 else -1 if sliding<0 else 0))
        remainder=exact(u_nom[i])-gain*error[i+3]-band*width*sat
        lower.append(remainder-adot_upper);upper.append(remainder-adot_lower)
    ideal=IntervalBox(tuple(lower),tuple(upper))
    defect=tuple(max(abs(exact(value)-a),abs(exact(value)-b))
        for value,a,b in zip(command,ideal.lower,ideal.upper))
    reserve=exact(numerical_reserve)
    if any(value>reserve for value in defect):
        raise ValueError('Held feedback evaluation exceeds declared numerical reserve; maximum excess='+str(max(defect)-reserve))
    record=dict(method='exact-sensed-command-evaluation-against-Arb-nominal-matrix-v1',
        coefficients_id=coefficients.fingerprint,measured_state=tuple(map(str,map(exact,measurement))),
        nominal_center=tuple(map(str,map(exact,center))),phi=tuple(map(str,map(exact,phi))),
        alpha=tuple(map(str,map(exact,alpha))),lambda_gain=tuple(map(str,map(exact,lam))),
        nominal_command=tuple(map(str,map(exact,u_nom))),executed_command=tuple(map(str,map(exact,command))),
        ideal_command=ideal.as_dict(),evaluation_defect_abs=tuple(map(str,defect)),
        numerical_reserve=str(reserve),scope='Known held-command evaluation only; no truth state or environmental sample bound')
    record['identity']=fingerprint(record)
    return record


def measurement_radius(box,point):
    return np.array([_upper(max(abs(lo-exact(v)),abs(hi-exact(v)))) for lo,hi,v in zip(box.lower,box.upper,point)])


def check_tube_observation(box,center,phi,omega,lam):
    for i in range(3):
        position=max(abs(box.lower[i]-exact(center[i])),abs(box.upper[i]-exact(center[i])))
        lo=box.lower[i+3]-exact(center[i+3])+exact(lam[i])*(box.lower[i]-exact(center[i]))
        hi=box.upper[i+3]-exact(center[i+3])+exact(lam[i])*(box.upper[i]-exact(center[i]))
        if position>exact(omega[i]) or max(abs(lo),abs(hi))>exact(phi[i]):
            raise ValueError('Complete navigation enclosure lies outside the performance tube')
