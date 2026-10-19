"""Sensed dense execution with separate nonlinear truth and finite fallback.

Planning/feedback sees navigation only. Truth is used by the numerical plant,
sensor generator and explicitly labeled post-command containment diagnostics.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass,field
from fractions import Fraction
from functools import cached_property
from pathlib import Path
from types import MappingProxyType
import numpy as np
from scipy.integrate import solve_ivp

from orbexa.core.reference_frame import exact,fingerprint
from orbexa.core.nonlinear_relative import NonlinearRelativeDynamics
from orbexa.control.reachability import (IntervalBox,HeldInputRequest,OrbitalModelBounds,
    ValidatedFlowEncloser)
from orbexa.control.nonlinear_flow import NonlinearFlowEncloser,FRAME
from orbexa.control.attitude_flow import _sqrt_upper
from orbexa.control.orbital_hold import OrbitalHoldContext,OrbitalHoldRequest,OrbitalHoldProofService
from orbexa.control.dynamic_tube_model import boundary_layer_feedback
from orbexa.control.sample_hold import sample_hold_bounds,SENSED_TUBE_MODEL_VERSION
from orbexa.control.fleet_fallback import (FleetAdmissionRuntime,FleetFallbackPlan,
    PerformanceProposal,PerformanceAttempt)
from orbexa.control.refuge import RefugeObservation
from orbexa.estimation.navigation import midpoint,intersect
from orbexa.estimation.nonlinear_membership import NonlinearTransitionObservation
from orbexa.simulation.coupled_execution import ExecutionStep
from orbexa.simulation.nonlinear_refuge import BlackoutObservationPropagator
from orbexa.simulation.mission_safety import MissionSafety

_ACTUATION_SOURCE_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


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
    record=dict(method='exact-rational-sensed-v2-stage-v1',source_sha256=_ACTUATION_SOURCE_SHA256,
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


@dataclass(frozen=True)
class CommandActuationSupport:
    command: tuple
    error_abs: tuple
    applied_input: IntervalBox
    prior_fingerprint: str

    def as_dict(self):
        return dict(command=tuple(map(str,self.command)),error_abs=tuple(map(str,self.error_abs)),
            applied_input=self.applied_input.as_dict(),prior_fingerprint=self.prior_fingerprint,
            method='declared-gain-tilt-bias-command-support-v1')

    @cached_property
    def fingerprint(self):return fingerprint(self.as_dict())


@dataclass(frozen=True)
class CommandDependentActuationPrior:
    """Prospective command-dependent support; no actual matrix sets its bounds.

    M=R D gives M-I=(R-I)D+(D-I). With ||R-I||2<=theta,
    diagonal gain error <=g and bias <=b, each component error is bounded
    by theta*(1+g)*||u||2+g*abs(u_i)+b, plus declared rounding reserves.
    """
    gain_error: object = '.001'
    tilt_rad: object = '.001'
    bias_abs: object = '.2'
    round_relative: object = '1e-10'
    round_absolute: object = '1e-9'
    command_abs_limit: object = 100000
    declaration: str = 'Prospective bounded synthetic diagonal gain/axis tilt/bias study prior; no device qualification'

    def __post_init__(self):
        for name in ('gain_error','tilt_rad','bias_abs','round_relative','round_absolute','command_abs_limit'):
            value=exact(getattr(self,name))
            if value<0:raise ValueError('Actuation prior bounds must be nonnegative')
            object.__setattr__(self,name,value)
        if self.gain_error>=1 or self.command_abs_limit<=0 or not isinstance(self.declaration,str) or not self.declaration.strip():
            raise ValueError('Invalid declared gain/tilt/bias command prior')

    def as_dict(self):
        return dict(model='declared-gain-tilt-bias-command-support-v1',source_sha256=_ACTUATION_SOURCE_SHA256,
            **{name:str(getattr(self,name)) for name in ('gain_error','tilt_rad','bias_abs','round_relative','round_absolute','command_abs_limit')},
            declaration=self.declaration,units='u and bias in m/rad^2; dimensionless gain and tilt in rad',
            law='eta_i=theta*(1+g)*sqrt_upper(sum(u_j^2))+g*abs(u_i)+b+round_relative*norm_upper+round_absolute')

    @cached_property
    def fingerprint(self):return fingerprint(self.as_dict())

    def error_for_component_limits(self,limits):
        values=tuple(map(exact,limits))
        if len(values)!=3 or any(value<0 or value>self.command_abs_limit for value in values):
            raise ValueError('Command escapes the declared actuation-prior cap')
        norm=_sqrt_upper(sum(value*value for value in values))
        common=(self.tilt_rad*(1+self.gain_error)+self.round_relative)*norm+self.bias_abs+self.round_absolute
        return tuple(common+self.gain_error*value for value in values)

    def support(self,command):
        values=tuple(map(exact,command))
        errors=self.error_for_component_limits(tuple(abs(value) for value in values))
        return CommandActuationSupport(values,errors,IntervalBox(tuple(v-e for v,e in zip(values,errors)),
            tuple(v+e for v,e in zip(values,errors))),self.fingerprint)

    def validate_realization(self,realization):
        """Exact independent support check, including binary64 dot rounding.

        Dproxy clips diag(M) into [1-g,1+g]. This does not estimate the
        prior: each row of M-Dproxy must fit the prospectively fixed slope.
        The test covers a superset of proper R*D realizations and avoids
        treating a rounded scipy rotation as exactly orthogonal.
        """
        if type(realization) is not AffineActuatorRealization:
            raise TypeError('A command-dependent prior requires the supported exact affine realization')
        gamma=Fraction(16,2**53)/(1-Fraction(16,2**53))
        tiny=Fraction(16,2**1074)*(1+3*self.command_abs_limit)
        diagonal=tuple(min(1+self.gain_error,max(1-self.gain_error,realization.matrix[i][i])) for i in range(3))
        allowed_slope=self.tilt_rad*(1+self.gain_error)+self.round_relative
        rows=[]
        for i,row in enumerate(realization.matrix):
            residual=tuple(value-(diagonal[i] if i==j else 0) for j,value in enumerate(row))
            residual_norm=_sqrt_upper(sum(value*value for value in residual))
            matrix_norm=_sqrt_upper(sum(value*value for value in row))
            required_slope=residual_norm+gamma*matrix_norm
            required_constant=(1+gamma)*abs(realization.bias[i])+tiny
            if (abs(realization.bias[i])>self.bias_abs or required_slope>allowed_slope
                    or required_constant>self.bias_abs+self.round_absolute):
                raise ValueError('Numerical actuator realization violates the declared command-dependent prior')
            rows.append(dict(diagonal_proxy=str(diagonal[i]),residual_norm_upper=str(residual_norm),
                binary64_rounding_slope_upper=str(gamma*matrix_norm),slope_slack=str(allowed_slope-required_slope),
                constant_slack=str(self.bias_abs+self.round_absolute-required_constant)))
        record=dict(status='EXACT_DECLARED_SUPPORT_CHECK_PASSED',prior_fingerprint=self.fingerprint,
            realization_fingerprint=realization.fingerprint,rows=rows,
            realization_matrix=tuple(tuple(map(str,row)) for row in realization.matrix),
            realization_bias=tuple(map(str,realization.bias)),realization_declaration=realization.declaration,
            method='exact-diagonal-residual-row-norm-plus-gamma16-binary64-v1',
            scope='Independent realization containment check; controller bounds remain the declared prior')
        record['identity']=fingerprint(record)
        return record


@dataclass(frozen=True)
class AffineActuatorRealization:
    matrix: tuple = ((1,0,0),(0,1,0),(0,0,1))
    bias: tuple = (0,0,0)
    declaration: str = 'Explicit fixed study gain/axis/bias realization'

    def __post_init__(self):
        matrix = tuple(tuple(map(exact,row)) for row in self.matrix)
        bias = tuple(map(exact,self.bias))
        if len(matrix)!=3 or any(len(row)!=3 for row in matrix) or len(bias)!=3 or not self.declaration:
            raise ValueError('Actuator realization needs finite three-dimensional affine declarations')
        object.__setattr__(self,'matrix',matrix);object.__setattr__(self,'bias',bias)

    @cached_property
    def fingerprint(self):
        return fingerprint(dict(matrix=tuple(tuple(map(str,row)) for row in self.matrix),
                                bias=tuple(map(str,self.bias)),declaration=self.declaration))

    def apply(self,agent,command,q):
        return np.array([sum(float(a)*float(u) for a,u in zip(row,command))+float(b)
                         for row,b in zip(self.matrix,self.bias)])


@dataclass(frozen=True)
class BlackoutNavigationEstimate:
    exact_event_q: object
    physical_time_box: IntervalBox
    relative_state_box_by_agent: object
    clock_fingerprint: str
    model_fingerprint: str
    status: str = 'BLACKOUT_CONDITIONAL'
    reason: str = 'Finite admitted feedback/observation propagation; target ephemeris not freshly enclosed'
    target_ephemeris_box: object = None

    @property
    def point_relative_by_agent(self):
        return {key:midpoint(box) for key,box in self.relative_state_box_by_agent.items()}


@dataclass
class NonlinearExecutionStep(ExecutionStep):
    control_sources: list = field(default_factory=list)
    actual_truth_relative: dict = field(default_factory=dict)
    navigation_estimate: object = None
    membership_transitions: list = field(default_factory=list)
    journal: list = field(default_factory=list)
    used_fallback: bool = False
    delivered_packets: list = field(default_factory=list)
    archive_events: list = field(default_factory=list)
    actual_committed_q: object = None
    failure_event: object = None


class NonlinearExecutionFailure(RuntimeError):
    """Failed run evidence, including any already applied physical suffix."""
    def __init__(self,message,*,outcome,event,staged_archive,actual_committed_q,
                 actual_truth_relative,target_state):
        super().__init__(message)
        self.outcome,self.event,self.staged_archive=outcome,event,staged_archive
        self.actual_committed_q=actual_committed_q
        self.actual_truth_relative=actual_truth_relative
        self.target_state=target_state


class NonlinearDenseExecutor:
    def __init__(self,scenario,*,navigation,truth_target,truth_relative_by_agent,clock,
                 model_bounds,point_truth,sensor_suite,actuator_realization=None,
                 admission_runtime=None,encloser=None,nominal_encloser=None,
                 certificate_sink=None,maximum_blackout_q='0.02',sensor_id=None,
                 initial_packets=(),planning_model_validator=None,retain_journal=True,actuation_prior=None,
                 successor_preparer=None,completed_hold_sink=None):
        self.scenario,self.config,self.navigation = scenario,scenario.config,navigation
        self.truth_target,self.clock,self.model = truth_target,clock,model_bounds
        self.truth_relative_by_agent = {key:np.array(value,dtype=float).copy() for key,value in truth_relative_by_agent.items()}
        if set(self.truth_relative_by_agent)!=set(navigation.boxes) or any(v.shape!=(6,) for v in self.truth_relative_by_agent.values()):
            raise ValueError('Truth/navigation fleets must have identical six-state identities')
        self.point_truth = point_truth if isinstance(point_truth,NonlinearRelativeDynamics) else NonlinearRelativeDynamics(point_truth,clock)
        if self.point_truth.clock.fingerprint!=clock.fingerprint or truth_target.reference_clock.fingerprint!=clock.fingerprint:
            raise ValueError('Physical truth uses a different commanded reference clock')
        self._check_realization()
        self.sensors,self.actuator = sensor_suite,actuator_realization or AffineActuatorRealization()
        if not callable(getattr(self.actuator,'apply',None)) or not getattr(self.actuator,'fingerprint',None):
            raise TypeError('Use a pure actuator realization with apply(agent,command,q) and fingerprint')
        self.actuation_prior=actuation_prior or CommandDependentActuationPrior()
        if type(self.actuation_prior) is not CommandDependentActuationPrior:
            raise TypeError('Use the declared command-dependent gain/tilt/bias prior')
        self.actuation_validation=self.actuation_prior.validate_realization(self.actuator)
        self._validated_actuation_prior_id=self.actuation_prior.fingerprint
        command_limits=tuple(max(abs(exact(b['lower'])),abs(exact(b['upper']))) for b in self.config.mpc.input_bounds)
        self.actuation_maximum_error_abs=self.actuation_prior.error_for_component_limits(command_limits)
        self.sensor_id = sensor_id or sensor_suite.profile.fingerprint
        self.planning_model_validator=planning_model_validator
        if successor_preparer is not None and not callable(successor_preparer):
            raise TypeError('successor_preparer must be callable or None')
        self.successor_preparer=successor_preparer
        self.truth_safety=MissionSafety(self.config,self.truth_target,
            chaser_body_radius=getattr(scenario.safety,'chaser_body_radius',0.))
        static_collision=truth_target.collision_params('docking',float(truth_target.current_q))
        self._truth_composite_center=(np.asarray(static_collision['center'],dtype=float)
            if getattr(truth_target,'geometry',None) is not None and static_collision.get('shape')=='composite' else None)
        self.encloser = encloser or NonlinearFlowEncloser()
        self.nominal_encloser = nominal_encloser or ValidatedFlowEncloser(self.encloser.config)
        self._nominal_services = {}
        self.certificate_sink = certificate_sink
        if completed_hold_sink is not None and not callable(completed_hold_sink):
            raise TypeError('completed_hold_sink must be callable or None')
        self.completed_hold_sink=completed_hold_sink
        if not isinstance(retain_journal,bool):raise TypeError('retain_journal must be boolean')
        self.retain_journal=retain_journal
        self.last_archive_event,self.last_assumption_violation=None,None
        self._active_outcome,self._active_staged,self._active_completion=None,None,None
        self._execution_stage='IDLE'
        self.maximum_blackout_q = exact(maximum_blackout_q)
        if self.maximum_blackout_q<=0:raise ValueError('Declare a finite positive blackout duration')
        # This full-domain rate bound is proved at construction, never freshly
        # computed by a blacked-out interval service.
        self.blackout_clock_rate = clock.interval_bounds(clock.q_domain)['rate_s_inv']
        self.runtime,self.current_target_flow,self.flow_cache = None,None,{}
        self._physical_tube_witnesses={}
        self._checking_performance,self.performance_regions=False,{}
        self.pending_packets,self.journal,self.blackout_holds = list(initial_packets),[],[]
        self.delivered_since_last_step=[]
        self.blackout_observer,self.blackout_time = None,None
        if self.sensors.last_acquisition_s is None:self._sample()
        self._consume_ready()
        if admission_runtime is not None:self.install_runtime(admission_runtime)

    def _check_realization(self):
        physical,bounds_model = self.point_truth.model,self.model
        for name in ('mu_m3_s2','j2','sigma_target','sigma_chaser'):
            lo,hi=getattr(physical,name);a,b=getattr(bounds_model,name)
            if not a<=lo==hi<=b:raise ValueError('Physical realization escapes declared '+name)
        for name in ('rho0_kg_m3','scale_height_m'):
            lo,hi=getattr(physical.atmosphere,name);a,b=getattr(bounds_model.atmosphere,name)
            if not a<=lo==hi<=b:raise ValueError('Physical atmosphere escapes '+name)
        for name in ('rotation_rad_s','wind_m_s'):
            if any(not a<=lo==hi<=b for (lo,hi),(a,b) in zip(getattr(physical.atmosphere,name),getattr(bounds_model.atmosphere,name))):
                raise ValueError('Physical atmosphere vector escapes '+name)
        if (physical.j2_axis!=bounds_model.j2_axis or physical.atmosphere.reference_radius_m!=bounds_model.atmosphere.reference_radius_m
                or physical.gravity_reference_radius_m!=bounds_model.gravity_reference_radius_m):
            raise ValueError('Physical realization changed the declared fixed axes/radii')

    def install_runtime(self,runtime):
        if not isinstance(runtime,FleetAdmissionRuntime):raise TypeError('Install a complete FleetAdmissionRuntime')
        if runtime.q!=self.observed_q:raise ValueError('Fallback runtime starts at another observation event')
        for segment in runtime.active_fallback.segments:
            for _,controller in segment.controllers:
                if any(value>bound for value,bound in zip(self.actuation_maximum_error_abs,
                        controller.certificate.spec.actuator_error_abs)):
                    raise ValueError('Declared command-dependent support exceeds the global fallback actuator prior')
        self.runtime=runtime
        self.runtime.flow_provider=self.flow_provider

    @property
    def actual_committed_q(self):
        return self.truth_target.current_q

    @property
    def observed_q(self):
        return self.blackout_observer.q if self.blackout_holds else self.navigation.q

    def observations(self):
        if self.blackout_holds:
            boxes,q = self.blackout_observer.boxes,self.blackout_observer.q
        else:boxes,q=self.navigation.boxes,self.navigation.q
        contexts = {} if self.runtime is None else dict(self.runtime.active_fallback.segment_at(q).controllers)
        return {key:RefugeObservation(q,midpoint(box),box,
            contexts[key].certificate.spec.context.sensor_id if key in contexts else self.sensor_id,
            FRAME,self.clock.fingerprint) for key,box in boxes.items()}

    def flow_provider(self,agent,request):
        if self.current_target_flow is None or not self.current_target_flow.proven:
            raise ValueError('No currently proved nonlinear target hold')
        key=(str(agent),fingerprint(request.as_dict()),self.current_target_flow.fingerprint)
        if key not in self.flow_cache:
            self.flow_cache[key]=self.encloser.enclose_relative(request,self.current_target_flow,self.model,self.clock)
        result=self.flow_cache[key]
        if self._checking_performance and result.proven and (agent not in self.performance_regions or
                not self.performance_regions[agent].contains(result.swept_state)):
            raise ValueError('Performance flow escapes the prospective nonlinear residual region')
        return result

    def _performance_successor(self,q,h,applied,observations,controllers,record,witnesses):
        """Prepare entry from full prospective flows without advancing sensing.

        The current stored suffix is preferred. A fresh builder is called only
        after the exact complete performance endpoints fail that entry gate.
        Admission subsequently replays these same requests through the cache.
        """
        suffix=self.runtime.active_fallback.suffix(q+h)
        if self.successor_preparer is None:return suffix
        from orbexa.simulation.nonlinear_safety import ProspectiveFleetEndpoint
        from orbexa.control.refuge import fingerprint as source_fingerprint
        try:
            flows={key:self.flow_provider(key,HeldInputRequest(
                controllers[key].observation_box(observations[key],q),applied[key],
                (q,q),(h,h),self.runtime.acceleration_disturbance)) for key in observations}
        except (RuntimeError,TimeoutError) as error:
            raise ValueError('Prospective performance flow service unavailable: '+str(error)) from error
        if any(not value.proven for value in flows.values()):
            raise ValueError('Prospective performance flow is unresolved')
        if suffix.contains({key:value.endpoint for key,value in flows.items()},q+h):return suffix
        projection=ProspectiveFleetEndpoint.from_flows(flows=flows,target_flow=self.current_target_flow,
            observations=observations,contexts={key:value.certificate.spec.context for key,value in controllers.items()},
            acceleration_disturbance=self.runtime.acceleration_disturbance,model=self.model,clock=self.clock)
        estimate=self.navigation.estimate()
        if (estimate.status!='VALID' or exact(estimate.exact_event_q)!=q
                or set(estimate.relative_state_box_by_agent)!=set(observations)
                or any(estimate.relative_state_box_by_agent[key]!=value.state_enclosure
                    for key,value in observations.items())):
            raise ValueError('Prospective successor needs the unchanged current navigation estimate')
        old_plan=self.runtime.active_fallback
        authoritative_runtime=self.runtime
        runtime_guard=(old_plan.fingerprint,self.runtime.q,source_fingerprint(self.runtime.remaining_fuel),
            self.runtime._pending,self.runtime.acceleration_disturbance)
        def navigation_id(value):
            return source_fingerprint({name:(dict(item) if isinstance(item,MappingProxyType) else item)
                for name,item in vars(value).items()})

        estimate_id=navigation_id(estimate)
        record.update(status='PREPARING',projection_id=projection.fingerprint,exact_start=str(q),
            exact_end=str(q+h),stored_suffix_id=suffix.fingerprint,stored_suffix_contains_endpoints=False,
            source_navigation_id=estimate_id,
            archive_storage='DURABLE_EXTERNAL_SINK' if self.certificate_sink is not None else 'STAGED_OUTCOME_MEMORY')
        acknowledged={}

        def archive_witness(value):
            if not isinstance(value,dict):raise TypeError('Successor witness must be a full record')
            witness=copy.deepcopy(value)
            identity=witness.get('identity')
            if (witness.get('kind')!='fallback_successor' or witness.get('projection_id')!=projection.fingerprint
                    or not isinstance(identity,str) or identity!=source_fingerprint(
                        {key:item for key,item in witness.items() if key!='identity'})
                    or witness.get('projection')!=projection.as_dict(include_flows=True)
                    or not isinstance(witness.get('plan'),dict)):
                raise ValueError('Full successor witness does not bind the exact prospective projection')
            if identity in acknowledged:raise ValueError('Duplicate successor witness acknowledgment')
            if self.certificate_sink is not None:self.certificate_sink(witness)
            else:witnesses[identity]=witness
            acknowledged[identity]=witness['plan'].get('identity')

        try:
            candidate,summary=self.successor_preparer(projection,sensing_prior=estimate,certificate_sink=archive_witness)
            projection.verify_integrity()
            if not isinstance(candidate,FleetFallbackPlan) or not isinstance(summary,dict):
                raise TypeError('Successor preparation requires a proved FleetFallbackPlan and witness summary')
            fingerprint(summary) # Compact metadata must be canonical JSON before final admission.
            candidate.verify_integrity()
            candidate_id=candidate.fingerprint
            if (not candidate.certificate.admissible or candidate.certificate.start!=q+h
                    or candidate.ids!=tuple(observations)
                    or not candidate.contains(dict(projection.endpoint_boxes),q+h)
                    or summary.get('projection_id')!=projection.fingerprint
                    or summary.get('full_witness_archived') is not True
                    or summary.get('plan',{}).get('identity')!=candidate_id
                    or acknowledged.get(summary.get('archived_witness_id'))!=candidate_id):
                raise ValueError('Prepared successor or its acknowledged full witness is stale, partial or inconsistent')
            # All actual source fields above were checked independently of the
            # callback's compact description. The runtime checks the plan,
            # endpoint boxes, resource budget and held flows again before motion.
            record.update(copy.deepcopy(summary))
            record.update(status='PREPARED',successor_id=candidate_id,
                projection_id=projection.fingerprint,projection_exact_start=str(q),projection_exact_end=str(q+h),
                stored_suffix_id=suffix.fingerprint,stored_suffix_contains_endpoints=False,source_navigation_id=estimate_id,
                archive_storage='DURABLE_EXTERNAL_SINK' if self.certificate_sink is not None else 'STAGED_OUTCOME_MEMORY')
            return candidate
        except Exception as error:
            record.update(status='REFUSED',error_type=type(error).__name__,reason=str(error),
                acknowledged_witness_ids=tuple(acknowledged))
            raise ValueError('Prospective successor refused: '+type(error).__name__+': '+str(error)) from error
        finally:
            # A builder receives immutable data and never owns current runtime
            # installation, fuel, observation time or navigation state.
            if (self.runtime is not authoritative_runtime or self.runtime.active_fallback is not old_plan
                    or old_plan.fingerprint!=runtime_guard[0]
                    or self.runtime.q!=runtime_guard[1]
                    or source_fingerprint(self.runtime.remaining_fuel)!=runtime_guard[2]
                    or self.runtime._pending is not runtime_guard[3]
                    or self.runtime.acceleration_disturbance!=runtime_guard[4]
                    or navigation_id(self.navigation.estimate())!=estimate_id):
                raise RuntimeError('Successor builder altered authoritative current execution state')

    def _navigation_actuation_supports(self,decision):
        if (self.actuation_prior.fingerprint!=self._validated_actuation_prior_id
                or self.actuator.fingerprint!=self.actuation_validation['realization_fingerprint']):
            raise ValueError('Actuation prior or realization changed after independent validation')
        full=dict(decision.applied_inputs)
        result={key:self.actuation_prior.support(command) for key,command in decision.float_commands().items()}
        if any(not full[key].contains(value.applied_input) for key,value in result.items()):
            raise ValueError('Navigation actuation support is not a subset of the admitted global support')
        return result

    def _stage(self,state,plan,key,h):
        tube,spec=plan.tube,plan.spec
        if tube.get('tube_model_version')!=SENSED_TUBE_MODEL_VERSION or h>exact(spec.hold_width):
            raise ValueError('Incompatible sensed tube model or feedback hold')
        if not np.array_equal(spec.lambda_gain,np.asarray(self.config.tube.sliding_gains)):
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

    def _stage_source_snapshot(self,planning,planner,key,*,_target_identity=None):
        from orbexa.control.refuge import exact_record
        spec=planning.plans[key].spec;context=planner.context
        fields={name:np.asarray(getattr(spec,name)).tolist() for name in
            ('A_abs','A_variation','delta_A','delta_d','lambda_gain','eta','measurement_error_abs')}
        budget=getattr(spec,'nominal_endpoint_error_abs',None)
        padding=getattr(spec,'stage_peak_padding_abs',None)
        applied=getattr(spec,'applied_actuator_input_error_abs',None)
        fields.update(hold_width=str(exact(spec.hold_width)),
            nominal_endpoint_error_abs=None if budget is None else np.asarray(budget).tolist(),
            stage_peak_padding_abs=None if padding is None else np.asarray(padding).tolist(),
            applied_actuator_input_error_abs=None if applied is None else np.asarray(applied).tolist())
        controller=dict(self.runtime.active_fallback.segment_at(self.observed_q).controllers)[key]
        if _target_identity is None:
            target_identity=context.target_flow.fingerprint
        else:
            target, target_identity=_target_identity
            if context.target_flow is not target:
                raise ValueError('Planning target proof object changed during the source-check batch')
        return dict(spec=exact_record(fields),nominal_parameters={k:str(v) for k,v in planning.parameters.items()},
            nominal_control=np.asarray(planning.plans[key].result.control_trajectory)[:,0].tolist(),
            context=dict(exact_q=str(context.exact_q),model=context.model.fingerprint,clock=context.clock.fingerprint,
                target=target_identity,observation_error_abs=context.observation_error_abs.tolist(),
                acceleration_error_abs=context.acceleration_error_abs.tolist()),
            numerical_reserve=str(exact(self.config.tube.numerical_reserve)),
            actuator_error_abs=tuple(map(str,controller.certificate.spec.actuator_error_abs)),
            acceleration_disturbance=self.runtime.acceleration_disturbance.as_dict(),region=planner.regions[key].as_dict(),
            residual=copy.deepcopy(planner.residuals[key][0]))

    def _bind_stage_sources(self,planning,planner,q0,stage):
        from orbexa.control.nonlinear_flow import _SOURCE_SHA256 as residual_source
        from orbexa.control.nonlinear_flow import bound_nominal_coefficients
        from orbexa.core.dynamics import orbital_ellp_drag
        params=planning.parameters
        try:
            coefficients=bound_nominal_coefficients((q0,q0+exact(self.config.anom_step)),params,self.encloser.config.precision_bits)
        except ImportError as error:raise ValueError('Arb nominal coefficient source unavailable') from error
        arb_abs,arb_change=coefficients.absolute_upper,coefficients.variation_upper
        limits=IntervalBox(tuple(b['lower'] for b in self.config.mpc.input_bounds),
            tuple(b['upper'] for b in self.config.mpc.input_bounds))
        matrix=orbital_ellp_drag(anom_step=float(self.config.anom_step),
            **{name:params[name] for name in ('eccentricity','alpha','beta','semi_major_axis','mu')},
            theta0=params.get('theta0',0))[0][0]
        bindings={}
        for key,data in stage.items():
            spec=planning.plans[key].spec;snapshot=self._stage_source_snapshot(planning,planner,key)
            residual=snapshot['residual'];ctx=residual.get('context',{})
            region=ctx.get('regional_state_box',{});inputs=ctx.get('input_box',{})
            regional=IntervalBox(region.get('exact_lower',()),region.get('exact_upper',()))
            input_box=IntervalBox(inputs.get('exact_lower',()),inputs.get('exact_upper',()))
            q=tuple(map(exact,ctx.get('q_interval',())))
            if (residual.get('status')!='PROVEN' or ctx.get('backend_source_sha256')!=residual_source
                    or ctx.get('plant_fingerprint')!=planner.context.model.fingerprint
                    or ctx.get('clock_fingerprint')!=self.clock.fingerprint
                    or ctx.get('target_flow_fingerprint')!=planner.context.target_flow.fingerprint
                    or ctx.get('nominal_parameters')!=snapshot['nominal_parameters']
                    or len(q)!=2 or not q[0]<=q0<=q0+exact(self.config.anom_step)<=q[1]
                    or not regional.contains(planner.regions[key]) or not input_box.contains(limits)
                    or np.any(spec.delta_A[0]!=0)):
                raise ValueError('Nonlinear stage residual does not bind the declared nominal/model/clock/region/input source')
            raw=residual.get('component_bounds',{})
            residual_box=IntervalBox(raw.get('exact_lower',()),raw.get('exact_upper',()))
            controller=dict(self.runtime.active_fallback.segment_at(q0).controllers)[key]
            applied=getattr(spec,'applied_actuator_input_error_abs',None)
            applied=np.asarray(np.zeros(3) if applied is None else applied,dtype=float)
            if (applied.shape!=(3,) or np.any(~np.isfinite(applied)) or np.any(applied<0)
                    or any(exact(value)<exact(bound) for value,bound in
                        zip(applied,controller.certificate.spec.actuator_error_abs))):
                raise ValueError('Prospective applied-input tightening does not cover the installed actuator support')
            eta_required=tuple(max(exact(declared),exact(global_))+max(abs(lo_),abs(hi_))+exact(self.config.tube.numerical_reserve)
                for declared,global_,lo_,hi_ in zip(planner.context.acceleration_error_abs,
                    controller.certificate.spec.actuator_error_abs,self.runtime.acceleration_disturbance.lower,
                    self.runtime.acceleration_disturbance.upper))
            if (residual_box.dimension!=3 or any(exact(value)<max(abs(a),abs(b))
                    for value,a,b in zip(spec.delta_d[0],residual_box.lower,residual_box.upper))
                    or any(exact(value)<bound for row,bounds_ in zip(spec.A_abs[0],arb_abs) for value,bound in zip(row,bounds_))
                    or any(exact(value)<bound for row,bounds_ in zip(spec.A_variation[0],arb_change) for value,bound in zip(row,bounds_))
                    or any(exact(value)<bound for value,bound in zip(spec.eta,eta_required))):
                raise ValueError('Effective stage coefficients or forcing do not dominate their independent source bounds')
            proof=dict(method='strict-nonlinear-first-stage-source-binding-v1',snapshot=snapshot,
                nominal_matrix_generator='orbexa.core.dynamics.orbital_ellp_drag',
                nominal_coefficients=coefficients.as_dict(),nominal_coefficients_id=coefficients.fingerprint,
                eta_required=tuple(map(str,eta_required)),precision_bits=self.encloser.config.precision_bits)
            proof['identity']=fingerprint(proof)
            data['source_binding']=proof
            data['proof_record']['source_binding_id']=proof['identity']
            data['proof_record']['identity']=fingerprint({name:value for name,value in data['proof_record'].items() if name!='identity'})
            bindings[key]=(fingerprint(snapshot),planning.matrices[0])
        return bindings,matrix

    def _check_stage_sources(self,planning,planner,bindings):
        # Every member shares the same complete prospective target proof.
        # Hash its full current payload before and after this synchronous
        # batch, while retaining every per-agent mutable-source snapshot.
        # No target identity persists between physical holds.
        target=planner.context.target_flow
        target_identity=target.fingerprint
        if (set(bindings)!=set(planning.plans) or any(planning.matrices[0] is not callback
                or fingerprint(self._stage_source_snapshot(planning,planner,key,
                    _target_identity=(target,target_identity)))!=identity
                for key,(identity,callback) in bindings.items())):
            raise ValueError('Planning stage/nominal/residual source changed after preflight binding')
        if planner.context.target_flow is not target or target.fingerprint!=target_identity:
            raise ValueError('Planning target proof payload changed during the source-check batch')

    def _feedback_evaluation(self,coefficients,measurement,center,phi,alpha,lam,u_nom,command):
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
        reserve=exact(self.config.tube.numerical_reserve)
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

    @staticmethod
    def _measurement_radius(box,point):
        return np.array([_upper(max(abs(lo-exact(v)),abs(hi-exact(v)))) for lo,hi,v in zip(box.lower,box.upper,point)])

    def _check_tube_observation(self,box,center,phi,omega,lam):
        for i in range(3):
            position=max(abs(box.lower[i]-exact(center[i])),abs(box.upper[i]-exact(center[i])))
            lo=box.lower[i+3]-exact(center[i+3])+exact(lam[i])*(box.lower[i]-exact(center[i]))
            hi=box.upper[i+3]-exact(center[i+3])+exact(lam[i])*(box.upper[i]-exact(center[i]))
            if position>exact(omega[i]) or max(abs(lo),abs(hi))>exact(phi[i]):
                raise ValueError('Complete navigation enclosure lies outside the performance tube')

    def _tube_membership(self,key,q,observation,state,lam):
        witness=self._physical_tube_witnesses.get(key)
        bound=(tuple(map(exact,state.centers[key])),tuple(map(exact,state.phi[key])),
            tuple(map(exact,state.omega[key])),tuple(map(exact,lam)))
        if (type(witness) is PhysicalTubeWitness and witness.agent==key and witness.exact_q==q
                and (witness.center,witness.phi,witness.omega,witness.lambda_gain)==bound
                and witness.model_fingerprint==self.model.fingerprint
                and witness.clock_fingerprint==self.clock.fingerprint):
            return dict(method='PRIOR_ADMITTED_PHYSICAL_TUBE',witness_id=witness.fingerprint,
                exact_q=str(q),current_observation_id=observation.fingerprint,
                scope='Relational physical inclusion; current navigation box corners need not all be reachable')
        self._check_tube_observation(observation.state_enclosure,state.centers[key],state.phi[key],state.omega[key],lam)
        return dict(method='COMPLETE_CURRENT_NAVIGATION_BOX',exact_q=str(q),
            current_observation_id=observation.fingerprint,
            discarded_prior_witness_id=None if witness is None else witness.fingerprint)

    def _nominal(self,center,u,q,h,params,phi,omega,stage,lam):
        budget=stage.get('nominal_endpoint_error_abs')
        if budget is not None and h!=stage['nominal_endpoint_budget_hold']:
            raise ValueError('Executed hold differs from the prospective nominal-endpoint budget hold')
        model=OrbitalModelBounds(eccentricity=(params['eccentricity'],)*2,
            alpha=(params['alpha'],)*2,beta=(params['beta'],)*2,
            semi_major_axis=params['semi_major_axis'],mu=params['mu'],theta0=(params.get('theta0',0),)*2)
        # The existing authenticated service retains its original Picard swept
        # witness and independently refines the endpoint by variation of
        # constants. This is solely the CH nominal surrogate, never a physical
        # nonlinear/frame certificate. Exact source checks remain in prove().
        source_id=fingerprint(dict(role='nominal-CH-surrogate-in-commanded-coordinates',model=model.as_dict()))
        service_key=(source_id,self.nominal_encloser.config)
        if service_key not in self._nominal_services:
            limits=IntervalBox(tuple(b['lower'] for b in self.config.mpc.input_bounds),
                               tuple(b['upper'] for b in self.config.mpc.input_bounds))
            context=OrbitalHoldContext(model,self.clock.q_domain,limits,IntervalBox.point((0,0,0)),
                source_id,self.nominal_encloser.config)
            self._nominal_services[service_key]=OrbitalHoldProofService(context)
        certificate=self._nominal_services[service_key].prove(OrbitalHoldRequest(
            IntervalBox.point(center),IntervalBox.point(u),q,h))
        if not certificate.proven:raise ValueError('Nominal Picard/VoC endpoint unresolved')
        chosen=np.array(midpoint(certificate.endpoint))
        defect=self._measurement_radius(certificate.endpoint,chosen)
        if budget is not None:
            if any(exact(value)>bound for value,bound in zip(defect,budget)):
                excess=max(exact(value)-bound for value,bound in zip(defect,budget))
                raise ValueError('Certified nominal endpoint defect exceeds prospective budget; maximum excess='+str(excess))
        if np.any(phi>stage['phi_peak']) or np.any(omega>stage['omega_peak']):
            excess=max(*(exact(a)-exact(b) for a,b in zip(phi,stage['phi_peak'])),
                *(exact(a)-exact(b) for a,b in zip(omega,stage['omega_peak'])))
            error=ValueError('Initial physical tube exceeds declared stage peaks; maximum excess='+repr(float(excess)))
            error.execution_details=dict(gate='INITIAL_PHYSICAL_TUBE_PEAK',exact_start=str(q),exact_hold_width=str(h),
                initial_phi=np.asarray(phi).tolist(),initial_omega=np.asarray(omega).tolist(),
                phi_peak=stage['phi_peak'].tolist(),omega_peak=stage['omega_peak'].tolist(),
                exact_maximum_excess=str(excess),exact_stage_id=stage['proof_record']['identity'])
            raise error
        next_phi,next_omega,phi_box,omega_box=_outward_tube_endpoint(phi,omega,stage['alpha'],lam,
            stage['disturbance_bound'],h,stage['phi_peak'],self.encloser.config.precision_bits)
        next_omega=np.array([_upper(exact(value)+exact(error)) for value,error in zip(next_omega,defect[:3])])
        next_phi=np.array([_upper(exact(value)+exact(ev)+exact(gain)*exact(ep))
            for value,ev,gain,ep in zip(next_phi,defect[3:],lam,defect[:3])])
        if np.any(next_phi>stage['phi_peak']) or np.any(next_omega>stage['omega_peak']):
            excess=max(float(np.max(next_phi-stage['phi_peak'])),float(np.max(next_omega-stage['omega_peak'])))
            error=ValueError('Certified nominal endpoint transfer exceeds solved tube peaks; maximum excess='+repr(excess))
            error.execution_details=dict(gate='TRANSFERRED_PHYSICAL_TUBE_PEAK',exact_start=str(q),exact_hold_width=str(h),
                transferred_phi=next_phi.tolist(),transferred_omega=next_omega.tolist(),nominal_defect_abs=defect.tolist(),
                phi_peak=stage['phi_peak'].tolist(),omega_peak=stage['omega_peak'].tolist(),maximum_excess=excess,
                exact_stage_id=stage['proof_record']['identity'])
            raise error
        transfer=dict(method='arb-positive-comparison-plus-certified-nominal-transfer-v1',
            exact_start=str(q),exact_hold_width=str(h),initial_phi=tuple(map(str,map(exact,phi))),
            initial_omega=tuple(map(str,map(exact,omega))),comparison_phi=phi_box.as_dict(),
            comparison_omega=omega_box.as_dict(),chosen_center=tuple(map(str,map(exact,chosen))),
            nominal_defect_abs=tuple(map(str,map(exact,defect))),
            transferred_phi=tuple(map(str,map(exact,next_phi))),transferred_omega=tuple(map(str,map(exact,next_omega))),
            exact_stage_id=stage['proof_record']['identity'],nominal_flow_id=certificate.fingerprint,
            nominal_endpoint_id=certificate.endpoint_certificate.fingerprint,
            nominal_endpoint_error_abs=None if budget is None else tuple(map(str,budget)),
            nominal_endpoint_budget_hold=None if budget is None else str(stage['nominal_endpoint_budget_hold']),
            precision_bits=self.encloser.config.precision_bits)
        transfer['identity']=fingerprint(transfer);stage['last_transfer']=transfer
        return chosen,next_phi,next_omega,certificate,defect

    def _numeric_hold(self,q,h,decision,navigation_supports):
        proposal=self.truth_target.propose_interval(q+h)
        commands,applied=decision.float_commands(),dict(decision.applied_inputs)
        results={};solutions={}
        for key,initial in self.truth_relative_by_agent.items():
            command=np.asarray(commands[key])
            def rhs(value,relative,key=key,command=command):
                actual=np.asarray(self.actuator.apply(key,command,value),dtype=float)
                if actual.shape!=(3,) or not np.all(np.isfinite(actual)) or not applied[key].contains(IntervalBox.point(actual)):
                    raise ValueError('Numerical actuator violates its declared applied-input support')
                if not navigation_supports[key].applied_input.contains(IntervalBox.point(actual)):
                    raise ValueError('Numerical actuator violates the declared navigation actuation subset')
                return self.point_truth.relative_rhs(value,relative,proposal.state_at(value)[:6],actual)
            solution=solve_ivp(rhs,(float(q),float(q+h)),initial,method='DOP853',dense_output=True,
                max_step=min(float(h),.0001),rtol=1e-11,atol=1e-10)
            if not solution.success or not np.all(np.isfinite(solution.y)):
                raise RuntimeError('Numerical relative truth refused: '+str(solution.message))
            results[key],solutions[key]=solution.y[:,-1],solution
        return proposal,results,solutions

    def _numeric_clearance(self,state,q,phase):
        # Same declared occupied-union distance as MissionSafety.composite_margin,
        # without copying a full serialized geometry for every offline sample.
        # This diagnostic never substitutes for continuous fleet admission.
        if self._truth_composite_center is not None:
            body=self.truth_target.rotation_matrix_at(q).T@(np.asarray(state[:3])-self._truth_composite_center)
            return float(self.truth_target.geometry.signed_distance(body)-self.truth_safety.chaser_body_radius)
        return self.truth_safety.active_margin_for_state(state,q,phase)

    def _sample(self):
        q=self.truth_target.current_q
        states={key:self.clock.to_inertial(value,q,self.truth_target.inertial_state_at(float(q)))[0]
                for key,value in self.truth_relative_by_agent.items()}
        packets=self.sensors.sample(q=q,physical_time_s=self.truth_target.physical_time_at(float(q)),
            relative_states=self.truth_relative_by_agent,target_rotation=self.truth_target.rotation_matrix_at(float(q)),
            target_angular_velocity=self.truth_target.body_angular_velocity_at(float(q)),chaser_inertial_states=states)
        self.pending_packets.extend(packets)

    def _consume_ready(self):
        ready=[packet for packet in self.pending_packets
               if packet.delivery_time_s.upper[0]<=self.navigation.time.lower[0]]
        estimate=self.navigation.consume(ready)
        identifiers={packet.packet_id for packet in ready}
        self.pending_packets=[packet for packet in self.pending_packets if packet.packet_id not in identifiers]
        self.delivered_since_last_step.extend(ready)
        return estimate

    def drain_delivered_packets(self):
        packets=tuple(self.delivered_since_last_step)
        self.delivered_since_last_step=[]
        return packets

    def _planning_model_valid(self,context):
        if context is None or context.clock.fingerprint!=self.clock.fingerprint:return False
        if context.model.fingerprint==self.model.fingerprint:return True
        sub=context.model
        for name in ('mu_m3_s2','j2','sigma_target','sigma_chaser'):
            lo,hi=getattr(sub,name);a,b=getattr(self.model,name)
            if not a<=lo<=hi<=b:return False
        if (sub.j2_axis!=self.model.j2_axis or sub.atmosphere.reference_radius_m!=self.model.atmosphere.reference_radius_m
                or sub.gravity_reference_radius_m!=self.model.gravity_reference_radius_m
                or sub.body_radius_m!=self.model.body_radius_m or sub.radius_domain_m!=self.model.radius_domain_m):return False
        for name in ('rho0_kg_m3','scale_height_m'):
            lo,hi=getattr(sub.atmosphere,name);a,b=getattr(self.model.atmosphere,name)
            if not a<=lo<=hi<=b:return False
        for name in ('rotation_rad_s','wind_m_s'):
            if any(not a<=lo<=hi<=b for (lo,hi),(a,b) in zip(getattr(sub.atmosphere,name),getattr(self.model.atmosphere,name))):return False
        return self.planning_model_validator is not None and bool(self.planning_model_validator(sub,self.model))

    def _blackout_proposal(self,decision,plan,observations):
        observer=(BlackoutObservationPropagator(observations,maximum_blackout_q=self.maximum_blackout_q)
            if self.blackout_observer is None else copy.deepcopy(self.blackout_observer))
        current_time=self.navigation.time if self.blackout_time is None else self.blackout_time
        updated=observer.advance(decision,plan)
        lo,hi=self.blackout_clock_rate.lower[0],self.blackout_clock_rate.upper[0]
        h=decision.hold_width
        time=IntervalBox((current_time.lower[0]+h/hi,),(current_time.upper[0]+h/lo,))
        return observer,time,updated

    def _commit_blackout(self,proposal,decision):
        self.blackout_observer,self.blackout_time,updated=proposal
        self.blackout_holds.append((decision,self.runtime.acceleration_disturbance))
        return updated

    def recover_navigation(self):
        if not self.blackout_holds:return self.navigation.estimate()
        candidate=copy.deepcopy(self.navigation)
        for decision,disturbance in self.blackout_holds:
            q,h=decision.start,decision.hold_width
            tf=self.encloser.enclose_target(candidate.target,(q,q),(h,h),self.model,self.clock,candidate.time)
            if not tf.proven:raise ValueError('Blackout recovery target flow unresolved: '+tf.reason)
            supports=self._navigation_actuation_supports(decision)
            flows={key:self.encloser.enclose_relative(HeldInputRequest(candidate.boxes[key],value.applied_input,(q,q),(h,h),disturbance),tf,self.model,self.clock)
                   for key,value in supports.items()}
            if any(not value.proven for value in flows.values()):raise ValueError('Blackout recovery relative inclusion unresolved')
            candidate.predict(flows,tf)
        ready=[packet for packet in self.pending_packets if packet.delivery_time_s.upper[0]<=candidate.time.lower[0]]
        estimate=candidate.consume(ready)
        if estimate.status!='VALID':raise ValueError('Recovered sensing contradicts blackout navigation')
        recovered=self.observations() # Keep blackout IDs until its correction is checked.
        recovered={key:RefugeObservation(candidate.q,midpoint(box),box,recovered[key].sensor_id,FRAME,self.clock.fingerprint)
                   for key,box in candidate.boxes.items()}
        self.blackout_observer.correct(recovered)
        self.navigation.__dict__.clear();self.navigation.__dict__.update(candidate.__dict__)
        self.blackout_holds=[];self.blackout_observer=None;self.blackout_time=None
        accepted={packet.packet_id for packet in ready}
        self.pending_packets=[packet for packet in self.pending_packets if packet.packet_id not in accepted]
        self.delivered_since_last_step.extend(ready)
        return estimate

    def _publish_archive(self,kind,record,outcome,*,target=None,flows=None,nominal=None,navigation_flows=None,
            successor_witnesses=None):
        event=dict(archive_event_kind=kind,staged_identity=record.get('staged_identity'),
            record=record,target=target,flows={} if flows is None else flows,
            nominal={} if nominal is None else nominal,
            navigation_flows={} if navigation_flows is None else navigation_flows,
            successor_witnesses={} if successor_witnesses is None else successor_witnesses)
        self.last_archive_event=event
        # A supplied sink must finish its durable write before returning. With
        # no sink, the bounded current outcome owns the full in-memory events.
        if self.certificate_sink is not None:
            self.certificate_sink(event)
            outcome.archive_events.append(dict(archive_event_kind=kind,staged_identity=record.get('staged_identity')))
        else:outcome.archive_events.append(event)
        return event

    def _stage_admitted_hold(self,outcome,q,h,decision,observations,tf,flows,navigation_flows,navigation_supports,nominal,blackout,service_error,
            successor_preparation=None,successor_witnesses=None,performance_tube_proofs=None,feedback_coefficients=None):
        from orbexa.control.refuge import exact_record
        record=dict(exact_start=str(q),exact_end=str(q+h),exact_hold_width=str(h),
            decision=decision.as_dict(),initial_observations={key:exact_record(value) for key,value in observations.items()},
            target_flow_id=None if tf is None else tf.fingerprint,
            flow_ids={key:value.fingerprint for key,value in flows.items()},
            navigation_flow_ids={key:value.fingerprint for key,value in navigation_flows.items()},
            navigation_prediction_input_supports={key:value.request['held_input'] for key,value in navigation_flows.items()},
            admission_input_supports={key:value.as_dict() for key,value in decision.applied_inputs},
            navigation_input_supports={key:value.as_dict() for key,value in navigation_supports.items()},
            actuation_prior_id=self.actuation_prior.fingerprint,actuation_prior=self.actuation_prior.as_dict(),
            actuation_realization_validation=self.actuation_validation,
            actuation_realization_validation_id=self.actuation_validation['identity'],
            nominal_flow_ids={key:value[3].fingerprint for key,value in nominal.items()} if decision.source=='PERFORMANCE' else {},
            plant_fingerprint=self.model.fingerprint,clock_fingerprint=self.clock.fingerprint,
            actuator_realization_id=self.actuator.fingerprint,sensor_profile_id=self.sensors.profile.fingerprint,
            numerical_delivery_at_start={key:np.asarray(self.actuator.apply(key,command,float(q))).tolist()
                for key,command in decision.float_commands().items()},
            proof_blackout=blackout,proof_service_error=service_error,
            prospective_successor={} if successor_preparation is None else successor_preparation,
            performance_tube_proofs={} if performance_tube_proofs is None else performance_tube_proofs,
            performance_feedback_coefficients=None if feedback_coefficients is None else feedback_coefficients.as_dict(),
            source='ADMITTED_DECISION_BEFORE_ANY_NUMERICAL_OR_PHYSICAL_MOTION')
        record['staged_identity']=fingerprint(record)
        self._active_staged=dict(archive_event_kind='ADMITTED_BEFORE_MOTION',staged_identity=record['staged_identity'],
            record=record,target=None if tf is None else tf.as_dict(),
            flows={key:value.as_dict() for key,value in flows.items()},
            navigation_flows={key:value.as_dict() for key,value in navigation_flows.items()},
            nominal={key:value[3].as_dict() for key,value in nominal.items()} if decision.source=='PERFORMANCE' else {},
            successor_witnesses={} if successor_witnesses is None else successor_witnesses)
        self._execution_stage='ARCHIVE_BEFORE_MOTION'
        self._publish_archive('ADMITTED_BEFORE_MOTION',record,outcome,
            target=self._active_staged['target'],flows=self._active_staged['flows'],nominal=self._active_staged['nominal'],
            navigation_flows=self._active_staged['navigation_flows'],
            successor_witnesses=self._active_staged['successor_witnesses'])

    def _raise_execution_failure(self,error):
        actual_q=self.actual_committed_q
        truth={key:value.copy() for key,value in self.truth_relative_by_agent.items()}
        target=np.asarray(self.truth_target._current_state,dtype=float).copy()
        outcome=self._active_outcome or NonlinearExecutionStep(float(actual_q),float(actual_q))
        staged=self._active_staged
        start=None if staged is None else exact(staged['record']['exact_start'])
        pending=None if self.runtime is None else self.runtime._pending
        event=dict(status='ASSUMPTION_VIOLATION',staged_identity=None if staged is None else staged['staged_identity'],
            execution_stage=self._execution_stage,error_type=type(error).__name__,error=str(error),
            physical_motion_committed=start is not None and actual_q>start,
            exact_actual_committed_q=str(actual_q),actual_physical_time_s=float(target[13]),
            actual_truth_relative={key:value.tolist() for key,value in truth.items()},target_full_state=target.tolist(),
            observed_q=str(self.observed_q),runtime_q=None if self.runtime is None else str(self.runtime.q),
            pending_decision_id=None if pending is None else pending[0].fingerprint,
            decision=None if staged is None else staged['record']['decision'],
            actual_input_at_start=(None if staged is None or start is None or actual_q<=start
                else staged['record']['numerical_delivery_at_start']),
            completion_record_unconfirmed=self._active_completion,
            continuation_authorized=False,scope='Already applied motion is retained; no successful endpoint or clock pause is inferred')
        outcome.success=False;outcome.messages.append(type(error).__name__+': '+str(error))
        outcome.end_anomaly=float(actual_q);outcome.actual_committed_q=actual_q
        outcome.actual_truth_relative=truth;outcome.failure_event=event
        outcome.navigation_estimate=self.navigation.estimate()
        outcome.delivered_packets=list(self.drain_delivered_packets())
        outcome.sampled_tube_contained=False
        self.last_assumption_violation=event
        try:self._publish_archive('ASSUMPTION_VIOLATION',event,outcome)
        except Exception as archive_error:
            event['failure_archive_error']=dict(type=type(archive_error).__name__,message=str(archive_error),persistence='UNCONFIRMED')
        raise NonlinearExecutionFailure(str(error),outcome=outcome,event=event,staged_archive=staged,
            actual_committed_q=actual_q,actual_truth_relative=truth,target_state=target) from error

    def _complete_admitted_hold(self,state,outcome,phase,q,h,decision,tf,flows,navigation_flows,navigation_supports,nominal,stage,feedbacks,
            initial_boxes,blackout,blackout_proposal,truth,solutions,lam,ids):
        self._execution_stage="SENSOR_GENERATION"
        self._sample()
        if blackout:
            endpoint_observations=self._commit_blackout(blackout_proposal,decision)
            estimate=BlackoutNavigationEstimate(q+h,self.blackout_time,MappingProxyType(dict(self.blackout_observer.boxes)),
                self.clock.fingerprint,self.model.fingerprint)
        else:
            self._execution_stage="NAVIGATION_UPDATE"
            self.navigation.predict(navigation_flows,tf)
            estimate=self._consume_ready()
            if estimate.status!='VALID':raise ValueError('Sensor/model assumption failed after committed physical hold: '+estimate.reason)
            endpoint_observations=self.observations()
        self._execution_stage="RUNTIME_COMMIT"
        committed=self.runtime.commit(decision,endpoint_observations,q=q+h)
        self._execution_stage="ENDPOINT_DIAGNOSTICS"
        centers,phi,omega,bandwidth={},{},{},{}
        for key in ids:
            box=endpoint_observations[key].state_enclosure
            if decision.source=='PERFORMANCE':
                centers[key],phi[key],omega[key],_,_=nominal[key]
                bandwidth[key]=stage[key]['alpha']
                defect=nominal[key][4]
                outcome.nominal_center_defect[key]=defect.copy()
                outcome.max_nominal_discretization_residual=max(outcome.max_nominal_discretization_residual,float(np.linalg.norm(defect)))
            else:
                centers[key]=np.array(midpoint(box));radius=self._measurement_radius(box,centers[key])
                omega[key]=radius[:3]
                phi[key]=np.array([_upper(exact(v)+exact(gain)*exact(p))
                    for v,gain,p in zip(radius[3:],lam,radius[:3])])
                bandwidth[key]=np.array(state.bandwidth[key])
            delivered=np.asarray(self.actuator.apply(key,decision.float_commands()[key],float(q)),dtype=float)
            outcome.applied[key].append(delivered);outcome.ancillary[key].append(feedbacks.get(key,np.zeros(3)))
            error=truth[key]-centers[key]
            outcome.containment[key]=min(outcome.containment[key],float(min(np.min(omega[key]-abs(error[:3])),np.min(phi[key]-abs(error[3:]+lam*error[:3])))))
            if flows and (not flows[key].endpoint.contains(IntervalBox.point(truth[key])) or any(not flows[key].swept_state.contains(IntervalBox.point(solutions[key].y[:,i])) for i in range(solutions[key].y.shape[1]))):
                outcome.success=False;outcome.messages.append(key+': numerical truth outside conditional flow enclosure')
            if navigation_flows and (not navigation_flows[key].endpoint.contains(IntervalBox.point(truth[key])) or any(not navigation_flows[key].swept_state.contains(IntervalBox.point(solutions[key].y[:,i])) for i in range(solutions[key].y.shape[1]))):
                outcome.success=False;outcome.messages.append(key+': numerical truth outside conditional navigation-subset flow')
            for index,value in enumerate(solutions[key].t):
                clearance=self._numeric_clearance(solutions[key].y[:,index],float(value),phase)
                outcome.clearance[key]=min(outcome.clearance[key],float(clearance))
            if not blackout:
                outcome.membership_transitions.append((key,NonlinearTransitionObservation(initial_boxes[key],box,
                    navigation_supports[key].applied_input,(q,q),(h,h),tf.initial_state,tf.physical_time_initial,
                    self.runtime.acceleration_disturbance,observation_id=fingerprint(dict(observation=endpoint_observations[key].fingerprint,
                        input_support=navigation_supports[key].fingerprint,actuation_prior=self.actuation_prior.fingerprint)))))
        if not outcome.success:
            raise ValueError('Committed numerical-truth inclusion assumption failed: '+'; '.join(outcome.messages))
        self._execution_stage="CONTROLLER_STATE_COMMIT"
        state.commit(float(q+h),states={key:np.array(value.measurement,dtype=float) for key,value in endpoint_observations.items()},
            centers=centers,phi=phi,omega=omega,bandwidth=bandwidth)
        outcome.end_anomaly=float(q+h);outcome.control_sources.append(decision.source)
        sample_clock=sorted({float(value) for solution in solutions.values() for value in solution.t})
        for value in sample_clock:
            positions={key:solutions[key].sol(value)[:3] for key in ids}
            for i,key in enumerate(ids):
                for other in ids[i+1:]:outcome.min_spacing=min(outcome.min_spacing,float(np.linalg.norm(positions[key]-positions[other])))
        record=dict(exact_start=str(q),exact_end=str(q+h),decision=decision.as_dict(),runtime_commit=committed,
            exact_hold_width=str(h),
            target_flow_id=None if tf is None else tf.fingerprint,flow_ids={key:value.fingerprint for key,value in flows.items()},
            navigation_flow_ids={key:value.fingerprint for key,value in navigation_flows.items()},
            navigation_prediction_input_supports={key:value.request['held_input'] for key,value in navigation_flows.items()},
            admission_input_supports={key:value.as_dict() for key,value in decision.applied_inputs},
            navigation_input_supports={key:value.as_dict() for key,value in navigation_supports.items()},
            actuation_prior_id=self.actuation_prior.fingerprint,
            actuation_realization_validation_id=self.actuation_validation['identity'],
            nominal_flow_ids={key:value[3].fingerprint for key,value in nominal.items()} if decision.source=='PERFORMANCE' else {},
            nominal_picard_flow_ids={key:value[3].picard_flow.fingerprint for key,value in nominal.items()} if decision.source=='PERFORMANCE' else {},
            nominal_endpoint_ids={key:value[3].endpoint_certificate.fingerprint for key,value in nominal.items()} if decision.source=='PERFORMANCE' else {},
            nominal_model_role='CH-surrogate-in-commanded-coordinates; physical admission uses separate nonlinear flows',
            tube_transition='CONTINUOUS_NOMINAL_ENDPOINT_TRANSFER' if decision.source=='PERFORMANCE' else 'RESET_FROM_CERTIFIED_NAVIGATION',
            actuator_realization_id=self.actuator.fingerprint,sensor_profile_id=self.sensors.profile.fingerprint,
            actual_input_at_start={key:np.asarray(self.actuator.apply(key,command,float(q))).tolist()
                                   for key,command in decision.float_commands().items()},
            navigation_status=estimate.status,proof_blackout=blackout,
            actual_truth_relative={key:value.tolist() for key,value in truth.items()})
        record['ancillary_control_semantics']='Executed performance ancillary; zero for fallback, whose full command is recorded separately'
        record['prospective_successor']=self._active_staged['record']['prospective_successor']
        next_witnesses={}
        if decision.source=='PERFORMANCE':
            for key in ids:
                proof=self._active_staged['record']['performance_tube_proofs'][key]
                source=dict(staged_identity=self._active_staged['staged_identity'],decision_id=decision.fingerprint,
                    parent_witness_id=proof['initial_membership'].get('witness_id'),
                    initial_membership_method=proof['initial_membership']['method'],
                    exact_stage_id=proof['exact_stage']['identity'],
                    feedback_evaluation_id=proof['feedback_evaluation']['identity'],
                    nominal_transfer_id=proof['nominal_transfer']['identity'],
                    nominal_endpoint_id=nominal[key][3].endpoint_certificate.fingerprint,
                    physical_flow_id=flows[key].fingerprint,residual_id=proof['regional_residual_id'])
                next_witnesses[key]=PhysicalTubeWitness(key,q+h,tuple(map(exact,centers[key])),
                    tuple(map(exact,phi[key])),tuple(map(exact,omega[key])),tuple(map(exact,lam)),
                    self.model.fingerprint,self.clock.fingerprint,tuple(source.items()))
        record['physical_tube_witnesses']={key:dict(identity=value.fingerprint,**value.as_dict())
            for key,value in next_witnesses.items()}
        record['staged_identity']=self._active_staged['staged_identity']
        self._active_completion=record
        self._execution_stage='ARCHIVE_COMPLETION'
        self._publish_archive('APPLIED_HOLD_COMPLETE',record,outcome)
        # Only an acknowledged completion belongs in the compact physical
        # ledger. A later outer-update failure must not discard this hold.
        self._active_completion=None
        outcome.journal.append(record)
        if self.retain_journal:self.journal.append(record)
        self._execution_stage='COMPLETED_HOLD_RECORDING'
        if self.completed_hold_sink is not None:self.completed_hold_sink(record)
        self._physical_tube_witnesses=next_witnesses
        self._active_staged,self._active_completion=None,None
        self._execution_stage='HOLD_COMPLETE'
        return estimate

    def advance(self,state,planning,phase,*,planner=None,failed_attempt=None,proof_blackout=False):
        if self.last_assumption_violation is not None:
            raise ValueError('A failed physical run cannot resume without an explicit recovery contract')
        self._active_outcome,self._active_staged,self._active_completion=None,None,None
        self._execution_stage='PRE_ADMISSION'
        try:
            return self._advance(state,planning,phase,planner=planner,failed_attempt=failed_attempt,
                proof_blackout=proof_blackout)
        except NonlinearExecutionFailure:
            raise
        except Exception as error:
            self._raise_execution_failure(error)

    def _advance(self,state,planning,phase,*,planner=None,failed_attempt=None,proof_blackout=False):
        if self.runtime is None:raise ValueError('Install a complete admitted fleet fallback before motion')
        q0=self.observed_q
        ids=tuple(self.truth_relative_by_agent)
        outcome=NonlinearExecutionStep(float(q0),float(q0),reserve_ratios={k:0. for k in ids},
            applied={k:[] for k in ids},ancillary={k:[] for k in ids},
            containment={k:np.inf for k in ids},clearance={k:np.inf for k in ids})
        self._active_outcome=outcome
        base_h=exact(self.config.anom_step)/int(self.config.tube.feedback_substeps)
        outer_end=q0+exact(self.config.anom_step)
        lam=np.asarray(self.config.tube.sliding_gains,dtype=float)
        performance=planning is not None and bool(planning.success) and not proof_blackout and not self.blackout_holds
        attempt=failed_attempt or PerformanceAttempt('READY' if performance else 'NO_READY_PLAN',
            'Sensed GEKKO candidate ready' if performance else 'Performance plan unavailable or proof blackout',solver_id='gekko-ipopt')
        stage,source_bindings,nominal_matrix={},{},None
        if performance:
            try:
                if (planner is None or not self._planning_model_valid(planner.context)
                        or exact(planner.context.exact_q)!=q0):
                    raise ValueError('Missing or changed nonlinear planning context')
                stage={key:self._stage(state,planning.plans[key],key,base_h) for key in ids}
                source_bindings,nominal_matrix=self._bind_stage_sources(planning,planner,q0,stage)
            except (ValueError,KeyError,TypeError) as error:
                performance=False;attempt=PerformanceAttempt('PREFLIGHT_REFUSED',str(error),solver_id='gekko-ipopt')
        while self.observed_q<outer_end:
            self._execution_stage='PRE_ADMISSION'
            q=self.observed_q
            segment=self.runtime.active_fallback.segment_at(q)
            h=min(base_h,outer_end-q,segment.end-q)
            if h<=0:raise ValueError('No positive hold remains in the installed feedback scope')
            observations=self.observations()
            if not self.blackout_holds:
                self.navigation.condition_on_admitted_fallback(self.runtime.active_fallback,observations)
                observations=self.observations()
            initial_boxes={key:value.state_enclosure for key,value in observations.items()}
            tf,nominal,feedbacks,proposal=None,{},{},None
            successor_preparation,successor_witnesses,tube_memberships={},{},{}
            feedback_coefficients,feedback_evaluations=None,{}
            service_error=None
            self.current_target_flow,self.flow_cache=None,{}
            self.performance_regions={} if planner is None else planner.regions
            if not proof_blackout and not self.blackout_holds:
                try:
                    tf=self.encloser.enclose_target(self.navigation.target,(q,q),(h,h),self.model,self.clock,self.navigation.time)
                except (RuntimeError,TimeoutError) as error:
                    service_error=dict(exception_type=type(error).__name__,message=str(error),service='target-flow')
                    performance=False;attempt=PerformanceAttempt('PROOF_SERVICE_UNAVAILABLE',str(error),solver_id='arb-nonlinear')
                self.current_target_flow=tf
                if tf is not None and not tf.proven:
                    performance=False;attempt=PerformanceAttempt('FLOW_UNRESOLVED',tf.reason,solver_id='arb-nonlinear')
            if performance:
                try:
                    context=planner.context if planner is not None else None
                    if not self._planning_model_valid(context) or exact(context.exact_q)!=q0:
                        raise ValueError('Missing or changed nonlinear planning context')
                    self._check_stage_sources(planning,planner,source_bindings)
                    applied={}
                    controllers=dict(self.runtime.active_fallback.segment_at(q,h).controllers)
                    from orbexa.control.nonlinear_flow import bound_nominal_coefficients
                    try:
                        feedback_coefficients=bound_nominal_coefficients((q,q),planning.parameters,self.encloser.config.precision_bits)
                    except (ImportError,RuntimeError,TimeoutError) as error:
                        raise ValueError('Feedback coefficient source unavailable: '+str(error)) from error
                    for key in ids:
                        sol,spec=planning.plans[key],planning.plans[key].spec
                        measurement=np.array(observations[key].measurement,dtype=float)
                        radius=self._measurement_radius(initial_boxes[key],measurement)
                        if np.any(radius>spec.measurement_error_abs):raise ValueError('Navigation error exceeds planned measurement support')
                        tube_memberships[key]=self._tube_membership(key,q,observations[key],state,lam)
                        a=np.asarray(nominal_matrix(float(q),planning.parameters['time_periapsis']),dtype=float)[3:]
                        fb=boundary_layer_feedback(a,spec.delta_A[0],measurement,state.centers[key],state.phi[key],stage[key]['alpha'],lam)
                        u_nom=np.asarray(sol.result.control_trajectory)[:,0];command=u_nom+fb
                        feedback_evaluations[key]=self._feedback_evaluation(feedback_coefficients,measurement,
                            state.centers[key],state.phi[key],stage[key]['alpha'],lam,u_nom,command)
                        reserve=stage[key]['input_reserve']
                        outcome.reserve_ratios[key]=max(outcome.reserve_ratios[key],float(np.max(np.abs(fb)/np.maximum(reserve,1e-12))))
                        if np.any(np.abs(fb)>reserve):raise ValueError('Sensed feedback exceeds solved input reserve')
                        if not np.allclose(self.scenario.safety.clip_control(command),command,rtol=0,atol=1e-9):raise ValueError('Sensed feedback would saturate')
                        applied[key]=IntervalBox.from_center_radius(command,controllers[key].certificate.spec.actuator_error_abs)
                        try:
                            nominal[key]=self._nominal(state.centers[key],u_nom,q,h,planning.parameters,state.phi[key],state.omega[key],stage[key],lam)
                        except (RuntimeError,TimeoutError) as error:
                            service_error=dict(exception_type=type(error).__name__,message=str(error),service='nominal-surrogate-flow')
                            raise ValueError('Nominal proof service unavailable: '+str(error)) from error
                        feedbacks[key]=fb
                    self._checking_performance=True
                    try:
                        successor=self._performance_successor(q,h,applied,observations,controllers,
                            successor_preparation,successor_witnesses)
                    finally:self._checking_performance=False
                    proposal=PerformanceProposal(q,h,applied,successor,
                        {key:value.fingerprint for key,value in observations.items()},
                        {key:value.certificate.spec.context.fingerprint for key,value in controllers.items()},attempt)
                except (ValueError,KeyError,TypeError,ArithmeticError) as error:
                    performance=False;attempt=PerformanceAttempt('PERFORMANCE_REFUSED',str(error),solver_id='gekko-ipopt',
                        details=getattr(error,'execution_details',()))
            old_plan=self.runtime.active_fallback
            self._checking_performance=True
            try:
                decision=self.runtime.decide(observations,q=q,hold_width=h,proposal=proposal,
                    failed_attempt=attempt,target_flow_fingerprint=None if tf is None else tf.fingerprint)
            finally:self._checking_performance=False
            if successor_preparation:
                successor_preparation['selected_for_motion']=(decision.source=='PERFORMANCE'
                    and decision.successor_fallback_id==successor_preparation.get('successor_id'))
                successor_preparation['selected_decision_id']=decision.fingerprint
            navigation_supports=self._navigation_actuation_supports(decision)
            flows={};blackout=proof_blackout or bool(self.blackout_holds) or tf is None or not tf.proven
            if not blackout:
                try:
                    flows={key:self.flow_provider(key,HeldInputRequest(initial_boxes[key],box,(q,q),(h,h),self.runtime.acceleration_disturbance))
                           for key,box in decision.applied_inputs}
                except (RuntimeError,TimeoutError) as error:
                    if decision.source=='PERFORMANCE':raise
                    service_error=dict(exception_type=type(error).__name__,message=str(error),service='fallback-navigation-flow')
                    blackout=True;flows={}
                if any(not value.proven for value in flows.values()):
                    if decision.source=='PERFORMANCE':raise ValueError('Admitted performance flow disappeared from its exact cache')
                    blackout=True
            navigation_flows={}
            if not blackout:
                try:
                    navigation_flows={key:self.flow_provider(key,HeldInputRequest(initial_boxes[key],value.applied_input,
                        (q,q),(h,h),self.runtime.acceleration_disturbance)) for key,value in navigation_supports.items()}
                    if any(not value.proven for value in navigation_flows.values()):
                        service_error=dict(service='navigation-subset-flow',message='Subset inclusion unresolved; complete admitted flow retained for prediction')
                        navigation_flows=flows
                except (RuntimeError,TimeoutError) as error:
                    service_error=dict(exception_type=type(error).__name__,message=str(error),service='navigation-subset-flow',
                        response='Complete admitted flow retained for conservative prediction')
                    navigation_flows=flows
            if decision.source!='PERFORMANCE':
                performance=False;outcome.used_fallback=True;feedbacks={}
            if decision.source=='PERFORMANCE' and planner is not None and any(not planner.regions[key].contains(flow.swept_state) for key,flow in flows.items()):
                raise ValueError('Admitted performance escaped its prospective nonlinear residual region')
            self._execution_stage='BLACKOUT_OBSERVATION_PREFLIGHT'
            blackout_proposal=self._blackout_proposal(decision,old_plan,observations) if blackout else None
            tube_proofs={}
            if decision.source=='PERFORMANCE':
                for key in ids:
                    residual=copy.deepcopy(planner.residuals[key][0])
                    tube_proofs[key]=dict(initial_membership=tube_memberships[key],
                        exact_stage=stage[key]['proof_record'],nominal_transfer=stage[key]['last_transfer'],
                        source_binding=stage[key]['source_binding'],
                        feedback_evaluation=feedback_evaluations[key],
                        regional_residual=residual,regional_residual_id=fingerprint(residual),
                        region=planner.regions[key].as_dict(),measurement_error_abs=tuple(map(str,map(exact,planning.plans[key].spec.measurement_error_abs))),
                        physical_flow_id=flows[key].fingerprint,source_observation_id=observations[key].fingerprint)
            self._stage_admitted_hold(outcome,q,h,decision,observations,tf,flows,navigation_flows,navigation_supports,
                nominal,blackout,service_error,successor_preparation,successor_witnesses,tube_proofs,
                feedback_coefficients if decision.source=='PERFORMANCE' else None)
            self._execution_stage='NUMERICAL_MOTION_PROPOSAL'
            target_proposal,truth,solutions=self._numeric_hold(q,h,decision,navigation_supports)
            self._execution_stage='PHYSICAL_TRUTH_COMMIT'
            try:self.truth_target.commit_interval(target_proposal)
            finally:
                # If a target commit hook raises after advancing its literal
                # clock, preserve the matching already computed relative truth.
                if self.actual_committed_q==q+h:self.truth_relative_by_agent=truth
            if self.actual_committed_q!=q+h:
                raise ValueError('Physical target commit did not advance the admitted exact clock')
            outcome.actual_committed_q=self.actual_committed_q
            estimate=self._complete_admitted_hold(state,outcome,phase,q,h,decision,tf,flows,navigation_flows,navigation_supports,nominal,
                stage,feedbacks,initial_boxes,blackout,blackout_proposal,truth,solutions,lam,ids)
            if not outcome.success:break
        outcome.actual_truth_relative={key:value.copy() for key,value in self.truth_relative_by_agent.items()}
        outcome.navigation_estimate=estimate if 'estimate' in locals() else self.navigation.estimate()
        outcome.delivered_packets=list(self.drain_delivered_packets())
        outcome.sampled_tube_contained=all(value>=-1e-8 for value in outcome.containment.values())
        return outcome
