"""Sensed dense execution with separate nonlinear truth and finite fallback.

Planning/feedback sees navigation only. Truth is used by the numerical plant,
sensor generator and explicitly labeled post-command containment diagnostics.
"""
from __future__ import annotations

import copy
from types import MappingProxyType
import numpy as np

from orbexa.core.dynamics.reference_frame import exact, fingerprint
from orbexa.core.dynamics.relative import NonlinearRelativeDynamics
from orbexa.control.certification.reachability import IntervalBox, HeldInputRequest, ValidatedFlowEncloser
from orbexa.control.certification.nonlinear import NonlinearFlowEncloser, FRAME
from orbexa.control.tubes.model import boundary_layer_feedback
from orbexa.control.fallback.fleet import FleetAdmissionRuntime, FleetFallbackPlan, PerformanceProposal, PerformanceAttempt
from orbexa.control.fallback.refuge import RefugeObservation
from orbexa.estimation.navigation.service import midpoint, intersect
from orbexa.estimation.membership.nonlinear import NonlinearTransitionObservation
from orbexa.simulation.nonlinear.safety.refuge import BlackoutObservationPropagator
from orbexa.simulation.approach.safety import MissionSafety

from orbexa.control.actuation import (CommandActuationSupport,CommandDependentActuationPrior,
    AffineActuatorRealization,_ACTUATION_SOURCE_SHA256)
from orbexa.control.tubes.sensed import PhysicalTubeWitness, _upper, _exact_sensed_stage, _outward_tube_endpoint, stage_from_plan, feedback_evaluation, measurement_radius, check_tube_observation
from orbexa.simulation.nonlinear.execution.contracts import BlackoutNavigationEstimate, NonlinearExecutionStep, NonlinearExecutionFailure
from orbexa.simulation.nonlinear.execution.plant import NonlinearFleetPlant, sampled_clearance
from orbexa.simulation.nonlinear.execution.sensing import MissionSensorDelivery, SensorAcquisitionAudit
from orbexa.simulation.nonlinear.execution.nominal import NominalTubePropagator
from orbexa.simulation.recording.journal import ExecutionJournal
from orbexa.simulation.nonlinear.execution.bindings import PerformanceStageBindings


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
        self.nominal_tube = NominalTubePropagator()
        self.archive = ExecutionJournal(certificate_sink)
        if completed_hold_sink is not None and not callable(completed_hold_sink):
            raise TypeError('completed_hold_sink must be callable or None')
        self.completed_hold_sink=completed_hold_sink
        if not isinstance(retain_journal,bool):raise TypeError('retain_journal must be boolean')
        self.retain_journal=retain_journal
        self.last_assumption_violation=None
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
        self.sensor_delivery=MissionSensorDelivery(initial_packets,acquisition_audit=SensorAcquisitionAudit(
            self.sensors,clock_fingerprint=self.clock.fingerprint,model_fingerprint=self.model.fingerprint,
            truth_model_fingerprint=self.point_truth.model.fingerprint,sink=self.archive.sink))
        self.sensor_delivery.record_initial()
        self.blackout_holds=[]
        self.blackout_observer,self.blackout_time = None,None
        if self.sensors.last_acquisition_s is None:self._sample()
        self._consume_ready()
        if admission_runtime is not None:self.install_runtime(admission_runtime)

    # Retain established execution attributes while their owners are explicit.
    @property
    def pending_packets(self):return self.sensor_delivery.pending
    @pending_packets.setter
    def pending_packets(self,value):self.sensor_delivery.pending=value

    @property
    def delivered_since_last_step(self):return self.sensor_delivery.delivered
    @delivered_since_last_step.setter
    def delivered_since_last_step(self,value):self.sensor_delivery.delivered=value

    @property
    def journal(self):return self.archive.completed
    @journal.setter
    def journal(self,value):self.archive.completed=value

    @property
    def certificate_sink(self):return self.archive.sink
    @certificate_sink.setter
    def certificate_sink(self,value):
        self.archive.sink=value
        self.sensor_delivery.acquisition_audit.sink=value

    @property
    def last_archive_event(self):return self.archive.last_event
    @last_archive_event.setter
    def last_archive_event(self,value):self.archive.last_event=value

    @property
    def _nominal_services(self):return self.nominal_tube.services
    @_nominal_services.setter
    def _nominal_services(self,value):self.nominal_tube.services=value

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

        The current stored suffix is preferred when both entry and its complete
        fuel reserve fit. A fresh builder can search for an affordable successor.
        Admission subsequently replays these same requests through the cache.
        """
        suffix=self.runtime.active_fallback.suffix(q+h)
        if self.successor_preparer is None:return suffix
        from orbexa.simulation.nonlinear.safety.planner import ProspectiveFleetEndpoint
        from orbexa.control.fallback.refuge import fingerprint as source_fingerprint

        def navigation_id(value):
            return source_fingerprint({name:(dict(item) if isinstance(item,MappingProxyType) else item)
                for name,item in vars(value).items()})

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
        estimate_id=navigation_id(estimate)

        def check_current_state():
            if (self.runtime is not authoritative_runtime or self.runtime.active_fallback is not old_plan
                    or old_plan.fingerprint!=runtime_guard[0]
                    or self.runtime.q!=runtime_guard[1]
                    or source_fingerprint(self.runtime.remaining_fuel)!=runtime_guard[2]
                    or self.runtime._pending is not runtime_guard[3]
                    or self.runtime.acceleration_disturbance!=runtime_guard[4]
                    or navigation_id(self.navigation.estimate())!=estimate_id):
                raise RuntimeError('Prospective proof service altered authoritative current execution state')

        try:
            flows={key:self.flow_provider(key,HeldInputRequest(
                controllers[key].observation_box(observations[key],q),applied[key],
                (q,q),(h,h),self.runtime.acceleration_disturbance)) for key in observations}
        except (RuntimeError,TimeoutError) as error:
            raise ValueError('Prospective performance flow service unavailable: '+str(error)) from error
        finally:check_current_state()
        if any(not value.proven for value in flows.values()):
            raise ValueError('Prospective performance flow is unresolved')
        from orbexa.control.fallback.fuel import FleetFuelBudget
        domains={value.certificate.spec.context.fuel_domain_id for value in controllers.values()}
        if len(domains)!=1:
            raise ValueError('Prospective fleet has inconsistent physical resource domains')
        remaining=dict(self.runtime.remaining_fuel)
        try:
            charges={key:self.runtime.performance_fuel_charge(key,applied[key],q,h,controllers[key])
                for key in observations}
        finally:check_current_state()
        budget=FleetFuelBudget(q+h,{key:remaining[key]-charges[key] for key in observations},next(iter(domains)))
        stored_contains=suffix.contains({key:value.endpoint for key,value in flows.items()},q+h)
        stored_affordable=not budget.shortfalls(suffix.remaining_fuel(q+h))
        record.update(performance_hold_fuel_upper={key:str(value) for key,value in charges.items()},
            prospective_fuel_budget=budget.as_dict(),stored_suffix_contains_endpoints=stored_contains,
            stored_suffix_fuel_feasible=stored_affordable)
        if stored_contains and stored_affordable:return suffix
        projection=ProspectiveFleetEndpoint.from_flows(flows=flows,target_flow=self.current_target_flow,
            observations=observations,contexts={key:value.certificate.spec.context for key,value in controllers.items()},
            acceleration_disturbance=self.runtime.acceleration_disturbance,model=self.model,clock=self.clock,
            fuel_budget=budget)
        record.update(status='PREPARING',projection_id=projection.fingerprint,exact_start=str(q),
            exact_end=str(q+h),stored_suffix_id=suffix.fingerprint,stored_suffix_contains_endpoints=stored_contains,
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
                stored_suffix_id=suffix.fingerprint,stored_suffix_contains_endpoints=stored_contains,source_navigation_id=estimate_id,
                archive_storage='DURABLE_EXTERNAL_SINK' if self.certificate_sink is not None else 'STAGED_OUTCOME_MEMORY')
            return candidate
        except Exception as error:
            record.update(status='REFUSED',error_type=type(error).__name__,reason=str(error),
                acknowledged_witness_ids=tuple(acknowledged))
            if getattr(error,'event',None) is not None:
                from orbexa.control.fallback.refuge import exact_record
                record['proof_refusal_event']=exact_record(error.event)
            raise ValueError('Prospective successor refused: '+type(error).__name__+': '+str(error)) from error
        finally:
            # A builder receives immutable data and never owns current runtime
            # installation, fuel, observation time or navigation state.
            check_current_state()

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
        return stage_from_plan(state,plan,key,h,sliding_gains=self.config.tube.sliding_gains)

    def _stage_bindings(self):
        return PerformanceStageBindings(config=self.config,clock=self.clock,runtime=self.runtime,
            observed_q=self.observed_q,precision_bits=self.encloser.config.precision_bits)

    def _stage_source_snapshot(self,planning,planner,key,*,_target_identity=None):
        return self._stage_bindings().snapshot(planning,planner,key,_target_identity=_target_identity)

    def _bind_stage_sources(self,planning,planner,q0,stage):
        return self._stage_bindings().bind(planning,planner,q0,stage)

    def _check_stage_sources(self,planning,planner,bindings):
        return self._stage_bindings().check(planning,planner,bindings)

    def _feedback_evaluation(self,coefficients,measurement,center,phi,alpha,lam,u_nom,command):
        return feedback_evaluation(coefficients,measurement,center,phi,alpha,lam,u_nom,command,
            numerical_reserve=self.config.tube.numerical_reserve)

    @staticmethod
    def _measurement_radius(box,point):
        return measurement_radius(box,point)

    def _check_tube_observation(self,box,center,phi,omega,lam):
        return check_tube_observation(box,center,phi,omega,lam)

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
        return self.nominal_tube.propagate(center,u,q,h,params,phi,omega,stage,lam,
            input_bounds=self.config.mpc.input_bounds,clock_domain=self.clock.q_domain,
            nominal_config=self.nominal_encloser.config,precision_bits=self.encloser.config.precision_bits)

    def _numeric_hold(self,q,h,decision,navigation_supports):
        plant=NonlinearFleetPlant(self.truth_target,self.point_truth,self.actuator)
        return plant.propose_hold(self.truth_relative_by_agent,q,h,decision,navigation_supports)

    def _numeric_clearance(self,state,q,phase):
        return sampled_clearance(state,q,phase,target=self.truth_target,safety=self.truth_safety,
            composite_center=self._truth_composite_center)

    def _sample(self):
        return self.sensor_delivery.sample(self.sensors,self.truth_target,self.clock,self.truth_relative_by_agent)

    def _consume_ready(self):
        return self.sensor_delivery.consume_ready(self.navigation)

    def drain_delivered_packets(self):
        return self.sensor_delivery.drain()

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
        audit=self.sensor_delivery.acquisition_audit
        audit.require_confirmed()
        if ready:audit.delivered(ready,candidate)
        estimate=candidate.consume(ready)
        if ready:audit.delivered(ready,candidate,estimate=estimate)
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
        return self.archive.publish(kind,record,outcome,target=target,flows=flows,nominal=nominal,
            navigation_flows=navigation_flows,successor_witnesses=successor_witnesses)

    def _stage_admitted_hold(self,outcome,q,h,decision,observations,tf,flows,navigation_flows,navigation_supports,nominal,blackout,service_error,
            successor_preparation=None,successor_witnesses=None,performance_tube_proofs=None,feedback_coefficients=None):
        from orbexa.control.fallback.refuge import exact_record
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
            if (estimate.exact_event_q!=q+h or estimate.clock_fingerprint!=self.clock.fingerprint
                    or estimate.model_fingerprint!=self.model.fingerprint):
                raise ValueError('Membership navigation source differs from the executed hold endpoint')
            membership_target=estimate.target_ephemeris_box
            membership_target_record=membership_target.as_dict()
            membership_navigation_id=fingerprint(dict(exact_event_q=str(estimate.exact_event_q),
                physical_time_box=estimate.physical_time_box.as_dict(),target_ephemeris_box=membership_target_record,
                clock_fingerprint=estimate.clock_fingerprint,model_fingerprint=estimate.model_fingerprint,
                packet_ids=estimate.packet_ids,sensor_fingerprints=estimate.sensor_fingerprints))
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
                        input_support=navigation_supports[key].fingerprint,actuation_prior=self.actuation_prior.fingerprint,
                        agent_id=key,actuation_persistence='fixed-affine-per-agent-v1',
                        navigation=membership_navigation_id,target_endpoint_q=str(estimate.exact_event_q),
                        target_observed=membership_target_record)),target_observed=membership_target,
                    agent_id=key,commanded_input=navigation_supports[key].command,
                    actuation_prior=self.actuation_prior,actuation_persistence='fixed-affine-per-agent-v1')))
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
        self.archive.record_completed(outcome,record,retain=self.retain_journal)
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
            h=min(base_h,outer_end-q,segment.end-q,
                *(controller.certificate.spec.maximum_hold for _,controller in segment.controllers))
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
                    from orbexa.control.certification.nonlinear import bound_nominal_coefficients
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
