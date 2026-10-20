"""Prepare complete nonlinear fleet refuge proofs from navigation snapshots."""
from __future__ import annotations

from dataclasses import dataclass,field,replace
from fractions import Fraction
from types import MappingProxyType
import numpy as np

from orbexa.control.attitude_flow import AttitudeFlowEncloser, _SOURCE_SHA256 as ATTITUDE_SOURCE_SHA256
from orbexa.control.fleet_fallback import FleetGeometryOracle,FleetAdmissionRuntime
from orbexa.control.fuel_budget import FleetFuelBudget
from orbexa.control.refuge import ControlContext,RefugeObservation
from orbexa.control.reachability import (IntervalBox,HeldInputRequest,FlowCertificate,
    SweptCollisionGuard,EnclosureStatus)
from orbexa.control.nonlinear_flow import (TargetEphemerisCertificate,
    _SOURCE_SHA256 as TARGET_SOURCE_SHA256)
from orbexa.core.reference_frame import exact
from orbexa.simulation.nonlinear_refuge import (C2RefugeReference,NonlinearZeroDriftBounds,
    StudyPropulsionBounds,CompositeOccupiedBoundsProvider,AnyProperTargetPose,
    AttitudePoseBoundsProvider,construct_refuge_plan,construct_patch_refuge_candidates,ZERO_NOMINAL_ID)
from orbexa.simulation.fleet_braking import coordinated_braking_references
from orbexa.simulation.contingency_reference import (
    AccelerationContingencyOptions, acceleration_contingency_references)
from orbexa.simulation.proof_provider import FixedProofProvider
from orbexa.control.refuge import exact_record,fingerprint


class _ForecastEnclosureRefusal(ValueError):
    """A declared target/pose enclosure could not prove its requested scope."""


class _ExtendedContingencyForecastRefusal(ValueError):
    """Only an optional longer proposal enlarged the refused forecast."""

    def __init__(self, reason, extended_width, legacy_width):
        super().__init__(reason)
        self.extended_width = extended_width
        self.legacy_width = legacy_width


def radius(box):
    return np.array([float((hi-lo)/2) for lo,hi in zip(box.lower,box.upper)])


def _named(value,name):
    pairs=tuple(value.items()) if hasattr(value,'items') else tuple(value)
    if (not pairs or any(not isinstance(key,str) or not key.strip() for key,_ in pairs)
            or len({key for key,_ in pairs})!=len(pairs)):
        raise ValueError('Complete distinct named '+name+' are required')
    return pairs


@dataclass(frozen=True)
class ProspectiveFleetEndpoint:
    """Full held-flow projection, never an acquired navigation observation.

    This study interface deliberately requires the complete declared global-input
    flows. Narrower command-dependent navigation flows cannot create a backup
    entry region. The owning runtime still admits the original flows and the
    resulting successor together before motion.
    """
    flows: object
    target_flow: TargetEphemerisCertificate
    observations: object
    contexts: object
    acceleration_disturbance: IntervalBox
    plant_fingerprint: str
    target_model_fingerprint: str
    clock_fingerprint: str
    frame_id: str='reference-orbit-frame-v1'
    fuel_budget: FleetFuelBudget | None=None
    _flow_ids: tuple=field(init=False,repr=False)
    _target_id: str=field(init=False,repr=False)
    _identity: str=field(init=False,repr=False)
    _fuel_budget_id: str | None=field(init=False,repr=False)

    @classmethod
    def from_flows(cls,*,flows,target_flow,observations,contexts,model,clock,
                   acceleration_disturbance=None,fuel_budget=None):
        value=cls(flows,target_flow,observations,contexts,
            IntervalBox.point((0,0,0)) if acceleration_disturbance is None else acceleration_disturbance,
            model.fingerprint,model.target_fingerprint,clock.fingerprint,fuel_budget=fuel_budget)
        clock.check_domain((value.start_q,value.end_q))
        return value

    def __post_init__(self):
        for name in ('flows','observations','contexts'):
            object.__setattr__(self,name,_named(getattr(self,name),name))
        ids=tuple(key for key,_ in self.contexts)
        if not ids or any(set(dict(getattr(self,name)))!=set(ids) for name in ('flows','observations')):
            raise ValueError('A prospective study successor requires the complete same declared fleet')
        if (not isinstance(self.acceleration_disturbance,IntervalBox)
                or self.acceleration_disturbance.dimension!=3):
            raise ValueError('Prospective acceleration disturbance needs three exact components')
        target=self.target_flow
        if (not isinstance(target,TargetEphemerisCertificate) or not target.proven
                or target.model!='nonlinear-target-ephemeris-v1' or target.frame!='inertial-cartesian-si-v1'
                or target.start_anomaly[0]!=target.start_anomaly[1]
                or target.hold_width[0]!=target.hold_width[1] or target.hold_width[0]<=0
                or target.model_bounds.get('target_model_fingerprint')!=self.target_model_fingerprint
                or target.model_bounds.get('clock_fingerprint')!=self.clock_fingerprint):
            raise ValueError('Prospective endpoint requires a complete exact same-model target hold')
        SweptCollisionGuard._check_flow_structure(target)
        for name in ('physical_time_initial','physical_time_endpoint','physical_time_swept'):
            box=getattr(target,name)
            if not isinstance(box,IntervalBox) or box.dimension!=1:
                raise ValueError('Prospective target requires its full physical-time projection')
        if (not target.physical_time_swept.contains(target.physical_time_initial)
                or not target.physical_time_swept.contains(target.physical_time_endpoint)
                or len(target.segment_time_boxes)!=len(target.segments)
                or any(not isinstance(box,IntervalBox) or box.dimension!=1
                    or not target.physical_time_swept.contains(box) for box in target.segment_time_boxes)):
            raise ValueError('Prospective target physical-time chain is incomplete')
        expected_target_request=dict(initial_state=target.initial_state.as_dict(),
            start_anomaly=tuple(map(str,target.start_anomaly)),hold_width=tuple(map(str,target.hold_width)),
            physical_time_initial=target.physical_time_initial.as_dict())
        if target.request!=expected_target_request:
            raise ValueError('Prospective target request differs from its proved initial/time boxes')
        target_id=target.fingerprint
        contexts,observations=dict(self.contexts),dict(self.observations)
        if self.fuel_budget is not None:
            if type(self.fuel_budget) is not FleetFuelBudget:
                raise TypeError('Prospective resources need an exact FleetFuelBudget')
            for context in contexts.values():
                self.fuel_budget.check_context(ids,self.end_q,context.fuel_domain_id)
        object.__setattr__(self,'_fuel_budget_id',None if self.fuel_budget is None else self.fuel_budget.fingerprint)
        for key,flow in self.flows:
            context,observation=contexts[key],observations[key]
            if not isinstance(context,ControlContext) or not isinstance(observation,RefugeObservation):
                raise TypeError('Prospective flow sources need typed controller contexts and observations')
            if (context.parameter_set_id!=self.plant_fingerprint
                    or context.target_model_id!=self.target_model_fingerprint
                    or context.clock_id!=self.clock_fingerprint or context.frame!=self.frame_id
                    or context.model_id!='nonlinear-reference-relative-v1'
                    or observation.q!=self.start_q or observation.clock_id!=self.clock_fingerprint
                    or observation.frame!=self.frame_id or observation.sensor_id!=context.sensor_id):
                raise ValueError('Prospective context/observation model, sensor, frame or exact clock changed')
            if (not isinstance(flow,FlowCertificate) or not flow.proven
                    or flow.start_anomaly!=target.start_anomaly or flow.hold_width!=target.hold_width
                    or flow.model!=context.model_id or flow.frame!=context.frame
                    or flow.model_bounds.get('plant_fingerprint')!=self.plant_fingerprint
                    or flow.model_bounds.get('target_model_fingerprint')!=self.target_model_fingerprint
                    or flow.model_bounds.get('clock_fingerprint')!=self.clock_fingerprint
                    or flow.model_bounds.get('target_flow_fingerprint')!=target_id):
                raise ValueError('Prospective full relative flow is unresolved, stale or from another model/frame/clock')
            SweptCollisionGuard._check_flow_structure(flow)
            if not flow.initial_state.contains(observation.state_enclosure):
                raise ValueError('Prospective flow omits part of its retained source observation')
            raw=flow.request.get('held_input',{})
            applied=IntervalBox(raw.get('exact_lower',()),raw.get('exact_upper',()))
            if applied.dimension!=3 or any((hi-lo)/2<300 for lo,hi in zip(applied.lower,applied.upper)):
                raise ValueError('Prospective successors require full global +/-300 input uncertainty')
            request=HeldInputRequest(flow.initial_state,applied,target.start_anomaly,target.hold_width,
                self.acceleration_disturbance)
            if flow.request!=request.as_dict():
                raise ValueError('Prospective flow request or additional disturbance changed')
        object.__setattr__(self,'_flow_ids',tuple((key,flow.fingerprint) for key,flow in self.flows))
        object.__setattr__(self,'_target_id',target_id)
        object.__setattr__(self,'_identity',fingerprint(self.as_dict(include_flows=False,include_identity=False)))

    @property
    def start_q(self):return self.target_flow.start_anomaly[0]
    @property
    def hold_width(self):return self.target_flow.hold_width[0]
    @property
    def end_q(self):return self.start_q+self.hold_width
    @property
    def ids(self):return tuple(key for key,_ in self.contexts)
    @property
    def endpoint_boxes(self):return MappingProxyType({key:flow.endpoint for key,flow in self.flows})
    @property
    def fingerprint(self):return self._identity

    def verify_integrity(self):
        if (self.target_flow.fingerprint!=self._target_id
                or tuple((key,flow.fingerprint) for key,flow in self.flows)!=self._flow_ids
                or (None if self.fuel_budget is None else self.fuel_budget.fingerprint)!=self._fuel_budget_id):
            raise ValueError('Prospective flow/request context changed after its projection was prepared')

    def as_dict(self,*,include_flows=True,include_identity=True):
        result=dict(method='complete-global-held-flow-projection-v1',
            exact_start_q=str(self.start_q),exact_end_q=str(self.end_q),hold_width=str(self.hold_width),
            plant_fingerprint=self.plant_fingerprint,target_model_fingerprint=self.target_model_fingerprint,
            clock_fingerprint=self.clock_fingerprint,frame_id=self.frame_id,
            flow_ids=self._flow_ids,target_flow_id=self._target_id,
            observation_ids=tuple((key,value.fingerprint) for key,value in self.observations),
            context_ids=tuple((key,value.fingerprint) for key,value in self.contexts),
            endpoint_boxes={key:box.as_dict() for key,box in self.endpoint_boxes.items()},
            target_endpoint=self.target_flow.endpoint.as_dict(),
            physical_time_endpoint=self.target_flow.physical_time_endpoint.as_dict(),
            acceleration_disturbance=self.acceleration_disturbance.as_dict(),
            global_actuator_error_abs=('300',)*3,
            scope='Prospective proved endpoint sets; no future sensor packet or navigation observation is asserted')
        if self.fuel_budget is not None:
            result.update(fuel_budget=self.fuel_budget.as_dict(),fuel_budget_id=self._fuel_budget_id)
        if include_identity:result['identity']=self.fingerprint
        if include_flows:
            result.update(flows={key:flow.as_dict() for key,flow in self.flows},
                target_flow=self.target_flow.as_dict(),source_observations=exact_record(self.observations),
                source_contexts=exact_record(self.contexts))
        return result


@dataclass(frozen=True)
class _SafetyInitialSet:
    exact_event_q: object
    relative_state_box_by_agent: object
    target_ephemeris_box: IntervalBox
    physical_time_box: IntervalBox
    frame_id: str
    source_kind: str
    source_id: str
    acceleration_disturbance: IntervalBox=field(default_factory=lambda:IntervalBox.point((0,0,0)))

    @property
    def point_relative_by_agent(self):
        return {key:tuple((a+b)/2 for a,b in zip(box.lower,box.upper))
                for key,box in self.relative_state_box_by_agent.items()}


class NonlinearSafetyPlanner:
    """The optimizer never installs its own unproved successor or a stopped clock."""
    def __init__(self,scenario,*,model,clock,encloser,study_options,design,
                 actuator_error_abs=(300.,300.,300.),attitude_encloser=None,
                 near_feedback_subdivisions=1,acceleration_contingency=None):
        self.scenario,self.model,self.clock=scenario,model,clock
        self.encloser,self.options,self.design=encloser,study_options,design
        self.actuator_error_abs=tuple(actuator_error_abs)
        if (isinstance(near_feedback_subdivisions,bool)
                or not isinstance(near_feedback_subdivisions,int)
                or near_feedback_subdivisions<1):
            raise ValueError('Near feedback subdivisions must be a positive integer')
        self.near_feedback_subdivisions=near_feedback_subdivisions
        if acceleration_contingency is not None and not isinstance(acceleration_contingency,AccelerationContingencyOptions):
            raise TypeError('Declare typed acceleration-contingency proposal options')
        self.acceleration_contingency=acceleration_contingency
        self.attitude_encloser=attitude_encloser or AttitudeFlowEncloser(encloser.config)
        self.propulsion=StudyPropulsionBounds(mass_upper_kg=design.chaser.mass_kg,
            installed_force_n=design.translation_force_n,specific_impulse_lower_s=design.specific_impulse_s,
            available_fuel_kg=study_options.fallback_fuel_reserve_kg,
            # Three torque couples can fire both signed-location channels on
            # every axis simultaneously; the allocator has no aggregate clamp.
            attitude_force_allocation_n=6*exact(design.attitude_nozzle_force_n))
        self.sensor_id=study_options.sensor.fingerprint
        self.actuator_id='bounded-q-gain-axis-bias-v1'
        self.installations=[]

    def observations(self,estimate):
        return {key:RefugeObservation(estimate.exact_event_q,estimate.point_relative_by_agent[key],box,
            self.sensor_id,estimate.frame_id,self.clock.fingerprint)
            for key,box in estimate.relative_state_box_by_agent.items()}

    def target_forecast(self,estimate,width):
        q,width=exact(estimate.exact_event_q),exact(width)
        value=self.encloser.enclose_target(estimate.target_ephemeris_box,(q,q),(width,width),
            self.model,self.clock,estimate.physical_time_box)
        if (not isinstance(value,TargetEphemerisCertificate)
                or value.model!='nonlinear-target-ephemeris-v1'
                or value.frame!='inertial-cartesian-si-v1'
                or value.start_anomaly!=(q,q) or value.hold_width!=(width,width)
                or value.initial_state!=estimate.target_ephemeris_box
                or value.physical_time_initial!=estimate.physical_time_box
                or value.model_bounds.get('backend_source_sha256')!=TARGET_SOURCE_SHA256
                or value.model_bounds.get('target_model_fingerprint')!=self.model.target_fingerprint
                or value.model_bounds.get('clock_fingerprint')!=self.clock.fingerprint):
            raise ValueError('Nonlinear safety target forecast source, model, request or clock changed')
        if not value.proven:
            if value.status==EnclosureStatus.UNRESOLVED:
                raise _ForecastEnclosureRefusal('Nonlinear safety target forecast unresolved: '+value.reason)
            raise ValueError('Nonlinear safety target forecast unavailable: '+value.reason)
        return value

    def prepare(self,estimate,*,flow_provider,remaining_fuel=None,near_target=None,certificate_sink=None):
        self._check_sensing_prior(estimate)
        source_id=fingerprint(self._prior_record(estimate))
        initial=_SafetyInitialSet(exact(estimate.exact_event_q),dict(estimate.relative_state_box_by_agent),
            estimate.target_ephemeris_box,estimate.physical_time_box,estimate.frame_id,
            'CURRENT_NAVIGATION_PRIOR',source_id)
        budget=(None if remaining_fuel is None else
            FleetFuelBudget(initial.exact_event_q,remaining_fuel,self.propulsion.fingerprint))
        plan,record,witness=self._build_plan(initial,sensing_prior=estimate,near_target=near_target,
            make_witness=certificate_sink is not None,fuel_budget=budget)
        runtime=FleetAdmissionRuntime(plan,self.observations(estimate),flow_provider=flow_provider,
            remaining_fuel=remaining_fuel,performance_resource_bound=self.propulsion.callback(self.clock))
        self._publish_witness(record,witness,certificate_sink,'fallback_installation')
        self.installations.append(record)
        return runtime,record

    def retain_runtime(self,runtime,estimate,*,width,flow_provider,certificate_sink=None):
        """Preflight an already committed invariant; never grant fresh entry.

        A new observation rectangle may protrude from the relational tube.
        Its original sensing support and consistency still have to pass. This
        method neither replaces the plan nor advances its clock/fuel ledger.
        """
        self._check_sensing_prior(estimate)
        if any(value!=self.sensor_id for value in estimate.sensor_fingerprints):
            raise ValueError('Reusable runtime navigation sensor-source declaration changed')
        if not isinstance(runtime,FleetAdmissionRuntime) or not callable(flow_provider):
            raise TypeError('Retaining fallback needs its existing typed runtime and actual flow provider')
        q,h=exact(estimate.exact_event_q),exact(width)
        if h<=0 or runtime.q!=q or runtime._pending is not None:
            raise ValueError('Reusable runtime has a stale clock, pending decision or nonpositive requested scope')
        if runtime.flow_provider!=flow_provider:
            raise ValueError('Reusable runtime flow-provider source changed')
        plan=runtime.active_fallback;plan.verify_integrity()
        if (not plan.certificate.admissible or set(plan.ids)!=set(estimate.relative_state_box_by_agent)
                or not plan.certificate.start<=q<q+h<=plan.certificate.end):
            raise ValueError('Reusable runtime lacks a complete fleet or the entire requested finite scope')
        self.clock.check_domain((q,q+h))
        oracle=plan.geometry_oracle
        expected_radii=tuple((key,exact(self.design.chaser.body_radius_m)) for key in plan.ids)
        if (oracle.geometry_id!=self.design.geometry.fingerprint or oracle.target_model_id!=self.model.target_fingerprint
                or oracle.frame!=estimate.frame_id or oracle.clock_id!=self.clock.fingerprint
                or oracle.component_ids!=tuple(part.id for part in self.design.geometry.components)
                or oracle.body_radii!=expected_radii
                or oracle.clearance!=exact(self.scenario.config.mission.target_barrier_clearance)
                or oracle.minimum_pair_center_distance!=exact(self.scenario.config.mpc.min_chaser_separation)):
            raise ValueError('Reusable runtime physical geometry, frame/clock or clearance declaration changed')
        dense=exact(self.scenario.config.anom_step)/self.scenario.config.tube.feedback_substeps
        declared_holds=(dense,dense/self.near_feedback_subdivisions)
        inputs=IntervalBox(tuple(float(b['lower']) for b in self.scenario.config.mpc.input_bounds),
                           tuple(float(b['upper']) for b in self.scenario.config.mpc.input_bounds))
        actuator=tuple(map(exact,self.actuator_error_abs))
        expected=dict(model_id='nonlinear-reference-relative-v1',nominal_model_id=ZERO_NOMINAL_ID,
            parameter_set_id=self.model.fingerprint,target_model_id=self.model.target_fingerprint,
            frame=estimate.frame_id,clock_id=self.clock.fingerprint,sensor_id=self.sensor_id,
            actuator_id=self.actuator_id,geometry_id=self.design.geometry.fingerprint,
            fuel_domain_id=self.propulsion.fingerprint)
        extra=tuple(max(abs(a),abs(b)) for a,b in zip(runtime.acceleration_disturbance.lower,runtime.acceleration_disturbance.upper))
        if runtime.acceleration_disturbance.dimension!=3:
            raise ValueError('Reusable runtime acceleration-disturbance domain changed')
        for segment in plan.segments:
            for _,controller in segment.controllers:
                cert=controller.certificate;spec=cert.spec
                if (any(getattr(spec.context,name)!=value for name,value in expected.items())
                        or spec.maximum_hold not in declared_holds or spec.input_limits!=inputs
                        or spec.actuator_error_abs!=actuator
                        or spec.available_fuel!=self.propulsion.available_fuel_kg):
                    raise ValueError('Reusable runtime plant/sensor/actuator/fuel/input or sampled-hold declaration changed')
                adapter=controller.builder.drift_bounds_provider
                if not isinstance(adapter,NonlinearZeroDriftBounds) or any(a>b for a,b in zip(extra,adapter.extra_acceleration_abs)):
                    raise ValueError('Reusable runtime extra disturbance lacks its original nonlinear bound')
                # The study adapter holds an immutable target proof behind a
                # lookup closure. This is source verification, not a new flow.
                target=adapter.target_provider(spec.start,spec.end)
                if not isinstance(target,TargetEphemerisCertificate) or not target.proven:
                    raise ValueError('Reusable runtime nonlinear target source is incomplete')
                source=fingerprint(dict(target=adapter._identities.identity(target),model=adapter.model.fingerprint,
                    clock=adapter.clock.fingerprint,region=exact_record(cert.state_box),
                    scope=(str(spec.start),str(spec.end)),extra=exact_record(adapter.extra_acceleration_abs),precision=adapter.precision_bits))
                if cert.drift_bounds.proof_reference!='Arb-actual-nonlinear-drift:'+source:
                    raise ValueError('Reusable runtime nonlinear drift/source declaration changed')
        if set(runtime.remaining_fuel)!=set(plan.ids):
            raise ValueError('Reusable runtime fuel ledger omitted a fleet member')
        needed=plan.remaining_fuel(q)
        if any(exact(runtime.remaining_fuel[key])<value for key,value in needed.items()):
            raise ValueError('Reusable runtime lacks fuel for its complete remaining finite route')
        observations=self.observations(estimate)
        for key,controller in plan.segment_at(q).controllers:
            controller.observation_box(observations[key],q)
        record=dict(kind='fallback_reuse',status='REUSED',exact_start=str(q),exact_required_end=str(q+h),
            coverage_end=str(plan.certificate.end),active_plan_id=plan.fingerprint,
            source_plan_id=plan.certificate.source_plan_id,navigation_id=fingerprint(self._prior_record(estimate)),
            model_id=self.model.fingerprint,clock_id=self.clock.fingerprint,
            remaining_fuel=exact_record(runtime.remaining_fuel),required_remaining_fuel=exact_record(needed),
            new_installation=False,full_witness_archived=False)
        identity=fingerprint(record)
        if certificate_sink is not None:
            if not callable(certificate_sink):raise TypeError('Fallback reuse archive sink must acknowledge a callable append')
            certificate_sink(dict(identity=identity,**record))
        return dict(record,reuse_record_id=identity,reuse_record_archived=certificate_sink is not None,
                    archived_witness_id=identity if certificate_sink is not None else None)

    def _check_sensing_prior(self,estimate):
        from orbexa.estimation.navigation import NavigationEstimate
        if (not isinstance(estimate,NavigationEstimate) or estimate.status!='VALID'
                or estimate.model_fingerprint!=self.model.fingerprint
                or estimate.clock_fingerprint!=self.clock.fingerprint
                or estimate.frame_id!='reference-orbit-frame-v1'):
            raise ValueError('Safety preparation needs a typed same-model/frame/clock valid current navigation prior')
        declared=tuple(c.chaser_id for c in self.scenario.fleet.chasers('multi'))
        expected=set(declared)
        if (not declared or len(expected)!=len(declared)
                or any(not isinstance(key,str) or not key.strip() for key in declared)
                or set(estimate.relative_state_box_by_agent)!=expected):
            raise ValueError('Safety preparation requires the complete unique declared navigation fleet')

    @staticmethod
    def _prior_record(estimate):
        return dict(exact_event_q=str(estimate.exact_event_q),
            physical_time_box=estimate.physical_time_box.as_dict(),
            relative_state_boxes={key:box.as_dict() for key,box in estimate.relative_state_box_by_agent.items()},
            target_ephemeris_box=estimate.target_ephemeris_box.as_dict(),
            target_orientation_matrix=exact_record(estimate.target_orientation_matrix),
            target_orientation_operator_error=exact_record(estimate.target_orientation_operator_error),
            target_angular_velocity_box=estimate.target_angular_velocity_box.as_dict(),
            packet_ids=tuple(estimate.packet_ids),sensor_fingerprints=tuple(estimate.sensor_fingerprints),
            model_fingerprint=estimate.model_fingerprint,clock_fingerprint=estimate.clock_fingerprint,
            frame_id=estimate.frame_id,status=estimate.status)

    def prepare_successor(self,projection,*,sensing_prior,certificate_sink=None):
        """Certify a future backup from full proved endpoints without installing it.

        The current navigation prior supplies current pose/rate and sensor
        identities only. No future packet arrival, observation contraction or
        VALID future estimate is manufactured. The existing fleet transaction
        must still admit the original performance hold and this complete plan.
        """
        if not isinstance(projection,ProspectiveFleetEndpoint):
            raise TypeError('A complete typed prospective fleet endpoint is required')
        if not callable(certificate_sink):
            raise ValueError('A full successor witness sink must acknowledge preparation before motion')
        self._check_sensing_prior(sensing_prior)
        projection.verify_integrity()
        if (projection.plant_fingerprint!=self.model.fingerprint
                or projection.target_model_fingerprint!=self.model.target_fingerprint
                or projection.clock_fingerprint!=self.clock.fingerprint
                or projection.frame_id!=sensing_prior.frame_id
                or projection.start_q!=exact(sensing_prior.exact_event_q)
                or set(projection.ids)!=set(sensing_prior.relative_state_box_by_agent)
                or projection.target_flow.initial_state!=sensing_prior.target_ephemeris_box
                or projection.target_flow.physical_time_initial!=sensing_prior.physical_time_box):
            raise ValueError('Prospective endpoint is not projected from this exact current navigation target/time prior')
        contexts=dict(projection.contexts)
        if any(value.sensor_id!=self.sensor_id or value.geometry_id!=self.design.geometry.fingerprint
               or value.actuator_id!=self.actuator_id or value.fuel_domain_id!=self.propulsion.fingerprint
               for value in contexts.values()):
            raise ValueError('Prospective endpoint changed sensor, actuator, geometry or fuel hardware')
        initial=_SafetyInitialSet(projection.end_q,dict(projection.endpoint_boxes),
            projection.target_flow.endpoint,projection.target_flow.physical_time_endpoint,
            projection.frame_id,'PROVED_FULL_PERFORMANCE_ENDPOINT',projection.fingerprint,
            projection.acceleration_disturbance)
        abandoned_forecast=None
        try:
            plan,record,witness=self._build_plan(initial,sensing_prior=sensing_prior,
                make_witness=True,projection=projection,fuel_budget=projection.fuel_budget)
        except _ExtendedContingencyForecastRefusal as error:
            # An optional proposal must not suppress the shorter, previously
            # available finite search merely by enlarging its common forecast.
            # No motion or witness publication has occurred at this point.
            abandoned_forecast=dict(status='REFUSED',reason=str(error),
                forecast_width_q=str(error.extended_width),
                legacy_forecast_width_q=str(error.legacy_width),
                scope='Optional acceleration-prefix forecast; no plan installed')
            plan,record,witness=self._build_plan(initial,sensing_prior=sensing_prior,
                make_witness=True,projection=projection,fuel_budget=projection.fuel_budget,
                allow_acceleration_prefix=False)
        # Recheck mutable producer payloads after the potentially long bound
        # construction, before acknowledging any candidate witness.
        projection.verify_integrity()
        if not plan.contains(projection.endpoint_boxes,projection.end_q):
            raise ValueError('Complete projected endpoints do not enter the prepared successor')
        record.update(projection_id=projection.fingerprint,status='SUCCESSOR_PREPARED',
            installed=False,scope='Prepared prospective candidate; only a committed fleet transaction installs it')
        witness.update(projection_id=projection.fingerprint,
            projection=projection.as_dict(include_flows=True))
        if abandoned_forecast is not None:
            record['abandoned_optional_forecast']=abandoned_forecast
            witness['abandoned_optional_forecast']=abandoned_forecast
        self._publish_witness(record,witness,certificate_sink,'fallback_successor')
        return plan,record

    @staticmethod
    def _publish_witness(record,witness,certificate_sink,kind):
        if witness is None:return
        witness=dict(witness,kind=kind)
        witness_id=fingerprint(witness)
        certificate_sink(dict(identity=witness_id,**witness))
        record.update(archived_witness_id=witness_id,full_witness_archived=True)

    def _build_plan(self,estimate,*,sensing_prior,near_target=None,make_witness=False,projection=None,fuel_budget=None,
                    allow_acceleration_prefix=True,_extended_near=False):
        q=exact(estimate.exact_event_q)
        points={key:np.asarray(point,dtype=float) for key,point in estimate.point_relative_by_agent.items()}
        if fuel_budget is not None:
            if type(fuel_budget) is not FleetFuelBudget:
                raise TypeError('Finite continuation search needs an exact FleetFuelBudget')
            fuel_budget.check_context(points,q,self.propulsion.fingerprint)
        near=bool(max(np.linalg.norm(v[:3]) for v in points.values())<30. if near_target is None else near_target)
        # One close member needs the observed-pose geometry for the whole
        # fleet, even while other members still need the far braking law.
        # Geometry selection must not switch the mixed fleet's tracking
        # gains, force bounds, reference policy or feedback hold.
        pose_conditioned_geometry=near or any(np.linalg.norm(v[:3])<30. for v in points.values())
        dense=exact(self.scenario.config.anom_step)/self.scenario.config.tube.feedback_substeps
        # All route and proof-window boundaries share the executed rational
        # clock. Decimal approximations can otherwise create microscopic
        # extra holds at an almost coincident proof boundary.
        retreat_width=(642 if near else 1600)*dense
        coordinated_width=relative_width=None
        if not near:
            # Stop the current momentum on a monotone velocity-easing route.
            # The complete nonlinear/reference/force/fuel/geometry proof below
            # decides whether this candidate is admissible at this location.
            velocity=max(np.linalg.norm(value[3:]) for value in points.values())
            duration=exact(3*velocity/60000.)
            window=64*dense;ratio=duration/window
            retreat_width=max(retreat_width,((ratio.numerator+ratio.denominator-1)//ratio.denominator)*window)
            velocities=np.asarray([value[3:] for value in points.values()])
            common_velocity=np.mean(velocities,axis=0)
            relative_velocity=max(np.linalg.norm(value-common_velocity) for value in velocities)
            def rounded_width(value):
                ratio=exact(value)/window
                return ((ratio.numerator+ratio.denominator-1)//ratio.denominator)*window
            # Separate the high shared approach speed from the much smaller
            # formation-closing velocities. These allocation targets select
            # proposals only; the unchanged full command/force proof decides.
            coordinated_width=max(1600*dense,rounded_width(3*np.linalg.norm(common_velocity)/50000.))
            relative_width=min(coordinated_width,max(window,rounded_width(3*relative_velocity/10000.)))
        hold_width=(96 if near else 320)*dense
        profiles=(
            ('coordinated-50k-10k','.15',30.,60000.,Fraction(1)),
            ('coordinated-retimed-9of10','.15',30.,60000.,Fraction(9,10)),
            ('coordinated-retimed-4of5','.15',30.,60000.,Fraction(4,5)),
            ('coordinated-retimed-2of3','.15',30.,60000.,Fraction(2,3)),
            ('original-60k','.15',30.,60000.,None),
            ('medium-60k','.12',24.,60000.,None),
            ('medium-63k','.12',24.,63000.,None))
        prefix=(self.acceleration_contingency
            if allow_acceleration_prefix and not near and projection is not None else None)
        prefix_width=Fraction(0)
        if prefix is not None:
            prefix_width=(prefix.coast_holds+prefix.transition_holds)*dense
            profiles=(('acceleration-prefix','.15',30.,60000.,Fraction(1)),*profiles)
        candidate_widths={}
        if not near:
            for label,_,_,reference_acceleration,retiming in profiles:
                candidate_widths[label]=(max(1600*dense,rounded_width(coordinated_width*retiming))
                    if retiming is not None else max(1600*dense,rounded_width(3*velocity/reference_acceleration)))
                if label=='acceleration-prefix':
                    candidate_widths[label]+=prefix_width
            eligible=[width for width in candidate_widths.values() if q+width+hold_width<=self.clock.q_domain[1]]
            if not eligible:
                raise ValueError('No complete finite retreat and holding candidate fits the declared clock domain')
            forecast_width=max(eligible)
        else:
            near_holds=(self.options.near_refuge_extra_holds if _extended_near else (480,642,800,1024))
            near_retreat_durations=tuple(value*dense for value in near_holds
                if q+value*dense+hold_width<=self.clock.q_domain[1])
            if not near_retreat_durations:
                raise ValueError('No complete finite patch retreat and holding candidate fits the declared clock domain')
            retreat_width=forecast_width=max(near_retreat_durations)
        end=q+forecast_width+hold_width
        if end>self.clock.q_domain[1]:
            raise ValueError('A complete retreat and holding suffix exceed the declared clock domain')
        legacy_widths=[width for label,width in candidate_widths.items()
            if label!='acceleration-prefix' and q+width+hold_width<=self.clock.q_domain[1]]
        legacy_forecast_width=max(legacy_widths) if legacy_widths else forecast_width
        extended_forecast=prefix is not None and forecast_width>legacy_forecast_width
        def forecast(initial,width):
            try:
                return self.target_forecast(initial,width)
            except _ForecastEnclosureRefusal as error:
                if extended_forecast:
                    raise _ExtendedContingencyForecastRefusal(str(error),
                        forecast_width+hold_width,legacy_forecast_width+hold_width) from error
                raise
        target_flow=forecast(estimate,forecast_width+hold_width)
        # A longer ineligible independent route must not prevent construction
        # of a shorter eligible coordinated route. Its placeholder is never
        # selected; the per-candidate clock guard below precedes every proof.
        base_retreat_width=min(retreat_width,forecast_width)
        attitude=pose_target=None
        if pose_conditioned_geometry:
            # A prospective endpoint has no newly observed pose. Continue the
            # genuine current pose/rate inclusion over the entire intervening
            # time, then query its tight quaternion windows at the future q.
            pose_q=exact(sensing_prior.exact_event_q)
            pose_target=(target_flow if pose_q==q else forecast(sensing_prior,end-pose_q))
            attitude=self.attitude_encloser.enclose_observation(sensing_prior.target_orientation_matrix,
                sensing_prior.target_orientation_operator_error,sensing_prior.target_angular_velocity_box,
                pose_target,self.design.geometry.metadata['mass_properties']['inertia_kg_m2'],
                self.model,self.clock,gravity_gradient=True)
            if (attitude.context.get('source_sha256')!=ATTITUDE_SOURCE_SHA256
                    or attitude.context.get('model')!='nonlinear-ephemeris-quaternion-euler-v1'
                    or attitude.context.get('quaternion_order')!='xyzw'
                    or attitude.context.get('plant_fingerprint')!=self.model.fingerprint
                    or attitude.context.get('target_model_fingerprint')!=self.model.target_fingerprint
                    or attitude.context.get('clock_fingerprint')!=self.clock.fingerprint
                    or attitude.context.get('target_flow_fingerprint')!=pose_target.fingerprint):
                raise ValueError('Nonlinear refuge pose source, model, quaternion order or clock changed')
            if not attitude.proven:
                if extended_forecast and attitude.status==EnclosureStatus.UNRESOLVED:
                    raise _ExtendedContingencyForecastRefusal(
                        'Nonlinear refuge pose forecast unresolved: '+attitude.reason,
                        forecast_width+hold_width,legacy_forecast_width+hold_width)
                raise ValueError('Nonlinear refuge pose forecast unresolved: '+attitude.reason)
            pose=AttitudePoseBoundsProvider(FixedProofProvider(attitude),clock=self.clock,
                geometry_id=self.design.geometry.fingerprint,target_model_id=self.model.target_fingerprint,
                frame=estimate.frame_id,clock_id=self.clock.fingerprint)
        else:
            pose=AnyProperTargetPose(geometry_id=self.design.geometry.fingerprint,
                target_model_id=self.model.target_fingerprint,frame=estimate.frame_id,clock_id=self.clock.fingerprint)
        occupied=CompositeOccupiedBoundsProvider(self.design.geometry,pose)
        oracle=FleetGeometryOracle(occupied,component_ids=[p.id for p in self.design.geometry.components],
            body_radii={key:self.design.chaser.body_radius_m for key in points},
            geometry_id=self.design.geometry.fingerprint,target_model_id=self.model.target_fingerprint,
            frame=estimate.frame_id,clock_id=self.clock.fingerprint,
            clearance=self.scenario.config.mission.target_barrier_clearance,
            minimum_pair_center_distance=self.scenario.config.mpc.min_chaser_separation)
        references,contexts,directions={},{},{}
        rotation=np.asarray(sensing_prior.target_orientation_matrix,dtype=float)
        assignments={c.chaser_id:c for c in self.scenario.fleet.chasers('multi')}
        for key,point in points.items():
            direction=(rotation@np.asarray(assignments[key].docking_normal_body,dtype=float) if near else point[:3])
            direction=direction/np.linalg.norm(direction)
            directions[key]=tuple(direction)
            refuge=point[:3]+2*direction
            if not near:refuge=refuge+.25*float(base_retreat_width)*point[3:]
            acceleration=(0.,)*3 if near else tuple(-3*value/float(base_retreat_width) for value in point[3:])
            references[key]=C2RefugeReference.retreat(start=q,retreat_end=q+base_retreat_width,hold_end=end,
                initial_state=tuple(point),refuge_position=tuple(refuge),
                initial_acceleration=acceleration,
                frame=estimate.frame_id,clock_id=self.clock.fingerprint)
            contexts[key]=ControlContext(model_id='nonlinear-reference-relative-v1',nominal_model_id=ZERO_NOMINAL_ID,
                parameter_set_id=self.model.fingerprint,target_model_id=self.model.target_fingerprint,
                frame=estimate.frame_id,clock_id=self.clock.fingerprint,sensor_id=self.sensor_id,
                actuator_id=self.actuator_id,geometry_id=self.design.geometry.fingerprint,
                reference_id=references[key].reference_id,fuel_domain_id=self.propulsion.fingerprint)
        widths=np.max(np.array([radius(box) for box in estimate.relative_state_box_by_agent.values()]),axis=0)
        # A small current navigation box is not a proof that all future
        # sensor/observer errors will stay below the installed far budget.
        # Keep that declared floor for near successors; the final transaction
        # also refuses any reduction of a larger original controller budget.
        floors=np.array((.03,)*3+(6.5,)*3)
        sensing=np.maximum(floors,1.1*widths+1e-8)
        # The far profile uses lambda*Omega == Phi exactly. A binary float
        # 0.15 is slightly smaller and would correctly fail the rational
        # inward inequality, so keep this declared boundary rational.
        position=(exact('.05' if near else '.15'),)*3
        phi=np.full(3,25. if near else 30.)
        gain=500. if near else 200.
        if np.any(sensing[3:]+gain*sensing[:3]>=phi):
            raise ValueError('Navigation uncertainty leaves no refuge sliding margin')
        drift=NonlinearZeroDriftBounds(model=self.model,clock=self.clock,
            target_provider=FixedProofProvider(target_flow),maximum_width='.001',
            extra_acceleration_abs=tuple(max(abs(a),abs(b))
                for a,b in zip(estimate.acceleration_disturbance.lower,estimate.acceleration_disturbance.upper)))
        tracking=dict(omega=position,phi=phi,lambda_gain=(gain,)*3,
            maximum_hold=dense/self.near_feedback_subdivisions if near else dense,
            input_limits=IntervalBox(tuple(float(b['lower']) for b in self.scenario.config.mpc.input_bounds),
                                    tuple(float(b['upper']) for b in self.scenario.config.mpc.input_bounds)),
            sensing_error_abs=tuple(sensing),actuator_error_abs=self.actuator_error_abs,
            command_rounding_abs=(1e-9,)*3,maximum_window=(6 if near else 64)*dense)
        attempts=[];refused_candidates=[];far_profile=near_profile=None
        if near:
            patch=self.design.geometry.attachment_patch(next(iter(assignments.values())).attachment_patch_id)
            u=rotation@np.asarray(patch.tangent_u_body)
            v=rotation@np.asarray(patch.tangent_v_body)
            plan=None
            # The original compact tube can have no sliding inward margin
            # under the installed far sensing floor: Ev+Lambda*Ep=21.5,
            # Phi=25, and held acceleration variation consumes the remainder.
            # Search a second tracking shape as a proposal; its larger box,
            # physical force, entry, fuel and every component/peer sweep still
            # have to pass the same complete independent certificate.
            near_profiles=(
                ('compact-original','.05',25.,500.,10000.,100000.),
                ('compact-high-gain','.05',25.,500.,35000.,120000.),
                ('compact-cancelled-reference','.05',25.,500.,40000.,110000.),
                ('compact-fast-sampling','.05',25.,500.,45000.,100000.),
                ('compact-fast-sampling-high-gain','.05',25.,500.,50000.,110000.),
                ('sensing-compatible','.10',30.,300.,22000.,120000.),
                ('sensing-compatible-high-gain','.10',30.,300.,35000.,120000.))
            near_offsets=(('.5',0),(1,0),(0,0),('-.5',0),(-1,0),
                (0,'.5'),(0,1),(0,'-.5'),(0,-1),
                ('.5','.5'),('.5','-.5'),('-.5','.5'),('-.5','-.5'))
            for label,omega,sliding,near_gain,beta,acceleration in near_profiles:
                candidate_tracking=dict(tracking,omega=(exact(omega),)*3,
                    phi=(sliding,)*3,lambda_gain=(near_gain,)*3)
                search=construct_patch_refuge_candidates(initial_states=points,
                    initial_enclosures=dict(estimate.relative_state_box_by_agent),outward_directions=directions,
                    tangent_u=tuple(u),tangent_v=tuple(v),start=q,contexts=contexts,drift_adapter=drift,
                    geometry_oracle=oracle,propulsion=self.propulsion,holding_duration=hold_width,
                    fuel_budget=fuel_budget,
                    retreat_durations=near_retreat_durations,
                    tangent_offsets=near_offsets,
                    maximum_candidates=len(near_offsets)*len(near_retreat_durations),
                    early_target_refusal=True,early_target_window_budget=4,
                    beta=(beta,)*3,acceleration_bound=(acceleration,)*3,**candidate_tracking)
                attempts.append(dict(profile=label,omega=omega,phi=sliding,
                    lambda_gain=near_gain,beta=beta,acceleration_bound=acceleration,
                    attempts=exact_record(search.attempts)))
                if search.early_refusals:
                    attempts[-1]['early_target_refusals']=exact_record(search.early_refusals)
                if search.admissible:
                    plan=search.plan
                    retreat_width=search.attempts[-1].retreat_duration
                    near_profile=dict(profile=label,omega=omega,phi=sliding,
                        lambda_gain=near_gain,beta=beta,acceleration_bound=acceleration,
                        tangent_offset=exact_record(search.attempts[-1].tangent_offset),
                        maximum_hold_q=str(candidate_tracking['maximum_hold']))
                    break
            if plan is None:
                if not _extended_near and self.options.near_refuge_extra_holds:
                    # Preserve the complete original search and its shorter
                    # pose forecast first. Only an exhausted valid search may
                    # request extra maneuver time; no state or clock advances.
                    prior=exact_record(attempts)
                    try:
                        extended_plan,extended_record,extended_witness=self._build_plan(estimate,
                            sensing_prior=sensing_prior,near_target=near_target,make_witness=make_witness,
                            projection=projection,fuel_budget=fuel_budget,
                            allow_acceleration_prefix=allow_acceleration_prefix,_extended_near=True)
                    except Exception as error:
                        event=dict(kind='ADDITIONAL_NEAR_REFUGE_REFUSED',exact_q=str(q),
                            initial_set_source_id=estimate.source_id,
                            preceding_near_candidate_search=prior,
                            additional_near_retreat_holds=self.options.near_refuge_extra_holds,
                            error_type=type(error).__name__,reason=str(error),
                            causal_event=getattr(error,'event',None))
                        event=exact_record(event)
                        error.event=dict(identity=fingerprint(event),**event)
                        raise
                    extended_record['preceding_near_candidate_search']=prior
                    extended_record['additional_near_retreat_holds']=self.options.near_refuge_extra_holds
                    if extended_witness is not None:
                        extended_witness['preceding_near_candidate_search']=prior
                        extended_witness['additional_near_retreat_holds']=self.options.near_refuge_extra_holds
                    return extended_plan,extended_record,extended_witness
                raise ValueError('All finite patch-refuge candidates refused: '+str(attempts[-1]['attempts'][-1]))
        else:
            # A finite prospective search changes only the declared reference
            # and tracking domain. Each member must independently satisfy all
            # original nonlinear, sensing, input, geometry and fuel guards.
            # A slow common stop can run into a target component even when
            # the faster relative stop protects every peer. Search a finite
            # set of shorter common durations before the independent routes.
            # Retiming changes a proposal, never the installed force limit,
            # uncertainty support, entry domain or geometry requirement.
            plan=None
            tried_coordinated_widths=set()
            for label,omega,sliding,reference_acceleration,retiming in profiles:
                candidate_width=candidate_widths[label]
                if q+candidate_width+hold_width>self.clock.q_domain[1]:
                    attempts.append(dict(profile=label,status='REFUSED',retreat_duration_q=str(candidate_width),
                        holding_duration_q=str(hold_width),reason='Complete candidate exceeds the declared clock domain'))
                    continue
                candidate_relative_width=relative_width
                candidate_references,candidate_contexts=references,contexts
                coordinated=label.startswith('coordinated') or label=='acceleration-prefix'
                braking_width=candidate_width-prefix_width if label=='acceleration-prefix' else candidate_width
                proposal_accelerations=None
                if coordinated:
                    if label!='acceleration-prefix' and candidate_width in tried_coordinated_widths:
                        continue
                    if label!='acceleration-prefix':
                        tried_coordinated_widths.add(candidate_width)
                    candidate_relative_width=min(relative_width,braking_width)
                    center=np.mean([value[:3] for value in points.values()],axis=0)
                    norm=np.linalg.norm(center)
                    outward=center/norm if norm>1e-12 else np.asarray(next(iter(directions.values())))
                    reference_arguments=dict(initial_states=points,start=q,
                        common_duration=braking_width,relative_duration=candidate_relative_width,
                        holding_duration=hold_width,outward_displacement=tuple(2*outward),
                        frame=estimate.frame_id,clock_id=self.clock.fingerprint)
                    if label=='acceleration-prefix':
                        limit=exact(prefix.acceleration_limit_m_per_rad2)
                        proposal_accelerations={key:tuple(max(-limit,min(limit,(lo+hi)/2))
                            for lo,hi in zip(flow.segments[-1].derivative.lower[3:6],
                                             flow.segments[-1].derivative.upper[3:6]))
                            for key,flow in projection.flows}
                        candidate_references=acceleration_contingency_references(**reference_arguments,
                            initial_accelerations=proposal_accelerations,
                            coast_duration=prefix.coast_holds*dense,
                            transition_duration=prefix.transition_holds*dense)
                    else:
                        candidate_references=coordinated_braking_references(**reference_arguments)
                    candidate_contexts={key:replace(contexts[key],reference_id=reference.reference_id)
                        for key,reference in candidate_references.items()}
                elif reference_acceleration!=60000.:
                    candidate_references={};candidate_contexts={}
                    for key,point in points.items():
                        refuge=point[:3]+.25*float(candidate_width)*point[3:]+2*np.asarray(directions[key])
                        reference=C2RefugeReference.retreat(start=q,retreat_end=q+candidate_width,
                            hold_end=q+candidate_width+hold_width,initial_state=tuple(point),
                            refuge_position=tuple(refuge),
                            initial_acceleration=tuple(-3*value/float(candidate_width) for value in point[3:]),
                            frame=estimate.frame_id,clock_id=self.clock.fingerprint)
                        candidate_references[key]=reference
                        candidate_contexts[key]=replace(contexts[key],reference_id=reference.reference_id)
                profile=dict(profile=label,omega=omega,phi=sliding,lambda_gain=gain,
                    beta=22000.,acceleration_bound=120000.,
                    reference_acceleration_target=reference_acceleration,
                    retreat_duration_q=str(candidate_width),holding_duration_q=str(hold_width),
                    reference_ids={key:value.reference_id for key,value in candidate_references.items()})
                if coordinated:
                    profile.update(common_duration_q=str(braking_width),relative_duration_q=str(candidate_relative_width),
                        common_duration_scale=str(retiming),
                        common_acceleration_target=float(50000/retiming),relative_acceleration_target=10000.,
                        reference_policy='shared-translation-with-faster-relative-braking')
                    if label=='acceleration-prefix':
                        profile.update(reference_policy='bounded-acceleration-prefix-then-coordinated-braking',
                            coast_duration_q=str(prefix.coast_holds*dense),
                            transition_duration_q=str(prefix.transition_holds*dense),
                            acceleration_limit_m_per_rad2=prefix.acceleration_limit_m_per_rad2,
                            proposal_accelerations=exact_record(proposal_accelerations),
                            proposal_source='Clipped prospective whole-hold interval derivative midpoint; not a new observation',
                            projection_id=projection.fingerprint)
                candidate=None
                try:
                    candidate_tracking=dict(tracking,omega=(exact(omega),)*3,phi=(sliding,)*3)
                    candidate=construct_refuge_plan(references=candidate_references,contexts=candidate_contexts,
                        drift_adapter=drift,geometry_oracle=oracle,propulsion=self.propulsion,beta=(22000.,)*3,
                        early_target_refusal=True,early_target_window_budget=4,
                        acceleration_bound=(120000.,)*3,**candidate_tracking)
                    if not candidate.certificate.admissible:
                        raise ValueError('; '.join(dict.fromkeys(candidate.certificate.reasons)))
                    if not candidate.contains(estimate.relative_state_box_by_agent,q):
                        raise ValueError('Complete initial state sets are outside this candidate fallback entry')
                    if fuel_budget is not None:
                        fuel_budget.require(candidate.remaining_fuel(q))
                except ValueError as error:
                    attempts.append(dict(profile,status='REFUSED',exception_type=type(error).__name__,
                        reason=str(error),plan_id=None if candidate is None else candidate.certificate.fingerprint))
                    if hasattr(error,'refusal_evidence'):
                        attempts[-1]['early_target_refusal']=exact_record(error.refusal_evidence)
                    if fuel_budget is not None and candidate is not None:
                        attempts[-1].update(required_fuel=exact_record(candidate.remaining_fuel(q)),
                            fuel_budget=fuel_budget.as_dict())
                    if make_witness:
                        refused_candidates.append(dict(attempt=attempts[-1],
                            references={key:dict(reference_id=value.reference_id,frame=value.frame,
                                clock_id=value.clock_id,pieces=exact_record(value.pieces))
                                for key,value in candidate_references.items()},
                            contexts=exact_record(candidate_contexts),
                            plan=None if candidate is None else candidate.certificate.as_dict()))
                    continue
                plan=candidate;far_profile=profile
                references,contexts=candidate_references,candidate_contexts
                retreat_width=candidate_width
                attempts.append(dict(profile,status='CONDITIONAL_PROVEN',plan_id=plan.certificate.fingerprint))
                if fuel_budget is not None:
                    attempts[-1].update(required_fuel=exact_record(plan.remaining_fuel(q)),
                        fuel_budget=fuel_budget.as_dict())
                break
            if plan is None:
                error=ValueError('All finite far-refuge candidates refused: '+str(attempts))
                error.finite_candidate_search=tuple(attempts)
                raise error
        if not plan.certificate.admissible:
            raise ValueError('Constructed fleet refuge refused: '+ '; '.join(plan.certificate.reasons[:5]))
        if not plan.contains(estimate.relative_state_box_by_agent,q):
            raise ValueError('Complete initial state sets are outside the constructed fallback entry')
        if fuel_budget is not None:
            fuel_budget.require(plan.remaining_fuel(q))
        summary=dict(status='CONDITIONAL_PROVEN',identity=plan.certificate.fingerprint,
            start=str(plan.certificate.start),end=str(plan.certificate.end),
            required_fuel=exact_record(plan.certificate.required_fuel),
            controller_ids=plan.certificate.controller_ids,
            geometry_ids=tuple(value.fingerprint for value in plan.certificate.geometry),
            terminal_holding_start=str(plan.certificate.terminal_holding_start),
            terminal_reference_ids=plan.certificate.terminal_reference_ids,method=plan.certificate.method)
        witness=None
        if make_witness:
            certificates={};providers={}
            for segment in plan.segments:
                for key,controller in segment.controllers:
                    proof=controller.certificate
                    certificates[proof.fingerprint]=proof.as_dict()
                    reference=controller.builder.reference_provider
                    providers[reference.reference_id]=dict(reference_id=reference.reference_id,
                        frame=reference.frame,clock_id=reference.clock_id,pieces=exact_record(reference.pieces))
            witness=dict(exact_start=str(q),
                plan=plan.certificate.as_dict(),controllers=certificates,references=providers,
                target=target_flow.as_dict(),attitude=None if attitude is None else attitude.as_dict(),
                attitude_target=None if pose_target is None else pose_target.as_dict(),
                model=self.model.as_dict(),clock=self.clock.as_dict(),geometry=self.design.geometry.as_dict(),
                initial_set=dict(source_kind=estimate.source_kind,source_id=estimate.source_id,
                    exact_q=str(q),relative_state_boxes={key:box.as_dict()
                        for key,box in estimate.relative_state_box_by_agent.items()},
                    target_box=estimate.target_ephemeris_box.as_dict(),
                    physical_time_box=estimate.physical_time_box.as_dict()),
                current_navigation_prior=self._prior_record(sensing_prior),
                endpoint_projection_id=None if projection is None else projection.fingerprint,
                finite_candidate_search=attempts,refused_candidates=refused_candidates)
        record=dict(exact_start=str(q),near_target=near,
            pose_conditioned_geometry=pose_conditioned_geometry,plan=summary,
            feedback_maximum_hold_q=str(tracking['maximum_hold']),
            near_feedback_subdivisions=self.near_feedback_subdivisions,
            retreat_duration_q=str(retreat_width),holding_duration_q=str(hold_width),
            refuge_policy='finite-patch-candidates' if near else 'finite-fleet-braking-candidates',
            far_candidate_profile=far_profile,near_candidate_profile=near_profile,
            target_flow_fingerprint=target_flow.fingerprint,
            attitude_flow_fingerprint=None if attitude is None else attitude.fingerprint,
            observation_packet_ids=list(sensing_prior.packet_ids[-32:]),sensing_error_abs=sensing.tolist(),
            initial_set_source_kind=estimate.source_kind,initial_set_source_id=estimate.source_id,
            finite_candidate_search=attempts,archived_witness_id=None,full_witness_archived=False)
        if fuel_budget is not None:
            record['fuel_budget']=fuel_budget.as_dict()
            if witness is not None:witness['fuel_budget']=fuel_budget.as_dict()
        return plan,record,witness
