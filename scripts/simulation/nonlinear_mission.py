"""Observation-driven nonlinear ADTMPC mission orchestration.

Physical truth, sensors, set navigation, nominal planning, operational fallback
and evidence recording have separate interfaces. Expensive planning/proof work
is an offline simulation; recorded wall time does not claim a flight deadline.
"""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import math
import time
import numpy as np

from orbexa.control.attitude_flow import AttitudeFlowEncloser
from orbexa.control.dynamic_tube_spec import ProspectiveContactConstraint
from orbexa.control.contact_admission import (ContactReferenceKinematics, ContactReachPolicy,
    contact_goal_speed_bound)
from orbexa.control.fleet_fallback import PerformanceAttempt
from orbexa.control.nonlinear_flow import NonlinearFlowEncloser
from orbexa.control.reachability import IntervalBox,ReachabilityConfig
from orbexa.control.refuge import fingerprint as source_fingerprint
from orbexa.core.nonlinear_target import NonlinearTargetPreview,NonlinearTargetTruth
from orbexa.core.reference_frame import exact,fingerprint
from orbexa.estimation.navigation import BoundedNavigationFilter,SensorSuite
from orbexa.estimation.nonlinear_adaptation import NonlinearMissionAdaptation
from orbexa.simulation.coupled_planning import PlanningStep
from orbexa.simulation.coupled_state import CoupledMissionState
from orbexa.simulation.nonlinear_execution import (NonlinearDenseExecutor,AffineActuatorRealization,
    CommandDependentActuationPrior)
from orbexa.simulation.nonlinear_planning import NonlinearMissionPlanner,NonlinearPlanningContext,outward_upper
from orbexa.simulation.nonlinear_profile import (NonlinearStudyOptions,IndependentTruthCase,
    reference_environment,reference_clock,initial_target_state,compose_deployment,controller_prior)
from orbexa.simulation.nonlinear_recording import NonlinearMissionRecorder
from orbexa.simulation.nonlinear_safety import NonlinearSafetyPlanner,radius
from orbexa.simulation.nonlinear_target_bounds import (preview_error_bounds,observed_point_goal_bounds,
    exact_norm_upper)


class NonlinearMissionRunner:
    """Run one prospective case/arm; retain failed solves and actual backup motion."""
    def __init__(self,config,design,*,case:IndependentTruthCase,options=None,
                 adaptation_mode='adaptive',bandwidth_mode='dynamic',progress_sink=None,
                 certificate_sink=None,checkpoint_sink=None,solver_failure_updates=(),
                 reachability_config=None,controller_factory=None,initial_target_spin_prior=None,
                 streamed_hold_index=False,retain_journal=True):
        if adaptation_mode not in ('adaptive','frozen') or bandwidth_mode not in ('dynamic','fixed'):
            raise ValueError('Use the explicit adaptive/frozen and dynamic/fixed factorial arms')
        if not isinstance(case,IndependentTruthCase):
            raise TypeError('Supply an independently selected physical truth/noise case')
        options=options or NonlinearStudyOptions()
        self.options=replace(options,sensor=replace(options.sensor,seed=case.seed))
        self.config=self.options.controller_config(config)
        self.design,self.case=design,case
        self.adaptation_mode,self.bandwidth_mode=adaptation_mode,bandwidth_mode
        self.checkpoint_sink,self.certificate_sink=checkpoint_sink,certificate_sink
        if streamed_hold_index and certificate_sink is None:
            raise ValueError('Lightweight hold indexes require a durable full certificate sink')
        self.solver_failure_updates=frozenset(solver_failure_updates)
        self.clock=reference_clock(self.config,self.options)
        # A prospective domain declaration, never a reachability claim. All
        # executed enclosures must still stay inside the selected domain.
        release_center=((0.,0.,self.options.release_distance_m) if self.options.deployment is None else
            self.options.deployment.carrier_center_lvlh_m)
        release_pitch=(self.options.release_pitch_m if self.options.deployment is None else
            self.config.mpc.min_chaser_separation+2*self.options.deployment.reference_tube_radius_m+
            self.options.deployment.separation_buffer_m)
        self.model=reference_environment(self.config,sigma_target=self.options.target_sigma_prior_m2_kg,
            sigma_chaser=self.options.chaser_sigma_prior_m2_kg,relative_position_limit_m=max(500.,
            float(np.linalg.norm(release_center))+25.+2*np.sqrt(self.config.num_chasers)*release_pitch))
        self.truth_model=replace(self.model,sigma_target=case.target_sigma_m2_kg,sigma_chaser=case.chaser_sigma_m2_kg)
        self.encloser=NonlinearFlowEncloser(reachability_config or ReachabilityConfig(max_step='0.00025'))
        self.attitude_encloser=AttitudeFlowEncloser(ReachabilityConfig(max_step='0.00002',max_substeps=4096,max_picard_iterations=32))
        template=design.target(self.config,initial_body_spin_rad_s=self.options.nominal_spin_vector,
            anomaly_domain=tuple(map(float,self.clock.q_domain)))
        belief=controller_prior(self.config,self.options)
        e0,e1=belief.feasible_sets['eccentricity']
        first,last=initial_target_state(self.config,e0),initial_target_state(self.config,e1)
        target_prior=IntervalBox((min(first[i],last[i]) for i in range(6)),
                                (max(first[i],last[i]) for i in range(6)))
        # A declared initial orbital-plane/phase prior, independent of the
        # chosen case. Delivered own-GNSS/relative packets subsequently narrow it.
        pad=(100.,0.,100.,0.,.1,.1)
        target_prior=IntervalBox(tuple(v-exact(p) for v,p in zip(target_prior.lower,pad)),
                                 tuple(v+exact(p) for v,p in zip(target_prior.upper,pad)))
        self.preview=NonlinearTargetPreview(template,self.clock,self.model,target_prior,preview_width_q='.03')
        self.deployment,self.scenario=compose_deployment(self.config,design,self.preview,self.options)
        self.scenario.adaptation_mode=adaptation_mode
        self.scenario.progress_sink=progress_sink
        if controller_factory is not None:
            if not callable(controller_factory):
                raise TypeError('controller_factory must construct the declared MPC controller')
            self.scenario.controller_factory=controller_factory
        self.chasers=self.scenario.fleet.chasers('multi')
        self.state=CoupledMissionState.initialize(self.chasers,self.config)
        nu=self.clock.point(0).rate_s_inv
        initial_radius=np.r_[np.full(3,self.options.initial_position_radius_m),
                             np.full(3,self.options.initial_velocity_radius_m_s/nu)]
        initial_boxes={c.chaser_id:IntervalBox.from_center_radius(c.initial_state,initial_radius) for c in self.chasers}
        spin_prior=(IntervalBox.from_center_radius(self.options.nominal_spin_vector,(.002,.002,.008))
                    if initial_target_spin_prior is None else initial_target_spin_prior)
        if (not isinstance(spin_prior,IntervalBox) or spin_prior.dimension!=3 or
                not spin_prior.contains(IntervalBox.point(case.spin_vector)) or
                any(max(abs(lo),abs(hi))>self.options.target_rate_bound_rad_s for lo,hi in zip(spin_prior.lower,spin_prior.upper))):
            raise ValueError('Declared initial spin prior must cover the selected case inside the configured rate domain')
        self.navigation=BoundedNavigationFilter(initial_boxes,target_prior,self.clock,self.model,
            initial_target_rotation=self.preview.rotation_matrix_at(0),initial_target_rotation_error=self.options.initial_target_orientation_error,
            target_rate_bound_rad_s=self.options.target_rate_bound_rad_s,target_acceleration_bound_rad_s2=self.options.target_acceleration_bound_rad_s2,
            initial_target_rate_box=spin_prior,
            sensor_fingerprints=(self.options.sensor.fingerprint,),encloser=self.encloser)
        for key in self.state.states:
            self.state.omega[key]=initial_radius[:3].copy()
            self.state.phi[key]=initial_radius[3:]+self.config.tube.sliding_gains*initial_radius[:3]
        realization_rng=np.random.default_rng(case.seed+938921)
        truth_relative={key:value+realization_rng.uniform(-.25,.25,6)*initial_radius
                        for key,value in self.state.states.items()}
        physical_template=replace(template,initial_body_angular_velocity=case.spin_vector)
        self.truth_target=NonlinearTargetTruth(physical_template,self.clock,self.truth_model,
            initial_target_state(self.config,case.initial_eccentricity))
        gain=np.diag(1+realization_rng.uniform(-self.options.actuator_gain_error,self.options.actuator_gain_error,3))
        from scipy.spatial.transform import Rotation
        axis=realization_rng.normal(size=3);axis/=np.linalg.norm(axis)
        tilt=Rotation.from_rotvec(axis*realization_rng.uniform(-self.options.actuator_axis_error_rad,
                                                               self.options.actuator_axis_error_rad)).as_matrix()
        actuator=AffineActuatorRealization(tuple(map(tuple,tilt@gain)),
            tuple(realization_rng.uniform(-self.options.actuator_bias_m_per_rad2,self.options.actuator_bias_m_per_rad2,3)))
        self.adaptation=NonlinearMissionAdaptation(self.model,self.scenario.initial_belief.copy(),self.clock,
            initial_boxes,mode=adaptation_mode,encloser=self.encloser,
            initial_orbit_conditioning=self.options.initial_orbit_conditioning,
            membership_policy=self.options.membership_policy,
            membership_window_holds=self.options.membership_window_holds,
            membership_max_nuisance_cells=self.options.membership_max_nuisance_cells,
            membership_max_flow_evaluations_per_candidate=self.options.membership_max_flow_evaluations_per_candidate)
        self.planner=NonlinearMissionPlanner(self.scenario,self.scenario.create_controller(),encloser=self.encloser,
            planning_workers=self.options.planning_workers,nominal_endpoint_error_abs=np.r_[
                np.full(3,self.options.nominal_position_transfer_budget_m),
                np.full(3,self.options.nominal_velocity_transfer_budget_m_per_rad)],
            stage_peak_padding_abs=np.r_[np.full(3,self.options.sliding_peak_padding_m_per_rad),
                np.full(3,self.options.position_peak_padding_m)])
        self.recorder=NonlinearMissionRecorder(self.scenario,self.chasers,self.state,self.adaptation.belief,
            truth_target=self.truth_target,truth_relatives=truth_relative,model_bounds=self.model,
            truth_model=self.truth_model,clock=self.clock,study_options=self.options,case=case,
            streamed_hold_index=streamed_hold_index)
        self.executor=NonlinearDenseExecutor(self.scenario,navigation=self.navigation,truth_target=self.truth_target,
            truth_relative_by_agent=truth_relative,clock=self.clock,model_bounds=self.model,point_truth=self.truth_model,
            sensor_suite=SensorSuite(self.options.sensor),actuator_realization=actuator,
            actuation_prior=CommandDependentActuationPrior(gain_error=self.options.actuator_gain_error,
                tilt_rad=self.options.actuator_axis_error_rad,bias_abs=self.options.actuator_bias_m_per_rad2,
                command_abs_limit=max(abs(float(bound[key])) for bound in self.config.mpc.input_bounds for key in ('lower','upper'))),
            encloser=self.encloser,certificate_sink=certificate_sink,maximum_blackout_q='.0005',
            planning_model_validator=self._validate_contracted_model,retain_journal=retain_journal,
            completed_hold_sink=self.recorder.record_completed_hold)
        self.safety=NonlinearSafetyPlanner(self.scenario,model=self.model,clock=self.clock,encloser=self.encloser,
            study_options=self.options,design=design,attitude_encloser=self.attitude_encloser,
            near_feedback_subdivisions=self.options.near_feedback_subdivisions,
            acceleration_contingency=self.options.far_contingency)
        self.executor.successor_preparer=self.safety.prepare_successor
        self.recorder.record_navigation(self.navigation.estimate())
        self.arrival_events=[]
        self.first_terminal_entry_q=None
        self.arrival_deadline_missed=False
        self.failure_events=[]
        self.update_wall_timing=[]
        self._attempt_timing=None

    def _finish_attempt_timing(self,status,used_fallback=None):
        """Include refused tails, execution proofs and durable checkpoints."""
        row=self._attempt_timing
        if row is None:return
        now=time.perf_counter();start=row['started'];planned=row.get('planned')
        period=self.clock.elapsed_bounds(row['q'],row['q']+exact(self.config.anom_step))
        lower,upper=float(period.lower[0]),float(period.upper[0])
        full=now-start
        ready=None if planned is None else planned-start
        classification=('MISS' if full>upper else 'WITHIN' if full<=lower else 'UNKNOWN')
        self.update_wall_timing.append(dict(update=row['update'],status=status,
            exact_start_q=str(row['q']),actual_end_exact_q=str(self.truth_target.current_q),
            planning_and_proof_wall_s=ready,
            execution_qualification_wall_s=None if planned is None else now-planned,
            full_attempt_wall_s=full,physical_outer_duration_s=(lower+upper)/2,
            physical_outer_duration_bounds_s=(lower,upper),
            actual_committed_physical_duration_s=float(self.truth_target.physical_time_at(
                self.truth_target.current_q)-row['physical_start']),used_fallback=used_fallback,
            ready_within_physical_outer_period=None if ready is None else ready<=lower,
            deadline_miss_full_update=classification=='MISS',full_update_deadline_status=classification,
            scope='Offline sequential attempted update through durable checkpoint; includes refused physical tails'))
        self._attempt_timing=None

    def _record_failure(self,stage,error):
        """Retain the physical tail even if downstream bookkeeping refuses."""
        self._finish_attempt_timing(stage)
        from orbexa.control.refuge import exact_record
        event=dict(stage=stage,error_type=type(error).__name__,reason=str(error),
            actual_exact_q=str(self.truth_target.current_q),
            actual_truth_relative={key:value.tolist() for key,value in self.executor.truth_relative_by_agent.items()},
            actual_target_state=self.truth_target._current_state.tolist(),
            execution_event=exact_record(getattr(error,'event',None)),
            recorded_outer_endpoint_q=self.recorder.arrays.q_history[-1],
            remaining_safe_q=None if self.executor.runtime is None else
                str(self.executor.runtime.active_fallback.certificate.end-self.executor.runtime.q))
        self.failure_events.append(event);self.recorder.all_certified=False
        return stage+': '+str(error)

    def _validate_contracted_model(self,model,global_model):
        return (global_model.fingerprint==self.model.fingerprint and model.fingerprint==self.adaptation.active_model.fingerprint
                and bool(self.adaptation.candidates))

    def _planning_context(self,estimate,phase):
        cfg=self.config;q=exact(estimate.exact_event_q);count=self.scenario.reference.horizon_steps_for_phase(phase)
        width=(count-1)*exact(cfg.anom_step)
        future=self.safety.target_forecast(estimate,width)
        attitude=self.attitude_encloser.enclose_observation(estimate.target_orientation_matrix,
            estimate.target_orientation_operator_error,estimate.target_angular_velocity_box,future,
            self.design.geometry.metadata['mass_properties']['inertia_kg_m2'],self.model,self.clock,gravity_gradient=True)
        if not attitude.proven:
            raise ValueError('Observed moving-target forecast unresolved: '+attitude.reason)
        acceleration=self.navigation.verify_angular_acceleration_bound(attitude)
        bounds=preview_error_bounds(attitude,self.preview,self.clock,self.chasers,
            [q+i*exact(cfg.anom_step) for i in range(count)],target_flow=future)
        serialized_bounds={k:(str(v) if isinstance(v,Fraction) else v) for k,v in bounds.items()}
        forecast_witness=dict(kind='moving_goal_forecast',exact_q=str(q),
            target_flow=future.as_dict(),attitude_flow=attitude.as_dict(),preview_error=serialized_bounds)
        forecast_witness['identity']=fingerprint(forecast_witness)
        forecast_sink=getattr(self,'certificate_sink',None)
        if forecast_sink is not None:forecast_sink(forecast_witness)
        retained_bounds=serialized_bounds
        if forecast_sink is not None:
            # Full rate/clock/point witnesses were acknowledged by the archive.
            # The retained mission context carries only their exact identity
            # and the bound actually charged to the planning formulation.
            retained_bounds={key:value for key,value in serialized_bounds.items()
                if key not in ('records','proper_rotation_comparison')}
            comparison=serialized_bounds['proper_rotation_comparison']
            retained_bounds['proper_rotation_comparison_id']=None if comparison is None else comparison['identity']
        observed=np.max(np.array([radius(box) for box in estimate.relative_state_box_by_agent.values()]),axis=0)
        floors=np.r_[np.full(3,self.options.observation_position_floor_m),
                      np.full(3,self.options.observation_velocity_floor_m_per_rad)]
        # A prospective measurement-product allowance prevents a temporarily
        # narrow intersection from being mistaken for the support of every
        # future packet. Actual support is still checked on every held action.
        sensor=self.options.sensor
        distance=max(np.linalg.norm(np.abs(np.asarray(estimate.point_relative_by_agent[key])[:3])+radius(box)[:3])
            for key,box in estimate.relative_state_box_by_agent.items())
        distance+=float(width)*max(np.linalg.norm(np.asarray(value)[3:]) for value in estimate.point_relative_by_agent.values())
        distance+=.5*100000.*float(width)**2
        radial=sensor.range_noise_m+sensor.range_bias_m+sensor.range_noise_relative*distance
        product=radial*(1+sensor.bearing_component_noise)+distance*sensor.bearing_component_noise
        age=sensor.max_latency_s+sensor.acquisition_period_s+float(self.config.anom_step)/self.clock.point(q).rate_s_inv
        propagated=age*self.clock.point(q).rate_s_inv*max(observed[3:])
        floors[:3]=np.maximum(floors[:3],1.5*product+propagated)
        observation=np.maximum(floors,1.5*observed+1e-7)
        reach=max(np.linalg.norm(np.array(v,dtype=float)) for c in self.design.geometry.components for v in c.vertices_body_m)
        rotation=outward_upper(bounds['rotation_operator_error_upper'])
        model=self.adaptation.active_model
        residual_target=future
        if model.fingerprint!=self.model.fingerprint:
            residual_target=self.encloser.enclose_target(estimate.target_ephemeris_box,(q,q),(width,width),model,self.clock,
                estimate.physical_time_box)
            if not residual_target.proven:raise ValueError('Adaptive physical forecast unresolved: '+residual_target.reason)
        def navigation_identity(value):
            return source_fingerprint({**vars(value),
                'relative_state_box_by_agent':dict(value.relative_state_box_by_agent)})
        contact_by_agent=None; contact_forecast=None; navigation_id=navigation_identity(estimate)
        if phase.operation=='docking':
            if estimate.status!='VALID':
                raise ValueError('Prospective contact forecast requires the current valid navigation event')
            global_target_id=future.fingerprint;attitude_id=attitude.fingerprint
            source=dict(kind='current-navigation-future-contact-forecast-v1',exact_start_q=str(q),
                exact_step_q=str(exact(cfg.anom_step)),navigation_q=str(q),navigation_fingerprint=navigation_id,
                clock_fingerprint=self.clock.fingerprint,global_model_fingerprint=self.model.fingerprint,
                global_target_flow_fingerprint=global_target_id,attitude_flow_fingerprint=attitude_id,
                moving_goal_forecast_id=forecast_witness['identity'],residual_model_fingerprint=model.fingerprint,
                residual_target_flow_fingerprint=residual_target.fingerprint,frame='reference-orbit-frame-v1')
            contact_by_agent={}
            for chaser in self.chasers:
                key=chaser.chaser_id;nodes=[]
                for index,point_record in enumerate(bounds['records']):
                    node_q=exact(point_record['exact_q'])
                    if node_q!=q+index*exact(cfg.anom_step):
                        raise ValueError('Future contact node differs from the declared exact prediction grid')
                    goal=tuple(map(exact,self.preview.docking_state(chaser.docking_point_body,float(node_q))))
                    goal_id=fingerprint(dict(kind='future-contact-goal-v1',agent_id=key,exact_q=str(node_q),
                        moving_goal_forecast_id=forecast_witness['identity'],point_record_id=fingerprint(point_record),
                        observed_goal=tuple(map(str,goal))))
                    reference=ContactReferenceKinematics(point_record['numerical_preview_rotation'],
                        point_record['numerical_preview_body_rate'],point_record['rotation_operator_error_upper'],
                        goal_id,self.clock.fingerprint,node_q)
                    point=point_record['point_state'];error=point_record['goals'][key]
                    speed=contact_goal_speed_bound(goal,error['position_norm_upper'],error['velocity_norm_upper'],
                        self.clock.interval_bounds((node_q,node_q))['rate_s_inv'],
                        IntervalBox(point['exact_lower'][4:],point['exact_upper'][4:]),reference_kinematics=reference)
                    nodes.append(dict(index=index,exact_q=str(node_q),goal_bound_id=goal_id,
                        goal_speed_bound=speed.as_dict()))
                # These are the unchanged capture/vehicle declarations also
                # used by the independent same-event and lifecycle gates.
                reach_policy=ContactReachPolicy(chaser.docking_standoff,self.design.arm_reach_m,.01)
                contact_by_agent[key]=ProspectiveContactConstraint.from_dict(dict(
                    schema='orbexa.prospective-contact-terminal.v1',source=dict(source,agent_id=key),nodes=nodes,
                    limits=dict(max_spin_error_rad_s=.001,gripper_offset_m=self.design.chaser.gripper_offset_m,
                        max_relative_speed_m_s=.01),reach_policy=reach_policy.as_dict()))
            if (future.fingerprint!=global_target_id or attitude.fingerprint!=attitude_id or
                    navigation_identity(estimate)!=navigation_id):
                raise ValueError('Prospective contact forecast sources changed during construction')
            contact_forecast=dict(kind='prospective_contact_forecast',exact_q=str(q),
                navigation_fingerprint=navigation_id,moving_goal_forecast_id=forecast_witness['identity'],
                constraints={key:value.as_dict() for key,value in contact_by_agent.items()})
            contact_forecast['identity']=fingerprint(contact_forecast)
            if forecast_sink is not None:forecast_sink(contact_forecast)
        context=NonlinearPlanningContext(q,model,self.clock,residual_target,observation,np.full(3,300.),
            reach*rotation,self.bandwidth_mode=='fixed',outward_upper(bounds['position_error_norm_upper']),
            outward_upper(bounds['velocity_error_norm_upper']),contact_terminal_by_agent=contact_by_agent,
            navigation_fingerprint=navigation_id if contact_by_agent is not None else None)
        record=dict(exact_q=str(q),plant_fingerprint=model.fingerprint,global_plant_fingerprint=self.model.fingerprint,
            target_flow_fingerprint=residual_target.fingerprint,attitude_flow_fingerprint=attitude.fingerprint,
            observation_error_abs=observation.tolist(),actuator_error_abs=[300.]*3,
            observation_planning_allowance=dict(predicted_range_m=distance,product_radius_m=product,
                propagated_position_allowance_m=propagated,
                scope='Prospective planning budget; per-hold navigation-support gate remains authoritative'),
            preview_error=retained_bounds,moving_goal_forecast_id=forecast_witness['identity'],
            moving_goal_forecast_archived=forecast_sink is not None,
            **({} if forecast_sink is not None else dict(moving_goal_forecast_witness=forecast_witness)),
            angular_acceleration_check=acceleration)
        if contact_forecast is not None:
            record.update(prospective_contact_forecast_id=contact_forecast['identity'],
                prospective_contact_forecast_archived=forecast_sink is not None,
                contact_terminal_ids={key:value.identity for key,value in contact_by_agent.items()},
                **({} if forecast_sink is not None else dict(prospective_contact_forecast_witness=contact_forecast)))
        return context,record

    def _terminal(self,estimate,phase):
        bounds={};observed_goals={}
        valid=(phase.operation=='docking' and estimate.status=='VALID'
               and self.planner.absolute_deadline_q is not None)
        exact_q=exact(estimate.exact_event_q);q=float(exact_q)
        goal_bounds=observed_point_goal_bounds(estimate,self.preview,self.clock,self.chasers)
        physical_rate_upper=self.clock.interval_bounds((estimate.exact_event_q,estimate.exact_event_q))['rate_s_inv'].upper[0]
        for chaser in self.chasers:
            key=chaser.chaser_id;goal=self.preview.docking_state(chaser.docking_point_body,q)
            observed_goals[key]=list(map(float,goal))
            p=exact_norm_upper(abs(exact(c)-exact(g))+exact(o) for c,g,o in
                zip(self.state.centers[key][:3],goal[:3],self.state.omega[key]))
            v=exact_norm_upper(abs(exact(c)-exact(g))+exact(f)+exact(l)*exact(o) for c,g,f,l,o in
                zip(self.state.centers[key][3:],goal[3:],self.state.phi[key],
                    self.config.tube.sliding_gains,self.state.omega[key]))
            p+=goal_bounds['goals'][key]['position_error_norm_upper']
            v+=goal_bounds['goals'][key]['velocity_error_norm_upper']
            bounds[key]=(outward_upper(p),outward_upper(v))
            saved_p,saved_v=map(exact,bounds[key])
            valid &= saved_p<=exact(self.config.mpc.docking_position_tolerance) and saved_v<=exact(self.config.mpc.docking_velocity_tolerance)
            # Inertial relative velocity error is nu*C*(ev+Z*ep).
            # Keep the physical contact admission stricter than 10 mm/s.
            valid &= physical_rate_upper*(saved_p+saved_v)<=exact('.01')
        deadline=self.planner.absolute_deadline_q
        contact_record=None
        from orbexa.control.refuge import exact_record
        goal_record=exact_record(goal_bounds)
        eligible=phase.operation=='docking' and deadline is not None
        if eligible:
            from orbexa.simulation.nonlinear_completion import terminal_contact_speed_record
            contact_record=terminal_contact_speed_record(navigation=self.recorder.navigation[-1],
                observed_goal_states=observed_goals,goal_bounds=goal_record,
                centers=self.state.centers,phi=self.state.phi,omega=self.state.omega,
                lambda_gain=self.config.tube.sliding_gains,
                docking_points={c.chaser_id:c.docking_point_body for c in self.chasers},
                docking_standoffs={c.chaser_id:c.docking_standoff for c in self.chasers},clock=self.clock,
                expected_model_fingerprint=self.model.fingerprint,vehicle_design=self.design.as_dict(),
                approach_bounds={key:dict(position_m=float(p),velocity_m_per_rad=float(v))
                    for key,(p,v) in bounds.items()})
            valid &= contact_record['admissible']
        if valid and self.first_terminal_entry_q is None:
            self.first_terminal_entry_q=exact_q
            self.arrival_events.append(dict(status='WHOLE_SET_ENTRY',exact_q=str(exact_q),
                by_deadline=deadline is not None and exact_q<=deadline))
        timely_entry=(deadline is not None and self.first_terminal_entry_q is not None
                      and self.first_terminal_entry_q<=deadline)
        if deadline is not None and exact_q>=deadline and (not valid or not timely_entry):
            self.arrival_deadline_missed=True
            self.arrival_events.append(dict(status='DEADLINE_MISSED',exact_q=str(exact_q),
                exact_deadline=str(deadline),whole_set_admissible=bool(valid)))
        valid &= not self.arrival_deadline_missed
        self.recorder.record_terminal(bounds,admissible=valid,phase=phase)
        self.recorder.terminal_records[-1].update(moving_goal_bound=goal_record,
            exact_q=str(exact_q),arrival_deadline=None if deadline is None else str(deadline),
            observed_goal_states=observed_goals if eligible else None,
            contact_speed_admission=contact_record)
        return bool(timely_entry and not self.arrival_deadline_missed
                    and self.recorder.dwell>=self.config.mpc.terminal_dwell_updates)

    def run(self,*,mission='multi',steps=None):
        if mission!='multi':raise ValueError('This profile qualifies the complete eight-agent deployment')
        maximum=self.options.maximum_updates if steps is None else int(steps)
        if maximum<1:raise ValueError('A positive finite update budget is required')
        previous={};phase=self.scenario.reference.initial_phase();messages=[];completed=False;valid=True
        wall_start=time.perf_counter()
        self.recorder.emit_progress('start')
        for update in range(maximum):
            started=time.perf_counter()
            self._attempt_timing=dict(update=update,started=started,q=exact(self.truth_target.current_q),
                physical_start=self.truth_target.physical_time_at(self.truth_target.current_q))
            if self.executor.blackout_holds:
                try:self.executor.recover_navigation()
                except Exception as error:
                    messages.append(self._record_failure('NAVIGATION_RECOVERY_REFUSED',error));valid=False;break
            estimate=self.navigation.estimate()
            self.preview.update_observation(estimate)
            phase=self.scenario.reference.select_phase(phase,self.state,self.chasers)
            if phase.operation=='docking' and self.planner.absolute_deadline_q is None:
                distances=[np.linalg.norm(self.state.centers[c.chaser_id][:3]-
                    self.preview.docking_state(c.docking_point_body,self.state.anomaly)[:3]) for c in self.chasers]
                if max(distances)<=.25:
                    self.planner.absolute_deadline_q=estimate.exact_event_q+(
                        self.scenario.reference.horizon_steps_for_phase(phase)-1)*exact(self.config.anom_step)
                    self.arrival_events.append(dict(status='ANCHORED',update=update,
                        exact_deadline=str(self.planner.absolute_deadline_q),nominal_center_distance_to_preview_m=max(distances)))
            try:
                reused=False
                if self.executor.runtime is not None:
                    try:
                        retained=self.safety.retain_runtime(self.executor.runtime,estimate,
                            width=exact(self.config.anom_step),flow_provider=self.executor.flow_provider,
                            certificate_sink=self.certificate_sink)
                        self.recorder.install_records.append(retained);reused=True
                    except (ValueError,ArithmeticError) as error:
                        self.recorder.install_records.append(dict(exact_start=str(estimate.exact_event_q),
                            status='REUSE_REFUSED',reason=str(error)))
                if not reused:
                    remaining=None if self.executor.runtime is None else self.executor.runtime.remaining_fuel
                    runtime,installed=self.safety.prepare(estimate,flow_provider=self.executor.flow_provider,
                        remaining_fuel=remaining,certificate_sink=self.certificate_sink)
                    self.executor.install_runtime(runtime);self.recorder.install_records.append(installed)
            except (ValueError,ArithmeticError) as error:
                if self.executor.runtime is None:
                    messages.append(self._record_failure('INITIAL_REFUGE_REFUSED',error));valid=False;break
                refusal=dict(exact_start=str(estimate.exact_event_q),status='REPLACEMENT_REFUSED',reason=str(error))
                if getattr(error,'event',None) is not None:
                    from orbexa.control.refuge import exact_record
                    refusal['proof_refusal_event']=exact_record(error.event)
                self.recorder.install_records.append(refusal)
            planning=None;context_record={};failed=None
            try:
                context,context_record=self._planning_context(estimate,phase)
                self.planner.install_context(context)
                if update==0 or update in self.solver_failure_updates:
                    failed=PerformanceAttempt('SENSOR_PRIMING' if update==0 else 'INJECTED_SOLVER_FAILURE',
                        'Initial delayed observations are acquired during certified holding' if update==0 else 'Predeclared solver fault injection',
                        solver_id='gekko-ipopt')
                else:
                    planning=self.planner.plan(self.state,self.adaptation.belief,phase,self.chasers,previous)
                    if not planning.success:
                        failed=PerformanceAttempt('SOLVER_FAILED','; '.join(planning.messages),solver_id='gekko-ipopt')
            except (ValueError,ArithmeticError) as error:
                failed=PerformanceAttempt('PLANNING_REFUSED',str(error),solver_id='gekko-ipopt')
            if planning is None:
                planning=PlanningStep(matrices=(),parameters={},success=False,messages=[] if failed is None else [failed.message])
            self.recorder.record_planning(planning,self.state,phase,context_record)
            planned=time.perf_counter()
            self._attempt_timing['planned']=planned
            try:
                outcome=self.executor.advance(self.state,planning,phase,planner=self.planner,failed_attempt=failed)
            except Exception as error:
                messages.append(self._record_failure('EXECUTION_REFUSED',error));valid=False;break
            if not outcome.committed:
                self._finish_attempt_timing('EXECUTION_UNCOMMITTED',outcome.used_fallback)
                valid=False;messages.extend(outcome.messages);break
            self.recorder.record_execution(self.state,outcome,planning,phase,
                truth_relatives=outcome.actual_truth_relative,hold_records=outcome.journal)
            accepted=self.navigation.estimate()
            self.adaptation.consume_delivered(outcome.delivered_packets,accepted_packet_ids=accepted.packet_ids,
                sensor_fingerprints=accepted.sensor_fingerprints)
            self.adaptation.retain_transitions(tuple(item for _,item in outcome.membership_transitions))
            if update>0 and update%20==0:
                # All agents contribute; no failed NLP is a negative truth label.
                consistent=self.adaptation.contract(update=update)
                valid &= consistent
            self.recorder.arrays.record_belief(self.adaptation.belief)
            estimate=self.navigation.estimate()
            self.recorder.record_navigation(estimate)
            self.preview.update_observation(estimate)
            completed=self._terminal(estimate,phase)
            valid &= outcome.success
            if self.arrival_deadline_missed:
                failure=dict(stage='ARRIVAL_DEADLINE_MISSED',actual_exact_q=str(estimate.exact_event_q),
                    exact_deadline=str(self.planner.absolute_deadline_q),
                    reason='Whole-set terminal entry or continuous endpoint dwell missed the fixed arrival requirement')
                self.failure_events.append(failure);messages.append(failure['stage']+': '+failure['reason'])
                valid=False
            # A shifted trajectory is only a fresh NLP initial guess. A
            # fallback does not authorize replay of an old command, but need
            # not discard useful optimizer initialization.
            previous=planning.plans if planning.success else {}
            self.recorder.emit_progress('committed')
            if self.checkpoint_sink is not None:self.checkpoint_sink(update,self,outcome)
            self._finish_attempt_timing('COMMITTED',outcome.used_fallback)
            if completed or not valid:break
        if not completed:messages.append('nonlinear_approach_not_complete_within_declared_update_budget')
        result=self.recorder.result(valid=valid,completed=completed,messages=messages,adaptation=self.adaptation)
        result.metadata.update(adaptation_mode=self.adaptation_mode,bandwidth_mode=self.bandwidth_mode,
            execution_wall_s=time.perf_counter()-wall_start,update_wall_timing=self.update_wall_timing,
            observed_deadline_misses=sum(v['deadline_miss_full_update'] for v in self.update_wall_timing),
            observed_deadline_unknown_updates=sum(v['full_update_deadline_status']=='UNKNOWN' for v in self.update_wall_timing),
            planner_ready_deadline_misses=sum(v['ready_within_physical_outer_period'] is False for v in self.update_wall_timing),
            attempted_updates=len(self.update_wall_timing),
            planning_schedule=dict(method=('Jacobi-prior-fleet-preview-with-joint-admission' if
                self.planner.planning_workers>1 else 'serial-Gauss-Seidel-with-joint-admission'),
                native_gekko_workers=self.planner.planning_workers),
            timing_scope='Offline sequential qualification through execution proofs and durable checkpoint, including refused attempts; no flight deadline claim',
            arrival_state=dict(method='fixed-deadline-whole-set-entry-v1',
                exact_deadline=None if self.planner.absolute_deadline_q is None else str(self.planner.absolute_deadline_q),
                first_admitted_exact_q=None if self.first_terminal_entry_q is None else str(self.first_terminal_entry_q),
                entry_by_deadline=bool(self.planner.absolute_deadline_q is not None and self.first_terminal_entry_q is not None
                    and self.first_terminal_entry_q<=self.planner.absolute_deadline_q),
                deadline_missed=self.arrival_deadline_missed,terminal_dwell_updates=self.config.mpc.terminal_dwell_updates,
                final_dwell=self.recorder.dwell,events=self.arrival_events,deadline_never_reset=True),
            failure_events=self.failure_events,actual_physical_end_exact_q=str(self.truth_target.current_q),
            injected_solver_failure_updates=sorted(self.solver_failure_updates))
        self.recorder.emit_progress('complete',result)
        self.result=result
        return result


def cfg_array(value):
    return np.asarray(value,dtype=float)
