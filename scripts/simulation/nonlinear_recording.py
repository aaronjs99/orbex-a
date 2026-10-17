"""Separate physical truth, observed navigation and controller mission records."""
from __future__ import annotations

from types import SimpleNamespace
import copy
import numpy as np

from orbexa.control.sample_hold import SENSED_TUBE_MODEL_VERSION
from orbexa.control.refuge import fingerprint
from orbexa.simulation.mission_recording import MissionRecorder
from orbexa.simulation.mission_result import ADTMPCMissionResult
from orbexa.simulation.coupled_state import CoupledMissionState
from orbexa.core.reference_frame import exact


class NonlinearMissionRecorder:
    """Reuse array formatting only; model/clock/safety claims are explicit here."""
    def __init__(self,scenario,chasers,state,belief,*,truth_target,truth_relatives,
                 model_bounds,truth_model,clock,study_options,case,streamed_hold_index=False):
        self.scenario,self.chasers=scenario,chasers
        self.truth_target,self.model_bounds,self.truth_model=truth_target,model_bounds,truth_model
        self.clock,self.study_options,self.case=clock,study_options,case
        if not isinstance(streamed_hold_index,bool):raise TypeError('streamed_hold_index must be boolean')
        self.streamed_hold_index=streamed_hold_index
        # This view is available only to the recorder. It is never passed to a
        # planner, navigator, feedback controller or admission provider.
        view=copy.copy(scenario); view.target=truth_target
        shadow=copy.copy(state); shadow.states={k:v.copy() for k,v in truth_relatives.items()}
        self.arrays=MissionRecorder(view,chasers,shadow,belief)
        self.navigation,self.hold_records,self.install_records=[],[],[]
        self.terminal_records,self.planning_contexts=[],[]
        self.all_certified=True
        self.dwell=0
        self._last_packet_count=0

    def record_navigation(self,estimate):
        self.navigation.append(dict(exact_q=str(estimate.exact_event_q),status=estimate.status,reason=estimate.reason,
            physical_time_box=estimate.physical_time_box.as_dict(),
            relative_state_boxes={k:v.as_dict() for k,v in estimate.relative_state_box_by_agent.items()},
            point_relative_by_agent={k:list(v) for k,v in estimate.point_relative_by_agent.items()},
            target_ephemeris_box=estimate.target_ephemeris_box.as_dict(),
            target_orientation_matrix=[list(map(float,row)) for row in estimate.target_orientation_matrix],
            target_orientation_operator_error=float(estimate.target_orientation_operator_error),
            target_angular_velocity_box=estimate.target_angular_velocity_box.as_dict(),
            packet_ids=list(estimate.packet_ids[self._last_packet_count:]),
            accepted_packet_count=len(estimate.packet_ids),sensor_fingerprints=list(estimate.sensor_fingerprints),
            model_fingerprint=estimate.model_fingerprint,clock_fingerprint=estimate.clock_fingerprint,
            frame=estimate.frame_id))
        self._last_packet_count=len(estimate.packet_ids)

    def record_planning(self,planning,state,phase,context_records):
        self.arrays.phases.append(phase.name)
        self.arrays.record_planning(planning,state.anomaly)
        self.planning_contexts.append(copy.deepcopy(context_records))

    def record_execution(self,state,outcome,planning,phase,*,truth_relatives,hold_records=()):
        if not outcome.committed:
            return
        a=self.arrays
        a.contained &= bool(outcome.sampled_tube_contained)
        a.dense_min_containment=min(a.dense_min_containment,min(outcome.containment.values()))
        a.dense_min_clearance=min(a.dense_min_clearance,min(outcome.clearance.values()))
        a.dense_min_spacing=min(a.dense_min_spacing,outcome.min_spacing)
        a.max_nominal_discretization_residual=max(a.max_nominal_discretization_residual,
            outcome.max_nominal_discretization_residual)
        # Refuge segment boundaries can split a dense feedback interval. Use
        # the exact executed widths, rather than assuming 32 equal samples.
        widths=[exact(record['exact_hold_width']) for record in hold_records]
        if not widths or sum(widths)!=exact(outcome.end_anomaly)-exact(outcome.start_anomaly):
            # The public outer endpoints are binary64 projections; compare
            # their exact journal endpoints when projection is inexact.
            if not widths or sum(widths)!=exact(hold_records[-1]['exact_end'])-exact(hold_records[0]['exact_start']):
                raise ValueError('Executed hold widths do not cover the recorded outer interval')
        weights=np.asarray([float(width) for width in widths]);duration=float(sum(widths))
        for key in a.ids:
            actual=np.asarray(truth_relatives[key],dtype=float)
            a.actual[key].append(actual.tolist())
            applied=np.asarray(outcome.applied[key],dtype=float)
            ancillary=np.asarray(outcome.ancillary[key],dtype=float)
            if len(applied)!=len(weights) or len(ancillary)!=len(weights):
                raise ValueError('Applied-control samples differ from the exact hold journal')
            a.controls[key].append((weights@applied/duration).tolist())
            data=a.diagnostics[key]
            a.record_tube_sample(data,actual,state.centers[key],state.phi[key],state.omega[key],state.bandwidth[key])
            nominal=(np.asarray(planning.plans[key].result.control_trajectory)[:,0]
                     if key in planning.plans else np.zeros(3))
            data['nominal_control'].append(nominal.tolist())
            data['max_dense_input_reserve_ratio'].append(float(outcome.reserve_ratios.get(key,0.)))
            data['ancillary_control_rms'].append(np.sqrt(weights@(ancillary**2)/duration).tolist())
            data['applied_control_rms'].append(np.sqrt(weights@(applied**2)/duration).tolist())
            data['applied_control_effort'].append(float(weights@np.sum(applied**2,axis=1)))
            data['dense_containment_margin'].append(float(outcome.containment[key]))
            data['dense_physical_clearance'].append(float(outcome.clearance[key]))
            data['nominal_center_defect'].append(outcome.nominal_center_defect.get(key,np.zeros(6)).tolist())
        q=state.anomaly
        a.q_history.append(float(q));a.sample_phases.append(phase.name)
        a.elapsed.append(float(self.truth_target.physical_time_at(q)))
        a.attitudes.append(self.truth_target.orientation_at(q).tolist())
        a.radius_history.append(max(float(np.linalg.norm(v)) for v in state.omega.values()))
        for record in hold_records:
            decision=record['decision']
            self.all_certified &= (record['runtime_commit']['status'] in ('COMMITTED','FINITE_SAFE_WINDOW_ENDED')
                and record['navigation_status'] in ('VALID','BLACKOUT_CONDITIONAL')
                and decision['source'] in ('PERFORMANCE','FALLBACK_TRACK','FALLBACK_HOLD')
                and bool(decision.get('geometry')) and not decision['geometry']['reasons']
                and exact(record['exact_end'])-exact(record['exact_start'])==exact(decision['hold_width']))
            if self.streamed_hold_index:
                from orbexa.simulation.nonlinear_archive import hold_index
                self.hold_records.append(hold_index(record))
                continue
            # Complete witnesses go to the injected proof archive. The mission
            # record retains their identities and executable/resource lineage.
            compact={key:copy.deepcopy(value) for key,value in record.items()
                     if key not in ('decision','runtime_commit')}
            compact['decision']={key:copy.deepcopy(value) for key,value in decision.items() if key!='geometry'}
            geometry=decision.get('geometry')
            compact['geometry_id']=None if geometry is None else fingerprint(geometry)
            commit=record['runtime_commit']
            compact['runtime_commit_status']=commit['status']
            compact['remaining_fuel']=copy.deepcopy(commit['remaining_fuel'])
            self.hold_records.append(compact)

    def record_terminal(self,bounds,*,admissible,phase):
        for key,(p,v) in bounds.items():
            self.arrays.diagnostics[key]['terminal_position_bound'].append(float(p))
            self.arrays.diagnostics[key]['terminal_velocity_bound'].append(float(v))
        self.dwell=self.dwell+1 if admissible and phase.operation=='docking' else 0
        self.terminal_records.append(dict(anomaly=self.arrays.q_history[-1],
            bounds={k:dict(position_m=float(v[0]),velocity_m_per_rad=float(v[1])) for k,v in bounds.items()},
            admitted=bool(admissible),dwell=self.dwell))

    def emit_progress(self,kind,result=None):
        sink=self.scenario.progress_sink
        if sink is None:
            return
        from orbexa.visualization.studio_data import StudioDataset
        record=result if result is not None else self._record(False,False,[],None)
        projection=StudioDataset(record).to_dict()
        if result is None:
            projection['outcome']['label']='Computing nonlinear mission'
        sink(dict(kind=kind,update=len(self.arrays.q_history)-1,anomaly=self.arrays.q_history[-1],projection=projection))

    def _record(self,valid,completed,messages,adaptation):
        a,cfg=self.arrays,self.scenario.config
        # Offline truth geometry is a diagnostic. Execution proof identities
        # are retained independently for every shared hold.
        safety=copy.copy(self.scenario.safety);safety.target=self.truth_target
        pr,pd,tr,td,active=safety.safety_margins(a.actual,a.q_history,a.sample_phases,a.radius_history)
        spacing=safety.spacing_history(a.actual,a.q_history)
        target_safe=safety.active_target_margins_safe(active) and a.dense_min_clearance>=0
        pair_safe=safety.pairwise_spacing_safe(spacing) and a.dense_min_spacing>=safety.min_chaser_separation
        kinematics=self.truth_target.recorded_kinematics(a.q_history)
        belief=a.initial_belief if adaptation is None else adaptation.belief
        metadata=dict(model_version='nonlinear-reference-relative-v1',controller='coupled-dynamic-tube-gekko',
            state_frame='reference-orbit-frame-v1',reference_clock=self.clock.as_dict(),
            reference_clock_fingerprint=self.clock.fingerprint,
            nonlinear_physical_bounds=self.model_bounds.as_dict(),nonlinear_truth=self.truth_model.as_dict(),
            numerical_truth_integration=dict(method='DOP853',relative_rtol=1e-11,relative_atol=1e-10,
                relative_max_step='min(exact physical hold q width,0.0001)',
                scope='Numerical truth path; continuous admission remains independently enclosed with Arb'),
            study_options=self.study_options.as_dict(),study_fingerprint=self.study_options.fingerprint,
            independent_case=vars(self.case),
            scenario=self.scenario.scenario_metadata(),target_kinematics=kinematics,
            navigation_history=self.navigation,planning_contexts=self.planning_contexts,
            fleet_hold_records=self.hold_records,fallback_installations=self.install_records,
            continuous_execution_certificate=dict(method='nonlinear-sensed-fleet-admission-v1',
                body_radius_m=safety.chaser_body_radius,start_anomaly=a.q_history[0],
                exact_end_anomaly=(self.navigation[-1]['exact_q'] if self.navigation else str(a.q_history[-1])),
                admitted_fleet_holds=len(self.hold_records),all_intervals_certified=self.all_certified,
                input_lower=[b['lower'] for b in cfg.mpc.input_bounds],input_upper=[b['upper'] for b in cfg.mpc.input_bounds],
                scope='Arb nonlinear physical flows and whole-component fleet sweeps for performance; '
                      'stored finite feedback invariant certificates for fallback; declared sensor and actuation supports'),
            tube_model_version=SENSED_TUBE_MODEL_VERSION,
            lambda_gain=np.asarray(cfg.tube.sliding_gains).tolist(),
            horizon_steps=self.scenario.reference.configured_horizon_metadata(),
            feedback_substeps=cfg.tube.feedback_substeps,anom_step=cfg.anom_step,
            dense_anom_step=cfg.anom_step/cfg.tube.feedback_substeps,
            docking_position_tolerance=cfg.mpc.docking_position_tolerance,
            docking_velocity_tolerance=cfg.mpc.docking_velocity_tolerance,
            terminal_dwell_updates=cfg.mpc.terminal_dwell_updates,terminal_dwell_observed=self.dwell,
            terminal_bound_records=self.terminal_records,mission_complete=bool(completed),
            active_target_margins_safe=bool(target_safe),pairwise_spacing_safe=bool(pair_safe),
            min_chaser_separation=safety.min_chaser_separation,sampled_tube_containment=bool(a.contained),
            dense_min_containment_margin=None if not np.isfinite(a.dense_min_containment) else float(a.dense_min_containment),
            dense_min_physical_clearance=None if not np.isfinite(a.dense_min_clearance) else float(a.dense_min_clearance),
            dense_min_spacing=None if not np.isfinite(a.dense_min_spacing) else float(a.dense_min_spacing),
            max_nominal_discretization_residual=a.max_nominal_discretization_residual,
            elapsed_time_seconds=a.elapsed,chaser_assignments=[c.assignment_metadata() for c in self.chasers],
            membership_records=[] if adaptation is None else adaptation.membership_records,
            physical_adaptation={} if adaptation is None else adaptation.metadata(),
            input_bounds=cfg.mpc.input_bounds,
            units=dict(position='m',velocity='m/rad scheduled reference anomaly',input='m/rad^2 scheduled reference anomaly',
                physical_acceleration='nu_ref(q)^2*C_ref(q)*u',physical_time='s'),
            guarantee_scope='Conditional simulated-domain evidence: observation-driven ADTMPC, declared central/J2/vector-drag '
                'plant and actuation supports, continuous physical swept admission and finite operational refuge policies. '
                'Whole-tube endpoint acceptance includes moving-goal preview error; flight, raw-image perception, '
                'global NLP feasibility and arbitrary-duration fuel-backed safety are not established.',
            controls_semantics='Recorded applied outer means; exact commands, supports and source retained per hold',
            nominal_center_defect_semantics='Validated nominal-flow endpoint radius transferred into Phi/Omega at every hold',
            terminal_dwell_semantics='Consecutive observed whole-tube endpoint bounds including target preview error; no success from one center sample')
        truth=self.scenario.truth.as_dict()
        coefficient=float(self.truth_model.atmosphere.rho0_kg_m3[0]*self.truth_model.atmosphere.reference_radius_m/2)
        truth.update(eccentricity=self.case.initial_eccentricity,
            alpha=float(self.truth_model.sigma_target[0])*coefficient,beta=float(self.truth_model.sigma_chaser[0])*coefficient)
        metadata['truth_parameter_semantics']=('Initial osculating eccentricity and reference-density drag descriptors; '
            'the actual propagated plant is nonlinear_physical_bounds/nonlinear_truth with the retained Cartesian ephemeris')
        return ADTMPCMissionResult(mission='multi',approximation='nonlinear',solver_backend='gekko',
            success=bool(valid and completed and target_safe and pair_safe and a.contained and self.all_certified),
            message='; '.join(messages),anom_history=a.q_history,phase_history=a.phases,sample_phase_history=a.sample_phases,
            actual_trajectories=a.actual,nominal_trajectories=a.nominal,controls=a.controls,solve_stats=a.stats,solver_costs=a.costs,
            truth=truth,target=self.truth_target.as_dict(),target_attitude_history=a.attitudes,
            target_angular_velocity_history=np.asarray(kinematics['body_angular_velocity']).tolist(),
            docking_points={c.chaser_id:c.docking_point_body.tolist() for c in self.chasers},
            initial_belief=a.initial_belief.as_metadata(),final_belief=belief.as_metadata(),
            feasible_set_history=a.feasible,parameter_estimate_history=a.estimates,
            smid_records=[] if adaptation is None else adaptation.smid_records,tube_profiles=a.profiles,
            tube_radius_history=a.radius_history,rendezvous_margins=pr,docking_cylinder_margins=pd,
            active_target_margins=active,pairwise_spacing=spacing,physical_rendezvous_margins=pr,
            physical_docking_cylinder_margins=pd,tightened_rendezvous_margins=tr,tightened_docking_cylinder_margins=td,
            metadata=metadata,tube_diagnostics=a.diagnostics)

    def result(self,*,valid,completed,messages,adaptation):
        return self._record(valid,completed,messages,adaptation)
