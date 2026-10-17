"""GEKKO ADTMPC adapter with explicit nonlinear regional residuals and sensing.

The optimizer remains the maintained multiple-shooting implementation. A
regional certificate is an input to that formulation, never a sampled residual
maximum or a substitute for admission of the actual executed nonlinear hold.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor
from types import MappingProxyType
from typing import Mapping
import numpy as np

from orbexa.control.dynamic_tube_spec import DynamicTubeSpec, ProspectiveContactConstraint
from orbexa.control.reachability import IntervalBox
from orbexa.control.nonlinear_flow import bound_nominal_coefficients
from orbexa.core.reference_frame import exact
from orbexa.simulation.coupled_planning import CoupledMissionPlanner,PlanningStep
from orbexa.contracts import SolverResult


def outward_upper(value):
    result = float(value)
    if Fraction(result) < value:
        result = np.nextafter(result, np.inf)
    return result


@dataclass(frozen=True)
class NonlinearPlanningContext:
    """Immutable observation/model snapshot used by one fleet solve."""
    exact_q: Fraction
    model: object
    clock: object
    target_flow: object
    observation_error_abs: np.ndarray
    acceleration_error_abs: np.ndarray
    target_position_preview_error_m: float
    frozen_bandwidth: bool = False
    goal_position_error_m: float = 0.
    goal_velocity_error_m_per_rad: float = 0.
    contact_terminal_by_agent: Mapping[str, ProspectiveContactConstraint] | None = None
    navigation_fingerprint: str | None = None

    def __post_init__(self):
        object.__setattr__(self, 'exact_q', exact(self.exact_q))
        for name, size in (('observation_error_abs',6),('acceleration_error_abs',3)):
            value = np.asarray(getattr(self,name), dtype=float)
            if value.shape != (size,) or np.any(value < 0) or np.any(~np.isfinite(value)):
                raise ValueError('Planning uncertainty must be finite and nonnegative')
            value = value.copy(); value.setflags(write=False)
            object.__setattr__(self,name,value)
        if (not self.target_flow.proven or self.target_position_preview_error_m < 0
                or not np.isfinite(self.target_position_preview_error_m)):
            raise ValueError('A bounded physical target preview is required')
        if (self.target_flow.start_anomaly!=(self.exact_q,self.exact_q)
                or self.target_flow.model_bounds.get('clock_fingerprint')!=self.clock.fingerprint
                or self.target_flow.model_bounds.get('target_model_fingerprint')!=self.model.target_fingerprint):
            raise ValueError('Planning ephemeris differs from the declared event/model/clock')
        if any(not np.isfinite(value) or value<0 for value in
                (self.goal_position_error_m,self.goal_velocity_error_m_per_rad)):
            raise ValueError('Moving-goal error bounds must be finite and nonnegative')
        if self.contact_terminal_by_agent is not None:
            if not isinstance(self.contact_terminal_by_agent, Mapping) or not self.contact_terminal_by_agent:
                raise ValueError('Prospective contact constraints require an explicit nonempty agent mapping')
            if not isinstance(self.navigation_fingerprint, str) or not self.navigation_fingerprint.strip():
                raise ValueError('Prospective contact constraints require the current navigation fingerprint')
            sealed = {}
            for agent_id, value in self.contact_terminal_by_agent.items():
                value = (ProspectiveContactConstraint.from_dict(value)
                         if isinstance(value, Mapping) else value)
                if not isinstance(value, ProspectiveContactConstraint):
                    raise ValueError('Use typed source-bound prospective contact constraints')
                if value.reach_policy() is None:
                    raise ValueError('Nonlinear contact forecasts require the declared capture reach/gap policy')
                record = value.as_dict(); source = record['source']
                if (source['agent_id'] != agent_id or source['navigation_fingerprint'] != self.navigation_fingerprint or
                        exact(source['exact_start_q']) != self.exact_q or
                        source['clock_fingerprint'] != self.clock.fingerprint or
                        source['residual_model_fingerprint'] != self.model.fingerprint or
                        source['residual_target_flow_fingerprint'] != self.target_flow.fingerprint):
                    raise ValueError('Prospective contact source differs from the active navigation/residual context')
                for node, bound in zip(record['nodes'], value.node_bounds()):
                    point_clock = self.clock.interval_bounds((exact(node['exact_q']), exact(node['exact_q'])))['rate_s_inv']
                    if (bound.clock_rate_s_inv.lower != point_clock.lower or
                            bound.clock_rate_s_inv.upper != point_clock.upper):
                        raise ValueError('Prospective contact node clock differs from the exact declared clock')
                sealed[agent_id] = value
            object.__setattr__(self, 'contact_terminal_by_agent', MappingProxyType(sealed))


class NonlinearMissionPlanner(CoupledMissionPlanner):
    """Bind every nominal/tube problem to its independently enclosed region."""
    def __init__(self, scenario, controller, *, encloser, planning_workers=8,nominal_endpoint_error_abs=None,
                 stage_peak_padding_abs=None):
        super().__init__(scenario,controller)
        if isinstance(planning_workers,bool) or not isinstance(planning_workers,int) or not 1<=planning_workers<=8:
            raise ValueError('Declare one to eight independent native GEKKO planning workers')
        self.planning_workers=planning_workers
        epsilon=np.zeros(6) if nominal_endpoint_error_abs is None else np.asarray(nominal_endpoint_error_abs,dtype=float)
        if epsilon.shape!=(6,) or not np.all(np.isfinite(epsilon)) or np.any(epsilon<0):
            raise ValueError('Nominal endpoint-transfer budget must be a nonnegative finite six-vector')
        self.nominal_endpoint_error_abs=epsilon.copy();self.nominal_endpoint_error_abs.setflags(write=False)
        padding=np.zeros(6) if stage_peak_padding_abs is None else np.asarray(stage_peak_padding_abs,dtype=float)
        if padding.shape!=(6,) or not np.all(np.isfinite(padding)) or np.any(padding<0):
            raise ValueError('Prospective stage-peak padding must be a nonnegative finite six-vector')
        self.stage_peak_padding_abs=padding.copy();self.stage_peak_padding_abs.setflags(write=False)
        self.encloser, self.context = encloser, None
        self.regions, self.residuals = {}, {}
        self.absolute_deadline_q = None
        self.operation = None

    def plan(self,state,belief,phase,chasers,previous_plans,**kwargs):
        self.operation=phase.operation
        if self.planning_workers==1 or len(chasers)<=1:
            return super().plan(state,belief,phase,chasers,previous_plans,**kwargs)
        if kwargs:
            raise ValueError('Parallel nonlinear planning uses its explicitly anchored deadline')
        # Jacobi candidates share only the preceding accepted planning snapshot.
        # All pair constraints remain in each NLP. A simultaneous tube check
        # and independent nonlinear fleet admission gate the whole candidate.
        def solve(chaser):
            worker=NonlinearMissionPlanner(self.scenario,self.scenario.create_controller(),
                encloser=self.encloser,planning_workers=1,nominal_endpoint_error_abs=self.nominal_endpoint_error_abs,
                stage_peak_padding_abs=self.stage_peak_padding_abs)
            worker.install_context(self.context);worker.absolute_deadline_q=self.absolute_deadline_q
            try:
                result=worker.plan(state,belief,phase,[chaser],previous_plans)
            except (ValueError,ArithmeticError,TimeoutError) as error:
                message=f'{chaser.chaser_id}: nonlinear planning setup refused: {error}'
                result=PlanningStep(matrices=(),parameters={},success=False,messages=[message],
                    attempts={chaser.chaser_id:SolverResult(success=False,message=message,
                        solver_info=dict(failure_stage='nonlinear_setup',exception_type=type(error).__name__))})
            return result,worker.regions,worker.residuals
        with ThreadPoolExecutor(max_workers=min(self.planning_workers,len(chasers))) as pool:
            results=list(pool.map(solve,chasers))
        first=next((value[0] for value in results if value[0].matrices),results[0][0])
        outcome=PlanningStep(matrices=first.matrices,parameters=first.parameters)
        for result,regions,residuals in results:
            outcome.plans.update(result.plans);outcome.attempts.update(result.attempts)
            outcome.messages.extend(result.messages);outcome.success &= result.success
            self.regions.update(regions);self.residuals.update(residuals)
        if outcome.success:
            outcome.messages.extend(self.scenario.safety.planned_separation_messages(outcome.plans))
            outcome.success=not outcome.messages
        return outcome

    def planning_peer_ids(self,chasers,state):
        return list(state.centers)

    def tube_profile_assumptions(self,spec):
        return (
            'Shared exact scheduled-reference clock and acceleration-input mapping; the nonlinear plant, sensing and actuation stay inside their declared supports',
            'Arb regional nonlinear-minus-nominal residual and coefficient/variation bounds cover the entire executed hold',
            'Strict stage peaks and sensed held-feedback comparison; actual nominal endpoint defects and executable command evaluation errors require their independent execution gates',
            'Whole-component and peer sweeps, admitted finite successor and remaining fuel require separate fleet admission; numerical optimization alone grants no physical motion',
        )

    def install_context(self, context):
        if not isinstance(context, NonlinearPlanningContext):
            raise TypeError('Install an explicit nonlinear planning context')
        self.context, self.regions, self.residuals = context, {}, {}

    def tube_spec(self, belief, params, anomaly, count, phi, omega, bandwidth, *, chaser_id=None, center=None):
        ctx, cfg, tube = self.context, self.config, self.config.tube
        if ctx is None or center is None or not chaser_id:
            raise ValueError('Nonlinear planning requires an observed state/context')
        h, q0 = exact(cfg.anom_step), ctx.exact_q
        center = np.asarray(center, dtype=float)
        duration = float((count-1)*h)
        limit = max(max(abs(float(b['lower'])), abs(float(b['upper']))) for b in cfg.mpc.input_bounds)
        # This is a chosen local domain, not a reachability claim. The NLP is
        # restricted to it, and actual swept admission checks it independently.
        pwidth = np.abs(center[3:])*duration + .65*limit*duration**2 + np.maximum(omega, .05)
        vwidth = np.full(3, 1.3*limit*duration) + np.asarray(phi)+tube.sliding_gains*omega + .1
        limits = np.r_[np.full(3,float(ctx.model.relative_position_limit_m)),
                       np.full(3,float(ctx.model.relative_velocity_limit_m_per_rad))]
        lower, upper = np.maximum(center-np.r_[pwidth,vwidth],-limits), np.minimum(center+np.r_[pwidth,vwidth],limits)
        if np.any(lower >= upper):
            raise ValueError('Empty nonlinear local planning domain')
        region = self.regions[chaser_id] = IntervalBox(tuple(lower),tuple(upper))
        a_abs, variation, d_abs, residuals, records = [], [], [], [], []
        for stage in range(count-1):
            lo, hi = q0+stage*h, q0+(stage+1)*h
            coefficients=bound_nominal_coefficients((lo,hi),params)
            a_abs.append([[outward_upper(value) for value in row]
                          for row in coefficients.absolute_upper])
            variation.append([[outward_upper(value) for value in row]
                              for row in coefficients.variation_upper])
            d_abs.append([outward_upper(value) for value in coefficients.drift_upper])
            residual = self.encloser.bound_residual((lo,hi),ctx.target_flow,region,
                IntervalBox.from_center_radius((0,0,0),(limit,)*3),params,ctx.model,ctx.clock)
            if not residual.proven:
                raise ValueError('Nonlinear regional residual unresolved: '+residual.reason)
            residuals.append([outward_upper(v) for v in residual.upper_bounds])
            records.append(residual.as_dict())
        self.residuals[chaser_id] = records
        bounds = ((float(bandwidth[0]),)*2 if ctx.frozen_bandwidth else
                  (tube.bandwidth_range['lower'],tube.bandwidth_range['upper']))
        if ctx.frozen_bandwidth and not np.all(np.asarray(bandwidth)==bandwidth[0]):
            raise ValueError('Fixed-bandwidth comparison requires one declared initial bandwidth')
        terminal_index = None
        if self.absolute_deadline_q is not None:
            remaining=(exact(self.absolute_deadline_q)-q0)/h
            terminal_index=max(0,(remaining.numerator+remaining.denominator-1)//remaining.denominator)
            if terminal_index>=count:
                raise ValueError('Inherited arrival deadline escapes the terminal horizon')
        position_tolerance=cfg.mpc.docking_position_tolerance-(ctx.goal_position_error_m if self.operation=='docking' else 0.)
        velocity_tolerance=cfg.mpc.docking_velocity_tolerance-(ctx.goal_velocity_error_m_per_rad if self.operation=='docking' else 0.)
        if min(position_tolerance,velocity_tolerance)<=0:
            raise ValueError('Moving-goal observation/preview uncertainty exceeds the complete terminal budget')
        contact = None
        if self.operation == 'docking' and terminal_index is not None and ctx.contact_terminal_by_agent is not None:
            if chaser_id not in ctx.contact_terminal_by_agent:
                raise ValueError('Missing prospective contact forecast for the terminal agent')
            contact = ctx.contact_terminal_by_agent[chaser_id]
            contact.validate_grid(q0, h, count)
        return DynamicTubeSpec(phi_initial=phi,omega_initial=omega,lambda_gain=tube.sliding_gains,
            alpha_initial=bandwidth,alpha_bounds=bounds,
            alpha_rate_limit=0. if ctx.frozen_bandwidth else tube.bandwidth_rate_limit,
            delta_A=np.zeros((count-1,3,6)),delta_d=np.asarray(residuals),
            A_abs=np.asarray(a_abs),A_variation=np.asarray(variation),d_abs=np.asarray(d_abs),
            hold_width=cfg.anom_step/tube.feedback_substeps,
            eta=np.array([outward_upper(exact(value)+exact(tube.numerical_reserve))
                for value in ctx.acceleration_error_abs]),
            bandwidth_weight=tube.bandwidth_weight,tube_weight=tube.tube_weight,
            terminal_position_tolerance=position_tolerance,
            terminal_velocity_tolerance=velocity_tolerance,
            enforce_terminal=terminal_index is not None,terminal_start_index=terminal_index,
            measurement_error_abs=ctx.observation_error_abs,
            nominal_endpoint_error_abs=self.nominal_endpoint_error_abs,
            stage_peak_padding_abs=self.stage_peak_padding_abs,
            contact_terminal=contact)

    def problem_bounds(self,chaser,state,spec):
        region = self.regions[chaser.chaser_id]
        lower,upper = region.float_bounds()
        bounds=[]
        for lo,hi,base in zip(lower,upper,self.config.mpc.state_bounds):
            bounds.append(dict(lower=max(lo,float(base.get('lower',-np.inf))),
                               upper=min(hi,float(base.get('upper',np.inf)))))
        return bounds,self.config.mpc.input_bounds

    def configure_terminal(self,spec,phase,state,goal,key):
        # Once anchored, neither a receding horizon nor a temporary retreat
        # disables or moves the inherited physical arrival requirement.
        spec.enforce_terminal = bool(phase.operation=='docking' and self.absolute_deadline_q is not None)

    def geometry_reserve(self,anomaly,count):
        if self.context is None:
            raise ValueError('No current target preview bound')
        return float(self.context.target_position_preview_error_m)
