"""Independent binding of an optimized tube stage to its declared sources.

The checker reads configuration, the current admission authority and one
observation clock. It cannot commit physical motion or navigation state.
"""
from __future__ import annotations

import copy
import numpy as np
from orbexa.core.reference_frame import exact, fingerprint
from orbexa.control.reachability import IntervalBox


class PerformanceStageBindings:
    def __init__(self,*,config,clock,runtime,observed_q,precision_bits):
        self.config,self.clock,self.runtime=config,clock,runtime
        self.observed_q,self.precision_bits=observed_q,precision_bits

    def snapshot(self,planning,planner,key,*,_target_identity=None):
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

    def bind(self,planning,planner,q0,stage):
        from orbexa.control.nonlinear_flow import _SOURCE_SHA256 as residual_source
        from orbexa.control.nonlinear_flow import bound_nominal_coefficients
        from orbexa.core.dynamics import orbital_ellp_drag
        params=planning.parameters
        try:
            coefficients=bound_nominal_coefficients((q0,q0+exact(self.config.anom_step)),params,self.precision_bits)
        except ImportError as error:raise ValueError('Arb nominal coefficient source unavailable') from error
        arb_abs,arb_change=coefficients.absolute_upper,coefficients.variation_upper
        limits=IntervalBox(tuple(b['lower'] for b in self.config.mpc.input_bounds),
            tuple(b['upper'] for b in self.config.mpc.input_bounds))
        matrix=orbital_ellp_drag(anom_step=float(self.config.anom_step),
            **{name:params[name] for name in ('eccentricity','alpha','beta','semi_major_axis','mu')},
            theta0=params.get('theta0',0))[0][0]
        bindings={}
        for key,data in stage.items():
            spec=planning.plans[key].spec;snapshot=self.snapshot(planning,planner,key)
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
                eta_required=tuple(map(str,eta_required)),precision_bits=self.precision_bits)
            proof['identity']=fingerprint(proof)
            data['source_binding']=proof
            data['proof_record']['source_binding_id']=proof['identity']
            data['proof_record']['identity']=fingerprint({name:value for name,value in data['proof_record'].items() if name!='identity'})
            bindings[key]=(fingerprint(snapshot),planning.matrices[0])
        return bindings,matrix

    def check(self,planning,planner,bindings):
        # Every member shares the same complete prospective target proof.
        # Hash its full current payload before and after this synchronous
        # batch, while retaining every per-agent mutable-source snapshot.
        # No target identity persists between physical holds.
        target=planner.context.target_flow
        target_identity=target.fingerprint
        if (set(bindings)!=set(planning.plans) or any(planning.matrices[0] is not callback
                or fingerprint(self.snapshot(planning,planner,key,
                    _target_identity=(target,target_identity)))!=identity
                for key,(identity,callback) in bindings.items())):
            raise ValueError('Planning stage/nominal/residual source changed after preflight binding')
        if planner.context.target_flow is not target or target.fingerprint!=target_identity:
            raise ValueError('Planning target proof payload changed during the source-check batch')
