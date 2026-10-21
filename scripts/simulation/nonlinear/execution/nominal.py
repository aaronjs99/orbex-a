"""Certified CH nominal propagation and physical tube endpoint transfer.

The nominal surrogate is distinct from the numerical nonlinear plant. Its cache
is keyed by the declared nominal model and enclosure configuration.
"""
from __future__ import annotations

import numpy as np
from orbexa.core.dynamics.reference_frame import exact, fingerprint
from orbexa.control.certification.reachability import IntervalBox, OrbitalModelBounds
from orbexa.control.certification.orbital import OrbitalHoldContext, OrbitalHoldRequest, OrbitalHoldProofService
from orbexa.control.tubes.sensed import _upper, _outward_tube_endpoint, measurement_radius
from orbexa.estimation.navigation.boxes import midpoint


class NominalTubePropagator:
    def __init__(self):
        self.services={}

    def propagate(self,center,u,q,h,params,phi,omega,stage,lam,*,input_bounds,clock_domain,nominal_config,precision_bits):
        budget=stage.get('nominal_endpoint_error_abs')
        h=exact(h)
        prospective_budget,prospective_hold=None,None
        if budget is not None:
            prospective_budget=tuple(map(exact,budget))
            prospective_hold=exact(stage['nominal_endpoint_budget_hold'])
            if not 0<h<=prospective_hold:
                raise ValueError('Executed hold escapes the positive prospective nominal-endpoint budget domain')
            if h<prospective_hold:
                rates=tuple(tuple(map(exact,values)) for values in (stage['alpha'],lam))
                if any(len(values)!=3 or any(value<0 for value in values) for values in rates):
                    raise ValueError('Shortened nominal budgets require three finite nonnegative alpha and lambda rates')
            # The planned reset forcing integrates to epsilon over H. For
            # every nonnegative decay rate a, w_a(t)=(1-exp(-a*t))/a is
            # concave with w_a(0)=0, so w_a(h)/w_a(H)>=h/H. Scaling all
            # endpoint components by h/H therefore preserves BOTH position
            # and sliding-error reset allowances at a shorter held cadence.
            # The actual certified defect must still fit this smaller budget.
            budget=tuple(value*h/prospective_hold for value in prospective_budget)
        model=OrbitalModelBounds(eccentricity=(params['eccentricity'],)*2,
            alpha=(params['alpha'],)*2,beta=(params['beta'],)*2,
            semi_major_axis=params['semi_major_axis'],mu=params['mu'],theta0=(params.get('theta0',0),)*2)
        # The existing authenticated service retains its original Picard swept
        # witness and independently refines the endpoint by variation of
        # constants. This is solely the CH nominal surrogate, never a physical
        # nonlinear/frame certificate. Exact source checks remain in prove().
        source_id=fingerprint(dict(role='nominal-CH-surrogate-in-commanded-coordinates',model=model.as_dict()))
        service_key=(source_id,nominal_config)
        if service_key not in self.services:
            limits=IntervalBox(tuple(b['lower'] for b in input_bounds),
                               tuple(b['upper'] for b in input_bounds))
            context=OrbitalHoldContext(model,clock_domain,limits,IntervalBox.point((0,0,0)),
                source_id,nominal_config)
            self.services[service_key]=OrbitalHoldProofService(context)
        certificate=self.services[service_key].prove(OrbitalHoldRequest(
            IntervalBox.point(center),IntervalBox.point(u),q,h))
        if not certificate.proven:raise ValueError('Nominal Picard/VoC endpoint unresolved')
        chosen=np.array(midpoint(certificate.endpoint))
        defect=measurement_radius(certificate.endpoint,chosen)
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
            stage['disturbance_bound'],h,stage['phi_peak'],precision_bits)
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
            nominal_endpoint_budget_hold=None if budget is None else str(h),
            precision_bits=precision_bits)
        if budget is not None and h<prospective_hold:
            transfer['nominal_endpoint_budget_origin']=dict(
                method='exact-positive-comparison-short-hold-scaling-v1',
                prospective_hold=str(prospective_hold),
                prospective_error_abs=tuple(map(str,prospective_budget)),
                scale=str(h/prospective_hold))
        transfer['identity']=fingerprint(transfer);stage['last_transfer']=transfer
        return chosen,next_phi,next_omega,certificate,defect
