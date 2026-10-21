"""Numerical truth propagation, separate from admission and navigation.

A proposal has no committed time/state effects. The executor commits only after
all admitted numerical trajectories have been computed. Clearance is diagnostic.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from orbexa.control.certification.reachability import IntervalBox


class NonlinearFleetPlant:
    def __init__(self,target,relative_dynamics,actuator):
        self.target,self.relative_dynamics,self.actuator=target,relative_dynamics,actuator

    def propose_hold(self,relative_states,q,h,decision,navigation_supports):
        proposal=self.target.propose_interval(q+h)
        commands,applied=decision.float_commands(),dict(decision.applied_inputs)
        results={};solutions={}
        for key,initial in relative_states.items():
            command=np.asarray(commands[key])
            def rhs(value,relative,key=key,command=command):
                actual=np.asarray(self.actuator.apply(key,command,value),dtype=float)
                if actual.shape!=(3,) or not np.all(np.isfinite(actual)) or not applied[key].contains(IntervalBox.point(actual)):
                    raise ValueError('Numerical actuator violates its declared applied-input support')
                if not navigation_supports[key].applied_input.contains(IntervalBox.point(actual)):
                    raise ValueError('Numerical actuator violates the declared navigation actuation subset')
                return self.relative_dynamics.relative_rhs(value,relative,proposal.state_at(value)[:6],actual)
            solution=solve_ivp(rhs,(float(q),float(q+h)),initial,method='DOP853',dense_output=True,
                max_step=min(float(h),.0001),rtol=1e-11,atol=1e-10)
            if not solution.success or not np.all(np.isfinite(solution.y)):
                raise RuntimeError('Numerical relative truth refused: '+str(solution.message))
            results[key],solutions[key]=solution.y[:,-1],solution
        return proposal,results,solutions


def sampled_clearance(state,q,phase,*,target,safety,composite_center):
    # Same declared occupied-union distance as MissionSafety.composite_margin,
    # without copying a full serialized geometry for every offline sample.
    # This diagnostic never substitutes for continuous fleet admission.
    if composite_center is not None:
        body=target.rotation_matrix_at(q).T@(np.asarray(state[:3])-composite_center)
        return float(target.geometry.signed_distance(body)-safety.chaser_body_radius)
    return safety.active_margin_for_state(state,q,phase)
