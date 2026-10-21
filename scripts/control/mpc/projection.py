"""Numerical hard-row QP proposals with explicit failure and progress slack.

This is not the mission's GEKKO optimizer or physical admission certificate.
There is no zero/nominal-action fallback on failure: command is then None.
Derived from the MIT Swarm CLF-CBF Playground; see THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize


@dataclass(frozen=True)
class CommandProjection:
    accepted: bool
    command: np.ndarray | None
    progress_slack: float | None
    maximum_hard_violation: float | None
    message: str
    physical_admission: bool = False


def project_command(nominal, rows, bounds, input_lower, input_upper, *,
                    clf_row=None, clf_bound=None, progress_weight=1.e4,
                    tolerance=1.e-8, max_iterations=200):
    """min .5||u-u_nom||² + .5*w*delta², hard Gu<=b and input box.

    Only an optional CLF row receives nonnegative delta. A successful numerical
    candidate still needs model, sampling, uncertainty and continuation checks.
    """
    nominal = np.asarray(nominal,dtype=float)
    if nominal.ndim!=1 or nominal.size<1 or not np.all(np.isfinite(nominal)):
        raise ValueError('Nominal command must be a finite vector')
    n = nominal.size
    lo,hi = np.asarray(input_lower,dtype=float),np.asarray(input_upper,dtype=float)
    g,b = np.asarray(rows,dtype=float),np.asarray(bounds,dtype=float)
    if g.size==0:
        g = np.empty((0,n))
    if (lo.shape!=(n,) or hi.shape!=(n,) or np.any(lo>hi) or g.ndim!=2 or
            g.shape[1]!=n or b.shape!=(g.shape[0],) or
            not all(np.all(np.isfinite(v)) for v in (lo,hi,g,b))):
        raise ValueError('Hard rows and finite input bounds have inconsistent dimensions')
    if (not np.isfinite(tolerance) or tolerance<=0 or isinstance(max_iterations,bool)
            or not isinstance(max_iterations,int) or max_iterations<1):
        raise ValueError('Declare a positive tolerance and finite iteration budget')
    soft = clf_row is not None
    if soft != (clf_bound is not None):
        raise ValueError('Both CLF row and bound are required')
    x0 = np.clip(nominal,lo,hi)
    constraints=[]
    hard = np.pad(g,((0,0),(0,int(soft))))
    if len(b):
        constraints.append(LinearConstraint(hard,-np.inf,b))
    if soft:
        c = np.asarray(clf_row,dtype=float)
        if (c.shape!=(n,) or not np.all(np.isfinite(c)) or not np.isfinite(clf_bound)
                or not np.isfinite(progress_weight) or progress_weight<=0):
            raise ValueError('CLF row/weight must be finite and dimensionally consistent')
        constraints.append(LinearConstraint(np.r_[c,-1.][None,:],-np.inf,float(clf_bound)))
        x0=np.r_[x0,max(0.,c @ x0-clf_bound)]
    def objective(x):
        return .5*float((x[:n]-nominal) @ (x[:n]-nominal)) + (
            .5*progress_weight*x[-1]**2 if soft else 0.)
    def gradient(x):
        return np.r_[x[:n]-nominal,progress_weight*x[-1]] if soft else x-nominal
    result=minimize(objective,x0,jac=gradient,method='SLSQP',constraints=constraints,
        bounds=Bounds(np.r_[lo,0.] if soft else lo,np.r_[hi,np.inf] if soft else hi),
        # Objective convergence is distinct from the hard-row readback below.
        # Squaring a 1e-8 feasibility tolerance asks SLSQP for machine-epsilon
        # objective changes and can reject a valid constrained minimizer.
        options=dict(maxiter=max_iterations,ftol=1.e-12,disp=False))
    if result.x is None or not np.all(np.isfinite(result.x)):
        return CommandProjection(False,None,None,None,str(result.message))
    u=result.x[:n]
    violation=float(max(0.,np.max(g @ u-b,initial=0.),np.max(lo-u),np.max(u-hi)))
    slack=float(result.x[-1]) if soft else 0.
    clf_violation=max(0.,float(c @ u-slack-clf_bound),-slack) if soft else 0.
    okay=bool(result.success and violation<=tolerance and clf_violation<=tolerance)
    return CommandProjection(okay,u.copy() if okay else None,slack if okay else None,
        violation,str(result.message))
