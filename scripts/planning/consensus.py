"""Synchronous consensus ADMM for copies of one shared decision vector.

Copies must name the SAME joint trajectory/assignment variables. Averaging
different chasers' private accelerations solves a different problem. Numerical
convergence does not establish convexity, collision safety or mission admission.
Inspired by the MIT Swarm CLF-CBF Playground; see THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence
import numpy as np


@dataclass(frozen=True)
class ConsensusResult:
    consensus: np.ndarray
    local_copies: tuple[np.ndarray,...]
    converged: bool
    iterations: int
    primal_residual: float
    dual_residual: float
    residual_history: tuple[tuple[float,float],...]
    status: str


@dataclass(frozen=True)
class ConsensusADMM:
    rho: float = 1.
    max_iterations: int = 100
    absolute_tolerance: float = 1.e-6
    relative_tolerance: float = 1.e-5

    def __post_init__(self):
        if any(not np.isfinite(v) or v<=0 for v in
               (self.rho,self.absolute_tolerance,self.relative_tolerance)) or (
                isinstance(self.max_iterations,bool) or not isinstance(self.max_iterations,int)
                or self.max_iterations<1):
            raise ValueError('ADMM requires finite positive rates/tolerances and an integer iteration budget')

    def solve(self, local_prox: Sequence[Callable], initial):
        """prox_i(v,rho) minimizes f_i(x)+(rho/2)||x-v||².

        Each callback must return a finite same-shaped numerical minimizer or
        raise; failure is not silently replaced with a zero trajectory.
        """
        z=np.asarray(initial,dtype=float)
        if z.ndim!=1 or not z.size or not np.all(np.isfinite(z)) or not local_prox:
            raise ValueError('Use a finite shared vector and at least one local proximal map')
        z=z.copy();count=len(local_prox);dual=np.zeros((count,len(z)))
        history=[];copies=np.repeat(z[None,:],count,axis=0)
        for iteration in range(1,self.max_iterations+1):
            # All maps see one frozen consensus/dual snapshot (Jacobi update).
            proposed=[np.asarray(fn(z-dual[i],self.rho),dtype=float) for i,fn in enumerate(local_prox)]
            if any(x.shape!=z.shape or not np.all(np.isfinite(x)) for x in proposed):
                raise ValueError('Local proximal solver returned an invalid shared-variable copy')
            copies=np.stack(proposed);old=z.copy()
            z=np.mean(copies+dual,axis=0);dual+=copies-z
            primal=float(np.linalg.norm(copies-z))
            dual_residual=float(self.rho*np.sqrt(count)*np.linalg.norm(z-old))
            history.append((primal,dual_residual))
            ep=np.sqrt(count*len(z))*self.absolute_tolerance+self.relative_tolerance*max(
                np.linalg.norm(copies),np.sqrt(count)*np.linalg.norm(z))
            ed=np.sqrt(count*len(z))*self.absolute_tolerance+self.relative_tolerance*self.rho*np.linalg.norm(dual)
            if primal<=ep and dual_residual<=ed:
                return ConsensusResult(z,tuple(x.copy() for x in copies),True,iteration,
                    primal,dual_residual,tuple(history),'NUMERICAL_CONSENSUS')
        return ConsensusResult(z,tuple(x.copy() for x in copies),False,iteration,
            primal,dual_residual,tuple(history),'ITERATION_LIMIT')


@dataclass(frozen=True)
class QuadraticProximalModel:
    """Local quadratic in the declared shared variables, with optional hard rows.

    Callers embed each chaser's private control blocks in the joint decision
    vector explicitly; consensus never forces different agents to use equal u.
    """
    hessian: np.ndarray
    linear: np.ndarray
    rows: np.ndarray | None = None
    upper_bounds: np.ndarray | None = None
    tolerance: float = 1.e-8

    def __call__(self,center,rho):
        q=np.asarray(self.hessian,dtype=float);c=np.asarray(self.linear,dtype=float)
        v=np.asarray(center,dtype=float)
        if v.ndim!=1 or not v.size:
            raise ValueError('Use a nonempty shared vector')
        n=len(v)
        if (not np.isfinite(self.tolerance) or self.tolerance<=0
                or c.shape!=(n,) or q.shape!=(n,n) or not np.allclose(q,q.T)
                or any(not np.all(np.isfinite(a)) for a in (q,c,v)) or np.min(np.linalg.eigvalsh(q)) < -1.e-12
                or not np.isfinite(rho) or rho<=0):
            raise ValueError('Local consensus objective must be a finite convex quadratic in the shared vector')
        h=q+rho*np.eye(n);g=c-rho*v;unconstrained=np.linalg.solve(h,-g)
        if (self.rows is None)!=(self.upper_bounds is None):
            raise ValueError('Local consensus rows and bounds must be supplied together')
        if self.rows is None:return unconstrained
        from scipy.optimize import LinearConstraint,minimize
        rows=np.asarray(self.rows,dtype=float);bounds=np.asarray(self.upper_bounds,dtype=float)
        if (rows.ndim!=2 or rows.shape[1]!=n or bounds.shape!=(len(rows),)
                or not np.all(np.isfinite(rows)) or not np.all(np.isfinite(bounds))):
            raise ValueError('Local consensus rows must match the shared vector')
        result=minimize(lambda x:.5*x @ h @ x+g @ x,unconstrained,jac=lambda x:h @ x+g,
            method='SLSQP',constraints=[LinearConstraint(rows,-np.inf,bounds)],options=dict(ftol=1.e-12,maxiter=200))
        if not result.success or not np.all(np.isfinite(result.x)) or np.max(rows @ result.x-bounds,initial=0.)>self.tolerance:
            raise ValueError('Local consensus proximal solve failed numerical feasibility')
        return result.x.copy()
