"""Dynamics-consistent initial guesses for the existing MPC problem boundary.

These hints own no motion authority. The actual moving reference, frame, input
units and RK4 coefficients come from MPCProblem. No toy simulator is substituted.
MIT source attribution: THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import json
import numpy as np


@dataclass(frozen=True)
class AffinePrediction:
    state_matrices: np.ndarray
    input_matrices: np.ndarray
    offsets: np.ndarray
    initial_state: np.ndarray

    @classmethod
    def from_problem(cls, problem):
        n,m,h=problem.num_states,problem.num_inputs,float(problem.anom_step)
        if not np.isfinite(h) or h<=0:
            raise ValueError('Prediction step must be positive and finite')
        start=float(problem.extra_params.get('start_anom',0.));tp=problem.extra_params.get('time_periapsis',0.)
        a_function=problem.extra_params.get('dynamics_callable')
        d_function=problem.extra_params.get('disturbance_callable')
        def coefficients(q):
            a=np.asarray(a_function(q,tp) if callable(a_function) else problem.dynamics_matrix,dtype=float)
            raw=problem.extra_params.get('disturbance_vector',np.zeros(n))
            if raw is None:raw=np.zeros(n)
            d=np.asarray(d_function(q,tp) if callable(d_function) else raw,dtype=float)
            if a.shape!=(n,n) or d.shape!=(n,) or not np.all(np.isfinite(a)) or not np.all(np.isfinite(d)):
                raise ValueError('Prediction needs finite affine coefficients')
            return a,d
        b=np.asarray(problem.input_matrix,dtype=float)
        if b.shape!=(n,m) or not np.all(np.isfinite(b)):
            raise ValueError('Invalid input matrix')
        aa=[];bb=[];dd=[]
        for k in range(problem.num_steps-1):
            q=start+k*h
            c0,cm,c1=coefficients(q),coefficients(q+h/2),coefficients(q+h)
            def step(x,u):
                if problem.dynamics_type=='discrete':
                    return c0[0] @ x+b @ u+c0[1]
                if problem.dynamics_type!='continuous':
                    raise ValueError('Unknown nominal dynamics type')
                def rhs(v,c):return c[0] @ v+b @ u+c[1]
                k1=rhs(x,c0);k2=rhs(x+h*k1/2,cm);k3=rhs(x+h*k2/2,cm);k4=rhs(x+h*k3,c1)
                return x+h*(k1+2*k2+2*k3+k4)/6
            zero=step(np.zeros(n),np.zeros(m))
            aa.append(np.column_stack([step(np.eye(n)[i],np.zeros(m))-zero for i in range(n)]))
            bb.append(np.column_stack([step(np.zeros(n),np.eye(m)[i])-zero for i in range(m)]))
            dd.append(zero)
        initial=np.asarray(problem.initial_state,dtype=float).copy()
        if initial.shape!=(n,) or not np.all(np.isfinite(initial)):
            raise ValueError('Prediction initial state must be finite')
        return cls(np.stack(aa),np.stack(bb),np.stack(dd),initial)

    def rollout(self, controls):
        u=np.asarray(controls,dtype=float)
        count=len(self.state_matrices);m=self.input_matrices.shape[2]
        if u.shape!=(m,count) or not np.all(np.isfinite(u)):
            raise ValueError('Initial controls must match the prediction intervals')
        x=np.empty((len(self.initial_state),count+1));x[:,0]=self.initial_state
        for k in range(count):
            x[:,k+1]=self.state_matrices[k] @ x[:,k]+self.input_matrices[k] @ u[:,k]+self.offsets[k]
        if not np.all(np.isfinite(x)):
            raise ValueError('Nominal initial-guess rollout overflowed')
        return x

    def condense(self):
        """X=[x1;...;xH]=M*x0+S*[u0;...;u(H-1)]+c."""
        n=len(self.initial_state);count=len(self.state_matrices);m=self.input_matrices.shape[2]
        state=np.eye(n);control=np.zeros((n,count*m));offset=np.zeros(n)
        mm=[];ss=[];cc=[]
        for k,(a,b,d) in enumerate(zip(self.state_matrices,self.input_matrices,self.offsets)):
            state=a @ state;control=a @ control;control[:,k*m:(k+1)*m]+=b
            offset=a @ offset+d;mm.append(state.copy());ss.append(control.copy());cc.append(offset.copy())
        return np.vstack(mm),np.vstack(ss),np.concatenate(cc)


@dataclass(frozen=True)
class TrajectoryHint:
    states: np.ndarray
    controls: np.ndarray
    source: dict


def input_limits(problem):
    bounds=problem.input_bounds
    if bounds is None:
        return np.full(problem.num_inputs,-np.inf),np.full(problem.num_inputs,np.inf)
    if len(bounds)!=problem.num_inputs:
        raise ValueError('One input-bound mapping is required per command')
    lo=np.array([-np.inf if b.get('lower') is None else b['lower'] for b in bounds],dtype=float)
    hi=np.array([np.inf if b.get('upper') is None else b['upper'] for b in bounds],dtype=float)
    if np.any(np.isnan(lo)) or np.any(np.isnan(hi)) or np.any(lo>hi):
        raise ValueError('Initial-guess input bounds are inconsistent')
    return lo,hi


def hint_identity(states,controls,method):
    payload=dict(method=method,states=states.tolist(),controls=controls.tolist())
    return sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class TrackingHint:
    """PD plus moving-goal feedforward, with optional bounded proposal shaping.

    Gains use the problem's independent variable (seconds or anomaly), not an
    assumed clock. Damping tracks reference velocity; it is not faded at contact.
    A correction(state, reference, k) may supply potential-field, graph, GP or
    learned preferences, always before native constrained planning.
    """
    position_gain: float = 1.
    velocity_gain: float = 2.
    correction: object | None = None

    def __post_init__(self):
        if any(not np.isfinite(v) or v<0 for v in (self.position_gain,self.velocity_gain)):
            raise ValueError('Initial-guess gains must be finite and nonnegative')
        if self.correction is not None and not callable(self.correction):
            raise TypeError('Hint correction must be callable')

    def propose(self, problem):
        if problem.num_states!=6 or problem.num_inputs!=3:
            raise ValueError('Position/velocity tracking hints require six states and three inputs')
        model=AffinePrediction.from_problem(problem);lo,hi=input_limits(problem)
        count=problem.num_steps-1;x=model.initial_state.copy();u=np.empty((3,count))
        for k in range(count):
            reference=problem.reference_states[:,k]
            command=problem.reference_inputs[:,k]+self.position_gain*(reference[:3]-x[:3])
            command+=self.velocity_gain*(reference[3:]-x[3:])
            if self.correction is not None:
                extra=np.asarray(self.correction(x.copy(),reference.copy(),k),dtype=float)
                if extra.shape!=(3,) or not np.all(np.isfinite(extra)):
                    raise ValueError('Hint correction returned an invalid acceleration-coordinate vector')
                command+=extra
            u[:,k]=np.clip(command,lo,hi)
            x=model.state_matrices[k] @ x+model.input_matrices[k] @ u[:,k]+model.offsets[k]
        states=model.rollout(u);method='moving-reference-feedback-rk4-hint-v1'
        return TrajectoryHint(states,u,dict(method=method,identity=hint_identity(states,u,method),
            role='Numerical initial guess only; no physical admission authority'))


@dataclass(frozen=True)
class PotentialRepulsion:
    """Bounded reciprocal-gap preference; local minima remain possible."""
    centers: np.ndarray
    radii: np.ndarray
    influence_m: float = 5.
    gain: float = 1.
    maximum: float = 1.

    def __call__(self,state,reference,k):
        centers=np.asarray(self.centers,dtype=float);radii=np.asarray(self.radii,dtype=float)
        if (centers.ndim!=2 or centers.shape[1]!=3 or radii.shape!=(len(centers),)
                or not np.all(np.isfinite(centers)) or not np.all(np.isfinite(radii)) or np.any(radii<0)
                or any(not np.isfinite(v) or v<=0 for v in (self.influence_m,self.gain,self.maximum))):
            raise ValueError('Potential preference requires finite geometry and positive scales')
        result=np.zeros(3)
        for center,radius in zip(centers,radii):
            r=state[:3]-center;distance=np.linalg.norm(r);gap=distance-radius
            if distance<=1.e-12:
                raise ValueError('Coincident center has no defined repulsion direction')
            if gap<self.influence_m:
                d=max(gap,1.e-6)
                result+=self.gain*(1/d-1/self.influence_m)/d**2*r/distance
        norm=np.linalg.norm(result)
        return result*min(1.,self.maximum/max(norm,1.e-30))


@dataclass(frozen=True)
class ConvexTrajectoryHint:
    """Condensed affine MPC proposal, optionally in supplied hard halfspaces.

    Rows constrain the stacked predicted states, not the inputs. Moving obstacle
    forecasts and tube/geometry supports must be supplied by the caller when
    building such halfspaces. They never replace the native nonconvex problem.
    """
    state_rows: np.ndarray | None = None
    state_lower_bounds: np.ndarray | None = None
    tolerance: float = 1.e-8
    max_iterations: int = 200

    def __post_init__(self):
        if not np.isfinite(self.tolerance) or self.tolerance<=0 or isinstance(self.max_iterations,bool) or not isinstance(self.max_iterations,int) or self.max_iterations<1:
            raise ValueError('Convex hint tolerance and iteration budget must be positive')

    def propose(self,problem):
        from scipy.linalg import block_diag
        from scipy.optimize import Bounds,LinearConstraint,minimize
        model=AffinePrediction.from_problem(problem);m=problem.num_inputs;n=problem.num_states
        count=problem.num_steps-1;mx,s,offset=model.condense()
        q=np.asarray(problem.state_cost_matrix,dtype=float);r=np.asarray(problem.input_cost_matrix,dtype=float)
        if q.ndim==1:q=np.diag(q)
        if r.ndim==1:r=np.diag(r)
        for weight,size,strict in ((q,n,False),(r,m,True)):
            if (weight.shape!=(size,size) or not np.all(np.isfinite(weight)) or not np.allclose(weight,weight.T)
                    or np.min(np.linalg.eigvalsh(weight)) < (1.e-14 if strict else -1.e-12)):
                raise ValueError('Convex proposal weights require PSD state and PD command matrices')
        weights=[q]*count
        if problem.terminal_cost_matrix is not None:
            terminal=np.asarray(problem.terminal_cost_matrix,dtype=float)
            if (terminal.shape!=(n,n) or not np.all(np.isfinite(terminal))
                    or not np.allclose(terminal,terminal.T)
                    or np.min(np.linalg.eigvalsh(terminal)) < -1.e-12):
                raise ValueError('Convex proposal terminal weight must be symmetric positive semidefinite')
            weights[-1]=q+terminal
        qb=block_diag(*weights);rb=block_diag(*([r]*count))
        target=problem.reference_states[:,1:].T.reshape(-1)
        reference=problem.reference_inputs.T.reshape(-1)
        error=mx @ model.initial_state+offset-target
        hessian=2*(s.T @ qb @ s+rb);gradient=2*s.T @ qb @ error-2*rb @ reference
        lo,hi=input_limits(problem);lower=np.tile(lo,count);upper=np.tile(hi,count)
        constraints=[]
        if (self.state_rows is None)!=(self.state_lower_bounds is None):
            raise ValueError('State halfspace rows and lower bounds must be supplied together')
        if self.state_rows is not None:
            rows=np.asarray(self.state_rows,dtype=float);bounds=np.asarray(self.state_lower_bounds,dtype=float)
            if (rows.ndim!=2 or rows.shape[1]!=n*count or bounds.shape!=(len(rows),)
                    or not np.all(np.isfinite(rows)) or not np.all(np.isfinite(bounds))):
                raise ValueError('Proposal halfspaces must match the stacked prediction states')
            constraints=[LinearConstraint(rows @ s,bounds-rows @ (mx @ model.initial_state+offset),np.inf)]
        result=minimize(lambda u:.5*u @ hessian @ u+gradient @ u,np.clip(reference,lower,upper),
            jac=lambda u:hessian @ u+gradient,method='SLSQP',bounds=Bounds(lower,upper),constraints=constraints,
            options=dict(maxiter=self.max_iterations,ftol=1.e-12,disp=False))
        if not result.success or not np.all(np.isfinite(result.x)):
            raise ValueError(f'Convex proposal solver refused: {result.message}')
        controls=result.x.reshape(count,m).T;states=model.rollout(controls)
        violation=max(0.,float(np.max(lower-result.x)),float(np.max(result.x-upper)))
        if self.state_rows is not None:
            violation=max(violation,float(np.max(bounds-rows @ states[:,1:].T.reshape(-1),initial=0.)))
        if violation>self.tolerance:raise ValueError('Convex initial guess failed its numerical constraint check')
        method='condensed-affine-convex-mpc-hint-v1'
        return TrajectoryHint(states,controls,dict(method=method,maximum_violation=violation,
            identity=hint_identity(states,controls,method),role='Initial guess; nonlinear native planning still required'))


@dataclass(frozen=True)
class ConsensusInitialGuess:
    """Extract one chaser block from a converged JOINT-control consensus vector."""
    consensus_result: object
    agent_index: int
    agent_count: int

    def propose(self,problem):
        result=self.consensus_result
        if not result.converged:raise ValueError('Unfinished consensus is not an accepted trajectory proposal')
        if (isinstance(self.agent_count,bool) or not isinstance(self.agent_count,int) or self.agent_count<1
                or isinstance(self.agent_index,bool) or not isinstance(self.agent_index,int)
                or not 0<=self.agent_index<self.agent_count):
            raise ValueError('Consensus extraction requires an explicit agent block')
        count=problem.num_steps-1;m=problem.num_inputs
        values=np.asarray(result.consensus,dtype=float)
        if values.shape!=(self.agent_count*count*m,) or not np.all(np.isfinite(values)):
            raise ValueError('Shared consensus vector does not contain the declared joint trajectory blocks')
        controls=values.reshape(self.agent_count,count,m)[self.agent_index].T.copy()
        lo,hi=input_limits(problem)
        if np.any(controls<lo[:,None]) or np.any(controls>hi[:,None]):
            raise ValueError('Consensus proposal exceeds the declared command box')
        states=AffinePrediction.from_problem(problem).rollout(controls);method='joint-consensus-mpc-initial-guess-v1'
        return TrajectoryHint(states,controls,dict(method=method,iterations=result.iterations,
            primal_residual=result.primal_residual,dual_residual=result.dual_residual,
            agent_index=self.agent_index,identity=hint_identity(states,controls,method),
            role='Numerical joint proposal only; native planning and physical admission still required'))
