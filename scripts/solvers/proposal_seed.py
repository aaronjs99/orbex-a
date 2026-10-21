"""Optional warm-start decorator preserving the authoritative solver boundary."""
from __future__ import annotations
from dataclasses import replace
import time
import numpy as np
from orbexa.solvers.base import SolverBase
from orbexa.planning.proposals.trajectory import AffinePrediction, input_limits


class ProposalSeededSolver(SolverBase):
    """Supply a hint to GEKKO, never replace its formulation or applied action.

    Existing caller guesses win by default. Invalid numerical hints are recorded
    and discarded, after which the original solver setup still runs unchanged.
    """
    def __init__(self,delegate,provider,*,replace_existing=False,maximum_preparation_s=None):
        if not isinstance(delegate,SolverBase) or not callable(getattr(provider,'propose',None)):
            raise TypeError('Use an existing solver backend and a propose(MPCProblem) provider')
        super().__init__(delegate.config)
        if maximum_preparation_s is not None and (
                not np.isfinite(maximum_preparation_s) or maximum_preparation_s<=0):
            raise ValueError('Optional proposal time budget must be finite and positive')
        self.delegate,self.provider,self.replace_existing=delegate,provider,bool(replace_existing)
        self.maximum_preparation_s=maximum_preparation_s
        self.proposal_record=None

    def setup(self,problem):
        self._problem=problem
        record=dict(status='PRESERVED_CALLER_GUESS',role='Initial guess only; native optimization and physical admission remain required')
        extra=dict(problem.extra_params)
        existing=any(extra.get(k) is not None for k in ('state_initial_guess','control_initial_guess'))
        started=time.perf_counter()
        if self.replace_existing or not existing:
            try:
                hint=self.provider.propose(problem)
                controls=np.asarray(hint.controls,dtype=float)
                if controls.shape!=(problem.num_inputs,problem.num_steps-1) or not np.all(np.isfinite(controls)):
                    raise ValueError('Proposal controls have invalid shape/values')
                lo,hi=input_limits(problem)
                if np.any(controls<lo[:,None]) or np.any(controls>hi[:,None]):
                    raise ValueError('Proposal controls exceed the declared input box')
                # Rebuild from current state and the actual problem model; do not
                # inherit another simulator's state rollout or stale first node.
                states=AffinePrediction.from_problem(problem).rollout(controls)
                record['source']=dict(hint.source)
                elapsed=time.perf_counter()-started
                if self.maximum_preparation_s is not None and elapsed>self.maximum_preparation_s:
                    record.update(status='DISCARDED_LATE_HINT',budget_s=self.maximum_preparation_s)
                else:
                    extra.update(state_initial_guess=states,control_initial_guess=controls)
                    record['status']='SUPPLIED_NATIVE_INITIAL_GUESS'
            except (ValueError,ArithmeticError,TimeoutError) as error:
                record.update(status='DISCARDED_INVALID_HINT',reason=f'{type(error).__name__}: {error}')
        record['preparation_wall_s']=time.perf_counter()-started
        # This is an acceptance budget, not preemption. A hard real-time caller
        # must prepare proposals on a separate worker with a bounded mailbox.
        self.proposal_record=record
        self.delegate.setup(replace(problem,extra_params=extra))
        self._is_setup=True

    def solve(self):
        if not self._is_setup:
            raise RuntimeError('Set up the decorated solver first')
        result=self.delegate.solve()
        result.solver_info['planning_proposal']=dict(self.proposal_record)
        return result

    def cleanup(self):
        self.delegate.cleanup()
        super().cleanup()
