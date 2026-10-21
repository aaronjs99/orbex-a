"""Conservative candidate-box membership for the declared nonlinear model.

Only proven transition/observation disjointness removes a candidate physical
parameter box. Failed inclusions and exhausted budgets retain unknown boxes.
"""
from __future__ import annotations

from dataclasses import dataclass
from orbexa.control.certification.reachability import IntervalBox, HeldInputRequest
from orbexa.control.certification.nonlinear import NonlinearFlowEncloser
from orbexa.core.dynamics.relative import NonlinearModelBounds
from orbexa.control.actuation import CommandDependentActuationPrior


def _disjoint(left, right):
    if left.dimension != right.dimension:
        raise ValueError('Membership observation dimensions disagree')
    return any(a > d or c > b for a, b, c, d in zip(left.lower, left.upper, right.lower, right.upper))


@dataclass(frozen=True)
class NonlinearTransitionObservation:
    relative_initial: IntervalBox
    relative_observed: IntervalBox
    held_input: IntervalBox
    start_q: tuple
    width_q: tuple
    target_initial: IntervalBox
    physical_time_initial: tuple = (0, 0)
    acceleration_disturbance: IntervalBox = IntervalBox.point((0, 0, 0))
    observation_id: str = ''
    target_observed: IntervalBox | None = None
    agent_id: str = ''
    commanded_input: tuple | None = None
    actuation_prior: CommandDependentActuationPrior | None = None
    actuation_persistence: str = ''


@dataclass(frozen=True)
class NonlinearMembershipResult:
    retained_models: tuple
    discarded_fingerprints: tuple
    records: tuple
    status: str


class NonlinearMembershipContractor:
    def __init__(self, encloser=None, max_boxes=64):
        if isinstance(max_boxes, bool) or not isinstance(max_boxes, int) or max_boxes < 1:
            raise ValueError('Membership needs a finite positive candidate budget')
        self.encloser, self.max_boxes = encloser or NonlinearFlowEncloser(), max_boxes

    def contract(self, candidate_models, transitions, observations=None, clock=None):
        if clock is None:
            raise ValueError('Supply the immutable commanded reference clock')
        models, transitions = tuple(candidate_models), tuple(transitions)
        if not models or not all(isinstance(model, NonlinearModelBounds) for model in models):
            raise ValueError('Supply nonempty typed physical parameter candidates')
        if not all(isinstance(item, NonlinearTransitionObservation) for item in transitions):
            raise TypeError('Transitions must bind actual observation and control boxes')
        if observations is not None:
            raise ValueError('Observation boxes belong in NonlinearTransitionObservation, never an unrelated side array')
        retained, discarded, records = [], [], []
        for index, model in enumerate(models):
            if index >= self.max_boxes:
                retained.append(model)
                records.append(dict(candidate=model.fingerprint, status='UNRESOLVED_BUDGET'))
                continue
            rejected = False
            for transition in transitions:
                target = self.encloser.enclose_target(transition.target_initial, transition.start_q,
                    transition.width_q, model, clock, transition.physical_time_initial)
                record = dict(candidate=model.fingerprint, observation_id=transition.observation_id,
                              target_status=target.status.value, target_fingerprint=target.fingerprint)
                if not target.proven:
                    record.update(status='UNRESOLVED_RETAINED', reason=target.reason)
                    records.append(record)
                    continue
                if transition.target_observed is not None and _disjoint(target.endpoint, transition.target_observed):
                    record.update(status='CERTIFIED_INCONSISTENT', reason='Complete target endpoint set disjoint from observation')
                    records.append(record)
                    rejected = True
                    break
                request = HeldInputRequest(transition.relative_initial, transition.held_input,
                    transition.start_q, transition.width_q, transition.acceleration_disturbance)
                flow = self.encloser.enclose_relative(request, target, model, clock)
                record.update(flow_status=flow.status.value, flow_fingerprint=flow.fingerprint)
                if flow.proven and _disjoint(flow.endpoint, transition.relative_observed):
                    record.update(status='CERTIFIED_INCONSISTENT', reason='Complete relative endpoint set disjoint from observation')
                    rejected = True
                else:
                    record.update(status='CONSISTENT_OR_UNRESOLVED_RETAINED', reason=flow.reason)
                records.append(record)
                if rejected:
                    break
            (discarded if rejected else retained).append(model.fingerprint if rejected else model)
        return NonlinearMembershipResult(tuple(retained), tuple(discarded), tuple(records),
            'INCONSISTENT' if not retained else 'RETAINED')
