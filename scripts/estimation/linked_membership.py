"""Bounded nonlinear set membership with shared affine-actuator hypotheses.

A branch contains a *fixed* matrix and bias for one agent over the retained
window. Interval propagation may overapproximate that dependency, but never
replaces a covering branch by a sampled point. Only complete disjointness can
remove a branch; every unevaluated or unresolved branch remains possible.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from fractions import Fraction

from orbexa.control.actuation import CommandDependentActuationPrior
from orbexa.control.nonlinear_flow import NonlinearFlowEncloser
from orbexa.control.reachability import HeldInputRequest, IntervalBox
from orbexa.core.nonlinear_relative import NonlinearModelBounds
from orbexa.core.reference_frame import exact, fingerprint
from orbexa.estimation.navigation import intersect
from orbexa.estimation.nonlinear_membership import (
    NonlinearMembershipContractor, NonlinearMembershipResult,
    NonlinearTransitionObservation,
)

METHOD = 'linked-affine-window-v1'
PERSISTENCE = 'fixed-affine-per-agent-v1'


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(name+' must be a positive integer')
    return value


@dataclass(frozen=True)
class SharedAffineActuationCell:
    """Twelve exact coordinates: row-major matrix, followed by fixed bias.

    The initial box covers every realization admitted by the existing
    ``CommandDependentActuationPrior.validate_realization`` check. It is a
    deliberately larger set than proper rotation times diagonal gain. The
    original prospective held-input support is intersected at every use.
    """
    parameters: IntervalBox
    prior_fingerprint: str

    def __post_init__(self):
        if self.parameters.dimension != 12 or not self.prior_fingerprint:
            raise ValueError('A shared affine cell needs twelve coordinates and its prior identity')

    @classmethod
    def from_prior(cls, prior):
        if type(prior) is not CommandDependentActuationPrior:
            raise TypeError('Use the supported declared affine actuation prior')
        residual = prior.tilt_rad*(1+prior.gain_error)+prior.round_relative
        lower, upper = [], []
        for row in range(3):
            for column in range(3):
                center = int(row == column)
                radius = residual+(prior.gain_error if row == column else 0)
                lower.append(center-radius)
                upper.append(center+radius)
        lower.extend((-prior.bias_abs,)*3)
        upper.extend((prior.bias_abs,)*3)
        return cls(IntervalBox(tuple(lower), tuple(upper)), prior.fingerprint)

    def as_dict(self):
        return dict(parameters=self.parameters.as_dict(), prior_fingerprint=self.prior_fingerprint,
            coordinates='row-major 3x3 matrix, then three fixed biases',
            persistence=PERSISTENCE)

    @property
    def fingerprint(self):
        return fingerprint(self.as_dict())

    def applied_input(self, command, original_support):
        values = tuple(map(exact, command))
        if len(values) != 3 or original_support.dimension != 3:
            raise ValueError('A shared actuator requires three held command components')
        # Production observations record the binary64 command actually passed
        # to apply(). A rational command rounded only here would need its own
        # conversion error and must not silently use the following bound.
        if any(exact(float(value)) != value for value in values):
            raise ValueError('Shared membership requires the actual binary64 delivered command')
        gamma = Fraction(16, 2**53)/(1-Fraction(16, 2**53))
        tiny = Fraction(16, 2**1074)*(1+3*max(map(abs, values)))
        lower, upper = [], []
        for row in range(3):
            lo = self.parameters.lower[9+row]
            hi = self.parameters.upper[9+row]
            magnitude = max(abs(lo), abs(hi))
            for column, value in enumerate(values):
                index = 3*row+column
                a, b = self.parameters.lower[index], self.parameters.upper[index]
                lo += min(a*value, b*value)
                hi += max(a*value, b*value)
                magnitude += max(abs(a), abs(b))*abs(value)
            rounding = gamma*magnitude+tiny
            lower.append(lo-rounding)
            upper.append(hi+rounding)
        return intersect(IntervalBox(tuple(lower), tuple(upper)), original_support)

    def split(self, commands):
        """Bisect one influential coordinate; both closed children cover it."""
        limits = tuple(max(abs(exact(command[j])) for command in commands) for j in range(3))
        weights = limits*3+(Fraction(1),)*3
        scores = tuple((hi-lo)*weight for lo, hi, weight in
            zip(self.parameters.lower, self.parameters.upper, weights))
        index = max(range(12), key=scores.__getitem__)
        if scores[index] == 0:
            return ()
        midpoint = (self.parameters.lower[index]+self.parameters.upper[index])/2
        left_upper, right_lower = list(self.parameters.upper), list(self.parameters.lower)
        left_upper[index] = right_lower[index] = midpoint
        return (type(self)(IntervalBox(self.parameters.lower, tuple(left_upper)), self.prior_fingerprint),
                type(self)(IntervalBox(tuple(right_lower), self.parameters.upper), self.prior_fingerprint))


class LinkedObservationWindow:
    """Bounded chronological history; no truth, sensor draw or point fit enters."""
    def __init__(self, agent_ids, holds_per_agent):
        self.holds_per_agent = _positive_integer(holds_per_agent, 'Window length')
        self.history = {str(agent): deque(maxlen=holds_per_agent) for agent in agent_ids}
        if not self.history:
            raise ValueError('Declare the finite observing fleet')

    def append(self, transitions):
        for item in transitions:
            if not isinstance(item, NonlinearTransitionObservation) or item.agent_id not in self.history:
                raise ValueError('Linked observations must identify a declared fleet agent')
            history = self.history[item.agent_id]
            if history and item == history[-1]:
                continue
            if history and item.start_q[0] < history[-1].start_q[0]+history[-1].width_q[0]:
                raise ValueError('Linked observation history must not overlap or move backward')
            history.append(item)

    def transitions(self):
        return tuple(item for history in self.history.values() for item in history)


class _FlowBudget:
    def __init__(self, maximum):
        self.maximum, self.used, self.cell_evaluations = maximum, 0, 0

    def take(self):
        if self.used >= self.maximum:
            return False
        self.used += 1
        return True

    def take_cell(self):
        if self.cell_evaluations >= self.maximum or self.used >= self.maximum:
            return False
        self.cell_evaluations += 1
        return True


class LinkedNonlinearMembershipContractor:
    """Cover-and-prune estimator with finite window, branches and flow budget.

    State and target/time endpoints are carried and intersected at contiguous
    observations. A gap restarts the state enclosure, while the same nuisance
    cell remains in force. Surviving cells persist between calls, conditioned
    on their physical-model candidate. No finite resolution guarantees useful
    identification, especially with the mission's weak drag signal.
    """
    def __init__(self, encloser=None, *, max_boxes=64, max_nuisance_cells=2,
                 max_flow_evaluations_per_candidate=256, max_agents=8):
        self.encloser = encloser or NonlinearFlowEncloser()
        self.max_boxes = _positive_integer(max_boxes, 'Candidate budget')
        self.max_nuisance_cells = _positive_integer(max_nuisance_cells, 'Nuisance cell budget')
        self.max_flow_evaluations_per_candidate = _positive_integer(
            max_flow_evaluations_per_candidate, 'Flow budget')
        self.max_agents = _positive_integer(max_agents, 'Fleet budget')
        self.agent_ids = None
        self.surviving_cells = {}

    @staticmethod
    def _supported(transitions):
        for item in transitions:
            if (not item.agent_id or item.commanded_input is None or
                    type(item.actuation_prior) is not CommandDependentActuationPrior or
                    item.actuation_persistence != PERSISTENCE or
                    item.start_q[0] != item.start_q[1] or item.width_q[0] != item.width_q[1] or
                    item.width_q[0] <= 0):
                return False
        return True

    def _target_sequence(self, model, transitions, clock, budget, records):
        result, endpoint, time_endpoint, previous_end = [], None, None, None
        for item in transitions:
            initial, initial_time = item.target_initial, item.physical_time_initial
            if not isinstance(initial_time, IntervalBox):
                initial_time = IntervalBox((initial_time[0],), (initial_time[1],))
            if previous_end == item.start_q[0] and endpoint is not None:
                initial = intersect(endpoint, initial)
                initial_time = intersect(time_endpoint, initial_time)
                if initial is None or initial_time is None:
                    records.append(dict(status='CERTIFIED_INCONSISTENT', observation_id=item.observation_id,
                        reason='Linked target/time endpoint and next initial observation are disjoint'))
                    return None, True
            if not budget.take():
                return None, False
            flow = self.encloser.enclose_target(initial, item.start_q, item.width_q, model, clock, initial_time)
            record = dict(observation_id=item.observation_id, target_status=flow.status.value,
                target_fingerprint=flow.fingerprint, linked_target_initial=initial.as_dict(),
                linked_time_initial=initial_time.as_dict())
            endpoint = intersect(flow.endpoint, item.target_observed) if (
                flow.proven and item.target_observed is not None) else flow.endpoint if flow.proven else None
            if flow.proven and endpoint is None:
                record.update(status='CERTIFIED_INCONSISTENT', reason='Complete linked target endpoint disjoint from observation')
                records.append(record)
                return None, True
            record['status'] = 'TARGET_LINK_RETAINED' if flow.proven else 'UNRESOLVED_RETAINED'
            records.append(record)
            time_endpoint = flow.physical_time_endpoint if flow.proven else None
            previous_end = item.start_q[0]+item.width_q[0]
            result.append(flow if flow.proven else None)
        return tuple(result), False

    def _evaluate_cell(self, cell, model, transitions, target_flows, clock, budget, records):
        endpoint, previous_end = None, None
        for item, target in zip(transitions, target_flows):
            initial = item.relative_initial
            if previous_end == item.start_q[0] and endpoint is not None:
                initial = intersect(endpoint, initial)
                if initial is None:
                    records.append(dict(nuisance_cell=cell.fingerprint, observation_id=item.observation_id,
                        status='CERTIFIED_INCONSISTENT', reason='Linked relative endpoint and next initial observation are disjoint'))
                    return True
            applied = cell.applied_input(item.commanded_input, item.held_input)
            if applied is None:
                records.append(dict(nuisance_cell=cell.fingerprint, observation_id=item.observation_id,
                    status='CERTIFIED_INCONSISTENT', reason='Shared actuator cell cannot produce admitted held-input support'))
                return True
            previous_end = item.start_q[0]+item.width_q[0]
            if target is None:
                endpoint = None
                continue
            if not budget.take():
                return False
            request = HeldInputRequest(initial, applied, item.start_q, item.width_q, item.acceleration_disturbance)
            flow = self.encloser.enclose_relative(request, target, model, clock)
            endpoint = intersect(flow.endpoint, item.relative_observed) if flow.proven else None
            rejected = flow.proven and endpoint is None
            records.append(dict(nuisance_cell=cell.fingerprint, observation_id=item.observation_id,
                flow_status=flow.status.value, flow_fingerprint=flow.fingerprint,
                status='CERTIFIED_INCONSISTENT' if rejected else 'CONSISTENT_OR_UNRESOLVED_RETAINED',
                linked_initial=initial.as_dict(), shared_input=applied.as_dict(),
                reason='Complete linked relative endpoint disjoint from observation' if rejected else flow.reason))
            if rejected:
                return True
        return False

    def contract(self, candidate_models, transitions, observations=None, clock=None):
        models, transitions = tuple(candidate_models), tuple(transitions)
        if clock is None or not models or not all(isinstance(m, NonlinearModelBounds) for m in models):
            raise ValueError('Supply typed candidate models and the immutable reference clock')
        if observations is not None or not all(isinstance(t, NonlinearTransitionObservation) for t in transitions):
            raise TypeError('Observations must be bound to typed held-input transitions')
        if not self._supported(transitions):
            legacy = NonlinearMembershipContractor(self.encloser, self.max_boxes).contract(models, transitions, clock=clock)
            return NonlinearMembershipResult(legacy.retained_models, legacy.discarded_fingerprints,
                (dict(method=METHOD, status='INDEPENDENT_FALLBACK',
                    reason='Missing exact clock, agent, command or explicit fixed affine declaration'),)+legacy.records, legacy.status)
        grouped = {}
        for item in transitions:
            grouped.setdefault(item.agent_id, []).append(item)
        if len(grouped) > self.max_agents:
            raise ValueError('Linked observations exceed the declared finite fleet budget')
        if grouped and self.agent_ids is None:
            self.agent_ids = frozenset(grouped)
        if self.agent_ids is not None and not frozenset(grouped).issubset(self.agent_ids):
            raise ValueError('A persistent actuator hypothesis cannot silently change its observing fleet')
        retained, discarded, records = [], [], []
        for index, model in enumerate(models):
            candidate_record = dict(method=METHOD, candidate=model.fingerprint, agents=[],
                maximum_flow_evaluations=self.max_flow_evaluations_per_candidate,
                maximum_nuisance_cells_per_agent=self.max_nuisance_cells)
            records.append(candidate_record)
            if index >= self.max_boxes:
                retained.append(model)
                candidate_record['status'] = 'UNRESOLVED_BUDGET'
                continue
            budget, rejected = _FlowBudget(self.max_flow_evaluations_per_candidate), False
            for agent, history in grouped.items():
                history.sort(key=lambda t: t.start_q[0])
                prior_ids = {t.actuation_prior.fingerprint for t in history}
                if len(prior_ids) != 1 or any(a.start_q[0]+a.width_q[0] > b.start_q[0] for a, b in zip(history, history[1:])):
                    raise ValueError('A linked window requires one immutable actuator prior and nonoverlapping observations')
                key = (model.fingerprint, agent, history[0].actuation_prior.fingerprint)
                cells = self.surviving_cells.get(key, (SharedAffineActuationCell.from_prior(history[0].actuation_prior),))
                entry = dict(agent_id=agent, observation_ids=[t.observation_id for t in history],
                    nuisance_cells_before=[cell.as_dict() for cell in cells], records=[])
                candidate_record['agents'].append(entry)
                targets, target_rejected = self._target_sequence(model, history, clock, budget, entry['records'])
                if target_rejected:
                    rejected = True
                    entry['status'] = 'CERTIFIED_INCONSISTENT'
                    break
                if targets is None:
                    entry['status'] = 'UNRESOLVED_BUDGET'
                    break
                pending, survivors = deque(cells), []
                while pending:
                    cell = pending.popleft()
                    if not budget.take_cell():
                        survivors.extend((cell, *pending))
                        pending.clear()
                        break
                    inconsistent = self._evaluate_cell(cell, model, history, targets, clock, budget, entry['records'])
                    if inconsistent:
                        continue
                    children = cell.split(tuple(t.commanded_input for t in history))
                    if children and len(survivors)+len(pending)+2 <= self.max_nuisance_cells and budget.used < budget.maximum:
                        pending.extend(children)
                    else:
                        survivors.append(cell)
                self.surviving_cells[key] = tuple(survivors)
                entry['nuisance_cells_after'] = [cell.as_dict() for cell in survivors]
                entry['status'] = 'CONSISTENT_OR_UNRESOLVED_RETAINED' if survivors else 'CERTIFIED_INCONSISTENT'
                if not survivors:
                    rejected = True
                    break
            candidate_record.update(status='CERTIFIED_INCONSISTENT' if rejected else 'CONSISTENT_OR_UNRESOLVED_RETAINED',
                flow_evaluations=budget.used, nuisance_cell_evaluations=budget.cell_evaluations,
                budget_exhausted=budget.used >= budget.maximum or budget.cell_evaluations >= budget.maximum)
            (discarded if rejected else retained).append(model.fingerprint if rejected else model)
        active_ids = {model.fingerprint for model in retained}
        self.surviving_cells = {key: cells for key, cells in self.surviving_cells.items() if key[0] in active_ids}
        return NonlinearMembershipResult(tuple(retained), tuple(discarded), tuple(records),
            'RETAINED' if retained else 'INCONSISTENT')
