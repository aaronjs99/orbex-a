"""Execution evidence and failures, independent of the execution coordinator."""
from __future__ import annotations

from dataclasses import dataclass, field
from orbexa.control.certification.reachability import IntervalBox
from orbexa.estimation.navigation.boxes import midpoint
from orbexa.simulation.approach.execution import ExecutionStep

@dataclass(frozen=True)
class BlackoutNavigationEstimate:
    exact_event_q: object
    physical_time_box: IntervalBox
    relative_state_box_by_agent: object
    clock_fingerprint: str
    model_fingerprint: str
    status: str = 'BLACKOUT_CONDITIONAL'
    reason: str = 'Finite admitted feedback/observation propagation; target ephemeris not freshly enclosed'
    target_ephemeris_box: object = None

    @property
    def point_relative_by_agent(self):
        return {key:midpoint(box) for key,box in self.relative_state_box_by_agent.items()}


@dataclass
class NonlinearExecutionStep(ExecutionStep):
    control_sources: list = field(default_factory=list)
    actual_truth_relative: dict = field(default_factory=dict)
    navigation_estimate: object = None
    membership_transitions: list = field(default_factory=list)
    journal: list = field(default_factory=list)
    used_fallback: bool = False
    delivered_packets: list = field(default_factory=list)
    archive_events: list = field(default_factory=list)
    actual_committed_q: object = None
    failure_event: object = None


class NonlinearExecutionFailure(RuntimeError):
    """Failed run evidence, including any already applied physical suffix."""
    def __init__(self,message,*,outcome,event,staged_archive,actual_committed_q,
                 actual_truth_relative,target_state):
        super().__init__(message)
        self.outcome,self.event,self.staged_archive=outcome,event,staged_archive
        self.actual_committed_q=actual_committed_q
        self.actual_truth_relative=actual_truth_relative
        self.target_state=target_state
