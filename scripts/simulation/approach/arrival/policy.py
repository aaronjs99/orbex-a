"""Finite arrival commitments without moving a receding-horizon deadline.

The retained nonlinear fleet candidate is numerical evidence, not an admitted
backup. Every actual hold still needs the ordinary shared continuous guard;
failure retains its prefix and never implies that a physical plant can pause.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from numbers import Integral, Real
import re

from orbexa.control.terminal.region import TerminalRegion, exact_scalar


def _fingerprint(value):
    from orbexa.simulation.approach.records.checkpoint import digest
    return digest(value)


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _sha(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must identify its exact source by SHA-256")
    return value


def exact_outer_step(config):
    return exact_scalar(config.anom_step / config.tube.feedback_substeps) * config.tube.feedback_substeps


def exact_mission_clock(scenario, state):
    guard = scenario.execution_guard
    if guard is None:
        raise ValueError("Arrival policy requires the exact shared clock of a continuous execution guard")
    if guard.exact_end is None:
        if state.anomaly != 0.:
            raise ValueError("A noninitial arrival state has no inherited exact guard clock")
        return Fraction(0)
    return exact_scalar(guard.exact_end)


@dataclass(frozen=True)
class ArrivalPolicyConfig:
    mode: str = "deadline_terminal_suffix"
    planning_fraction: float = 0.8
    required_terminal_region: TerminalRegion | None = None

    def __post_init__(self):
        if self.mode != "deadline_terminal_suffix":
            raise ValueError("Unsupported arrival policy mode")
        if (isinstance(self.planning_fraction, bool) or not isinstance(self.planning_fraction, Real)
                or not math.isfinite(float(self.planning_fraction)) or not 0 < self.planning_fraction < 1):
            raise ValueError("Arrival planning_fraction must be a finite real strictly between zero and one")
        object.__setattr__(self, "planning_fraction", float(self.planning_fraction))
        region = self.required_terminal_region
        if isinstance(region, dict):
            region = TerminalRegion.from_dict(region)
            object.__setattr__(self, "required_terminal_region", region)
        if region is not None and (not isinstance(region, TerminalRegion) or region.frame != "physical-lvlh-v1"):
            raise ValueError("Required arrival region must use the physical-lvlh-v1 state frame")

    def as_dict(self):
        return dict(mode=self.mode, planning_fraction=self.planning_fraction,
            required_terminal_region=None if self.required_terminal_region is None else self.required_terminal_region.as_dict())

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise TypeError("Arrival policy must be an explicit mapping")
        return cls(**value)

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class ArrivalCommitment:
    policy_fingerprint: str
    commitment_update: int
    deadline_update: int
    exact_start_anomaly: object
    exact_deadline_anomaly: object
    exact_outer_step: object
    candidate_fingerprint: str
    transition_identity: str | None = None
    schema: str = "orbexa.arrival_commitment.v1"

    def __post_init__(self):
        if self.schema != "orbexa.arrival_commitment.v1":
            raise ValueError("Unsupported arrival commitment schema")
        for name in ("policy_fingerprint", "candidate_fingerprint"):
            _sha(getattr(self, name), name)
        if self.transition_identity is not None:
            _sha(self.transition_identity, "transition_identity")
        for name in ("commitment_update", "deadline_update"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        for name in ("exact_start_anomaly", "exact_deadline_anomaly", "exact_outer_step"):
            object.__setattr__(self, name, exact_scalar(getattr(self, name)))
        if (self.deadline_update <= self.commitment_update or self.exact_outer_step <= 0
                or self.exact_deadline_anomaly != self.exact_start_anomaly
                    + (self.deadline_update - self.commitment_update) * self.exact_outer_step):
            raise ValueError("Arrival deadline, exact clock and update indices disagree")

    def as_dict(self):
        return {name: str(value) if isinstance(value, Fraction) else value for name, value in vars(self).items()}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise TypeError("Arrival commitment must be an explicit mapping")
        return cls(**value)

    def check_clock(self, update, anomaly, outer_step):
        update = _integer(update, "current arrival update")
        if (update < self.commitment_update or exact_scalar(outer_step) != self.exact_outer_step
                or exact_scalar(anomaly) != self.exact_start_anomaly
                    + (update - self.commitment_update) * self.exact_outer_step):
            raise ValueError("Arrival restart is not on its inherited exact outer grid; the deadline cannot be reset")


@dataclass(frozen=True)
class ArrivalPlanningStep:
    policy: ArrivalPolicyConfig
    current_update: int
    exact_anomaly: Fraction
    deadline_update: int
    exact_deadline_anomaly: Fraction
    exact_outer_step: Fraction
    terminal_start_index: int

    def __post_init__(self):
        if not isinstance(self.policy, ArrivalPolicyConfig):
            raise TypeError('Arrival planning requires its typed policy')
        for name in ('current_update', 'deadline_update', 'terminal_start_index'):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        for name in ('exact_anomaly', 'exact_deadline_anomaly', 'exact_outer_step'):
            object.__setattr__(self, name, exact_scalar(getattr(self, name)))
        if (self.exact_outer_step <= 0 or self.terminal_start_index != max(0, self.deadline_update - self.current_update)
                or self.exact_deadline_anomaly != self.exact_anomaly
                    + (self.deadline_update - self.current_update) * self.exact_outer_step):
            raise ValueError('Arrival planning suffix, deadline and exact clock disagree')

    def as_dict(self):
        return dict(current_update=self.current_update, exact_anomaly=str(self.exact_anomaly),
            deadline_update=self.deadline_update, exact_deadline_anomaly=str(self.exact_deadline_anomaly),
            exact_outer_step=str(self.exact_outer_step), terminal_start_index=self.terminal_start_index,
            policy_fingerprint=self.policy.fingerprint)


class ArrivalRuntime:
    """Own one immutable commitment, retained candidates and a separate event ledger."""

    def __init__(self, policy: ArrivalPolicyConfig, *, transition_identity=None,
                 enabled_at_update=0, exact_enabled_anomaly=0):
        if not isinstance(policy, ArrivalPolicyConfig):
            raise TypeError("Inject a typed ArrivalPolicyConfig")
        if transition_identity is not None:
            _sha(transition_identity, "transition_identity")
        self.policy, self.transition_identity = policy, transition_identity
        self.enabled_at_update = _integer(enabled_at_update, 'arrival enablement update')
        self.exact_enabled_anomaly = exact_scalar(exact_enabled_anomaly)
        self.commitment = None
        self.events = []
        self.committed_candidate = self.latest_candidate = None

    def planning_step(self, update, anomaly, outer_step, horizon):
        update = _integer(update, "current arrival update")
        horizon = _integer(horizon, "arrival horizon")
        q, h = exact_scalar(anomaly), exact_scalar(outer_step)
        if horizon < 2 or h <= 0:
            raise ValueError("Arrival requires a positive outer step and at least two prediction nodes")
        if (update < self.enabled_at_update or q != self.exact_enabled_anomaly
                + (update - self.enabled_at_update) * h):
            raise ValueError('Arrival planning moved away from its immutable enablement clock and outer grid')
        if self.commitment is None:
            deadline, end = update + horizon - 1, q + (horizon - 1) * h
        else:
            if self.commitment.policy_fingerprint != self.policy.fingerprint:
                raise ValueError("An arrival commitment cannot change its policy")
            self.commitment.check_clock(update, q, h)
            deadline, end = self.commitment.deadline_update, self.commitment.exact_deadline_anomaly
        index = max(0, deadline - update)
        if index >= horizon:
            raise ValueError("The fixed arrival deadline is outside the unchanged planning horizon")
        return ArrivalPlanningStep(self.policy, update, q, deadline, end, h, index)

    def accept_candidate(self, step, planning, *, expected_chaser_ids):
        from orbexa.simulation.approach.records.checkpoint import plain
        ids = tuple(expected_chaser_ids)
        if (not isinstance(step, ArrivalPlanningStep) or step.policy.fingerprint != self.policy.fingerprint
                or not ids or len(set(ids)) != len(ids) or tuple(planning.plans) != ids or not planning.success):
            raise ValueError("Only a complete numerically accepted fleet candidate can anchor arrival")
        if self.commitment is not None:
            self.commitment.check_clock(step.current_update, step.exact_anomaly, step.exact_outer_step)
        for plan in planning.plans.values():
            spec, info = plan.spec, plan.result.solver_info.get('terminal_suffix')
            if (not plan.result.success or not spec.enforce_terminal or spec.terminal_start_index != step.terminal_start_index
                    or spec.terminal_planning_fraction != self.policy.planning_fraction or not isinstance(info, dict)
                    or info.get('start_index') != step.terminal_start_index
                    or info.get('planning_fraction') != self.policy.planning_fraction
                    or [row.get('index') for row in info.get('checked_nodes', [])] != list(range(step.terminal_start_index, spec.horizon_steps))
                    or info.get('required_region') != (None if spec.terminal_region is None else spec.terminal_region.as_dict())):
                raise ValueError('Arrival candidate is missing the requested native whole-suffix readback')
        payload = plain(dict(step=step.as_dict(), parameters=planning.parameters,
            plans={key: dict(state_trajectory=plan.result.state_trajectory,
                control_trajectory=plan.result.control_trajectory, dynamic_tube=plan.tube,
                specification=plan.spec.to_extra_params(), planning_context=plan.arrival_context,
                solver_acceptance={name: value for name, value in plan.result.solver_info.items() if name != 'dynamic_tube'})
                for key, plan in planning.plans.items()}))
        fingerprint = _fingerprint(payload)  # rejects nonfinite retained data
        artifact = dict(fingerprint=fingerprint, payload=payload,
            scope="Retained numerical fleet candidate only; not an admitted held-feedback backup")
        if self.commitment is None:
            self.commitment = ArrivalCommitment(self.policy.fingerprint, step.current_update, step.deadline_update,
                step.exact_anomaly, step.exact_deadline_anomaly, step.exact_outer_step, fingerprint, self.transition_identity)
            self.committed_candidate = copy.deepcopy(artifact)
        elif (step.deadline_update != self.commitment.deadline_update
                or step.exact_deadline_anomaly != self.commitment.exact_deadline_anomaly):
            raise ValueError("An accepted arrival deadline is immutable")
        self.latest_candidate = artifact
        self.events.append(dict(**step.as_dict(), status="NUMERICAL_FLEET_CANDIDATE_ACCEPTED",
                                candidate_fingerprint=fingerprint))

    def refuse_candidate(self, step, messages):
        self.events.append(dict(**step.as_dict(), status="NUMERICAL_FLEET_CANDIDATE_REFUSED", messages=list(messages)))

    def record_execution(self, step, execution, exact_end):
        self.events.append(dict(**step.as_dict(), status="EXECUTION_ACCEPTED" if execution.success else "EXECUTION_STOPPED",
            committed=bool(execution.committed), exact_end_anomaly=str(exact_scalar(exact_end)),
            messages=list(execution.messages)))

    def as_metadata(self):
        return dict(schema="orbexa.arrival_state.v1", policy_fingerprint=self.policy.fingerprint,
            transition_identity=self.transition_identity,
            commitment=None if self.commitment is None else self.commitment.as_dict(),
            committed_candidate=copy.deepcopy(self.committed_candidate), latest_candidate=copy.deepcopy(self.latest_candidate),
            events=copy.deepcopy(self.events),
            scope="Fixed numerical arrival deadline; every actual hold still needs continuous admission; no retained backup theorem")

    def activation_marker(self):
        """Separate durable origin/first-acceptance binding for checkpoint reload."""
        return dict(schema='orbexa.arrival_activation.v1', policy_fingerprint=self.policy.fingerprint,
            enabled_at_update=self.enabled_at_update, exact_enabled_anomaly=str(self.exact_enabled_anomaly),
            transition_identity=self.transition_identity,
            commitment_fingerprint=None if self.commitment is None else _fingerprint(self.commitment.as_dict()))

    @classmethod
    def from_metadata(cls, policy, value, *, activation_marker=None):
        if (not isinstance(value, dict) or value.get("schema") != "orbexa.arrival_state.v1"
                or value.get("policy_fingerprint") != policy.fingerprint):
            raise ValueError("Enabled arrival requires its complete policy-bound persisted state")
        marker = activation_marker
        if (not isinstance(marker, dict) or marker.get('schema') != 'orbexa.arrival_activation.v1'
                or marker.get('policy_fingerprint') != policy.fingerprint):
            raise ValueError('Enabled arrival requires its separate durable activation marker')
        result = cls(policy, transition_identity=value.get("transition_identity"),
            enabled_at_update=marker['enabled_at_update'], exact_enabled_anomaly=marker['exact_enabled_anomaly'])
        if marker.get('transition_identity') != result.transition_identity:
            raise ValueError('Arrival activation and runtime changed their source transition')
        if marker['commitment_fingerprint'] is not None:
            _sha(marker['commitment_fingerprint'], 'activation commitment_fingerprint')
        result.commitment = None if value.get("commitment") is None else ArrivalCommitment.from_dict(value["commitment"])
        if result.commitment is not None and (result.commitment.policy_fingerprint != policy.fingerprint
                or result.commitment.transition_identity != result.transition_identity
                or result.commitment.commitment_update < result.enabled_at_update
                or result.commitment.exact_start_anomaly != result.exact_enabled_anomaly
                    + (result.commitment.commitment_update - result.enabled_at_update) * result.commitment.exact_outer_step):
            raise ValueError("Saved arrival commitment changed its policy or source transition")
        if marker != result.activation_marker():
            raise ValueError('Arrival activation lost or changed its original accepted commitment')
        result.events = copy.deepcopy(value["events"])
        if not isinstance(result.events, list) or any(not isinstance(event, dict) for event in result.events):
            raise ValueError("Arrival events must retain their ordered ledger")
        for name in ("committed_candidate", "latest_candidate"):
            artifact = copy.deepcopy(value[name])
            if artifact is not None and (not isinstance(artifact, dict)
                    or _fingerprint(artifact["payload"]) != artifact["fingerprint"]):
                raise ValueError("Retained arrival candidate identity does not match its data")
            setattr(result, name, artifact)
        if result.commitment is not None:
            artifact = result.committed_candidate
            if artifact is None or artifact["fingerprint"] != result.commitment.candidate_fingerprint:
                raise ValueError("The immutable arrival commitment lost its original fleet candidate")
            step = artifact["payload"]["step"]
            if (step["policy_fingerprint"] != policy.fingerprint
                    or step["current_update"] != result.commitment.commitment_update
                    or step["deadline_update"] != result.commitment.deadline_update
                    or exact_scalar(step["exact_anomaly"]) != result.commitment.exact_start_anomaly
                    or exact_scalar(step["exact_deadline_anomaly"]) != result.commitment.exact_deadline_anomaly
                    or exact_scalar(step["exact_outer_step"]) != result.commitment.exact_outer_step
                    or step['terminal_start_index'] != result.commitment.deadline_update - result.commitment.commitment_update):
                raise ValueError("Arrival commitment and original numerical candidate disagree")
            if result.latest_candidate is None:
                raise ValueError('An accepted arrival commitment lost its latest fleet candidate')
        elif result.committed_candidate is not None or result.latest_candidate is not None:
            raise ValueError("A retained arrival candidate requires its immutable deadline")
        result._validate_events()
        _fingerprint(result.as_metadata())  # finite JSON, without trusting serialized status flags
        return result

    def _saved_step(self, value):
        if value['policy_fingerprint'] != self.policy.fingerprint:
            raise ValueError('Arrival event or candidate changed its policy')
        step = ArrivalPlanningStep(self.policy, **{name: value[name] for name in (
            'current_update', 'exact_anomaly', 'deadline_update', 'exact_deadline_anomaly',
            'exact_outer_step', 'terminal_start_index')})
        if (step.current_update < self.enabled_at_update or step.exact_anomaly != self.exact_enabled_anomaly
                + (step.current_update - self.enabled_at_update) * step.exact_outer_step):
            raise ValueError('Arrival event or candidate changed its enablement clock/grid')
        return step

    def _validate_events(self):
        accepted, last_step, last_execution_end = [], None, None
        last_update = self.enabled_at_update
        for event in self.events:
            status = event.get('status')
            if status == 'POLICY_REFUSED':
                update = _integer(event['update'], 'refused arrival update')
                if update < last_update:
                    raise ValueError('Arrival event updates are not ordered')
                last_update = update
                continue
            if status == 'REQUIRED_ENDPOINT_REGION_SCREEN':
                if (not accepted or last_execution_end is None
                        or exact_scalar(event['exact_anomaly']) != last_execution_end):
                    raise ValueError('Arrival endpoint screen has no matching executed candidate')
                continue
            if status not in ('NUMERICAL_FLEET_CANDIDATE_ACCEPTED', 'NUMERICAL_FLEET_CANDIDATE_REFUSED',
                              'EXECUTION_ACCEPTED', 'EXECUTION_STOPPED'):
                raise ValueError('Unknown arrival event status')
            step = self._saved_step(event)
            if step.current_update < last_update:
                raise ValueError('Arrival event updates are not ordered')
            last_update = step.current_update
            if accepted:
                commitment = self.commitment
                commitment.check_clock(step.current_update, step.exact_anomaly, step.exact_outer_step)
                if (step.deadline_update != commitment.deadline_update
                        or step.exact_deadline_anomaly != commitment.exact_deadline_anomaly):
                    raise ValueError('Arrival event reset its accepted deadline')
            if status == 'NUMERICAL_FLEET_CANDIDATE_ACCEPTED':
                if self.commitment is None:
                    raise ValueError('Accepted arrival evidence cannot lose its immutable commitment')
                _sha(event['candidate_fingerprint'], 'accepted candidate_fingerprint')
                accepted.append(event)
                last_step, last_execution_end = step.as_dict(), None
            elif status.startswith('EXECUTION_'):
                if (not accepted or last_step != step.as_dict() or type(event.get('committed')) is not bool):
                    raise ValueError('Arrival execution has no matching accepted candidate')
                end = exact_scalar(event['exact_end_anomaly'])
                if not step.exact_anomaly <= end <= step.exact_anomaly + step.exact_outer_step:
                    raise ValueError('Arrival execution clock is outside its candidate interval')
                last_execution_end = end
        if self.commitment is None:
            return
        if not accepted:
            raise ValueError('The immutable arrival commitment lost its accepted event ledger')
        first, last = accepted[0], accepted[-1]
        original, latest = self.committed_candidate, self.latest_candidate
        for event, artifact in ((first, original), (last, latest)):
            if (event['candidate_fingerprint'] != artifact['fingerprint']
                    or self._saved_step(event).as_dict() != self._saved_step(artifact['payload']['step']).as_dict()):
                raise ValueError('Arrival accepted events and retained original/latest candidates disagree')
        commitment = self.commitment
        latest_step = self._saved_step(latest['payload']['step'])
        commitment.check_clock(latest_step.current_update, latest_step.exact_anomaly, latest_step.exact_outer_step)
        if (first['current_update'] != commitment.commitment_update
                or latest_step.deadline_update != commitment.deadline_update
                or latest_step.exact_deadline_anomaly != commitment.exact_deadline_anomaly):
            raise ValueError('Arrival accepted candidates changed their immutable deadline')
