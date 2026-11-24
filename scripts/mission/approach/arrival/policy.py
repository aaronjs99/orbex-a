"""policy.py: Finite arrival commitments without moving a receding-horizon deadline.

The retained nonlinear fleet candidate is numerical evidence, not an admitted
backup. Every actual hold still needs the ordinary shared continuous guard;
failure retains its prefix and never implies that a physical plant can pause.

Copyright: Aaron John Sabu
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral, Real

from orbexa.control.terminal.region import TerminalRegion, exact_scalar


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _sha(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must identify its exact source by SHA-256")
    return value


def exact_outer_step(config):
    return (
        exact_scalar(config.anom_step / config.tube.feedback_substeps)
        * config.tube.feedback_substeps
    )


@dataclass(frozen=True)
class ArrivalPolicyConfig:
    mode: str = "deadline_terminal_suffix"
    planning_fraction: float = 0.8
    required_terminal_region: TerminalRegion | None = None

    def __post_init__(self):
        if self.mode != "deadline_terminal_suffix":
            raise ValueError("Unsupported arrival policy mode")
        if (
            isinstance(self.planning_fraction, bool)
            or not isinstance(self.planning_fraction, Real)
            or not math.isfinite(float(self.planning_fraction))
            or not 0 < self.planning_fraction < 1
        ):
            raise ValueError(
                "Arrival planning_fraction must be a finite real strictly between zero and one"
            )
        object.__setattr__(self, "planning_fraction", float(self.planning_fraction))
        region = self.required_terminal_region
        if isinstance(region, dict):
            region = TerminalRegion.from_dict(region)
            object.__setattr__(self, "required_terminal_region", region)
        if region is not None and (
            not isinstance(region, TerminalRegion) or region.frame != "physical-lvlh-v1"
        ):
            raise ValueError(
                "Required arrival region must use the physical-lvlh-v1 state frame"
            )

    def as_dict(self):
        return {
            "mode": self.mode,
            "planning_fraction": self.planning_fraction,
            "required_terminal_region": None
            if self.required_terminal_region is None
            else self.required_terminal_region.as_dict(),
        }

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise TypeError("Arrival policy must be an explicit mapping")
        return cls(**value)

    @property
    def fingerprint(self):
        return hashlib.sha256(
            json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


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
        for name in (
            "exact_start_anomaly",
            "exact_deadline_anomaly",
            "exact_outer_step",
        ):
            object.__setattr__(self, name, exact_scalar(getattr(self, name)))
        if (
            self.deadline_update <= self.commitment_update
            or self.exact_outer_step <= 0
            or self.exact_deadline_anomaly
            != self.exact_start_anomaly
            + (self.deadline_update - self.commitment_update) * self.exact_outer_step
        ):
            raise ValueError(
                "Arrival deadline, exact clock and update indices disagree"
            )

    def as_dict(self):
        return {
            name: str(value) if isinstance(value, Fraction) else value
            for name, value in vars(self).items()
        }

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise TypeError("Arrival commitment must be an explicit mapping")
        return cls(**value)


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
            raise TypeError("Arrival planning requires its typed policy")
        for name in ("current_update", "deadline_update", "terminal_start_index"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        for name in ("exact_anomaly", "exact_deadline_anomaly", "exact_outer_step"):
            object.__setattr__(self, name, exact_scalar(getattr(self, name)))
        if (
            self.exact_outer_step <= 0
            or self.terminal_start_index
            != max(0, self.deadline_update - self.current_update)
            or self.exact_deadline_anomaly
            != self.exact_anomaly
            + (self.deadline_update - self.current_update) * self.exact_outer_step
        ):
            raise ValueError(
                "Arrival planning suffix, deadline and exact clock disagree"
            )

    def as_dict(self):
        return {
            "current_update": self.current_update,
            "exact_anomaly": str(self.exact_anomaly),
            "deadline_update": self.deadline_update,
            "exact_deadline_anomaly": str(self.exact_deadline_anomaly),
            "exact_outer_step": str(self.exact_outer_step),
            "terminal_start_index": self.terminal_start_index,
            "policy_fingerprint": self.policy.fingerprint,
        }
