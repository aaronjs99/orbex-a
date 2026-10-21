"""Immutable performance attempts, proposals, decisions and resource evidence."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping
from orbexa.control.certification.reachability import IntervalBox, _exact
from orbexa.control.fallback.refuge import _identity, exact_record, fingerprint
from orbexa.control.fallback.common import _pairs
from orbexa.control.fallback.geometry import FleetGeometryCertificate
from orbexa.control.fallback.plan import FleetFallbackPlan


def _freeze(value):
    if isinstance(value, Mapping):
        return tuple((str(key), _freeze(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (float, Fraction)):
        return _exact(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise TypeError("Attempt details must be immutable JSON-like evidence")


@dataclass(frozen=True)
class PerformanceAttempt:
    status: str
    message: str
    solver_id: str = "unspecified-performance-planner"
    details: Any = ()

    def __post_init__(self):
        _identity(self.status, "performance status")
        _identity(self.solver_id, "performance solver identity")
        if not isinstance(self.message, str):
            raise TypeError("Performance message must be text")
        object.__setattr__(self, "details", _freeze(self.details))


class PerformanceAdmissionRefusal(ValueError):
    """An admission failure with exact evidence independent of solver outcome."""
    def __init__(self, message, *, details=()):
        super().__init__(message)
        self.admission_details = _freeze(details)


@dataclass(frozen=True)
class PerformanceProposal:
    start: Any
    hold_width: Any
    applied_inputs: Any
    successor: FleetFallbackPlan
    observation_ids: Any
    context_ids: Any
    attempt: PerformanceAttempt

    def __post_init__(self):
        q, h = _exact(self.start), _exact(self.hold_width)
        if h <= 0:
            raise ValueError("A positive performance hold is required")
        object.__setattr__(self, "start", q)
        object.__setattr__(self, "hold_width", h)
        inputs = _pairs(self.applied_inputs, "performance held inputs")
        if any(not isinstance(box, IntervalBox) or box.dimension != 3 for _, box in inputs):
            raise ValueError("Performance inputs need applied three-input uncertainty boxes")
        object.__setattr__(self, "applied_inputs", inputs)
        object.__setattr__(self, "observation_ids", _pairs(self.observation_ids, "proposal observations"))
        object.__setattr__(self, "context_ids", _pairs(self.context_ids, "proposal contexts"))
        if not isinstance(self.successor, FleetFallbackPlan) or not isinstance(self.attempt, PerformanceAttempt):
            raise TypeError("Proposals require a prepared successor and retained attempt")

    @property
    def fingerprint(self):
        return fingerprint(dict(start=self.start, hold_width=self.hold_width, applied_inputs=self.applied_inputs,
            successor=self.successor.fingerprint, observation_ids=self.observation_ids,
            context_ids=self.context_ids, attempt=self.attempt))


@dataclass(frozen=True)
class HeldFleetDecision:
    start: Fraction
    hold_width: Fraction
    source: str
    reason: str
    commands: tuple[tuple[str, tuple[Fraction, ...]], ...]
    applied_inputs: tuple[tuple[str, IntervalBox], ...]
    attempt: PerformanceAttempt
    active_fallback_id: str
    successor_fallback_id: str
    context_ids: tuple[tuple[str, str], ...]
    observation_ids: tuple[tuple[str, str], ...]
    charged_fuel: tuple[tuple[str, Fraction], ...]
    proposal_id: str | None = None
    flow_fingerprints: tuple[str, ...] = ()
    geometry: FleetGeometryCertificate | None = None

    @property
    def fingerprint(self):
        return fingerprint(self)

    def float_commands(self):
        return {key: tuple(float(value) for value in command) for key, command in self.commands}

    def as_dict(self):
        return dict(identity=self.fingerprint, **exact_record(self))


@dataclass(frozen=True)
class CommandResourceBound:
    start: Any
    hold_width: Any
    applied_input: IntervalBox
    fuel_upper_kg: Any
    force_margin_n: Any
    fuel_domain_id: str
    proof_reference: str

    def __post_init__(self):
        for name in ("start", "hold_width", "fuel_upper_kg", "force_margin_n"):
            object.__setattr__(self, name, _exact(getattr(self, name)))
        if self.hold_width <= 0 or self.fuel_upper_kg < 0 or not isinstance(self.applied_input, IntervalBox):
            raise ValueError("A finite, nonnegative command resource proof is required")
        _identity(self.fuel_domain_id, "fuel domain identity")
        _identity(self.proof_reference, "command resource proof reference")
