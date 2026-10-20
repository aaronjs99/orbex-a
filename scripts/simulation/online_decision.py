"""Prepared evidence for an already admitted command; no proof gate is skipped.

The canonical decision hash/evidence is expensive because it includes the full
geometry proof. This optional view prepares it once and checks the complete
live data graph before reuse. Exact rational scalars are copied as integers;
container links and dataclass field schemas are checked. Mutation rebuilds the
original record. Unknown data/code semantics use the original uncached path.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, is_dataclass
from fractions import Fraction
import hashlib
import json
import weakref

from orbexa.control import refuge
from orbexa.control.fleet_fallback import FleetAdmissionRuntime, HeldFleetDecision
from orbexa.control.reachability import IntervalBox

_EXACT_RECORD = refuge.exact_record
_FINGERPRINT = refuge.fingerprint
_DECISION_FINGERPRINT = HeldFleetDecision.fingerprint
_DECISION_RECORD = HeldFleetDecision.as_dict
_MISSING = object()


def _same_code():
    return (refuge.exact_record is _EXACT_RECORD and refuge.fingerprint is _FINGERPRINT
        and HeldFleetDecision.fingerprint is _DECISION_FINGERPRINT
        and HeldFleetDecision.as_dict is _DECISION_RECORD)


class _GraphGuard:
    """Snapshot exact data, not a hash cached under object identity alone."""
    def __init__(self, value):
        self.nodes, self.classes, self.seen = [], {}, set()
        self._visit(value, root=True)

    def _visit(self, value, *, root=False):
        if id(value) in self.seen:
            return
        self.seen.add(id(value))
        kind = type(value)
        if kind is Fraction:
            self.nodes.append(('fraction', value, value.numerator, value.denominator))
            return
        if kind in (str, int, float, bool, type(None)):
            return
        if kind is IntervalBox:
            names = ('lower', 'upper')
        elif (is_dataclass(value) and kind.__module__.startswith('orbexa.')
                and kind.__getattribute__ is object.__getattribute__):
            names = tuple(field.name for field in fields(value))
            self.classes[kind] = names
        elif kind in (tuple, list):
            items = tuple(value)
            self.nodes.append(('sequence', value, kind, items))
            for item in items:
                self._visit(item)
            return
        elif kind is dict:
            if any(type(key) is not str for key in value):
                raise TypeError('Prepared decision records require ordinary string map keys')
            items = tuple(value.items())
            self.nodes.append(('mapping', value, items))
            for _, item in items:
                self._visit(item)
            return
        else:
            raise TypeError('Unknown data/serialization semantics require the original decision path')
        items = tuple((name, getattr(value, name)) for name in names)
        self.nodes.append(('root_attributes' if root else 'attributes',
            weakref.ref(value) if root else value, kind, items))
        for _, item in items:
            self._visit(item)

    def valid(self):
        if not _same_code() or any(tuple(field.name for field in fields(kind)) != names
                for kind, names in self.classes.items()):
            return False
        for entry in self.nodes:
            tag, value, *data = entry
            if tag == 'root_attributes':
                value, tag = value(), 'attributes'
                if value is None:
                    return False
            if tag == 'fraction':
                numerator, denominator = data
                if (type(value) is not Fraction or type(value.numerator) is not int
                        or type(value.denominator) is not int or value.numerator != numerator
                        or value.denominator != denominator):
                    return False
            elif tag == 'attributes':
                kind, items = data
                if type(value) is not kind or any(getattr(value, name, _MISSING) is not item
                        for name, item in items):
                    return False
            elif tag == 'sequence':
                kind, items = data
                if (type(value) is not kind or len(value) != len(items)
                        or any(a is not b for a, b in zip(value, items))):
                    return False
            elif tag == 'mapping':
                items, = data
                if (type(value) is not dict or len(value) != len(items)
                        or any(value.get(key, _MISSING) is not item for key, item in items)):
                    return False
        return True


class _PreparedRecord:
    def __init__(self, decision):
        # The view owns this record; a weak reverse link avoids retaining one
        # complete decision/guard cycle per hold until a garbage collection.
        self.decision, self.guard = weakref.ref(decision), None
        self._rebuild()

    def _rebuild(self):
        # Build privately before publication. A failure cannot replace a valid
        # decision by stale data or partially prepared evidence.
        decision = self.decision()
        if decision is None:
            raise ValueError('The prepared decision owner no longer exists')
        record = _EXACT_RECORD(decision)
        wire = json.dumps(record, sort_keys=True, separators=(',', ':'),
            allow_nan=False).encode('utf-8')
        guard = _GraphGuard(decision)
        self.record, self.identity, self.guard = record, hashlib.sha256(wire).hexdigest(), guard

    def ready(self):
        if not _same_code():
            return False
        if self.guard is None or not self.guard.valid():
            try:
                self._rebuild()
            except TypeError:
                self.guard = None
                return False
        return True


class PreparedHeldFleetDecision(HeldFleetDecision):
    """Same exact decision fields and identity, with guarded record preparation.

    Instances are created only by prepare_pending_decision. The original
    runtime remains the sole owner of admission, current observations, clock,
    complete-hold support, fuel debit and endpoint commit.
    """
    @property
    def fingerprint(self):
        prepared = self._prepared_record
        if prepared.ready():
            return prepared.identity
        return HeldFleetDecision.fingerprint.fget(self)

    def as_dict(self):
        prepared = self._prepared_record
        if prepared.ready():
            return dict(identity=prepared.identity, **deepcopy(prepared.record))
        return HeldFleetDecision.as_dict(self)


def prepare_pending_decision(runtime, decision):
    """Replace only the pending value by an equivalent view, without advancing.

    Unknown payload semantics retain the original decision. The original
    fingerprints are compared before publishing the prepared value. No proof,
    resource or sensor result is invented by this record optimization.
    """
    if (not isinstance(runtime, FleetAdmissionRuntime) or not isinstance(decision, HeldFleetDecision)
            or runtime._pending is None or runtime._pending[0] is not decision):
        raise ValueError('Prepare only the exact already admitted current pending decision')
    if type(runtime) is not FleetAdmissionRuntime or type(decision) is not HeldFleetDecision or not _same_code():
        return decision
    prepared = object.__new__(PreparedHeldFleetDecision)
    for field in fields(HeldFleetDecision):
        object.__setattr__(prepared, field.name, getattr(decision, field.name))
    try:
        record = _PreparedRecord(prepared)
    except TypeError:
        return decision
    object.__setattr__(prepared, '_prepared_record', record)
    if prepared.fingerprint != decision.fingerprint:
        raise ValueError('Prepared decision differs from its original canonical identity')
    _, successor, endpoints = runtime._pending
    runtime._pending = (prepared, successor, endpoints)
    return prepared
