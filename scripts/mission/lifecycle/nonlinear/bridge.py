"""bridge.py: An explicit verified old-source approach boundary for a later completion.

This adapter consumes a trusted independently executed closed-arm audit. It
never edits an approach or rebrands its old arrival certificates as new ones.
Every bound artifact is rehashed before the physical boundary is used.

Copyright: Aaron John Sabu
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from pathlib import Path

from orbexa.common.serialization import plain
from orbexa.evidence.records.archive import semantic_sha256
from orbexa.models.dynamics.reference_frame import fingerprint


def _require(condition, message):
    if not condition:
        raise ValueError("Verified approach bridge: " + message)


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


_ISSUER = object()


@dataclass(frozen=True, init=False)
class VerifiedClosedApproachBoundary:
    """Conditional authorization from an exact-source independent audit.

    The audit's implementation is an explicit trusted input. Hashes alone are
    not a new differential-inclusion proof or a cryptographic signature.
    """

    record: dict
    guarded_files: tuple[tuple[str, str], ...]
    record_identity: str

    def __init__(self, record, guarded_files, record_identity, *, _issuer=None):
        _require(
            _issuer is _ISSUER,
            "construct this boundary from an independently passed receipt",
        )
        object.__setattr__(self, "record", copy.deepcopy(record))
        object.__setattr__(self, "guarded_files", tuple(guarded_files))
        object.__setattr__(self, "record_identity", record_identity)

    def verify(self, approach, *, model, clock, design, hold_records):
        """Recheck immutable provenance and the complete physical boundary."""
        _require(
            fingerprint(self.record) == self.record_identity, "bridge record changed"
        )
        for path, expected in self.guarded_files:
            _require(_sha(path) == expected, "a bound old-source artifact changed")
        boundary = self.record["boundary"]
        _require(
            semantic_sha256(vars(approach)) == boundary["source_approach_sha256"],
            "physical approach changed",
        )
        _require(
            fingerprint(model.as_dict()) == boundary["physical_model_fingerprint"],
            "physical model changed",
        )
        _require(
            fingerprint(clock.as_dict())
            == boundary["reference_clock_record_fingerprint"],
            "reference clock changed",
        )
        _require(
            fingerprint(design.as_dict()) == boundary["vehicle_design_fingerprint"],
            "vehicle design changed",
        )
        _require(
            getattr(hold_records, "identity", None)
            == self.record["old_archive_identity"],
            "full old journal changed",
        )
        from orbexa.mission.lifecycle.nonlinear.boundary import (
            capture_truth_from_nonlinear,
        )

        actual = capture_truth_from_nonlinear(approach, model=model, clock=clock)
        _require(
            fingerprint(plain(actual)) == boundary["capture_truth_fingerprint"],
            "actual physical endpoint changed",
        )
        return {
            "method": self.record["method"],
            "bridge_identity": self.record_identity,
            "old_source_arrival": copy.deepcopy(boundary["arrival_verification"]),
            "old_source_manifest_sha256": self.record["old_source_manifest_sha256"],
            "old_auditor_sha256": self.record["old_auditor_sha256"],
            "scope": self.record["scope"],
        }
