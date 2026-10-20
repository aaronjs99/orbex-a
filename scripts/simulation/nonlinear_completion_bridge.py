"""An explicit verified old-source approach boundary for a later completion.

This adapter consumes a trusted independently executed closed-arm audit. It
never edits an approach or rebrands its old arrival certificates as new ones.
Every bound artifact is rehashed before the physical boundary is used.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path

from orbexa.core.reference_frame import fingerprint
from orbexa.simulation.mission_checkpoint import plain
from orbexa.simulation.nonlinear_archive import semantic_sha256


def _require(condition, message):
    if not condition:
        raise ValueError('Verified approach bridge: '+message)


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
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
        _require(_issuer is _ISSUER, 'construct this boundary from an independently passed receipt')
        object.__setattr__(self, 'record', copy.deepcopy(record))
        object.__setattr__(self, 'guarded_files', tuple(guarded_files))
        object.__setattr__(self, 'record_identity', record_identity)

    @classmethod
    def from_receipt(cls, receipt_path, *, arm_directory, auditor_source):
        receipt_path, root = Path(receipt_path).resolve(), Path(arm_directory).resolve()
        report = json.loads(receipt_path.read_text(encoding='utf-8'))
        _require(report.get('schema') == 'orbexa.closed-mission-verification.v2'
                 and report.get('verification_status') == 'VERIFIED', 'a passed v2 independent audit is required')
        _require(report.get('auditor_sha256') == _sha(auditor_source), 'independent audit implementation changed')
        _require(Path(report['arm_directory']).resolve() == root, 'audit belongs to another arm')
        boundary = copy.deepcopy(report.get('approach_boundary'))
        _require(isinstance(boundary, dict) and boundary.get('arrival_verified') is True,
                 'a genuinely successful old-source arrival is required')
        _require(boundary.get('complete_outer_endpoint') is True,
                 'a partial outer update cannot enter physical completion')
        _require(report['archive_ledger'].get('unconfirmed_tail') is False,
                 'unconfirmed physical tail cannot enter completion')
        _require(report.get('all_verified_artifacts_stable') is True,
                 'the audited artifacts were not stable')
        guards = [(str(receipt_path), _sha(receipt_path)), (str(Path(auditor_source).resolve()), _sha(auditor_source))]
        for relative, expected in report['artifact_sha256'].items():
            path = (root/relative).resolve()
            _require(path.is_relative_to(root), 'an audit artifact escapes its arm directory')
            _require(_sha(path) == expected, 'audited artifact changed: '+relative)
            guards.append((str(path), expected))
        request = report['producer_request']
        closure = report['producer_closure']
        for value in (request, closure):
            path = Path(value['path']).resolve()
            _require(_sha(path) == value['sha256'], 'producer request or closure changed')
            guards.append((str(path), value['sha256']))
        source_manifest = json.loads((root/'source-manifest.json').read_text(encoding='utf-8'))
        producer_request = json.loads(Path(request['path']).read_text(encoding='utf-8'))
        old_source = Path(producer_request['source']).resolve()
        for relative, expected in source_manifest['package_python_sha256'].items():
            for folder in (old_source, root/'runtime/orbexa'):
                path = (folder/relative).resolve()
                _require(path.is_relative_to(folder.resolve()) and _sha(path) == expected,
                         'old source or its runtime package changed: '+relative)
                guards.append((str(path), expected))
        record = dict(method='verified-old-source-approach-to-new-completion-v1',
            original_arm=str(root), old_auditor_sha256=report['auditor_sha256'],
            old_source_manifest_sha256=report['source_inputs']['source_manifest_sha256'],
            old_study_manifest_sha256=report['source_inputs']['study_manifest_sha256'],
            old_archive_identity=report['archive_ledger']['identity'], boundary=boundary,
            scope='Unchanged actual endpoint/model/clock/design and complete old-source arrival; later contact/assembly implementation is separately source-bound')
        return cls(record, tuple(guards), fingerprint(record), _issuer=_ISSUER)

    def verify(self, approach, *, model, clock, design, hold_records):
        """Recheck immutable provenance and the complete physical boundary."""
        _require(fingerprint(self.record) == self.record_identity, 'bridge record changed')
        for path, expected in self.guarded_files:
            _require(_sha(path) == expected, 'a bound old-source artifact changed')
        boundary = self.record['boundary']
        _require(semantic_sha256(vars(approach)) == boundary['source_approach_sha256'], 'physical approach changed')
        _require(fingerprint(model.as_dict()) == boundary['physical_model_fingerprint'], 'physical model changed')
        _require(fingerprint(clock.as_dict()) == boundary['reference_clock_record_fingerprint'], 'reference clock changed')
        _require(fingerprint(design.as_dict()) == boundary['vehicle_design_fingerprint'], 'vehicle design changed')
        _require(getattr(hold_records, 'identity', None) == self.record['old_archive_identity'], 'full old journal changed')
        from orbexa.simulation.nonlinear_completion_boundary import capture_truth_from_nonlinear
        actual = capture_truth_from_nonlinear(approach, model=model, clock=clock)
        _require(fingerprint(plain(actual)) == boundary['capture_truth_fingerprint'], 'actual physical endpoint changed')
        return dict(method=self.record['method'], bridge_identity=self.record_identity,
            old_source_arrival=copy.deepcopy(boundary['arrival_verification']),
            old_source_manifest_sha256=self.record['old_source_manifest_sha256'],
            old_auditor_sha256=self.record['old_auditor_sha256'],
            scope=self.record['scope'])
