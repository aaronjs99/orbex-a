"""Explicit, prospective inputs for one nonlinear study arm.

This declaration layer uses only the standard library so CLI help and request
validation do not load numerical or solver dependencies.
"""
from __future__ import annotations
from orbexa.layout import PACKAGE_ROOT

from dataclasses import dataclass
import json
from pathlib import Path


STUDY_SCHEMA = 'orbexa.nonlinear-study.v1'


@dataclass(frozen=True)
class StudyArmRequest:
    config: Path
    spacecraft: Path
    study: Path
    case: str
    adaptation: str
    bandwidth: str
    output_dir: Path
    steps: int | None = None

    def __post_init__(self):
        for name in ('config', 'spacecraft', 'study'):
            path = Path(getattr(self, name)).expanduser().resolve()
            if not path.is_file():
                raise ValueError('Missing explicit '+name+' input: '+str(path))
            object.__setattr__(self, name, path)
        output = Path(self.output_dir).expanduser().resolve()
        if output.is_relative_to(PACKAGE_ROOT.parent):
            raise ValueError('Study outputs must be outside the maintained checkout')
        if not self.case or self.adaptation not in ('adaptive', 'frozen') or self.bandwidth not in ('dynamic', 'fixed'):
            raise ValueError('Select an explicit case and adaptive/frozen by dynamic/fixed arm')
        if self.steps is not None and (isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 1):
            raise ValueError('steps is a positive execution cap, independent of the prospective reference domain')
        object.__setattr__(self, 'output_dir', output)


def load_declaration(path, selected):
    """Require one selected case in a prospectively declared study."""
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    allowed = {'schema', 'study_id', 'options', 'cases', 'initial_target_spin_prior', 'description'}
    if (not isinstance(value, dict) or set(value)-allowed or value.get('schema') != STUDY_SCHEMA or
            not isinstance(value.get('study_id'), str) or not value['study_id'].strip() or
            not isinstance(value.get('options'), dict) or not isinstance(value.get('cases'), list)):
        raise ValueError('Study JSON requires schema, study_id, explicit options and a predeclared cases list')
    ids = [case.get('case_id') for case in value['cases'] if isinstance(case, dict)]
    if len(ids) != len(value['cases']) or not ids or len(set(ids)) != len(ids) or selected not in ids:
        raise ValueError('The explicit selected case must occur exactly once in the predeclared study')
    return value
