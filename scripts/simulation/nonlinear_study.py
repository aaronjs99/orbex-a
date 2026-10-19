"""Freeze and execute one explicit prospective nonlinear study arm.

This lightweight entry point preserves the established public and private
imports. Request declarations, source freezing, records, metrics, and execution
live in focused study modules; numerical dependencies load only at execution.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from .study_metrics import collect_study_metrics as _metrics
from .study_records import write_record as _write
from .study_request import STUDY_SCHEMA, StudyArmRequest, load_declaration as _declaration
from .study_source import StudySourceFreezer, hash_file as _sha, runtime_environment as _runtime_environment


_ARM_STARTED = False


def _freeze(request):
    """Reserve a new directory and copy coherent Python/input source bytes."""
    return StudySourceFreezer(Path(__file__).resolve().parents[1]).freeze(request)


def run_study_arm(request: StudyArmRequest):
    """Freeze then execute one arm; never choose a case or compare winners."""
    global _ARM_STARTED
    if _ARM_STARTED:
        raise RuntimeError('Run each study arm in a separate process')
    if not isinstance(request, StudyArmRequest):
        raise TypeError('Supply StudyArmRequest')
    _declaration(request.study, request.case)
    destination, frozen = _freeze(request)
    _ARM_STARTED = True
    for name in tuple(sys.modules):
        if name == 'orbexa' or name.startswith('orbexa.'):
            del sys.modules[name]
    spec = importlib.util.spec_from_file_location('orbexa', destination/'__init__.py',
        submodule_search_locations=[str(destination)])
    package = importlib.util.module_from_spec(spec)
    sys.modules['orbexa'] = package
    spec.loader.exec_module(package)
    from orbexa.simulation.nonlinear_study import _execute_arm
    return _execute_arm(**frozen)


def _execute_arm(*, config, spacecraft, study, case, adaptation, bandwidth, output_dir, steps):
    global _ARM_STARTED
    if _ARM_STARTED:
        raise RuntimeError('Run each study arm in a separate process')
    _ARM_STARTED = True
    from .study_execution import StudyArmExecution
    return StudyArmExecution(config=config, spacecraft=spacecraft, study=study, case=case,
        adaptation=adaptation, bandwidth=bandwidth, output_dir=output_dir, steps=steps).run()
