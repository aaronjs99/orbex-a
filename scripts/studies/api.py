"""api.py: Freeze and execute one explicit prospective nonlinear study arm.

This lightweight entry point preserves the established public and private
imports. Request declarations, source freezing, records, metrics, and execution
live in focused study modules; numerical dependencies load only at execution.

Copyright: Aaron John Sabu
"""

from __future__ import annotations

import importlib.util
import sys

from orbexa.common.layout import PACKAGE_ROOT
from orbexa.evidence.source import (
    StudySourceFreezer,
)
from orbexa.studies.request import (
    StudyArmRequest,
)
from orbexa.studies.request import (
    load_declaration as _declaration,
)

_ARM_STARTED = False


def _freeze(request):
    """Reserve a new directory and copy coherent Python/input source bytes."""
    return StudySourceFreezer(PACKAGE_ROOT).freeze(request)


def run_study_arm(request: StudyArmRequest):
    """Freeze then execute one arm; never choose a case or compare winners."""
    global _ARM_STARTED
    if _ARM_STARTED:
        raise RuntimeError("Run each study arm in a separate process")
    if not isinstance(request, StudyArmRequest):
        raise TypeError("Supply StudyArmRequest")
    _declaration(request.study, request.case)
    destination, frozen = _freeze(request)
    _ARM_STARTED = True
    for name in tuple(sys.modules):
        if name == "orbexa" or name.startswith("orbexa."):
            del sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        "orbexa",
        destination / "__init__.py",
        submodule_search_locations=[str(destination)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules["orbexa"] = package
    spec.loader.exec_module(package)
    from orbexa.studies.api import _execute_arm

    return _execute_arm(**frozen)


def _execute_arm(
    *, config, spacecraft, study, case, adaptation, bandwidth, output_dir, steps
):
    """Run one study arm in this process and return its summary.

    Inputs:
        Keyword arguments naming the frozen config, spacecraft and study files, the
        case, adaptation, bandwidth, output directory and steps cap.
    Outputs:
        Returns the result of ``StudyArmExecution.run``. Raises RuntimeError if an
        arm has already started in this process; sets the module's started flag.
    """
    global _ARM_STARTED
    if _ARM_STARTED:
        raise RuntimeError("Run each study arm in a separate process")
    _ARM_STARTED = True
    from orbexa.studies.execution import StudyArmExecution

    return StudyArmExecution(
        config=config,
        spacecraft=spacecraft,
        study=study,
        case=case,
        adaptation=adaptation,
        bandwidth=bandwidth,
        output_dir=output_dir,
        steps=steps,
    ).run()
