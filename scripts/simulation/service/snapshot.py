"""Coherent Python and explicit-input snapshots before worker execution."""

from __future__ import annotations

import shutil

from orbexa.layout import PACKAGE_ROOT
from orbexa.simulation.recording.artifacts import _sha, _write_json


def _freeze_job(settings, directory):
    """Capture a coherent per-job package and inputs before scientific imports."""
    from dataclasses import replace

    source = PACKAGE_ROOT
    destination = directory / "runtime" / "orbexa"

    def inventory():
        return {
            str(path.relative_to(source)): _sha(path)
            for path in source.rglob("*.py")
            if "__pycache__" not in path.parts
        }

    before = inventory()
    destination.mkdir(parents=True, exist_ok=False)
    for name in before:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    if inventory() != before or any(
        _sha(destination / name) != value for name, value in before.items()
    ):
        raise RuntimeError(
            "Source changed during job snapshot; retry with a coherent source"
        )
    inputs = directory / "inputs"
    inputs.mkdir()
    changes, input_records = {}, {}
    for name in (
        "config_path",
        "spacecraft_path",
    ):
        original = getattr(settings, name)
        frozen = inputs / (name + original.suffix)
        expected = _sha(original)
        shutil.copyfile(original, frozen)
        if _sha(original) != expected or _sha(frozen) != expected:
            raise RuntimeError("An explicit input changed during job snapshot")
        changes[name] = frozen
        input_records[name] = {
            "original": str(original),
            "frozen": str(frozen.relative_to(directory)),
            "sha256": expected,
        }
    _write_json(
        directory / "snapshot-manifest.json",
        {
            "package_files": before,
            "inputs": input_records,
            "package_scope": "Python runtime; viewer assets are exported separately for the selected UI port",
            "scope": "Per-job coherent Python/input snapshot; scientific modules execute only these frozen bytes",
        },
    )
    return replace(settings, **changes)
