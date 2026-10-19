"""Freeze coherent study source, explicit inputs, and runtime identities."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys


def hash_file(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def runtime_environment():
    """Read installed dependency identities without importing scientific code."""
    packages = {}
    for name in ('numpy', 'scipy', 'gekko', 'python-flint', 'PyYAML', 'zstandard'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    native = None
    try:
        distribution = importlib.metadata.distribution('gekko')
        executable = 'apm.exe' if os.name == 'nt' else 'apm_mac' if sys.platform == 'darwin' else 'apm'
        relative = 'gekko/bin/'+executable
        path = Path(distribution.locate_file(relative))
        if path.is_file():
            native = dict(distribution='gekko', path=relative, sha256=hash_file(path))
    except importlib.metadata.PackageNotFoundError:
        pass
    return dict(python=sys.version, platform=platform.platform(), packages=packages,
        native_gekko_executable=native,
        scope='Installed version and native binary identities before motion; unavailable metadata remains null')


class StudySourceFreezer:
    """Reserve one new output directory and bind its executable source bytes."""

    def __init__(self, package_source):
        self.package_source = Path(package_source)

    def freeze(self, request):
        directory = request.output_dir
        directory.mkdir(parents=True, exist_ok=False)
        source = self.package_source
        destination = directory/'runtime'/'orbexa'
        before = {path.relative_to(source).as_posix(): hash_file(path) for path in source.rglob('*.py')
                  if '__pycache__' not in path.parts}
        for name, expected in before.items():
            target = destination/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source/name, target)
            if hash_file(target) != expected:
                raise RuntimeError('Python source changed while freezing '+name)
        after = {path.relative_to(source).as_posix(): hash_file(path) for path in source.rglob('*.py')
                 if '__pycache__' not in path.parts}
        if before != after:
            raise RuntimeError('Source changed during snapshot; no arm was executed')
        inputs, frozen = {}, {}
        (directory/'inputs').mkdir()
        for name in ('config', 'spacecraft', 'study'):
            original = getattr(request, name)
            target = directory/'inputs'/(name+original.suffix)
            expected = hash_file(original)
            shutil.copyfile(original, target)
            if hash_file(original) != expected or hash_file(target) != expected:
                raise RuntimeError('Explicit '+name+' input changed during snapshot')
            inputs[name] = dict(original=str(original), frozen=str(target.relative_to(directory)), sha256=expected)
            frozen[name] = str(target)
        # This standard-library record intentionally precedes scientific imports.
        manifest = dict(schema='orbexa.nonlinear-study-source.v1', package_python_sha256=before, inputs=inputs,
            selected_case=request.case, adaptation=request.adaptation, bandwidth=request.bandwidth, steps=request.steps,
            runtime_environment=runtime_environment(),
            scope='One explicit predeclared arm; coherent source/input bytes before any motion; a cap is not completion')
        (directory/'source-manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        return destination, dict(**frozen, case=request.case, adaptation=request.adaptation, bandwidth=request.bandwidth,
                                 output_dir=str(directory), steps=request.steps)
