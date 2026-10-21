"""Shared native solver transcript retention for isolated study and live jobs."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from orbexa.simulation.recording.artifacts import _sha


def native_controller(directory, messages):
    """Retain native solver evidence inside the isolated job process only."""
    from orbexa.control.mpc.controller import MPCController
    from orbexa.solvers.gekko.tube import GekkoDynamicTube
    from orbexa.simulation.approach.records.checkpoint import plain
    import tempfile
    import gekko
    native = directory / 'native'; native.mkdir(exist_ok=True)
    tempfile.tempdir = str(native)
    # Preserve every local APMonitor working directory, including failed/cold
    # retry attempts. This changes file lifetime only, not the NLP or solver.
    gekko.GEKKO.cleanup = lambda _self: None
    original_solve = GekkoDynamicTube._solve_once
    attempts, calls = 0, 0
    counter_lock, current_call = threading.Lock(), threading.local()
    def retained_attempt(problem):
        nonlocal attempts
        with counter_lock:
            attempts += 1
            number = attempts
        problem.model.solver_options += ['output_file native-ipopt.txt', 'file_print_level 5']
        result = None
        try:
            result = original_solve(problem)
            return result
        finally:
            folder = Path(problem.model.path)
            record = dict(attempt=number, controller_call=getattr(current_call, 'number', None), working_directory=str(folder.relative_to(directory)),
                success=None if result is None else bool(result.success),
                message=None if result is None else result.message,
                solver_info=None if result is None else plain(result.solver_info),
                files={str(p.relative_to(folder)): _sha(p) for p in folder.rglob('*') if p.is_file()},
                scope='Unchanged native problem/acceptance; only transcript output and directory lifetime instrumented')
            with counter_lock:
                with (directory / 'native-attempts.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(record, allow_nan=False) + '\n')
    GekkoDynamicTube._solve_once = retained_attempt
    class TrackedController:
        solver_backend = 'gekko'
        def __init__(self, **kwargs):
            self.inner = MPCController(**kwargs)
        def solve_step(self, **kwargs):
            nonlocal calls
            with counter_lock:
                calls += 1
                number = calls
            current_call.number = number
            messages.put(dict(kind='native', message=f'Actual nonlinear controller call {number}: solving fleet member',
                native=dict(controller_call=number, anomaly=float(kwargs['start_anom']), status='solving')))
            result = self.inner.solve_step(**kwargs)
            messages.put(dict(kind='native', message=f'Nonlinear controller call {number}: '+result.message,
                native=dict(controller_call=number, anomaly=float(kwargs['start_anom']), status='accepted' if result.success else 'refused')))
            return result
    return TrackedController
