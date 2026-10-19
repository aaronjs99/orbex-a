"""Atomic study records and acknowledged execution progress."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from fractions import Fraction
import json
from pathlib import Path
import threading
import time


def write_record(path, value):
    """Publish finite JSON only after its complete temporary record is written."""
    def convert(item):
        if is_dataclass(item):
            return asdict(item)
        if isinstance(item, (Path, Fraction)):
            return str(item)
        if callable(getattr(item, 'tolist', None)):
            return item.tolist()
        if callable(getattr(item, 'item', None)):
            return item.item()
        raise TypeError('Unsupported result value: '+type(item).__name__)

    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, default=convert, allow_nan=False, separators=(',', ':'))
        stream.write('\n')
    replace_deadline = time.monotonic()+2.0
    while True:
        try:
            temporary.replace(path)
            break
        except OSError as error:
            if getattr(error, 'winerror', None) not in (5, 32, 33):
                raise
            remaining = replace_deadline-time.monotonic()
            if remaining <= 0.0:
                raise
            time.sleep(min(0.02, remaining))
            if time.monotonic() >= replace_deadline:
                raise


class StudyTelemetry:
    """Append small events and publish progress after archive acknowledgment."""

    def __init__(self, directory, started):
        self.directory = Path(directory)
        self.started = started
        self.lock = threading.Lock()

    def put(self, value):
        # Native attempts have their own durable source transcript. Keep only
        # small progress events here, never trajectory or proof copies.
        with self.lock:
            with (self.directory/'progress.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(value, allow_nan=False)+'\n')

    def checkpoint(self, archive, update, runner, outcome):
        import numpy as np

        archive.flush()
        if archive.holds != len(runner.recorder.hold_records):
            raise ValueError('Durable archive count differs from the committed exact hold index')
        runner.executor.journal.clear()
        runner.executor.runtime.events.clear()
        phase = runner.recorder.arrays.sample_phases[-1]
        distance = max(float(np.linalg.norm(runner.state.centers[chaser.chaser_id][:3]-
            runner.preview.docking_state(chaser.docking_point_body, runner.state.anomaly)[:3])) for chaser in runner.chasers)
        progress = dict(update=update, exact_q=str(runner.truth_target.current_q),
            physical_time_s=runner.truth_target.physical_time_at(runner.truth_target.current_q),
            phase=phase, maximum_nominal_goal_distance_m=distance,
            performance_holds=sum(source == 'PERFORMANCE' for source in outcome.control_sources),
            fallback_holds=sum(source != 'PERFORMANCE' for source in outcome.control_sources),
            elapsed_wall_s=time.perf_counter()-self.started,
            completed_holds=archive.holds, archive_entries=archive.entries,
            retained_full_journal_holds=len(runner.executor.journal), retained_runtime_events=len(runner.executor.runtime.events))
        self.put({'kind': 'checkpoint', 'schema': 'orbexa.nonlinear-progress-checkpoint.v1',
                  'committed_progress': progress})
        write_record(self.directory/'progress.json', progress)

    def stage_completed(self, store, stage, result):
        store.record(stage, result)
        write_record(self.directory/'stage-progress.json',
                     dict(stage=stage, elapsed_wall_s=time.perf_counter()-self.started))
