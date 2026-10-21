"""One replaceable worker with retained jobs and monotone state revisions."""
from __future__ import annotations

import copy
import math
import multiprocessing as mp
import queue
import threading
import time
import uuid
from pathlib import Path
from orbexa.simulation.recording.artifacts import _write_json
from orbexa.simulation.service.contracts import LiveParameters, LiveServiceSettings
from orbexa.simulation.service.jobs.resume import _resume_source
from orbexa.simulation.service.process import _WorkerTree, _run_job
from orbexa.simulation.service.snapshot import _freeze_job


class LiveSimulationService:
    """One replaceable worker, retained job directories and monotone revisions."""

    def __init__(self, settings: LiveServiceSettings):
        if not isinstance(settings, LiveServiceSettings):
            raise TypeError("Inject LiveServiceSettings")
        self.settings = settings
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        self._condition = threading.Condition(threading.RLock())
        self._operation = threading.Lock()
        self._process = self._tree = self._messages = self._run_event = None
        self._resume_pending = settings.resume_path is not None
        self._state = dict(revision=0, session_id=uuid.uuid4().hex, job_id=None, status="idle", parameters=None,
            projection=None, lifecycle=None, progress={}, message="Ready for a new bounded simulation", paused=False,
            scope="Actual maintained nonlinear ADTMPC simulation under declared study-model assumptions; no flight qualification",
            execution_model='historical_analytical_resume' if self._resume_pending else 'sensed_nonlinear',
            resume_pending=self._resume_pending,
            resume_source=None if settings.resume_path is None else str(settings.resume_path))

    @property
    def worker_alive(self):
        return self._process is not None and self._process.is_alive()

    def state(self, *, after=None, timeout=0.):
        if after is not None and (isinstance(after, bool) or not isinstance(after, int) or not 0 <= after < 2**63):
            raise ValueError("after must be a nonnegative revision integer")
        if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 <= timeout <= 20:
            raise ValueError("Long-poll timeout must be between zero and20 seconds")
        with self._condition:
            self._condition.wait_for(lambda: after is None or self._state["revision"] > after, timeout=timeout)
            return copy.deepcopy(self._state)

    def _publish(self, **changes):
        with self._condition:
            self._state.update(changes)
            self._state["revision"] += 1
            snapshot = copy.deepcopy(self._state)
            if snapshot.get("output_dir"):
                _write_json(Path(snapshot["output_dir"]) / "job-status.json", {
                    key: value for key, value in snapshot.items() if key not in ("projection", "lifecycle")})
            self._condition.notify_all()
        return snapshot

    def _stop(self, message):
        if self._tree is not None:
            alive = self.worker_alive
            terminal = self._state['status'] in ('completed', 'stopped', 'failed')
            self._tree.close(cancel=alive)
            if alive and not terminal:
                self._publish(status="cancelled", message=message, paused=False)
        self._tree = self._process = self._run_event = None
        # The per-job collector owns and closes its queue after process exit.
        self._messages = None

    def start(self, parameters):
        parameters = parameters if isinstance(parameters, LiveParameters) else LiveParameters.from_dict(parameters)
        with self._operation:
            from dataclasses import replace
            if self._resume_pending:
                _resume_source(self.settings.resume_path, parameters)
            job_settings = self.settings if self._resume_pending else replace(self.settings, resume_path=None,
                historical_analytical_resume=False)
            self._stop("Replaced by an explicitly requested new parameter set; outputs preserved")
            job_id = time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:12]
            directory = self.settings.output_dir / job_id
            directory.mkdir(exist_ok=False)
            _write_json(directory / "parameters.json", parameters.as_dict())
            try:
                frozen_settings = _freeze_job(job_settings, directory)
            except (OSError, RuntimeError) as error:
                _write_json(directory / 'job-status.json', dict(job_id=job_id, status='failed',
                    message=str(error), worker_started=False))
                raise
            context = mp.get_context("spawn")
            messages, run_event, launch_event = context.Queue(8), context.Event(), context.Event()
            run_event.set()
            process = context.Process(target=_run_job, args=(frozen_settings, parameters,
                directory, messages, run_event, launch_event), name="orbexa-live-" + job_id)
            self._publish(job_id=job_id, output_dir=str(directory), status="starting", parameters=parameters.as_dict(),
                projection=None, lifecycle=None, progress={}, paused=False, success=None,
                execution_model='historical_analytical_resume' if frozen_settings.resume_path else 'sensed_nonlinear',
                message="Constructing the declared physical model, sensor prior and finite safety policy")
            try:
                process.start()
                tree = _WorkerTree(process)
            except BaseException:
                if process.pid is not None and process.is_alive():
                    process.terminate(); process.join(3)
                self._publish(status="failed", message="Worker process isolation failed before solver launch")
                raise
            self._process, self._tree, self._messages, self._run_event = process, tree, messages, run_event
            self._resume_pending = False
            self._publish(resume_pending=False, origin='historical_analytical_checkpoint' if frozen_settings.resume_path else 'fresh_nonlinear_parameters')
            threading.Thread(target=self._collect, args=(job_id, process, messages), daemon=True).start()
            launch_event.set()
            return self.state()

    def _collect(self, job_id, process, messages):
        terminal = False
        try:
            while True:
                try:
                    event = messages.get(timeout=.25)
                except queue.Empty:
                    if not process.is_alive():
                        break
                    continue
                with self._condition:
                    if self._state["job_id"] != job_id or self._state['status'] in ('cancelled', 'idle'):
                        return
                    paused = self._state["paused"]
                kind = event.get("kind")
                changes = {}
                if "projection" in event:
                    changes["projection"] = event["projection"]
                if "lifecycle" in event:
                    changes["lifecycle"] = event["lifecycle"]
                if kind in ("ready", "committed", "running"):
                    changes.update(status="pausing" if paused else "running",
                        message=event.get("message", "Actual committed simulation states"))
                elif kind == "paused":
                    changes.update(status="paused", message="Paused at a committed outer boundary")
                elif kind == "stage":
                    changes.update(status="completing", message=event.get("message", "Running bounded physical completion"))
                elif kind == "native":
                    changes.update(native=event.get('native', {}), message=event.get('message', 'Native nonlinear solve'))
                elif kind in ("finished", "error"):
                    terminal = True
                    changes.update(status="completed" if event.get("success") else "stopped" if kind == "finished" else "failed",
                        success=bool(event.get("success")), message=event.get("message", "Worker stopped"), paused=False)
                if "update" in event or "anomaly" in event:
                    changes["progress"] = {name: event[name] for name in ("update", "anomaly", "phase") if name in event}
                if changes:
                    self._publish(**changes)
        finally:
            with self._condition:
                current = self._state["job_id"] == job_id
                status = self._state["status"]
            if current and not terminal and status not in ("cancelled", "idle"):
                self._publish(status="failed", paused=False,
                    message=f"Worker exited without a closed completion result (exit {process.exitcode}); committed output retained")
            messages.close()

    def pause(self, paused=None):
        with self._operation:
            if not self.worker_alive or self._run_event is None:
                raise RuntimeError("There is no running job to pause")
            if paused is None:
                paused = not self._state["paused"]
            if not isinstance(paused, bool):
                raise ValueError("paused must be a boolean")
            self._run_event.clear() if paused else self._run_event.set()
            return self._publish(paused=paused, status="pausing" if paused else "running",
                message="Pause requested at next safe outer boundary" if paused else "Resumed actual simulation")

    def reset(self):
        with self._operation:
            self._stop("Reset requested; worker/solver descendants terminated and outputs retained")
            return self._publish(job_id=None, output_dir=None, status="idle", parameters=None,
                projection=None, lifecycle=None, progress={}, paused=False, success=None,
                message="Reset; retained job directories are unchanged")

    def close(self):
        with self._operation:
            self._stop("Loopback service closed; outputs preserved")
