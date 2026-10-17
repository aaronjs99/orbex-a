"""Bounded loopback control of one real, source-recorded ORBEX-A worker.

The HTTP process never integrates a spacecraft or fabricates a trajectory.
Only the maintained mission runner's committed progress is exposed as motion.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import contextlib
import copy
import ctypes
import gzip
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import queue
import shutil
import signal
import sys
import threading
import time
from urllib.parse import parse_qs, urlsplit
import uuid


LIVE_SERVICE = 'orbexa-live-v2'
LIVE_PROTOCOL_VERSION = 2
ALLOWED_ORIGINS = frozenset((
    "http://localhost:8879", "http://127.0.0.1:8879",
    "http://localhost:8881", "http://127.0.0.1:8881",
    "https://orbex-a-mission-studio.aaronjs99.chatgpt.site",
))
PARAMETER_LIMITS = dict(release_distance_m=(100., 1000.),
    cluster_spacing_m=(1.4, 3.), target_spin_deg_s=(.2, 3.))


@dataclass(frozen=True)
class LiveParameters:
    release_distance_m: float = 300.
    cluster_spacing_m: float = 1.8
    target_spin_deg_s: float = 2.4

    def __post_init__(self):
        for name, (lower, upper) in PARAMETER_LIMITS.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"{name} must be a finite number in [{lower}, {upper}]")
            object.__setattr__(self, name, float(value))

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) - set(PARAMETER_LIMITS):
            raise ValueError("Supply only release_distance_m, cluster_spacing_m and target_spin_deg_s")
        return cls(**value)

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class LiveServiceSettings:
    output_dir: Path
    config_path: Path
    prior_path: Path
    admission_path: Path
    deployment_path: Path
    spacecraft_path: Path
    maximum_updates: int = 1200
    resume_path: Path | None = None
    historical_analytical_resume: bool = False

    def __post_init__(self):
        for name in ("output_dir", "config_path", "prior_path", "admission_path", "deployment_path", "spacecraft_path"):
            path = Path(getattr(self, name)).expanduser().resolve()
            if name != "output_dir" and not path.is_file():
                raise ValueError(f"Missing explicit {name}: {path}")
            object.__setattr__(self, name, path)
        if self.resume_path is not None:
            path = Path(self.resume_path).expanduser().resolve()
            if not path.is_file():
                raise ValueError(f"Missing explicit resume record: {path}")
            object.__setattr__(self, 'resume_path', path)
        if not isinstance(self.historical_analytical_resume, bool) or bool(self.resume_path) != self.historical_analytical_resume:
            raise ValueError('Saved analytical resume requires the explicit historical_analytical_resume flag and a resume path')
        if isinstance(self.maximum_updates, bool) or not isinstance(self.maximum_updates, int) or not 1 <= self.maximum_updates <= 1200:
            raise ValueError("maximum_updates must be an integer in [1, 1200]")
        checkout = Path(__file__).resolve().parents[2]
        if (checkout / "pyproject.toml").is_file() and self.output_dir.is_relative_to(checkout):
            raise ValueError("Live outputs must be outside the maintained checkout")


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _resume_source(path, parameters=None):
    """Read only the explicitly selected local record and its adjacent journal."""
    path = Path(path).resolve()
    payload = json.loads(path.read_bytes())
    record = payload.get('result', payload)
    try:
        deployment = record['metadata']['scenario']['deployment']
        options = deployment['options']
        center = options['carrier_center_lvlh_m']
        pitch = deployment['initial_pitch_m']
        spin = record['target']['initial_body_angular_velocity_rad_s']
        if (record['mission'] != 'multi' or record['metadata'].get('mission_complete')
                or record['target']['shape'] != 'composite_spacecraft'
                or deployment['kind'] != 'carrier_cluster_to_component_aware_patch_routes'
                or len(center) != 3 or center[:2] != [0., 0.]
                or len(spin) != 3 or spin[:2] != [0., 0.]
                or record['metadata'].get('model_version') == 'nonlinear-reference-relative-v1'
                or record['metadata'].get('continuous_execution_certificate', {}).get('method') != 'arb-picard-continuous-v1'):
            raise ValueError('Resume requires the same unfinished eight-chaser carrier deployment')
        if parameters is not None and (center[2] != parameters.release_distance_m
                or pitch != parameters.cluster_spacing_m
                or spin[2] != math.radians(parameters.target_spin_deg_s)):
            raise ValueError('This launcher must first resume the saved distance, spacing and spin; restart without --resume to use different knobs')
        # A raw result is either beside its journal or in the runner's approach/
        # directory. Never search elsewhere or infer a journal from a checkout.
        candidates = [path.parent / 'continuous-certificates.jsonl.gz']
        if path.parent.name == 'approach':
            candidates.append(path.parent.parent / 'continuous-certificates.jsonl.gz')
        journals = [candidate for candidate in candidates if candidate.is_file()]
        if len(journals) != 1:
            raise ValueError('Resume requires one complete adjacent continuous-certificates.jsonl.gz journal')
        return record, journals[0]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError('Resume record lacks the explicit carrier/target/source declarations') from error


def _freeze_job(settings, directory):
    """Capture a coherent per-job package and inputs before scientific imports."""
    from dataclasses import replace
    source = Path(__file__).resolve().parents[1]
    destination = directory / 'runtime' / 'orbexa'
    def inventory():
        return {str(path.relative_to(source)): _sha(path) for path in source.rglob('*.py')
            if '__pycache__' not in path.parts}
    before = inventory()
    destination.mkdir(parents=True,exist_ok=False)
    for name in before:
        target=destination/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/name,target)
    if inventory() != before or any(_sha(destination / name) != value for name, value in before.items()):
        raise RuntimeError('Source changed during job snapshot; retry with a coherent source')
    inputs = directory / 'inputs'; inputs.mkdir()
    changes, input_records = {}, {}
    for name in ('config_path', 'prior_path', 'admission_path', 'deployment_path', 'spacecraft_path'):
        original = getattr(settings, name)
        frozen = inputs / (name + original.suffix)
        expected = _sha(original)
        shutil.copyfile(original, frozen)
        if _sha(original) != expected or _sha(frozen) != expected:
            raise RuntimeError('An explicit input changed during job snapshot')
        changes[name] = frozen
        input_records[name] = dict(original=str(original), frozen=str(frozen.relative_to(directory)), sha256=expected)
    if settings.resume_path is not None:
        _record, source_journal = _resume_source(settings.resume_path)
        for name, original, filename in (('resume_path', settings.resume_path, 'resume-mission_data.json'),
                ('resume_journal', source_journal, 'continuous-certificates.jsonl.gz')):
            frozen = inputs / filename
            expected = _sha(original)
            shutil.copyfile(original, frozen)
            if _sha(original) != expected or _sha(frozen) != expected:
                raise RuntimeError('A resume source changed during job snapshot')
            if name == 'resume_path':
                changes[name] = frozen
            input_records[name] = dict(original=str(original), frozen=str(frozen.relative_to(directory)), sha256=expected)
    _write_json(directory / 'snapshot-manifest.json', dict(package_files=before, inputs=input_records,
        package_scope='Python runtime; viewer assets are exported separately for the selected UI port',
        scope='Per-job coherent Python/input snapshot; scientific modules execute only these frozen bytes'))
    return replace(settings, **changes)


def _append_resume_journal(checkpoint, path, destination):
    """Bind the complete saved prefix before appending newly executed holds.

    This authenticates the retained record/clock/enclosure linkage. The public
    propellant accountant separately reproduces every flow and swept witness
    before consuming the combined journal for a smaller fuel allowance.
    """
    from orbexa.control.reachability import IntervalBox, OrbitalModelBounds
    from orbexa.simulation.certified_execution import CertifiedExecutionGuard
    from orbexa.simulation.mission_checkpoint import digest
    record = checkpoint.source_record
    guard = record['metadata']['continuous_execution_certificate']
    ids = tuple(record['actual_trajectories'])
    q, end = Fraction(record['anom_history'][0]), Fraction(guard['exact_end_anomaly'])
    previous, observed, sample_index, last_context, last_width = {}, 0, 1, None, None
    def require(condition, reason):
        if not condition:
            raise ValueError('Resume journal refused: ' + reason)
    def box(value, size):
        result = IntervalBox(tuple(map(Fraction, value['exact_lower'])), tuple(map(Fraction, value['exact_upper'])))
        require(result.dimension == size, 'box dimension changed')
        return result
    model = OrbitalModelBounds(**guard['model_bounds'])
    disturbance = IntervalBox.from_center_radius([0.]*3, guard['acceleration_radius'])
    inputs = IntervalBox(guard['input_lower'], guard['input_upper'])
    require(len(ids) == 8 and guard['method'] == 'arb-picard-continuous-v1'
        and guard['rejected_holds'] == 0 and guard['start_anomaly'] == float(q), 'missing complete admitted prefix')
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            require(row.get('admitted') is True and tuple(row['chaser_ids']) == ids
                and set(row['flows']) == set(ids), 'refused or incomplete fleet hold')
            q0, q1, width = Fraction(row['exact_start_anomaly']), Fraction(row['exact_end_anomaly']), Fraction(row['width'])
            require(q0 == q and q1 == q0 + width and width > 0 and q1 <= end
                and abs(float(q0)-row['anomaly']) <= 1e-10, 'noncontiguous exact hold clock')
            next_boxes = {}
            for key in ids:
                flow = row['flows'][key]
                request = flow['request']
                initial = box(request['initial_state'], 6)
                require(flow['status'] == 'PROVEN' and flow['method'] == guard['method']
                    and OrbitalModelBounds(**flow['model_bounds']) == model
                    and tuple(map(Fraction, request['start_anomaly'])) == (q0, q0)
                    and tuple(map(Fraction, request['hold_width'])) == (width, width)
                    and box(request['acceleration_disturbance'], 3) == disturbance
                    and inputs.contains(box(request['held_input'], 3)), 'flow/model/actuator request changed')
                if previous:
                    require(previous[key].contains(initial), 'request leaves preceding endpoint')
                else:
                    require(initial == IntervalBox.from_center_radius(record['actual_trajectories'][key][0],
                        guard['measurement_radius']), 'release measurement differs from source')
                next_boxes[key] = box(flow['endpoint'], 6)
            require(row['collision']['status'] == 'PROVEN_SAFE'
                and CertifiedExecutionGuard._shape_identity(row['target_bound_context'])
                == CertifiedExecutionGuard._shape_identity(guard['target_bound_context']), 'swept target admission changed')
            if sample_index < len(record['anom_history']) and abs(float(q1)-record['anom_history'][sample_index]) <= 1e-10:
                require(all(next_boxes[key].contains(IntervalBox.point(record['actual_trajectories'][key][sample_index]))
                    for key in ids), 'endpoint misses source sample')
                sample_index += 1
            elif sample_index < len(record['anom_history']):
                require(float(q1) < record['anom_history'][sample_index]-1e-10, 'source sample skipped')
            destination.write(line if line.endswith('\n') else line + '\n')
            q, previous, observed = q1, next_boxes, observed+1
            last_context, last_width = row['target_bound_context'], width
    require(observed == guard['admitted_holds'] and q == end and sample_index == len(record['anom_history'])
        and last_width == Fraction(guard['last_hold_width'])
        and digest(last_context) == digest(guard['target_bound_context'])
        and all(previous[key] == box(guard['endpoint_boxes'][key], 6) for key in ids), 'incomplete final clock/enclosures/count')
    return dict(admitted_holds=observed, committed_updates=checkpoint.committed_updates,
        exact_start_anomaly=str(Fraction(record['anom_history'][0])), exact_end_anomaly=str(q),
        scope='Unchanged source journal prefix; clock/request/sample/endpoint linkage checked, not newly executed or relabeled')


class _WorkerTree:
    """Windows job lifetime includes APMonitor descendants, not other jobs."""

    def __init__(self, process):
        self.process, self.handle, self.kernel = process, None, None
        if os.name != "nt":
            return
        from ctypes import wintypes as w
        class Basic(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                ("flags", w.DWORD), ("min_ws", ctypes.c_size_t), ("max_ws", ctypes.c_size_t),
                ("active", w.DWORD), ("affinity", ctypes.c_size_t),
                ("priority", w.DWORD), ("scheduling", w.DWORD)]
        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in
                ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]
        class Extended(ctypes.Structure):
            _fields_ = [("basic", Basic), ("io", IO), ("process_mem", ctypes.c_size_t),
                ("job_mem", ctypes.c_size_t), ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.argtypes, kernel.CreateJobObjectW.restype = [ctypes.c_void_p, w.LPCWSTR], w.HANDLE
        kernel.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
        kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        kernel.OpenProcess.argtypes, kernel.OpenProcess.restype = [w.DWORD, w.BOOL, w.DWORD], w.HANDLE
        kernel.TerminateJobObject.argtypes = [w.HANDLE, w.UINT]
        kernel.CloseHandle.argtypes = [w.HANDLE]
        handle = kernel.CreateJobObjectW(None, None)
        information = Extended()
        information.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        child = kernel.OpenProcess(0x0101, False, process.pid)
        try:
            if not handle or not child or not kernel.SetInformationJobObject(handle, 9, ctypes.byref(information), ctypes.sizeof(information)) or not kernel.AssignProcessToJobObject(handle, child):
                raise OSError(ctypes.get_last_error(), "Cannot isolate the live solver process tree")
        except BaseException:
            if handle:
                kernel.CloseHandle(handle)
            raise
        finally:
            if child:
                kernel.CloseHandle(child)
        self.handle, self.kernel = handle, kernel

    def close(self, *, cancel=False):
        if self.handle is not None:
            if cancel:
                self.kernel.TerminateJobObject(self.handle, 1)
            self.kernel.CloseHandle(self.handle)
            self.handle = None
        elif cancel and self.process.is_alive():
            if os.name != "nt":
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    self.process.terminate()
            else:
                self.process.terminate()
        self.process.join(3)
        if cancel and self.process.is_alive():
            self.process.kill()
            self.process.join(3)


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


def _arrival_policy(profile, prior, admission, directory):
    """Derive final-port query errors for this job's exact spin and domain.

    The global controller may refuse; only its globally derived reference jets
    are used here as a planning/entry screen, never as terminal certification.
    """
    from orbexa.control.composite_terminal import CompositePhysicalTerminalBoundsBuilder
    from orbexa.control.physical_terminal import PhysicalTerminalErrorBounds, _encode
    from orbexa.control.reachability import IntervalBox, OrbitalModelBounds
    from orbexa.simulation.arrival_policy import ArrivalPolicyConfig
    target, cfg = profile.target, profile.config
    chasers = profile.scenario.fleet.chasers("multi")
    ports = tuple(tuple(c.docking_point_body) for c in chasers)
    index = max(range(len(ports)), key=lambda i: sum(Fraction(float(v))**2 for v in ports[i]))
    common = dict(mu=target.mu, semi_major_axis=target.semi_major_axis)
    orbit = OrbitalModelBounds(eccentricity=(target.eccentricity,)*2, alpha=(target.alpha,)*2, beta=(0, 0), **common)
    plant = OrbitalModelBounds(**prior.feasible_sets, **common)
    nominal = OrbitalModelBounds(**{key: (value, value) for key, value in prior.estimates.items()}, **common)
    errors = PhysicalTerminalErrorBounds(sensing_error_abs=admission.measurement_radius,
        reference_state_error_abs=(1e-6,)*3+(1e-4,)*3, anomaly_evaluation_error_abs=1e-10,
        declaration="Caller-declared uniform direct reference evaluation errors1e-6m/1e-4m per rad, sensor bounds from admission and initial-spin containment1e-8rad/s. Arb derives only finite-domain query-displacement inflation. No flight metrology or DOP853 tolerance certificate.")
    builder = CompositePhysicalTerminalBoundsBuilder(start_anomaly=target.anomaly_domain[0],
        end_anomaly=target.anomaly_domain[1], target_orbit=orbit, plant_parameters=plant, nominal_parameters=nominal,
        inertia=target.inertia, entry_spin=IntervalBox.from_center_radius(target.initial_body_angular_velocity, (1e-8,)*3),
        body_ports=ports, port_index=index, errors=errors, reference_id="live-global-final-port-errors",
        target_geometry=target.geometry, body_radius=profile.options.body_radius_m,
        maximum_hold=cfg.anom_step/cfg.tube.feedback_substeps,
        input_limits=IntervalBox(tuple(b['lower'] for b in cfg.mpc.input_bounds), tuple(b['upper'] for b in cfg.mpc.input_bounds)),
        bandwidth_candidates=(80,), precision_bits=192)
    result = builder.build()
    if not result.derived or "effective_reference_state_error_abs" not in result.derived:
        raise ValueError("Spin-specific global reference-error derivation refused: " + result.reason)
    rounded = []
    for value in result.derived["effective_reference_state_error_abs"]:
        number = float(value)
        if Fraction(number) < value:
            number = math.nextafter(number, math.inf)
        if not math.isfinite(number) or Fraction(number) < value:
            raise ValueError("Cannot represent the outward reference-error bound")
        rounded.append(number)
    payload = dict(mode="deadline_terminal_suffix", planning_fraction=.8, required_terminal_region=dict(
        omega=[.00115]*3, phi=[.02]*3, lambda_gain=[20]*3,
        reference_state_error_abs=rounded, maximum_query_displacement=1e-10,
        anomaly_domain=list(target.anomaly_domain), frame="physical-lvlh-v1",
        error_source=errors.declaration + " Full-domain rigid-body energy/orbital Arb bounds at the longest exact-norm final COM port dominate all eight port query jets; the global physical-controller certificate is not claimed."))
    _write_json(directory / "arrival-policy.json", payload)
    with gzip.open(directory / "reference-error-derivation.json.gz", "wt", encoding="utf-8") as stream:
        json.dump(_encode(dict(builder=vars(builder), result=result.as_dict(), selected_port=index,
            all_ports=ports, scope="Global final-port planning error bound only; controller refusal retained")), stream, allow_nan=False)
    return ArrivalPolicyConfig.from_dict(payload)


def _run_job(settings, parameters, directory, messages, run_event, launch_event):
    if os.name != "nt":
        os.setsid()
    launch_event.wait()
    with (directory / "worker.log").open("a", encoding="utf-8", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            frozen = directory / 'runtime' / 'orbexa'
            for name in tuple(sys.modules):
                if name == 'orbexa' or name.startswith('orbexa.'):
                    del sys.modules[name]
            spec = importlib.util.spec_from_file_location('orbexa', frozen / '__init__.py',
                submodule_search_locations=[str(frozen)])
            package = importlib.util.module_from_spec(spec)
            sys.modules['orbexa'] = package
            spec.loader.exec_module(package)
            from orbexa.simulation.live_service import _execute_job as frozen_execute
            frozen_execute(settings, parameters, directory, messages, run_event)
        except Exception as error:
            import traceback
            traceback.print_exc()
            payload = dict(kind="error", success=False, message=f"{type(error).__name__}: {error}")
            _write_json(directory / "worker-result.json", payload)
            messages.put(payload)


def _execute_job(settings, parameters, directory, messages, run_event):
    if settings.resume_path is not None:
        if not settings.historical_analytical_resume:
            raise ValueError('Analytical resume must use the explicit historical route')
        return _execute_historical_analytical_job(settings, parameters, directory, messages, run_event)
    return _execute_nonlinear_job(settings, parameters, directory, messages, run_event)


def _native_controller(directory, messages):
    """Retain native solver evidence inside the isolated job process only."""
    from orbexa.control.mpc_controller import MPCController
    from orbexa.solvers.gekko_dynamic_tube import GekkoDynamicTube
    from orbexa.simulation.mission_checkpoint import plain
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


def _execute_nonlinear_job(settings, parameters, directory, messages, run_event):
    """Fresh observed nonlinear run, durable proofs and ordinary finite capture."""
    from dataclasses import replace
    import numpy as np
    from orbexa.control.reachability import IntervalBox
    from orbexa.core.config import SimulationConfig
    from orbexa.simulation.complete_mission import CompleteMissionStore
    from orbexa.simulation.mission_result import MissionResultStore
    from orbexa.simulation.nonlinear_archive import NonlinearProofArchive
    from orbexa.simulation.nonlinear_completion import complete_nonlinear_mission
    from orbexa.simulation.nonlinear_mission import NonlinearMissionRunner
    from orbexa.simulation.nonlinear_profile import NonlinearStudyOptions, IndependentTruthCase
    from orbexa.simulation.spacecraft_design import SpacecraftMissionDesign
    from orbexa.visualization.studio_data import StudioDataset
    cfg = SimulationConfig.load(str(settings.config_path))
    if cfg.solver.remote:
        raise ValueError('The live service requires local GEKKO/IPOPT; remote solvers are not enabled')
    cfg.num_chasers = 8
    cfg.__post_init__()
    design = SpacecraftMissionDesign.from_json(settings.spacecraft_path)
    cfg.target.docking_standoff = design.com_standoff_m
    cfg.target.inertia = np.asarray(design.geometry.metadata['mass_properties']['inertia_kg_m2'])
    options = replace(NonlinearStudyOptions(), release_distance_m=parameters.release_distance_m,
        release_pitch_m=parameters.cluster_spacing_m, nominal_spin_deg_s=parameters.target_spin_deg_s)
    # These physical realization values are fixed prospectively, independent
    # of the observed controller prior and the success/failure of this run.
    case = IndependentTruthCase('live-prospective-study-v1', .0012, .026, .036,
        parameters.target_spin_deg_s, 3817)
    prior = IntervalBox.from_center_radius((0., 0., math.radians(parameters.target_spin_deg_s)), (.002, .002, .008))
    inputs = {name: str(getattr(settings, name)) for name in
        ('config_path', 'prior_path', 'admission_path', 'deployment_path', 'spacecraft_path')}
    _write_json(directory / 'source-inputs.json', dict(paths=inputs,
        sha256={name: _sha(path) for name, path in inputs.items()},
        runtime_python_sha256={str(path.relative_to(Path(__file__).parents[1])): _sha(path)
            for path in Path(__file__).parents[1].rglob('*.py')}, parameters=parameters.as_dict(),
        maximum_updates=settings.maximum_updates, origin='fresh_nonlinear_parameters', execution_model='sensed_nonlinear',
        nonlinear_study_options=options.as_dict(), independent_truth_case=asdict(case), initial_spin_prior=prior.as_dict(),
        historical_only_inputs=('prior_path', 'admission_path', 'deployment_path'),
        scope='Shared declared nonlinear physics/sensor/safety profile; requested study knobs and nonzero observed spin prior; no flight qualification'))
    controller_factory = _native_controller(directory, messages)
    with NonlinearProofArchive(directory / 'nonlinear-certificates.jsonl.zst') as archive:
        def progress(event):
            projection = event.get('projection')
            if projection is None and event.get('result') is not None:
                projection = StudioDataset(event['result']).to_dict()
            if projection is None:
                raise ValueError('Nonlinear runner must provide an actual committed projection')
            _write_json(directory / 'committed-projection.json', projection)
            archive.flush()
            messages.put(dict(kind='ready' if event.get('kind') == 'start' else 'committed', projection=projection,
                update=event.get('update', max(0, len(projection.get('q', []))-1)),
                anomaly=projection['q'][-1], phase=projection.get('phases', ['unknown'])[-1],
                message='Actual sensed nonlinear mission states'))
            if not run_event.is_set():
                messages.put(dict(kind='paused'))
                run_event.wait()
                messages.put(dict(kind='running'))

        def checkpoint(update, runner, outcome):
            archive.flush()
            if archive.holds != len(runner.recorder.hold_records):
                raise ValueError('Durable completed-hold archive does not match the committed mission index')
            # Both owners confirmed these lists are evidence-only. Their
            # full records already have durable archive receipts at this point.
            runner.executor.journal.clear()
            if runner.executor.runtime is not None:
                runner.executor.runtime.events.clear()
            _write_json(directory / 'archive-progress.json', dict(update=update, entries=archive.entries,
                completed_holds=archive.holds, event_kinds=archive.kinds,
                retained_executor_holds=len(runner.executor.journal),
                retained_runtime_events=0 if runner.executor.runtime is None else len(runner.executor.runtime.events),
                execution_exact_q=str(runner.truth_target.current_q), proof_scope='Producer witnesses streamed before motion'))

        runner = NonlinearMissionRunner(cfg, design, case=case, options=options, progress_sink=progress,
            certificate_sink=archive.append, checkpoint_sink=checkpoint, controller_factory=controller_factory,
            initial_target_spin_prior=prior, streamed_hold_index=True, retain_journal=False)
        approach = runner.run(mission='multi', steps=settings.maximum_updates)
        source = archive.close()
    # Publish the closed archive receipt after recording; no archive identity
    # depends on the result that later names it.
    approach.metadata['proof_archive'] = source.as_dict()
    approach.metadata['live_execution_route'] = 'sensed_nonlinear'
    manifest = directory / 'snapshot-manifest.json'
    approach.metadata['live_source_binding'] = dict(source_inputs_sha256=_sha(directory / 'source-inputs.json'),
        snapshot_manifest_sha256=_sha(manifest) if manifest.is_file() else None,
        frozen_package=str(Path(__file__).resolve().parents[1]),
        scope='Closed archive receipt and immutable job source/input manifests; recorded after production to avoid self-reference')
    MissionResultStore.save(approach, directory / 'approach')
    projection = StudioDataset(approach).to_dict()
    _write_json(directory / 'committed-projection.json', projection)
    messages.put(dict(kind='committed', projection=projection, update=len(approach.anom_history)-1,
        anomaly=approach.anom_history[-1], phase=approach.sample_phase_history[-1]))
    store = CompleteMissionStore(directory / 'completion')
    def stage_sink(stage, result):
        store.record(stage, result)
        messages.put(dict(kind='stage', message='Physical completion: '+stage))
    completion = complete_nonlinear_mission(approach, nonlinear_runner=runner, hold_records=source, stage_sink=stage_sink)
    # The early incomplete result deliberately performs no post-approach stage;
    # retain its original source in the same result directory nevertheless.
    if not approach.success or not approach.metadata.get('mission_complete', False):
        store.record('approach', approach)
    store.finish(completion)
    payload = dict(kind='finished', success=bool(completion.success), message=completion.message, projection=projection)
    if completion.lifecycle is not None:
        from orbexa.visualization.lifecycle_renderer import LifecycleDataset
        payload['lifecycle'] = LifecycleDataset(completion.lifecycle).to_dict()
        _write_json(directory / 'lifecycle-projection.json', payload['lifecycle'])
    _write_json(directory / 'worker-result.json', {key: value for key, value in payload.items()
        if key not in ('projection', 'lifecycle')})
    messages.put(payload)


def _execute_historical_analytical_job(settings, parameters, directory, messages, run_event):
    """Explicit retained analytical restart; it never supplies a fresh run."""
    from dataclasses import replace
    import numpy as np
    from orbexa.core.config import SimulationConfig
    from orbexa.estimation.belief import ParameterBelief
    from orbexa.simulation.adtmpc_mission import ADTMPCMissionRunner
    from orbexa.simulation.composite_deployment import CompositeDeploymentOptions
    from orbexa.simulation.complete_mission import CompleteMissionStore
    from orbexa.simulation.mission_result import MissionResultStore
    from orbexa.simulation.physical_approach import ContinuousAdmissionOptions, PhysicalApproachOptions, PhysicalApproachScenario
    from orbexa.simulation.spacecraft_design import SpacecraftMissionDesign
    from orbexa.visualization.studio_data import StudioDataset
    from orbexa.simulation.mission_checkpoint import MissionCheckpoint, physical_target_from_snapshot, plain
    if not settings.historical_analytical_resume or settings.resume_path is None:
        raise ValueError('Historical analytical execution requires its explicitly selected saved source')
    TrackedController = _native_controller(directory, messages)
    inputs = {name: str(getattr(settings, name)) for name in
        ("config_path", "prior_path", "admission_path", "deployment_path", "spacecraft_path")}
    _write_json(directory / "source-inputs.json", dict(paths=inputs,
        sha256={name: _sha(path) for name, path in inputs.items()},
        runtime_python_sha256={str(path.relative_to(Path(__file__).parents[1])): _sha(path)
            for path in Path(__file__).parents[1].rglob("*.py")},
        parameters=parameters.as_dict(), maximum_updates=settings.maximum_updates,
        origin='historical_analytical_checkpoint', execution_model='historical_analytical_resume',
        resume_source=None if settings.resume_path is None else str(settings.resume_path),
        transfer_duration_rule=".18*sqrt(release_distance_m/300)rad; unchanged hardware/input caps"))
    cfg = SimulationConfig.load(str(settings.config_path))
    if cfg.solver.remote:
        raise ValueError('The live service requires local GEKKO/IPOPT; remote solvers are not enabled')
    cfg.num_chasers = 8
    cfg.__post_init__()
    prior = ParameterBelief(**json.loads(settings.prior_path.read_text(encoding="utf-8")))
    admission = ContinuousAdmissionOptions(**json.loads(settings.admission_path.read_text(encoding="utf-8")))
    deployment = CompositeDeploymentOptions(**json.loads(settings.deployment_path.read_text(encoding="utf-8")))
    buffer = parameters.cluster_spacing_m - cfg.mpc.min_chaser_separation - 2*deployment.reference_tube_radius_m
    if buffer <= 0:
        raise ValueError("Requested spacing cannot fit the unchanged pair/tube limits")
    deployment = replace(deployment, carrier_center_lvlh_m=(0., 0., parameters.release_distance_m),
        carrier_transfer_duration_rad=.18*math.sqrt(parameters.release_distance_m/300.), separation_buffer_m=buffer)
    design = SpacecraftMissionDesign.from_json(settings.spacecraft_path)
    cfg.target.docking_standoff = design.com_standoff_m
    cfg.target.inertia = np.asarray(design.geometry.metadata['mass_properties']['inertia_kg_m2'])
    spin = (0., 0., math.radians(parameters.target_spin_deg_s))
    end = (max(settings.maximum_updates, cfg.num_update_steps) + max(cfg.mpc.horizon_steps.values()) + 2) * cfg.anom_step
    checkpoint, target_scope, additional_updates = None, None, settings.maximum_updates
    if settings.resume_path is None:
        target = design.target(cfg, initial_body_spin_rad_s=spin, anomaly_domain=(0., end), gravity_gradient=True)
    else:
        from orbexa.simulation.arrival_policy import exact_outer_step
        _resume_source(settings.resume_path, parameters)
        manifest = json.loads((directory / 'snapshot-manifest.json').read_bytes())
        for name in ('resume_path', 'resume_journal'):
            row = manifest['inputs'][name]
            if _sha(directory / row['frozen']) != row['sha256']:
                raise ValueError('Frozen resume input identity changed before checkpoint construction')
        checkpoint = MissionCheckpoint.from_file(settings.resume_path, source_context={
            'source_reference': manifest['inputs']['resume_path']['original']})
        saved = checkpoint.source_record
        target = physical_target_from_snapshot(saved['target'])
        guard = saved['metadata']['continuous_execution_certificate']
        target_scope = guard['target_bound_context']['scope']
        remaining_cap = saved['metadata']['checkpoint_context']['configuration']['num_update_steps'] - checkpoint.committed_updates
        remaining_domain = ((Fraction(target.anomaly_domain[1])-Fraction(guard['exact_end_anomaly']))
            // exact_outer_step(cfg)) - max(cfg.mpc.horizon_steps.values()) - 1
        additional_updates = min(settings.maximum_updates, remaining_cap, remaining_domain)
        if additional_updates < 1:
            raise ValueError('No new update fits the original total cap and target preview domain')
    options = PhysicalApproachOptions(body_radius_m=design.chaser.body_radius_m,
        com_standoff_m=design.com_standoff_m, initial_body_spin_rad_s=spin, gravity_gradient=True)
    with gzip.open(directory / "continuous-certificates.jsonl.gz", "wt", encoding="utf-8") as journal:
        def certificate_sink(record):
            journal.write(json.dumps(record, allow_nan=False) + "\n")
        profile = PhysicalApproachScenario(cfg, initial_belief=prior, maximum_updates=additional_updates,
            options=options, target=target, deployment_options=deployment, admission=admission,
            certificate_sink=certificate_sink, target_bound_scope=target_scope,
            vehicle_design=design.as_dict(), controller_factory=TrackedController)
        profile.scenario.arrival_policy = _arrival_policy(profile, prior, admission, directory)
        if checkpoint is not None:
            checkpoint.validate_scenario(profile.scenario, 'multi')
            checkpoint.save(directory / 'source-checkpoint.json')
            prefix = _append_resume_journal(checkpoint, settings.resume_path.parent / 'continuous-certificates.jsonl.gz', journal)
            journal.flush()
            if any(_sha(directory / manifest['inputs'][name]['frozen']) != manifest['inputs'][name]['sha256']
                    for name in ('resume_path', 'resume_journal')):
                raise ValueError('Frozen resume identity changed during prefix validation')
            _write_json(directory / 'resume-source.json', dict(checkpoint_identity=checkpoint.identity,
                source_record_sha256=checkpoint.source_record_sha256,
                source_canonical_sha256=checkpoint.source_canonical_sha256,
                frozen_inputs={name:manifest['inputs'][name] for name in ('resume_path', 'resume_journal')},
                prefix=prefix, additional_update_cap=additional_updates,
                original_total_update_cap=remaining_cap+checkpoint.committed_updates,
                target_anomaly_domain=list(target.anomaly_domain),
                scope='Numerical continuation with the public strict checkpoint; original failed attempts/source history retained. Cold plans and dense adaptation-window reset are explicit.'))
        def progress(event):
            kind = event.get("kind", event.get("type", "committed"))
            projection = event.get("projection")
            if projection is None and event.get("result") is not None:
                projection = StudioDataset(event["result"]).to_dict()
            if projection is None:
                raise ValueError("Runner live callback must provide committed projection or result")
            _write_json(directory / "committed-projection.json", projection)
            journal.flush()
            payload = dict(kind="ready" if kind == "start" else "committed", projection=projection,
                update=event.get("update", max(0, len(projection.get("q", []))-1)),
                anomaly=projection["q"][-1], phase=projection.get("phases", ["unknown"])[-1])
            messages.put(payload)
            if not run_event.is_set():
                messages.put(dict(kind="paused"))
                run_event.wait()
                messages.put(dict(kind="running"))
        profile.scenario.progress_sink = progress
        runner = ADTMPCMissionRunner(scenario=profile.scenario)
        approach = (runner.run(mission="multi", steps=additional_updates) if checkpoint is None
            else runner.resume(checkpoint, steps=additional_updates))
        MissionResultStore.save(approach, directory / "approach")
        # Close the complete source journal before authenticating and replaying
        # its interval proofs for conservative held-command fuel accounting.
        journal.close()
        projection = StudioDataset(approach).to_dict()
        _write_json(directory / "committed-projection.json", projection)
        messages.put(dict(kind="committed", projection=projection, update=len(approach.anom_history)-1,
            anomaly=approach.anom_history[-1], phase=approach.sample_phase_history[-1]))
        store = CompleteMissionStore(directory / "completion")
        def stage_sink(stage, result):
            store.record(stage, result)
            messages.put(dict(kind="stage", message="Physical completion: " + stage))
        bound = max(abs(float(b[k])) for b in cfg.mpc.input_bounds for k in ("lower", "upper"))
        accountant = None
        if approach.success and approach.metadata.get('mission_complete') is True:
            from orbexa.simulation.held_propellant import HeldInputPropellantAccountant
            raw_path = directory / 'approach' / MissionResultStore.filename
            journal_path = directory / 'continuous-certificates.jsonl.gz'
            accountant = HeldInputPropellantAccountant.from_files(raw_path, [journal_path],
                expected_approach_sha256=_sha(raw_path), expected_journal_sha256=[_sha(journal_path)],
                producer_package_root=Path(__file__).parents[1])
            messages.put(dict(kind='stage', message='Verifying held-command propellant bounds'))
        completion = design.completion_runner(input_component_bound=bound, stage_sink=stage_sink,
            translation_accountant=accountant).complete(approach)
        store.finish(completion)
        payload = dict(kind="finished", success=bool(completion.success), message=completion.message,
            projection=projection)
        if completion.lifecycle is not None:
            from orbexa.visualization.lifecycle_renderer import LifecycleDataset
            payload['lifecycle'] = LifecycleDataset(completion.lifecycle).to_dict()
            _write_json(directory / "lifecycle-projection.json", payload['lifecycle'])
        _write_json(directory / "worker-result.json", {key: value for key, value in payload.items()
            if key not in ('projection', 'lifecycle')})
        messages.put(payload)


class _LiveHandler(BaseHTTPRequestHandler):
    server_version = "ORBEXA-Loopback/1"
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass

    def _origin(self):
        origin = self.headers.get("Origin")
        host = self.headers.get("Host", "")
        if host not in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"):
            raise PermissionError("Only the literal loopback host is accepted")
        if origin is not None and origin not in self.server.allowed_origins:
            raise PermissionError("This browser origin is not allowed")
        return origin

    def _reply(self, status, payload=None, *, preflight=False):
        raw = b"" if status == 204 else json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        origin = self.headers.get("Origin")
        if origin in self.server.allowed_origins:
            self.send_header("Access-Control-Allow-Origin", origin)
            if preflight:
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Max-Age", "600")
                if self.headers.get("Access-Control-Request-Private-Network") == "true":
                    self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Vary", "Origin, Access-Control-Request-Headers, Access-Control-Request-Private-Network")
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if raw:
            self.wfile.write(raw)

    def _body(self):
        if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
            raise ValueError("POST requests require application/json")
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("A valid Content-Length is required") from error
        if not 0 < size <= 16384 or self.headers.get("Transfer-Encoding"):
            raise ValueError("JSON request size must be between1 and16384 bytes")
        self.connection.settimeout(5)
        def pairs(rows):
            result = {}
            for key, value in rows:
                if key in result:
                    raise ValueError("Duplicate JSON field")
                result[key] = value
            return result
        def invalid(_value):
            raise ValueError("JSON numbers must be finite")
        try:
            value = json.loads(self.rfile.read(size), object_pairs_hook=pairs, parse_constant=invalid)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("Malformed JSON request") from error
        if not isinstance(value, dict):
            raise ValueError("JSON request must be an object")
        return value

    def do_OPTIONS(self):
        try:
            self._origin()
            if self.headers.get("Origin") is None:
                raise PermissionError("Browser preflight requires an allowed origin")
            if urlsplit(self.path).path not in ("/run", "/pause", "/reset", "/state", "/health"):
                return self._reply(404, dict(error="Unknown endpoint"))
            requested = {x.strip().lower() for x in self.headers.get("Access-Control-Request-Headers", "").split(",") if x.strip()}
            if requested - {"content-type"} or self.headers.get("Access-Control-Request-Method", "GET") not in ("GET", "POST"):
                raise PermissionError("Unsupported preflight method or headers")
            self._reply(204, preflight=True)
        except PermissionError as error:
            self._reply(403, dict(error=str(error)))

    def do_GET(self):
        try:
            self._origin()
            url = urlsplit(self.path)
            if url.path == "/health":
                state = self.server.service.state()
                return self._reply(200, dict(service=LIVE_SERVICE, status="ready",
                    protocol_version=LIVE_PROTOCOL_VERSION, execution_model=state['execution_model'],
                    bind="127.0.0.1", port=self.server.server_port, worker_alive=self.server.service.worker_alive,
                    session_id=state['session_id'],
                    parameter_limits=PARAMETER_LIMITS, maximum_updates=self.server.service.settings.maximum_updates,
                    source="Maintained sensed nonlinear GEKKO/IPOPT with finite fleet admission and contact; explicit historical analytical resume available; outcomes may refuse"))
            if url.path != "/state":
                return self._reply(404, dict(error="Unknown endpoint"))
            query = parse_qs(url.query, keep_blank_values=True)
            if set(query) - {"after", "timeout"} or any(len(v) != 1 for v in query.values()):
                raise ValueError("Supply only one after and timeout query value")
            after = int(query['after'][0]) if 'after' in query else None
            timeout = float(query.get('timeout', ['20' if after is not None else '0'])[0])
            self._reply(200, self.server.service.state(after=after, timeout=timeout))
        except PermissionError as error:
            self._reply(403, dict(error=str(error)))
        except (ValueError, TypeError) as error:
            self._reply(400, dict(error=str(error)))
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        try:
            self._origin()
            envelope = self._body()
            if (set(envelope) != {'protocol_version','session_id','payload'}
                    or type(envelope.get('protocol_version')) is not int
                    or envelope['protocol_version'] != LIVE_PROTOCOL_VERSION
                    or envelope.get('session_id') != self.server.service.state()['session_id']):
                raise RuntimeError('Simulator session or protocol changed; reconnect before a command')
            value = envelope['payload']
            if not isinstance(value,dict):
                raise ValueError('Command payload must be a JSON object')
            if self.path == "/run":
                return self._reply(202, self.server.service.start(LiveParameters.from_dict(value)))
            if self.path == "/pause":
                if set(value) - {"paused"}:
                    raise ValueError("Supply only paused")
                return self._reply(200, self.server.service.pause(value.get('paused')))
            if self.path == "/reset":
                if value:
                    raise ValueError("Reset does not accept parameters")
                return self._reply(200, self.server.service.reset())
            self._reply(404, dict(error="Unknown endpoint"))
        except PermissionError as error:
            self._reply(403, dict(error=str(error)))
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (ValueError, TypeError, OSError) as error:
            self._reply(400, dict(error=str(error)))
        except RuntimeError as error:
            self._reply(409, dict(error=str(error)))


def make_server(service, *, port=8882, viewer_port=None):
    """Bind only IPv4 loopback; port0 is useful for isolated caller probes."""
    if not isinstance(service, LiveSimulationService):
        raise TypeError("Inject LiveSimulationService")
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("Invalid loopback port")
    if viewer_port is not None and (isinstance(viewer_port,bool) or not isinstance(viewer_port,int)
            or not 1 <= viewer_port <= 65535):
        raise ValueError('Invalid loopback viewer port')
    server = ThreadingHTTPServer(("127.0.0.1", port), _LiveHandler)
    server.daemon_threads = True
    server.service = service
    server.allowed_origins = ALLOWED_ORIGINS | (frozenset() if viewer_port is None else frozenset((
        f'http://127.0.0.1:{viewer_port}',f'http://localhost:{viewer_port}')))
    return server
