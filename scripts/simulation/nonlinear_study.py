"""One explicit prospective nonlinear study arm per isolated process.

Only the standard library is imported until execution. The CLI can therefore
show its declaration/help without loading numerical or solver dependencies.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from fractions import Fraction
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback


STUDY_SCHEMA = 'orbexa.nonlinear-study.v1'
_ARM_STARTED = False


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
        if output.is_relative_to(Path(__file__).resolve().parents[2]):
            raise ValueError('Study outputs must be outside the maintained checkout')
        if not self.case or self.adaptation not in ('adaptive', 'frozen') or self.bandwidth not in ('dynamic', 'fixed'):
            raise ValueError('Select an explicit case and adaptive/frozen by dynamic/fixed arm')
        if self.steps is not None and (isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 1):
            raise ValueError('steps is a positive execution cap, independent of the prospective reference domain')
        object.__setattr__(self, 'output_dir', output)


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def _write(path, value):
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


def _runtime_environment():
    """Read installed dependency identities without importing scientific code."""
    packages={}
    for name in ('numpy','scipy','gekko','python-flint','PyYAML','zstandard'):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    native=None
    try:
        distribution=importlib.metadata.distribution('gekko')
        executable='apm.exe' if os.name=='nt' else 'apm_mac' if sys.platform=='darwin' else 'apm'
        relative='gekko/bin/'+executable
        path=Path(distribution.locate_file(relative))
        if path.is_file():native=dict(distribution='gekko',path=relative,sha256=_sha(path))
    except importlib.metadata.PackageNotFoundError:pass
    return dict(python=sys.version,platform=platform.platform(),packages=packages,
        native_gekko_executable=native,
        scope='Installed version and native binary identities before motion; unavailable metadata remains null')


def _declaration(path, selected):
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


def _freeze(request):
    """Reserve a new directory and copy coherent Python/input source bytes."""
    directory = request.output_dir
    directory.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[1]
    destination = directory/'runtime'/'orbexa'
    before = {path.relative_to(source).as_posix(): _sha(path) for path in source.rglob('*.py')
              if '__pycache__' not in path.parts}
    for name, expected in before.items():
        target = destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/name, target)
        if _sha(target) != expected:
            raise RuntimeError('Python source changed while freezing '+name)
    after = {path.relative_to(source).as_posix(): _sha(path) for path in source.rglob('*.py')
             if '__pycache__' not in path.parts}
    if before != after:
        raise RuntimeError('Source changed during snapshot; no arm was executed')
    inputs, frozen = {}, {}
    (directory/'inputs').mkdir()
    for name in ('config', 'spacecraft', 'study'):
        original = getattr(request, name)
        target = directory/'inputs'/(name+original.suffix)
        expected = _sha(original)
        shutil.copyfile(original, target)
        if _sha(original) != expected or _sha(target) != expected:
            raise RuntimeError('Explicit '+name+' input changed during snapshot')
        inputs[name] = dict(original=str(original), frozen=str(target.relative_to(directory)), sha256=expected)
        frozen[name] = str(target)
    # Standard-library output here intentionally precedes any scientific import.
    manifest = dict(schema='orbexa.nonlinear-study-source.v1', package_python_sha256=before, inputs=inputs,
        selected_case=request.case, adaptation=request.adaptation, bandwidth=request.bandwidth, steps=request.steps,
        runtime_environment=_runtime_environment(),
        scope='One explicit predeclared arm; coherent source/input bytes before any motion; a cap is not completion')
    (directory/'source-manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return destination, dict(**frozen, case=request.case, adaptation=request.adaptation, bandwidth=request.bandwidth,
                             output_dir=str(directory), steps=request.steps)


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


def _metrics(approach, completion, runner, directory, error=None):
    """Report recorded physics and refusals without success-only filtering."""
    import numpy as np
    from orbexa.simulation.nonlinear_archive import HOLD_INDEX_SCHEMA
    native = dict(total=0, successful=0, refused=0, unknown=0)
    calls = set()
    path = directory/'native-attempts.jsonl'
    if path.is_file():
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                native['total'] += 1
                native['successful' if row.get('success') is True else 'refused' if row.get('success') is False else 'unknown'] += 1
                if row.get('controller_call') is not None:
                    calls.add(row['controller_call'])
    native['controller_calls'] = len(calls)
    result = dict(schema='orbexa.nonlinear-study-metrics.v1', success=False,
        native_attempts=native,
        error=error, scope='Recorded simulated physical prefix and actual failures; no successful-case filtering or flight deadline claim')
    if approach is None:
        result.update(status='SETUP_FAILED', physical_elapsed_s=None, safety=None, tubes={}, bandwidth={},
            set_width={}, effort={}, fuel_bound=None, fallback_counts={}, deadline_misses=None, failures=[error])
        return result
    metadata = approach.metadata
    elapsed = metadata.get('elapsed_time_seconds', [])
    actual_time = None
    if runner is not None:
        actual_time = float(runner.truth_target.physical_time_at(runner.truth_target.current_q))
    start_time = float(elapsed[0]) if elapsed else 0.
    end_time, end_stage = actual_time, 'approach'
    if completion is not None and completion.lifecycle is not None:
        lifecycle = completion.lifecycle
        contact = lifecycle.capture.audit.get('finite_contact')
        if contact is not None:
            end_time, end_stage = float(contact['final_target']['time_s']), 'finite_contact'
        if lifecycle.capture.state is not None:
            end_time, end_stage = float(lifecycle.capture.state.time_s), 'attachment'
        for name in ('detumbling', 'deorbit'):
            stage = getattr(lifecycle, name)
            if stage is not None:
                end_time, end_stage = float(stage.final_state.time_s), name
    widths = {}
    for key in approach.initial_belief.get('feasible_sets', {}):
        history = [float(value[key][1]-value[key][0]) for value in approach.feasible_set_history if key in value]
        first, last = approach.initial_belief['feasible_sets'][key], approach.final_belief['feasible_sets'][key]
        widths[key] = dict(initial=float(first[1]-first[0]), final=float(last[1]-last[0]), history=history)
    tubes, bandwidth, effort = {}, {}, {}
    for key, values in approach.tube_diagnostics.items():
        p, w, a = (np.asarray(values.get(name, []), dtype=float) for name in ('phi', 'omega', 'bandwidth'))
        tubes[key] = dict(max_position_radius_norm_m=float(np.max(np.linalg.norm(w, axis=1))) if w.size else None,
            max_sliding_radius_norm=float(np.max(np.linalg.norm(p, axis=1))) if p.size else None,
            final_position_radius_m=None if not w.size else w[-1].tolist(), final_sliding_radius=None if not p.size else p[-1].tolist())
        bandwidth[key] = dict(minimum_per_rad=None if not a.size else np.min(a, axis=0).tolist(),
            maximum_per_rad=None if not a.size else np.max(a, axis=0).tolist(), history_per_rad=a.tolist())
        effort[key] = dict(applied_input_squared_integral_per_reference_anomaly=float(sum(values.get('applied_control_effort', []))),
            applied_control_rms=values.get('applied_control_rms', []), ancillary_control_rms=values.get('ancillary_control_rms', []),
            semantics='Actual delivered held inputs weighted by exact reference-anomaly hold widths; not physical force or measured fuel')
    sources = {}
    for row in metadata.get('fleet_hold_records', []):
        source = row['command_source'] if row.get('schema') == HOLD_INDEX_SCHEMA else row['decision']['source']
        sources[source] = sources.get(source, 0)+1
    failure = [] if completion is None or completion.lifecycle is None else [completion.lifecycle.failure] if completion.lifecycle.failure else []
    failure += metadata.get('failure_events', [])
    for message in approach.message.split('; '):
        if message and message != 'nonlinear_approach_not_complete_within_declared_update_budget':
            failure.append(dict(stage='approach', message=message))
    for name in ('active_target_margins_safe', 'pairwise_spacing_safe', 'sampled_tube_containment'):
        if metadata.get(name) is False:
            failure.append(dict(stage='approach_safety', gate=name, passed=False))
    if metadata.get('continuous_execution_certificate', {}).get('all_intervals_certified') is False:
        failure.append(dict(stage='approach_safety', gate='all_intervals_certified', passed=False))
    approach_complete = bool(approach.success and metadata.get('mission_complete'))
    if completion is not None and not completion.success and approach_complete and not failure:
        failure.append(dict(stage='completion', message=completion.message))
    if error is not None:
        failure.append(error)
    fuel_unavailable = directory/'fuel-bound-refused.json'
    physical_complete=bool(completion is not None and completion.success)
    result.update(success=physical_complete and error is None and not failure,
        status='FAILED' if failure else 'COMPLETED' if physical_complete else 'INCOMPLETE',
        physical_mission_complete=physical_complete,
        approach_complete=approach_complete, message=approach.message if completion is None else completion.message,
        physical_elapsed_s=None if end_time is None else end_time-start_time,
        approach_physical_elapsed_s=None if actual_time is None else actual_time-start_time,
        physical_end_time_s=end_time, physical_end_stage=end_stage,
        committed_outer_elapsed_s=None if not elapsed else float(elapsed[-1]-elapsed[0]),
        physical_end_exact_q=metadata.get('actual_physical_end_exact_q'), updates=len(approach.anom_history)-1,
        safety={name: metadata.get(name) for name in ('active_target_margins_safe', 'pairwise_spacing_safe', 'sampled_tube_containment',
            'dense_min_containment_margin', 'dense_min_physical_clearance', 'dense_min_spacing')},
        continuous_execution=metadata.get('continuous_execution_certificate'), tubes=tubes, bandwidth=bandwidth,
        set_width=widths, physical_adaptation=metadata.get('physical_adaptation'), effort=effort,
        fuel_bound=None if completion is None or completion.translation_budget is None else completion.translation_budget.to_dict(),
        fuel_bound_unavailable=None if not fuel_unavailable.is_file() else json.loads(fuel_unavailable.read_text(encoding='utf-8')),
        fallback_counts=sources, deadline_misses=metadata.get('observed_deadline_misses'),
        deadline_unknown_updates=metadata.get('observed_deadline_unknown_updates'),
        planner_only_deadline_misses=metadata.get('planner_ready_deadline_misses'),
        deadline_scope=metadata.get('timing_scope'),attempted_updates=metadata.get('attempted_updates'),
        update_wall_timing=metadata.get('update_wall_timing', []), failures=failure,
        completion=None if completion is None else completion.to_dict())
    return result


def _execute_arm(*, config, spacecraft, study, case, adaptation, bandwidth, output_dir, steps):
    global _ARM_STARTED
    if _ARM_STARTED:
        raise RuntimeError('Run each study arm in a separate process')
    _ARM_STARTED = True
    directory, started = Path(output_dir), time.perf_counter()
    runner = approach = completion = source = None
    error = None
    try:
        from dataclasses import replace
        import math
        import threading
        import numpy as np
        from orbexa.control.reachability import IntervalBox
        from orbexa.core.config import SimulationConfig
        from orbexa.estimation.navigation import StudySensorProfile
        from orbexa.simulation.complete_mission import CompleteMissionOptions, CompleteMissionStore
        from orbexa.simulation.live_service import _native_controller
        from orbexa.simulation.mission_result import MissionResultStore
        from orbexa.simulation.nonlinear_archive import NonlinearProofArchive
        from orbexa.simulation.nonlinear_completion import NonlinearTranslationAccountant, complete_nonlinear_mission
        from orbexa.simulation.nonlinear_mission import NonlinearMissionRunner
        from orbexa.simulation.nonlinear_profile import NonlinearStudyOptions, IndependentTruthCase
        from orbexa.simulation.spacecraft_design import SpacecraftMissionDesign
        from orbexa.simulation.chaser_attitude import ChaserAttitudeTracker
        declaration = _declaration(study, case)
        options_record = dict(declaration['options'])
        options_record.pop('scope', None)
        if 'sensor' in options_record:
            sensor = dict(options_record['sensor'])
            sensor.pop('scope', None)
            options_record['sensor'] = StudySensorProfile(**sensor)
        options = NonlinearStudyOptions(**options_record)
        cases = [IndependentTruthCase(**record) for record in declaration['cases']]
        selected = next(value for value in cases if value.case_id == case)
        cfg, design = SimulationConfig.load(config), SpacecraftMissionDesign.from_json(Path(spacecraft))
        if cfg.solver.remote:
            raise ValueError('Study execution requires local GEKKO/IPOPT')
        cfg.num_chasers = 8
        cfg.__post_init__()
        cfg.target.docking_standoff = design.com_standoff_m
        cfg.target.inertia = np.asarray(design.geometry.metadata['mass_properties']['inertia_kg_m2'])
        prior_record = declaration.get('initial_target_spin_prior')
        if prior_record is None:
            center, radius = (0., 0., math.radians(options.nominal_spin_deg_s)), (.002, .002, .008)
        else:
            if set(prior_record) != {'center_rad_s', 'radius_rad_s'}:
                raise ValueError('Initial target spin prior needs center_rad_s and radius_rad_s')
            center, radius = prior_record['center_rad_s'], prior_record['radius_rad_s']
        prior = IntervalBox.from_center_radius(center, radius)
        cap = options.maximum_updates if steps is None else steps
        if cap > options.maximum_updates:
            raise ValueError('Execution cap exceeds the prospectively declared profile update limit')
        class NativeEvents:
            def __init__(self):
                self.lock = threading.Lock()
            def put(self, value):
                # Native attempts have a separate durable source transcript.
                # Keep only small progress events, never trajectory/proof copies.
                with self.lock:
                    with (directory/'progress.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(value, allow_nan=False)+'\n')
        native_events = NativeEvents()
        controller = _native_controller(directory, native_events)
        store = CompleteMissionStore(directory/'completion')
        with NonlinearProofArchive(directory/'nonlinear-certificates.jsonl.zst') as archive:
            def checkpoint(update, value, outcome):
                archive.flush()
                if archive.holds != len(value.recorder.hold_records):
                    raise ValueError('Durable archive count differs from the committed exact hold index')
                value.executor.journal.clear()
                value.executor.runtime.events.clear()
                phase=value.recorder.arrays.sample_phases[-1]
                distance=max(float(np.linalg.norm(value.state.centers[chaser.chaser_id][:3]-
                    value.preview.docking_state(chaser.docking_point_body,value.state.anomaly)[:3])) for chaser in value.chasers)
                progress = dict(update=update, exact_q=str(value.truth_target.current_q),
                    physical_time_s=value.truth_target.physical_time_at(value.truth_target.current_q),
                    phase=phase,maximum_nominal_goal_distance_m=distance,
                    performance_holds=sum(source=='PERFORMANCE' for source in outcome.control_sources),
                    fallback_holds=sum(source!='PERFORMANCE' for source in outcome.control_sources),
                    elapsed_wall_s=time.perf_counter()-started,
                    completed_holds=archive.holds, archive_entries=archive.entries,
                    retained_full_journal_holds=len(value.executor.journal), retained_runtime_events=len(value.executor.runtime.events))
                native_events.put({'kind':'checkpoint','schema':'orbexa.nonlinear-progress-checkpoint.v1',
                                   'committed_progress':progress})
                _write(directory/'progress.json', progress)
            runner = NonlinearMissionRunner(cfg, design, case=selected, options=options,
                adaptation_mode=adaptation, bandwidth_mode=bandwidth, certificate_sink=archive.append,
                checkpoint_sink=checkpoint, controller_factory=controller, initial_target_spin_prior=prior,
                streamed_hold_index=True, retain_journal=False)
            _write(directory/'study-manifest.json', dict(schema=STUDY_SCHEMA, study_id=declaration['study_id'],
                cases=declaration['cases'], selected_case=case, selected_arm=dict(adaptation=adaptation, bandwidth=bandwidth),
                factorial_axes=dict(adaptation=('adaptive', 'frozen'), bandwidth=('dynamic', 'fixed')),
                resolved_options=runner.options.as_dict(), design=design.as_dict(), initial_belief=runner.adaptation.belief.as_metadata(),
                initial_target_spin_prior=prior.as_dict(), model_bounds=runner.model.as_dict(), point_truth=runner.truth_model.as_dict(),
                reference_clock=runner.clock.as_dict(), source_manifest_sha256=_sha(directory/'source-manifest.json'),
                execution_cap_updates=cap, scope='Prospective declaration before any actual motion; one selected arm, no winner selection'))
            approach = runner.run(mission='multi', steps=cap)
            source = archive.close()
        approach.metadata['proof_archive'] = source.as_dict()
        approach.metadata['study_source_binding'] = dict(source_manifest_sha256=_sha(directory/'source-manifest.json'),
            study_manifest_sha256=_sha(directory/'study-manifest.json'), study_id=declaration['study_id'], selected_case=case,
            adaptation=adaptation, bandwidth=bandwidth)
        MissionResultStore.save(approach, directory/'approach')
        def stage_sink(stage, result):
            store.record(stage, result)
            _write(directory/'stage-progress.json', dict(stage=stage, elapsed_wall_s=time.perf_counter()-started))
        completion = complete_nonlinear_mission(approach, nonlinear_runner=runner, hold_records=source, stage_sink=stage_sink)
        if completion.translation_budget is None and len(approach.anom_history) > 1:
            try:
                accountant = NonlinearTranslationAccountant(approach, model=runner.truth_model, clock=runner.clock,
                    hold_records=source, expected_actuator_id=runner.executor.actuator.fingerprint,
                    expected_sensor_id=runner.options.sensor.fingerprint)
                bound = max(abs(float(value)) for row in runner.config.mpc.input_bounds for value in row.values())
                budget = accountant.translation_budget(approach, options=CompleteMissionOptions(bound,
                    translation_force_n=design.translation_force_n, specific_impulse_s=design.specific_impulse_s,
                    target_mass_kg=design.target_mass_kg), attitude_tracker=ChaserAttitudeTracker(default_chaser=design.chaser))
                completion = replace(completion, translation_budget=budget)
                store.record('translation_budget', budget)
            except (ValueError, RuntimeError, ArithmeticError, ImportError) as refused:
                _write(directory/'fuel-bound-refused.json', dict(status='UNAVAILABLE', reason=str(refused),
                    scope='No zero consumption or successful whole-prefix accounting inferred'))
        if not approach.success or not approach.metadata.get('mission_complete'):
            store.record('approach', approach)
        store.finish(completion)
    except Exception as exception:
        error = dict(type=type(exception).__name__, message=str(exception))
        (directory/'execution-error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        if runner is not None and approach is None:
            try:
                runner._finish_attempt_timing('UNEXPECTED_EXECUTION_FAILURE')
                approach = runner.recorder.result(valid=False, completed=False,
                    messages=['Uncompleted execution: '+str(exception)], adaptation=runner.adaptation)
                approach.metadata['failure_events'] = list(runner.failure_events)+[error]
                approach.metadata['actual_physical_end_exact_q'] = str(runner.truth_target.current_q)
                approach.metadata.update(update_wall_timing=runner.update_wall_timing,
                    observed_deadline_misses=sum(v['deadline_miss_full_update'] for v in runner.update_wall_timing),
                    observed_deadline_unknown_updates=sum(v['full_update_deadline_status']=='UNKNOWN' for v in runner.update_wall_timing),
                    planner_ready_deadline_misses=sum(v['ready_within_physical_outer_period'] is False for v in runner.update_wall_timing),
                    attempted_updates=len(runner.update_wall_timing),
                    timing_scope='Offline attempted updates including refused/exceptional tails; no flight deadline claim')
                from orbexa.simulation.mission_result import MissionResultStore
                MissionResultStore.save(approach, directory/'approach')
            except Exception as recording_error:
                error['prefix_recording_error'] = str(recording_error)
    try:
        metrics = _metrics(approach, completion, runner, directory, error)
    except Exception as metrics_error:
        metrics = dict(schema='orbexa.nonlinear-study-metrics.v1', success=False, status='METRICS_REFUSED',
            error=error, metrics_error=dict(type=type(metrics_error).__name__, message=str(metrics_error)),
            scope='Source artifacts retained; unavailable metrics are not zero or success')
    metrics['wall_time_s'] = time.perf_counter()-started
    _write(directory/'study-metrics.json', metrics)
    summary = dict(success=bool(metrics.get('success')), status=metrics.get('status'),
        message=metrics.get('message', error), case=case, adaptation=adaptation, bandwidth=bandwidth,
        output_dir=str(directory), metrics='study-metrics.json', source_manifest='source-manifest.json',
        study_manifest='study-manifest.json' if (directory/'study-manifest.json').is_file() else None)
    _write(directory/'run-result.json', summary)
    return summary
