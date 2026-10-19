"""Execution lifecycle for one already frozen nonlinear study arm."""
from __future__ import annotations

from functools import partial
from pathlib import Path
import time
import traceback

from .study_metrics import collect_study_metrics
from .study_records import StudyTelemetry, write_record
from .study_request import STUDY_SCHEMA, load_declaration
from .study_source import hash_file


class StudyArmExecution:
    """Own setup, approach, completion, and failure retention for one arm."""

    def __init__(self, *, config, spacecraft, study, case, adaptation, bandwidth, output_dir, steps):
        self.config_path = config
        self.spacecraft_path = spacecraft
        self.study_path = study
        self.case = case
        self.adaptation = adaptation
        self.bandwidth = bandwidth
        self.directory = Path(output_dir)
        self.steps = steps
        self.started = time.perf_counter()
        self._run_started = False
        self.runner = self.approach = self.completion = self.source = None
        self.error = None

    def run(self):
        if self._run_started:
            raise RuntimeError('Run each study arm in a separate process')
        self._run_started = True
        try:
            self._prepare()
            self._run_approach()
            self._complete()
        except Exception as exception:
            self._retain_failure(exception)
        return self._finalize()

    def _prepare(self):
        # Load the complete execution dependency set before any physical motion.
        # Later stage-local imports make each stage's dependencies explicit and
        # resolve from this same, already loaded frozen package.
        from dataclasses import replace
        import math
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

        self.declaration = load_declaration(self.study_path, self.case)
        options_record = dict(self.declaration['options'])
        options_record.pop('scope', None)
        if 'sensor' in options_record:
            sensor = dict(options_record['sensor'])
            sensor.pop('scope', None)
            options_record['sensor'] = StudySensorProfile(**sensor)
        self.options = NonlinearStudyOptions(**options_record)
        cases = [IndependentTruthCase(**record) for record in self.declaration['cases']]
        self.selected_case = next(value for value in cases if value.case_id == self.case)
        self.config = SimulationConfig.load(self.config_path)
        self.design = SpacecraftMissionDesign.from_json(Path(self.spacecraft_path))
        if self.config.solver.remote:
            raise ValueError('Study execution requires local GEKKO/IPOPT')
        self.config.num_chasers = 8
        self.config.__post_init__()
        self.config.target.docking_standoff = self.design.com_standoff_m
        self.config.target.inertia = np.asarray(self.design.geometry.metadata['mass_properties']['inertia_kg_m2'])
        prior_record = self.declaration.get('initial_target_spin_prior')
        if prior_record is None:
            center, radius = (0., 0., math.radians(self.options.nominal_spin_deg_s)), (.002, .002, .008)
        else:
            if set(prior_record) != {'center_rad_s', 'radius_rad_s'}:
                raise ValueError('Initial target spin prior needs center_rad_s and radius_rad_s')
            center, radius = prior_record['center_rad_s'], prior_record['radius_rad_s']
        self.prior = IntervalBox.from_center_radius(center, radius)
        self.cap = self.options.maximum_updates if self.steps is None else self.steps
        if self.cap > self.options.maximum_updates:
            raise ValueError('Execution cap exceeds the prospectively declared profile update limit')
        self.telemetry = StudyTelemetry(self.directory, self.started)
        self.controller = _native_controller(self.directory, self.telemetry)
        self.store = CompleteMissionStore(self.directory/'completion')

    def _run_approach(self):
        from orbexa.simulation.mission_result import MissionResultStore
        from orbexa.simulation.nonlinear_archive import NonlinearProofArchive
        from orbexa.simulation.nonlinear_mission import NonlinearMissionRunner

        with NonlinearProofArchive(self.directory/'nonlinear-certificates.jsonl.zst') as archive:
            self.runner = NonlinearMissionRunner(self.config, self.design, case=self.selected_case, options=self.options,
                adaptation_mode=self.adaptation, bandwidth_mode=self.bandwidth, certificate_sink=archive.append,
                checkpoint_sink=partial(self.telemetry.checkpoint, archive), controller_factory=self.controller,
                initial_target_spin_prior=self.prior, streamed_hold_index=True, retain_journal=False)
            self._write_manifest()
            self.approach = self.runner.run(mission='multi', steps=self.cap)
            self.source = archive.close()
        self.approach.metadata['proof_archive'] = self.source.as_dict()
        self.approach.metadata['study_source_binding'] = dict(
            source_manifest_sha256=hash_file(self.directory/'source-manifest.json'),
            study_manifest_sha256=hash_file(self.directory/'study-manifest.json'),
            study_id=self.declaration['study_id'], selected_case=self.case,
            adaptation=self.adaptation, bandwidth=self.bandwidth)
        MissionResultStore.save(self.approach, self.directory/'approach')

    def _write_manifest(self):
        write_record(self.directory/'study-manifest.json', dict(schema=STUDY_SCHEMA,
            study_id=self.declaration['study_id'], cases=self.declaration['cases'], selected_case=self.case,
            selected_arm=dict(adaptation=self.adaptation, bandwidth=self.bandwidth),
            factorial_axes=dict(adaptation=('adaptive', 'frozen'), bandwidth=('dynamic', 'fixed')),
            resolved_options=self.runner.options.as_dict(), design=self.design.as_dict(),
            initial_belief=self.runner.adaptation.belief.as_metadata(), initial_target_spin_prior=self.prior.as_dict(),
            model_bounds=self.runner.model.as_dict(), point_truth=self.runner.truth_model.as_dict(),
            reference_clock=self.runner.clock.as_dict(), source_manifest_sha256=hash_file(self.directory/'source-manifest.json'),
            execution_cap_updates=self.cap,
            scope='Prospective declaration before any actual motion; one selected arm, no winner selection'))

    def _complete(self):
        from dataclasses import replace
        from orbexa.simulation.chaser_attitude import ChaserAttitudeTracker
        from orbexa.simulation.complete_mission import CompleteMissionOptions
        from orbexa.simulation.nonlinear_completion import NonlinearTranslationAccountant, complete_nonlinear_mission

        self.completion = complete_nonlinear_mission(self.approach, nonlinear_runner=self.runner,
            hold_records=self.source, stage_sink=partial(self.telemetry.stage_completed, self.store))
        if self.completion.translation_budget is None and len(self.approach.anom_history) > 1:
            try:
                accountant = NonlinearTranslationAccountant(self.approach, model=self.runner.truth_model,
                    clock=self.runner.clock, hold_records=self.source,
                    expected_actuator_id=self.runner.executor.actuator.fingerprint,
                    expected_sensor_id=self.runner.options.sensor.fingerprint)
                bound = max(abs(float(value)) for row in self.runner.config.mpc.input_bounds for value in row.values())
                budget = accountant.translation_budget(self.approach, options=CompleteMissionOptions(bound,
                    translation_force_n=self.design.translation_force_n, specific_impulse_s=self.design.specific_impulse_s,
                    target_mass_kg=self.design.target_mass_kg),
                    attitude_tracker=ChaserAttitudeTracker(default_chaser=self.design.chaser))
                self.completion = replace(self.completion, translation_budget=budget)
                self.store.record('translation_budget', budget)
            except (ValueError, RuntimeError, ArithmeticError, ImportError) as refused:
                write_record(self.directory/'fuel-bound-refused.json', dict(status='UNAVAILABLE', reason=str(refused),
                    scope='No zero consumption or successful whole-prefix accounting inferred'))
        if not self.approach.success or not self.approach.metadata.get('mission_complete'):
            self.store.record('approach', self.approach)
        self.store.finish(self.completion)

    def _retain_failure(self, exception):
        self.error = dict(type=type(exception).__name__, message=str(exception))
        (self.directory/'execution-error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        if self.runner is not None and self.approach is None:
            try:
                self.runner._finish_attempt_timing('UNEXPECTED_EXECUTION_FAILURE')
                self.approach = self.runner.recorder.result(valid=False, completed=False,
                    messages=['Uncompleted execution: '+str(exception)], adaptation=self.runner.adaptation)
                self.approach.metadata['failure_events'] = list(self.runner.failure_events)+[self.error]
                self.approach.metadata['actual_physical_end_exact_q'] = str(self.runner.truth_target.current_q)
                timing = self.runner.update_wall_timing
                self.approach.metadata.update(update_wall_timing=timing,
                    observed_deadline_misses=sum(value['deadline_miss_full_update'] for value in timing),
                    observed_deadline_unknown_updates=sum(value['full_update_deadline_status'] == 'UNKNOWN' for value in timing),
                    planner_ready_deadline_misses=sum(value['ready_within_physical_outer_period'] is False for value in timing),
                    attempted_updates=len(timing),
                    timing_scope='Offline attempted updates including refused/exceptional tails; no flight deadline claim')
                from orbexa.simulation.mission_result import MissionResultStore
                MissionResultStore.save(self.approach, self.directory/'approach')
            except Exception as recording_error:
                self.error['prefix_recording_error'] = str(recording_error)

    def _finalize(self):
        try:
            metrics = collect_study_metrics(self.approach, self.completion, self.runner, self.directory, self.error)
        except Exception as metrics_error:
            metrics = dict(schema='orbexa.nonlinear-study-metrics.v1', success=False, status='METRICS_REFUSED',
                error=self.error, metrics_error=dict(type=type(metrics_error).__name__, message=str(metrics_error)),
                scope='Source artifacts retained; unavailable metrics are not zero or success')
        metrics['wall_time_s'] = time.perf_counter()-self.started
        write_record(self.directory/'study-metrics.json', metrics)
        summary = dict(success=bool(metrics.get('success')), status=metrics.get('status'),
            message=metrics.get('message', self.error), case=self.case, adaptation=self.adaptation, bandwidth=self.bandwidth,
            output_dir=str(self.directory), metrics='study-metrics.json', source_manifest='source-manifest.json',
            study_manifest='study-manifest.json' if (self.directory/'study-manifest.json').is_file() else None)
        write_record(self.directory/'run-result.json', summary)
        return summary
