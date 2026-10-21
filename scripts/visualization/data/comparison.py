"""Verify and describe every arm of a predeclared nonlinear 2-by-2 study.

Matching is a source/input and declaration check. It does not establish a
causal superiority claim or discard an arm because it failed or reached a cap.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from orbexa.visualization.plots.nonlinear import reference_density_coefficient
from orbexa.visualization.plots.canvas import artifact_subplots
from orbexa.visualization.style import _STYLE


ARMS = (('adaptive', 'dynamic'), ('adaptive', 'fixed'), ('frozen', 'dynamic'), ('frozen', 'fixed'))
_COLORS = ('#0072B2', '#56B4E9', '#D55E00', '#D99B00')


def _read(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def _inside(directory, relative):
    path = (directory/relative).resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        raise ValueError('Manifest source/input must be a retained file inside its own arm directory: '+str(relative))
    return path


def _finite(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if np.isfinite(number) else None


def _sum_values(values):
    values = list(values)
    numbers = [_finite(value) for value in values]
    return sum(numbers) if values and all(value is not None for value in numbers) else None


def recorded_outcome_evidence(metrics):
    """Describe recorded termination; never infer physical infeasibility.

    Explicit exceptions and physical-gate records are independent evidence.
    A reported simulated completion still needs the existing delivery verifier.
    """
    status = metrics.get('status', 'UNKNOWN')
    error = metrics.get('error')
    safety = metrics.get('safety') or {}
    failed_gates = [key for key in ('active_target_margins_safe',
                    'pairwise_spacing_safe', 'sampled_tube_containment')
                    if safety.get(key) is False]
    if (metrics.get('continuous_execution') or {}).get('all_intervals_certified') is False:
        failed_gates.append('all_intervals_certified')
    if error is not None:
        termination = 'EXECUTION_ERROR'
    elif status == 'METRICS_REFUSED' or metrics.get('metrics_error') is not None:
        termination = 'METRICS_UNAVAILABLE'
    elif failed_gates:
        termination = 'RECORDED_GATE_FAILURE'
    elif status == 'COMPLETED':
        termination = 'COMPLETION_REPORTED'
    elif status == 'INCOMPLETE':
        termination = 'INCOMPLETE'
    elif status in ('FAILED', 'SETUP_FAILED'):
        termination = 'RECORDED_RUN_FAILURE'
    else:
        termination = 'UNKNOWN'
    complete = (status == 'COMPLETED' and metrics.get('success') is True
                and metrics.get('physical_mission_complete') is True
                and error is None and metrics.get('metrics_error') is None
                and not metrics.get('failures') and not failed_gates)
    markers = {'COMPLETED': 'C', 'INCOMPLETE': 'I', 'FAILED': 'F',
               'SETUP_FAILED': 'S', 'METRICS_REFUSED': 'M'}
    marker = markers.get(status, '?')
    if termination == 'EXECUTION_ERROR':
        marker = 'E'
    elif termination == 'METRICS_UNAVAILABLE':
        marker = 'M'
    elif failed_gates:
        marker = 'F'
    if status == 'COMPLETED' and not complete and marker == 'C':
        marker = '?'
    return dict(termination_class=termination,
        physical_feasibility='SIMULATED_COMPLETION_REPORTED' if complete else 'UNKNOWN',
        success_time_eligible=complete, display_marker=marker,
        execution_error_type=error.get('type', 'UNKNOWN') if isinstance(error, dict)
            else 'UNKNOWN' if error is not None else None,
        recorded_gate_failures=failed_gates,
        scope='Recorded evidence only; a failed/capped/error run is not a physical infeasibility certificate; '
              'reported completion remains subject to existing full verification')


class MatchedNonlinearComparison:
    """Strict complete-grid verifier with descriptive, failure-inclusive plots.

    ``verify()`` reports every mismatch without plotting. ``render()`` writes
    that report and refuses plots if any arm/source/input/declaration differs.
    Setup failures remain grid entries when their prospective copied inputs
    and source are available; missing metrics are never inferred to be zero.
    """
    def __init__(self, arm_directories, output_dir, *, dpi=180):
        self.directories = tuple(Path(value).resolve() for value in arm_directories)
        if not self.directories or len(set(self.directories)) != len(self.directories):
            raise ValueError('Supply each retained arm directory exactly once')
        if not isinstance(dpi, int) or isinstance(dpi, bool) or dpi < 50:
            raise ValueError('dpi must be an integer of at least 50')
        self.output_dir, self.dpi = Path(output_dir), dpi
        self.records, self.report, self.artifacts = {}, None, []
        self.case_ids = ()

    @staticmethod
    def _load(directory):
        source_path, study_path = directory/'source-manifest.json', directory/'study-manifest.json'
        source = _read(source_path)
        if source.get('schema') != 'orbexa.nonlinear-study-source.v1':
            raise ValueError('An explicit nonlinear study source manifest is required')
        package = source.get('package_python_sha256', {})
        if not package or not isinstance(package, dict):
            raise ValueError('Source manifest lacks its complete Python source map')
        retained = {path.relative_to(directory/'runtime'/'orbexa').as_posix()
                    for path in (directory/'runtime'/'orbexa').rglob('*.py') if '__pycache__' not in path.parts}
        if retained != set(package):
            raise ValueError('Retained Python source files differ from the complete manifest map')
        for name, identity in package.items():
            if _sha(_inside(directory, 'runtime/orbexa/'+name)) != identity:
                raise ValueError('Retained Python source hash changed: '+name)
        inputs = source.get('inputs', {})
        if not {'config', 'spacecraft', 'study'} <= set(inputs):
            raise ValueError('Source manifest lacks explicit config/spacecraft/study inputs')
        for name, item in inputs.items():
            if _sha(_inside(directory, item['frozen'])) != item['sha256']:
                raise ValueError('Retained explicit input hash changed: '+name)
        declaration = _read(_inside(directory, inputs['study']['frozen']))
        if declaration.get('schema') != 'orbexa.nonlinear-study.v1' or not declaration.get('study_id'):
            raise ValueError('Copied input does not declare a nonlinear study')
        case_ids = tuple(row['case_id'] for row in declaration['cases'])
        if not case_ids or len(set(case_ids)) != len(case_ids):
            raise ValueError('Copied study input must declare unique case IDs')
        key = (source['selected_case'], source['adaptation'], source['bandwidth'])
        if key[0] not in case_ids or key[1:] not in ARMS:
            raise ValueError('Source manifest selected an undeclared case or arm')
        metrics, summary = _read(directory/'study-metrics.json'), _read(directory/'run-result.json')
        if metrics.get('schema') != 'orbexa.nonlinear-study-metrics.v1' or not isinstance(metrics.get('status'), str):
            raise ValueError('Every retained arm needs explicit status and metrics, including failures/caps')
        if ((summary.get('case'), summary.get('adaptation'), summary.get('bandwidth')) != key or
                summary.get('status') != metrics['status'] or summary.get('success') != metrics.get('success')):
            raise ValueError('Run summary selection/outcome disagrees with its source or metrics')
        study = _read(study_path) if study_path.is_file() else None
        if study is None:
            if metrics['status'] != 'SETUP_FAILED' or (directory/'approach'/'mission_data.json').is_file():
                raise ValueError('A motion-bearing arm lacks its prospective resolved study manifest')
        else:
            required = ('resolved_options', 'design', 'initial_belief', 'initial_target_spin_prior',
                        'model_bounds', 'point_truth', 'reference_clock', 'execution_cap_updates')
            if any(study.get(name) is None for name in required):
                raise ValueError('Resolved prospective manifest omits required profile/prior/design/model/clock fields')
            if (study.get('schema') != declaration['schema'] or study.get('study_id') != declaration['study_id'] or
                    study.get('cases') != declaration['cases'] or study.get('selected_case') != key[0] or
                    study.get('selected_arm') != dict(adaptation=key[1], bandwidth=key[2]) or
                    study.get('factorial_axes') != dict(adaptation=['adaptive', 'frozen'], bandwidth=['dynamic', 'fixed']) or
                    study.get('source_manifest_sha256') != _sha(source_path)):
                raise ValueError('Resolved prospective manifest disagrees with the complete copied declaration/source')
            cap = source.get('steps') if source.get('steps') is not None else study['resolved_options'].get('maximum_updates')
            if study['execution_cap_updates'] != cap:
                raise ValueError('Resolved execution cap differs from the prospective source request')
            declared_case = next(row for row in declaration['cases'] if row['case_id'] == key[0])
            sensor_seed = study['resolved_options'].get('sensor', {}).get('seed')
            if sensor_seed != declared_case['seed'] or isinstance(sensor_seed, bool):
                raise ValueError('Resolved sensor seed must equal its explicitly declared independent case seed')
        approach_path = directory/'approach'/'mission_data.json'
        binding_verified = False
        if approach_path.is_file():
            payload = _read(approach_path)
            approach = payload.get('result', payload)
            metadata = approach.get('metadata', {})
            binding = metadata.get('study_source_binding')
            if binding is not None:
                if (binding.get('source_manifest_sha256') != _sha(source_path) or
                        binding.get('study_manifest_sha256') != _sha(study_path) or
                        (binding.get('selected_case'), binding.get('adaptation'), binding.get('bandwidth')) != key or
                        binding.get('study_id') != declaration['study_id']):
                    raise ValueError('Saved approach source/profile/selection binding differs from this arm')
                binding_verified = True
            elif metrics.get('error') is None:
                raise ValueError('A normal saved approach lost its source/profile binding')
            modes = (metadata.get('adaptation_mode'), metadata.get('bandwidth_mode'))
            exceptional_missing_modes = modes == (None, None) and metrics.get('error') is not None
            if ((modes != key[1:] and not exceptional_missing_modes) or
                    metadata.get('independent_case') != next(row for row in declaration['cases'] if row['case_id'] == key[0])):
                raise ValueError('Saved actual approach differs from the explicitly selected case/arm')
        elif metrics['status'] not in ('SETUP_FAILED', 'FAILED', 'METRICS_REFUSED') or metrics.get('physical_elapsed_s') is not None:
            raise ValueError('An outcome claiming a physical prefix lacks its actual saved approach')
        return dict(key=key, directory=directory, source=source, study=study, declaration=declaration, metrics=metrics,
            source_sha256=_sha(source_path), metrics_sha256=_sha(directory/'study-metrics.json'),
            study_sha256=None if study is None else _sha(study_path), approach_binding_verified=binding_verified)

    def verify(self):
        errors, loaded, records = [], [], {}
        for directory in self.directories:
            try:
                row = self._load(directory)
                if row['key'] in records:
                    raise ValueError('Duplicate case/arm selection '+str(row['key']))
                records[row['key']] = row
                loaded.append(row)
            except (OSError, ValueError, TypeError, KeyError) as error:
                errors.append(dict(directory=str(directory), reason=str(error)))
        reference = loaded[0] if loaded else None
        self.case_ids = () if reference is None else tuple(row['case_id'] for row in reference['declaration']['cases'])
        expected = {(case, *arm) for case in self.case_ids for arm in ARMS}
        if reference is not None:
            source_inputs = lambda row: {name: value['sha256'] for name, value in row['source']['inputs'].items()}
            for row in loaded[1:]:
                if row['source']['package_python_sha256'] != reference['source']['package_python_sha256']:
                    errors.append(dict(directory=str(row['directory']), reason='Python source hashes differ across arms'))
                if row['source'].get('runtime_environment')!=reference['source'].get('runtime_environment'):
                    errors.append(dict(directory=str(row['directory']),reason='Installed runtime versions or native solver binary differ across arms'))
                if source_inputs(row) != source_inputs(reference):
                    errors.append(dict(directory=str(row['directory']), reason='Explicit config/spacecraft/study input hashes differ across arms'))
                if row['declaration'] != reference['declaration']:
                    errors.append(dict(directory=str(row['directory']), reason='Full case/profile/prior study declarations differ'))
            resolved = [row for row in loaded if row['study'] is not None]
            if resolved:
                base = resolved[0]['study']
                common = ('design', 'initial_target_spin_prior', 'model_bounds', 'reference_clock', 'execution_cap_updates')
                def case_normalized_options(study):
                    options = copy.deepcopy(study['resolved_options'])
                    del options['sensor']['seed']
                    return options
                for row in resolved[1:]:
                    for name in common:
                        if row['study'].get(name) != base.get(name):
                            errors.append(dict(directory=str(row['directory']), reason='Resolved prospective '+name+' differs'))
                    if case_normalized_options(row['study']) != case_normalized_options(base):
                        errors.append(dict(directory=str(row['directory']), reason='Resolved options differ after removing only the validated case sensor seed'))
                for case in self.case_ids:
                    cohort = [row for row in resolved if row['key'][0] == case]
                    for row in cohort[1:]:
                        for name in ('initial_belief', 'point_truth', 'resolved_options'):
                            if row['study'].get(name) != cohort[0]['study'].get(name):
                                errors.append(dict(directory=str(row['directory']), reason='Matched case '+name+' differs'))
        missing, extra = expected-set(records), set(records)-expected
        if missing:
            errors.append(dict(reason='Required predeclared case/arm outcomes are missing', selections=sorted(missing)))
        if extra:
            errors.append(dict(reason='Unexpected undeclared case/arm outcomes', selections=sorted(extra)))
        self.records = records
        self.report = dict(schema='orbexa.nonlinear-comparison-verification.v1', matched=bool(reference is not None and not errors),
            study_id=None if reference is None else reference['declaration']['study_id'], cases=list(self.case_ids),
            required_arms=[dict(adaptation=a, bandwidth=b) for a, b in ARMS], expected_outcomes=len(expected),
            retained_outcomes=len(records), errors=errors,
            outcomes=[dict(case=key[0], adaptation=key[1], bandwidth=key[2], status=row['metrics']['status'],
                success=row['metrics'].get('success'), outcome_evidence=recorded_outcome_evidence(row['metrics']),
                directory=str(row['directory']),
                source_manifest_sha256=row['source_sha256'], study_manifest_sha256=row['study_sha256'],
                metrics_sha256=row['metrics_sha256'], resolved_manifest_available=row['study'] is not None,
                approach_binding_verified=row['approach_binding_verified'],
                recorded_physical_duration_s=row['metrics'].get('physical_elapsed_s')) for key, row in sorted(records.items())],
            exposure_semantics='Each failed/capped metric has its own recorded prefix exposure; unequal-duration effort is not a method improvement; only completed full missions have a time-to-success',
            case_seed_normalization=dict(field='resolved_options.sensor.seed',
                rule='Every seed equals its selected declared case.seed; complete resolved options match within each case; only this seed is removed for cross-case comparison',
                declared_seeds={} if reference is None else {case['case_id']: case['seed'] for case in reference['declaration']['cases']}),
            scope='Complete predeclared case-by-four-arm source/input/declaration matching; all failures/caps retained; no superiority or full-mission claim')
        return self.report

    def _save(self, figure, filename, label):
        try:
            figure.tight_layout(rect=(0, .035, 1, .92))
            path = self.output_dir/filename
            figure.savefig(path, dpi=self.dpi)
        finally:
            figure.clear()
        self.artifacts.append((path, label))

    def render(self):
        report = self.verify()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir/'nonlinear-comparison-verification.json'
        path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        if not report['matched']:
            raise ValueError('Nonlinear comparison refused: unmatched declarations/source/grid; see '+str(path))
        self.artifacts = []
        with plt.rc_context(_STYLE):
            self._outcomes()
            self._metrics()
            self._widths()
            self._success_times()
        manifest = dict(verification_sha256=_sha(path), artifacts=[dict(path=p.name, label=label) for p, label in self.artifacts],
            scope='Descriptive matched comparisons; incomplete physical prefixes and unavailable values remain explicit; no case filtering or superiority threshold')
        (self.output_dir/'nonlinear-comparison-plots.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        return list(self.artifacts)

    def _outcomes(self):
        figure, axis = artifact_subplots(figsize=(12, max(3.5, 1.1*len(self.case_ids)+1.8)))
        palette = {'COMPLETED': '#CBE8DE', 'INCOMPLETE': '#F8E7BA', 'FAILED': '#F1C8CC', 'SETUP_FAILED': '#E0D5E8', 'METRICS_REFUSED': '#D9DFE4'}
        for y, case in enumerate(self.case_ids):
            for x, arm in enumerate(ARMS):
                metrics = self.records[(case, *arm)]['metrics']
                status = metrics['status']
                evidence = recorded_outcome_evidence(metrics)
                execution_error = evidence['termination_class'] == 'EXECUTION_ERROR'
                contradictory_completion = status == 'COMPLETED' and not evidence['success_time_eligible']
                color = '#D9DFE4' if execution_error or contradictory_completion else palette.get(status, '#D9DFE4')
                axis.add_patch(plt.Rectangle((x-.47, y-.43), .94, .86, facecolor=color,
                    edgecolor='#687580' if execution_error else color, hatch='xx' if execution_error else None))
                label = status.replace('_', '\n')
                if execution_error:
                    label += '\nexecution error\nfeasibility UNKNOWN'
                elif contradictory_completion:
                    label += '\nconflicting recorded evidence\nfeasibility UNKNOWN'
                axis.text(x, y, label, ha='center', va='center', fontsize=8 if execution_error or contradictory_completion else 9)
        axis.set(xticks=range(4), xticklabels=[a+' / '+b for a, b in ARMS], yticks=range(len(self.case_ids)), yticklabels=self.case_ids,
                 xlim=(-.55, 3.55), ylim=(len(self.case_ids)-.5, -.5))
        axis.tick_params(length=0)
        figure.suptitle(str(self.report['study_id'])+' — every predeclared outcome', fontsize=14)
        figure.text(.03, .015, 'Operational failure is retained; execution errors have UNKNOWN physical feasibility. A completed approach alone is not a completed mission.', fontsize=8)
        self._save(figure, 'nonlinear_comparison_outcomes.png', 'Complete declared case-by-arm outcome matrix, including failed and capped runs.')

    def _bars(self, axis, values, title):
        x, width = np.arange(len(self.case_ids)), .19
        any_known = False
        for number, arm in enumerate(ARMS):
            rows = [self.records[(case, *arm)] for case in self.case_ids]
            heights = [_finite(values(row)) for row in rows]
            any_known |= any(value is not None for value in heights)
            if any(value is not None and value < 0 for value in heights):
                raise ValueError('A recorded nonnegative comparison metric is negative: '+title)
            positions = x+(number-1.5)*width
            bars = axis.bar(positions, [np.nan if value is None else value for value in heights], width=.175, color=_COLORS[number])
            abbreviations = {'COMPLETED': 'C', 'INCOMPLETE': 'I', 'FAILED': 'F', 'SETUP_FAILED': 'S', 'METRICS_REFUSED': 'M'}
            for position, value, row, bar in zip(positions, heights, rows, bars):
                status = row['metrics']['status']
                evidence = recorded_outcome_evidence(row['metrics'])
                execution_error = evidence['termination_class'] == 'EXECUTION_ERROR'
                short = evidence['display_marker']
                if execution_error:
                    bar.set_hatch('xx')
                elif not evidence['success_time_eligible']:
                    bar.set_hatch('//')
                axis.annotate(('N/A '+short) if value is None else short, (position, 0 if value is None else value),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=7,
                    rotation=90 if value is None else 0, color='#687580')
        axis.set(xticks=x, xticklabels=self.case_ids, title=title, xlim=(-.6, len(self.case_ids)-.4))
        axis.tick_params(axis='x', labelsize=8)
        axis.grid(True, axis='y', alpha=.6)
        axis.set_axisbelow(True)
        axis.set_ylim(bottom=0)
        low, high = axis.get_ylim()
        axis.set_ylim(low, high*1.15)
        if not any_known:
            axis.set_ylim(0, 1)
            axis.set_yticks([])

    def _metrics(self):
        def fuel(row):
            return _sum_values((row['metrics'].get('fuel_bound') or {}).get('translation_fuel_debit_kg', {}).values())
        def effort(row):
            return _sum_values(value.get('applied_input_squared_integral_per_reference_anomaly') for value in row['metrics'].get('effort', {}).values())
        def omega(row):
            values = [value.get('final_position_radius_m') for value in row['metrics'].get('tubes', {}).values()]
            return max(float(np.linalg.norm(value)) for value in values) if values and all(value is not None for value in values) else None
        def deadline(row):
            metrics = row['metrics']
            timings = metrics.get('update_wall_timing', [])
            if (timings and all(value.get('full_update_deadline_status') in ('WITHIN', 'MISS') for value in timings)
                    and all(isinstance(value.get('deadline_miss_full_update'), bool) for value in timings)):
                return sum(value['deadline_miss_full_update'] for value in timings)
            return None
        fields = (('Recorded physical duration (s; prefixes retained)', lambda row: _finite(row['metrics'].get('physical_elapsed_s'))),
                  ('Translation propellant upper debit (kg)', fuel),
                  ('Delivered reference-input effort (m²/rad³)', effort),
                  ('Final largest position-tube radius ‖Ω‖ (m)', omega),
                  ('Native solver attempts, all outcomes', lambda row: _finite(row['metrics'].get('native_attempts', {}).get('total'))),
                  ('Recorded full-update offline deadline misses', deadline))
        figure, axes = artifact_subplots(3, 2, figsize=(14, 11))
        figure.suptitle(str(self.report['study_id'])+' — recorded metrics across every arm', fontsize=14)
        figure.legend(handles=[Patch(color=color, label=a+' / '+b) for (a, b), color in zip(ARMS, _COLORS)], loc='upper center', ncol=4, bbox_to_anchor=(.5, .955), fontsize=9)
        for axis, (title, function) in zip(axes.flat, fields):
            self._bars(axis, function, title)
        figure.text(.02, .017, 'C completed / I incomplete / F run failed / E execution error (feasibility UNKNOWN) / S setup / M metrics / ? unknown. N/A is unavailable.', fontsize=8)
        figure.text(.02, .004, 'Prefix exposures differ: these duration/effort bars are not success times or method improvements. No common-window effort is inferred.', fontsize=8)
        self._save(figure, 'nonlinear_comparison_metrics.png', 'All-arm recorded durations, fuel bounds, delivered effort, tubes, native attempts and explicitly full-update deadline misses.')

    def _widths(self):
        def width(row, key):
            value = _finite(row['metrics'].get('set_width', {}).get(key, {}).get('final'))
            if value is None or key == 'eccentricity':
                return value
            model = None if row['study'] is None else row['study'].get('model_bounds')
            if model is None:
                return None
            return value/reference_density_coefficient(model)
        figure, axes = artifact_subplots(3, 1, figsize=(12, 9))
        figure.suptitle(str(self.report['study_id'])+' — final recorded marginal widths', fontsize=14)
        figure.legend(handles=[Patch(color=color, label=a+' / '+b) for (a, b), color in zip(ARMS, _COLORS)], loc='upper center', ncol=4, bbox_to_anchor=(.5, .955), fontsize=9)
        for axis, key, label in zip(axes, ('eccentricity', 'alpha', 'beta'),
                ('Initial osculating eccentricity width', 'Reference-mapped target CdA/m width (m²/kg)', 'Reference-mapped chaser CdA/m width (m²/kg)')):
            self._bars(axis, lambda row, key=key: width(row, key), label)
        figure.text(.02, .01, 'Reference density maps alpha/beta to sigma=CdA/m. Prefix marginal widths are not set volumes; E denotes execution error, feasibility UNKNOWN.', fontsize=8)
        self._save(figure, 'nonlinear_comparison_set_widths.png', 'All-arm final initial-eccentricity and reference-mapped physical drag marginal widths, including prefixes.')

    def _success_times(self):
        figure, axis = artifact_subplots(figsize=(12, 5))
        figure.suptitle(str(self.report['study_id'])+' — time to full mission success', fontsize=14)
        figure.legend(handles=[Patch(color=color, label=a+' / '+b) for (a, b), color in zip(ARMS, _COLORS)],
                      loc='upper center', ncol=4, bbox_to_anchor=(.5, .94), fontsize=9)
        def successful_time(row):
            metrics = row['metrics']
            evidence = recorded_outcome_evidence(metrics)
            return metrics.get('physical_elapsed_s') if evidence['success_time_eligible'] else None
        self._bars(axis, successful_time, 'Full simulated mission success time (s); other outcomes remain N/A')
        figure.text(.02, .01, 'Every arm remains visible. Failed/capped/error prefix duration is not a success time; E means execution error, feasibility UNKNOWN.', fontsize=8)
        self._save(figure, 'nonlinear_comparison_time_to_success.png', 'Full-mission success times only; every failed, capped or unknown arm remains explicitly unavailable.')
