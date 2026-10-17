"""Static diagnostics from aligned, saved nonlinear mission observations.

Plots are numerical readouts of the retained records. They do not replay a
controller, create contact/lifecycle states, or establish a new certificate.
"""
from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from orbexa.visualization.plot_canvas import artifact_subplots
from orbexa.visualization.style import _STYLE, chaser_color


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _array(value, shape, label):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(label+' must have finite, exactly aligned shape '+str(shape))
    return result


def _box(value, dimension, label):
    if not isinstance(value, Mapping):
        raise ValueError(label+' lacks its saved interval box')
    lower, upper = (_array(value.get(name), (dimension,), label+' '+name) for name in ('lower', 'upper'))
    if np.any(lower > upper):
        raise ValueError(label+' has reversed bounds')
    return lower, upper


def _number(value):
    return float(Fraction(value)) if isinstance(value, str) else float(value)


def reference_density_coefficient(model):
    """Declared alpha=(rho_reference*radius_reference/2)*sigma mapping."""
    atmosphere = model['atmosphere']
    rho = atmosphere['rho0_kg_m3']
    if len(rho) != 2 or Fraction(rho[0]) != Fraction(rho[1]):
        raise ValueError('Physical marginal widths require a point reference density')
    coefficient = _number(rho[0])*_number(atmosphere['reference_radius_m'])/2
    if not np.isfinite(coefficient) or coefficient <= 0:
        raise ValueError('Physical marginal widths require a positive reference density/radius')
    return coefficient


class NonlinearMissionPlots:
    """Export saved navigation, tube, parameter and execution diagnostics.

    ``render()`` returns ``[(path, label), ...]`` and writes an explicit
    alignment/skipped-artifact manifest. An unavailable or misaligned series
    is refused, never truncated, interpolated, or replaced with a zero series.
    """
    def __init__(self, result, output_dir, *, dpi=180, lifecycle=None):
        if not isinstance(dpi, int) or isinstance(dpi, bool) or dpi < 50:
            raise ValueError('dpi must be an integer of at least 50')
        self.result, self.output_dir, self.dpi = result, Path(output_dir), dpi
        self.metadata = _get(result, 'metadata', {})
        if self.metadata.get('model_version') != 'nonlinear-reference-relative-v1':
            raise ValueError('Nonlinear plots require saved NonlinearMissionRunner data')
        if self.metadata.get('state_frame') != 'reference-orbit-frame-v1':
            raise ValueError('Nonlinear plots require explicitly recorded reference-orbit-frame states')
        self.ids = tuple(_get(result, 'actual_trajectories', {}))
        self.q = np.asarray(_get(result, 'anom_history', []), dtype=float)
        if self.q.ndim != 1 or not len(self.q) or not np.all(np.isfinite(self.q)) or np.any(np.diff(self.q) <= 0):
            raise ValueError('Saved reference anomaly nodes must be finite and strictly increasing')
        self.time = _array(self.metadata.get('elapsed_time_seconds'), self.q.shape, 'Saved physical times')
        if np.any(np.diff(self.time) <= 0):
            raise ValueError('Saved physical times must increase with reference anomaly')
        if not self.ids:
            raise ValueError('Saved nonlinear data has no declared chaser trajectories')
        self.artifacts, self.skipped_artifacts, self.alignment = [], {}, {}
        self.source, self.lifecycle, self.lifecycle_source = None, lifecycle, None

    @classmethod
    def from_directory(cls, directory, output_dir, **kwargs):
        from orbexa.simulation.mission_result import MissionResultStore
        directory = Path(directory)
        path = directory/'approach'/'mission_data.json' if directory.is_dir() else directory
        lifecycle_path = directory/'completion'/'lifecycle_data.json' if directory.is_dir() else None
        loaded_lifecycle = lifecycle_path is not None and lifecycle_path.is_file() and 'lifecycle' not in kwargs
        if loaded_lifecycle:
            with lifecycle_path.open(encoding='utf-8') as stream:
                value = json.load(stream)
            if value.get('schema') != 'orbexa.physical-mission-lifecycle.v1':
                raise ValueError('Unknown saved completion lifecycle schema')
            kwargs['lifecycle'] = value['result']
        result = cls(MissionResultStore.load(path), output_dir, **kwargs)
        result.source = dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if loaded_lifecycle:
            result.lifecycle_source = dict(path=str(lifecycle_path.resolve()), sha256=hashlib.sha256(lifecycle_path.read_bytes()).hexdigest())
        return result

    def _save(self, figure, filename, label):
        try:
            figure.tight_layout(rect=(0, .035, 1, .95))
            path = self.output_dir/filename
            figure.savefig(path, dpi=self.dpi)
        finally:
            figure.clear()
        self.artifacts.append((path, label))

    def _decorate(self, figure, title):
        complete = bool(_get(self.result, 'success') and self.metadata.get('mission_complete'))
        figure.suptitle(title+' — '+('approach complete' if complete else 'recorded approach prefix'), fontsize=13)
        figure.text(.02, .008, 'Saved simulated records; unavailable series are refused. No flight or new safety certificate.', fontsize=7, color='#687580')

    @staticmethod
    def _axes(axes, xlabel='Elapsed simulated physical time (s)'):
        for axis in np.asarray(axes, dtype=object).flat:
            axis.grid(True, alpha=.6)
            axis.set_xlabel(xlabel)
            if axis.get_legend_handles_labels()[0]:
                axis.legend(fontsize=7, ncol=4)

    def render(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts, self.skipped_artifacts, self.alignment = [], {}, {}
        with plt.rc_context(_STYLE):
            for name, method in (('nonlinear_navigation.png', self._navigation),
                                 ('nonlinear_tubes_bandwidth.png', self._tubes),
                                 ('nonlinear_physical_set_widths.png', self._widths),
                                 ('nonlinear_execution_timeline.png', self._execution),
                                 ('nonlinear_finite_contact.png', self._contact)):
                try:
                    method()
                except (ValueError, KeyError, TypeError, IndexError) as error:
                    self.skipped_artifacts[name] = str(error)
        record = dict(schema='orbexa.nonlinear-plot-manifest.v1', source=self.source, lifecycle_source=self.lifecycle_source,
            model=self.metadata.get('model_version'), clock_fingerprint=self.metadata.get('reference_clock_fingerprint'),
            approach_complete=bool(_get(self.result, 'success') and self.metadata.get('mission_complete')),
            recorded_nodes=len(self.q), artifacts=[dict(path=path.name, label=label) for path, label in self.artifacts],
            skipped_artifacts=self.skipped_artifacts, alignment=self.alignment,
            scope='Numerical diagnostics from saved matched records; no missing data or completed lifecycle is synthesized')
        (self.output_dir/'nonlinear-plot-manifest.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        return list(self.artifacts)

    def navigation_series(self):
        """Return matched numerical error/bound images in m and m/s."""
        from orbexa.core.reference_frame import ReferenceClockFrame
        record = self.metadata['reference_clock']
        if record.get('model') != 'commanded-reference-orbit-clock-v1' or record.get('frame') != 'reference-orbit-frame-v1':
            raise ValueError('Unknown saved clock/frame cannot define a physical navigation error')
        fields = ('semi_major_axis_m', 'eccentricity', 'alpha', 'mu_m3_s2', 'phase', 'epoch_orientation', 'q_domain')
        clock = ReferenceClockFrame(**{key: record[key] for key in fields})
        actual = {key: _array(_get(self.result, 'actual_trajectories')[key], (len(self.q), 6), key+' actual states') for key in self.ids}
        rows, refused = [], []
        seen = set()
        for row in self.metadata.get('navigation_history', []):
            if (row.get('clock_fingerprint') != self.metadata.get('reference_clock_fingerprint') or
                    row.get('frame') != 'reference-orbit-frame-v1'):
                raise ValueError('Navigation record clock/frame differs from the saved physical-state coordinates')
            exact_q = Fraction(row['exact_q'])
            matches = np.flatnonzero(self.q == float(exact_q))
            if len(matches) != 1 or exact_q in seen:
                raise ValueError('Navigation nodes must match one unique saved truth node, without interpolation')
            seen.add(exact_q)
            index = int(matches[0])
            if row.get('status') != 'VALID':
                refused.append(dict(exact_q=str(exact_q), status=row.get('status'), reason=row.get('reason')))
                continue
            time_lower, time_upper = _box(row['physical_time_box'], 1, 'Navigation physical time')
            tolerance = 1e-9+1e-10*abs(self.time[index])
            if not time_lower[0]-tolerance <= self.time[index] <= time_upper[0]+tolerance:
                raise ValueError('Navigation physical time box disagrees with its matched saved truth node')
            boxes, points = row['relative_state_boxes'], row['point_relative_by_agent']
            if set(boxes) != set(self.ids) or set(points) != set(self.ids):
                raise ValueError('Navigation row must name the exact recorded fleet')
            per_agent = {}
            rate = clock.point(exact_q).rate_s_inv
            for key in self.ids:
                lo, hi = _box(boxes[key], 6, key+' navigation')
                point = _array(points[key], (6,), key+' observed point')
                if np.any(point < lo) or np.any(point > hi):
                    raise ValueError('Observed navigation point lies outside its own saved box')
                radius = np.maximum(point-lo, hi-point)
                error = actual[key][index]-point
                zp = np.array((-error[1], error[0], 0.))
                velocity_radius = radius[3:]+np.array((radius[1], radius[0], 0.))
                ratio = np.divide(np.abs(error), radius, out=np.full(6, np.inf), where=radius > 0)
                ratio[(radius == 0) & (error == 0)] = 0
                per_agent[key] = dict(position_error_m=float(np.linalg.norm(error[:3])),
                    position_bound_m=float(np.linalg.norm(radius[:3])),
                    velocity_error_m_s=float(rate*np.linalg.norm(error[3:]+zp)),
                    velocity_bound_m_s=float(rate*np.linalg.norm(velocity_radius)),
                    component_ratio=None if not np.all(np.isfinite(ratio)) else float(np.max(ratio)),
                    zero_width_violation=bool(np.any(~np.isfinite(ratio))))
            rows.append(dict(index=index, exact_q=str(exact_q), time_s=float(self.time[index]), agents=per_agent))
        if not rows:
            raise ValueError('No VALID navigation row matches retained numerical truth')
        if any(a['index'] >= b['index'] for a, b in zip(rows, rows[1:])):
            raise ValueError('Matched navigation records are not in strictly increasing node order')
        self.alignment['navigation'] = dict(matched_nodes=len(rows), saved_truth_nodes=len(self.q), refused_statuses=refused,
            exact_q=[row['exact_q'] for row in rows],
            velocity_transform='nu(q)*(error_reference_derivative + Z*error_position); upper image nu*norm(rv+abs(Z)*rp)',
            semantics='Bounds are numerical images of saved boxes, not a new outward interval proof')
        return rows

    def _navigation(self):
        rows = self.navigation_series()
        time = self.time
        indices = [row['index'] for row in rows]
        def matched(values):
            result = np.full(len(self.q), np.nan)
            result[indices] = values
            return result
        figure, axes = artifact_subplots(3, 1, figsize=(10.8, 10), sharex=True)
        self._decorate(figure, 'Navigation error and its own declared box')
        for number, key in enumerate(self.ids):
            color = chaser_color(number)
            data = [row['agents'][key] for row in rows]
            for axis, name in zip(axes[:2], ('position', 'velocity')):
                unit = 'm' if name == 'position' else 'm_s'
                axis.plot(time, matched([1000*item[name+'_error_'+unit] for item in data]), color=color, label=key, marker='.', markersize=3)
                axis.plot(time, matched([1000*item[name+'_bound_'+unit] for item in data]), color=color, linestyle='--', alpha=.75)
            axes[2].plot(time, matched([np.nan if item['component_ratio'] is None else item['component_ratio'] for item in data]), color=color, label=key, marker='.', markersize=3)
            if any(item['zero_width_violation'] for item in data):
                axes[2].text(.01, .96-.05*number, key+': nonzero error at a zero-width bound', transform=axes[2].transAxes, color='#BA4D53', fontsize=8)
        axes[0].set_ylabel('Position norm (mm)')
        axes[1].set_ylabel('Physical velocity norm (mm/s)')
        axes[2].set_ylabel('Max component |error| / own radius')
        axes[2].axhline(1, color='#BA4D53', linestyle='--', linewidth=1)
        axes[0].set_title('Solid: recorded error; dashed: own box bound image. Missing/non-VALID nodes are gaps.', fontsize=10)
        self._axes(axes)
        self._save(figure, 'nonlinear_navigation.png', 'Matched navigation errors against saved position/physical velocity bounds and componentwise coverage.')

    def _tubes(self):
        diagnostics = _get(self.result, 'tube_diagnostics', {})
        if set(diagnostics) != set(self.ids):
            raise ValueError('Tube diagnostics do not name the exact saved fleet')
        data = {key: {name: _array(diagnostics[key].get(name), (len(self.q), 3), key+' '+name)
                      for name in ('omega', 'phi', 'bandwidth')} for key in self.ids}
        if any(np.any(values < 0) for agent in data.values() for values in agent.values()):
            raise ValueError('Saved tube radii and bandwidths must be nonnegative')
        figure, axes = artifact_subplots(3, 1, figsize=(10.8, 9), sharex=True)
        self._decorate(figure, 'Dynamic tube and bandwidth records')
        for number, key in enumerate(self.ids):
            for axis, name in zip(axes, ('omega', 'phi', 'bandwidth')):
                values = data[key][name]
                axis.plot(self.time, np.max(values, axis=1) if name == 'bandwidth' else np.linalg.norm(values, axis=1), color=chaser_color(number), label=key)
        for axis, label in zip(axes, ('Position radius ‖Ω‖ (m)', 'Sliding radius ‖Φ‖ (m/rad)', 'Largest component bandwidth (1/rad)')):
            axis.set_ylabel(label)
        self.alignment['tubes'] = dict(nodes=len(self.q), fleet=list(self.ids), semantics='Each agent uses exactly its own saved node arrays')
        self._axes(axes)
        self._save(figure, 'nonlinear_tubes_bandwidth.png', 'Recorded per-agent position/sliding tube radii and largest component bandwidth.')

    def _widths(self):
        history = _get(self.result, 'feasible_set_history', [])
        if len(history) != len(self.q):
            raise ValueError('Feasible-set history must have one record per saved outer node')
        adaptation = self.metadata['physical_adaptation']
        coefficient = reference_density_coefficient(adaptation['physical_prior'])
        keys = ('eccentricity', 'alpha', 'beta')
        bands = {key: _array([row[key] for row in history], (len(self.q), 2), key+' feasible sets') for key in keys}
        if any(np.any(value[:, 1] < value[:, 0]) for value in bands.values()):
            raise ValueError('Saved feasible-set interval endpoints are reversed')
        figure, axes = artifact_subplots(3, 1, figsize=(10.8, 8.5), sharex=True)
        self._decorate(figure, 'Recorded physical marginal widths')
        for axis, key, label in zip(axes, keys, ('Initial osculating\neccentricity width', 'Reference-mapped target\nCdA/m width (m²/kg)', 'Reference-mapped chaser\nCdA/m width (m²/kg)')):
            width = bands[key][:, 1]-bands[key][:, 0]
            axis.step(self.time, width if key == 'eccentricity' else width/coefficient, where='post', color='#0072B2')
            axis.set_ylabel(label, fontsize=9)
        self.alignment['physical_set_widths'] = dict(nodes=len(self.q), coefficient_per_sigma=coefficient,
            semantics='Controller reference-density marginal intervals mapped with alpha/beta=(rho_reference*radius_reference/2)*sigma; not the volume of retained candidate sets')
        self._axes(axes)
        self._save(figure, 'nonlinear_physical_set_widths.png', 'Initial eccentricity and reference-density mapped CdA/m marginal widths; no candidate-set volume claim.')

    def _execution(self):
        records = self.metadata.get('fleet_hold_records', [])
        if not records:
            raise ValueError('No exact completed hold index/records were retained')
        segments, previous = [], None
        for row in records:
            start, end = Fraction(row['exact_start']), Fraction(row['exact_end'])
            if end <= start or (previous is not None and start != previous):
                raise ValueError('Saved completed holds must form a contiguous increasing exact clock prefix')
            source = row.get('command_source', row.get('decision', {}).get('source'))
            if not isinstance(source, str):
                raise ValueError('Saved completed hold has no explicit command source')
            if Fraction(row.get('hold_width', row.get('decision', {}).get('hold_width'))) != end-start:
                raise ValueError('Saved completed hold width differs from its exact clock endpoints')
            if segments and segments[-1][2] == source:
                segments[-1] = (segments[-1][0], end, source)
            else:
                segments.append((start, end, source))
            previous = end
        figure, axes = artifact_subplots(2, 1, figsize=(11, 7.5))
        self._decorate(figure, 'Actual held command sources and offline update timing')
        colors = {'PERFORMANCE': '#0072B2', 'FALLBACK_TRACK': '#D55E00', 'FALLBACK_HOLD': '#D99B00'}
        sources = tuple(dict.fromkeys(segment[2] for segment in segments))
        for index, source in enumerate(sources):
            values = [(float(start), float(end-start)) for start, end, value in segments if value == source]
            axes[0].broken_barh(values, (index-.35, .7), facecolors=colors.get(source, '#687580'))
        axes[0].set_yticks(range(len(sources)), sources)
        axes[0].set_xlabel('Commanded reference anomaly q (rad); exact contiguous hold endpoints')
        axes[0].set_xlim(float(segments[0][0]), float(segments[-1][1]))
        timing = self.metadata.get('update_wall_timing', [])
        full = bool(timing) and all('full_attempt_wall_s' in row for row in timing)
        if timing:
            updates = _array([row['update'] for row in timing], (len(timing),), 'Update timing indices')
            if np.any(np.diff(updates) <= 0):
                raise ValueError('Saved update wall timings must have unique increasing update indices')
            field = 'full_attempt_wall_s' if full else 'planning_and_proof_wall_s'
            wall = _array([row[field] for row in timing], (len(timing),), field)
            duration = np.asarray([row.get('physical_outer_duration_s', np.nan) for row in timing], dtype=float)
            axes[1].plot(updates, wall, marker='.', color='#0072B2', label='Full attempt including qualification' if full else 'Planner/proof only (historical timing scope)')
            axes[1].plot(updates, duration, linestyle='--', color='#D55E00', label='Scheduled physical outer period' if full else 'Recorded simulated outer duration')
            axes[1].set(xlabel='Attempted outer update index', ylabel='Offline wall time / physical duration (s)')
            axes[1].legend(fontsize=8)
        else:
            axes[1].text(.5, .5, 'No update wall-timing records; no deadline count inferred', ha='center', transform=axes[1].transAxes)
            axes[1].set_axis_off()
        self.alignment['execution'] = dict(completed_holds=len(records), exact_start=str(segments[0][0]), exact_end=str(segments[-1][1]),
            update_attempts=len(timing), wall_timing_scope='full_attempt' if full else 'planner_only' if timing else 'unavailable',
            full_update_deadline_status_counts={status: sum(row.get('full_update_deadline_status') == status for row in timing)
                                               for status in ('WITHIN', 'MISS', 'UNKNOWN')} if full else None,
            semantics='Actual shared fleet hold sources; timing is offline evidence, not a flight deadline guarantee')
        for axis in axes:
            axis.grid(True, alpha=.6)
        self._save(figure, 'nonlinear_execution_timeline.png', 'Exact completed hold command sources and explicitly scoped offline planning/update timing.')

    def _contact(self):
        from orbexa.simulation.nonlinear_archive import semantic_sha256
        if self.lifecycle is None:
            raise ValueError('No actual completion lifecycle was supplied; contact qualification records are not substituted')
        approach = _get(self.lifecycle, 'approach')
        source = self.result if isinstance(self.result, Mapping) else vars(self.result)
        if approach is None or semantic_sha256(approach) != semantic_sha256(source):
            raise ValueError('Completion lifecycle must bind the exact plotted saved approach')
        capture = _get(self.lifecycle, 'capture', _get(self.lifecycle, 'attachment'))
        contact = _get(capture, 'audit', {}).get('finite_contact')
        if not contact or contact.get('audit', {}).get('model') != 'finite_coupled_rigid_body_compliant_adhesive.v1':
            raise ValueError('This actual lifecycle has no supported finite-contact trace')
        history, audit = contact['history'], contact['audit']
        count = len(history)
        if count < 2:
            raise ValueError('A finite-contact trace requires at least two recorded physical states')
        times = _array([row['time_s'] for row in history], (count,), 'Contact physical times')
        if np.any(np.diff(times) <= 0) or times[0] < self.time[-1]-1e-8:
            raise ValueError('Contact trace clock is reversed or precedes its actual approach endpoint')
        ledger = _array([row['ledger'] for row in history], (count, 19), 'Contact v1 cumulative ledger')
        kinetic = _array([row['kinetic_energy_j'] for row in history], (count,), 'Contact kinetic energy')
        potential = _array([row['contact_potential_j'] for row in history], (count,), 'Contact elastic potential')
        for row in history:
            if set(row['pads']) != set(self.ids):
                raise ValueError('Finite-contact trace must name the exact plotted fleet')
        pads = {key: dict(
            gap=_array([row['pads'][key]['gap_m'] for row in history], (count,), key+' gap'),
            force=_array([row['pads'][key]['force_eci_n'] for row in history], (count, 3), key+' contact force'),
            extension=_array([row['pads'][key]['extension_body_m'] for row in history], (count, 3), key+' extension'),
            activation=_array([row['pads'][key]['activation'] for row in history], (count,), key+' bond activation')) for key in self.ids}
        if any(np.any((row['activation'] < -1e-12) | (row['activation'] > 1+1e-12)) for row in pads.values()):
            raise ValueError('Finite-contact activation is outside its declared unit interval')
        elapsed = times-times[0]
        change = kinetic+potential-kinetic[0]-potential[0]
        work = np.sum(ledger[:, 12:16], axis=1)-np.sum(ledger[:, 16:19], axis=1)
        figure, axes = artifact_subplots(4, 2, figsize=(13, 14), sharex=True)
        figure.suptitle('Actual saved finite contact — '+('latched' if contact.get('success') else 'not latched'), fontsize=14)
        for number, key in enumerate(self.ids):
            row, color = pads[key], chaser_color(number)
            for axis, values in ((axes[0, 0], 1000*row['gap']), (axes[0, 1], np.linalg.norm(row['force'], axis=1)),
                                 (axes[1, 0], 1000*np.linalg.norm(row['extension'], axis=1)), (axes[1, 1], row['activation'])):
                axis.plot(elapsed, values, color=color, label=key)
        for axis, label in ((axes[0, 0], 'Signed pad gap (mm)'), (axes[0, 1], 'Pad force norm (N)'),
                            (axes[1, 0], 'Finite tip extension norm (mm)'), (axes[1, 1], 'Adhesive bond activation')):
            axis.set_ylabel(label)
        axes[0, 0].axhline(0, color='#687580', linewidth=.8, linestyle='--')
        axes[1, 1].set_ylim(-.02, 1.05)
        axes[2, 0].plot(elapsed, change, label='Δ(kinetic + elastic)', color='#0072B2')
        axes[2, 0].plot(elapsed, work, label='External + flux + positioner + engage − dissipation', color='#D55E00', linestyle='--')
        axes[2, 0].set_ylabel('Contact-frame energy change (J)')
        axes[2, 1].plot(elapsed, change-work, color='#0072B2', label='Energy balance residual')
        axes[2, 1].axhline(0, color='#687580', linewidth=.8)
        axes[2, 1].set_ylabel('Energy balance residual (J)')
        for axis, start, label in ((axes[3, 0], 0, 'Cumulative linear impulse norm (kg m/s)'),
                                   (axes[3, 1], 6, 'Cumulative angular impulse norm (kg m²/s)')):
            axis.plot(elapsed, np.linalg.norm(ledger[:, start:start+3], axis=1), label='External wrench integral', color='#0072B2')
            axis.plot(elapsed, np.linalg.norm(ledger[:, start+3:start+6], axis=1), label='Fuel flux integral', color='#D55E00', linestyle='--')
            axis.set_ylabel(label, fontsize=9)
        for axis, field in ((axes[3, 0], 'linear_momentum_balance_residual_kg_m_s'),
                            (axes[3, 1], 'angular_momentum_balance_residual_kg_m2_s')):
            axis.text(.02, .96, 'Recorded final balance residual: '+format(float(audit[field]), '.3g'),
                      transform=axis.transAxes, va='top', fontsize=8)
        self._axes(axes, 'Elapsed finite-contact physical time (s)')
        figure.text(.02, .009, 'Original translating nonrotating contact frame and fuel/work ledgers. Representative simulated pad model; no hardware/flight qualification.', fontsize=8)
        self.alignment['finite_contact'] = dict(nodes=count, physical_start_time_s=float(times[0]), physical_end_time_s=float(times[-1]),
            lifecycle_approach_sha256=semantic_sha256(approach), contact_model=audit['model'],
            contact_success=bool(contact.get('success')), balance_frame=audit.get('balance_frame'),
            ledger_semantics='v1 columns: external P, fuel P, external H, fuel H, external/fuel/positioner/engagement work, compression/friction/bond dissipation',
            scope='Actual lifecycle trace only; no standalone contact qualification is imported')
        self._save(figure, 'nonlinear_finite_contact.png', 'Actual lifecycle finite-contact force, gap, extension, activation and cumulative energy/impulse/flux ledger traces.')
