"""Bind and authenticate the complete saved nonlinear hold journal.

Authentication yields one hold at a time so resource checks remain interleaved
with journal checks in their original execution order.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, zip_longest

import numpy as np

from orbexa.control.certification.reachability import IntervalBox
from orbexa.control.fallback.refuge import fingerprint as exact_fingerprint
from orbexa.core.dynamics.relative import NonlinearModelBounds
from orbexa.core.dynamics.reference_frame import ReferenceClockFrame, exact
from orbexa.simulation.approach.records.checkpoint import digest, plain
from orbexa.simulation.nonlinear.records.archive import HOLD_INDEX_SCHEMA, NonlinearHoldArchive, hold_index, semantic_sha256
from orbexa.simulation.completion.nonlinear.boundary import capture_truth_from_nonlinear
from orbexa.simulation.completion.nonlinear.contracts import _box, _compact_hold, _identity, _pairs, _require


@dataclass(frozen=True)
class NonlinearJournalBinding:
    model: NonlinearModelBounds
    clock: ReferenceClockFrame
    records: NonlinearHoldArchive | tuple[dict, ...]
    source_hash: str
    journal_hash: str
    actuator_id: str
    sensor_id: str


@dataclass(frozen=True)
class NonlinearJournalTimeline:
    ids: tuple[str, ...]
    start: Fraction
    end: Fraction
    samples: np.ndarray


@dataclass(frozen=True)
class AuthenticatedNonlinearHold:
    exact_start: Fraction
    exact_end: Fraction
    input_boxes: dict[str, IntervalBox]
    actual_inputs: dict[str, tuple]

    def input_box(self, agent_id):
        """Check delivered support at the original per-body resource boundary."""
        box = self.input_boxes[agent_id]
        _require(box.contains(IntervalBox.point(self.actual_inputs[agent_id])),
                 agent_id+': delivered actuator value escapes its support')
        return box


def bind_nonlinear_journal(approach, *, model, clock, hold_records,
                           expected_actuator_id=None, expected_sensor_id=None):
    capture_truth_from_nonlinear(approach, model=model, clock=clock)
    records = (hold_records if isinstance(hold_records, NonlinearHoldArchive) else
               tuple(copy.deepcopy(plain(record)) for record in hold_records))
    _require(len(records) > 0, 'complete full nonlinear hold records are required')
    saved = approach.metadata.get('fleet_hold_records', ())
    _require(len(saved) == len(records), 'full journal hold count differs from the saved approach')
    for row, value in zip_longest(records, saved):
        projection = (None if row is None or value is None else
                      hold_index(row) if value.get('schema') == HOLD_INDEX_SCHEMA else _compact_hold(row))
        _require(row is not None and value is not None and digest(projection) == digest(value),
                 'full journal differs from the saved compact approach')
    source_hash = semantic_sha256(vars(approach))
    journal_hash = records.identity if isinstance(records, NonlinearHoldArchive) else digest(records)
    actuator_id = _identity(expected_actuator_id or saved[0]['actuator_realization_id'], 'actuator realization')
    sensor_id = _identity(expected_sensor_id or saved[0]['sensor_profile_id'], 'sensor profile')

    return NonlinearJournalBinding(model, clock, records, source_hash, journal_hash, actuator_id, sensor_id)
def verify_nonlinear_timeline(approach, *, binding):
    current_journal = binding.records.identity if isinstance(binding.records, NonlinearHoldArchive) else digest(binding.records)
    _require(semantic_sha256(vars(approach)) == binding.source_hash and current_journal == binding.journal_hash,
             'bound approach or full journal changed')
    metadata, ids = approach.metadata, tuple(approach.actual_trajectories)
    certificate = metadata['continuous_execution_certificate']
    _require(certificate.get('method') == 'nonlinear-sensed-fleet-admission-v1' and
             certificate.get('all_intervals_certified') is True and
             certificate.get('admitted_fleet_holds') == len(binding.records), 'not every retained hold was admitted')
    start = exact(metadata['fleet_hold_records'][0]['exact_start'])
    end = exact(certificate['exact_end_anomaly'])
    _require(abs(float(start)-float(approach.anom_history[0])) < 1e-10, 'journal release clock changed')
    samples = np.asarray(approach.anom_history, dtype=float)
    _require(samples.ndim == 1 and len(samples) >= 2 and np.all(np.isfinite(samples)) and
             np.all(np.diff(samples) > 0) and all(np.asarray(approach.actual_trajectories[key]).shape ==
             (len(samples), 6) for key in ids), 'saved actual fleet timeline is invalid')
    return NonlinearJournalTimeline(ids, start, end, samples)
def verified_nonlinear_holds(approach, *, binding, timeline, body_radii):
    metadata, ids = approach.metadata, timeline.ids
    start, end, samples = timeline.start, timeline.end, timeline.samples
    q, sample_index, decision_ids = start, 1, set()
    model_bounds = metadata['nonlinear_physical_bounds']
    component_ids = tuple(value['id'] for value in approach.target['geometry']['components'])
    expected_separations = {('target_component', key, component) for key in ids for component in component_ids}
    expected_separations.update(('peer', *sorted(pair)) for pair in combinations(ids, 2))
    expected_target_id = dict(model_bounds)
    for key in ('sigma_chaser', 'relative_position_limit_m', 'relative_velocity_limit_m_per_rad'):
        expected_target_id.pop(key)
    from orbexa.core.dynamics.reference_frame import fingerprint
    target_id = fingerprint(expected_target_id)
    _require(any(binding.sensor_id in row.get('sensor_fingerprints', ()) for row in metadata['navigation_history']),
             'journal sensor source was never accepted by navigation')
    for row in binding.records:
        q0, q1 = exact(row['exact_start']), exact(row['exact_end'])
        decision, commit = row['decision'], row['runtime_commit']
        h = exact(decision['hold_width'])
        _require(q0 == q and q1 == q0+h and h > 0 and q1 <= end and exact(decision['start']) == q0,
                 'full journal is not the complete exact contiguous shared clock')
        identity = _identity(decision['identity'], 'fleet decision')
        _require(identity not in decision_ids and identity == exact_fingerprint(
            {key: value for key, value in decision.items() if key != 'identity'}), 'decision identity changed or repeated')
        decision_ids.add(identity)
        _require(commit.get('status') in ('COMMITTED', 'FINITE_SAFE_WINDOW_ENDED') and
                 exact(commit['exact_end']) == q1 and digest(commit['decision']) == digest(decision),
                 'hold lacks its matching committed fleet admission')
        _require(row['actuator_realization_id'] == binding.actuator_id and row['sensor_profile_id'] == binding.sensor_id and
                 row['navigation_status'] in ('VALID', 'BLACKOUT_CONDITIONAL'), 'hold source or navigation status changed')
        for key in ('active_fallback_id', 'successor_fallback_id'):
            _identity(decision[key], key)
        for key in ('context_ids', 'observation_ids'):
            for value in _pairs(decision[key], ids, key).values():
                _identity(value, key)
        for key in ('commands', 'applied_inputs', 'charged_fuel'):
            _pairs(decision[key], ids, key)
        _pairs(commit['remaining_fuel'], ids, 'remaining fuel')
        _require(all(exact(value) >= 0 for _, value in decision['charged_fuel']) and
                 all(exact(value) >= 0 for value in commit['remaining_fuel'].values()), 'negative committed fuel')
        geometry = decision['geometry']
        _require(geometry and not geometry['reasons'] and geometry['records'] and
                 geometry['clock_id'] == binding.clock.fingerprint and geometry['target_model_id'] == target_id and
                 geometry['geometry_id'] == approach.target['geometry_fingerprint'] and
                 set(geometry['component_ids']) == set(component_ids) and
                 geometry['method'] == 'exact-component-box-fleet-sweep-v1' and
                 geometry['frame'] == 'reference-orbit-frame-v1' and
                 set(geometry['object_ids']) == set(ids) and exact(geometry['start']) <= q0 and exact(geometry['end']) >= q1,
                 'hold lacks complete matching fleet geometry admission')
        _require(all(exact(value['distance_squared_lower']) > exact(value['required_squared']) for value in geometry['records']),
                 'a retained component/peer separation was not admitted')
        coverage = {key: [] for key in expected_separations}
        for value in geometry['records']:
            key = (('peer', *sorted((value['first_id'], value['second_id']))) if value['kind'] == 'peer' else
                   (value['kind'], value['first_id'], value['second_id']))
            _require(key in coverage, 'geometry admission contains an unknown body or component')
            _identity(value['occupied_bounds_id'], 'occupied geometry bounds')
            coverage[key].append((exact(value['start']), exact(value['end'])))
        for intervals in coverage.values():
            covered = q0
            for lower, upper in sorted(intervals):
                if lower <= covered:
                    covered = max(covered, upper)
            _require(covered >= q1, 'a component or peer has missing whole-hold separation coverage')
        radii = _pairs(geometry['body_radii'], ids, 'finite body radii')
        _require(all(exact(radii[key]) >= exact(body_radii[key]) for key in ids), 'admitted finite bus envelope is too small')
        _require(decision['source'] in ('PERFORMANCE', 'FALLBACK_TRACK', 'FALLBACK_HOLD'), 'unrecognized executed command source')
        if decision['source'] == 'PERFORMANCE':
            _identity(row['target_flow_id'], 'target flow')
            _require(set(row['flow_ids']) == set(row['nominal_flow_ids']) == set(ids), 'performance proof fleet is incomplete')
            for value in (*row['flow_ids'].values(), *row['nominal_flow_ids'].values()):
                _identity(value, 'flow source')
        boxes = {key: _box(value, 3) for key, value in decision['applied_inputs']}
        actual = _pairs(row['actual_input_at_start'], ids, 'delivered actuator input')
        actual_end = _pairs(row['actual_truth_relative'], ids, 'retained actual endpoint')
        if sample_index < len(samples) and abs(float(q1)-samples[sample_index]) <= 1e-12:
            _require(all(np.array_equal(np.asarray(actual_end[key], dtype=float),
                np.asarray(approach.actual_trajectories[key][sample_index], dtype=float)) for key in ids),
                'actual journal endpoint differs from the saved approach fleet')
            sample_index += 1
        elif sample_index < len(samples):
            _require(float(q1) < samples[sample_index], 'journal skipped a saved actual fleet event')
        yield AuthenticatedNonlinearHold(q0, q1, boxes, actual)
        q = q1
    _require(q == end and sample_index == len(samples), 'full journal does not cover every saved actual endpoint')
