"""Physical continuation of an authenticated sensed nonlinear approach.

The adapter uses the executed Cartesian endpoint and the commanded reference
clock. It never reconstructs a Clohessy-Hill ephemeris or fuel history. Sensor
callbacks are declared synthetic realizations; rotational replay stays decoupled.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction
import copy
from itertools import combinations
from itertools import zip_longest
from math import isqrt
import re

import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

from orbexa.control.reachability import IntervalBox
from orbexa.control.refuge import fingerprint as exact_fingerprint
from orbexa.core.attitude import body_rate_derivative
from orbexa.core.nonlinear_relative import ExponentialAtmosphere, NonlinearModelBounds, acceleration, parameters
from orbexa.core.reference_frame import FloatArithmetic, ReferenceClockFrame, exact
from orbexa.simulation.assembly import STANDARD_GRAVITY, positive
from orbexa.simulation.capture import CaptureTruth
from orbexa.simulation.complete_mission import ApproachPropellantBudget, CompleteMissionResult, _outward_float
from orbexa.simulation.contact import ContactBodyState
from orbexa.simulation.mission_checkpoint import digest, plain
from orbexa.simulation.nonlinear_archive import HOLD_INDEX_SCHEMA, NonlinearHoldArchive, hold_index, semantic_sha256


def _require(condition, message):
    if not condition:
        raise ValueError('Nonlinear completion: '+message)


def _identity(value, label):
    _require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None,
             label+' must be a retained SHA-256 identity')
    return value


def _pairs(value, ids, label):
    items = tuple(value.items()) if isinstance(value, dict) else tuple(value)
    _require(len(items) == len(ids) and {key for key, _ in items} == set(ids), label+' must name the exact fleet')
    return dict(items)


def _box(value, dimension):
    _require(isinstance(value, dict), 'an exact saved interval box is required')
    result = IntervalBox(value.get('exact_lower', value.get('lower')),
                         value.get('exact_upper', value.get('upper')))
    _require(result.dimension == dimension, 'saved interval has the wrong dimension')
    if 'exact_lower' in value:
        _require(digest(result.as_dict()) == digest(value), 'saved exact and display interval bounds disagree')
    return result


def _sqrt_upper(value, bits=128):
    scale = 1 << bits
    numerator = isqrt(value.numerator*scale*scale//value.denominator)
    if numerator*numerator*value.denominator < value.numerator*scale*scale:
        numerator += 1
    return Fraction(numerator, scale)


def _dyadic_upper(value, bits=128):
    scale = 1 << bits
    return Fraction(-(-value.numerator*scale//value.denominator), scale)


def _compact_hold(record):
    result = {key: copy.deepcopy(value) for key, value in record.items()
              if key not in ('decision', 'runtime_commit')}
    decision, commit = record['decision'], record['runtime_commit']
    result['decision'] = {key: copy.deepcopy(value) for key, value in decision.items() if key != 'geometry'}
    result['geometry_id'] = None if decision.get('geometry') is None else exact_fingerprint(decision['geometry'])
    result['runtime_commit_status'], result['remaining_fuel'] = commit['status'], copy.deepcopy(commit['remaining_fuel'])
    return result


def verify_nonlinear_arrival(approach, *, clock):
    """Check the saved fixed deadline and every observed terminal endpoint.

    This rechecks the retained whole-set acceptance numbers and their exact
    clock/dwell lineage. It does not replace the producer's flow certificates.
    A historical record without this evidence cannot enter physical completion.
    """
    metadata = approach.metadata
    arrival = metadata.get('arrival_state')
    _require(isinstance(clock, ReferenceClockFrame) and
             metadata.get('reference_clock_fingerprint') == clock.fingerprint,
             'arrival requires the original typed reference clock')
    _require(isinstance(arrival, dict) and arrival.get('method') == 'fixed-deadline-whole-set-entry-v1' and
             arrival.get('entry_by_deadline') is True and arrival.get('deadline_missed') is False and
             arrival.get('deadline_never_reset') is True, 'missing or unsuccessful fixed arrival evidence')

    def q_value(value, label):
        _require(isinstance(value, str) and bool(value.strip()), label+' must be an exact saved clock string')
        try:
            return exact(value)
        except (ValueError, TypeError, ZeroDivisionError) as error:
            raise ValueError('Nonlinear completion: malformed '+label) from error

    def integer(value):
        return isinstance(value, int) and not isinstance(value, bool)

    deadline = q_value(arrival.get('exact_deadline'), 'arrival deadline')
    entry = q_value(arrival.get('first_admitted_exact_q'), 'first terminal entry')
    dwell_required, final_dwell = arrival.get('terminal_dwell_updates'), arrival.get('final_dwell')
    _require(integer(dwell_required) and dwell_required > 0 and integer(final_dwell) and
             final_dwell >= dwell_required and integer(metadata.get('terminal_dwell_updates')) and
             integer(metadata.get('terminal_dwell_observed')) and metadata.get('terminal_dwell_updates') == dwell_required and
             metadata.get('terminal_dwell_observed') == final_dwell,
             'arrival dwell declaration disagrees with the saved approach')
    rows, navigation = metadata.get('terminal_bound_records'), metadata.get('navigation_history')
    samples, phases = approach.anom_history, approach.sample_phase_history
    _require(isinstance(rows, list) and isinstance(navigation, list) and len(samples) >= 2 and
             len(rows) == len(samples)-1 and len(navigation) == len(samples) and len(phases) == len(samples),
             'arrival history must cover every committed outer endpoint')
    events = arrival.get('events')
    _require(isinstance(events, list) and all(isinstance(event, dict) for event in events),
             'arrival events must be a saved ordered list')
    anchors = [event for event in events if event.get('status') == 'ANCHORED']
    entries = [event for event in events if event.get('status') == 'WHOLE_SET_ENTRY']
    _require(len(anchors) == len(entries) == 1 and len(events) == 2 and events[0] == anchors[0] and
             events[1] == entries[0] and integer(anchors[0].get('update')) and
             0 <= anchors[0]['update'] < len(rows) and entries[0].get('by_deadline') is True and
             q_value(anchors[0].get('exact_deadline'), 'anchored deadline') == deadline and
             q_value(entries[0].get('exact_q'), 'entry event') == entry and entry <= deadline,
             'arrival was not one unchanged deadline followed by a timely whole-set entry')
    _require(not any(isinstance(event, dict) and event.get('stage') == 'ARRIVAL_DEADLINE_MISSED'
                     for event in metadata.get('failure_events', ())), 'arrival failure evidence cannot be overridden')
    exact_samples = []
    for index, nav in enumerate(navigation):
        _require(isinstance(nav, dict) and nav.get('clock_fingerprint') == clock.fingerprint,
                 'arrival navigation clock lineage changed')
        value = q_value(nav.get('exact_q'), 'navigation event')
        _require(np.isfinite(float(samples[index])) and abs(float(value)-float(samples[index])) <= 1e-12 and
                 (not exact_samples or value > exact_samples[-1]), 'arrival event clock is stale or unordered')
        exact_samples.append(value)
    count = metadata.get('horizon_steps', {}).get('docking', metadata.get('horizon_steps', {}).get('all'))
    step = metadata.get('anom_step')
    _require(isinstance(step, (int, float)) and not isinstance(step, bool) and np.isfinite(step) and step > 0 and
             integer(count) and count >= 2 and
             deadline == exact_samples[anchors[0]['update']]+(count-1)*exact(metadata.get('anom_step')),
             'anchored arrival deadline differs from the declared fixed horizon')
    _require(entry > exact_samples[anchors[0]['update']], 'terminal entry precedes the anchored update')
    _require(q_value(metadata.get('continuous_execution_certificate', {}).get('exact_end_anomaly'),
                     'committed endpoint') == exact_samples[-1] and
             q_value(metadata.get('actual_physical_end_exact_q'), 'actual physical endpoint') == exact_samples[-1],
             'arrival final endpoint differs from the actual committed physical clock')
    ids, dwell, first = set(approach.actual_trajectories), 0, None
    position_limit, velocity_limit = metadata.get('docking_position_tolerance'), metadata.get('docking_velocity_tolerance')
    _require(all(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value) and value > 0
                 for value in (position_limit, velocity_limit)), 'terminal tolerances are malformed')
    for index, row in enumerate(rows):
        _require(isinstance(row, dict) and q_value(row.get('exact_q'), 'terminal event') == exact_samples[index+1] and
                 np.isfinite(float(row.get('anomaly', np.nan))) and
                 abs(float(row['anomaly'])-float(exact_samples[index+1])) <= 1e-12,
                 'terminal history is stale or misaligned')
        saved_deadline = row.get('arrival_deadline')
        _require((saved_deadline is None if index < anchors[0]['update'] else
                  q_value(saved_deadline, 'terminal deadline') == deadline), 'terminal deadline changed or is missing')
        bounds = row.get('bounds')
        _require(isinstance(bounds, dict) and set(bounds) == ids, 'terminal bounds must cover the exact fleet')
        admissible = phases[index+1] == 'docking' and navigation[index+1].get('status') == 'VALID'
        rate = clock.interval_bounds((exact_samples[index+1], exact_samples[index+1]))['rate_s_inv'].upper[0]
        for bound in bounds.values():
            _require(isinstance(bound, dict), 'terminal fleet bound must be a saved mapping')
            p, v = bound.get('position_m'), bound.get('velocity_m_per_rad')
            _require(all(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value) and value >= 0
                         for value in (p, v)), 'terminal whole-set bounds are malformed')
            admissible &= exact(p) <= exact(position_limit) and exact(v) <= exact(velocity_limit) and rate*(exact(p)+exact(v)) <= exact('.01')
        _require(isinstance(row.get('admitted'), bool) and row['admitted'] == bool(admissible),
                 'terminal admitted flag disagrees with its whole-set bounds or navigation status')
        dwell = dwell+1 if admissible else 0
        _require(integer(row.get('dwell')) and row['dwell'] == dwell, 'terminal dwell is not a consecutive endpoint count')
        if admissible and first is None:
            first = exact_samples[index+1]
        _require(exact_samples[index+1] < deadline or admissible,
                 'an endpoint at or after the fixed arrival deadline was not admissible')
    _require(first == entry and dwell == final_dwell and dwell >= dwell_required,
             'final whole-set dwell or first-entry lineage changed')
    return dict(method='verified-fixed-deadline-whole-set-entry-v1', exact_deadline=str(deadline),
        first_admitted_exact_q=str(entry), final_exact_q=str(exact_samples[-1]),
        terminal_dwell_updates=dwell_required, final_dwell=dwell, checked_outer_endpoints=len(rows),
        arrival_state_sha256=semantic_sha256(arrival), terminal_history_sha256=semantic_sha256(rows),
        scope='Rechecked retained whole-set endpoint bounds, exact clock and consecutive dwell; not a new flow proof')


def capture_truth_from_nonlinear(approach, *, model, clock):
    """Use retained actual r/v/Q/omega, with the original exact reference clock."""
    _require(isinstance(model, NonlinearModelBounds) and isinstance(clock, ReferenceClockFrame),
             'typed physical model and commanded reference clock are required')
    metadata = approach.metadata
    _require(digest(metadata.get('nonlinear_truth')) == digest(model.as_dict()), 'point physical model differs from the saved truth')
    _require(metadata.get('reference_clock_fingerprint') == clock.fingerprint and
             digest(metadata.get('reference_clock')) == digest(clock.as_dict()), 'reference clock identity changed')
    # Reject interval-valued truth instead of silently selecting a midpoint.
    par = parameters(model, FloatArithmetic())
    q = exact(metadata['continuous_execution_certificate']['exact_end_anomaly'])
    _require(abs(float(q)-float(approach.anom_history[-1])) < 1e-10, 'physical endpoint and exact committed clock disagree')
    kinematics = metadata['target_kinematics']
    count = len(approach.anom_history)
    saved_q = np.asarray(kinematics['anomaly'], dtype=float)
    times = np.asarray(kinematics['physical_time'], dtype=float)
    states = np.asarray(kinematics['target_inertial_states'], dtype=float)
    quaternions = np.asarray(kinematics['inertial_quaternions'], dtype=float)
    rates = np.asarray(kinematics['body_angular_velocity'], dtype=float)
    _require(saved_q.shape == times.shape == (count,) and states.shape == (count, 6) and
             quaternions.shape == (count, 4) and rates.shape == (count, 3) and
             np.allclose(saved_q, approach.anom_history, atol=1e-12, rtol=0) and
             all(np.all(np.isfinite(value)) for value in (times, states, quaternions, rates)) and
             np.all(np.diff(times) > 0), 'actual Cartesian target boundary is missing or misaligned')
    point = clock.point(q)
    return CaptureTruth(float(q), times[-1], point.rate_s_inv, states[-1, :3], states[-1, 3:],
        quaternions[-1], rates[-1], np.asarray(point.rotation), par['mu'])


class NonlinearEnvironment:
    """The original point J2/vector-drag law, as extra acceleration beyond central.

    Default sigma=CdA/m remains the constant declared by the approach model.
    Optional explicit CdA values instead use each current dry+fuel mass. The
    radius domain extends to the declared delivery interface's numerical event
    bracket; the force law and inclined physical ECI axes are unchanged.
    """
    def __init__(self, model, *, target_mass_kg, chasers, drag_area_m2_by_name=None,
                 minimum_altitude_m=80000.):
        _require(isinstance(model, NonlinearModelBounds), 'environment requires a typed point model')
        self.original_model = model
        minimum = float(model.body_radius_m)+positive(minimum_altitude_m, 'Continuation minimum altitude')
        self.model = replace(model, radius_domain_m=(min(minimum, float(model.radius_domain_m[0])), model.radius_domain_m[1]))
        self.arithmetic, self.par = FloatArithmetic(), parameters(model, FloatArithmetic())
        self.dry_mass = dict(target=positive(target_mass_kg, 'Target mass'),
                             **{key: value.dry_mass_kg for key, value in chasers.items()})
        _require('target' not in chasers and chasers, 'environment needs the actual named fleet')
        self.drag_area = None if drag_area_m2_by_name is None else dict(drag_area_m2_by_name)
        if self.drag_area is not None:
            _require(set(self.drag_area) == set(self.dry_mass), 'explicit CdA values must name every body')
            self.drag_area = {key: positive(value, key+' CdA', allow_zero=True) for key, value in self.drag_area.items()}
        self.calls = 0

    def __call__(self, time_s, bodies):
        _require(np.isfinite(time_s) and set(bodies) == set(self.dry_mass), 'environment body IDs or physical clock changed')
        result = {}
        for key, state in bodies.items():
            _require(isinstance(state, ContactBodyState) and abs(state.time_s-time_s) < 1e-8,
                     'environment needs current physical body states')
            vector = tuple(np.r_[state.position_eci_m, state.velocity_eci_m_s])
            sigma = self.par['sigma_target' if key == 'target' else 'sigma_chaser']
            if self.drag_area is not None:
                sigma = self.drag_area[key]/(self.dry_mass[key]+state.fuel_kg)
            full = np.asarray(acceleration(vector, self.model, self.par, self.arithmetic, sigma))
            radius = np.linalg.norm(state.position_eci_m)
            central = -self.par['mu']*state.position_eci_m/radius**3
            result[key] = dict(acceleration_eci_m_s2=full-central, torque_body_nm=np.zeros(3))
        self.calls += 1
        return result

    def as_dict(self):
        return dict(method='same-declared-point-j2-vector-drag-minus-central-v1',
            source_model=self.original_model.as_dict(), source_model_fingerprint=self.original_model.fingerprint,
            continuation_radius_domain_m=tuple(map(str, self.model.radius_domain_m)),
            dry_mass_kg=self.dry_mass, drag_area_m2_by_name=self.drag_area,
            drag_semantics=('declared constant CdA/m from approach' if self.drag_area is None else
                            'explicit CdA divided by actual current dry plus fuel mass'),
            frame='unchanged physical ECI basis; original inclined J2 and atmosphere rotation axes',
            intrinsic_extra_torque='zero; configured central gravity-gradient torque remains separate',
            scope='Same numerical force law through delivery; radius-domain extension is not atmospheric demise or flight qualification')


@dataclass(frozen=True)
class NonlinearPropellantBudget(ApproachPropellantBudget):
    held_accounting: dict = field(default_factory=dict)

    def to_dict(self):
        record = super().to_dict()
        record.update(method='source-bound-nonlinear-admitted-hold-propellant-v1',
            scope='Conservative upper debit from every authenticated nonlinear applied-input support and exact reference-clock hold; not measured fuel',
            held_input_accounting=plain(self.held_accounting))
        return record


class NonlinearTranslationAccountant:
    """Bind a complete full nonlinear journal to its compact saved approach.

    This verifies committed admission bookkeeping and source identities. It
    does not re-run the independent interval-flow/geometry proof service.
    """
    def __init__(self, approach, *, model, clock, hold_records,
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
        self.model, self.clock, self.records = model, clock, records
        self.source_hash = semantic_sha256(vars(approach))
        self.journal_hash = records.identity if isinstance(records, NonlinearHoldArchive) else digest(records)
        self.actuator_id = _identity(expected_actuator_id or saved[0]['actuator_realization_id'], 'actuator realization')
        self.sensor_id = _identity(expected_sensor_id or saved[0]['sensor_profile_id'], 'sensor profile')

    def translation_budget(self, approach, *, options, attitude_tracker, terminal_handoff=None):
        _require(terminal_handoff is None, 'a CH terminal continuation cannot be mixed with this nonlinear journal')
        current_journal = self.records.identity if isinstance(self.records, NonlinearHoldArchive) else digest(self.records)
        _require(semantic_sha256(vars(approach)) == self.source_hash and current_journal == self.journal_hash,
                 'bound approach or full journal changed')
        metadata, ids = approach.metadata, tuple(approach.actual_trajectories)
        certificate = metadata['continuous_execution_certificate']
        _require(certificate.get('method') == 'nonlinear-sensed-fleet-admission-v1' and
                 certificate.get('all_intervals_certified') is True and
                 certificate.get('admitted_fleet_holds') == len(self.records), 'not every retained hold was admitted')
        start = exact(metadata['fleet_hold_records'][0]['exact_start'])
        end, q = exact(certificate['exact_end_anomaly']), start
        _require(abs(float(start)-float(approach.anom_history[0])) < 1e-10, 'journal release clock changed')
        impulse, peak, decision_ids = dict.fromkeys(ids, Fraction(0)), dict.fromkeys(ids, Fraction(0)), set()
        samples = np.asarray(approach.anom_history, dtype=float)
        _require(samples.ndim == 1 and len(samples) >= 2 and np.all(np.isfinite(samples)) and
                 np.all(np.diff(samples) > 0) and all(np.asarray(approach.actual_trajectories[key]).shape ==
                 (len(samples), 6) for key in ids), 'saved actual fleet timeline is invalid')
        sample_index = 1
        model_bounds = metadata['nonlinear_physical_bounds']
        component_ids = tuple(value['id'] for value in approach.target['geometry']['components'])
        expected_separations = {('target_component', key, component) for key in ids for component in component_ids}
        expected_separations.update(('peer', *sorted(pair)) for pair in combinations(ids, 2))
        expected_target_id = dict(model_bounds)
        for key in ('sigma_chaser', 'relative_position_limit_m', 'relative_velocity_limit_m_per_rad'):
            expected_target_id.pop(key)
        from orbexa.core.reference_frame import fingerprint
        target_id = fingerprint(expected_target_id)
        _require(any(self.sensor_id in row.get('sensor_fingerprints', ()) for row in metadata['navigation_history']),
                 'journal sensor source was never accepted by navigation')
        for row in self.records:
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
            _require(row['actuator_realization_id'] == self.actuator_id and row['sensor_profile_id'] == self.sensor_id and
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
                     geometry['clock_id'] == self.clock.fingerprint and geometry['target_model_id'] == target_id and
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
            _require(all(exact(radii[key]) >= exact(attitude_tracker.chasers.get(key,
                attitude_tracker.default_chaser).body_radius_m) for key in ids), 'admitted finite bus envelope is too small')
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
            duration = self.clock.elapsed_bounds(q0, q1)
            rate = self.clock.interval_bounds((q0, q1))['rate_s_inv'].upper[0]
            for key in ids:
                box = boxes[key]
                _require(box.contains(IntervalBox.point(actual[key])), key+': delivered actuator value escapes its support')
                u = tuple(max(abs(lo), abs(hi)) for lo, hi in zip(box.lower, box.upper))
                _require(max(u) <= exact(options.input_component_bound), key+': saved actuator support exceeds the supplied component bound')
                prop = attitude_tracker.chasers.get(key, attitude_tracker.default_chaser)
                force = exact(prop.mass_kg)*rate**2*_sqrt_upper(3*sum(value*value for value in u))
                _require(force <= exact(options.translation_force_n), key+': per-hold body-force allowance exceeds installed authority')
                impulse[key] += _dyadic_upper(force*duration.upper[0])
                peak[key] = max(peak[key], force)
            q = q1
        _require(q == end and sample_index == len(samples), 'full journal does not cover every saved actual endpoint')
        denominator = exact(STANDARD_GRAVITY)*exact(options.specific_impulse_s)
        fuel = {key: value/denominator for key, value in impulse.items()}
        binding = dict(source_approach_sha256=self.source_hash, full_journal_sha256=self.journal_hash,
            full_journal_archive=self.records.as_dict() if isinstance(self.records, NonlinearHoldArchive) else None,
            exact_start_anomaly=str(start), exact_end_anomaly=str(end), holds=len(self.records),
            actuator_realization_id=self.actuator_id, sensor_profile_id=self.sensor_id,
            clock_fingerprint=self.clock.fingerprint, physical_truth_fingerprint=self.model.fingerprint,
            decision_identities_verified=True, complete_committed_clock_verified=True,
            exact_impulse_upper_n_s={key: str(value) for key, value in impulse.items()},
            exact_fuel_upper_kg={key: str(value) for key, value in fuel.items()},
            formula='F1 <= m_initial_wet*nu_hold_upper^2*sqrt(3*sum(max(abs(u_lo),abs(u_hi))^2)); sum(F1*dt_hold_upper)/(g0*Isp)',
            proof_scope='Authenticated retained admission bookkeeping; independent interval witnesses are retained by the producer, not re-executed here')
        elapsed = self.clock.elapsed_bounds(start, end)
        return NonlinearPropellantBudget(dict(method='original-commanded-reference-clock-v1',
            physical_duration_s=elapsed.as_dict(), exact_start_anomaly=str(start), exact_end_anomaly=str(end),
            clock_fingerprint=self.clock.fingerprint), {key: _outward_float(value) for key, value in peak.items()},
            {key: _outward_float(value) for key, value in fuel.items()}, float(options.input_component_bound),
            float(options.specific_impulse_s), float(options.translation_force_n), binding)


class PastNavigationAttitudeObserver:
    """Causal past accepted pose/rate prediction and bounded own-bus sensing."""
    def __init__(self, approach, *, model, clock):
        self.clock = clock
        # Prediction uses the public prior, never the independently chosen
        # point realization that is supplied to the physical continuation.
        public = approach.metadata['nonlinear_physical_bounds']
        atmosphere = ExponentialAtmosphere(**{key: public['atmosphere'][key] for key in
            ('rho0_kg_m3', 'reference_radius_m', 'scale_height_m', 'rotation_rad_s', 'wind_m_s', 'declaration')})
        prior = NonlinearModelBounds(atmosphere=atmosphere, **{key: public[key] for key in
            ('mu_m3_s2', 'j2', 'body_radius_m', 'gravity_reference_radius_m', 'j2_axis', 'sigma_target',
             'sigma_chaser', 'radius_domain_m', 'relative_position_limit_m', 'relative_velocity_limit_m_per_rad', 'declaration')})
        _require(digest(prior.as_dict()) == digest(public), 'public navigation model source changed')
        self.public_model_id, self.model = prior.fingerprint, prior.midpoint_model()
        self.inertia = np.asarray(approach.target['inertia_kg_m2'], dtype=float)
        self.gravity_gradient = bool(approach.target.get('gravity_gradient', False))
        self.par = parameters(self.model, FloatArithmetic())
        self.records = []
        for record in approach.metadata['navigation_history']:
            if record.get('status') != 'VALID':
                continue
            _require(record['clock_fingerprint'] == clock.fingerprint and
                     record['model_fingerprint'] == self.public_model_id and
                     record['frame'] == 'reference-orbit-frame-v1', 'navigation model, clock or frame source changed')
            box, ephemeris = _box(record['physical_time_box'], 1), _box(record['target_ephemeris_box'], 6)
            rate = _box(record['target_angular_velocity_box'], 3)
            q = exact(record['exact_q'])
            rotation = np.asarray(clock.point(q).rotation)@np.asarray(record['target_orientation_matrix'], dtype=float)
            state = np.r_[[float((lo+hi)/2) for lo, hi in zip(ephemeris.lower, ephemeris.upper)],
                Rotation.from_matrix(rotation).as_quat(),
                [float((lo+hi)/2) for lo, hi in zip(rate.lower, rate.upper)]]
            self.records.append((float(box.upper[0]), float((box.lower[0]+box.upper[0])/2), state,
                                 int(record.get('accepted_packet_count', 0)), str(q)))
        _require(self.records and all(a[0] <= b[0] for a, b in zip(self.records, self.records[1:])),
                 'ordered accepted past navigation history is required')
        self.calls, self.max_age_s, self.used_indices = 0, 0., set()
        self.prior_calls = 0
        self._cached = None

    def _rhs(self, time_s, state):
        rotation = Rotation.from_quat(state[6:10]).as_matrix()
        torque = np.zeros(3)
        if self.gravity_gradient:
            radius = np.linalg.norm(state[:3])
            radial = rotation.T@state[:3]/radius
            torque = 3*self.par['mu']/radius**3*np.cross(radial, self.inertia@radial)
        q, w = state[6:10], state[10:13]
        return np.r_[state[3:6], acceleration(tuple(state[:6]), self.model, self.par, FloatArithmetic(), self.par['sigma_target']),
            .5*np.r_[q[3]*w+np.cross(q[:3], w), -q[:3]@w], body_rate_derivative(w, self.inertia, torque)]

    def __call__(self, time_s, snapshot):
        eligible = [index for index, record in enumerate(self.records) if record[0] <= time_s+1e-10]
        _require(eligible, 'no past accepted target observation is available at this replay time')
        index = eligible[-1]
        available, epoch, initial, packets, q = self.records[index]
        if self._cached is not None and self._cached[0] == index and self._cached[1] <= time_s:
            _, epoch, initial = self._cached
        state = initial.copy()
        if time_s > epoch+1e-12:
            solved = solve_ivp(self._rhs, (epoch, time_s), state, method='DOP853', rtol=1e-10, atol=1e-12,
                               max_step=min(.5, time_s-epoch))
            _require(solved.success and np.all(np.isfinite(solved.y)), 'past observed target prediction failed')
            state = solved.y[:, -1]
        self._cached = (index, time_s, state.copy())
        result = dict(time_s=float(time_s), target_rotation_body_to_eci=Rotation.from_quat(state[6:10]).as_matrix(),
            target_angular_velocity_body_rad_s=state[10:13],
            target_angular_acceleration_body_rad_s2=self._rhs(time_s, state)[10:13], chasers={})
        for number, (key, sensed) in enumerate(snapshot['chasers'].items()):
            phase = .7*number+.03*time_s
            direction = np.array((np.cos(phase), np.sin(phase), 0.))
            rotation = Rotation.from_quat(sensed['quaternion_body_to_eci'])*Rotation.from_rotvec(.0003*direction)
            result['chasers'][key] = dict(quaternion_body_to_eci=rotation.as_quat(),
                angular_velocity_body_rad_s=np.asarray(sensed['angular_velocity_body_rad_s'])+1e-5*direction)
        self.calls += 1
        self.prior_calls += int(packets == 0)
        self.max_age_s = max(self.max_age_s, time_s-self.records[index][1])
        self.used_indices.add(index)
        return result

    def as_dict(self):
        return dict(method='past-accepted-navigation-rigid-body-extrapolation-v1',
            rule='latest VALID navigation record whose physical-time upper endpoint is no later than the replay query',
            future_truth_or_interpolation_used=False, own_bus_orientation_error_norm_rad=.0003,
            own_bus_rate_error_norm_rad_s=1e-5, calls=self.calls, maximum_prediction_age_s=self.max_age_s,
            initial_declared_prior_calls=self.prior_calls,
            prediction_model='midpoint of public declared nonlinear bounds, independent of selected truth',
            public_model_fingerprint=self.public_model_id, prediction_model_fingerprint=self.model.fingerprint,
            used_navigation_records=len(self.used_indices), source_navigation_sha256=digest(self.records),
            scope='Observed-center rigid-body/GG prediction; decoupled replay, no coupled translational-attitude flight claim')


class BiasedContactObserver:
    """Truth-side synthetic sensor realization; commands see only its outputs."""
    def __init__(self):
        self.calls = 0

    def __call__(self, time_s, snapshot):
        common_p, common_v = np.array((.35, -.4, .2)), np.array((.001, -.001, 0.))
        target = snapshot['target']
        target_rotation = Rotation.from_quat(target.quaternion_body_to_eci)*Rotation.from_rotvec((.001, 0., 0.))
        measured = dict(target=ContactBodyState(time_s, target.position_eci_m+common_p,
            target.velocity_eci_m_s+common_v, target_rotation.as_quat(),
            target.angular_velocity_body_rad_s+np.array((0., 1e-4, 0.)), target.fuel_kg), chasers={},
            contact_gaps_m={key: value+.001 for key, value in snapshot['contact_gaps_m'].items()},
            contact_wrenches_eci={})
        for number, (key, state) in enumerate(snapshot['chasers'].items()):
            sign = (-1.)**number
            direction = np.array((np.cos(.7*number), np.sin(.7*number), 0.))
            camera_p = np.array((sign*.0015, -.002, .0015))
            rotation = Rotation.from_quat(state.quaternion_body_to_eci)*Rotation.from_rotvec(.0003*direction)
            measured['chasers'][key] = ContactBodyState(time_s, state.position_eci_m+common_p+camera_p,
                state.velocity_eci_m_s+common_v, rotation.as_quat(), state.angular_velocity_body_rad_s+1e-5*direction,
                state.fuel_kg)
            wrench = snapshot['contact_wrenches_eci'][key]
            measured['contact_wrenches_eci'][key] = dict(force_n=np.asarray(wrench['force_n'])+np.array((.003, -.002, .003)),
                couple_nm=np.asarray(wrench['couple_nm'])+np.array((.0001, -.0001, .0001)))
        self.calls += 1
        return measured

    def as_dict(self):
        return dict(method='declared-biased-synthetic-contact-observations-v1', calls=self.calls,
            common_gnss_position_bias_eci_m=(.35, -.4, .2), common_gnss_velocity_bias_eci_m_s=(.001, -.001, 0.),
            individual_camera_position_bias_norm_upper_m=.003, target_pose_error_rad=.001,
            target_rate_error_norm_rad_s=1e-4, own_bus_pose_error_norm_rad=.0003,
            own_bus_rate_error_norm_rad_s=1e-5, local_range_bias_m=.001,
            wrench_force_bias_eci_n=(.003, -.002, .003), wrench_couple_bias_eci_nm=(.0001, -.0001, .0001),
            scope='Synthetic bounded processed products; no raw perception, calibrated material sensing or hardware claim')


def complete_nonlinear_mission(approach, nonlinear_runner=None, *, capture_truth=None, model=None, clock=None,
                               design=None, hold_records=None, completion_runner=None, stage_sink=None,
                               drag_area_m2_by_name=None):
    """Run actual saved approach -> observed attitude -> finite contact -> disposal.

    Supply the live nonlinear runner, or explicit typed model/clock/design and
    its full admitted hold journal. Incomplete approaches remain incomplete.
    """
    if not approach.success or not approach.metadata.get('mission_complete', False):
        return CompleteMissionResult(False, 'nonlinear_approach_incomplete: '+approach.message, approach)
    _require(approach.metadata.get('model_version') == 'nonlinear-reference-relative-v1', 'a saved sensed nonlinear approach is required')
    if nonlinear_runner is not None:
        _require(all(value is None for value in (model, clock, design)), 'use a live runner or explicit boundary declarations')
        model, clock, design = nonlinear_runner.truth_model, nonlinear_runner.clock, nonlinear_runner.design
        if hold_records is None:
            hold_records = nonlinear_runner.executor.journal
    _require(model is not None and clock is not None and design is not None and hold_records is not None,
             'supply model, clock, design and the complete full nonlinear hold journal')
    arrival_verification = verify_nonlinear_arrival(approach, clock=clock)
    _require(digest(approach.metadata['scenario']['vehicle_design']) == digest(design.as_dict()), 'declared vehicle differs from the saved approach')
    saved_truth = capture_truth_from_nonlinear(approach, model=model, clock=clock)
    if capture_truth is None:
        capture_truth = saved_truth
    else:
        _require(isinstance(capture_truth, CaptureTruth) and digest(plain(capture_truth)) == digest(plain(saved_truth)),
                 'explicit physical boundary differs from retained actual nonlinear truth')
    accountant = NonlinearTranslationAccountant(approach, model=model, clock=clock, hold_records=hold_records,
        expected_actuator_id=None if nonlinear_runner is None else nonlinear_runner.executor.actuator.fingerprint,
        expected_sensor_id=None if nonlinear_runner is None else nonlinear_runner.options.sensor.fingerprint)
    certificate = approach.metadata['continuous_execution_certificate']
    bound = max(abs(float(value)) for value in (*certificate['input_lower'], *certificate['input_upper']))
    complete = completion_runner or design.completion_runner(input_component_bound=float(np.nextafter(bound, np.inf)),
        stage_sink=stage_sink, translation_accountant=accountant)
    _require(complete.options.target_mass_kg == design.target_mass_kg and
             complete.options.translation_force_n == design.translation_force_n and
             complete.options.specific_impulse_s == design.specific_impulse_s and
             complete.attitude_tracker.actuators.nozzle_force_limit_n == design.attitude_nozzle_force_n and
             complete.attitude_tracker.actuators.moment_arm_m == design.attitude_moment_arm_m and
             complete.attitude_tracker.actuators.specific_impulse_s == design.specific_impulse_s,
             'completion hardware differs from the recorded prospective vehicle')
    _require(all(complete.attitude_tracker.chasers.get(key, complete.attitude_tracker.default_chaser) == design.chaser
                 for key in approach.actual_trajectories), 'completion bus mass/envelope differs from the saved vehicle')
    complete.translation_accountant = accountant
    if stage_sink is not None:
        complete.stage_sink = stage_sink
    complete.capture_options = replace(complete.capture_options, arm_reach_m=design.arm_reach_m,
        max_relative_speed_m_s=.01, max_orientation_error_rad=.01, max_spin_error_rad_s=.001,
        idealized_instantaneous=False)
    complete.attitude_tracker.options = replace(complete.attitude_tracker.options,
        max_final_orientation_error_rad=.01, max_final_spin_error_rad_s=.001)
    _require(float(approach.metadata['docking_position_tolerance']) <= .04+1e-12,
             'saved whole-position endpoint enclosure exceeds the declared 40 mm contact profile')
    chasers = {key: complete.attitude_tracker.chasers.get(key, complete.attitude_tracker.default_chaser)
               for key in approach.actual_trajectories}
    environment = NonlinearEnvironment(model, target_mass_kg=design.target_mass_kg, chasers=chasers,
        drag_area_m2_by_name=drag_area_m2_by_name)
    attitude_observer = PastNavigationAttitudeObserver(approach, model=model, clock=clock)
    contact_observer = BiasedContactObserver()
    result = complete.complete(approach, capture_truth=capture_truth, contact_feedback=contact_observer,
        contact_environment=environment, attitude_observations=attitude_observer)
    audit = dict(method='nonlinear-actual-boundary-physical-completion-v1',
        source_approach_sha256=semantic_sha256(vars(approach)), environment=environment.as_dict(),
        arrival_verification=arrival_verification,
        environment_rhs_calls=environment.calls, attitude_observations=attitude_observer.as_dict(),
        contact_observations=contact_observer.as_dict(),
        capture_truth=plain(capture_truth), admission=dict(position_enclosure_m=.04, relative_speed_m_s=.01,
            orientation_rad=.01, spin_error_rad_s=.001, arm_reach_m=design.arm_reach_m),
        scope='Simulated nonlinear continuation with explicit sensors and retained physical gates; separate attitude replay and representative rigid compliant contact, no hardware or flight qualification')
    return replace(result, completion_audit=audit)
