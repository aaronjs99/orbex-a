"""Authenticate arrival evidence and recover the retained Cartesian boundary."""
from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace

import numpy as np

from orbexa.control.terminal.contact import ContactAdmissionInputs, ContactAdmissionSource, ContactReachPolicy, contact_reach_admission, contact_speed_admission
from orbexa.control.fallback.refuge import exact_record
from orbexa.core.dynamics.relative import NonlinearModelBounds, parameters
from orbexa.core.dynamics.reference_frame import FloatArithmetic, ReferenceClockFrame, exact, fingerprint
from orbexa.simulation.contact.capture import CaptureTruth
from orbexa.simulation.completion.coordinator import _outward_float
from orbexa.simulation.approach.records.checkpoint import digest
from orbexa.simulation.nonlinear.records.archive import semantic_sha256
from orbexa.simulation.completion.nonlinear.contracts import _box, _require, nonlinear_completion_source_identity
from orbexa.simulation.nonlinear.safety.target_bounds import observed_point_goal_bounds, exact_norm_upper, _SOURCE_SHA256 as _GOAL_BOUNDS_SOURCE_SHA256


_CONTACT_ADAPTER_SOURCE_SHA256 = nonlinear_completion_source_identity()['sha256']


def terminal_contact_speed_record(*, navigation, observed_goal_states, goal_bounds, centers,
                                 phi, omega, lambda_gain, docking_points, docking_standoffs, clock,
                                 expected_model_fingerprint, vehicle_design, approach_bounds):
    """Recompute same-event sensing-only bridges to capture speed and reach.

    Numerical reference states are declared separately from physical truth.
    Their point-goal errors are rederived from the saved pose/rate supports;
    neither the producer nor the saved-arrival verifier trusts copied errors.
    """
    from orbexa.control.terminal.contact import ContactReferenceKinematics
    _require(isinstance(docking_points, Mapping) and docking_points and
             all(isinstance(values, Mapping) for values in
                 (centers, phi, omega, observed_goal_states, docking_standoffs, approach_bounds)),
             'contact-speed inputs require complete named fleet mappings')
    ids = tuple(docking_points)
    _require(all(set(values) == set(ids) for values in
             (centers, phi, omega, observed_goal_states, docking_standoffs, approach_bounds)),
             'contact-speed inputs must name the complete assigned fleet')
    _require(isinstance(navigation, dict) and navigation.get('status') == 'VALID' and
             navigation.get('frame') == 'reference-orbit-frame-v1' and
             navigation.get('clock_fingerprint') == clock.fingerprint and
             navigation.get('model_fingerprint') == expected_model_fingerprint,
             'contact-speed navigation source, model, clock or frame changed')
    q = exact(navigation['exact_q'])
    _require(isinstance(goal_bounds, dict) and exact(goal_bounds['exact_q']) == q and
             goal_bounds.get('clock_fingerprint') == clock.fingerprint and
             set(goal_bounds.get('goals', {})) == set(ids),
             'contact-speed goal bounds must share the complete fleet and event')
    rotation = np.asarray([[float(exact(value)) for value in row]
                           for row in goal_bounds['numerical_preview_rotation']], dtype=float)
    rates = np.asarray([float(exact(value)) for value in goal_bounds['numerical_preview_body_rate']], dtype=float)
    goals = {key: np.asarray(values, dtype=float) for key, values in observed_goal_states.items()}
    points = {key: np.asarray(values, dtype=float) for key, values in docking_points.items()}
    _require(rotation.shape == (3, 3) and rates.shape == (3,) and
             all(value.shape == (6,) and np.all(np.isfinite(value)) for value in goals.values()) and
             all(value.shape == (3,) and np.all(np.isfinite(value)) for value in points.values()) and
             np.all(np.isfinite(rotation)) and np.all(np.isfinite(rates)),
             'contact-speed numerical reference has an invalid shape or value')

    class SavedPointPreview:
        current_q = q
        def rotation_matrix_at(self, query):
            _require(float(query) == float(q), 'contact-speed preview queried at another event')
            return rotation
        def body_angular_velocity_at(self, query):
            _require(float(query) == float(q), 'contact-speed rate queried at another event')
            return rates
        def docking_state(self, point, query):
            _require(float(query) == float(q), 'contact-speed point queried at another event')
            matches = [key for key in ids if np.array_equal(points[key], np.asarray(point, dtype=float))]
            _require(matches and all(np.array_equal(goals[key], goals[matches[0]]) for key in matches),
                     'contact-speed point is not the assigned numerical reference')
            return goals[matches[0]]

    estimate = SimpleNamespace(status=navigation['status'], exact_event_q=q,
        clock_fingerprint=navigation['clock_fingerprint'],
        target_orientation_matrix=navigation['target_orientation_matrix'],
        target_orientation_operator_error=navigation['target_orientation_operator_error'],
        target_angular_velocity_box=_box(navigation['target_angular_velocity_box'], 3))
    chasers = [SimpleNamespace(chaser_id=key, docking_point_body=points[key]) for key in ids]
    recomputed = exact_record(observed_point_goal_bounds(estimate, SavedPointPreview(), clock, chasers))
    _require(fingerprint(recomputed) == fingerprint(goal_bounds),
             'contact-speed goal-error support differs from the sensed point-goal derivation')
    _require(isinstance(vehicle_design, dict) and isinstance(vehicle_design.get('chaser'), dict),
             'contact-speed hardware must come from the recorded vehicle declaration')
    gripper = vehicle_design['chaser']['gripper_offset_m']
    navigation_id = fingerprint(navigation)
    goal_id = fingerprint(dict(goal_bounds=recomputed, observed_goal_states=observed_goal_states,
        goal_bound_source_sha256=_GOAL_BOUNDS_SOURCE_SHA256))
    certificates, reach_certificates, derived_bounds = {}, {}, {}
    clock_rate = clock.interval_bounds((q, q))['rate_s_inv']
    for key in ids:
        source = ContactAdmissionSource(agent_id=key, exact_q=q, clock_fingerprint=clock.fingerprint,
            model_fingerprint=expected_model_fingerprint, navigation_fingerprint=navigation_id, goal_bound_id=goal_id)
        support = recomputed['goals'][key]
        inputs = ContactAdmissionInputs(nominal_center=centers[key], observed_goal=goals[key],
            phi=phi[key], omega=omega[key], lambda_gain=lambda_gain,
            goal_position_error_norm_upper=support['position_error_norm_upper'],
            goal_velocity_error_norm_upper=support['velocity_error_norm_upper'],
            clock_rate_s_inv=clock_rate, target_body_rate_rad_s=estimate.target_angular_velocity_box,
            max_spin_error_rad_s=.001, gripper_offset_m=gripper, max_relative_speed_m_s=.01)
        kinematics = ContactReferenceKinematics(rotation_matrix=rotation, body_rate_rad_s=rates,
            rotation_operator_error_upper=recomputed['rotation_operator_error_upper'],
            goal_bound_id=source.goal_bound_id, clock_fingerprint=source.clock_fingerprint, exact_q=source.exact_q)
        certificates[key] = contact_speed_admission(inputs, source, reference_kinematics=kinematics).as_dict()
        reach_policy = ContactReachPolicy(standoff_m=docking_standoffs[key],
            arm_reach_m=vehicle_design['arm_reach_m'], max_orientation_error_rad=.01)
        reach_certificates[key] = contact_reach_admission(inputs, source, policy=reach_policy).as_dict()
        p = exact_norm_upper(abs(c-g)+o for c, g, o in
            zip(inputs.nominal_center[:3], inputs.observed_goal[:3], inputs.omega))
        v = exact_norm_upper(abs(c-g)+f+l*o for c, g, f, l, o in
            zip(inputs.nominal_center[3:], inputs.observed_goal[3:], inputs.phi, inputs.lambda_gain, inputs.omega))
        derived_bounds[key] = dict(position_m=_outward_float(p+exact(support['position_error_norm_upper'])),
            velocity_m_per_rad=_outward_float(v+exact(support['velocity_error_norm_upper'])))
    _require(fingerprint(derived_bounds) == fingerprint(approach_bounds),
             'contact-speed nominal/tube inputs differ from the saved approach bounds')
    result = dict(method='source-bound-terminal-contact-speed-reach-v1', source_sha256=_CONTACT_ADAPTER_SOURCE_SHA256,
        exact_q=str(q), navigation_id=navigation_id, goal_bound_id=goal_id,
        goal_bound_source_sha256=_GOAL_BOUNDS_SOURCE_SHA256,
        vehicle_design_id=fingerprint(vehicle_design), certificates=certificates,
        reach_certificates=reach_certificates,
        admissible=all(value['admissible'] for value in certificates.values()) and
                   all(value['admissible'] for value in reach_certificates.values()))
    result['identity'] = fingerprint(result)
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
        eligible = index >= anchors[0]['update'] and phases[index+1] == 'docking'
        admissible = eligible and navigation[index+1].get('status') == 'VALID'
        if eligible:
            assignments = metadata.get('chaser_assignments', ())
            points = {value['chaser_id']: value['docking_point_body'] for value in assignments}
            standoffs = {value['chaser_id']: value['docking_standoff'] for value in assignments}
            _require(len(points) == len(assignments) and set(points) == ids,
                     'contact-speed assigned points do not name the complete saved fleet')
            diagnostic = approach.tube_diagnostics
            certificate = terminal_contact_speed_record(navigation=navigation[index+1],
                observed_goal_states=row.get('observed_goal_states'), goal_bounds=row.get('moving_goal_bound'),
                centers={key: diagnostic[key]['nominal_center'][index+1] for key in ids},
                phi={key: diagnostic[key]['phi'][index+1] for key in ids},
                omega={key: diagnostic[key]['omega'][index+1] for key in ids},
                lambda_gain=metadata['lambda_gain'], docking_points=points, docking_standoffs=standoffs, clock=clock,
                expected_model_fingerprint=fingerprint(metadata['nonlinear_physical_bounds']),
                vehicle_design=metadata['scenario']['vehicle_design'], approach_bounds=bounds)
            saved = row.get('contact_speed_admission')
            _require(isinstance(saved, dict) and fingerprint(saved) == fingerprint(certificate),
                     'saved contact-speed certificate differs from its exact sensed source derivation')
            admissible &= certificate['admissible']
        else:
            _require(row.get('contact_speed_admission') is None and row.get('observed_goal_states') is None,
                     'contact-speed admission requires an already fixed arrival anchor and docking phase')
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
