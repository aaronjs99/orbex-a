"""Study metrics from recorded physical prefixes, sampled histories and refusals."""
from __future__ import annotations

import json
from fractions import Fraction


def completion_progress_endpoint(lifecycle, approach_end_time_s):
    """Retain admitted physical progress when a later hybrid stage refuses."""
    end_time, end_stage = approach_end_time_s, 'approach'
    contact = lifecycle.capture.audit.get('finite_contact')
    if contact is not None:
        end_time, end_stage = float(contact['final_target']['time_s']), 'finite_contact'
        lock = lifecycle.capture.audit.get('finite_lock', {})
        records = lock.get('step_records', ())
        if records:
            previous = Fraction(0)
            for record in records:
                start, end = map(Fraction, (record['exact_start_time_s'], record['exact_end_time_s']))
                if start != previous or end <= start:
                    raise ValueError('Finite-lock progress must be an exact contiguous admitted prefix')
                previous = end
            if lock.get('accepted_steps') != len(records):
                raise ValueError('Finite-lock progress count differs from its admitted prefix')
            end_time, end_stage = end_time+float(previous), 'finite_lock'
    if lifecycle.capture.state is not None:
        time_s = float(lifecycle.capture.state.time_s)
        if end_time is not None and time_s < end_time-1e-9:
            raise ValueError('Attachment state precedes its retained contact/lock prefix')
        end_time, end_stage = time_s, 'attachment'
    for name in ('detumbling', 'deorbit'):
        stage = getattr(lifecycle, name)
        if stage is not None:
            time_s = float(stage.final_state.time_s)
            if end_time is not None and time_s < end_time-1e-9:
                raise ValueError('Completion stage clock precedes its admitted physical prefix')
            end_time, end_stage = time_s, name
    return end_time, end_stage


def native_attempt_metrics(directory):
    """Count every native attempt, including refused and unknown outcomes."""
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
    return native


def collect_study_metrics(approach, completion, runner, directory, error=None):
    """Report recorded physics and refusals without success-only filtering."""
    import numpy as np
    from orbexa.simulation.nonlinear_archive import HOLD_INDEX_SCHEMA

    result = dict(schema='orbexa.nonlinear-study-metrics.v1', success=False,
        native_attempts=native_attempt_metrics(directory),
        error=error, scope='Recorded simulated physical prefix and actual failures; no successful-case filtering or flight deadline claim')
    if approach is None:
        result.update(status='SETUP_FAILED', physical_elapsed_s=None, safety=None, tubes={}, bandwidth={},
            set_width={}, effort={}, fuel_bound=None, fallback_counts={}, deadline_misses=None, failures=[error])
        return result
    metadata = approach.metadata
    accounting = metadata.get('partial_execution_accounting')
    held_effort = None if accounting is None else accounting['applied_input_squared_integral_per_reference_anomaly']
    elapsed = metadata.get('elapsed_time_seconds', [])
    actual_time = None
    if runner is not None:
        actual_time = float(runner.truth_target.physical_time_at(runner.truth_target.current_q))
    start_time = float(elapsed[0]) if elapsed else 0.
    end_time, end_stage = actual_time, 'approach'
    if completion is not None and completion.lifecycle is not None:
        end_time, end_stage = completion_progress_endpoint(completion.lifecycle, actual_time)
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
        outer_effort = float(sum(values.get('applied_control_effort', [])))
        effort[key] = dict(applied_input_squared_integral_per_reference_anomaly=(outer_effort if held_effort is None else held_effort[key]),
            complete_outer_applied_input_squared_integral_per_reference_anomaly=outer_effort,
            applied_control_rms=values.get('applied_control_rms', []), ancillary_control_rms=values.get('ancillary_control_rms', []),
            semantics=('Complete-outer history only; acknowledged-hold accounting unavailable' if accounting is None else
                'Every acknowledged completed held input weighted by its exact reference-anomaly width; unconfirmed motion excluded')+'; not physical force or measured fuel',
            rms_scope='Complete outer updates only')
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
    physical_complete = bool(completion is not None and completion.success)
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
        partial_execution_accounting=accounting,
        sampled_diagnostics_scope=dict(exact_end_q=None if accounting is None else accounting['complete_outer_end_exact_q'],
            updates=len(approach.anom_history)-1, scope='Safety minima, tube/bandwidth histories and RMS arrays cover complete outer samples only'),
        set_width=widths, physical_adaptation=metadata.get('physical_adaptation'), effort=effort,
        fuel_bound=None if completion is None or completion.translation_budget is None else completion.translation_budget.to_dict(),
        fuel_bound_unavailable=None if not fuel_unavailable.is_file() else json.loads(fuel_unavailable.read_text(encoding='utf-8')),
        fallback_counts=sources, deadline_misses=metadata.get('observed_deadline_misses'),
        deadline_unknown_updates=metadata.get('observed_deadline_unknown_updates'),
        planner_only_deadline_misses=metadata.get('planner_ready_deadline_misses'),
        deadline_scope=metadata.get('timing_scope'), attempted_updates=metadata.get('attempted_updates'),
        update_wall_timing=metadata.get('update_wall_timing', []), failures=failure,
        completion=None if completion is None else completion.to_dict())
    return result
