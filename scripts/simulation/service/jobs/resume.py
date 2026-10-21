"""Explicit historical source selection and authenticated journal retention."""
from __future__ import annotations

import gzip
import json
import math
from fractions import Fraction
from pathlib import Path


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


def _append_resume_journal(checkpoint, path, destination):
    """Bind the complete saved prefix before appending newly executed holds.

    This authenticates the retained record/clock/enclosure linkage. The public
    propellant accountant separately reproduces every flow and swept witness
    before consuming the combined journal for a smaller fuel allowance.
    """
    from orbexa.control.certification.reachability import IntervalBox, OrbitalModelBounds
    from orbexa.simulation.approach.physical.admission import CertifiedExecutionGuard
    from orbexa.simulation.approach.records.checkpoint import digest
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
