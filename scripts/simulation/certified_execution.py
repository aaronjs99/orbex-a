"""Conditional continuous-flow admission for the sampled mission executor.

The state estimator supplies a declared valid componentwise measurement box.
After the first hold that box is intersected with the previous certified
endpoint, never substituted for it. A missing proof refuses the next command.
This guard proves admitted motion; admissible continuation is a separate
controller contract in ``orbexa.control.continuation``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
import json
from typing import Callable, Mapping, Any
import numpy as np

from orbexa.control.reachability import (
    IntervalBox, OrbitalModelBounds, HeldInputRequest, ValidatedFlowEncloser,
    SweptCollisionGuard, RotatingCylinderBounds,
)
from orbexa.control.composite_geometry import CompositeSweptCollisionGuard, RotatingCompositeBounds


@dataclass(frozen=True)
class GuardDecision:
    admissible: bool
    reason: str
    evidence: dict


class CertifiedExecutionGuard:
    """Admit all fleet commands together before their physical hold starts.

    ``measurement_radius`` bounds [position m, anomaly velocity m/rad] errors;
    ``acceleration_radius`` bounds additive acceleration in m/rad². Pose/rate
    bounds returned by ``target_bounds`` must hold throughout the supplied
    interval. These are explicit model assumptions, not estimates made by this
    class. A record sink may retain full rational certificates outside the repo.
    """

    def __init__(self, *, model: OrbitalModelBounds, measurement_radius,
                 acceleration_radius, body_radius: float,
                 input_lower, input_upper, minimum_pair_separation: float,
                 target_bounds: Callable[[float, float], RotatingCylinderBounds | RotatingCompositeBounds],
                 flow_encloser=None, record_sink: Callable[[dict], None] | None = None):
        self.model, self.target_bounds = model, target_bounds
        self.measurement_radius = self._vector(measurement_radius, 6, 'measurement_radius', nonnegative=True)
        self.acceleration_radius = self._vector(acceleration_radius, 3, 'acceleration_radius', nonnegative=True)
        self.lower = self._vector(input_lower, 3, 'input_lower')
        self.upper = self._vector(input_upper, 3, 'input_upper')
        if np.any(self.lower > self.upper):
            raise ValueError('Invalid actuator interval')
        if (not np.isfinite(body_radius) or body_radius < 0 or
                not np.isfinite(minimum_pair_separation) or minimum_pair_separation < 2 * body_radius):
            raise ValueError('Finite bodies require valid center separation')
        self.body_radius, self.pair_separation = float(body_radius), float(minimum_pair_separation)
        self.encloser = flow_encloser if flow_encloser is not None else ValidatedFlowEncloser()
        self.collision = SweptCollisionGuard()
        self.composite_collision = CompositeSweptCollisionGuard()
        self.record_sink = record_sink
        self.endpoints: dict[str, IntervalBox] = {}
        self.last_end: float | None = None
        self.exact_end: Fraction | None = None
        self.max_reported_clock_difference = 0.0
        self.pending: tuple[float, dict[str, IntervalBox], dict] | None = None
        self.admitted_holds = self.rejected_holds = 0
        self.last_rejection: str | None = None
        self.clock_start = None
        self.last_hold_width = None
        self.target_bound_context = None
        self.restoration_history = []

    @staticmethod
    def _vector(value, count, name, nonnegative=False):
        value = np.asarray(value, dtype=float)
        if value.shape != (count,) or not np.all(np.isfinite(value)) or (nonnegative and np.any(value < 0)):
            raise ValueError(f'{name} must be a finite {count}-vector' + (' with nonnegative components' if nonnegative else ''))
        return value.copy()

    @staticmethod
    def _target_context(target):
        if isinstance(target, RotatingCompositeBounds):
            names = ('shape', 'geometry', 'geometry_fingerprint', 'angular_rate_per_anomaly',
                     'center_speed_per_anomaly', 'orientation_matrix_error', 'frame', 'scope')
        elif isinstance(target, RotatingCylinderBounds):
            # Preserve the exact context schema of the qualified cylinder baseline.
            names = ('radius', 'half_length', 'angular_rate_per_anomaly', 'center_speed_per_anomaly',
                     'orientation_matrix_error', 'frame', 'scope')
        else:
            raise TypeError('Unsupported occupied target geometry; explicit cylinder or composite bounds required')
        record = target.as_dict()
        result = {name: record[name] for name in names}
        if isinstance(target, RotatingCompositeBounds) and 'rotation_rate_bounds' in record:
            result['rotation_rate_bounds'] = record['rotation_rate_bounds']
        return result

    @staticmethod
    def _shape_identity(context):
        if context.get('shape') == 'composite_spacecraft':
            return ('composite_spacecraft', context.get('geometry_fingerprint'), context.get('geometry'))
        if 'shape' in context:
            raise ValueError('Unsupported continuous target geometry identity')
        return ('cylinder', context.get('radius'), context.get('half_length'))

    def prepare(self, states: Mapping[str, Any], inputs: Mapping[str, Any], anomaly: float, width: float) -> GuardDecision:
        """Prove a complete hold without advancing the committed proof clock."""
        self.pending = None
        ids = tuple(states)
        if not ids or set(ids) != set(inputs) or (self.endpoints and set(ids) != set(self.endpoints)):
            return self._reject('Fleet changed inside a continuous certificate')
        tolerance = 32 * np.spacing(max(1.0, abs(anomaly))) if np.isfinite(anomaly) else 0.0
        if (not np.isfinite(anomaly) or not np.isfinite(width) or width <= 0 or
                (self.last_end is not None and abs(anomaly - self.last_end) > tolerance)):
            return self._reject('Noncontiguous or invalid continuous proof clock')
        exact_start = Fraction.from_float(float(anomaly)) if self.exact_end is None else self.exact_end
        exact_width = Fraction.from_float(float(width))
        discrepancy = abs(float(exact_start) - anomaly)
        if discrepancy > 1e-10:
            return self._reject('Reported anomaly departed from the exact accumulated hold clock')
        self.max_reported_clock_difference = max(self.max_reported_clock_difference, discrepancy)
        flows = []
        evidence = {'anomaly': float(anomaly), 'width': float(width), 'chaser_ids': ids, 'flows': {}}
        for key in ids:
            report = self._vector(states[key], 6, f'{key} measurement')
            command = self._vector(inputs[key], 3, f'{key} command')
            if np.any(command < self.lower) or np.any(command > self.upper):
                return self._reject(f'{key}: command outside actuator box', evidence)
            measured = IntervalBox.from_center_radius(report, self.measurement_radius)
            if key in self.endpoints:
                old = self.endpoints[key]
                lo = tuple(max(a, b) for a, b in zip(old.lower, measured.lower))
                hi = tuple(min(a, b) for a, b in zip(old.upper, measured.upper))
                if any(a > b for a, b in zip(lo, hi)):
                    return self._reject(f'{key}: measurement contradicts certified endpoint', evidence)
                initial = IntervalBox(lo, hi)
            else:
                initial = measured
            request = HeldInputRequest(initial_state=initial, held_input=IntervalBox.point(command),
                start_anomaly=(exact_start, exact_start), hold_width=(exact_width, exact_width),
                acceleration_disturbance=IntervalBox.from_center_radius(np.zeros(3), self.acceleration_radius))
            flow = self.encloser.enclose_orbital(request, self.model)
            evidence['flows'][key] = flow.as_dict()
            if not flow.proven:
                return self._reject(f'{key}: {flow.reason}', evidence)
            flows.append(flow)
        try:
            target = self.target_bounds(float(exact_start), float(width))
            target_context = self._target_context(target)
            if (self.target_bound_context is not None and self._shape_identity(target_context)
                    != self._shape_identity(self.target_bound_context)):
                return self._reject('Occupied target geometry changed inside a continuous certificate', evidence)
        except (ValueError, TypeError, KeyError) as error:
            return self._reject(f'Unsupported or invalid target geometry: {error}', evidence)
        evidence['target_bound_context'] = target_context
        pose_clock_error = abs(Fraction.from_float(float(exact_start)) - exact_start)
        if pose_clock_error:
            # The supplier evaluates the pose at a rounded binary label. Carry
            # that clock difference into the proper-rotation operator bound.
            if target.orientation_matrix_at_start is None:
                return self._reject('Canonical clock adjustment requires matrix pose bounds', evidence)
            center_padding = target.center_speed_per_anomaly * pose_clock_error
            center = IntervalBox(tuple(v - center_padding for v in target.center_at_start.lower),
                                 tuple(v + center_padding for v in target.center_at_start.upper))
            target = replace(target, center_at_start=center,
                orientation_matrix_error=target.orientation_matrix_error + target.angular_rate_per_anomaly * pose_clock_error)
        collision_guard = self.composite_collision if isinstance(target, RotatingCompositeBounds) else self.collision
        collision = collision_guard.check(flows, [self.body_radius] * len(ids),
            target=target, minimum_pair_center_distance=self.pair_separation)
        evidence['collision'] = collision.as_dict()
        if not collision.admissible:
            return self._reject(collision.reason, evidence)
        end = float(anomaly + width)
        evidence['exact_start_anomaly'] = str(exact_start)
        evidence['exact_end_anomaly'] = str(exact_start + exact_width)
        self.pending = (end, {key: flow.endpoint for key, flow in zip(ids, flows)}, evidence)
        return GuardDecision(True, collision.reason, evidence)

    def commit(self, anomaly: float):
        """Commit only after the mission commits that exact simultaneous hold."""
        if self.pending is None or anomaly != self.pending[0]:
            raise ValueError('No matching certified hold is pending')
        end, endpoints, evidence = self.pending
        self.endpoints, self.last_end, self.pending = endpoints, end, None
        self.exact_end = Fraction(evidence['exact_end_anomaly'])
        self.last_hold_width = evidence['width']
        self.target_bound_context = evidence['target_bound_context']
        self.admitted_holds += 1
        if self.clock_start is None:
            self.clock_start = evidence['anomaly']
        if self.record_sink is not None:
            self.record_sink({'admitted': True, **evidence})

    def _reject(self, reason: str, evidence=None):
        self.rejected_holds += 1
        self.last_rejection = reason
        record = {'admitted': False, 'reason': reason, **(evidence or {})}
        if self.record_sink is not None:
            self.record_sink(record)
        return GuardDecision(False, reason, record)

    def restore(self, metadata, *, chaser_ids, anomaly, hold_width,
                source_context=None, checkpoint_identity=None):
        """Restore a trusted, source-bound committed proof endpoint atomically.

        A numerical point never replaces the saved exact endpoint enclosure.
        Older records missing actuator bounds require explicit frozen-source
        context. This restores conditional proof continuity; it does not prove
        historical metadata or create an admissible fallback on solver failure.
        """
        if self.endpoints or self.pending is not None or self.admitted_holds or self.exact_end is not None:
            raise ValueError('Restore continuous state only into a fresh execution guard')
        context = dict(source_context or {})
        if metadata.get('method') != 'arb-picard-continuous-v1':
            raise ValueError('Unsupported continuous certificate method')
        recorded_model = OrbitalModelBounds(**metadata['model_bounds'])
        if recorded_model.as_dict() != self.model.as_dict():
            raise ValueError('Continuous restart changed the retained physical parameter model')
        for name, current, count in (('measurement_radius', self.measurement_radius, 6),
                                     ('acceleration_radius', self.acceleration_radius, 3)):
            if not np.array_equal(self._vector(metadata[name], count, name), current):
                raise ValueError(f'Continuous restart changed {name}')
        if (metadata['body_radius_m'] != self.body_radius or
                metadata['minimum_pair_separation_m'] != self.pair_separation):
            raise ValueError('Continuous restart changed occupied-body or pair geometry')
        bounds = metadata
        if 'input_lower' not in metadata or 'input_upper' not in metadata:
            if not context.get('source_reference') or 'input_lower' not in context or 'input_upper' not in context:
                raise ValueError('Older continuous records require explicit frozen-source actuator context')
            bounds = context
        if (not np.array_equal(self._vector(bounds['input_lower'], 3, 'source input_lower'), self.lower) or
                not np.array_equal(self._vector(bounds['input_upper'], 3, 'source input_upper'), self.upper)):
            raise ValueError('Continuous restart changed actuator assumptions')
        if metadata.get('last_hold_width') is not None and metadata['last_hold_width'] != hold_width:
            raise ValueError('Continuous restart changed the held execution interval')
        ids = tuple(chaser_ids)
        count, rejected = metadata['admitted_holds'], metadata['rejected_holds']
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (count, rejected)):
            raise ValueError('Invalid historical guard counters')
        if not np.isfinite(anomaly) or not np.isfinite(hold_width) or hold_width <= 0:
            raise ValueError('Invalid restored hold clock')
        endpoints = {key: IntervalBox(value['exact_lower'], value['exact_upper'])
                     for key, value in metadata['endpoint_boxes'].items()}
        exact = None if metadata.get('exact_end_anomaly') is None else Fraction(metadata['exact_end_anomaly'])
        last = metadata.get('last_committed_anomaly')
        if count:
            tolerance = 32 * np.spacing(max(1., abs(anomaly)))
            if (set(endpoints) != set(ids) or any(box.dimension != 6 for box in endpoints.values()) or
                    exact is None or last is None or not np.isfinite(last) or abs(last - anomaly) > tolerance or
                    abs(float(exact) - anomaly) > 1e-10):
                raise ValueError('Continuous checkpoint clock, fleet and exact endpoint boxes disagree')
        elif endpoints or exact is not None or last is not None or anomaly != 0.:
            raise ValueError('A noninitial numerical state has no committed continuous endpoint')
        target_context = metadata.get('target_bound_context')
        if target_context is None and count:
            if not context.get('source_reference') or not context.get('target_bound_context'):
                raise ValueError('Older continuous records require explicit frozen-source target bound context')
            target_context = context['target_bound_context']
        if target_context is not None:
            if not isinstance(target_context, dict):
                raise ValueError('Complete continuous target bound declarations are required')
            proposed = self._target_context(self.target_bounds(float(exact) if exact is not None else anomaly, hold_width))
            # Compare the same JSON representation for live tuple-based proof
            # fields and persisted lists; retain all exact rational values.
            normalized_proposed = json.loads(json.dumps(proposed, allow_nan=False))
            normalized_recorded = json.loads(json.dumps(target_context, allow_nan=False))
            if normalized_proposed != normalized_recorded:
                raise ValueError('Continuous restart changed target occupied-shape, pose or rate assumptions')
        clock_difference = float(metadata.get('max_reported_clock_difference_rad', 0))
        if not np.isfinite(clock_difference) or not 0 <= clock_difference <= 1e-10:
            raise ValueError('Invalid historical clock discrepancy')
        history = metadata.get('restoration_history', [])
        if not isinstance(history, list) or not all(isinstance(value, dict) for value in history):
            raise ValueError('Invalid continuous restoration history')
        restoration_history = list(history) + [dict(
            checkpoint_identity=checkpoint_identity, exact_end_anomaly=None if exact is None else str(exact),
            source_reference=context.get('source_reference'), policy='Restore exact committed boxes and clock; no numerical-point replacement')]
        # All compatibility decisions above precede the atomic restoration.
        self.endpoints, self.exact_end, self.last_end = endpoints, exact, last
        self.admitted_holds, self.rejected_holds = count, rejected
        self.last_rejection, self.clock_start = metadata.get('last_rejection'), metadata.get('start_anomaly')
        self.last_hold_width = metadata.get('last_hold_width', hold_width)
        self.target_bound_context = target_context
        self.max_reported_clock_difference = max(clock_difference, 0. if exact is None else abs(float(exact) - anomaly))
        self.restoration_history = restoration_history

    def as_metadata(self):
        return dict(method='arb-picard-continuous-v1', admitted_holds=self.admitted_holds,
            rejected_holds=self.rejected_holds, last_rejection=self.last_rejection,
            start_anomaly=self.clock_start, last_committed_anomaly=self.last_end,
            exact_end_anomaly=None if self.exact_end is None else str(self.exact_end),
            max_reported_clock_difference_rad=self.max_reported_clock_difference,
            model_bounds=self.model.as_dict(), measurement_radius=self.measurement_radius.tolist(),
            acceleration_radius=self.acceleration_radius.tolist(), body_radius_m=self.body_radius,
            input_lower=self.lower.tolist(), input_upper=self.upper.tolist(),
            last_hold_width=self.last_hold_width, target_bound_context=self.target_bound_context,
            restoration_history=list(self.restoration_history),
            minimum_pair_separation_m=self.pair_separation,
            endpoint_boxes={key: box.as_dict() for key, box in self.endpoints.items()},
            assumptions=['Every measurement box contains the continuous physical state at the canonical shared hold event',
                'All spacecraft start and stop each hold at the same physical events; proof clock is the exact sum of binary hold widths',
                'Initial prior contains fixed physical parameters; no controller contraction is used to narrow this guard',
                'Target pose, rotation-rate and occupied-shape bounds are valid over each complete hold',
                'Command is held exactly; additive acceleration obeys its declared bound'],
            scope='Continuous collision exclusion for admitted holds; this guard alone does not guarantee a feasible successor command')
