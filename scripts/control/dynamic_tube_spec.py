# /***********************************************************
# *                                                         *
# * Copyright (c) 2026                                      *
# *                                                         *
# * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
# * University of California, Los Angeles                   *
# *                                                         *
# * Authors: Aaron John Sabu, Brett T. Lopez                *
# * Contact: {aaronjs, btlopez}@ucla.edu                    *
# *                                                         *
# ***********************************************************/

"""Typed coupled-tube data at the solver boundary.

Without optional declarations, conversion preserves the existing
``MPCProblem.extra_params['dynamic_tube']`` mapping and backend expressions.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import json
from typing import Any, Dict, Mapping, Tuple
from numbers import Real
import numpy as np
from orbexa.control.terminal_region import TerminalRegion
from orbexa.control.contact_admission import ContactGoalSpeedBound
from orbexa.core.reference_frame import exact, fingerprint


@dataclass(frozen=True)
class ProspectiveContactConstraint:
    """Sealed future-node speed declarations, without a future Nav claim.

    The caller authenticates the complete current-navigation/global-attitude
    forecast before constructing this record.  The planning context separately
    checks its residual model/target identities: adaptation need not make those
    identical to the global physical forecast.  Every export freshly verifies
    the helper algebra and complete record identity; no source cache is used.
    """
    _record_json: str

    @staticmethod
    def _validated(record):
        if not isinstance(record, Mapping):
            raise ValueError('Prospective contact evidence must be a mapping')
        required = {'schema', 'source', 'limits', 'nodes'}
        if not required <= set(record) <= required | {'identity', 'reach_policy'}:
            raise ValueError('Malformed prospective contact evidence fields')
        if record['schema'] != 'orbexa.prospective-contact-terminal.v1':
            raise ValueError('Unsupported prospective contact evidence schema')
        source = dict(record['source'])
        fields_required = {'kind', 'agent_id', 'exact_start_q', 'exact_step_q',
            'navigation_q', 'navigation_fingerprint', 'clock_fingerprint',
            'global_model_fingerprint', 'global_target_flow_fingerprint',
            'attitude_flow_fingerprint', 'moving_goal_forecast_id',
            'residual_model_fingerprint', 'residual_target_flow_fingerprint', 'frame'}
        if (set(source) != fields_required or
                source['kind'] != 'current-navigation-future-contact-forecast-v1' or
                source['frame'] != 'reference-orbit-frame-v1'):
            raise ValueError('Prospective contact source kind/frame/fields differ')
        if any(not isinstance(value, str) or not value.strip() for value in source.values()):
            raise ValueError('Prospective contact source identities must be nonempty strings')
        start, step, nav_q = map(exact, (source['exact_start_q'], source['exact_step_q'], source['navigation_q']))
        if step <= 0 or nav_q != start:
            raise ValueError('A future contact forecast must start at its current navigation event')
        source.update(exact_start_q=str(start), exact_step_q=str(step), navigation_q=str(nav_q))
        limits = dict(record['limits'])
        if set(limits) != {'max_spin_error_rad_s', 'gripper_offset_m', 'max_relative_speed_m_s'}:
            raise ValueError('Prospective contact limits must come from the capture declaration')
        for name, value in limits.items():
            if isinstance(value, bool):
                raise ValueError('Prospective contact limits must be finite exact scalars')
            value = exact(value)
            if value < 0 or (name == 'max_relative_speed_m_s' and value == 0) or not np.isfinite(float(value)):
                raise ValueError('Prospective contact limits must be finite and nonnegative, with positive speed')
            limits[name] = str(value)
        nodes = []
        for index, item in enumerate(record['nodes']):
            if (not isinstance(item, Mapping) or set(item) != {'index', 'exact_q', 'goal_bound_id', 'goal_speed_bound'} or
                    isinstance(item['index'], bool) or not isinstance(item['index'], int) or item['index'] != index):
                raise ValueError('Prospective contact nodes must be the complete ordered prediction grid')
            q = exact(item['exact_q'])
            if q != start + index * step:
                raise ValueError('Prospective contact node differs from its exact future grid')
            goal_id = item['goal_bound_id']
            if not isinstance(goal_id, str) or not goal_id.strip():
                raise ValueError('Each future contact goal needs its forecast bound identity')
            bound = ContactGoalSpeedBound.from_dict(item['goal_speed_bound'])
            reference = bound.reference_kinematics
            if reference is not None and (reference.exact_q != q or
                    reference.clock_fingerprint != source['clock_fingerprint'] or
                    reference.goal_bound_id != goal_id or reference.frame != source['frame']):
                raise ValueError('Future reference kinematics differ from their node/clock/goal source')
            nodes.append(dict(index=index, exact_q=str(q), goal_bound_id=goal_id,
                              goal_speed_bound=bound.as_dict()))
        if len(nodes) < 2:
            raise ValueError('A prospective contact forecast needs at least two prediction nodes')
        result = dict(schema=record['schema'], source=source, limits=limits, nodes=nodes)
        if record.get('reach_policy') is not None:
            from orbexa.control.contact_admission import ContactReachPolicy
            result['reach_policy'] = ContactReachPolicy.from_dict(record['reach_policy']).as_dict()
        result['identity'] = fingerprint(result)
        if 'identity' in record and record['identity'] != result['identity']:
            raise ValueError('Prospective contact evidence identity differs from its contents')
        return result

    @classmethod
    def from_dict(cls, record):
        try:
            return cls(json.dumps(cls._validated(record), sort_keys=True, separators=(',', ':'), allow_nan=False))
        except (TypeError, KeyError, OverflowError, ArithmeticError) as error:
            raise ValueError('Malformed prospective contact evidence') from error

    def as_dict(self):
        try:
            return self._validated(json.loads(self._record_json))
        except (TypeError, KeyError, OverflowError, ArithmeticError) as error:
            raise ValueError('Malformed sealed prospective contact evidence') from error

    @property
    def identity(self):
        return self.as_dict()['identity']

    def node_bounds(self):
        return tuple(ContactGoalSpeedBound.from_dict(node['goal_speed_bound'])
                     for node in self.as_dict()['nodes'])

    def reach_policy(self):
        record = self.as_dict().get('reach_policy')
        if record is None:
            return None
        from orbexa.control.contact_admission import ContactReachPolicy
        return ContactReachPolicy.from_dict(record)

    def validate_grid(self, start_q, step_q, count):
        record = self.as_dict()
        if (exact(record['source']['exact_start_q']) != exact(start_q) or
                exact(record['source']['exact_step_q']) != exact(step_q) or len(record['nodes']) != count):
            raise ValueError('Prospective contact evidence differs from the complete problem grid')
        return record


@dataclass
class DynamicTubeSpec:
    phi_initial: np.ndarray
    omega_initial: np.ndarray
    lambda_gain: np.ndarray
    alpha_initial: np.ndarray
    alpha_bounds: Tuple[float, float]
    alpha_rate_limit: float
    delta_A: np.ndarray
    delta_d: np.ndarray
    A_abs: np.ndarray
    A_variation: np.ndarray
    d_abs: np.ndarray
    hold_width: float
    eta: np.ndarray
    bandwidth_weight: float
    tube_weight: float
    terminal_position_tolerance: float
    terminal_velocity_tolerance: float
    enforce_terminal: bool = False
    terminal_start_index: int | None = None
    terminal_region: TerminalRegion | None = None
    terminal_planning_fraction: float | None = None
    measurement_error_abs: np.ndarray | None = None
    nominal_endpoint_error_abs: np.ndarray | None = None
    stage_peak_padding_abs: np.ndarray | None = None
    contact_terminal: ProspectiveContactConstraint | None = None

    def __post_init__(self) -> None:
        delta_a = np.asarray(self.delta_A, dtype=float)
        if delta_a.ndim != 3 or delta_a.shape[1:] != (3, 6) or len(delta_a) < 1:
            raise ValueError("dynamic_tube.delta_A must have shape (intervals, 3, 6) with at least one interval")
        intervals = len(delta_a)
        if self.measurement_error_abs is not None:
            observed = np.asarray(self.measurement_error_abs, dtype=float)
            if observed.shape != (6,) or not np.all(np.isfinite(observed)) or np.any(observed < 0):
                raise ValueError('dynamic_tube.measurement_error_abs must be a finite nonnegative six-vector')
            self.measurement_error_abs = observed.copy()
        if self.nominal_endpoint_error_abs is not None:
            endpoint = np.asarray(self.nominal_endpoint_error_abs,dtype=float)
            if endpoint.shape != (6,) or not np.all(np.isfinite(endpoint)) or np.any(endpoint < 0):
                raise ValueError('dynamic_tube.nominal_endpoint_error_abs must be a finite nonnegative six-vector')
            self.nominal_endpoint_error_abs = endpoint.copy()
        if self.stage_peak_padding_abs is not None:
            padding=np.asarray(self.stage_peak_padding_abs,dtype=float)
            if padding.shape!=(6,) or not np.all(np.isfinite(padding)) or np.any(padding<0):
                raise ValueError('dynamic_tube.stage_peak_padding_abs must be a finite nonnegative six-vector')
            self.stage_peak_padding_abs=padding.copy()
        shapes = {
            "phi_initial": (3,), "omega_initial": (3,), "lambda_gain": (3,),
            "alpha_initial": (3,), "eta": (3,), "delta_A": (intervals, 3, 6),
            "delta_d": (intervals, 3), "A_abs": (intervals, 3, 6),
            "A_variation": (intervals, 3, 6), "d_abs": (intervals, 3),
        }
        for name, shape in shapes.items():
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != shape or not np.all(np.isfinite(value)) or np.any(value < 0):
                raise ValueError(f"dynamic_tube.{name} must be finite, nonnegative and have shape {shape}")
            setattr(self, name, value.copy())
        if np.any(self.lambda_gain <= 0):
            raise ValueError("dynamic_tube.lambda_gain must be positive")
        self.hold_width = float(self.hold_width)
        if not np.isfinite(self.hold_width) or self.hold_width <= 0:
            raise ValueError("dynamic_tube.hold_width must be finite and positive")
        bounds = np.asarray(self.alpha_bounds, dtype=float)
        if (bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or
                not 0 < bounds[0] <= bounds[1]):
            raise ValueError("dynamic_tube.alpha_bounds must be finite, ordered and positive")
        self.alpha_bounds = (float(bounds[0]), float(bounds[1]))
        if np.any(self.alpha_initial < bounds[0]) or np.any(self.alpha_initial > bounds[1]):
            raise ValueError("dynamic_tube.alpha_initial must belong to alpha_bounds")
        for name in ("alpha_rate_limit", "bandwidth_weight", "tube_weight"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"dynamic_tube.{name} must be finite and nonnegative")
        for name in ("terminal_position_tolerance", "terminal_velocity_tolerance"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"dynamic_tube.{name} must be finite and positive")
        if not isinstance(self.enforce_terminal, bool):
            raise ValueError("dynamic_tube.enforce_terminal must be a boolean")
        if self.terminal_start_index is not None:
            if (isinstance(self.terminal_start_index, bool) or
                    not isinstance(self.terminal_start_index, (int, np.integer)) or
                    not 0 <= self.terminal_start_index < self.horizon_steps or not self.enforce_terminal):
                raise ValueError("An enabled terminal suffix requires an integer start index inside the horizon")
            self.terminal_start_index = int(self.terminal_start_index)
        if self.terminal_region is not None:
            if isinstance(self.terminal_region, dict):
                self.terminal_region = TerminalRegion.from_dict(self.terminal_region)
            if not isinstance(self.terminal_region, TerminalRegion) or not self.enforce_terminal:
                raise ValueError("A required terminal region needs enabled terminal constraints")
        if self.terminal_planning_fraction is not None:
            fraction = self.terminal_planning_fraction
            if (isinstance(fraction, bool) or not isinstance(fraction, Real) or not np.isfinite(float(fraction))
                    or not 0 < fraction <= 1 or not self.enforce_terminal):
                raise ValueError("An enabled terminal planning fraction must lie in (0,1]")
            self.terminal_planning_fraction = float(fraction)
        if self.contact_terminal is not None:
            if isinstance(self.contact_terminal, Mapping):
                self.contact_terminal = ProspectiveContactConstraint.from_dict(self.contact_terminal)
            if not isinstance(self.contact_terminal, ProspectiveContactConstraint) or not self.enforce_terminal:
                raise ValueError('Prospective contact speed constraints require an enabled terminal suffix')
            if len(self.contact_terminal.as_dict()['nodes']) != self.horizon_steps:
                raise ValueError('Prospective contact evidence must cover the entire prediction grid')

    @property
    def horizon_steps(self) -> int:
        return len(self.delta_A) + 1

    def to_extra_params(self) -> Dict[str, Any]:
        """Return the current backend's dynamic-tube mapping without reformulation."""
        result = {item.name: getattr(self, item.name) for item in fields(self)
                  if item.name not in ("terminal_start_index", "terminal_region", "terminal_planning_fraction", "measurement_error_abs", "nominal_endpoint_error_abs", "stage_peak_padding_abs", "contact_terminal")}
        if self.measurement_error_abs is not None:
            result['measurement_error_abs'] = self.measurement_error_abs.copy()
        if self.nominal_endpoint_error_abs is not None:
            result['nominal_endpoint_error_abs'] = self.nominal_endpoint_error_abs.copy()
        if self.stage_peak_padding_abs is not None:
            result['stage_peak_padding_abs']=self.stage_peak_padding_abs.copy()
        if self.terminal_start_index is not None:
            result["terminal_start_index"] = self.terminal_start_index
        if self.terminal_region is not None:
            result["terminal_region"] = self.terminal_region.as_dict()
        if self.terminal_planning_fraction is not None:
            result["terminal_planning_fraction"] = self.terminal_planning_fraction
        if self.contact_terminal is not None:
            result['contact_terminal'] = self.contact_terminal.as_dict()
        return result

    @classmethod
    def from_extra_params(cls, parameters: Mapping[str, Any]) -> "DynamicTubeSpec":
        return cls(**dict(parameters))
