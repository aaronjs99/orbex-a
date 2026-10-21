"""Conservative parameter-box membership for recorded held-input RK4 motion.

The guarantee is conditional on the physical-lvlh-v1 model, its fixed context,
the recorded RK4 subintervals and controls, and a componentwise measurement
error bound. It does not bound RK4's error relative to continuous dynamics or
unmodeled disturbances. A retained box is not a proof that every point in it
fits; exclusion alone is certified by an empty observation intersection.
"""

from collections import deque
from typing import Mapping

import numpy as np

from orbexa.control.tubes.uncertainty import coefficient_intervals


_PARAMETERS = ("eccentricity", "alpha", "beta")


def _add(left, right):
    return (
        np.nextafter(left[0] + right[0], -np.inf),
        np.nextafter(left[1] + right[1], np.inf),
    )


def _multiply(left, right):
    products = np.stack((
        left[0] * right[0], left[0] * right[1],
        left[1] * right[0], left[1] * right[1],
    ))
    return (
        np.nextafter(np.min(products, axis=0), -np.inf),
        np.nextafter(np.max(products, axis=0), np.inf),
    )


def _scale(interval, scalar):
    return _multiply(interval, (scalar, scalar))


def _observation_interval(state, error):
    return _add((state, state), (-error, error))


def _derivative(state, control, coefficients):
    lower, upper, disturbance_lower, disturbance_upper = coefficients
    acceleration = (np.zeros(3), np.zeros(3))
    # Round each product AND each accumulated sum outwards. A final rounded
    # np.sum alone would not enclose its preceding accumulation roundoff.
    for column in range(6):
        term = _multiply(
            (lower[:, column], upper[:, column]),
            (state[0][column], state[1][column]),
        )
        acceleration = _add(acceleration, term)
    acceleration = _add(acceleration, (disturbance_lower, disturbance_upper))
    acceleration = _add(acceleration, (control, control))
    return (
        np.concatenate((state[0][3:], acceleration[0])),
        np.concatenate((state[1][3:], acceleration[1])),
    )


def _rk4_interval(state, control, step, stage_coefficients):
    first, midpoint, last = stage_coefficients
    k1 = _derivative(state, control, first)
    k2 = _derivative(_add(state, _scale(k1, 0.5 * step)), control, midpoint)
    k3 = _derivative(_add(state, _scale(k2, 0.5 * step)), control, midpoint)
    k4 = _derivative(_add(state, _scale(k3, step)), control, last)
    combined = _add(_add(_add(k1, _scale(k2, 2.0)), _scale(k3, 2.0)), k4)
    return _add(state, _scale(combined, step / 6.0))


def _box_consistent(box, states, controls, anomalies, steps, context, error):
    nominal = {key: 0.5 * lo + 0.5 * hi for key, (lo, hi) in box.items()}
    nominal["theta0"] = float(context.get("theta0", 0.0))
    axis = context.get("semi_major_axis")
    if axis is None:
        mu = float(context.get("mu", 3.986004418e14))
        motion = float(context.get("mean_motion", 0.001))
        if mu <= 0.0 or motion <= 0.0:
            raise ValueError("positive mu and mean motion are required")
        axis = (mu / motion**2) ** (1.0 / 3.0)
    for index, step in enumerate(steps):
        q = anomalies[index]
        stage_coefficients = [
            coefficient_intervals(
                stage_q, stage_q, nominal, float(axis),
                e_range=box["eccentricity"],
                a_range=box["alpha"], b_range=box["beta"],
            )
            for stage_q in (q, q + 0.5 * step, q + step)
        ]
        initial = _observation_interval(states[index], error)
        predicted = _rk4_interval(initial, controls[index], step, stage_coefficients)
        if not all(np.all(np.isfinite(endpoint)) for endpoint in predicted):
            raise FloatingPointError("nonfinite interval prediction")
        observation = _observation_interval(states[index + 1], error)
        if np.any(predicted[1] < observation[0]) or np.any(observation[1] < predicted[0]):
            return False
    return True


def contract_parameter_box(
    feasible_sets: Mapping,
    states,
    controls,
    start_anom,
    anom_step,
    dynamics_context: Mapping,
    error_bound=1.0e-7,
    max_boxes=32,
):
    """Return a conservative outer parameter box and bounded-work diagnostics.

    Required ranges are ``eccentricity``, ``alpha`` and ``beta``. Observed
    states have shape (N+1,6), held anomaly accelerations have shape (N,3).
    ``anom_step`` is a positive scalar or the N exact recorded subinterval
    widths. Only the latest eight transitions are considered. The measurement
    error is a nonnegative scalar or six componentwise bounds, applied at
    both endpoints of each transition. No plant truth is accepted or used.

    At most ``max_boxes`` candidate boxes are evaluated. Subdivision is along
    the widest width relative to the initial parameter range. Every retained
    and unevaluated box contributes to the returned hull, including the whole
    queue when the budget ends. If all boxes are excluded, the original ranges
    are returned with ``status='inconsistent'``; an empty set is never emitted.
    Invalid model domains are retained as unresolved rather than excluded.
    """
    original = {key: tuple(map(float, value)) for key, value in feasible_sets.items()}
    for key in _PARAMETERS:
        if key not in original or len(original[key]) != 2:
            raise ValueError(f"A two-endpoint {key} range is required")
        lo, hi = original[key]
        if not np.isfinite(lo) or not np.isfinite(hi) or lo > hi:
            raise ValueError(f"Invalid {key} range")
    original_box = {key: original[key] for key in _PARAMETERS}
    samples = np.asarray(states, dtype=float)
    inputs = np.asarray(controls, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 6:
        raise ValueError("states must have shape (N+1, 6)")
    count = max(0, len(samples) - 1)
    if count and (inputs.ndim != 2 or inputs.shape != (count, 3)):
        raise ValueError("controls must have shape (N, 3), one held input per transition")
    if not np.all(np.isfinite(samples)) or not np.all(np.isfinite(inputs)):
        raise ValueError("observations and held controls must be finite")
    error = np.broadcast_to(np.asarray(error_bound, dtype=float), (6,))
    if np.any(~np.isfinite(error)) or np.any(error < 0.0):
        raise ValueError("measurement error bounds must be finite and nonnegative")
    step_values = np.asarray(anom_step, dtype=float)
    if step_values.ndim == 0:
        steps = np.full(count, float(step_values))
    elif step_values.shape == (count,):
        steps = step_values
    else:
        raise ValueError("anom_step must be a scalar or one width per transition")
    if np.any(~np.isfinite(steps)) or np.any(steps <= 0.0) or not np.isfinite(start_anom):
        raise ValueError("finite anomaly and strictly positive subinterval widths are required")
    budget = int(max_boxes)
    if budget < 0 or budget != max_boxes:
        raise ValueError("max_boxes must be a nonnegative integer")

    diagnostics = {
        "method": "outward_interval_held_input_rk4",
        "guarantee": "conditional_on_physical_lvlh_v1_recorded_rk4_and_componentwise_measurement_error",
        "status": "unchanged",
        "used_substeps": min(count, 8),
        "evaluated_boxes": 0,
        "pruned_boxes": 0,
        "retained_boxes": 0,
        "unprocessed_boxes": 0,
        "unresolved_boxes": 0,
        "max_boxes": budget,
    }
    if not count:
        diagnostics["status"] = "insufficient_data"
        return original, diagnostics

    # Reconstruct floating stage angles with the same sequential held-step
    # convention as the recorded RK4 map, preserving arbitrary step widths.
    anomalies = np.empty(count)
    q = float(start_anom)
    for index, step in enumerate(steps):
        anomalies[index] = q
        q += float(step)
    offset = max(0, count - 8)
    samples, inputs = samples[offset:], inputs[offset:]
    anomalies, steps = anomalies[offset:], steps[offset:]
    initial_widths = {key: hi - lo for key, (lo, hi) in original_box.items()}
    queue = deque([original_box])
    retained = []
    while queue and diagnostics["evaluated_boxes"] < budget:
        box = queue.popleft()
        diagnostics["evaluated_boxes"] += 1
        try:
            consistent = _box_consistent(
                box, samples, inputs, anomalies, steps, dynamics_context, error,
            )
        except (ValueError, FloatingPointError, OverflowError):
            retained.append(box)
            diagnostics["unresolved_boxes"] += 1
            continue
        if not consistent:
            diagnostics["pruned_boxes"] += 1
            continue
        split_key = max(
            _PARAMETERS,
            key=lambda key: (
                (box[key][1] - box[key][0]) / initial_widths[key]
                if initial_widths[key] > 0.0 else -1.0
            ),
        )
        lo, hi = box[split_key]
        midpoint = 0.5 * lo + 0.5 * hi
        if not lo < midpoint < hi:
            retained.append(box)
            continue
        lower_box, upper_box = dict(box), dict(box)
        lower_box[split_key], upper_box[split_key] = (lo, midpoint), (midpoint, hi)
        queue.extend((lower_box, upper_box))

    diagnostics["unprocessed_boxes"] = len(queue)
    retained.extend(queue)
    diagnostics["retained_boxes"] = len(retained)
    if not retained:
        diagnostics["status"] = "inconsistent"
        return original, diagnostics
    result = dict(original)
    for key in _PARAMETERS:
        result[key] = (
            min(box[key][0] for box in retained),
            max(box[key][1] for box in retained),
        )
    contracted = any(result[key] != original[key] for key in _PARAMETERS)
    diagnostics["status"] = (
        "contracted" if contracted else
        "unresolved_domain" if diagnostics["unresolved_boxes"] else "unchanged"
    )
    return result, diagnostics
