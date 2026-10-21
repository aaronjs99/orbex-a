"""Exact fleet identity, ordered member and plant-context checks."""
from __future__ import annotations

from typing import Mapping
from orbexa.control.fallback.refuge import ControlContext


def _pairs(value, name):
    items = tuple(value.items()) if isinstance(value, Mapping) else tuple(value)
    if not items or any(not isinstance(key, str) or not key.strip() for key, _ in items):
        raise ValueError(f"{name} requires nonempty named fleet members")
    if len({key for key, _ in items}) != len(items):
        raise ValueError(f"{name} contains duplicate identities")
    return items


def _same_plant(left: ControlContext, right: ControlContext):
    return all(getattr(left, name) == getattr(right, name) for name in
        ("model_id", "parameter_set_id", "target_model_id", "frame", "clock_id", "sensor_id",
         "actuator_id", "geometry_id", "fuel_domain_id"))
