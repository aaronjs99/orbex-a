"""common.py: Exact fleet identity, ordered member and plant-context checks.

Copyright: Aaron John Sabu
"""

from __future__ import annotations

from collections.abc import Mapping

from orbexa.control.fallback.refuge import ControlContext


def _pairs(value, name):
    """Return the named members of a fleet mapping or pair sequence as a tuple.

    Inputs: ``value`` is a mapping or an iterable of ``(key, item)`` pairs; ``name``
    labels the error messages.
    Outputs: ``tuple`` of ``(key, item)`` pairs in input order. Raises ValueError
    when there are no members, a key is not a nonblank string, or a key repeats."""
    items = tuple(value.items()) if isinstance(value, Mapping) else tuple(value)
    if not items or any(
        not isinstance(key, str) or not key.strip() for key, _ in items
    ):
        raise ValueError(f"{name} requires nonempty named fleet members")
    if len({key for key, _ in items}) != len(items):
        raise ValueError(f"{name} contains duplicate identities")
    return items


def _same_plant(left: ControlContext, right: ControlContext):
    """Return True when two control contexts agree on every plant identity field.

    The compared fields are ``model_id``, ``parameter_set_id``, ``target_model_id``,
    ``frame``, ``clock_id``, ``sensor_id``, ``actuator_id``, ``geometry_id`` and
    ``fuel_domain_id``."""
    return all(
        getattr(left, name) == getattr(right, name)
        for name in (
            "model_id",
            "parameter_set_id",
            "target_model_id",
            "frame",
            "clock_id",
            "sensor_id",
            "actuator_id",
            "geometry_id",
            "fuel_domain_id",
        )
    )
