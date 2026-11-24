"""encoding.py: Exact-record encoding of rationals, records and containers for certificate dictionaries.

Copyright: Aaron John Sabu
"""

from __future__ import annotations

from fractions import Fraction


def _encode(value):
    if isinstance(value, Fraction):
        return str(value)
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_encode(item) for item in value)
    return value
