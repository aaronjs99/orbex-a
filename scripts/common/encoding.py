"""encoding.py: Exact-record encoding of rationals, records and containers for certificate dictionaries.

Copyright: Aaron John Sabu
"""

from __future__ import annotations

from fractions import Fraction


def _encode(value):
    """Encode a value for a certificate dictionary.

    A Fraction becomes its exact string, an object with ``as_dict`` becomes that
    dictionary, dictionary values are encoded recursively, and lists and tuples
    become tuples of encoded items. Any other value is returned unchanged.
    """
    if isinstance(value, Fraction):
        return str(value)
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_encode(item) for item in value)
    return value
