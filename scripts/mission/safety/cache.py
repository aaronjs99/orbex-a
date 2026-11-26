"""cache.py: Frozen proof identities with mutable-context guards.

Copyright: Aaron John Sabu
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import fields, is_dataclass

from orbexa.control.fallback.refuge import fingerprint


class _FrozenProofIdentityCache:
    """Bounded identities for frozen proof payloads with mutable-context guards.

    Flow/attitude dataclass fields, segment tuples and interval boxes are frozen.
    Their mapping/list contexts are not: every lookup deeply hashes those small
    contexts before reusing the expensive full continuous-segment identity.
    """

    def __init__(self, maximum_objects=4):
        """Store the object limit and an empty least-recently-used entry map."""
        self.maximum_objects, self.entries = maximum_objects, OrderedDict()

    def identity(self, value):
        """Return the fingerprint of a frozen proof dataclass, reusing a cached one.

        Reuse requires the same object and an unchanged deep fingerprint of its Mapping
        and list fields. Otherwise value.fingerprint is recomputed, stored, and the
        oldest entries beyond maximum_objects are evicted.

        Inputs: value, a frozen dataclass exposing a fingerprint attribute.
        Outputs: the fingerprint. Raises TypeError for a non-dataclass or unfrozen
            value.
        """
        if not is_dataclass(value) or not value.__dataclass_params__.frozen:
            raise TypeError(
                "Only frozen proof dataclass payloads can reuse a cached identity"
            )
        contexts = {
            f.name: getattr(value, f.name)
            for f in fields(value)
            if isinstance(getattr(value, f.name), (Mapping, list))
        }
        guard = fingerprint(contexts)
        key = id(value)
        entry = self.entries.get(key)
        if entry is not None and entry[0] is value and entry[1] == guard:
            self.entries.move_to_end(key)
            return entry[2]
        identity = value.fingerprint
        self.entries[key] = (value, guard, identity)
        self.entries.move_to_end(key)
        while len(self.entries) > self.maximum_objects:
            self.entries.popitem(last=False)
        return identity
