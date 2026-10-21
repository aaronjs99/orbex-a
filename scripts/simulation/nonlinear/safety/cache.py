"""Frozen proof identities with mutable-context guards."""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from collections import OrderedDict
from collections.abc import Mapping
from orbexa.control.fallback.refuge import fingerprint


class _FrozenProofIdentityCache:
    """Bounded identities for frozen proof payloads with mutable-context guards.

    Flow/attitude dataclass fields, segment tuples and interval boxes are frozen.
    Their mapping/list contexts are not: every lookup deeply hashes those small
    contexts before reusing the expensive full continuous-segment identity.
    """
    def __init__(self,maximum_objects=4):
        self.maximum_objects,self.entries = maximum_objects,OrderedDict()

    def identity(self,value):
        if not is_dataclass(value) or not value.__dataclass_params__.frozen:
            raise TypeError("Only frozen proof dataclass payloads can reuse a cached identity")
        contexts = {f.name:getattr(value,f.name) for f in fields(value)
                    if isinstance(getattr(value,f.name),(Mapping,list))}
        guard = fingerprint(contexts)
        key = id(value)
        entry = self.entries.get(key)
        if entry is not None and entry[0] is value and entry[1] == guard:
            self.entries.move_to_end(key)
            return entry[2]
        identity = value.fingerprint
        self.entries[key] = (value,guard,identity)
        self.entries.move_to_end(key)
        while len(self.entries) > self.maximum_objects:
            self.entries.popitem(last=False)
        return identity
