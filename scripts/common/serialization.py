"""serialization.py: Plain JSON conversion and canonical SHA-256 digests of records.

Copyright: Aaron John Sabu
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np


def plain(value):
    """Convert a value to JSON-compatible builtins.

    Dataclasses go through ``asdict``, arrays through ``tolist``, NumPy scalars
    through ``item``, dictionary keys become strings, tuples and lists become
    lists and paths become strings. Any other value is returned unchanged.
    """
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def digest(value):
    """Return the SHA-256 hex digest of the canonical JSON form of ``plain(value)``.

    The JSON uses sorted keys and compact separators. Raises ValueError for a
    non-finite float.
    """
    return hashlib.sha256(
        json.dumps(
            plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
