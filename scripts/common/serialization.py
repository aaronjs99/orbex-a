"""Source-qualified numerical continuation of a committed mission record.

This is a numerical restart, not the admissible safe-backup theorem. Actual
states, persistent nominal centers, tube widths/gains, belief, clock, phase and
assignments are retained. Unavailable full control warm starts and dense SMID
measurements are explicitly reset; neither is reconstructed from mean inputs.
No callables or pickle payloads are serialized.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

CHECKPOINT_SCHEMA = "orbexa.mission_checkpoint.v1"


def plain(value):
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
    return hashlib.sha256(
        json.dumps(
            plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
