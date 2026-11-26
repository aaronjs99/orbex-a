"""artifacts.py: Atomic worker records and streamed source-file identities.

Copyright: Aaron John Sabu
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _write_json(path, value):
    """Write value as strict JSON to path through a temporary file.

    Inputs:
        path: destination file; ``<name>.tmp`` beside it is written first.
        value: JSON-serializable object; NaN and infinities are refused.
    Outputs:
        Replaces path with the complete record. Raises ValueError for non-finite
        floats and TypeError for unserializable values.
    """
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _sha(path):
    """Return the SHA-256 hex digest of the file at path, read in 1 MiB blocks."""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
