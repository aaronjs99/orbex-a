# /***********************************************************
# *                                                         *
# * Copyright (c) 2026                                      *
# *                                                         *
# * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
# * University of California, Los Angeles                   *
# *                                                         *
# * Authors: Aaron John Sabu                                *
# * Contact: aaronjs@ucla.edu                               *
# *                                                         *
# ***********************************************************/

"""Atomic JSON persistence, data conversion and result-folder naming."""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("orbexa.utils.io_utils")


def save_data(filename: str | Path, data: Any) -> bool:
    """
    Save data to a JSON file. Handles conversion and path creation.

    Args:
        filename (str | Path): File path.
        data (Any): Data to save.

    Returns:
        bool: True if success, False otherwise.
    """
    temporary = None
    try:
        path = Path(filename)
        # Serialize before touching the destination, so unsupported values cannot truncate it.
        payload = json.dumps(convert_numpy_to_list(data), ensure_ascii=False, indent=2)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        return True
    except Exception as error:
        logger.error("[io_utils] Error saving %s: %s", filename, error)
        return False
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                logger.warning("Could not remove incomplete save %s", temporary)


def load_data(filename: str | Path) -> Any:
    """
    Load data from a JSON file.

    Args:
        filename (str | Path): File path.

    Returns:
        Any: Parsed data.
    """
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def convert_numpy_to_list(obj: Any) -> Any:
    """
    Recursively convert numpy arrays/scalars in data to native Python types.

    Args:
        obj (Any): Data to convert.

    Returns:
        Any: Converted data.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_to_list(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_numpy_to_list(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj
