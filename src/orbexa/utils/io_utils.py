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

import os
import sys
import json
import yaml
import shutil
import datetime
import time
import logging
from pathlib import Path
from typing import Optional, Sequence, Union, Any, Tuple
from queue import Queue
import numpy as np

logger = logging.getLogger(__name__)

# Try to import keyboard safely
try:
    import keyboard
except ImportError:
    keyboard = None


def load_config(path="config/default.yaml"):
    """
    Load a YAML configuration file.

    Args:
        path (str, optional): Path to the YAML configuration file. Defaults to "config/default.yaml".

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(Path(path), "r") as file:
        return yaml.safe_load(file)


def is_key_pressed(key: str) -> bool:
    """
    Check if a specific keyboard key is currently pressed.
    Safe to use even if keyboard module is missing or headless.

    Args:
        key (str): Key name (e.g., 'q', 'space').

    Returns:
        bool: True if the key is pressed, False otherwise.
    """
    if keyboard is None:
        return False
    try:
        return keyboard.is_pressed(key)
    except Exception:
        return False


def thread_worker(result_queue: Queue, func, *args, **kwargs):
    """
    Run a function in a thread and store the result in a queue.
    Catches exceptions and returns (success_bool, result_or_error).

    Args:
        result_queue (Queue): A queue to store the result.
        func (callable): Function to execute.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.
    """
    try:
        result = func(*args, **kwargs)
        result_queue.put((True, result))
    except Exception as e:
        result_queue.put((False, e))


def stream_printer(text):
    """
    Write text to stdout immediately.

    Args:
        text (str): Text to print.
    """
    sys.stdout.write(text)
    sys.stdout.flush()


def get_next_test_folder(base_dir="./results/tests") -> Path:
    """
    Get the next available test folder path.

    Args:
        base_dir (str): Base directory for test folders.

    Returns:
        Path: Path to the next test folder.
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    existing = [d.name for d in base.iterdir() if d.is_dir() and d.name.startswith("t")]
    numbers = sorted(int(d[1:]) for d in existing if d[1:].isdigit())
    next_id = (numbers[-1] + 1) if numbers else 1
    return base / f"t{next_id:02d}"


def save_test_result(
    properties: dict, files_to_copy=None, base_dir="./results/tests"
) -> str:
    """
    Creates a new test result folder, saves the properties.yaml, and copies optional files.

    Args:
        properties (dict): Metadata to save as properties.yaml
        files_to_copy (list of tuples): Each tuple is (src_path, dst_filename)
        base_dir (str): Path to the results/tests directory
    """
    target_folder = get_next_test_folder(base_dir)
    target_folder.mkdir(parents=True)

    # Save properties.yaml
    with open(target_folder / "properties.yaml", "w") as f:
        yaml.dump(properties, f, default_flow_style=False)

    # Copy additional files
    if files_to_copy:
        for src_path, dst_name in files_to_copy:
            try:
                shutil.copy(src_path, target_folder / dst_name)
            except Exception as e:
                logger.warning(f"Failed to copy {src_path}: {e}")

    logger.info(f"[✓] Saved test result to: {target_folder}")
    return str(target_folder)


def save_data(filename: Union[str, Path], data: Any) -> bool:
    """
    Save data to a JSON file. Handles conversion and path creation.

    Args:
        filename (str | Path): File path.
        data (Any): Data to save.

    Returns:
        bool: True if success, False otherwise.
    """
    try:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types if needed
        safe_data = convert_numpy_to_list(data)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(safe_data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"[io_utils] Error saving {filename}: {e}")
        return False


def load_data(filename: Union[str, Path]) -> Any:
    """
    Load data from a JSON file.

    Args:
        filename (str | Path): File path.

    Returns:
        Any: Parsed data.
    """
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def latest_data_file(
    folder: Union[str, Path], suffixes: Optional[Sequence[str]] = None
) -> str:
    """
    Get the most recent file in a folder (by modification time).

    Args:
        folder (str | Path): Path to folder.
        suffixes (Sequence[str], optional): Valid suffixes (e.g. ['.json']).

    Returns:
        str: Path to latest file.
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    candidates = [p for p in folder.iterdir() if p.is_file()]
    if suffixes:
        cands_filtered = [p for p in candidates if p.suffix in suffixes]
        # Fallback if no specific suffix found?
        # User requested filtering. If empty, maybe fall back to all?
        # But if user asks for .json and only .txt exists, returning .txt is wrong.
        candidates = cands_filtered

    if not candidates:
        raise FileNotFoundError(f"No files found in: {folder}")

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest)


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


def create_filename(folder: Union[str, Path], suffix: str) -> str:
    """
    Generate a timestamped filename.

    Args:
        folder (str | Path): Folder path.
        suffix (str): File extension (e.g., '.json').

    Returns:
        str: Full filename with timestamp.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(folder / f"{ts}{suffix}")
