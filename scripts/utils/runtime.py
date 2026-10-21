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

"""Optional keyboard polling, queued execution and immediate console output."""

import sys
from queue import Queue

try:
    import keyboard
except ImportError:
    keyboard = None


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
