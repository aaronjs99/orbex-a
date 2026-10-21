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

"""Explicit YAML loading and relocated-checkout configuration fallback."""

from pathlib import Path
import yaml


def load_config(path=None):
    """
    Load a YAML configuration file.

    Args:
        path (str, optional): YAML path, or the relocated checkout's default scenario.

    Returns:
        dict: Parsed configuration dictionary.
    """
    if path is None:
        from orbexa.layout import default_config_path
        path = default_config_path()
    with open(Path(path), "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
