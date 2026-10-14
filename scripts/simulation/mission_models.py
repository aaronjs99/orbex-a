# /***********************************************************
# *                                                         *
# * Copyright (c) 2026                                      *
# *                                                         *
# * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
# * University of California, Los Angeles                   *
# *                                                         *
# * Authors: Aaron John Sabu, Brett T. Lopez                *
# * Contact: {aaronjs, btlopez}@ucla.edu                    *
# *                                                         *
# ***********************************************************/

"""Plain chaser assignments and phase records shared by runtime services."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import numpy as np


@dataclass(frozen=True)
class ChaserConfig:
    """One chaser's initial condition and assigned docking point."""

    chaser_id: str
    initial_state: np.ndarray
    docking_point_body: np.ndarray
    docking_normal_body: np.ndarray
    docking_azimuth: float
    docking_surface: str = "cylinder_side"
    docking_standoff: float = 0.0
    docking_candidate_index: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.chaser_id, str) or not self.chaser_id:
            raise ValueError("chaser_id must be a nonempty string")
        for name, size in (("initial_state", 6), ("docking_point_body", 3), ("docking_normal_body", 3)):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (size,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite vector of length {size}")
            object.__setattr__(self, name, value.copy())
        if not np.isfinite(self.docking_azimuth):
            raise ValueError("docking_azimuth must be finite")
        if not np.isfinite(self.docking_standoff) or self.docking_standoff < 0:
            raise ValueError("docking_standoff must be finite and nonnegative")

    def assignment_metadata(self) -> Dict[str, Any]:
        return {
            "chaser_id": self.chaser_id,
            "initial_state": self.initial_state.tolist(),
            "docking_point_body": self.docking_point_body.tolist(),
            "docking_normal_body": self.docking_normal_body.tolist(),
            "docking_azimuth": float(self.docking_azimuth),
            "docking_surface": self.docking_surface,
            "docking_standoff": float(self.docking_standoff),
            "docking_candidate_index": int(self.docking_candidate_index),
        }


@dataclass(frozen=True)
class MissionPhase:
    """Controller phase metadata for one MPC update."""

    name: str
    operation: str
