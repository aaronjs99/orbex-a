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

"""Typed coupled-tube data at the solver boundary.

Conversion preserves the existing ``MPCProblem.extra_params['dynamic_tube']``
mapping, so backend expressions and solver assembly remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Mapping, Tuple
import numpy as np


@dataclass
class DynamicTubeSpec:
    phi_initial: np.ndarray
    omega_initial: np.ndarray
    lambda_gain: np.ndarray
    alpha_initial: np.ndarray
    alpha_bounds: Tuple[float, float]
    alpha_rate_limit: float
    delta_A: np.ndarray
    delta_d: np.ndarray
    A_abs: np.ndarray
    d_abs: np.ndarray
    eta: np.ndarray
    bandwidth_weight: float
    tube_weight: float
    terminal_position_tolerance: float
    terminal_velocity_tolerance: float
    enforce_terminal: bool = False

    def __post_init__(self) -> None:
        delta_a = np.asarray(self.delta_A, dtype=float)
        if delta_a.ndim != 3 or delta_a.shape[1:] != (3, 6) or len(delta_a) < 1:
            raise ValueError("dynamic_tube.delta_A must have shape (intervals, 3, 6) with at least one interval")
        intervals = len(delta_a)
        shapes = {
            "phi_initial": (3,), "omega_initial": (3,), "lambda_gain": (3,),
            "alpha_initial": (3,), "eta": (3,), "delta_A": (intervals, 3, 6),
            "delta_d": (intervals, 3), "A_abs": (intervals, 3, 6), "d_abs": (intervals, 3),
        }
        for name, shape in shapes.items():
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != shape or not np.all(np.isfinite(value)) or np.any(value < 0):
                raise ValueError(f"dynamic_tube.{name} must be finite, nonnegative and have shape {shape}")
            setattr(self, name, value.copy())
        if np.any(self.lambda_gain <= 0):
            raise ValueError("dynamic_tube.lambda_gain must be positive")
        bounds = np.asarray(self.alpha_bounds, dtype=float)
        if (bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or
                not 0 < bounds[0] <= bounds[1]):
            raise ValueError("dynamic_tube.alpha_bounds must be finite, ordered and positive")
        self.alpha_bounds = (float(bounds[0]), float(bounds[1]))
        if np.any(self.alpha_initial < bounds[0]) or np.any(self.alpha_initial > bounds[1]):
            raise ValueError("dynamic_tube.alpha_initial must belong to alpha_bounds")
        for name in ("alpha_rate_limit", "bandwidth_weight", "tube_weight"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"dynamic_tube.{name} must be finite and nonnegative")
        for name in ("terminal_position_tolerance", "terminal_velocity_tolerance"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"dynamic_tube.{name} must be finite and positive")
        if not isinstance(self.enforce_terminal, bool):
            raise ValueError("dynamic_tube.enforce_terminal must be a boolean")

    @property
    def horizon_steps(self) -> int:
        return len(self.delta_A) + 1

    def to_extra_params(self) -> Dict[str, Any]:
        """Return the current backend's dynamic-tube mapping without reformulation."""
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @classmethod
    def from_extra_params(cls, parameters: Mapping[str, Any]) -> "DynamicTubeSpec":
        return cls(**dict(parameters))
