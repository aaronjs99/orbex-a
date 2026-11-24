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

"""execution.py: Dense sampled ancillary feedback on one shared fleet clock.

Copyright: Aaron John Sabu
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


@dataclass
class ExecutionStep:
    start_anomaly: float
    end_anomaly: float
    success: bool = True
    messages: list[str] = field(default_factory=list)
    reserve_ratios: dict[str, float] = field(default_factory=dict)
    applied: dict[str, list[np.ndarray]] = field(default_factory=dict)
    ancillary: dict[str, list[np.ndarray]] = field(default_factory=dict)
    containment: dict[str, float] = field(default_factory=dict)
    clearance: dict[str, float] = field(default_factory=dict)
    min_spacing: float = np.inf
    sampled_tube_contained: bool = True
    max_nominal_discretization_residual: float = 0.0
    nominal_center_defect: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def committed(self) -> bool:
        return self.end_anomaly != self.start_anomaly
