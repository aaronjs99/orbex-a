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

"""parameters.py: Fixed plant parameters, independent of mission orchestration and estimators.

Copyright: Aaron John Sabu
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from orbexa.models.configuration.model import SimulationConfig

ADTMPC_PARAMETER_KEYS = ("eccentricity", "alpha", "beta")


def specific_angular_momentum(
    mu: float, semi_major_axis: float, eccentricity: float
) -> float:
    """Return sqrt(mu * semi_major_axis * (1 - eccentricity^2)) as a float."""
    return float(np.sqrt(mu * semi_major_axis * (1.0 - eccentricity**2)))


@dataclass(frozen=True)
class OrbitalPlantTruth:
    """Fixed orbital/density parameters used only by truth propagation."""

    eccentricity: float
    alpha: float
    beta: float
    mu: float
    semi_major_axis: float
    mean_motion: float
    time_periapsis: float

    def __post_init__(self) -> None:
        """Validate the plant parameters.

        Raises ValueError if eccentricity is not finite and in [0, 1), alpha or beta
        is negative or not finite, mu, semi_major_axis or mean_motion is not finite
        and positive, or time_periapsis is not finite.
        """
        if not np.isfinite(self.eccentricity) or not 0 <= self.eccentricity < 1:
            raise ValueError("Plant eccentricity must belong to [0, 1)")
        for name in ("alpha", "beta"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"Plant {name} must be finite and nonnegative")
        for name in ("mu", "semi_major_axis", "mean_motion"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"Plant {name} must be finite and positive")
        if not np.isfinite(self.time_periapsis):
            raise ValueError("Plant time_periapsis must be finite")

    @property
    def specific_angular_momentum(self) -> float:
        """Return the specific angular momentum from mu, semi_major_axis and e."""
        return specific_angular_momentum(
            self.mu, self.semi_major_axis, self.eccentricity
        )

    def dynamics_params(self) -> dict[str, float]:
        """Return the plant values and specific angular momentum as a dict."""
        return {
            "mean_motion": self.mean_motion,
            "eccentricity": self.eccentricity,
            "alpha": self.alpha,
            "beta": self.beta,
            "mu": self.mu,
            "semi_major_axis": self.semi_major_axis,
            "specific_angular_momentum": self.specific_angular_momentum,
            "time_periapsis": self.time_periapsis,
        }

    def as_dict(self) -> dict[str, float]:
        """Return the plant values and specific angular momentum as floats in a dict."""
        return {
            "eccentricity": float(self.eccentricity),
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "mu": float(self.mu),
            "semi_major_axis": float(self.semi_major_axis),
            "mean_motion": float(self.mean_motion),
            "time_periapsis": float(self.time_periapsis),
            "specific_angular_momentum": float(self.specific_angular_momentum),
        }

    @classmethod
    def from_config(cls, config: SimulationConfig) -> OrbitalPlantTruth:
        """Build the plant from the orbit section of a simulation configuration.

        Uses orbit.target_drag as alpha and orbit.chaser_drag as beta.
        """
        return cls(
            eccentricity=float(config.orbit.eccentricity),
            alpha=float(config.orbit.target_drag),
            beta=float(config.orbit.chaser_drag),
            mu=float(config.orbit.mu),
            semi_major_axis=float(config.orbit.semi_major_axis),
            mean_motion=float(config.orbit.mean_motion),
            time_periapsis=float(config.orbit.time_periapsis),
        )
