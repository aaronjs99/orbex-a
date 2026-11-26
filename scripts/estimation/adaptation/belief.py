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

"""belief.py: Controller feasible sets and point estimates with reproducible priors.

Copyright: Aaron John Sabu
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from orbexa.models.dynamics.parameters import (
    ADTMPC_PARAMETER_KEYS,
    OrbitalPlantTruth,
    specific_angular_momentum,
)


def serial_ranges(ranges: dict[str, tuple[float, float]]) -> dict[str, list[float]]:
    """Return each (lower, upper) range as a list of two floats."""
    return {key: [float(value[0]), float(value[1])] for key, value in ranges.items()}


def serial_estimates(estimates: dict[str, float]) -> dict[str, float]:
    """Return each estimate converted to float."""
    return {key: float(value) for key, value in estimates.items()}


@dataclass
class ParameterBelief:
    """Controller-side feasible sets and point estimates used by MPC/SMID."""

    feasible_sets: dict[str, tuple[float, float]]
    estimates: dict[str, float]

    def __post_init__(self) -> None:
        """Validate the belief and normalize its values to floats in place.

        Outputs:
            Replaces each feasible set with a float tuple and each estimate with a
            float. Raises ValueError when either mapping lacks exactly the
            ADTMPC parameter keys, an interval is not a finite pair with 0 <= lower <=
            upper, a point estimate is non-finite or outside its interval, or the
            eccentricity upper bound is 1 or more.
        """
        if set(self.feasible_sets) != set(ADTMPC_PARAMETER_KEYS) or set(
            self.estimates
        ) != set(ADTMPC_PARAMETER_KEYS):
            raise ValueError("Parameter belief requires eccentricity, alpha and beta")
        for key in ADTMPC_PARAMETER_KEYS:
            interval = np.asarray(self.feasible_sets[key], dtype=float)
            point = float(self.estimates[key])
            if (
                interval.shape != (2,)
                or not np.all(np.isfinite(interval))
                or interval[0] < 0
                or interval[0] > interval[1]
                or not np.isfinite(point)
                or not interval[0] <= point <= interval[1]
            ):
                raise ValueError(f"Invalid {key} belief interval or point estimate")
            if key == "eccentricity" and interval[1] >= 1:
                raise ValueError("Eccentricity belief must belong to [0, 1)")
            self.feasible_sets[key] = (float(interval[0]), float(interval[1]))
            self.estimates[key] = point

    def copy(self) -> ParameterBelief:
        """Return a new ParameterBelief holding copies of the sets and estimates."""
        return ParameterBelief(
            feasible_sets={
                key: (float(value[0]), float(value[1]))
                for key, value in self.feasible_sets.items()
            },
            estimates=dict(self.estimates),
        )

    def dynamics_params(self, truth_context: OrbitalPlantTruth) -> dict[str, float]:
        """Return the dynamics parameter dict built from the estimated parameters.

        Mean motion, mu, semi-major axis and periapsis time come from truth_context.
        The specific angular momentum is computed from mu, the semi-major axis and
        the estimated eccentricity.
        """
        return {
            "mean_motion": truth_context.mean_motion,
            "eccentricity": float(self.estimates["eccentricity"]),
            "alpha": float(self.estimates["alpha"]),
            "beta": float(self.estimates["beta"]),
            "mu": truth_context.mu,
            "semi_major_axis": truth_context.semi_major_axis,
            "specific_angular_momentum": specific_angular_momentum(
                truth_context.mu,
                truth_context.semi_major_axis,
                float(self.estimates["eccentricity"]),
            ),
            "time_periapsis": truth_context.time_periapsis,
        }

    def as_metadata(self) -> dict[str, dict[str, Any]]:
        """Return the feasible sets and estimates as plain-float dicts."""
        return {
            "feasible_sets": serial_ranges(self.feasible_sets),
            "estimates": serial_estimates(self.estimates),
        }
