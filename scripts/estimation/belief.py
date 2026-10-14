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

"""Controller feasible sets and point estimates with reproducible priors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np
from orbexa.core.orbital_parameters import OrbitalPlantTruth, specific_angular_momentum
from orbexa.core.orbital_parameters import ADTMPC_PARAMETER_KEYS

if TYPE_CHECKING:
    from orbexa.core.config import SimulationConfig


def serial_ranges(ranges: Dict[str, Tuple[float, float]]) -> Dict[str, List[float]]:
    return {key: [float(value[0]), float(value[1])] for key, value in ranges.items()}


def serial_estimates(estimates: Dict[str, float]) -> Dict[str, float]:
    return {key: float(value) for key, value in estimates.items()}


@dataclass
class ParameterBelief:
    """Controller-side feasible sets and point estimates used by MPC/SMID."""

    feasible_sets: Dict[str, Tuple[float, float]]
    estimates: Dict[str, float]

    def __post_init__(self) -> None:
        if set(self.feasible_sets) != set(ADTMPC_PARAMETER_KEYS) or set(self.estimates) != set(ADTMPC_PARAMETER_KEYS):
            raise ValueError("Parameter belief requires eccentricity, alpha and beta")
        for key in ADTMPC_PARAMETER_KEYS:
            interval = np.asarray(self.feasible_sets[key], dtype=float)
            point = float(self.estimates[key])
            if (interval.shape != (2,) or not np.all(np.isfinite(interval)) or
                    interval[0] < 0 or interval[0] > interval[1] or
                    not np.isfinite(point) or not interval[0] <= point <= interval[1]):
                raise ValueError(f"Invalid {key} belief interval or point estimate")
            if key == "eccentricity" and interval[1] >= 1:
                raise ValueError("Eccentricity belief must belong to [0, 1)")
            self.feasible_sets[key] = (float(interval[0]), float(interval[1]))
            self.estimates[key] = point

    @classmethod
    def from_config_prior(cls, config: Optional[SimulationConfig] = None) -> "ParameterBelief":
        """Build the initial adaptive belief from broad or config-centered priors."""
        if config is None:
            ranges = {
                "eccentricity": (0.02, 0.38),
                "alpha": (0.0, 5.5e-7),
                "beta": (0.0, 8.55e-7),
            }
            estimates = {key: float(np.mean(value)) for key, value in ranges.items()}
        else:
            eccentricity = float(config.orbit.eccentricity)
            alpha = float(config.orbit.target_drag)
            beta = float(config.orbit.chaser_drag)
            ranges = {
                "eccentricity": (
                    max(0.0, eccentricity - config.adaptation.prior_eccentricity_half_width),
                    min(0.99, eccentricity + config.adaptation.prior_eccentricity_half_width,
                        1.0 - (config.orbit.central_body_radius + 1.0) /
                        config.orbit.semi_major_axis),
                ),
                "alpha": (max(0.0, config.adaptation.prior_drag_factors[0] * alpha), config.adaptation.prior_drag_factors[1] * alpha),
                "beta": (max(0.0, config.adaptation.prior_drag_factors[0] * beta), config.adaptation.prior_drag_factors[1] * beta),
            }
            # Start from a reproducible feasible point rather than the plant
            # truth; SMID/time-series updates are responsible for adaptation.
            rng = np.random.default_rng(int(config.seed))
            estimates = {
                key: float(rng.uniform(low=value[0], high=value[1]))
                for key, value in ranges.items()
            }
        return cls(feasible_sets=ranges, estimates=estimates)

    def copy(self) -> "ParameterBelief":
        return ParameterBelief(
            feasible_sets={
                key: (float(value[0]), float(value[1]))
                for key, value in self.feasible_sets.items()
            },
            estimates=dict(self.estimates),
        )

    def dynamics_params(self, truth_context: OrbitalPlantTruth) -> Dict[str, float]:
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

    def as_metadata(self) -> Dict[str, Dict[str, Any]]:
        return {
            "feasible_sets": serial_ranges(self.feasible_sets),
            "estimates": serial_estimates(self.estimates),
        }
