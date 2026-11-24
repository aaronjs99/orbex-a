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

"""Simulation configuration composition and stable public model exports."""

from dataclasses import dataclass, field

import numpy as np
from orbexa.models.configuration.types import (
    AdaptationConfig,
    MissionPolicyConfig,
    MPCConfig,
    OrbitConfig,
    SolverConfig,
    TargetConfig,
    TubeMPCConfig,
    _positive_integer,
)


@dataclass
class SimulationConfig:
    """Main simulation configuration."""

    seed: int
    anom_step: float
    total_anom: float
    num_update_steps: int
    decoupled_mode: bool

    orbit: OrbitConfig
    mpc: MPCConfig
    target: TargetConfig
    tube: TubeMPCConfig

    num_chasers: int
    mission: MissionPolicyConfig = field(default_factory=MissionPolicyConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)

    def __post_init__(self) -> None:
        _positive_integer(self.seed, "seed", 0)
        _positive_integer(self.num_chasers, "num_chasers")
        _positive_integer(self.num_update_steps, "num_update_steps")
        if not isinstance(self.decoupled_mode, bool):
            raise ValueError("decoupled_mode must be a boolean")
        if any(
            not np.isfinite(value) or value <= 0
            for value in (self.anom_step, self.total_anom)
        ):
            raise ValueError(
                "Anomaly step and total anomaly must be finite and positive"
            )

    @property
    def iter_anomaly(self) -> float:
        return self.num_update_steps * self.anom_step

    def dynamics_params(self) -> dict[str, float]:
        params = self.orbit.dynamics_params()
        params["anom_step"] = self.anom_step
        return params

    @classmethod
    def load(cls, path: str | None = None) -> "SimulationConfig":
        """Decode a scenario through the dedicated YAML loader."""
        from orbexa.models.configuration.loader import load_simulation_config

        return load_simulation_config(path, config_type=cls)
