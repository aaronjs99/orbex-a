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

"""
Control Mode Configurations

Defines the available control modes for spacecraft rendezvous simulations.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ControlModeConfig:
    """Configuration for a control mode."""

    name: str
    description: str
    tube_mpc_enabled: bool
    adaptive_enabled: bool
    num_mpc_steps: int
    num_act_steps: int


# Control mode definitions
CONTROL_MODES: Dict[str, ControlModeConfig] = {
    "oc": ControlModeConfig(
        name="Optimal Control",
        description="Single-shot trajectory optimization (full horizon)",
        tube_mpc_enabled=False,
        adaptive_enabled=False,
        num_mpc_steps=200,
        num_act_steps=200,
    ),
    "mpc": ControlModeConfig(
        name="Model Predictive Control",
        description="Receding horizon MPC without tube constraints",
        tube_mpc_enabled=False,
        adaptive_enabled=False,
        num_mpc_steps=80,
        num_act_steps=20,
    ),
    "tube": ControlModeConfig(
        name="Dynamic Tube MPC",
        description="Tube MPC with robustness to bounded disturbances",
        tube_mpc_enabled=True,
        adaptive_enabled=False,
        num_mpc_steps=80,
        num_act_steps=20,
    ),
    "adtmpc": ControlModeConfig(
        name="Adaptive Dynamic Tube MPC",
        description="Tube MPC with online adaptation of uncertainty bounds",
        tube_mpc_enabled=True,
        adaptive_enabled=True,
        num_mpc_steps=80,
        num_act_steps=20,
    ),
}


def get_mode_config(mode: str) -> ControlModeConfig:
    """Get configuration for a control mode."""
    if mode not in CONTROL_MODES:
        available = list(CONTROL_MODES.keys())
        raise ValueError(f"Unknown mode '{mode}'. Available: {available}")
    return CONTROL_MODES[mode]


def list_modes() -> Dict[str, str]:
    """Return dict of mode names to descriptions."""
    return {k: v.description for k, v in CONTROL_MODES.items()}
