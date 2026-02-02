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
ORBEX-A: Orbital Rendezvous and Capture Experiment with Adaptive Tube MPC

A Python package for spacecraft rendezvous simulations using Model Predictive
Control with configurable solver backends (GEKKO, CasADi, SciPy).

Package Structure:
    - orbexa.core: Core data structures (params, dynamics, spacecraft)
    - orbexa.control: MPC controllers
    - orbexa.solvers: Modular solver backends
    - orbexa.estimation: State estimation and adaptation
    - orbexa.planning: Mission planning and task allocation
    - orbexa.visualization: Plotting and simulation
    - orbexa.utils: Shared utilities

Example:
    >>> from orbexa.solvers import get_solver
    >>> solver = get_solver("gekko")
"""

__version__ = "2.0.0"
__author__ = "Aaron John Sabu, Brett T. Lopez"
__email__ = "aaronjs@ucla.edu, btlopez@ucla.edu"

# =============================================================================
# Solver Module Exports (always available, no circular deps)
# =============================================================================
from orbexa.solvers import (
    get_solver,
    get_solver_from_config,
    register_solver,
    SolverBase,
    SolverResult,
    MPCProblem,
    GekkoSolver,
    ScipySolver,
    SOLVER_REGISTRY,
)


# =============================================================================
# Package-Level Utilities (lazy loaded to avoid circular deps)
# =============================================================================
def load_config(path="config/default.yaml"):
    """Load configuration from YAML file."""
    from orbexa.utils import load_config as _load_config

    return _load_config(path)


def discretize(A, B, dt):
    """Discretize continuous-time system."""
    from orbexa.utils import discretize as _discretize

    return _discretize(A, B, dt)


# =============================================================================
# Lazy Loading for Heavy Modules (avoid circular deps)
# =============================================================================
def __getattr__(name):
    """Lazy load submodules to avoid circular imports."""
    if name == "core":
        from orbexa import core

        return core
    elif name == "control":
        from orbexa import control

        return control
    elif name == "estimation":
        from orbexa import estimation

        return estimation
    elif name == "planning":
        from orbexa import planning

        return planning
    elif name == "visualization":
        from orbexa import visualization

        return visualization
    elif name == "utils":
        from orbexa import utils

        return utils
    elif name == "params":
        from orbexa.core import params

        return params
    elif name == "dynamics":
        from orbexa.core import dynamics

        return dynamics
    elif name == "spacecraft":
        from orbexa.core import spacecraft

        return spacecraft
    elif name in ("Spacecraft", "Target", "Chaser"):
        from orbexa.core.spacecraft import Spacecraft, Target, Chaser

        return {"Spacecraft": Spacecraft, "Target": Target, "Chaser": Chaser}[name]
    raise AttributeError(f"module 'orbexa' has no attribute '{name}'")


__all__ = [
    # Version
    "__version__",
    "__author__",
    # Solvers
    "get_solver",
    "get_solver_from_config",
    "register_solver",
    "SolverBase",
    "SolverResult",
    "MPCProblem",
    "GekkoSolver",
    "ScipySolver",
    "SOLVER_REGISTRY",
    # Utilities
    "load_config",
    "discretize",
    # Submodules (lazy loaded)
    "core",
    "control",
    "solvers",
    "estimation",
    "planning",
    "visualization",
    "utils",
]
