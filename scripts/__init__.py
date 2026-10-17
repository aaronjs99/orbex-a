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

"""
ORBEX-A: Orbital Rendezvous and Capture Experiment with Adaptive Tube MPC

A Python package for spacecraft rendezvous simulations using Model Predictive
Control with configurable solver backends (GEKKO, CasADi, SciPy).

Package Structure:
    - orbexa.core: Structured configuration, dynamics and spacecraft models
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

__version__ = "2.3.0"
__author__ = "Aaron John Sabu"
__email__ = "aaronjs@ucla.edu"

# =============================================================================
# Solver Module Exports (always available, no circular deps)
# =============================================================================
_SOLVER_EXPORTS = {
    "get_solver", "get_solver_from_config", "register_solver", "SolverBase",
    "SolverResult", "MPCProblem", "GekkoSolver", "ScipySolver", "SOLVER_REGISTRY",
}
_CONTRACT_EXPORTS = {"MPCProblem", "SolverResult"}
_MODEL_EXPORTS = {
    "SimulationConfig": ".core.config",
    "ADTMPCMissionRunner": ".simulation.adtmpc_mission",
    "ADTMPCMissionResult": ".simulation.mission_result",
}


# =============================================================================
# Package-Level Utilities (lazy loaded to avoid circular deps)
# =============================================================================
def load_config(path=None):
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
    from importlib import import_module

    if name in _CONTRACT_EXPORTS:
        value = getattr(import_module(".contracts", __name__), name)
        globals()[name] = value
        return value
    if name in _MODEL_EXPORTS:
        value = getattr(import_module(_MODEL_EXPORTS[name], __name__), name)
        globals()[name] = value
        return value
    if name in _SOLVER_EXPORTS:
        value = getattr(import_module(".solvers", __name__), name)
        globals()[name] = value
        return value
    if name in ("core", "control", "solvers", "estimation", "planning", "simulation", "visualization", "utils", "layout"):
        module = import_module("." + name, __name__)
        globals()[name] = module
        return module
    if name == "dynamics":
        from orbexa.core import dynamics

        return dynamics
    elif name == "spacecraft":
        from orbexa.core import spacecraft

        return spacecraft
    elif name in ("Spacecraft", "Target", "Chaser"):
        return getattr(import_module(".core.spacecraft", __name__), name)
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
    "SimulationConfig",
    "ADTMPCMissionRunner",
    "ADTMPCMissionResult",
    "Spacecraft",
    "Target",
    "Chaser",
    # Utilities
    "load_config",
    "discretize",
    # Submodules (lazy loaded)
    "core",
    "control",
    "solvers",
    "estimation",
    "planning",
    "simulation",
    "visualization",
    "utils",
    "layout",
]
