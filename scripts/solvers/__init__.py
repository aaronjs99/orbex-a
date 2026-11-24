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

"""solvers/__init__.py: ORBEX-A Solvers Module

Modular MPC solver backends with unified interface.
Supports GEKKO, CasADi, and SciPy.

Copyright: Aaron John Sabu
"""

from orbexa.solvers.base import MPCProblem, SolverBase, SolverResult
from orbexa.solvers.gekko.solver import GekkoSolver
from orbexa.solvers.scipy_solver import ScipySolver

# Solver registry
SOLVER_REGISTRY = {
    "gekko": GekkoSolver,
    "scipy": ScipySolver,
}

# Try to import CasADi solver (optional dependency)
try:
    from orbexa.solvers.casadi_solver import CasadiSolver

    SOLVER_REGISTRY["casadi"] = CasadiSolver
    CASADI_AVAILABLE = True
except ImportError:
    CASADI_AVAILABLE = False
    CasadiSolver = None


def get_solver(backend: str = "gekko", config: dict = None) -> SolverBase:
    """
    Get a solver instance by backend name.

    Args:
        backend: Solver backend name ("gekko", "casadi", "scipy")
        config: Solver-specific configuration dictionary

    Returns:
        Configured solver instance

    Raises:
        ValueError: If backend is not available
    """
    if backend not in SOLVER_REGISTRY:
        available = list(SOLVER_REGISTRY.keys())
        raise ValueError(f"Unknown solver: {backend}. Available: {available}")
    return SOLVER_REGISTRY[backend](config or {})


def get_solver_from_config(config_path=None) -> SolverBase:
    """
    Get a solver configured from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configured solver instance
    """
    from orbexa.common.io.configuration import load_config

    config = load_config(config_path)
    solver_config = config.get("solver", {})
    if not isinstance(solver_config, dict):
        raise ValueError("solver configuration must be a mapping")
    backend = solver_config.get("backend", "gekko")
    backend_config = solver_config.get(backend)
    if backend_config is None:
        backend_config = {
            key: value for key, value in solver_config.items() if key != "backend"
        }
    if not isinstance(backend_config, dict):
        raise ValueError(f"solver.{backend} options must be a mapping")
    return get_solver(backend, backend_config)


def register_solver(name: str, solver_class: type) -> None:
    """
    Register a custom solver backend.

    Args:
        name: Name for the solver
        solver_class: Solver class (must inherit from SolverBase)
    """
    if not issubclass(solver_class, SolverBase):
        raise TypeError("solver_class must inherit from SolverBase")
    SOLVER_REGISTRY[name] = solver_class


__all__ = [
    "CASADI_AVAILABLE",
    "SOLVER_REGISTRY",
    "CasadiSolver",
    "GekkoSolver",
    "MPCProblem",
    "ScipySolver",
    "SolverBase",
    "SolverResult",
    "get_solver",
    "get_solver_from_config",
    "register_solver",
]
