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
Abstract base classes and data structures for MPC solvers.
"""

import logging
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Callable
from orbexa.contracts import MPCProblem, SolverResult

logger = logging.getLogger(__name__)


class SolverBase(ABC):
    """
    Abstract base class for MPC solvers.

    All solver backends must inherit from this class and implement
    the setup() and solve() methods.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize solver with configuration.

        Args:
            config: Solver-specific configuration dictionary
        """
        self.config = config or {}
        self._problem: Optional[MPCProblem] = None
        self._is_setup = False

    @property
    def name(self) -> str:
        """Solver backend name."""
        return self.__class__.__name__

    @abstractmethod
    def setup(self, problem: MPCProblem) -> None:
        """
        Set up the optimization problem.

        Args:
            problem: MPC problem definition
        """
        pass

    @abstractmethod
    def solve(self) -> SolverResult:
        """
        Solve the optimization problem.

        Returns:
            SolverResult with trajectory and status
        """
        pass

    def cleanup(self) -> None:
        """Clean up solver resources. Override if needed."""
        self._is_setup = False

    def solve_problem(self, problem: MPCProblem) -> SolverResult:
        """
        Convenience method to setup and solve in one call.

        Args:
            problem: MPC problem definition

        Returns:
            SolverResult with trajectory and status
        """
        logger.debug(f"Solving MPC problem with {self.name} solver")
        self.setup(problem)
        result = self.solve()
        self.cleanup()
        if result.success:
            logger.debug(f"{self.name} solver succeeded in {result.solve_time:.4f}s")
        else:
            logger.warning(f"{self.name} solver failed: {result.message}")
        return result

    def _discretize(
        self, A: np.ndarray, B: np.ndarray, anom_step: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Discretize continuous-time system using matrix exponential.

        Args:
            A: Continuous state matrix
            B: Continuous input matrix
            anom_step: Independent variable step (time or anomaly)

        Returns:
            Tuple of (A_d, B_d) discrete matrices
        """
        from scipy import linalg

        n = A.shape[0]
        m = B.shape[1]

        # Build augmented matrix for discretization
        em_upper = np.hstack([A, B]) * anom_step
        em_lower = np.zeros((m, n + m))
        em = np.vstack([em_upper, em_lower])

        # Matrix exponential
        ms = linalg.expm(em)
        A_d = ms[:n, :n]
        B_d = ms[:n, n:]

        return A_d, B_d


class SolverError(Exception):
    """Exception raised by solvers."""

    pass
