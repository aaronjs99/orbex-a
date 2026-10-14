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

logger = logging.getLogger(__name__)


@dataclass
class MPCProblem:
    """
    Defines an MPC optimization problem.

    Attributes:
        dynamics_matrix: State transition matrix (n x n) or callable A(t)
        input_matrix: Input matrix (n x m)
        state_cost_matrix: State cost matrix (n x n)
        input_cost_matrix: Input cost matrix (m x m)
        initial_state: Initial state (n,)
        final_state: Final/reference state (n,)
        num_steps: Number of state samples, including the initial state
        anom_step: True anomaly step size (independent variable)
        state_bounds: Optional state constraints
        input_bounds: Optional input constraints
        dynamics_type: "continuous" or "discrete"
        terminal_cost_matrix: Optional quadratic penalty on the final state error
        state_reference: Optional state preview (n x T), including the initial node
        input_reference: Optional interval input preview (m x (T-1))
    """

    dynamics_matrix: np.ndarray
    input_matrix: np.ndarray
    state_cost_matrix: np.ndarray
    input_cost_matrix: np.ndarray
    initial_state: np.ndarray
    final_state: np.ndarray
    num_steps: int
    anom_step: float
    state_bounds: Optional[List[Dict[str, float]]] = None
    input_bounds: Optional[List[Dict[str, float]]] = None
    dynamics_type: str = "continuous"
    extra_params: Dict[str, Any] = field(default_factory=dict)
    terminal_cost_matrix: Optional[np.ndarray] = None
    state_reference: Optional[np.ndarray] = None
    input_reference: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.num_steps < 2:
            raise ValueError("MPC requires at least two state samples and one control interval")
        if self.terminal_cost_matrix is not None:
            self.terminal_cost_matrix = np.asarray(self.terminal_cost_matrix, dtype=float)
            if self.terminal_cost_matrix.shape != (self.num_states, self.num_states):
                raise ValueError("Terminal cost must have one row and column per state")
            if not np.all(np.isfinite(self.terminal_cost_matrix)):
                raise ValueError("Terminal cost must be finite")
        for name, shape in (
            ("state_reference", (self.num_states, self.num_steps)),
            ("input_reference", (self.num_inputs, self.num_steps - 1)),
        ):
            reference = getattr(self, name)
            if reference is not None:
                reference = np.asarray(reference, dtype=float)
                if reference.shape != shape:
                    raise ValueError(f"{name} must have shape {shape}")
                if not np.all(np.isfinite(reference)):
                    raise ValueError(f"{name} must be finite")
                setattr(self, name, reference)

    @property
    def num_states(self) -> int:
        return len(self.initial_state)

    @property
    def num_inputs(self) -> int:
        return self.input_matrix.shape[1]

    @property
    def reference_states(self) -> np.ndarray:
        """State reference at every node, falling back to the constant goal."""
        if self.state_reference is not None:
            return self.state_reference
        return np.repeat(
            np.asarray(self.final_state, dtype=float)[:, None], self.num_steps, axis=1
        )

    @property
    def reference_inputs(self) -> np.ndarray:
        """Input reference for each state transition, falling back to zero."""
        if self.input_reference is not None:
            return self.input_reference
        return np.zeros((self.num_inputs, self.num_steps - 1))


@dataclass
class SolverResult:
    """
    Result from MPC solver.

    Attributes:
        success: Whether optimization succeeded
        state_trajectory: State trajectory (n x T)
        control_trajectory: Interval inputs (m x (T-1)); column k drives
            state_trajectory[:, k] to state_trajectory[:, k+1]
        cost: Optimal cost value
        solve_time: Solver execution time in seconds
        message: Status message or error description
        solver_info: Additional solver-specific info
    """

    success: bool
    state_trajectory: Optional[np.ndarray] = None
    control_trajectory: Optional[np.ndarray] = None
    cost: Optional[float] = None
    solve_time: float = 0.0
    message: str = ""
    solver_info: Dict[str, Any] = field(default_factory=dict)


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
