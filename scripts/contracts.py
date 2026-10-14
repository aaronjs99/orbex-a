"""Backend-neutral optimization records and maintained mission identities."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

PHYSICAL_MODEL_ID = "physical-lvlh-v1"
COUPLED_CONTROLLER_ID = "coupled-adaptive-dynamic-tube"
MISSION_RESULT_SCHEMA = "orbexa.adtmpc_mission.v1"

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
