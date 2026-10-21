"""Structured numeric and policy configuration contracts."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from orbexa.utils.solver_options import DEFAULT_GEKKO_MAX_MEMORY, gekko_memory_exponent, composite_geometry_mode

def _positive_integer(value: Any, name: str, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")


def _finite_vector(value: Any, name: str, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {size} finite values")
    return array


def _bounds_list(bounds: List[Dict[str, Any]], name: str, size: int) -> None:
    if not isinstance(bounds, (list, tuple)) or len(bounds) != size:
        raise ValueError(f"{name} must contain {size} bounds")
    for index, bound in enumerate(bounds):
        if not isinstance(bound, dict):
            raise ValueError(f"{name}[{index}] must be a bound mapping")
        try:
            low = -np.inf if bound.get("lower") is None else float(bound["lower"])
            high = np.inf if bound.get("upper") is None else float(bound["upper"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name}[{index}] must use numeric lower/upper bounds") from error
        if np.isnan(low) or np.isnan(high) or low > high:
            raise ValueError(f"{name}[{index}] has invalid ordered bounds")


def _as_square_weight_matrix(value: Any, size: int) -> np.ndarray:
    """Normalize scalar/vector/matrix YAML weights into a square matrix."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.eye(size) * float(arr)
    if arr.ndim == 1:
        if arr.shape[0] != size:
            raise ValueError(f"Expected {size} weights, got {arr.shape[0]}")
        return np.diag(arr)
    if arr.shape == (size, size):
        return arr
    raise ValueError(f"Expected scalar, length-{size}, or {size}x{size} weight matrix")


@dataclass
class OrbitConfig:
    """Orbital mechanics configuration."""

    mu: float
    """Standard gravitational parameter for Earth (m^3/s^2)."""

    semi_major_axis: float
    """Semi-major axis (meters)."""

    eccentricity: float
    """Orbital eccentricity."""

    time_periapsis: float
    """Time of periapsis passage."""

    initial_conditions: np.ndarray
    """Physical LVLH state [x, y, z, dx/dq, dy/dq, dz/dq]."""

    target_drag: float = 0.0
    """Target quadratic drag constant alpha."""

    chaser_drag: float = 0.0
    """Chaser quadratic drag constant beta."""

    central_body_radius: float = 0.0
    """Reference surface radius in meters; zero disables surface admissibility."""

    def __post_init__(self) -> None:
        if not (np.isfinite(self.mu) and self.mu > 0 and
                np.isfinite(self.semi_major_axis) and self.semi_major_axis > 0 and
                0 <= self.eccentricity < 1):
            raise ValueError("The reference orbit must be a finite bound ellipse")
        if self.central_body_radius < 0 or not np.isfinite(self.central_body_radius):
            raise ValueError("central_body_radius must be finite and nonnegative")
        if self.semi_major_axis * (1 - self.eccentricity) <= self.central_body_radius:
            raise ValueError("Reference-orbit perigee intersects the central body")
        if any(not np.isfinite(value) or value < 0 for value in (self.target_drag, self.chaser_drag)):
            raise ValueError("Drag coefficients must be finite and nonnegative")
        if not np.isfinite(self.time_periapsis):
            raise ValueError("time_periapsis must be finite")
        self.initial_conditions = _finite_vector(self.initial_conditions, "orbit.initial_conditions", 6)

    @property
    def mean_motion(self) -> float:
        """Mean motion (rad/s) derived from mu and semi-major axis."""
        return np.sqrt(self.mu / self.semi_major_axis**3)

    @property
    def specific_angular_momentum(self) -> float:
        """Specific angular momentum for the configured Keplerian orbit."""
        return float(
            np.sqrt(self.mu * self.semi_major_axis * (1.0 - self.eccentricity**2))
        )

    def dynamics_params(self) -> Dict[str, float]:
        """Runtime parameters for the paper's relative dynamics model."""
        return {
            "mean_motion": self.mean_motion,
            "eccentricity": self.eccentricity,
            "time_periapsis": self.time_periapsis,
            "mu": self.mu,
            "semi_major_axis": self.semi_major_axis,
            "specific_angular_momentum": self.specific_angular_momentum,
            "alpha": self.target_drag,
            "beta": self.chaser_drag,
        }


@dataclass
class MPCConfig:
    """Model Predictive Control configuration."""

    horizon_steps: Dict[str, int]
    actuation_steps: Dict[str, int]

    state_bounds: List[Dict[str, Any]]
    input_bounds: List[Dict[str, float]]
    goal_bounds: Optional[List[float]] = None
    """Optional shell radii for standalone Chaser goal allocation, in meters."""
    min_chaser_separation: float = 0.35

    Q: np.ndarray = field(default_factory=lambda: np.eye(6))
    R: np.ndarray = field(default_factory=lambda: np.eye(3) * 1e-4)
    terminal_cost_enabled: bool = True
    reference_preview_enabled: bool = True
    docking_position_tolerance: float = 0.002
    docking_velocity_tolerance: float = 0.08
    terminal_dwell_updates: int = 5

    def __post_init__(self) -> None:
        for phase in ("rendezvous", "docking"):
            _positive_integer(self.horizon_steps[phase], f"mpc.horizon_steps.{phase}", 2)
            _positive_integer(self.actuation_steps[phase], f"mpc.actuation_steps.{phase}")
        _bounds_list(self.state_bounds, "mpc.state_bounds", 6)
        _bounds_list(self.input_bounds, "mpc.input_bounds", 3)
        for name, size, positive in (("Q", 6, False), ("R", 3, True)):
            matrix = _as_square_weight_matrix(getattr(self, name), size)
            if not np.all(np.isfinite(matrix)) or not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0):
                raise ValueError(f"mpc.{name} must be finite and symmetric")
            eigenvalues = np.linalg.eigvalsh(matrix)
            if (positive and np.min(eigenvalues) <= 0) or (not positive and np.min(eigenvalues) < -1e-12):
                raise ValueError(f"mpc.{name} must be {'positive definite' if positive else 'positive semidefinite'}")
            setattr(self, name, matrix)
        if not np.isfinite(self.min_chaser_separation) or self.min_chaser_separation <= 0:
            raise ValueError("mpc.min_chaser_separation must be finite and positive")
        if self.goal_bounds is not None:
            goal = _finite_vector(self.goal_bounds, "mpc.goal_bounds", 2)
            if not 0 <= goal[0] < goal[1]:
                raise ValueError("mpc.goal_bounds must be increasing nonnegative shell radii")
        if not isinstance(self.terminal_cost_enabled, bool):
            raise ValueError("mpc.terminal_cost_enabled must be a boolean")
        if not isinstance(self.reference_preview_enabled, bool):
            raise ValueError("mpc.reference_preview_enabled must be a boolean")
        if (not isinstance(self.terminal_dwell_updates, int) or
                isinstance(self.terminal_dwell_updates, bool) or self.terminal_dwell_updates <= 0):
            raise ValueError("mpc.terminal_dwell_updates must be a positive integer")
        if any(not np.isfinite(value) or value <= 0 for value in
               (self.docking_position_tolerance, self.docking_velocity_tolerance)):
            raise ValueError("Docking tolerances must be finite and positive")

    def bounds(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, float]]]:
        return self.state_bounds, self.input_bounds


@dataclass
class TargetConfig:
    """Target spacecraft configuration."""

    initial_orientation: np.ndarray
    initial_angular_velocity: np.ndarray
    """Prescribed Euler angle derivatives per anomaly, not body rates in rad/s."""
    inertia: np.ndarray
    shape: str
    center: np.ndarray
    limits: Dict[str, float]
    tolerance: float
    docking_standoff: float
    assignment_strategy: str
    stop_time: float
    discretize_dockers: bool

    observation_error: Dict[str, float]

    def __post_init__(self) -> None:
        self.initial_orientation = _finite_vector(self.initial_orientation, "target.initial_orientation", 3)
        self.initial_angular_velocity = _finite_vector(self.initial_angular_velocity, "target.initial_angular_velocity", 3)
        self.center = _finite_vector(self.center, "target.center", 3)
        self.inertia = np.asarray(self.inertia, dtype=float)
        if (self.inertia.shape != (3, 3) or not np.all(np.isfinite(self.inertia)) or
                not np.allclose(self.inertia, self.inertia.T, atol=1e-12, rtol=0) or
                np.min(np.linalg.eigvalsh(self.inertia)) <= 0):
            raise ValueError("target.inertia must be finite, symmetric and positive definite")
        if any(not np.isfinite(value) or value < 0 for value in (self.tolerance, self.docking_standoff)):
            raise ValueError("Target tolerance and standoff must be finite and nonnegative")
        if self.shape == "cylinder" and (not np.isfinite(self.radius) or not np.isfinite(self.half_length) or
                                          self.radius <= 0 or self.half_length <= 0):
            raise ValueError("Cylinder radius and half-length must be finite and positive")

    @property
    def radius(self) -> float:
        return float(self.limits.get("target_radius", self.limits.get("r_T", 0.0)))

    @property
    def height(self) -> float:
        if "target_height" in self.limits:
            return float(self.limits["target_height"])
        if "target_half_length" in self.limits:
            return 2.0 * float(self.limits["target_half_length"])
        if "half_length" in self.limits:
            return 2.0 * float(self.limits["half_length"])
        if "l_T" in self.limits:
            return 2.0 * float(self.limits["l_T"])
        return 0.0

    @property
    def half_length(self) -> float:
        if "target_half_length" in self.limits:
            return float(self.limits["target_half_length"])
        if "half_length" in self.limits:
            return float(self.limits["half_length"])
        if "l_T" in self.limits:
            return float(self.limits["l_T"])
        return self.height / 2.0

    @property
    def bounding_sphere_radius(self) -> float:
        return float(np.sqrt(self.radius**2 + self.half_length**2))

    @property
    def rendezvous_radius(self) -> float:
        return self.bounding_sphere_radius * (1.0 + self.tolerance)

    def collision_params(self, operation: str = "rendezvous") -> Dict[str, Any]:
        """Translate target geometry into solver constraint parameters."""
        radius = self.radius * (1.0 + self.tolerance)
        half_length = self.half_length * (1.0 + self.tolerance)
        return {
            "operation": operation,
            "shape": self.shape,
            "center": self.center,
            "orientation": self.initial_orientation,
            "target_radius": radius,
            "target_height": self.height * (1.0 + self.tolerance),
            "target_half_length": half_length,
            "rendezvous_radius": self.rendezvous_radius,
            "docking_standoff": self.docking_standoff,
            "assignment_strategy": self.assignment_strategy,
            "tube_radius": 0.0,
        }


@dataclass
class TubeMPCConfig:
    """Tube MPC settings."""

    run_tube: bool
    bandwidth_0: List[float]
    boundary_layer_0: List[float]
    sliding_gains: List[float]

    # Adaptation ranges
    bandwidth_range: Dict[str, float]
    feedback_substeps: int = 32
    bandwidth_rate_limit: float = 1500.0
    bandwidth_weight: float = 1.0e-7
    tube_weight: float = 10.0
    numerical_reserve: float = 1.0e-5
    membership_error_bound: float = 1.0e-7
    membership_box_budget: int = 32
    membership_update_period: int = 5

    def __post_init__(self) -> None:
        if not isinstance(self.run_tube, bool):
            raise ValueError("tube.run_tube must be a boolean")
        for name in ("bandwidth_0", "sliding_gains", "boundary_layer_0"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f"tube.{name} must contain three finite values")
            if np.any(value <= 0):
                raise ValueError(f"tube.{name} must be positive")
        low, high = self.bandwidth_range["lower"], self.bandwidth_range["upper"]
        if not np.isfinite(low) or not np.isfinite(high) or not 0 < low <= high:
            raise ValueError("Invalid tube bandwidth range or feedback sampling")
        if np.any(np.asarray(self.bandwidth_0) < low) or np.any(np.asarray(self.bandwidth_0) > high):
            raise ValueError("tube.bandwidth_0 must fit bandwidth_range")
        _positive_integer(self.feedback_substeps, "tube.feedback_substeps")
        if any(not np.isfinite(value) or value < 0 for value in
               (self.numerical_reserve, self.membership_error_bound, self.bandwidth_rate_limit,
                self.bandwidth_weight, self.tube_weight)):
            raise ValueError("Tube numerical reserves, limits and weights must be finite and nonnegative")
        _positive_integer(self.membership_box_budget, "tube.membership_box_budget")
        _positive_integer(self.membership_update_period, "tube.membership_update_period")

    def runtime_params(
        self,
        *,
        enabled: bool,
        eccentricity_range: Tuple[float, float],
        target_drag_range: Tuple[float, float],
        chaser_drag_range: Tuple[float, float],
    ) -> Optional[Dict[str, Any]]:
        if not enabled:
            return None
        return {
            "enabled": True,
            "lambda_gain": self.sliding_gains,
            "alpha": self.bandwidth_0,
            "phi": self.boundary_layer_0,
            "eccentricity_range": eccentricity_range,
            "aRange": target_drag_range,
            "bRange": chaser_drag_range,
            "tighten_inputs": True,
        }


@dataclass
class MissionPolicyConfig:
    """Rendezvous, terminal activation and sampled safety policy."""

    rendezvous_standoff: float = 0.5
    rendezvous_capture_radius: float = 3.5
    rendezvous_position_tolerance: float = 0.15
    rendezvous_velocity_tolerance: float = 5.0
    docking_position_weight_factor: float = 6.0
    docking_velocity_weight_factor: float = 1.5
    target_safety_tolerance: float = 1.0e-7
    target_barrier_buffer: float = 0.04
    target_barrier_clearance_fraction: float = 0.05
    terminal_activation_distance: float = 0.035

    def __post_init__(self) -> None:
        for name in ("rendezvous_standoff", "rendezvous_capture_radius", "rendezvous_position_tolerance",
                     "rendezvous_velocity_tolerance", "docking_position_weight_factor", "docking_velocity_weight_factor",
                     "target_barrier_buffer", "terminal_activation_distance"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"mission.{name} must be finite and positive")
        if not np.isfinite(self.target_safety_tolerance) or self.target_safety_tolerance < 0:
            raise ValueError("mission.target_safety_tolerance must be finite and nonnegative")
        if not np.isfinite(self.target_barrier_clearance_fraction) or not 0 < self.target_barrier_clearance_fraction <= 1:
            raise ValueError("mission.target_barrier_clearance_fraction must be in (0, 1]")

    @property
    def target_barrier_clearance(self) -> float:
        return self.target_barrier_buffer * self.target_barrier_clearance_fraction


@dataclass
class AdaptationConfig:
    """Point estimation and reproducible initial-belief policy."""

    error_bound: float = 0.15
    prediction_error_threshold: float = 0.005
    max_iter: int = 90
    minimum_widths: Dict[str, float] = field(default_factory=lambda: {
        "eccentricity": 0.04, "alpha": 1.0e-7, "beta": 1.5e-7,
    })
    prior_eccentricity_half_width: float = 0.08
    prior_drag_factors: List[float] = field(default_factory=lambda: [0.5, 1.5])
    window_intervals: int = 8

    def __post_init__(self) -> None:
        for name in ("error_bound", "prediction_error_threshold", "prior_eccentricity_half_width"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"adaptation.{name} must be finite and positive")
        if set(self.minimum_widths) != {"eccentricity", "alpha", "beta"} or any(
                not np.isfinite(value) or value < 0 for value in self.minimum_widths.values()):
            raise ValueError("adaptation.minimum_widths requires finite nonnegative eccentricity, alpha and beta widths")
        factors = _finite_vector(self.prior_drag_factors, "adaptation.prior_drag_factors", 2)
        if not 0 <= factors[0] <= 1 <= factors[1]:
            raise ValueError("adaptation.prior_drag_factors must bracket one with nonnegative factors")
        _positive_integer(self.max_iter, "adaptation.max_iter")
        _positive_integer(self.window_intervals, "adaptation.window_intervals")


@dataclass
class SolverConfig:
    """Default coupled GEKKO backend options, distinct from mission policy."""

    remote: bool = False
    disp: bool = False
    max_iter: int = 400
    max_memory: int = DEFAULT_GEKKO_MAX_MEMORY
    """APMonitor base-ten sparsity-storage exponent, in the range 2 through 10."""
    composite_geometry_mode: str = "free"
    """Optional reference corridor fixes component support normals; free is the default."""

    def __post_init__(self) -> None:
        if not isinstance(self.remote, bool) or not isinstance(self.disp, bool):
            raise ValueError("solver.remote and solver.disp must be booleans")
        _positive_integer(self.max_iter, "solver.max_iter")
        gekko_memory_exponent(self.max_memory)
        composite_geometry_mode(self.composite_geometry_mode)

    def options(self) -> Dict[str, Any]:
        return dict(remote=self.remote, disp=self.disp, max_iter=self.max_iter, max_memory=self.max_memory,
                    composite_geometry_mode=self.composite_geometry_mode)
