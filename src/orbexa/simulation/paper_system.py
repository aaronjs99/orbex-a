#!/usr/bin/env python3
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

"""Paper-native ORBEX-A ADTMPC mission execution.

This module is the executable surface for the paper system.  It keeps the
plant truth fixed, lets the controller belief/FSS adapt through SMID, uses the
elliptical quadratic-drag dynamics model, applies dynamic tube tightening, and
supports both single-chaser rendezvous/docking and a coordinated multi-chaser
scenario.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from orbexa.control import (
    CylinderConstraint,
    MPCController,
    ancillary_controller,
    input_tightening_from_profile,
    linearize_cylinder_constraint,
    linearize_rendezvous_constraint,
    propagate_tube_profile,
    rendezvous_margin,
    rotating_body_point_velocity,
    rotating_docking_point,
    target_frame_position,
    tighten_box_bounds,
    tighten_target_params,
)
from orbexa.core.config import SimulationConfig
from orbexa.core.dynamics import orbital_ellp_drag
from orbexa.estimation.adaptor import SMIDAdaptor, SMIDRecord
from orbexa.solvers import SolverResult
from orbexa.utils.io_utils import load_data, save_data


PAPER_PARAMETER_KEYS = ("eccentricity", "alpha", "beta")


def _specific_angular_momentum(mu: float, semi_major_axis: float, eccentricity: float) -> float:
    return float(np.sqrt(mu * semi_major_axis * (1.0 - eccentricity**2)))


def _serial_ranges(ranges: Dict[str, Tuple[float, float]]) -> Dict[str, List[float]]:
    return {key: [float(value[0]), float(value[1])] for key, value in ranges.items()}


def _serial_estimates(estimates: Dict[str, float]) -> Dict[str, float]:
    return {key: float(value) for key, value in estimates.items()}


@dataclass(frozen=True)
class PaperTruth:
    """Fixed plant parameters used only by actual propagation."""

    eccentricity: float
    alpha: float
    beta: float
    mu: float
    semi_major_axis: float
    mean_motion: float
    time_periapsis: float

    @property
    def specific_angular_momentum(self) -> float:
        return _specific_angular_momentum(
            self.mu, self.semi_major_axis, self.eccentricity
        )

    def dynamics_params(self) -> Dict[str, float]:
        return {
            "mean_motion": self.mean_motion,
            "eccentricity": self.eccentricity,
            "alpha": self.alpha,
            "beta": self.beta,
            "mu": self.mu,
            "semi_major_axis": self.semi_major_axis,
            "specific_angular_momentum": self.specific_angular_momentum,
            "time_periapsis": self.time_periapsis,
        }

    def as_dict(self) -> Dict[str, float]:
        return {
            "eccentricity": float(self.eccentricity),
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "mu": float(self.mu),
            "semi_major_axis": float(self.semi_major_axis),
            "mean_motion": float(self.mean_motion),
            "time_periapsis": float(self.time_periapsis),
            "specific_angular_momentum": float(self.specific_angular_momentum),
        }

    @classmethod
    def from_config(cls, config: SimulationConfig) -> "PaperTruth":
        return cls(
            eccentricity=0.18,
            alpha=2.0e-7,
            beta=4.5e-7,
            mu=float(config.orbit.mu),
            semi_major_axis=float(config.orbit.semi_major_axis),
            mean_motion=float(config.orbit.mean_motion),
            time_periapsis=float(config.orbit.time_periapsis),
        )


@dataclass
class ParameterBelief:
    """Controller-side mutable feasible sets and point estimates."""

    feasible_sets: Dict[str, Tuple[float, float]]
    estimates: Dict[str, float]

    @classmethod
    def paper_default(cls, config: Optional[SimulationConfig] = None) -> "ParameterBelief":
        if config is None:
            ranges = {
                "eccentricity": (0.02, 0.38),
                "alpha": (0.0, 5.5e-7),
                "beta": (0.0, 8.55e-7),
            }
        else:
            eccentricity = float(config.orbit.eccentricity)
            alpha = float(config.orbit.target_drag)
            beta = float(config.orbit.chaser_drag)
            ranges = {
                "eccentricity": (
                    max(0.0, eccentricity - 0.08),
                    min(0.99, eccentricity + 0.08),
                ),
                "alpha": (max(0.0, 0.5 * alpha), 1.5 * alpha),
                "beta": (max(0.0, 0.5 * beta), 1.5 * beta),
            }
        estimates = {key: float(np.mean(value)) for key, value in ranges.items()}
        if config is not None:
            estimates.update(
                {
                    "eccentricity": float(config.orbit.eccentricity),
                    "alpha": float(config.orbit.target_drag),
                    "beta": float(config.orbit.chaser_drag),
                }
            )
        return cls(feasible_sets=ranges, estimates=estimates)

    def copy(self) -> "ParameterBelief":
        return ParameterBelief(
            feasible_sets={
                key: (float(value[0]), float(value[1]))
                for key, value in self.feasible_sets.items()
            },
            estimates=dict(self.estimates),
        )

    def dynamics_params(self, truth_context: PaperTruth) -> Dict[str, float]:
        return {
            "mean_motion": truth_context.mean_motion,
            "eccentricity": float(self.estimates["eccentricity"]),
            "alpha": float(self.estimates["alpha"]),
            "beta": float(self.estimates["beta"]),
            "mu": truth_context.mu,
            "semi_major_axis": truth_context.semi_major_axis,
            "specific_angular_momentum": _specific_angular_momentum(
                truth_context.mu,
                truth_context.semi_major_axis,
                float(self.estimates["eccentricity"]),
            ),
            "time_periapsis": truth_context.time_periapsis,
        }

    def as_metadata(self) -> Dict[str, Dict[str, Any]]:
        return {
            "feasible_sets": _serial_ranges(self.feasible_sets),
            "estimates": _serial_estimates(self.estimates),
        }


@dataclass(frozen=True)
class TumblingCylinderTarget:
    """Tumbling cylindrical target geometry and docking-point kinematics."""

    radius: float
    half_length: float
    initial_orientation: np.ndarray
    angular_velocity: np.ndarray
    tolerance: float = 0.0
    docking_standoff: float = 0.12

    @classmethod
    def from_config(cls, config: SimulationConfig) -> "TumblingCylinderTarget":
        return cls(
            radius=float(config.target.radius),
            half_length=float(config.target.half_length),
            initial_orientation=np.asarray(config.target.initial_orientation, dtype=float),
            angular_velocity=np.asarray(config.target.initial_angular_velocity, dtype=float),
            tolerance=float(config.target.tolerance),
            docking_standoff=float(config.target.docking_standoff),
        )

    @property
    def height(self) -> float:
        return 2.0 * self.half_length

    @property
    def bounding_sphere_radius(self) -> float:
        return float(np.sqrt(self.radius**2 + self.half_length**2))

    @property
    def rendezvous_radius(self) -> float:
        return self.bounding_sphere_radius * (1.0 + self.tolerance)

    def orientation_at(self, anom: float) -> np.ndarray:
        return self.initial_orientation + self.angular_velocity * float(anom)

    def angular_velocity_at(self, anom: float) -> np.ndarray:
        return np.asarray(self.angular_velocity, dtype=float)

    def approach_azimuth(self, initial_state: np.ndarray) -> float:
        position_body = self.approach_position_body(initial_state)
        radial_norm = float(np.linalg.norm(position_body[:2]))
        if radial_norm <= 1.0e-9:
            return 0.0
        return float(np.mod(np.arctan2(position_body[1], position_body[0]), 2.0 * np.pi))

    def approach_position_body(self, initial_state: np.ndarray) -> np.ndarray:
        return target_frame_position(
            np.asarray(initial_state[:3], dtype=float), self.initial_orientation
        )

    def approach_axial_coordinate(self, initial_state: np.ndarray) -> float:
        position_body = self.approach_position_body(initial_state)
        axial_limit = 0.8 * self.half_length
        return float(np.clip(position_body[2], -axial_limit, axial_limit))

    def side_docking_geometry(
        self, azimuth: float, axial: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        normal = np.array([np.cos(azimuth), np.sin(azimuth), 0.0], dtype=float)
        point = normal * (self.radius + self.docking_standoff)
        point[2] = float(np.clip(axial, -0.8 * self.half_length, 0.8 * self.half_length))
        return point, normal

    def docking_points_body(self, count: int, start_azimuth: float = 0.0) -> List[np.ndarray]:
        points = []
        for idx in range(count):
            angle = start_azimuth + 2.0 * np.pi * idx / max(count, 1)
            point, _ = self.side_docking_geometry(angle)
            points.append(point)
        return points

    def assign_docking_targets(
        self, initial_states: Sequence[np.ndarray]
    ) -> List[Dict[str, Any]]:
        count = len(initial_states)
        if count == 0:
            return []

        approach_angles = [self.approach_azimuth(state) for state in initial_states]
        approach_axials = [
            self.approach_axial_coordinate(state) for state in initial_states
        ]
        if count == 1:
            start_azimuth = approach_angles[0]
            candidate_angles = [start_azimuth]
            candidate_geometry = [self.side_docking_geometry(start_azimuth)]
            assignment = [(0, 0)]
        else:
            best_cost = np.inf
            best_angles: List[float] = []
            best_assignment: List[Tuple[int, int]] = []
            start_options = [0.0]
            start_options.extend(
                angle - 2.0 * np.pi * offset / count
                for angle in approach_angles
                for offset in range(count)
            )
            for start_azimuth in start_options:
                angles = [
                    float(
                        np.mod(
                            start_azimuth + 2.0 * np.pi * idx / count,
                            2.0 * np.pi,
                        )
                    )
                    for idx in range(count)
                ]
                costs = np.zeros((count, count), dtype=float)
                for state_idx, approach_angle in enumerate(approach_angles):
                    for point_idx, candidate_angle in enumerate(angles):
                        costs[state_idx, point_idx] = abs(
                            np.mod(
                                approach_angle - candidate_angle + np.pi,
                                2.0 * np.pi,
                            )
                            - np.pi
                        )
                rows, cols = linear_sum_assignment(costs)
                total_cost = float(costs[rows, cols].sum())
                if total_cost < best_cost - 1.0e-12:
                    best_cost = total_cost
                    best_angles = angles
                    best_assignment = sorted(
                        zip(rows.tolist(), cols.tolist()), key=lambda item: item[0]
                    )
            candidate_angles = best_angles
            candidate_geometry = [
                self.side_docking_geometry(angle) for angle in candidate_angles
            ]
            assignment = best_assignment

        assigned: List[Dict[str, Any]] = []
        for state_idx, point_idx in assignment:
            _, normal = candidate_geometry[point_idx]
            point, normal = self.side_docking_geometry(
                candidate_angles[point_idx], approach_axials[state_idx]
            )
            assigned.append(
                {
                    "body_position": point,
                    "body_normal": normal,
                    "azimuth": candidate_angles[point_idx],
                    "surface": "cylinder_side",
                    "standoff": self.docking_standoff,
                    "candidate_index": int(point_idx),
                }
            )
        return assigned

    def docking_state(self, docking_point_body: np.ndarray, anom: float) -> np.ndarray:
        orientation = self.orientation_at(anom)
        position = rotating_docking_point(docking_point_body, orientation)
        velocity = rotating_body_point_velocity(
            docking_point_body,
            orientation,
            self.angular_velocity_at(anom),
        )
        return np.concatenate((position, velocity))

    def collision_params(self, operation: str, anom: float) -> Dict[str, Any]:
        radius = self.radius * (1.0 + self.tolerance)
        half_length = self.half_length * (1.0 + self.tolerance)
        return {
            "operation": operation,
            "shape": "cylinder",
            "active_safety_model": (
                "bounding_sphere" if operation == "rendezvous" else "rotating_cylinder_union"
            ),
            "center": [0.0, 0.0, 0.0],
            "orientation": self.orientation_at(anom).tolist(),
            "angular_velocity": self.angular_velocity_at(anom).tolist(),
            "target_radius": radius,
            "target_height": self.height * (1.0 + self.tolerance),
            "target_half_length": half_length,
            "rendezvous_radius": self.rendezvous_radius,
            "bounding_sphere_radius": self.bounding_sphere_radius,
            "docking_standoff": self.docking_standoff,
            "tube_radius": 0.0,
        }

    def cylinder_constraint(self, anom: float, tube_radius: float = 0.0) -> CylinderConstraint:
        return CylinderConstraint(
            radius=self.radius * (1.0 + self.tolerance),
            half_length=self.half_length * (1.0 + self.tolerance),
            orientation=self.orientation_at(anom),
            tube_radius=tube_radius,
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "shape": "cylinder",
            "radius": float(self.radius),
            "height": float(self.height),
            "half_length": float(self.half_length),
            "bounding_sphere_radius": float(self.bounding_sphere_radius),
            "rendezvous_sphere_radius": float(self.rendezvous_radius),
            "initial_orientation": self.initial_orientation.tolist(),
            "angular_velocity": self.angular_velocity.tolist(),
            "tolerance": float(self.tolerance),
            "docking_standoff": float(self.docking_standoff),
            "docking_clearance": float(self.docking_standoff),
            "assignment_strategy": "cylinder_side_approach_hungarian",
        }


@dataclass(frozen=True)
class ChaserConfig:
    """One chaser's initial condition and assigned docking point."""

    chaser_id: str
    initial_state: np.ndarray
    docking_point_body: np.ndarray
    docking_normal_body: np.ndarray
    docking_azimuth: float
    docking_surface: str = "cylinder_side"
    docking_standoff: float = 0.0
    docking_candidate_index: int = 0

    def assignment_metadata(self) -> Dict[str, Any]:
        return {
            "chaser_id": self.chaser_id,
            "initial_state": self.initial_state.tolist(),
            "docking_point_body": self.docking_point_body.tolist(),
            "docking_normal_body": self.docking_normal_body.tolist(),
            "docking_azimuth": float(self.docking_azimuth),
            "docking_surface": self.docking_surface,
            "docking_standoff": float(self.docking_standoff),
            "docking_candidate_index": int(self.docking_candidate_index),
        }


@dataclass(frozen=True)
class MissionPhase:
    """Controller phase metadata for one MPC update."""

    name: str
    operation: str


@dataclass
class PaperSystemResult:
    """Serializable paper-system mission result."""

    mission: str
    approximation: str
    solver_backend: str
    success: bool
    message: str
    anom_history: List[float]
    phase_history: List[str]
    sample_phase_history: List[str]
    actual_trajectories: Dict[str, List[List[float]]]
    nominal_trajectories: Dict[str, List[List[List[float]]]]
    controls: Dict[str, List[List[float]]]
    solve_stats: Dict[str, List[Dict[str, Any]]]
    solver_costs: Dict[str, List[Optional[float]]]
    truth: Dict[str, float]
    target: Dict[str, Any]
    target_attitude_history: List[List[float]]
    target_angular_velocity_history: List[List[float]]
    docking_points: Dict[str, List[float]]
    initial_belief: Dict[str, Dict[str, Any]]
    final_belief: Dict[str, Dict[str, Any]]
    feasible_set_history: List[Dict[str, List[float]]]
    parameter_estimate_history: List[Dict[str, float]]
    smid_records: List[Dict[str, Any]]
    tube_profiles: Dict[str, List[Dict[str, Any]]]
    tube_radius_history: List[float]
    rendezvous_margins: Dict[str, List[float]]
    docking_cylinder_margins: Dict[str, List[float]]
    active_target_margins: Dict[str, List[float]]
    pairwise_spacing: List[Dict[str, float]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.__dict__)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PaperSystemResult":
        payload = copy.deepcopy(data)
        if "sample_phase_history" not in payload:
            payload["sample_phase_history"] = payload.get("metadata", {}).get(
                "sample_phase_history",
                ["rendezvous"] * len(payload.get("anom_history", [])),
            )
        if "active_target_margins" not in payload:
            payload["active_target_margins"] = payload.get("rendezvous_margins", {})
        return cls(**payload)


class PaperSystemRunner:
    """Run the paper ADTMPC system for single or multi-chaser missions."""

    def __init__(
        self,
        *,
        solver_backend: str = "gekko",
        approximation: str = "nonlinear",
        config: Optional[SimulationConfig] = None,
        horizon_steps: Optional[int] = 10,
        actuation_steps: Optional[int] = 1,
        smid_window: int = 4,
    ):
        self.solver_backend = solver_backend
        self.approximation = approximation
        self.config = config or SimulationConfig.load()
        self.truth = PaperTruth.from_config(self.config)
        self.target = TumblingCylinderTarget.from_config(self.config)
        self.horizon_steps = None if horizon_steps is None else int(horizon_steps)
        self.actuation_steps = None if actuation_steps is None else int(actuation_steps)
        self.smid_window = int(max(2, smid_window))
        self.smid = SMIDAdaptor(
            error_bound=0.15,
            prediction_error_threshold=0.005,
            max_iter=90,
            minimum_widths={
                "eccentricity": 0.04,
                "alpha": 1.0e-7,
                "beta": 1.5e-7,
            },
        )
        self.rendezvous_standoff = 0.5
        self.rendezvous_capture_radius = 3.5
        self.rendezvous_position_tolerance = 0.15
        self.docking_position_tolerance = 0.08
        self.docking_velocity_tolerance = 0.08
        self.min_chaser_separation = float(self.config.mpc.min_chaser_separation)
        self.target_safety_tolerance = 1.0e-7
        self.target_barrier_buffer = 0.04

    @property
    def linearized(self) -> bool:
        return self.approximation == "linearized"

    def run(self, *, mission: str, steps: int) -> PaperSystemResult:
        if mission == "single":
            return self._run_chasers(mission="single", chasers=self._single_chaser(), steps=steps)
        if mission == "multi":
            return self._run_chasers(mission="multi", chasers=self._multi_chasers(), steps=steps)
        raise ValueError(f"Unknown paper mission: {mission}")

    def save_result(self, result: PaperSystemResult, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "mission_data.json"
        payload = {
            "schema": "orbexa.paper_system.v1",
            "result": result.to_dict(),
        }
        save_data(path, payload)
        return path

    @staticmethod
    def load_result(path: Path) -> PaperSystemResult:
        payload = load_data(path)
        if "result" in payload:
            return PaperSystemResult.from_dict(payload["result"])
        return PaperSystemResult.from_dict(payload)

    def _run_chasers(
        self,
        *,
        mission: str,
        chasers: Sequence[ChaserConfig],
        steps: int,
    ) -> PaperSystemResult:
        controller = MPCController(
            solver_backend=self.solver_backend,
            solver_config=self._solver_config(),
        )
        belief = ParameterBelief.paper_default(self.config)
        initial_belief = belief.copy()

        states = {
            chaser.chaser_id: np.asarray(chaser.initial_state, dtype=float).copy()
            for chaser in chasers
        }
        actual = {
            chaser_id: [state.copy().tolist()] for chaser_id, state in states.items()
        }
        nominal: Dict[str, List[List[List[float]]]] = {
            chaser.chaser_id: [] for chaser in chasers
        }
        controls: Dict[str, List[List[float]]] = {chaser.chaser_id: [] for chaser in chasers}
        tube_profiles: Dict[str, List[Dict[str, Any]]] = {
            chaser.chaser_id: [] for chaser in chasers
        }
        solve_stats: Dict[str, List[Dict[str, Any]]] = {
            chaser.chaser_id: [] for chaser in chasers
        }
        solver_costs: Dict[str, List[Optional[float]]] = {
            chaser.chaser_id: [] for chaser in chasers
        }

        anom = 0.0
        anom_history = [anom]
        phase_history: List[str] = []
        active_phase = MissionPhase(name="rendezvous", operation="rendezvous")
        sample_phase_history: List[str] = [active_phase.name]
        feasible_history = [_serial_ranges(belief.feasible_sets)]
        estimate_history = [_serial_estimates(belief.estimates)]
        smid_records: List[Dict[str, Any]] = []
        tube_radius_history = [0.0]
        attitude_history = [self.target.orientation_at(anom).tolist()]
        angular_velocity_history = [self.target.angular_velocity_at(anom).tolist()]

        solver_success = True
        mission_complete = False
        messages: List[str] = []
        previous_anom = anom

        for update_index in range(int(steps)):
            if active_phase.operation == "rendezvous" and self._rendezvous_goal_satisfied(
                states, chasers
            ):
                active_phase = MissionPhase(name="docking", operation="docking")
                sample_phase_history[-1] = active_phase.name
            if active_phase.operation == "docking" and self._docking_goal_satisfied(
                states, chasers, anom
            ):
                mission_complete = True
                break

            phase_history.append(active_phase.name)
            update_solutions: Dict[str, Dict[str, Any]] = {}
            predicted_terminal_states: Dict[str, np.ndarray] = {}

            for chaser in chasers:
                chaser_id = chaser.chaser_id
                state = states[chaser_id]
                belief_params = belief.dynamics_params(self.truth)
                matrices, _, _ = orbital_ellp_drag(anom_step=self.config.anom_step, **belief_params)
                solve_bounds = self.config.mpc.bounds()
                horizon_steps = self._horizon_steps_for_phase(active_phase)
                tube_config, profile = self._tube_for_belief(
                    belief, belief_params, anom, horizon_steps
                )
                target_params = self.target.collision_params(active_phase.operation, anom)
                final_state = self._phase_reference_state(
                    active_phase,
                    chaser,
                    state,
                    anom,
                    tube_radius=profile.max_position_radius,
                )

                solve_kwargs: Dict[str, Any] = {
                    "mean_motion": belief_params["mean_motion"],
                    "state_cost_matrix": self._phase_state_weights(active_phase),
                    "input_cost_matrix": self.config.mpc.R,
                    "target_params": target_params,
                }
                tube_profiles[chaser_id].append(self._tube_metadata(anom, profile))
                solve_bounds = self._tightened_bounds(solve_bounds, profile)
                if active_phase.operation == "rendezvous":
                    solve_kwargs["target_params"] = tighten_target_params(
                        solve_kwargs["target_params"], profile
                    )
                solve_kwargs["tube_mpc"] = tube_config

                pairwise = self._pairwise_constraints(
                    chaser_id=chaser_id,
                    states={**states, **predicted_terminal_states},
                    horizon_steps=horizon_steps,
                )
                if pairwise:
                    solve_kwargs["pairwise_constraints"] = pairwise

                if self.linearized:
                    solve_kwargs.update(
                        self._linearized_constraints(
                            phase=active_phase,
                            state=state,
                            target_params=target_params,
                            tube_radius=(
                                profile.max_position_radius
                                if active_phase.operation == "rendezvous"
                                else 0.0
                            ),
                            anom=anom,
                        )
                    )

                result = controller.solve_step(
                    initial_state=state,
                    final_state=final_state,
                    control_input_0=np.zeros(3),
                    start_anom=anom,
                    anom_step=self.config.anom_step,
                    num_steps=horizon_steps,
                    dynamics=matrices,
                    bounds=solve_bounds,
                    time_periapsis=belief_params["time_periapsis"],
                    eccentricity=belief_params["eccentricity"],
                    use_anomaly_scaling=False,
                    **solve_kwargs,
                )
                solve_stats[chaser_id].append(
                    {
                        "success": bool(result.success),
                        "solve_time": float(result.solve_time),
                        "message": result.message,
                        "solver_info": result.solver_info,
                    }
                )
                solver_costs[chaser_id].append(
                    None if result.cost is None else float(result.cost)
                )

                if not result.success:
                    solver_success = False
                    messages.append(f"{chaser_id}:{result.message}")
                    break

                nominal_trajectory = np.asarray(result.state_trajectory, dtype=float)
                nominal[chaser_id].append(nominal_trajectory.T.tolist())
                if nominal_trajectory.ndim == 2 and nominal_trajectory.shape[1] > 0:
                    predicted_terminal_states[chaser_id] = nominal_trajectory[:, -1]
                update_solutions[chaser_id] = {
                    "result": result,
                    "belief_params": belief_params,
                    "tube_config": tube_config,
                    "tube_profile": profile,
                    "reference_state": final_state,
                    "phase": active_phase,
                }

            if not solver_success:
                break

            act_count = self._actuation_steps_for_phase(active_phase, update_solutions)
            for act_idx in range(act_count):
                controls_to_apply: Dict[str, np.ndarray] = {}
                sample_tube_radius = 0.0
                for chaser in chasers:
                    chaser_id = chaser.chaser_id
                    solution = update_solutions[chaser_id]
                    profile = solution["tube_profile"]
                    sample_tube_radius = max(
                        sample_tube_radius,
                        float(profile.max_position_radius),
                    )
                    control = self._control_at(solution["result"], act_idx)
                    control = self._apply_ancillary(
                        control=control,
                        state=states[chaser_id],
                        result=solution["result"],
                        anom=anom,
                        belief_params=solution["belief_params"],
                        tube_config=solution["tube_config"],
                        nominal_index=act_idx,
                    )
                    control = self._apply_tracking_feedback(
                        control=control,
                        state=states[chaser_id],
                        reference_state=self._tracking_reference_state(solution, act_idx),
                        phase=solution["phase"],
                        anom=anom,
                        belief_params=solution["belief_params"],
                    )
                    controls_to_apply[chaser_id] = self._clip_control(control)

                next_states = {}
                applied_controls: Dict[str, np.ndarray] = {}
                for chaser_id, control in controls_to_apply.items():
                    applied_control = self._apply_target_barrier_control(
                        state=states[chaser_id],
                        control=control,
                        anom=anom,
                        phase=active_phase,
                    )
                    applied_control = self._clip_control(applied_control)
                    applied_controls[chaser_id] = applied_control
                    next_states[chaser_id] = self._propagate_truth(
                        states[chaser_id],
                        applied_control,
                        anom,
                    )
                for chaser_id, control in applied_controls.items():
                    controls[chaser_id].append(control.tolist())

                states.update(next_states)
                for chaser_id, state in states.items():
                    actual[chaser_id].append(state.tolist())

                previous_anom = anom
                anom += self.config.anom_step
                anom_history.append(float(anom))
                sample_phase_history.append(active_phase.name)
                tube_radius_history.append(float(sample_tube_radius))
                attitude_history.append(self.target.orientation_at(anom).tolist())
                angular_velocity_history.append(self.target.angular_velocity_at(anom).tolist())

                if act_idx == act_count - 1:
                    primary_id = chasers[0].chaser_id
                    primary_states = np.asarray(actual[primary_id], dtype=float)
                    primary_controls = np.asarray(controls[primary_id], dtype=float)
                    if len(primary_states) >= 2 and len(primary_controls) >= 1:
                        window_len = min(self.smid_window, len(primary_states))
                        window_states = primary_states[-window_len:]
                        window_controls = primary_controls[-(window_len - 1):]
                        window_start = anom_history[-window_len]
                        fss, estimates, record = self.smid.update(
                            feasible_sets=belief.feasible_sets,
                            estimates=belief.estimates,
                            states=window_states,
                            controls=window_controls,
                            start_anom=window_start,
                            anom_step=self.config.anom_step,
                            dynamics_context=self._smid_context(),
                        )
                        belief = ParameterBelief(feasible_sets=fss, estimates=estimates)
                        smid_records.append(record.to_dict())

                feasible_history.append(_serial_ranges(belief.feasible_sets))
                estimate_history.append(_serial_estimates(belief.estimates))

                if active_phase.operation == "rendezvous" and self._rendezvous_goal_satisfied(
                    states, chasers
                ):
                    active_phase = MissionPhase(name="docking", operation="docking")
                    break
                if active_phase.operation == "docking" and self._docking_goal_satisfied(
                    states, chasers, anom
                ):
                    mission_complete = True
                    break

            if mission_complete:
                break

        rendezvous_margins, docking_margins, active_margins = self._safety_margins(
            actual, anom_history, sample_phase_history, tube_radius_history
        )
        pairwise_spacing = self._spacing_history(actual, anom_history)

        active_safety_ok = self._active_target_margins_safe(active_margins)
        spacing_ok = self._pairwise_spacing_safe(pairwise_spacing)
        success = solver_success and mission_complete and active_safety_ok and spacing_ok
        if solver_success and not mission_complete:
            messages.append("max_mpc_updates_reached_before_goal")
        if mission_complete and not active_safety_ok:
            messages.append("active_target_margin_negative")
        if mission_complete and not spacing_ok:
            messages.append("pairwise_separation_below_configured_minimum")

        return PaperSystemResult(
            mission=mission,
            approximation=self.approximation,
            solver_backend=self.solver_backend,
            success=success,
            message="; ".join(messages),
            anom_history=anom_history,
            phase_history=phase_history,
            sample_phase_history=sample_phase_history,
            actual_trajectories=actual,
            nominal_trajectories=nominal,
            controls=controls,
            solve_stats=solve_stats,
            solver_costs=solver_costs,
            truth=self.truth.as_dict(),
            target=self.target.as_dict(),
            target_attitude_history=attitude_history,
            target_angular_velocity_history=angular_velocity_history,
            docking_points={
                chaser.chaser_id: chaser.docking_point_body.tolist()
                for chaser in chasers
            },
            initial_belief=initial_belief.as_metadata(),
            final_belief=belief.as_metadata(),
            feasible_set_history=feasible_history,
            parameter_estimate_history=estimate_history,
            smid_records=smid_records,
            tube_profiles=tube_profiles,
            tube_radius_history=tube_radius_history,
            rendezvous_margins=rendezvous_margins,
            docking_cylinder_margins=docking_margins,
            active_target_margins=active_margins,
            pairwise_spacing=pairwise_spacing,
            metadata={
                "horizon_steps": self._configured_horizon_metadata(),
                "actuation_steps": self.actuation_steps,
                "anom_step": self.config.anom_step,
                "max_mpc_updates": int(steps),
                "mission_complete": mission_complete,
                "active_target_margins_safe": active_safety_ok,
                "target_safety_tolerance": self.target_safety_tolerance,
                "min_chaser_separation": self.min_chaser_separation,
                "pairwise_spacing_safe": spacing_ok,
                "target_barrier": {
                    "enabled": True,
                    "radial_buffer": self.target_barrier_buffer,
                },
                "phase_updates": phase_history,
                "sample_phase_history": sample_phase_history,
                "chaser_assignments": [
                    chaser.assignment_metadata() for chaser in chasers
                ],
                "primary_smid_chaser": chasers[0].chaser_id,
                "previous_anom": previous_anom if "previous_anom" in locals() else 0.0,
            },
        )

    def _single_chaser(self) -> List[ChaserConfig]:
        initial_states = [
            np.asarray(self.config.orbit.initial_conditions, dtype=float),
        ]
        return self._build_chaser_configs(initial_states)

    def _multi_chasers(self) -> List[ChaserConfig]:
        initial_states = [
            np.array([-6.0, 1.4, 0.5, 0.02, -0.008, 0.004], dtype=float),
            np.array([-5.6, -1.6, 0.8, 0.015, 0.010, -0.006], dtype=float),
            np.array([-6.3, 0.1, -1.4, 0.018, -0.002, 0.012], dtype=float),
        ]
        return self._build_chaser_configs(initial_states[: int(self.config.num_chasers)])

    def _build_chaser_configs(
        self, initial_states: Sequence[np.ndarray]
    ) -> List[ChaserConfig]:
        assignments = self.target.assign_docking_targets(initial_states)
        return [
            ChaserConfig(
                chaser_id=f"chaser_{idx + 1}",
                initial_state=np.asarray(initial_states[idx], dtype=float),
                docking_point_body=np.asarray(assignment["body_position"], dtype=float),
                docking_normal_body=np.asarray(assignment["body_normal"], dtype=float),
                docking_azimuth=float(assignment["azimuth"]),
                docking_surface=str(assignment["surface"]),
                docking_standoff=float(assignment["standoff"]),
                docking_candidate_index=int(assignment["candidate_index"]),
            )
            for idx, assignment in enumerate(assignments)
        ]

    def _phase_reference_state(
        self,
        phase: MissionPhase,
        chaser: ChaserConfig,
        state: np.ndarray,
        anom: float,
        tube_radius: float = 0.0,
    ) -> np.ndarray:
        if phase.operation == "docking":
            return self.target.docking_state(chaser.docking_point_body, anom)
        direction = np.asarray(state[:3], dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-9:
            direction = np.array([1.0, 0.0, 0.0])
            norm = 1.0
        physical_radius = self.target.rendezvous_radius + self.rendezvous_standoff
        expanded_radius = self.target.rendezvous_radius + float(tube_radius) + 0.2
        approach_radius = max(physical_radius, expanded_radius)
        return np.concatenate((direction / norm * approach_radius, np.zeros(3)))

    def _rendezvous_goal_satisfied(
        self,
        states: Dict[str, np.ndarray],
        chasers: Sequence[ChaserConfig],
    ) -> bool:
        desired_radius = max(
            self.target.rendezvous_radius + self.rendezvous_standoff,
            self.rendezvous_capture_radius,
        )
        for chaser in chasers:
            state = np.asarray(states[chaser.chaser_id], dtype=float)
            distance = float(np.linalg.norm(state[:3]))
            velocity = float(np.linalg.norm(state[3:]))
            outside_target = distance >= self.target.rendezvous_radius
            near_shell = distance <= desired_radius + self.rendezvous_position_tolerance
            if not (outside_target and near_shell and velocity <= 5.0):
                return False
        return True

    def _docking_goal_satisfied(
        self,
        states: Dict[str, np.ndarray],
        chasers: Sequence[ChaserConfig],
        anom: float,
    ) -> bool:
        for chaser in chasers:
            state = np.asarray(states[chaser.chaser_id], dtype=float)
            goal = self.target.docking_state(chaser.docking_point_body, anom)
            position_error = float(np.linalg.norm(state[:3] - goal[:3]))
            velocity_error = float(np.linalg.norm(state[3:] - goal[3:]))
            if (
                position_error > self.docking_position_tolerance
                or velocity_error > self.docking_velocity_tolerance
            ):
                return False
        return True

    def _phase_state_weights(self, phase: MissionPhase) -> np.ndarray:
        weights = np.asarray(np.diag(np.diag(self.config.mpc.Q)), dtype=float).copy()
        if phase.operation == "docking":
            weights[:3, :3] *= 6.0
            weights[3:, 3:] *= 1.5
        return weights

    def _horizon_steps_for_phase(self, phase: MissionPhase) -> int:
        if self.horizon_steps is not None:
            return int(self.horizon_steps)
        return int(self.config.mpc.horizon_steps.get(phase.operation, 10))

    def _configured_horizon_metadata(self) -> Dict[str, int]:
        if self.horizon_steps is not None:
            return {"all": int(self.horizon_steps)}
        return {
            key: int(value)
            for key, value in self.config.mpc.horizon_steps.items()
        }

    def _tube_for_belief(
        self,
        belief: ParameterBelief,
        belief_params: Dict[str, float],
        anom: float,
        horizon_steps: int,
    ):
        tube_config = {
            "enabled": True,
            "lambda_gain": [float(value) for value in self.config.tube.sliding_gains],
            "alpha": [float(value) for value in self.config.tube.bandwidth_0],
            "phi": [float(value) for value in self.config.tube.boundary_layer_0],
            "eccentricity_range": belief.feasible_sets["eccentricity"],
            "aRange": belief.feasible_sets["alpha"],
            "bRange": belief.feasible_sets["beta"],
            "tighten_inputs": True,
        }
        profile = propagate_tube_profile(
            start_anom=anom,
            num_steps=horizon_steps,
            anom_step=self.config.anom_step,
            mean_motion=belief_params["mean_motion"],
            t_periapsis=belief_params["time_periapsis"],
            lambda_gain=tube_config["lambda_gain"],
            alpha=tube_config["alpha"],
            phi_0=tube_config["phi"],
            eccentricity_range=tube_config["eccentricity_range"],
            aRange=tube_config["aRange"],
            bRange=tube_config["bRange"],
            mu=belief_params["mu"],
            semi_major_axis=belief_params["semi_major_axis"],
            specific_angular_momentum=belief_params["specific_angular_momentum"],
        )
        tube_config["profile"] = profile
        return tube_config, profile

    def _tightened_bounds(self, bounds, profile):
        state_bounds, input_bounds = bounds
        input_tightening = input_tightening_from_profile(
            profile, self.config.tube.sliding_gains
        )
        tightened_inputs = tighten_box_bounds(input_bounds, input_tightening)
        for bound in tightened_inputs or []:
            lower = bound.get("lower")
            upper = bound.get("upper")
            if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
                if lower >= upper:
                    center = 0.5 * (float(lower) + float(upper))
                    bound["lower"] = center - 1.0e-6
                    bound["upper"] = center + 1.0e-6
        return tighten_box_bounds(state_bounds, profile.max_state_error), tightened_inputs

    def _tube_metadata(self, anom: float, profile) -> Dict[str, Any]:
        return {
            "start_anom": float(anom),
            "time_grid": profile.time_grid.tolist(),
            "phi": profile.phi.tolist(),
            "omega": profile.omega.tolist(),
            "position_radius": profile.position_radius.tolist(),
            "max_position_radius": profile.max_position_radius,
            "max_state_error": profile.max_state_error.tolist(),
            "input_tightening": input_tightening_from_profile(
                profile, self.config.tube.sliding_gains
            ).tolist(),
        }

    def _tube_radius_at(self, profile, index: int) -> float:
        radii = profile.position_radius
        if len(radii) == 0:
            return 0.0
        return float(radii[min(index, len(radii) - 1)])

    def _linearized_constraints(
        self,
        *,
        phase: MissionPhase,
        state: np.ndarray,
        target_params: Dict[str, Any],
        tube_radius: float,
        anom: float,
    ) -> Dict[str, Any]:
        if phase.operation == "rendezvous":
            affine = linearize_rendezvous_constraint(
                state[:3],
                target_radius=float(target_params["rendezvous_radius"]),
                tube_radius=tube_radius,
            )
        else:
            affine = linearize_cylinder_constraint(
                state[:3],
                self.target.cylinder_constraint(anom, tube_radius=tube_radius),
            )
        return {"target_params": None, "affine_constraints": [affine]}

    def _pairwise_constraints(
        self,
        *,
        chaser_id: str,
        states: Dict[str, np.ndarray],
        horizon_steps: int,
    ) -> List[Dict[str, Any]]:
        constraints = []
        for other_id, other_state in states.items():
            if other_id == chaser_id:
                continue
            reference = np.repeat(
                np.asarray(other_state[:3], dtype=float).reshape(1, 3),
                horizon_steps,
                axis=0,
            )
            constraints.append(
                {
                    "other_chaser": other_id,
                    "reference_positions": reference.tolist(),
                    "min_separation": self.min_chaser_separation,
                }
            )
        return constraints

    def _first_control(self, result: SolverResult) -> np.ndarray:
        return self._control_at(result, 0)

    def _control_at(self, result: SolverResult, index: int) -> np.ndarray:
        controls = np.asarray(result.control_trajectory, dtype=float)
        if controls.ndim == 1:
            return controls.copy()
        if controls.shape[1] == 0:
            return np.zeros(controls.shape[0], dtype=float)
        return controls[:, min(index, controls.shape[1] - 1)].copy()

    def _nominal_state_at(
        self, result: SolverResult, index: int, fallback: np.ndarray
    ) -> np.ndarray:
        if result.state_trajectory is None:
            return np.asarray(fallback, dtype=float)
        nominal = np.asarray(result.state_trajectory, dtype=float)
        if nominal.ndim != 2 or nominal.shape[1] == 0:
            return np.asarray(fallback, dtype=float)
        return nominal[:, min(index, nominal.shape[1] - 1)].copy()

    def _tracking_reference_state(
        self, solution: Dict[str, Any], index: int
    ) -> np.ndarray:
        final_reference = np.asarray(solution["reference_state"], dtype=float)
        if solution["phase"].operation != "docking":
            return final_reference
        nominal_reference = self._nominal_state_at(
            solution["result"], index + 1, final_reference
        )
        return 0.25 * nominal_reference + 0.75 * final_reference

    def _actuation_steps_for_phase(
        self,
        phase: MissionPhase,
        update_solutions: Dict[str, Dict[str, Any]],
    ) -> int:
        configured = (
            int(self.actuation_steps)
            if self.actuation_steps is not None
            else int(self.config.mpc.actuation_steps.get(phase.operation, 1))
        )
        if configured <= 0:
            configured = int(self.config.mpc.actuation_steps.get(phase.operation, 1))
        available = [self._horizon_steps_for_phase(phase)]
        for solution in update_solutions.values():
            controls = solution["result"].control_trajectory
            if controls is None:
                continue
            controls = np.asarray(controls, dtype=float)
            if controls.ndim == 2:
                available.append(controls.shape[1])
        return max(1, min(configured, *available))

    def _apply_ancillary(
        self,
        *,
        control: np.ndarray,
        state: np.ndarray,
        result: SolverResult,
        anom: float,
        belief_params: Dict[str, float],
        tube_config: Dict[str, Any],
        nominal_index: int = 0,
    ) -> np.ndarray:
        if result.state_trajectory is None:
            return control
        nominal = np.asarray(result.state_trajectory, dtype=float)
        if nominal.ndim != 2 or nominal.shape[1] == 0:
            return control
        nominal_index = min(nominal_index, nominal.shape[1] - 1)
        local_tube_config = dict(tube_config)
        profile = local_tube_config.get("profile")
        if profile is not None and len(profile.phi) > 0:
            profile_index = min(nominal_index, len(profile.phi) - 1)
            local_tube_config["phi"] = np.asarray(profile.phi[profile_index], dtype=float)
        correction = ancillary_controller(
            t=anom,
            t_p=belief_params["time_periapsis"],
            t_f=anom + self.config.anom_step,
            dt=self.config.anom_step,
            mean_motion=belief_params["mean_motion"],
            nom_state=nominal[:, nominal_index],
            act_state=state,
            mu=belief_params["mu"],
            semi_major_axis=belief_params["semi_major_axis"],
            specific_angular_momentum=belief_params["specific_angular_momentum"],
            **local_tube_config,
        )
        return np.asarray(control, dtype=float) + np.asarray(correction, dtype=float)

    def _apply_tracking_feedback(
        self,
        *,
        control: np.ndarray,
        state: np.ndarray,
        reference_state: np.ndarray,
        phase: MissionPhase,
        anom: float,
        belief_params: Dict[str, float],
    ) -> np.ndarray:
        matrices, _, _ = orbital_ellp_drag(
            anom_step=self.config.anom_step,
            **belief_params,
        )
        A_func, _, _, _, d_func = matrices
        A_val = np.asarray(
            A_func(anom, belief_params["time_periapsis"]),
            dtype=float,
        )
        d_val = np.asarray(
            d_func(anom, belief_params["time_periapsis"]),
            dtype=float,
        )
        state = np.asarray(state, dtype=float)
        reference_state = np.asarray(reference_state, dtype=float)
        position_error = reference_state[:3] - state[:3]
        velocity_error = reference_state[3:] - state[3:]
        if phase.operation == "docking":
            kp, kd = 8.0, 8.0
        else:
            kp, kd = 5.0, 5.0
        desired_accel = kp * position_error + kd * velocity_error
        model_accel = A_val[3:] @ state + d_val[3:]
        feedback = desired_accel - model_accel
        if phase.operation == "docking":
            orientation = self.target.orientation_at(anom)
            body_position_error = target_frame_position(position_error, orientation)
            body_velocity_error = target_frame_position(velocity_error, orientation)
            axial_body_accel = np.array(
                [
                    0.0,
                    0.0,
                    30.0 * body_position_error[2] + 18.0 * body_velocity_error[2],
                ],
                dtype=float,
            )
            feedback = feedback + rotating_docking_point(axial_body_accel, orientation)
        return np.asarray(control, dtype=float) + feedback

    def _clip_control(self, control: np.ndarray) -> np.ndarray:
        clipped = np.asarray(control, dtype=float).copy()
        for idx, bound in enumerate(self.config.mpc.input_bounds[: len(clipped)]):
            lower = bound.get("lower", -np.inf)
            upper = bound.get("upper", np.inf)
            if lower != "-Inf":
                clipped[idx] = max(float(lower), clipped[idx])
            if upper != "+Inf":
                clipped[idx] = min(float(upper), clipped[idx])
        return clipped

    def _apply_target_barrier_control(
        self,
        *,
        state: np.ndarray,
        control: np.ndarray,
        anom: float,
        phase: MissionPhase,
    ) -> np.ndarray:
        if phase.operation != "docking":
            return np.asarray(control, dtype=float)

        position = np.asarray(state[:3], dtype=float)
        velocity = np.asarray(state[3:], dtype=float)
        orientation = self.target.orientation_at(anom)
        p_body = target_frame_position(position, orientation)
        v_body = target_frame_position(velocity, orientation)
        radial_norm = float(np.linalg.norm(p_body[:2]))
        if radial_norm <= 1.0e-9:
            return np.asarray(control, dtype=float)

        half_length = self.target.half_length * (1.0 + self.target.tolerance)
        if abs(float(p_body[2])) > half_length + self.target_barrier_buffer:
            return np.asarray(control, dtype=float)

        radial_clearance = radial_norm - self.target.radius * (1.0 + self.target.tolerance)
        if radial_clearance >= self.target_barrier_buffer:
            return np.asarray(control, dtype=float)

        radial_dir_body = np.array(
            [p_body[0] / radial_norm, p_body[1] / radial_norm, 0.0],
            dtype=float,
        )
        radial_velocity = float(np.dot(v_body, radial_dir_body))
        min_outward_velocity = -1.5 * max(radial_clearance, 0.0)
        if radial_velocity >= min_outward_velocity:
            return np.asarray(control, dtype=float)

        required_delta_v = min_outward_velocity - radial_velocity
        correction_body = (
            required_delta_v / max(self.config.anom_step, 1.0e-9)
        ) * radial_dir_body
        correction_lvlh = rotating_docking_point(correction_body, orientation)
        return np.asarray(control, dtype=float) + correction_lvlh

    def _propagate_truth(self, state: np.ndarray, control: np.ndarray, anom: float) -> np.ndarray:
        matrices, _, _ = orbital_ellp_drag(
            anom_step=self.config.anom_step, **self.truth.dynamics_params()
        )
        A_func, B_func, _, _, d_func = matrices
        B_val = np.asarray(B_func(), dtype=float)

        def derivative(local_state: np.ndarray, local_anom: float) -> np.ndarray:
            A_val = np.asarray(
                A_func(local_anom, self.truth.time_periapsis),
                dtype=float,
            )
            d_val = np.asarray(
                d_func(local_anom, self.truth.time_periapsis),
                dtype=float,
            )
            return A_val @ local_state + B_val @ control + d_val

        dt = self.config.anom_step
        state = np.asarray(state, dtype=float)
        k1 = derivative(state, anom)
        k2 = derivative(state + 0.5 * dt * k1, anom + 0.5 * dt)
        k3 = derivative(state + 0.5 * dt * k2, anom + 0.5 * dt)
        k4 = derivative(state + dt * k3, anom + dt)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _active_margin_for_state(
        self, state: np.ndarray, anom: float, phase: MissionPhase
    ) -> float:
        position = np.asarray(state[:3], dtype=float)
        if phase.operation == "docking":
            cylinder = self.target.cylinder_constraint(anom, tube_radius=0.0)
            radial, axial = cylinder.margins(position)
            return float(max(radial, axial))
        return rendezvous_margin(
            position,
            target_radius=self.target.rendezvous_radius,
            tube_radius=0.0,
        )

    def _smid_context(self) -> Dict[str, Any]:
        return {
            "mean_motion": self.truth.mean_motion,
            "mu": self.truth.mu,
            "semi_major_axis": self.truth.semi_major_axis,
            "time_periapsis": self.truth.time_periapsis,
        }

    def _safety_margins(
        self,
        actual: Dict[str, List[List[float]]],
        anom_history: Sequence[float],
        sample_phase_history: Sequence[str],
        tube_radius_history: Sequence[float],
    ) -> Tuple[
        Dict[str, List[float]],
        Dict[str, List[float]],
        Dict[str, List[float]],
    ]:
        rendezvous = {}
        docking = {}
        active = {}
        for chaser_id, states in actual.items():
            rv_values = []
            dock_values = []
            active_values = []
            for idx, state in enumerate(states):
                anom = anom_history[min(idx, len(anom_history) - 1)]
                tube_radius = tube_radius_history[min(idx, len(tube_radius_history) - 1)]
                phase = sample_phase_history[min(idx, len(sample_phase_history) - 1)]
                position = np.asarray(state[:3], dtype=float)
                rv_values.append(
                    rendezvous_margin(
                        position,
                        target_radius=self.target.rendezvous_radius,
                        tube_radius=tube_radius,
                    )
                )
                cylinder = self.target.cylinder_constraint(anom, tube_radius=tube_radius)
                radial, axial = cylinder.margins(position)
                dock_values.append(float(max(radial, axial)))
                if phase == "docking":
                    physical_cylinder = self.target.cylinder_constraint(anom, tube_radius=0.0)
                    physical_radial, physical_axial = physical_cylinder.margins(position)
                    active_values.append(float(max(physical_radial, physical_axial)))
                else:
                    active_values.append(
                        rendezvous_margin(
                            position,
                            target_radius=self.target.rendezvous_radius,
                            tube_radius=0.0,
                        )
                    )
            rendezvous[chaser_id] = rv_values
            docking[chaser_id] = dock_values
            active[chaser_id] = active_values
        return rendezvous, docking, active

    def _active_target_margins_safe(
        self, active_margins: Dict[str, Sequence[float]]
    ) -> bool:
        for values in active_margins.values():
            arr = np.asarray(values, dtype=float)
            if arr.size and float(np.nanmin(arr)) < -self.target_safety_tolerance:
                return False
        return True

    def _pairwise_spacing_safe(self, pairwise_spacing: Sequence[Dict[str, float]]) -> bool:
        values = [
            float(value)
            for entry in pairwise_spacing
            for value in entry.values()
            if np.isfinite(value)
        ]
        if not values:
            return True
        return min(values) >= self.min_chaser_separation - self.target_safety_tolerance

    def _spacing_history(
        self,
        actual: Dict[str, List[List[float]]],
        anom_history: Sequence[float],
    ) -> List[Dict[str, float]]:
        chaser_ids = list(actual.keys())
        spacing = []
        for idx in range(len(anom_history)):
            entry: Dict[str, float] = {}
            for left_idx, left_id in enumerate(chaser_ids):
                for right_id in chaser_ids[left_idx + 1:]:
                    left = np.asarray(actual[left_id][min(idx, len(actual[left_id]) - 1)][:3])
                    right = np.asarray(actual[right_id][min(idx, len(actual[right_id]) - 1)][:3])
                    entry[f"{left_id}:{right_id}"] = float(np.linalg.norm(left - right))
            spacing.append(entry)
        return spacing

    def _solver_config(self) -> Dict[str, Any]:
        if self.solver_backend == "gekko":
            return {"remote": False, "disp": False, "max_iter": 400, "max_memory": 512}
        if self.solver_backend == "scipy":
            return {"method": "SLSQP", "max_iter": 250, "ftol": 1.0e-7}
        return {}


def load_paper_system_result(output_dir: Path) -> PaperSystemResult:
    return PaperSystemRunner.load_result(Path(output_dir) / "mission_data.json")


def result_chaser_ids(result: PaperSystemResult) -> List[str]:
    return list(result.actual_trajectories.keys())
