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

"""ADTMPC rendezvous and docking mission execution.

This module owns the mission-level orchestration for the adaptive dynamic tube
MPC demo: fixed plant truth, controller-side parameter belief, nonlinear
collision constraints, tube tightening, SMID adaptation, and single/multi-chaser
rendezvous-to-docking flows.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from orbexa.control import (
    CylinderConstraint,
    input_tightening_from_profile,
    rendezvous_margin,
    rotating_body_point_velocity,
    rotating_docking_point,
    target_frame_position,
)
from orbexa.core.config import SimulationConfig
from orbexa.control.constraints import rotating_body_point_acceleration
from orbexa.control.tracking_reference import sample_tracking_reference
from orbexa.estimation.adaptor import SMIDAdaptor
from orbexa.utils.io_utils import load_data, save_data


ADTMPC_PARAMETER_KEYS = ("eccentricity", "alpha", "beta")


def _specific_angular_momentum(mu: float, semi_major_axis: float, eccentricity: float) -> float:
    return float(np.sqrt(mu * semi_major_axis * (1.0 - eccentricity**2)))


def _serial_ranges(ranges: Dict[str, Tuple[float, float]]) -> Dict[str, List[float]]:
    return {key: [float(value[0]), float(value[1])] for key, value in ranges.items()}


def _serial_estimates(estimates: Dict[str, float]) -> Dict[str, float]:
    return {key: float(value) for key, value in estimates.items()}


@dataclass(frozen=True)
class OrbitalPlantTruth:
    """Fixed orbital/density parameters used only by truth propagation."""

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
    def from_config(cls, config: SimulationConfig) -> "OrbitalPlantTruth":
        return cls(
            eccentricity=float(config.orbit.eccentricity),
            alpha=float(config.orbit.target_drag),
            beta=float(config.orbit.chaser_drag),
            mu=float(config.orbit.mu),
            semi_major_axis=float(config.orbit.semi_major_axis),
            mean_motion=float(config.orbit.mean_motion),
            time_periapsis=float(config.orbit.time_periapsis),
        )


@dataclass
class ParameterBelief:
    """Controller-side feasible sets and point estimates used by MPC/SMID."""

    feasible_sets: Dict[str, Tuple[float, float]]
    estimates: Dict[str, float]

    @classmethod
    def from_config_prior(cls, config: Optional[SimulationConfig] = None) -> "ParameterBelief":
        """Build the initial adaptive belief from broad or config-centered priors."""
        if config is None:
            ranges = {
                "eccentricity": (0.02, 0.38),
                "alpha": (0.0, 5.5e-7),
                "beta": (0.0, 8.55e-7),
            }
            estimates = {key: float(np.mean(value)) for key, value in ranges.items()}
        else:
            eccentricity = float(config.orbit.eccentricity)
            alpha = float(config.orbit.target_drag)
            beta = float(config.orbit.chaser_drag)
            ranges = {
                "eccentricity": (
                    max(0.0, eccentricity - 0.08),
                    min(0.99, eccentricity + 0.08,
                        1.0 - (config.orbit.central_body_radius + 1.0) /
                        config.orbit.semi_major_axis),
                ),
                "alpha": (max(0.0, 0.5 * alpha), 1.5 * alpha),
                "beta": (max(0.0, 0.5 * beta), 1.5 * beta),
            }
            # Start from a reproducible feasible point rather than the plant
            # truth; SMID/time-series updates are responsible for adaptation.
            rng = np.random.default_rng(int(config.seed))
            estimates = {
                key: float(rng.uniform(low=value[0], high=value[1]))
                for key, value in ranges.items()
            }
        return cls(feasible_sets=ranges, estimates=estimates)

    def copy(self) -> "ParameterBelief":
        return ParameterBelief(
            feasible_sets={
                key: (float(value[0]), float(value[1]))
                for key, value in self.feasible_sets.items()
            },
            estimates=dict(self.estimates),
        )

    def dynamics_params(self, truth_context: OrbitalPlantTruth) -> Dict[str, float]:
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

    def docking_acceleration(self, docking_point_body: np.ndarray, anom: float) -> np.ndarray:
        return rotating_body_point_acceleration(
            docking_point_body,
            self.orientation_at(anom),
            self.angular_velocity_at(anom),
        )

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
class ADTMPCMissionResult:
    """Serializable ADTMPC mission result."""

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
    physical_rendezvous_margins: Dict[str, List[float]] = field(default_factory=dict)
    physical_docking_cylinder_margins: Dict[str, List[float]] = field(default_factory=dict)
    tightened_rendezvous_margins: Dict[str, List[float]] = field(default_factory=dict)
    tightened_docking_cylinder_margins: Dict[str, List[float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tube_diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.__dict__)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ADTMPCMissionResult":
        payload = copy.deepcopy(data)
        if "sample_phase_history" not in payload:
            payload["sample_phase_history"] = payload.get("metadata", {}).get(
                "sample_phase_history",
                ["rendezvous"] * len(payload.get("anom_history", [])),
            )
        if "active_target_margins" not in payload:
            payload["active_target_margins"] = payload.get("rendezvous_margins", {})
        payload.setdefault("physical_rendezvous_margins", payload.get("rendezvous_margins", {}))
        payload.setdefault(
            "physical_docking_cylinder_margins",
            payload.get("docking_cylinder_margins", {}),
        )
        payload.setdefault("tightened_rendezvous_margins", payload.get("rendezvous_margins", {}))
        payload.setdefault(
            "tightened_docking_cylinder_margins",
            payload.get("docking_cylinder_margins", {}),
        )
        return cls(**payload)


class ADTMPCMissionRunner:
    """Run single- or multi-chaser ADTMPC rendezvous/docking missions."""

    def __init__(
        self,
        *,
        solver_backend: str = "gekko",
        approximation: str = "nonlinear",
        config: Optional[SimulationConfig] = None,
        horizon_steps: Optional[int] = None,
        actuation_steps: Optional[int] = None,
    ):
        self.solver_backend = solver_backend
        self.approximation = approximation
        self.config = config or SimulationConfig.load()
        self.truth = OrbitalPlantTruth.from_config(self.config)
        self.target = TumblingCylinderTarget.from_config(self.config)
        self.horizon_steps = None if horizon_steps is None else int(horizon_steps)
        self.actuation_steps = None if actuation_steps is None else int(actuation_steps)
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
        self.docking_position_tolerance = self.config.mpc.docking_position_tolerance
        self.docking_velocity_tolerance = self.config.mpc.docking_velocity_tolerance
        self.min_chaser_separation = float(self.config.mpc.min_chaser_separation)
        self.target_safety_tolerance = 1.0e-7
        self.target_barrier_buffer = 0.04
        self.target_barrier_clearance = 0.05 * self.target_barrier_buffer


    def run(self, *, mission: str, steps: int) -> ADTMPCMissionResult:
        if mission not in ("single", "multi"):
            raise ValueError(f"Unknown ADTMPC mission: {mission}")
        if self.approximation != "nonlinear" or self.solver_backend != "gekko":
            raise ValueError("ADTMPC missions require the nonlinear GEKKO coupled dynamic-tube controller")
        if not self.config.tube.run_tube:
            raise ValueError("ADTMPC missions require tube.run_tube=True; use MPCController for ordinary MPC")
        from orbexa.simulation.coupled_mission import CoupledTubeMission
        return CoupledTubeMission(self).run(mission=mission, steps=steps)

    def save_result(self, result: ADTMPCMissionResult, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "mission_data.json"
        payload = {
            "schema": "orbexa.adtmpc_mission.v1",
            "result": result.to_dict(),
        }
        if not save_data(path, payload):
            raise OSError(f"Could not save mission data to {path}")
        return path

    @staticmethod
    def load_result(path: Path) -> ADTMPCMissionResult:
        payload = load_data(path)
        if "result" in payload:
            return ADTMPCMissionResult.from_dict(payload["result"])
        return ADTMPCMissionResult.from_dict(payload)


    def _single_chaser(self) -> List[ChaserConfig]:
        initial_states = [
            np.asarray(self.config.orbit.initial_conditions, dtype=float),
        ]
        return self._build_chaser_configs(initial_states)

    def _multi_chasers(self) -> List[ChaserConfig]:
        return self._build_chaser_configs(
            self._multi_initial_states(int(self.config.num_chasers))
        )

    def _multi_initial_states(self, count: int) -> List[np.ndarray]:
        """Create deterministic far-field starts for a configurable chaser fleet."""
        if count <= 0:
            raise ValueError("num_chasers must be positive")

        base_state = np.asarray(self.config.orbit.initial_conditions, dtype=float)
        base_radius = max(float(np.linalg.norm(base_state[:2])), 5.5)
        base_angle = float(np.arctan2(base_state[1], base_state[0]))
        axial_span = max(abs(float(base_state[2])), 0.7)
        inward_speed = max(float(np.linalg.norm(base_state[3:5])), 0.018)
        axial_speed = max(abs(float(base_state[5])), 0.004)

        states: List[np.ndarray] = []
        golden_offset = np.pi * (3.0 - np.sqrt(5.0))
        for idx in range(count):
            angle = base_angle + 2.0 * np.pi * idx / count
            radial = base_radius * (1.0 + 0.04 * np.sin(idx * golden_offset))
            vertical_phase = angle + 0.5 * idx * golden_offset
            position = np.array(
                [
                    radial * np.cos(angle),
                    radial * np.sin(angle),
                    axial_span * np.sin(vertical_phase),
                ],
                dtype=float,
            )
            radial_unit = position[:2] / max(float(np.linalg.norm(position[:2])), 1.0e-9)
            tangential_unit = np.array([-radial_unit[1], radial_unit[0]], dtype=float)
            velocity_xy = -inward_speed * radial_unit + 0.25 * inward_speed * tangential_unit
            velocity = np.array(
                [
                    velocity_xy[0],
                    velocity_xy[1],
                    axial_speed * np.cos(vertical_phase),
                ],
                dtype=float,
            )
            states.append(np.concatenate([position, velocity]))
        return states

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

    def _phase_reference_preview(
        self,
        phase: MissionPhase,
        chaser: ChaserConfig,
        current_reference: np.ndarray,
        anom: float,
        horizon_steps: int,
        matrices: Tuple,
        time_periapsis: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Preview target motion and the nominal inputs needed to track it."""
        if phase.operation == "docking":
            def state_at(value):
                return self.target.docking_state(chaser.docking_point_body, value)

            def derivative_at(value):
                state = state_at(value)
                acceleration = self.target.docking_acceleration(chaser.docking_point_body, value)
                return np.concatenate((state[3:], acceleration))
        else:
            def state_at(value):
                return current_reference

            def derivative_at(value):
                return np.zeros_like(current_reference)

        anomalies = anom + np.arange(horizon_steps) * self.config.anom_step
        return sample_tracking_reference(
            anomalies,
            state_at,
            derivative_at,
            lambda value: matrices[0](value, time_periapsis),
            matrices[1](),
            lambda value: matrices[4](value, time_periapsis),
        )

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


    def _tube_metadata(self, anom: float, profile) -> Dict[str, Any]:
        return {
            "start_anom": float(anom),
            "time_grid": profile.time_grid.tolist(),
            "phi": profile.phi.tolist(),
            "omega": profile.omega.tolist(),
            "position_radius": profile.position_radius.tolist(),
            "max_position_radius": profile.max_position_radius,
            "max_state_error": profile.max_state_error.tolist(),
            "lambda_gain": None if profile.lambda_gain is None else np.asarray(profile.lambda_gain).tolist(),
            "bandwidth": None if profile.alpha is None else np.asarray(profile.alpha).tolist(),
            "input_reserve": None if profile.input_reserve is None else np.asarray(profile.input_reserve).tolist(),
            "assumptions": list(profile.assumptions),
            "input_tightening": input_tightening_from_profile(
                profile, self.config.tube.sliding_gains
            ).tolist(),
        }


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
        Dict[str, List[float]],
        Dict[str, List[float]],
    ]:
        """Compute physical safety margins separately from tube-tightened margins.

        Physical margins answer whether the actual chaser is outside the solid
        target.  Tightened margins include the dynamic tube radius and are a
        robust-planning diagnostic, so they can be negative while the physical
        active margin remains safe.
        """
        physical_rendezvous = {}
        physical_docking = {}
        tightened_rendezvous = {}
        tightened_docking = {}
        active = {}
        for chaser_id, states in actual.items():
            physical_rv_values = []
            physical_dock_values = []
            tightened_rv_values = []
            tightened_dock_values = []
            active_values = []
            for idx, state in enumerate(states):
                anom = anom_history[min(idx, len(anom_history) - 1)]
                tube_radius = tube_radius_history[min(idx, len(tube_radius_history) - 1)]
                phase = sample_phase_history[min(idx, len(sample_phase_history) - 1)]
                position = np.asarray(state[:3], dtype=float)
                physical_rv_values.append(
                    rendezvous_margin(
                        position,
                        target_radius=self.target.rendezvous_radius,
                        tube_radius=0.0,
                    )
                )
                tightened_rv_values.append(
                    rendezvous_margin(
                        position,
                        target_radius=self.target.rendezvous_radius,
                        tube_radius=tube_radius,
                    )
                )
                physical_cylinder = self.target.cylinder_constraint(anom, tube_radius=0.0)
                physical_radial, physical_axial = physical_cylinder.margins(position)
                physical_dock_values.append(float(max(physical_radial, physical_axial)))
                tightened_cylinder = self.target.cylinder_constraint(
                    anom, tube_radius=tube_radius
                )
                tightened_radial, tightened_axial = tightened_cylinder.margins(position)
                tightened_dock_values.append(float(max(tightened_radial, tightened_axial)))
                if phase == "docking":
                    active_values.append(float(max(physical_radial, physical_axial)))
                else:
                    active_values.append(physical_rv_values[-1])
            physical_rendezvous[chaser_id] = physical_rv_values
            physical_docking[chaser_id] = physical_dock_values
            tightened_rendezvous[chaser_id] = tightened_rv_values
            tightened_docking[chaser_id] = tightened_dock_values
            active[chaser_id] = active_values
        return (
            physical_rendezvous,
            physical_docking,
            tightened_rendezvous,
            tightened_docking,
            active,
        )

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


def load_adtmpc_mission_result(output_dir: Path) -> ADTMPCMissionResult:
    return ADTMPCMissionRunner.load_result(Path(output_dir) / "mission_data.json")


def result_chaser_ids(result: ADTMPCMissionResult) -> List[str]:
    return list(result.actual_trajectories.keys())
