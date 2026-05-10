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

from orbexa.control import (
    CylinderConstraint,
    MPCController,
    ancillary_controller,
    input_tightening_from_profile,
    linearize_cylinder_constraint,
    linearize_rendezvous_constraint,
    propagate_tube_profile,
    rendezvous_margin,
    rotating_docking_point,
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
    def paper_default(cls) -> "ParameterBelief":
        ranges = {
            "eccentricity": (0.02, 0.38),
            "alpha": (0.0, 5.5e-7),
            "beta": (0.0, 8.55e-7),
        }
        estimates = {key: float(np.mean(value)) for key, value in ranges.items()}
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
            radius=0.8,
            half_length=0.3,
            initial_orientation=np.asarray(config.target.initial_orientation, dtype=float),
            angular_velocity=np.asarray(config.target.initial_angular_velocity, dtype=float),
            tolerance=float(config.target.tolerance),
        )

    @property
    def rendezvous_radius(self) -> float:
        return max(self.radius, self.half_length) * (1.0 + self.tolerance)

    def orientation_at(self, anom: float) -> np.ndarray:
        return self.initial_orientation + self.angular_velocity * float(anom)

    def angular_velocity_at(self, anom: float) -> np.ndarray:
        return np.asarray(self.angular_velocity, dtype=float)

    def docking_points_body(self, count: int) -> List[np.ndarray]:
        points = []
        dock_radius = self.radius + self.docking_standoff
        for idx in range(count):
            angle = 2.0 * np.pi * idx / max(count, 1)
            points.append(
                np.array(
                    [
                        dock_radius * np.cos(angle),
                        dock_radius * np.sin(angle),
                        0.35 * self.half_length * ((idx % 2) * 2 - 1),
                    ],
                    dtype=float,
                )
            )
        return points

    def docking_state(self, docking_point_body: np.ndarray, anom: float) -> np.ndarray:
        orientation = self.orientation_at(anom)
        position = rotating_docking_point(docking_point_body, orientation)
        velocity = np.cross(self.angular_velocity_at(anom), position)
        return np.concatenate((position, velocity))

    def collision_params(self, operation: str, anom: float) -> Dict[str, Any]:
        radius = self.radius * (1.0 + self.tolerance)
        half_length = self.half_length * (1.0 + self.tolerance)
        return {
            "operation": operation,
            "shape": "cylinder",
            "center": [0.0, 0.0, 0.0],
            "orientation": self.orientation_at(anom).tolist(),
            "target_radius": radius,
            "target_half_length": half_length,
            "rendezvous_radius": max(radius, half_length),
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
            "half_length": float(self.half_length),
            "initial_orientation": self.initial_orientation.tolist(),
            "angular_velocity": self.angular_velocity.tolist(),
            "tolerance": float(self.tolerance),
            "docking_standoff": float(self.docking_standoff),
        }


@dataclass(frozen=True)
class ChaserConfig:
    """One chaser's initial condition and assigned docking point."""

    chaser_id: str
    initial_state: np.ndarray
    docking_point_body: np.ndarray

    def assignment_metadata(self) -> Dict[str, Any]:
        return {
            "chaser_id": self.chaser_id,
            "initial_state": self.initial_state.tolist(),
            "docking_point_body": self.docking_point_body.tolist(),
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
    pairwise_spacing: List[Dict[str, float]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.__dict__)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PaperSystemResult":
        return cls(**copy.deepcopy(data))


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
        )
        self.rendezvous_standoff = 0.5
        self.rendezvous_capture_radius = 3.5
        self.rendezvous_position_tolerance = 0.15
        self.docking_position_tolerance = 0.08
        self.docking_velocity_tolerance = 0.08

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
        belief = ParameterBelief.paper_default()
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
        feasible_history = [_serial_ranges(belief.feasible_sets)]
        estimate_history = [_serial_estimates(belief.estimates)]
        smid_records: List[Dict[str, Any]] = []
        tube_radius_history = [0.0]
        attitude_history = [self.target.orientation_at(anom).tolist()]
        angular_velocity_history = [self.target.angular_velocity_at(anom).tolist()]

        solver_success = True
        mission_complete = False
        active_phase = MissionPhase(name="rendezvous", operation="rendezvous")
        messages: List[str] = []
        previous_anom = anom

        for update_index in range(int(steps)):
            if active_phase.operation == "rendezvous" and self._rendezvous_goal_satisfied(
                states, chasers
            ):
                active_phase = MissionPhase(name="docking", operation="docking")
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
                            tube_radius=profile.max_position_radius,
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
                        self._tube_radius_at(profile, act_idx),
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
                        reference_state=solution["reference_state"],
                        phase=solution["phase"],
                        anom=anom,
                        belief_params=solution["belief_params"],
                    )
                    controls_to_apply[chaser_id] = self._clip_control(control)

                next_states = {
                    chaser_id: self._propagate_truth(
                        states[chaser_id], control, anom
                    )
                    for chaser_id, control in controls_to_apply.items()
                }
                for chaser_id, control in controls_to_apply.items():
                    controls[chaser_id].append(control.tolist())

                states.update(next_states)
                for chaser_id, state in states.items():
                    actual[chaser_id].append(state.tolist())

                previous_anom = anom
                anom += self.config.anom_step
                anom_history.append(float(anom))
                tube_radius_history.append(float(sample_tube_radius))
                attitude_history.append(self.target.orientation_at(anom).tolist())
                angular_velocity_history.append(self.target.angular_velocity_at(anom).tolist())

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

        rendezvous_margins, docking_margins = self._safety_margins(
            actual, anom_history, tube_radius_history
        )
        pairwise_spacing = self._spacing_history(actual, anom_history)

        success = solver_success and mission_complete
        if solver_success and not mission_complete:
            messages.append("max_mpc_updates_reached_before_goal")

        return PaperSystemResult(
            mission=mission,
            approximation=self.approximation,
            solver_backend=self.solver_backend,
            success=success,
            message="; ".join(messages),
            anom_history=anom_history,
            phase_history=phase_history,
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
            pairwise_spacing=pairwise_spacing,
            metadata={
                "horizon_steps": self._configured_horizon_metadata(),
                "actuation_steps": self.actuation_steps,
                "anom_step": self.config.anom_step,
                "max_mpc_updates": int(steps),
                "mission_complete": mission_complete,
                "phase_updates": phase_history,
                "chaser_assignments": [
                    chaser.assignment_metadata() for chaser in chasers
                ],
                "primary_smid_chaser": chasers[0].chaser_id,
                "previous_anom": previous_anom if "previous_anom" in locals() else 0.0,
            },
        )

    def _single_chaser(self) -> List[ChaserConfig]:
        docking_point = self.target.docking_points_body(1)[0]
        return [
            ChaserConfig(
                chaser_id="chaser_1",
                initial_state=np.asarray(self.config.orbit.initial_conditions, dtype=float),
                docking_point_body=docking_point,
            )
        ]

    def _multi_chasers(self) -> List[ChaserConfig]:
        points = self.target.docking_points_body(3)
        initial_states = [
            np.array([-6.0, 1.4, 0.5, 0.02, -0.008, 0.004], dtype=float),
            np.array([-5.6, -1.6, 0.8, 0.015, 0.010, -0.006], dtype=float),
            np.array([-6.3, 0.1, -1.4, 0.018, -0.002, 0.012], dtype=float),
        ]
        return [
            ChaserConfig(
                chaser_id=f"chaser_{idx + 1}",
                initial_state=initial_states[idx],
                docking_point_body=points[idx],
            )
            for idx in range(3)
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
                    "min_separation": 0.35,
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
            **tube_config,
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
        return 0.25 * np.asarray(control, dtype=float) + feedback

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

    def _propagate_truth(self, state: np.ndarray, control: np.ndarray, anom: float) -> np.ndarray:
        matrices, _, _ = orbital_ellp_drag(
            anom_step=self.config.anom_step, **self.truth.dynamics_params()
        )
        A_func, B_func, _, _, d_func = matrices
        A_val = np.asarray(
            A_func(anom, self.truth.time_periapsis),
            dtype=float,
        )
        B_val = np.asarray(B_func(), dtype=float)
        d_val = np.asarray(
            d_func(anom, self.truth.time_periapsis),
            dtype=float,
        )
        return np.asarray(state, dtype=float) + (
            A_val @ state + B_val @ control + d_val
        ) * self.config.anom_step

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
        tube_radius_history: Sequence[float],
    ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        rendezvous = {}
        docking = {}
        for chaser_id, states in actual.items():
            rv_values = []
            dock_values = []
            for idx, state in enumerate(states):
                anom = anom_history[min(idx, len(anom_history) - 1)]
                tube_radius = tube_radius_history[min(idx, len(tube_radius_history) - 1)]
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
            rendezvous[chaser_id] = rv_values
            docking[chaser_id] = dock_values
        return rendezvous, docking

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
