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

"""Serializable mission records and standalone result I/O."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from orbexa.contracts import MISSION_RESULT_SCHEMA
from orbexa.utils.io.data import load_data, save_data


@dataclass
class ADTMPCMissionResult:
    """Serializable ADTMPC mission result."""

    mission: str
    approximation: str
    solver_backend: str
    success: bool
    message: str
    anom_history: list[float]
    phase_history: list[str]
    sample_phase_history: list[str]
    actual_trajectories: dict[str, list[list[float]]]
    nominal_trajectories: dict[str, list[list[list[float]]]]
    controls: dict[str, list[list[float]]]
    solve_stats: dict[str, list[dict[str, Any]]]
    solver_costs: dict[str, list[float | None]]
    truth: dict[str, float]
    target: dict[str, Any]
    target_attitude_history: list[list[float]]
    target_angular_velocity_history: list[list[float]]
    docking_points: dict[str, list[float]]
    initial_belief: dict[str, dict[str, Any]]
    final_belief: dict[str, dict[str, Any]]
    feasible_set_history: list[dict[str, list[float]]]
    parameter_estimate_history: list[dict[str, float]]
    smid_records: list[dict[str, Any]]
    tube_profiles: dict[str, list[dict[str, Any]]]
    tube_radius_history: list[float]
    rendezvous_margins: dict[str, list[float]]
    docking_cylinder_margins: dict[str, list[float]]
    active_target_margins: dict[str, list[float]]
    pairwise_spacing: list[dict[str, float]]
    physical_rendezvous_margins: dict[str, list[float]] = field(default_factory=dict)
    physical_docking_cylinder_margins: dict[str, list[float]] = field(
        default_factory=dict
    )
    tightened_rendezvous_margins: dict[str, list[float]] = field(default_factory=dict)
    tightened_docking_cylinder_margins: dict[str, list[float]] = field(
        default_factory=dict
    )
    metadata: dict[str, Any] = field(default_factory=dict)
    tube_diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ADTMPCMissionResult:
        payload = copy.deepcopy(data)
        # Partial and older recorded runs can omit optional diagnostic histories.
        # No trajectory, physical context, or current-model identity is invented.
        for name in (
            "phase_history",
            "target_attitude_history",
            "target_angular_velocity_history",
            "feasible_set_history",
            "parameter_estimate_history",
            "smid_records",
            "tube_radius_history",
            "pairwise_spacing",
        ):
            payload.setdefault(name, [])
        for name in (
            "nominal_trajectories",
            "controls",
            "solve_stats",
            "solver_costs",
            "docking_points",
            "initial_belief",
            "final_belief",
            "tube_profiles",
            "rendezvous_margins",
            "docking_cylinder_margins",
        ):
            payload.setdefault(name, {})
        if "sample_phase_history" not in payload:
            payload["sample_phase_history"] = payload.get("metadata", {}).get(
                "sample_phase_history",
                ["rendezvous"] * len(payload.get("anom_history", [])),
            )
        if "active_target_margins" not in payload:
            payload["active_target_margins"] = payload.get("rendezvous_margins", {})
        payload.setdefault(
            "physical_rendezvous_margins", payload.get("rendezvous_margins", {})
        )
        payload.setdefault(
            "physical_docking_cylinder_margins",
            payload.get("docking_cylinder_margins", {}),
        )
        payload.setdefault(
            "tightened_rendezvous_margins", payload.get("rendezvous_margins", {})
        )
        payload.setdefault(
            "tightened_docking_cylinder_margins",
            payload.get("docking_cylinder_margins", {}),
        )
        return cls(**payload)


class MissionResultStore:
    """Read and atomically save plain mission records without a mission engine."""

    filename = "mission_data.json"

    @staticmethod
    def save(result: ADTMPCMissionResult, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / MissionResultStore.filename
        payload = {"schema": MISSION_RESULT_SCHEMA, "result": result.to_dict()}
        if not save_data(path, payload):
            raise OSError(f"Could not save mission data to {path}")
        return path

    @staticmethod
    def load(path: Path) -> ADTMPCMissionResult:
        payload = load_data(Path(path))
        if "result" in payload:
            return ADTMPCMissionResult.from_dict(payload["result"])
        return ADTMPCMissionResult.from_dict(payload)
