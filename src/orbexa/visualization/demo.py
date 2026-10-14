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

"""Render ADTMPC mission artifacts and run/session reports."""

from __future__ import annotations

import json
import logging
import tempfile
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

_mpl_cache_dir = Path(os.environ.get("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "orbexa-matplotlib")))
_mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache_dir))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from plotly.subplots import make_subplots

from orbexa.control import rotating_docking_point
from orbexa.core.config import SimulationConfig
from orbexa.sessions import MissionSession, validate_session_id
from orbexa.simulation.adtmpc_mission import (
    ADTMPCMissionResult,
    ADTMPCMissionRunner,
    load_adtmpc_mission_result,
)
from orbexa.utils.math_utils import tait_bryan_to_rotation_matrix
from orbexa.utils.io_utils import save_data
from orbexa.visualization.mission_insights import MissionInsights, chaser_color
from orbexa.visualization.mission_replay import MissionReplay
from orbexa.visualization.mission_gallery import MissionGallery
from orbexa.visualization.tube_diagnostics import MatchedTubeDiagnostics


logger = logging.getLogger(__name__)

_CORE_ARTIFACTS = (
    "trajectory.html", "diagnostics.html", "tube_trajectory.html",
    "trajectory.mp4", "trajectory.gif", "actual_nominal_trajectories.png",
    "tube_geometry.png", "control_effort.png", "rendezvous_margin.png",
    "docking_cylinder_margin.png", "tightened_rendezvous_margin.png",
    "tightened_docking_cylinder_margin.png", "active_target_margin.png",
    "smid_fss_widths.png", "parameter_estimates_vs_truth.png",
    "target_attitude.png", "target_angular_velocity.png", "multi_chaser_spacing.png",
)


def _raw_data_reference(data_file: Optional[Path], root: Path) -> Dict:
    if data_file is None:
        return {}
    try:
        return {"data_href": Path(os.path.relpath(data_file, root)).as_posix()}
    except ValueError:
        # Windows cannot form relative links between different drives.
        return {"notes": [f"Raw mission data is saved on another drive: {data_file}"]}


@dataclass(frozen=True)
class DemoConfig:
    """Configuration for ADTMPC mission artifact generation."""

    output_dir: Path = Path("results")
    data_dir: Path = Path("data")
    session_id: Optional[str] = None
    config_path: Optional[Path] = None
    steps: int = 450
    """Maximum number of receding-horizon MPC updates."""
    mission: str = "all"
    primary_solver: str = "gekko"
    secondary_solver: str = "scipy"
    run_linearized: bool = False
    linearized_steps: Optional[int] = 20
    from_data: bool = False
    clean_generated: bool = False
    require_primary_success: bool = True
    fps: int = 8

    def __post_init__(self) -> None:
        for name in ("steps", "fps", "linearized_steps"):
            value = getattr(self, name)
            if name == "linearized_steps" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.mission not in ("single", "multi", "all"):
            raise ValueError(f"Unknown mission: {self.mission}")
        if self.session_id is not None:
            validate_session_id(self.session_id, allow_latest=self.from_data)


@dataclass
class ADTMPCMissionManifest:
    """Summary of generated files for one mission/solver family."""

    output_dir: Path
    mission: str
    approximation: str
    solver_backend: str
    data_file: Optional[Path] = None
    artifacts: List[Path] = field(default_factory=list)
    artifact_labels: Dict[str, str] = field(default_factory=dict)
    skipped_artifacts: Dict[str, str] = field(default_factory=dict)
    success: bool = False
    mission_complete: bool = False
    sample_count: int = 0
    chaser_count: int = 0
    message: str = ""

    def add(self, path: Path, label: Optional[str] = None) -> Path:
        self.artifacts.append(path)
        if label is not None:
            self.artifact_labels[path.name] = label
        return path

    def write(self) -> Path:
        path = self.output_dir / "manifest.json"
        payload = {
            "schema": "orbexa.adtmpc_mission.manifest.v1",
            "mission": self.mission,
            "approximation": self.approximation,
            "solver_backend": self.solver_backend,
            "data_file": None if self.data_file is None else str(self.data_file),
            "artifacts": [str(path) for path in self.artifacts],
            "required_entries": ["index.html", "mission_data.json", *_CORE_ARTIFACTS],
            "artifact_labels": dict(self.artifact_labels),
            "skipped_artifacts": dict(self.skipped_artifacts),
            "success": self.success,
            "mission_complete": self.mission_complete,
            "sample_count": self.sample_count,
            "chaser_count": self.chaser_count,
            "message": self.message,
        }
        payload["required_entries"] = [name for name in payload["required_entries"] if name not in self.skipped_artifacts]
        if not save_data(path, payload):
            raise OSError(f"Could not save artifact manifest to {path}")
        self.add(path)
        return path


class ADTMPCMissionRenderer:
    """Create HTML, plots, and video for one saved ADTMPC mission result."""

    def __init__(self, output_dir: Path, *, fps: int = 8):
        self.output_dir = Path(output_dir)
        self.fps = int(fps)

    def render(
        self,
        result: ADTMPCMissionResult,
        *,
        data_file: Optional[Path] = None,
    ) -> ADTMPCMissionManifest:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = ADTMPCMissionManifest(
            output_dir=self.output_dir,
            mission=result.mission,
            approximation=result.approximation,
            solver_backend=result.solver_backend,
            data_file=data_file,
            success=result.success,
            mission_complete=bool(result.metadata.get("mission_complete", result.success)),
            sample_count=len(result.anom_history),
            chaser_count=len(result.actual_trajectories),
            message=result.message,
        )
        if data_file is not None:
            manifest.add(Path(data_file))
        insights = MissionInsights(result, self.output_dir, fps=self.fps)
        usable_states = bool(len(insights.samples.q)) and all(
            np.asarray(values).ndim == 2 and np.asarray(values).shape[1] >= 6
            and len(values) > 0 and np.all(np.isfinite(np.asarray(values, dtype=float)))
            for values in result.actual_trajectories.values()
        )
        if usable_states:
            self._render_static_plots(result, manifest)
            self._render_interactive_html(result, manifest)
            self._render_interactive_diagnostics(result, manifest)
            self._render_tube_trajectory_html(result, manifest)
            self._render_video(result, manifest)
            self._render_gif(result, manifest, speed_multiplier=2.0)
        else:
            manifest.skipped_artifacts.update({
                name: "No complete usable recorded state trajectories; see saved-data insight notes."
                for name in _CORE_ARTIFACTS
            })
        for path, label in insights.render():
            manifest.add(path, label)
        manifest.skipped_artifacts.update(insights.skipped_artifacts)
        tube_diagnostics = MatchedTubeDiagnostics(result, self.output_dir)
        for path, label in tube_diagnostics.render():
            manifest.add(path, label)
        manifest.skipped_artifacts.update(tube_diagnostics.skipped_artifacts)
        manifest.add(MissionReplay(result).write(self.output_dir / "replay.html"),
                     "Animated replay of recorded states with target rotation and update slider.")
        self._render_index(result, manifest)
        manifest.write()
        return manifest

    def _render_static_plots(
        self, result: ADTMPCMissionResult, manifest: ADTMPCMissionManifest
    ) -> None:
        fig = plt.figure(figsize=(8.5, 7))
        ax = fig.add_subplot(111, projection="3d")
        for chaser_index, (chaser_id, states) in enumerate(result.actual_trajectories.items()):
            positions = np.asarray(states, dtype=float)[:, :3]
            color = chaser_color(chaser_index)
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], color=color, label=f"{chaser_id} actual")
            ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], s=18, color=color)
            nominal_labeled = False
            for nominal in result.nominal_trajectories.get(chaser_id, []):
                nom = np.asarray(nominal, dtype=float)
                if nom.size:
                    ax.plot(
                        nom[:, 0],
                        nom[:, 1],
                        nom[:, 2],
                        linestyle=":",
                        color=color,
                        alpha=0.25,
                        label=f"{chaser_id} nominal horizons" if not nominal_labeled else None,
                    )
                    nominal_labeled = True
        frame_index = max(0, len(result.anom_history) - 1)
        self._plot_rendezvous_sphere(ax, result)
        self._plot_target_cylinder(ax, result, frame_index=frame_index)
        self._plot_docking_points(ax, result, frame_index=frame_index)
        self._plot_phase_switch_markers(ax, result)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend(fontsize=8)
        manifest.add(
            self._save_matplotlib(fig, "actual_nominal_trajectories.png"),
            "Actual/nominal trajectories with rotating cylinder target, rendezvous sphere, docking points, and phase switch markers.",
        )

        self._line_plot(
            result,
            filename="tube_geometry.png",
            ylabel="tube position radius",
            values_by_label=self._tube_values(result),
            manifest=manifest,
        )
        self._line_plot(
            result,
            filename="control_effort.png",
            ylabel="control norm",
            values_by_label={
                chaser_id: np.linalg.norm(np.asarray(values, dtype=float), axis=1).tolist()
                if len(values)
                else []
                for chaser_id, values in result.controls.items()
            },
            manifest=manifest,
            input_aligned=True,
        )
        self._line_plot(
            result,
            filename="rendezvous_margin.png",
            ylabel="physical rendezvous margin",
            values_by_label=result.rendezvous_margins,
            manifest=manifest,
            zero_line=True,
            phase_markers=True,
        )
        self._line_plot(
            result,
            filename="docking_cylinder_margin.png",
            ylabel="physical docking cylinder union margin",
            values_by_label=result.docking_cylinder_margins,
            manifest=manifest,
            zero_line=True,
            phase_markers=True,
        )
        self._line_plot(
            result,
            filename="tightened_rendezvous_margin.png",
            ylabel="tube-tightened rendezvous margin",
            values_by_label=result.tightened_rendezvous_margins,
            manifest=manifest,
            zero_line=True,
            phase_markers=True,
        )
        self._line_plot(
            result,
            filename="tightened_docking_cylinder_margin.png",
            ylabel="tube-tightened docking cylinder union margin",
            values_by_label=result.tightened_docking_cylinder_margins,
            manifest=manifest,
            zero_line=True,
            phase_markers=True,
        )
        self._line_plot(
            result,
            filename="active_target_margin.png",
            ylabel="active target safety margin",
            values_by_label=result.active_target_margins,
            manifest=manifest,
            zero_line=True,
            phase_markers=True,
        )
        self._plot_smid_widths(result, manifest)
        self._plot_parameter_estimates(result, manifest)
        self._target_plot(
            result,
            manifest,
            filename="target_attitude.png",
            values=result.target_attitude_history,
            ylabel="Euler angle (rad)",
            labels=["roll", "pitch", "yaw"],
        )
        self._target_plot(
            result,
            manifest,
            filename="target_angular_velocity.png",
            values=result.target_angular_velocity_history,
            ylabel="Euler angle rate (rad/rad anomaly)",
            labels=["roll rate", "pitch rate", "yaw rate"],
        )
        self._plot_spacing(result, manifest)

    def _line_plot(
        self,
        result: ADTMPCMissionResult,
        *,
        filename: str,
        ylabel: str,
        values_by_label: Dict[str, Sequence[float]],
        manifest: ADTMPCMissionManifest,
        input_aligned: bool = False,
        zero_line: bool = False,
        phase_markers: bool = False,
    ) -> None:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        for label, values in values_by_label.items():
            y_values = np.asarray(values, dtype=float)
            if len(y_values) == 0:
                continue
            if input_aligned:
                x_values = np.asarray(result.anom_history[: len(y_values)], dtype=float)
            else:
                x_values = np.asarray(result.anom_history[: len(y_values)], dtype=float)
            chaser_ids = list(result.actual_trajectories)
            color = chaser_color(chaser_ids.index(label)) if label in chaser_ids else "#596780"
            ax.plot(x_values, y_values, label=label, color=color,
                    linestyle="--" if label == "max_tube" else "-")
        if zero_line:
            ax.axhline(0.0, color="black", linestyle="--", linewidth=0.9, alpha=0.75)
        if phase_markers:
            self._add_phase_switch_lines(ax, result)
        ax.set_xlabel("true anomaly")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
        manifest.add(self._save_matplotlib(fig, filename))

    def _add_phase_switch_lines(self, ax, result: ADTMPCMissionResult) -> None:
        for index in self._phase_switch_indices(result):
            if index < len(result.anom_history):
                ax.axvline(
                    result.anom_history[index],
                    color="tab:red",
                    linestyle=":",
                    linewidth=1.0,
                    alpha=0.8,
                )

    def _phase_switch_indices(self, result: ADTMPCMissionResult) -> List[int]:
        phases = result.sample_phase_history or []
        return [
            idx
            for idx in range(1, len(phases))
            if phases[idx] != phases[idx - 1]
        ]

    def _target_dimensions(self, result: ADTMPCMissionResult) -> Dict[str, float]:
        target = result.target
        radius = float(target.get("radius", target.get("target_radius", 0.0)))
        height = float(target.get("height", target.get("target_height", 0.0)))
        half_length = float(target.get("half_length", height / 2.0))
        if height <= 0.0:
            height = 2.0 * half_length
        rendezvous_radius = float(
            target.get(
                "rendezvous_sphere_radius",
                target.get(
                    "rendezvous_radius",
                    np.sqrt(radius**2 + half_length**2),
                ),
            )
        )
        return {
            "radius": radius,
            "height": height,
            "half_length": half_length,
            "rendezvous_radius": rendezvous_radius,
        }

    def _orientation_at_index(self, result: ADTMPCMissionResult, frame_index: int) -> np.ndarray:
        attitudes = np.asarray(result.target_attitude_history, dtype=float)
        if attitudes.size == 0:
            return np.zeros(3)
        return attitudes[min(frame_index, len(attitudes) - 1)]

    def _cylinder_mesh(
        self,
        result: ADTMPCMissionResult,
        frame_index: int,
        *,
        theta_count: int = 48,
        z_count: int = 12,
    ):
        dims = self._target_dimensions(result)
        theta = np.linspace(0.0, 2.0 * np.pi, theta_count)
        z_values = np.linspace(-dims["half_length"], dims["half_length"], z_count)
        theta_grid, z_grid = np.meshgrid(theta, z_values)
        points = np.vstack(
            (
                dims["radius"] * np.cos(theta_grid).ravel(),
                dims["radius"] * np.sin(theta_grid).ravel(),
                z_grid.ravel(),
            )
        )
        rotation = tait_bryan_to_rotation_matrix(self._orientation_at_index(result, frame_index))
        rotated = rotation @ points
        x = rotated[0].reshape(theta_grid.shape)
        y = rotated[1].reshape(theta_grid.shape)
        z = rotated[2].reshape(theta_grid.shape)
        return x, y, z

    def _sphere_mesh(self, result: ADTMPCMissionResult, *, count: int = 36):
        radius = self._target_dimensions(result)["rendezvous_radius"]
        theta = np.linspace(0.0, 2.0 * np.pi, count)
        phi = np.linspace(0.0, np.pi, count // 2)
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        x = radius * np.cos(theta_grid) * np.sin(phi_grid)
        y = radius * np.sin(theta_grid) * np.sin(phi_grid)
        z = radius * np.cos(phi_grid)
        return x, y, z

    def _offset_sphere_mesh(
        self, center: np.ndarray, radius: float, *, count: int = 18
    ):
        """Return a Plotly-ready sphere mesh centered at ``center``."""
        theta = np.linspace(0.0, 2.0 * np.pi, count)
        phi = np.linspace(0.0, np.pi, max(count // 2, 4))
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        center = np.asarray(center, dtype=float)
        x = center[0] + radius * np.cos(theta_grid) * np.sin(phi_grid)
        y = center[1] + radius * np.sin(theta_grid) * np.sin(phi_grid)
        z = center[2] + radius * np.cos(phi_grid)
        return x, y, z

    def _docking_points_lvlh(
        self, result: ADTMPCMissionResult, frame_index: int
    ) -> Dict[str, np.ndarray]:
        orientation = self._orientation_at_index(result, frame_index)
        return {
            chaser_id: rotating_docking_point(np.asarray(point, dtype=float), orientation)
            for chaser_id, point in result.docking_points.items()
        }

    def _plot_target_cylinder(
        self,
        ax,
        result: ADTMPCMissionResult,
        frame_index: int,
        *,
        add_label: bool = True,
    ):
        x, y, z = self._cylinder_mesh(result, frame_index)
        surface = ax.plot_surface(
            x,
            y,
            z,
            color="tab:gray",
            alpha=0.35,
            linewidth=0,
            shade=True,
        )
        if add_label:
            ax.plot(
                [],
                [],
                [],
                color="tab:gray",
                linewidth=6,
                label="rotating target cylinder",
            )
        return surface

    def _plot_rendezvous_sphere(self, ax, result: ADTMPCMissionResult) -> None:
        x, y, z = self._sphere_mesh(result)
        ax.plot_wireframe(x, y, z, color="tab:blue", alpha=0.16, linewidth=0.45)
        ax.plot([], [], [], color="tab:blue", alpha=0.45, label="rendezvous sphere")

    def _plot_docking_points(
        self, ax, result: ADTMPCMissionResult, frame_index: int
    ) -> None:
        points = self._docking_points_lvlh(result, frame_index)
        if not points:
            return
        arr = np.asarray(list(points.values()), dtype=float)
        ax.scatter(
            arr[:, 0],
            arr[:, 1],
            arr[:, 2],
            color="tab:red",
            marker="D",
            s=36,
            label="docking points",
        )

    def _plot_phase_switch_markers(self, ax, result: ADTMPCMissionResult) -> None:
        switch_indices = self._phase_switch_indices(result)
        if not switch_indices:
            return
        label_used = False
        for chaser_id, states in result.actual_trajectories.items():
            positions = np.asarray(states, dtype=float)[:, :3]
            for index in switch_indices:
                if index >= len(positions):
                    continue
                ax.scatter(
                    positions[index, 0],
                    positions[index, 1],
                    positions[index, 2],
                    color="tab:red",
                    marker="x",
                    s=50,
                    label="rendezvous-to-docking switch" if not label_used else None,
                )
                label_used = True

    def _tube_values(self, result: ADTMPCMissionResult) -> Dict[str, Sequence[float]]:
        values = {"max_tube": result.tube_radius_history}
        for chaser_id, profiles in result.tube_profiles.items():
            values[chaser_id] = [
                float(profile.get("max_position_radius", 0.0)) for profile in profiles
            ]
        return values

    def _plot_smid_widths(
        self, result: ADTMPCMissionResult, manifest: ADTMPCMissionManifest
    ) -> None:
        keys = ["eccentricity", "alpha", "beta"]
        fig, axes = plt.subplots(len(keys), 1, figsize=(9, 8), sharex=True)
        for axis, key in zip(axes, keys):
            widths = [
                feasible[key][1] - feasible[key][0]
                for feasible in result.feasible_set_history
            ]
            axis.plot(result.anom_history[: len(widths)], widths, label=key)
            axis.set_ylabel(f"{key} width")
            axis.grid(True, alpha=0.25)
        axes[-1].set_xlabel("true anomaly")
        manifest.add(self._save_matplotlib(fig, "smid_fss_widths.png"))

    def _plot_parameter_estimates(
        self, result: ADTMPCMissionResult, manifest: ADTMPCMissionManifest
    ) -> None:
        keys = ["eccentricity", "alpha", "beta"]
        fig, axes = plt.subplots(len(keys), 1, figsize=(9, 8), sharex=True)
        for axis, key in zip(axes, keys):
            count = min(
                len(result.parameter_estimate_history),
                len(result.feasible_set_history),
                len(result.anom_history),
            )
            times = result.anom_history[:count]
            values = [
                estimates[key] for estimates in result.parameter_estimate_history
            ][:count]
            lower = [feasible[key][0] for feasible in result.feasible_set_history[:count]]
            upper = [feasible[key][1] for feasible in result.feasible_set_history[:count]]
            axis.fill_between(times, lower, upper, alpha=0.16, label="feasible set")
            axis.plot(times, values, label="belief")
            axis.axhline(result.truth[key], color="black", linestyle="--", linewidth=1.0, label="truth")
            axis.set_ylabel(key)
            axis.grid(True, alpha=0.25)
        axes[-1].set_xlabel("true anomaly")
        axes[0].legend(fontsize=8)
        manifest.add(self._save_matplotlib(fig, "parameter_estimates_vs_truth.png"))

    def _target_plot(
        self,
        result: ADTMPCMissionResult,
        manifest: ADTMPCMissionManifest,
        *,
        filename: str,
        values: Sequence[Sequence[float]],
        ylabel: str,
        labels: Optional[Sequence[str]] = None,
    ) -> None:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        arr = np.asarray(values, dtype=float)
        labels = list(labels or ["x", "y", "z"])
        for idx in range(min(arr.shape[1], 3)):
            ax.plot(result.anom_history[: len(arr)], arr[:, idx], label=labels[idx])
        ax.set_xlabel("true anomaly")
        ax.set_ylabel(ylabel)
        if "prescribed" in str(result.metadata.get("target_attitude_model", "")).lower():
            ax.set_title("Prescribed Euler-angle target motion in true anomaly")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        manifest.add(self._save_matplotlib(fig, filename))

    def _plot_spacing(
        self, result: ADTMPCMissionResult, manifest: ADTMPCMissionManifest
    ) -> None:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        if result.pairwise_spacing and result.pairwise_spacing[0]:
            keys = sorted(result.pairwise_spacing[0].keys())
            for key in keys:
                values = [entry.get(key, np.nan) for entry in result.pairwise_spacing]
                ax.plot(result.anom_history[: len(values)], values, label=key)
        ax.set_xlabel("true anomaly")
        ax.set_ylabel("pairwise separation")
        ax.grid(True, alpha=0.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
        manifest.add(self._save_matplotlib(fig, "multi_chaser_spacing.png"))

    def _render_interactive_html(
        self, result: ADTMPCMissionResult, manifest: ADTMPCMissionManifest
    ) -> None:
        fig = go.Figure()
        for chaser_index, (chaser_id, states) in enumerate(result.actual_trajectories.items()):
            color = chaser_color(chaser_index)
            positions = np.asarray(states, dtype=float)[:, :3]
            fig.add_trace(
                go.Scatter3d(
                    x=positions[:, 0],
                    y=positions[:, 1],
                    z=positions[:, 2],
                    mode="lines+markers",
                    name=f"{chaser_id} actual",
                    line=dict(color=color, width=4),
                    marker=dict(color=color, size=2),
                )
            )
            nominal_labeled = False
            for idx, nominal in enumerate(result.nominal_trajectories.get(chaser_id, [])):
                nom = np.asarray(nominal, dtype=float)
                if nom.size:
                    fig.add_trace(
                        go.Scatter3d(
                            x=nom[:, 0],
                            y=nom[:, 1],
                            z=nom[:, 2],
                            mode="lines",
                            name=f"{chaser_id} nominal horizons" if not nominal_labeled else f"{chaser_id} nominal",
                            showlegend=not nominal_labeled,
                            opacity=0.22,
                            line=dict(color=color, dash="dot"),
                        )
                    )
                    nominal_labeled = True
        frame_index = max(0, len(result.anom_history) - 1)
        cx, cy, cz = self._cylinder_mesh(result, frame_index)
        fig.add_trace(
            go.Surface(
                x=cx,
                y=cy,
                z=cz,
                name="rotating target cylinder",
                showscale=False,
                opacity=0.35,
                colorscale=[[0, "rgb(115,115,115)"], [1, "rgb(115,115,115)"]],
            )
        )
        sx, sy, sz = self._sphere_mesh(result)
        fig.add_trace(
            go.Surface(
                x=sx,
                y=sy,
                z=sz,
                name="rendezvous sphere",
                showscale=False,
                opacity=0.12,
                colorscale=[[0, "rgb(70,130,180)"], [1, "rgb(70,130,180)"]],
            )
        )
        docking_points = self._docking_points_lvlh(result, frame_index)
        if docking_points:
            dock_arr = np.asarray(list(docking_points.values()), dtype=float)
            fig.add_trace(
                go.Scatter3d(
                    x=dock_arr[:, 0],
                    y=dock_arr[:, 1],
                    z=dock_arr[:, 2],
                    mode="markers",
                    name="docking points",
                    marker=dict(size=5, color="red", symbol="diamond"),
                )
            )
        for switch_index in self._phase_switch_indices(result):
            xs, ys, zs = [], [], []
            for states in result.actual_trajectories.values():
                positions = np.asarray(states, dtype=float)[:, :3]
                if switch_index < len(positions):
                    xs.append(positions[switch_index, 0])
                    ys.append(positions[switch_index, 1])
                    zs.append(positions[switch_index, 2])
            if xs:
                fig.add_trace(
                    go.Scatter3d(
                        x=xs,
                        y=ys,
                        z=zs,
                        mode="markers",
                        name="rendezvous-to-docking switch",
                        marker=dict(size=6, color="red", symbol="x"),
                    )
                )
        fig.update_layout(
            title=f"ORBEX-A ADTMPC Mission ({result.mission}, {result.approximation})",
            scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        path = self.output_dir / "trajectory.html"
        fig.write_html(path)
        manifest.add(path, "Interactive trajectory with cylinder target, rendezvous sphere, docking points, and phase switch markers.")

    def _render_interactive_diagnostics(
        self, result: ADTMPCMissionResult, manifest: ADTMPCMissionManifest
    ) -> None:
        """Save an interactive Plotly version of the main safety/control plots."""
        fig = make_subplots(
            rows=5,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.035,
            subplot_titles=(
                "Tube radius",
                "Control norm",
                "Physical target margins",
                "Tube-tightened target margins",
                "Pairwise spacing",
            ),
        )
        anom = np.asarray(result.anom_history, dtype=float)

        for label, values in self._tube_values(result).items():
            fig.add_trace(
                go.Scatter(x=anom[: len(values)], y=list(values), mode="lines", name=f"tube {label}"),
                row=1,
                col=1,
            )
        for chaser_id, values in result.controls.items():
            norms = (
                np.linalg.norm(np.asarray(values, dtype=float), axis=1).tolist()
                if len(values)
                else []
            )
            fig.add_trace(
                go.Scatter(x=anom[: len(norms)], y=norms, mode="lines", name=f"{chaser_id} control"),
                row=2,
                col=1,
            )
        for prefix, margins in (
            ("physical rendezvous", result.rendezvous_margins),
            ("physical cylinder", result.docking_cylinder_margins),
            ("active physical", result.active_target_margins),
        ):
            for chaser_id, values in margins.items():
                fig.add_trace(
                    go.Scatter(x=anom[: len(values)], y=values, mode="lines", name=f"{prefix} {chaser_id}"),
                    row=3,
                    col=1,
                )
        for prefix, margins in (
            ("tightened rendezvous", result.tightened_rendezvous_margins),
            ("tightened cylinder", result.tightened_docking_cylinder_margins),
        ):
            for chaser_id, values in margins.items():
                fig.add_trace(
                    go.Scatter(x=anom[: len(values)], y=values, mode="lines", name=f"{prefix} {chaser_id}"),
                    row=4,
                    col=1,
                )
        spacing_keys = sorted({key for entry in result.pairwise_spacing for key in entry})
        for key in spacing_keys:
            values = [entry.get(key, np.nan) for entry in result.pairwise_spacing]
            fig.add_trace(
                go.Scatter(x=anom[: len(values)], y=values, mode="lines", name=f"spacing {key}"),
                row=5,
                col=1,
            )
        for row in (3, 4):
            fig.add_hline(y=0.0, line_dash="dash", line_color="black", opacity=0.55, row=row, col=1)
        for index in self._phase_switch_indices(result):
            if index < len(anom):
                for row in range(1, 6):
                    fig.add_vline(
                        x=float(anom[index]),
                        line_dash="dot",
                        line_color="red",
                        opacity=0.45,
                        row=row,
                        col=1,
                    )
        fig.update_layout(
            title=f"ADTMPC diagnostics ({result.mission}, {result.approximation})",
            height=1050,
            hovermode="x unified",
            margin=dict(l=40, r=20, t=70, b=40),
        )
        fig.update_xaxes(title_text="true anomaly", row=5, col=1)
        path = self.output_dir / "diagnostics.html"
        fig.write_html(path)
        manifest.add(path, "Interactive safety, tube, control, and spacing diagnostics.")

    def _render_tube_trajectory_html(
        self, result: ADTMPCMissionResult, manifest: ADTMPCMissionManifest
    ) -> None:
        """Save an interactive 3D trajectory view with sparse tube envelopes."""
        fig = go.Figure()
        for chaser_index, (chaser_id, states) in enumerate(result.actual_trajectories.items()):
            color = chaser_color(chaser_index)
            positions = np.asarray(states, dtype=float)[:, :3]
            fig.add_trace(
                go.Scatter3d(
                    x=positions[:, 0],
                    y=positions[:, 1],
                    z=positions[:, 2],
                    mode="lines",
                    name=f"{chaser_id} actual",
                    line=dict(width=5, color=color),
                )
            )
            nominal_labeled = False
            for nominal in result.nominal_trajectories.get(chaser_id, []):
                nom = np.asarray(nominal, dtype=float)
                if nom.size:
                    fig.add_trace(
                        go.Scatter3d(
                            x=nom[:, 0],
                            y=nom[:, 1],
                            z=nom[:, 2],
                            mode="lines",
                            name=f"{chaser_id} nominal horizons" if not nominal_labeled else f"{chaser_id} nominal",
                            showlegend=not nominal_labeled,
                            opacity=0.18,
                            line=dict(width=2, dash="dot", color=color),
                        )
                    )
                    nominal_labeled = True
            if len(positions) and result.tube_radius_history:
                sample_count = min(12, len(positions), len(result.tube_radius_history))
                sample_indices = np.unique(
                    np.linspace(0, min(len(positions), len(result.tube_radius_history)) - 1, sample_count).astype(int)
                )
                for surface_idx, idx in enumerate(sample_indices):
                    radius = float(result.tube_radius_history[idx])
                    if radius <= 1.0e-9:
                        continue
                    sx, sy, sz = self._offset_sphere_mesh(positions[idx], radius)
                    fig.add_trace(
                        go.Surface(
                            x=sx,
                            y=sy,
                            z=sz,
                            name=f"{chaser_id} tube envelope" if surface_idx == 0 else f"{chaser_id} tube",
                            showscale=False,
                            opacity=0.08,
                            colorscale=[[0, color], [1, color]],
                        )
                    )
        frame_index = max(0, len(result.anom_history) - 1)
        cx, cy, cz = self._cylinder_mesh(result, frame_index)
        fig.add_trace(
            go.Surface(
                x=cx,
                y=cy,
                z=cz,
                name="rotating target cylinder",
                showscale=False,
                opacity=0.32,
                colorscale=[[0, "rgb(115,115,115)"], [1, "rgb(115,115,115)"]],
            )
        )
        sx, sy, sz = self._sphere_mesh(result)
        fig.add_trace(
            go.Surface(
                x=sx,
                y=sy,
                z=sz,
                name="rendezvous sphere",
                showscale=False,
                opacity=0.08,
                colorscale=[[0, "rgb(70,130,180)"], [1, "rgb(70,130,180)"]],
            )
        )
        docking_points = self._docking_points_lvlh(result, frame_index)
        if docking_points:
            dock_arr = np.asarray(list(docking_points.values()), dtype=float)
            fig.add_trace(
                go.Scatter3d(
                    x=dock_arr[:, 0],
                    y=dock_arr[:, 1],
                    z=dock_arr[:, 2],
                    mode="markers",
                    name="docking points",
                    marker=dict(size=5, color="red", symbol="diamond"),
                )
            )
        fig.update_layout(
            title=f"ADTMPC tube trajectory ({result.mission}, {result.approximation})",
            scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        path = self.output_dir / "tube_trajectory.html"
        fig.write_html(path)
        manifest.add(path, "Interactive trajectory with actual path, nominal horizons, target geometry, and sparse tube envelopes.")

    def _render_video(
        self, result: ADTMPCMissionResult, manifest: ADTMPCMissionManifest
    ) -> None:
        path = self.output_dir / "trajectory.mp4"
        if shutil.which("ffmpeg") is None:
            reason = "FFmpeg is unavailable; no current MP4 was generated. The trajectory GIF remains available."
            manifest.skipped_artifacts[path.name] = reason
            logger.warning(reason)
            return

        positions_by_id = {
            chaser_id: np.asarray(states, dtype=float)[:, :3]
            for chaser_id, states in result.actual_trajectories.items()
        }
        all_positions = np.vstack(list(positions_by_id.values()))
        mins = all_positions.min(axis=0)
        maxs = all_positions.max(axis=0)
        center = (mins + maxs) / 2.0
        target_radius = self._target_dimensions(result)["rendezvous_radius"]
        radius = max(float(np.max(maxs - mins)) / 2.0, target_radius * 1.2, 1.0)
        max_frames = max(len(values) for values in positions_by_id.values())

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        lines: Dict[str, object] = {}
        points: Dict[str, object] = {}
        for chaser_index, chaser_id in enumerate(positions_by_id):
            color = chaser_color(chaser_index)
            (line,) = ax.plot([], [], [], label=chaser_id, color=color)
            (point,) = ax.plot([], [], [], marker="o", color=color)
            lines[chaser_id] = line
            points[chaser_id] = point
        self._plot_rendezvous_sphere(ax, result)
        target_surface = [self._plot_target_cylinder(ax, result, frame_index=0)]
        (dock_points_artist,) = ax.plot(
            [],
            [],
            [],
            linestyle="",
            marker="D",
            color="tab:red",
            label="docking points",
        )
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend(fontsize=8)

        def update(frame: int):
            if target_surface[0] is not None:
                target_surface[0].remove()
            target_surface[0] = self._plot_target_cylinder(
                ax, result, frame_index=frame, add_label=False
            )
            docking_points = self._docking_points_lvlh(result, frame)
            if docking_points:
                dock_arr = np.asarray(list(docking_points.values()), dtype=float)
                dock_points_artist.set_data(dock_arr[:, 0], dock_arr[:, 1])
                dock_points_artist.set_3d_properties(dock_arr[:, 2])
            for chaser_id, positions in positions_by_id.items():
                idx = min(frame, len(positions) - 1)
                segment = positions[: idx + 1]
                lines[chaser_id].set_data(segment[:, 0], segment[:, 1])
                lines[chaser_id].set_3d_properties(segment[:, 2])
                point = positions[idx]
                points[chaser_id].set_data([point[0]], [point[1]])
                points[chaser_id].set_3d_properties([point[2]])
            return list(lines.values()) + list(points.values()) + [dock_points_artist]

        animation = FuncAnimation(fig, update, frames=max_frames, interval=125)
        animation.save(path, writer=FFMpegWriter(fps=self.fps), dpi=130)
        plt.close(fig)
        manifest.add(path)

    def _render_gif(
        self,
        result: ADTMPCMissionResult,
        manifest: ADTMPCMissionManifest,
        *,
        speed_multiplier: float = 2.0,
    ) -> None:
        """Render a compact GIF trajectory preview using the same scene as MP4."""
        path = self.output_dir / "trajectory.gif"
        positions_by_id = {
            chaser_id: np.asarray(states, dtype=float)[:, :3]
            for chaser_id, states in result.actual_trajectories.items()
        }
        if not positions_by_id:
            manifest.skipped_artifacts[path.name] = "No recorded trajectory samples are available."
            return

        all_positions = np.vstack(list(positions_by_id.values()))
        mins = all_positions.min(axis=0)
        maxs = all_positions.max(axis=0)
        center = (mins + maxs) / 2.0
        target_radius = self._target_dimensions(result)["rendezvous_radius"]
        radius = max(float(np.max(maxs - mins)) / 2.0, target_radius * 1.2, 1.0)
        max_frames = max(len(values) for values in positions_by_id.values())
        frame_step = max(1, int(np.ceil(max_frames / 80)))
        frames = list(range(0, max_frames, frame_step))
        if frames[-1] != max_frames - 1:
            frames.append(max_frames - 1)

        fig = plt.figure(figsize=(6.5, 5.6))
        ax = fig.add_subplot(111, projection="3d")
        lines: Dict[str, object] = {}
        points: Dict[str, object] = {}
        for chaser_index, chaser_id in enumerate(positions_by_id):
            color = chaser_color(chaser_index)
            (line,) = ax.plot([], [], [], label=chaser_id, color=color)
            (point,) = ax.plot([], [], [], marker="o", color=color)
            lines[chaser_id] = line
            points[chaser_id] = point
        self._plot_rendezvous_sphere(ax, result)
        target_surface = [self._plot_target_cylinder(ax, result, frame_index=0)]
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend(fontsize=7)

        def update(frame: int):
            if target_surface[0] is not None:
                target_surface[0].remove()
            target_surface[0] = self._plot_target_cylinder(
                ax, result, frame_index=frame, add_label=False
            )
            for chaser_id, positions in positions_by_id.items():
                idx = min(frame, len(positions) - 1)
                segment = positions[: idx + 1]
                lines[chaser_id].set_data(segment[:, 0], segment[:, 1])
                lines[chaser_id].set_3d_properties(segment[:, 2])
                point = positions[idx]
                points[chaser_id].set_data([point[0]], [point[1]])
                points[chaser_id].set_3d_properties([point[2]])
            return list(lines.values()) + list(points.values())

        fps = max(1, int(round(self.fps * speed_multiplier)))
        animation = FuncAnimation(fig, update, frames=frames, interval=1000 / fps)
        animation.save(path, writer=PillowWriter(fps=fps), dpi=100)
        plt.close(fig)
        manifest.add(path, f"Trajectory GIF preview at {speed_multiplier:g}x playback speed.")

    def _render_index(
        self, result: ADTMPCMissionResult, manifest: ADTMPCMissionManifest
    ) -> None:
        entry = {
            "mission": result.mission, "approximation": result.approximation,
            "backend": result.solver_backend, "success": result.success,
            "mission_complete": manifest.mission_complete,
            "sample_count": manifest.sample_count, "chaser_count": manifest.chaser_count,
            "message": result.message, "relative_dir": ".",
            "artifacts": [{"path": path.name,
                           "label": manifest.artifact_labels.get(path.name, path.stem.replace("_", " "))}
                          for path in manifest.artifacts if path.parent == self.output_dir],
            "skipped_artifacts": manifest.skipped_artifacts,
            **_raw_data_reference(manifest.data_file, self.output_dir),
        }
        manifest.add(MissionGallery(self.output_dir).render([entry]))

    def _save_matplotlib(self, fig, filename: str) -> Path:
        path = self.output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path


class OrbexaDemo:
    """Public entrypoint used by the ORBEX-A CLI."""

    def __init__(self, config: DemoConfig):
        self.config = config

    def run(self) -> List[ADTMPCMissionManifest]:
        if self.config.from_data and self.config.clean_generated:
            raise ValueError("--from-data cannot be combined with --clean-generated")
        session = MissionSession.select(
            self.config.output_dir, self.config.data_dir,
            session_id=self.config.session_id, from_data=self.config.from_data,
            mission=self.config.mission, solver=self.config.primary_solver,
        )
        output_session_dir, data_session_dir = session.output_dir, session.data_dir
        approximations = ["nonlinear", "linearized"] if self.config.run_linearized else ["nonlinear"]
        session.preflight([(mission, approximation) for mission in self._missions() for approximation in approximations])
        gallery_entries = self._read_gallery_catalog(output_session_dir)
        if self.config.from_data:
            # Read every requested family before changing any rendered artifacts or notes.
            self._saved_results = {}
            for mission in self._missions():
                for approximation in approximations:
                    result = load_adtmpc_mission_result(data_session_dir / mission / approximation)
                    if result.mission != mission or result.approximation != approximation:
                        raise ValueError(f"Saved data does not match {mission}/{approximation}")
                    self._saved_results[(mission, approximation)] = result
        else:
            self._mission_config = SimulationConfig.load(self.config.config_path)
            if self.config.clean_generated:
                clean_generated_outputs(session.output_root, session.data_root, scenario_path=self.config.config_path)
            session.create()
        self._write_run_readme(output_session_dir, data_session_dir, session.name)

        manifests: List[ADTMPCMissionManifest] = []
        for mission in self._missions():
            manifests.append(
                self._run_or_render(
                    mission=mission,
                    approximation="nonlinear",
                    solver_backend=self.config.primary_solver,
                    family_dir=output_session_dir / mission / "nonlinear",
                    data_family_dir=data_session_dir / mission / "nonlinear",
                )
            )
            if self.config.run_linearized:
                manifests.append(
                    self._run_or_render(
                        mission=mission,
                        approximation="linearized",
                        solver_backend=self.config.secondary_solver,
                        family_dir=output_session_dir / mission / "linearized",
                        data_family_dir=data_session_dir / mission / "linearized",
                    )
                )
        for manifest in manifests:
            entry = self._gallery_entry(manifest, output_session_dir)
            gallery_entries[entry["relative_dir"]] = entry
        entries = list(gallery_entries.values())
        MissionGallery(output_session_dir).render(entries)
        if not save_data(output_session_dir / "gallery.json", {
            "schema": "orbexa.mission.gallery.v1", "entries": entries,
        }):
            raise OSError("Could not save mission gallery catalog")
        self._link_session_demo_gif(output_session_dir)
        if not self.config.from_data:
            session.publish()
            failures = [manifest for manifest in manifests
                        if self._requires_success(manifest.approximation, manifest.solver_backend)
                        and not manifest.success]
            if failures:
                failed = failures[0]
                raise RuntimeError(
                    f"{failed.mission}/{failed.approximation}/{failed.solver_backend} did not complete: "
                    f"{failed.message or 'mission_complete=false'}. "
                    f"Inspect the saved gallery at {output_session_dir / 'index.html'}"
                )
        return manifests

    @staticmethod
    def _read_gallery_catalog(output_session_dir: Path) -> Dict[str, Dict]:
        path = output_session_dir / "gallery.json"
        entries = []
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("schema") != "orbexa.mission.gallery.v1":
                raise ValueError(f"Refusing to replace an unrecognized gallery catalog: {path}")
            entries = payload.get("entries")
            if not isinstance(entries, list):
                raise ValueError(f"Invalid mission gallery catalog: {path}")
        catalog = {}
        families = ("single/nonlinear", "single/linearized", "multi/nonlinear", "multi/linearized")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"Invalid mission gallery entry in {path}")
            relative = entry.get("relative_dir")
            if relative not in families:
                raise ValueError(f"Invalid mission gallery family in {path}")
            MissionSession._check_path(output_session_dir / relative, output_session_dir)
            catalog[relative] = entry
        # Older sessions have family manifests but no session catalog. Retain
        # their views during a partial render without loading unrequested data.
        for relative in families:
            if relative in catalog:
                continue
            family = output_session_dir / relative
            manifest_path = family / "manifest.json"
            MissionSession._check_path(manifest_path, output_session_dir)
            if not manifest_path.is_file():
                continue
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if payload.get("schema") != "orbexa.adtmpc_mission.manifest.v1":
                continue
            artifacts = []
            for value in payload.get("artifacts", []):
                # Legacy manifests stored absolute paths; maintained family
                # artifacts are flat, so bind their filenames to this session.
                artifact = family / Path(value).name
                MissionSession._check_path(artifact, output_session_dir)
                if artifact.is_file():
                    artifacts.append({"path": artifact.relative_to(output_session_dir).as_posix(),
                                      "label": payload.get("artifact_labels", {}).get(artifact.name, artifact.stem.replace("_", " "))})
            mission, approximation = relative.split("/")
            catalog[relative] = {
                "mission": mission, "approximation": approximation,
                "backend": payload.get("solver_backend", "unspecified"),
                "success": payload.get("success"), "mission_complete": payload.get("mission_complete"),
                "sample_count": payload.get("sample_count"), "chaser_count": payload.get("chaser_count"),
                "message": payload.get("message", "Outcome not recorded in this older manifest."),
                "relative_dir": relative, "artifacts": artifacts,
                "skipped_artifacts": payload.get("skipped_artifacts", {}),
            }
        return catalog

    @staticmethod
    def _gallery_entry(manifest: ADTMPCMissionManifest, root: Path) -> Dict:
        artifacts = [{"path": path.relative_to(root).as_posix(),
                      "label": manifest.artifact_labels.get(path.name, path.stem.replace("_", " "))}
                     for path in manifest.artifacts if path.is_relative_to(root)]
        return {
            "mission": manifest.mission, "approximation": manifest.approximation,
            "backend": manifest.solver_backend, "success": manifest.success,
            "mission_complete": manifest.mission_complete, "message": manifest.message,
            "sample_count": manifest.sample_count, "chaser_count": manifest.chaser_count,
            "relative_dir": manifest.output_dir.relative_to(root).as_posix(),
            "artifacts": artifacts, "skipped_artifacts": manifest.skipped_artifacts,
            **_raw_data_reference(manifest.data_file, root),
        }

    def _write_run_readme(
        self, output_session_dir: Path, data_session_dir: Path, session_id: str
    ) -> None:
        """Create editable run notes in both the rendered and raw-data session roots."""
        content = self._run_readme_template(session_id)
        for directory in (output_session_dir, data_session_dir):
            directory.mkdir(parents=True, exist_ok=True)
            readme = directory / "README.md"
            if not readme.exists():
                readme.write_text(content, encoding="utf-8")

    def _run_readme_template(self, session_id: str) -> str:
        missions = ", ".join(self._missions())
        return f"""# ADTMPC Run Notes

![Multi-agent ADTMPC demo](demo.gif)

Session: `{session_id}`
Created: `{datetime.now().isoformat(timespec="seconds")}`

## Run Summary

- Mission(s): `{missions}`
- Primary solver: `{self.config.primary_solver}`
- Linearized comparison: `{self.config.run_linearized}`
- Max nonlinear updates: `{self.config.steps}`
- Notes:

## Scenario Options

- Target rotation / attitude notes:
- Initial condition notes:
- Docking-point assignment notes:
- Tube / SMID settings:
- Solver settings:

## Results Checklist

- Mission completion:
- Active physical target margins nonnegative:
- Tube-tightened diagnostics reviewed:
- Pairwise spacing reviewed:
- Parameter belief behavior:

## Follow-Up Ideas

- Plots to inspect:
- Parameter or controller changes to try:
- Open questions:
"""

    def _link_session_demo_gif(self, output_session_dir: Path) -> None:
        """Expose the multi-agent nonlinear trajectory GIF at the session root."""
        source = output_session_dir / "multi" / "nonlinear" / "trajectory.gif"
        if not source.exists():
            return
        link = output_session_dir / "demo.gif"
        if link.is_symlink() or link.is_file():
            link.unlink()
        elif link.exists():
            logger.warning("Preserving existing directory at %s", link)
            return
        try:
            link.symlink_to(source.relative_to(output_session_dir))
        except OSError:
            shutil.copy2(source, link)

    def _missions(self) -> List[str]:
        if self.config.mission == "all":
            return ["single", "multi"]
        return [self.config.mission]

    def _run_or_render(
        self,
        *,
        mission: str,
        approximation: str,
        solver_backend: str,
        family_dir: Path,
        data_family_dir: Path,
    ) -> ADTMPCMissionManifest:
        if self.config.from_data:
            result = self._saved_results[(mission, approximation)]
            data_file = data_family_dir / "mission_data.json"
        else:
            runner_kwargs = {}
            run_steps = self.config.steps
            if approximation == "linearized":
                runner_kwargs = {"horizon_steps": 2, "smid_window": 2}
                if self.config.linearized_steps is not None:
                    run_steps = min(self.config.steps, int(self.config.linearized_steps))
            runner = ADTMPCMissionRunner(
                config=self._mission_config,
                solver_backend=solver_backend,
                approximation=approximation,
                **runner_kwargs,
            )
            result = runner.run(mission=mission, steps=run_steps)
            data_file = runner.save_result(result, data_family_dir)
        return ADTMPCMissionRenderer(family_dir, fps=self.config.fps).render(
            result,
            data_file=data_file,
        )

    def _requires_success(self, approximation: str, solver_backend: str) -> bool:
        return (
            self.config.require_primary_success
            and approximation == "nonlinear"
            and solver_backend == self.config.primary_solver
        )


def clean_generated_outputs(output_dir: Path, data_dir: Path, *, scenario_path: Optional[Path] = None) -> None:
    """Remove only the selected generated roots, rejecting source and overlapping paths."""
    roots = [Path(output_dir).expanduser().resolve(), Path(data_dir).expanduser().resolve()]
    checkout = Path(__file__).resolve().parents[3]
    protected = [checkout / name for name in ("src", "config", "paper", "assets", ".git")]
    for root in roots:
        if root.exists() and not root.is_dir():
            raise ValueError(f"Generated output root is not a directory: {root}")
        if scenario_path is not None and root in Path(scenario_path).expanduser().resolve().parents:
            raise ValueError(f"Refusing to clean the selected scenario YAML: {scenario_path}")
        if root == Path(root.anchor) or root == Path.home() or root == checkout or root in checkout.parents:
            raise ValueError(f"Refusing to clean protected folder: {root}")
        if (root / ".git").exists() or any(
            root == item or root in item.parents or item in root.parents for item in protected
        ):
            raise ValueError(f"Refusing to clean source folder: {root}")
    if roots[0] == roots[1] or roots[0] in roots[1].parents or roots[1] in roots[0].parents:
        raise ValueError("Generated output roots must not overlap")
    for root in roots:
        if root.is_dir():
            shutil.rmtree(root)


def main() -> int:
    from orbexa.cli.generate_demo import main as cli_main
    return cli_main()


DemoManifest = ADTMPCMissionManifest
MissionDemo = ADTMPCMissionResult


if __name__ == "__main__":
    raise SystemExit(main())
