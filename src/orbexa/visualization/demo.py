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

"""Render paper-system ORBEX-A mission artifacts."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

_mpl_cache_dir = Path(os.environ.get("MPLCONFIGDIR", "/tmp/orbexa-matplotlib"))
_mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache_dir))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.animation import FFMpegWriter, FuncAnimation

from orbexa.simulation.paper_system import (
    PaperSystemResult,
    PaperSystemRunner,
    load_paper_system_result,
)


@dataclass(frozen=True)
class DemoConfig:
    """Configuration for paper-system artifact generation."""

    output_dir: Path = Path("results/paper_system")
    data_dir: Path = Path("data/paper_system")
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


@dataclass
class PaperSystemManifest:
    """Summary of generated files for one mission/solver family."""

    output_dir: Path
    mission: str
    approximation: str
    solver_backend: str
    data_file: Optional[Path] = None
    artifacts: List[Path] = field(default_factory=list)

    def add(self, path: Path) -> Path:
        self.artifacts.append(path)
        return path

    def write(self) -> Path:
        path = self.output_dir / "manifest.json"
        payload = {
            "schema": "orbexa.paper_system.manifest.v1",
            "mission": self.mission,
            "approximation": self.approximation,
            "solver_backend": self.solver_backend,
            "data_file": None if self.data_file is None else str(self.data_file),
            "artifacts": [str(path) for path in self.artifacts],
            "required_entries": [
                "index.html",
                "trajectory.html",
                "trajectory.mp4",
                "mission_data.json",
                "actual_nominal_trajectories.png",
                "tube_geometry.png",
                "control_effort.png",
                "rendezvous_margin.png",
                "docking_cylinder_margin.png",
                "smid_fss_widths.png",
                "parameter_estimates_vs_truth.png",
                "target_attitude.png",
                "target_angular_velocity.png",
                "multi_chaser_spacing.png",
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.add(path)
        return path


class PaperSystemRenderer:
    """Create HTML, plots, and video for one saved paper-system result."""

    def __init__(self, output_dir: Path, *, fps: int = 8):
        self.output_dir = Path(output_dir)
        self.fps = int(fps)

    def render(
        self,
        result: PaperSystemResult,
        *,
        data_file: Optional[Path] = None,
    ) -> PaperSystemManifest:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = PaperSystemManifest(
            output_dir=self.output_dir,
            mission=result.mission,
            approximation=result.approximation,
            solver_backend=result.solver_backend,
            data_file=data_file,
        )
        if data_file is not None:
            manifest.add(Path(data_file))
        self._render_static_plots(result, manifest)
        self._render_interactive_html(result, manifest)
        self._render_video(result, manifest)
        self._render_index(result, manifest)
        manifest.write()
        return manifest

    def _render_static_plots(
        self, result: PaperSystemResult, manifest: PaperSystemManifest
    ) -> None:
        fig = plt.figure(figsize=(8.5, 7))
        ax = fig.add_subplot(111, projection="3d")
        for chaser_id, states in result.actual_trajectories.items():
            positions = np.asarray(states, dtype=float)[:, :3]
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label=f"{chaser_id} actual")
            ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], s=18)
            for nominal in result.nominal_trajectories.get(chaser_id, []):
                nom = np.asarray(nominal, dtype=float)
                if nom.size:
                    ax.plot(nom[:, 0], nom[:, 1], nom[:, 2], linestyle=":", alpha=0.45)
        ax.scatter([0], [0], [0], color="black", s=48, label="target")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend(fontsize=8)
        manifest.add(self._save_matplotlib(fig, "actual_nominal_trajectories.png"))

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
            ylabel="rendezvous margin",
            values_by_label=result.rendezvous_margins,
            manifest=manifest,
        )
        self._line_plot(
            result,
            filename="docking_cylinder_margin.png",
            ylabel="docking cylinder union margin",
            values_by_label=result.docking_cylinder_margins,
            manifest=manifest,
        )
        self._plot_smid_widths(result, manifest)
        self._plot_parameter_estimates(result, manifest)
        self._target_plot(
            result,
            manifest,
            filename="target_attitude.png",
            values=result.target_attitude_history,
            ylabel="attitude",
        )
        self._target_plot(
            result,
            manifest,
            filename="target_angular_velocity.png",
            values=result.target_angular_velocity_history,
            ylabel="angular velocity",
        )
        self._plot_spacing(result, manifest)

    def _line_plot(
        self,
        result: PaperSystemResult,
        *,
        filename: str,
        ylabel: str,
        values_by_label: Dict[str, Sequence[float]],
        manifest: PaperSystemManifest,
        input_aligned: bool = False,
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
            ax.plot(x_values, y_values, label=label)
        ax.set_xlabel("true anomaly")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
        manifest.add(self._save_matplotlib(fig, filename))

    def _tube_values(self, result: PaperSystemResult) -> Dict[str, Sequence[float]]:
        values = {"max_tube": result.tube_radius_history}
        for chaser_id, profiles in result.tube_profiles.items():
            values[chaser_id] = [
                float(profile.get("max_position_radius", 0.0)) for profile in profiles
            ]
        return values

    def _plot_smid_widths(
        self, result: PaperSystemResult, manifest: PaperSystemManifest
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
        self, result: PaperSystemResult, manifest: PaperSystemManifest
    ) -> None:
        keys = ["eccentricity", "alpha", "beta"]
        fig, axes = plt.subplots(len(keys), 1, figsize=(9, 8), sharex=True)
        for axis, key in zip(axes, keys):
            values = [
                estimates[key] for estimates in result.parameter_estimate_history
            ]
            axis.plot(result.anom_history[: len(values)], values, label="belief")
            axis.axhline(result.truth[key], color="black", linestyle="--", linewidth=1.0, label="truth")
            axis.set_ylabel(key)
            axis.grid(True, alpha=0.25)
        axes[-1].set_xlabel("true anomaly")
        axes[0].legend(fontsize=8)
        manifest.add(self._save_matplotlib(fig, "parameter_estimates_vs_truth.png"))

    def _target_plot(
        self,
        result: PaperSystemResult,
        manifest: PaperSystemManifest,
        *,
        filename: str,
        values: Sequence[Sequence[float]],
        ylabel: str,
    ) -> None:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        arr = np.asarray(values, dtype=float)
        labels = ["x", "y", "z"]
        for idx in range(min(arr.shape[1], 3)):
            ax.plot(result.anom_history[: len(arr)], arr[:, idx], label=labels[idx])
        ax.set_xlabel("true anomaly")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        manifest.add(self._save_matplotlib(fig, filename))

    def _plot_spacing(
        self, result: PaperSystemResult, manifest: PaperSystemManifest
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
        self, result: PaperSystemResult, manifest: PaperSystemManifest
    ) -> None:
        fig = go.Figure()
        for chaser_id, states in result.actual_trajectories.items():
            positions = np.asarray(states, dtype=float)[:, :3]
            fig.add_trace(
                go.Scatter3d(
                    x=positions[:, 0],
                    y=positions[:, 1],
                    z=positions[:, 2],
                    mode="lines+markers",
                    name=f"{chaser_id} actual",
                )
            )
            for idx, nominal in enumerate(result.nominal_trajectories.get(chaser_id, [])):
                nom = np.asarray(nominal, dtype=float)
                if nom.size:
                    fig.add_trace(
                        go.Scatter3d(
                            x=nom[:, 0],
                            y=nom[:, 1],
                            z=nom[:, 2],
                            mode="lines",
                            name=f"{chaser_id} nominal {idx}",
                            opacity=0.35,
                        )
                    )
        fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers", name="target"))
        fig.update_layout(
            title=f"ORBEX-A Paper System ({result.mission}, {result.approximation})",
            scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        path = self.output_dir / "trajectory.html"
        fig.write_html(path)
        manifest.add(path)

    def _render_video(
        self, result: PaperSystemResult, manifest: PaperSystemManifest
    ) -> None:
        path = self.output_dir / "trajectory.mp4"
        if shutil.which("ffmpeg") is None:
            path.write_bytes(b"")
            manifest.add(path)
            return

        positions_by_id = {
            chaser_id: np.asarray(states, dtype=float)[:, :3]
            for chaser_id, states in result.actual_trajectories.items()
        }
        all_positions = np.vstack(list(positions_by_id.values()))
        mins = all_positions.min(axis=0)
        maxs = all_positions.max(axis=0)
        center = (mins + maxs) / 2.0
        radius = max(float(np.max(maxs - mins)) / 2.0, 1.0)
        max_frames = max(len(values) for values in positions_by_id.values())

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        lines: Dict[str, object] = {}
        points: Dict[str, object] = {}
        for chaser_id in positions_by_id:
            (line,) = ax.plot([], [], [], label=chaser_id)
            (point,) = ax.plot([], [], [], marker="o")
            lines[chaser_id] = line
            points[chaser_id] = point
        ax.scatter([0], [0], [0], color="black", s=48, label="target")
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend(fontsize=8)

        def update(frame: int):
            for chaser_id, positions in positions_by_id.items():
                idx = min(frame, len(positions) - 1)
                segment = positions[: idx + 1]
                lines[chaser_id].set_data(segment[:, 0], segment[:, 1])
                lines[chaser_id].set_3d_properties(segment[:, 2])
                point = positions[idx]
                points[chaser_id].set_data([point[0]], [point[1]])
                points[chaser_id].set_3d_properties([point[2]])
            return list(lines.values()) + list(points.values())

        animation = FuncAnimation(fig, update, frames=max_frames, interval=125)
        animation.save(path, writer=FFMpegWriter(fps=self.fps), dpi=130)
        plt.close(fig)
        manifest.add(path)

    def _render_index(
        self, result: PaperSystemResult, manifest: PaperSystemManifest
    ) -> None:
        rows = "\n".join(
            (
                f"<tr><td>{chaser_id}</td><td>{len(states)}</td>"
                f"<td>{len(result.controls.get(chaser_id, []))}</td>"
                f"<td>{sum(item['solve_time'] for item in result.solve_stats.get(chaser_id, [])):.3f}</td></tr>"
            )
            for chaser_id, states in result.actual_trajectories.items()
        )
        title = f"ORBEX-A Paper System: {result.mission} {result.approximation}"
        html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #17202a; }}
    table {{ border-collapse: collapse; margin-bottom: 24px; }}
    th, td {{ border: 1px solid #c8d0d8; padding: 6px 10px; text-align: left; }}
    iframe {{ border: 1px solid #c8d0d8; }}
    img {{ max-width: 900px; width: 100%; margin: 10px 0 22px; display: block; }}
  </style>
</head>
<body>
<h1>{title}</h1>
<p>Solver: {result.solver_backend}. Success: {result.success}. {result.message}</p>
<table>
<tr><th>Chaser</th><th>State Samples</th><th>Controls</th><th>Solve Time (s)</th></tr>
{rows}
</table>
<h2>Interactive Trajectory</h2>
<iframe src="trajectory.html" width="100%" height="650"></iframe>
<h2>Video</h2>
<video controls width="900" src="trajectory.mp4"></video>
<h2>Diagnostics</h2>
<img src="actual_nominal_trajectories.png" alt="Actual and nominal trajectories">
<img src="tube_geometry.png" alt="Tube geometry">
<img src="control_effort.png" alt="Control effort">
<img src="rendezvous_margin.png" alt="Rendezvous margin">
<img src="docking_cylinder_margin.png" alt="Docking cylinder margin">
<img src="smid_fss_widths.png" alt="SMID feasible-set widths">
<img src="parameter_estimates_vs_truth.png" alt="Parameter estimates">
<img src="target_attitude.png" alt="Target attitude">
<img src="target_angular_velocity.png" alt="Target angular velocity">
<img src="multi_chaser_spacing.png" alt="Multi-chaser spacing">
</body>
</html>
"""
        path = self.output_dir / "index.html"
        path.write_text(html, encoding="utf-8")
        manifest.add(path)

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

    def run(self) -> List[PaperSystemManifest]:
        output_dir = Path(self.config.output_dir)
        data_dir = Path(self.config.data_dir)
        if self.config.from_data and self.config.clean_generated:
            raise ValueError("--from-data cannot be combined with --clean-generated")
        if self.config.clean_generated:
            clean_generated_outputs(output_dir, data_dir)

        manifests: List[PaperSystemManifest] = []
        for mission in self._missions():
            manifests.append(
                self._run_or_render(
                    mission=mission,
                    approximation="nonlinear",
                    solver_backend=self.config.primary_solver,
                    family_dir=output_dir / mission / "nonlinear",
                    data_family_dir=data_dir / mission / "nonlinear",
                )
            )
            if self.config.run_linearized:
                manifests.append(
                    self._run_or_render(
                        mission=mission,
                        approximation="linearized",
                        solver_backend=self.config.secondary_solver,
                        family_dir=output_dir / mission / "linearized",
                        data_family_dir=data_dir / mission / "linearized",
                    )
                )
        return manifests

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
    ) -> PaperSystemManifest:
        if self.config.from_data:
            result = load_paper_system_result(data_family_dir)
            data_file = data_family_dir / "mission_data.json"
        else:
            runner_kwargs = {}
            run_steps = self.config.steps
            if approximation == "linearized":
                runner_kwargs = {"horizon_steps": 2, "smid_window": 2}
                if self.config.linearized_steps is not None:
                    run_steps = min(self.config.steps, int(self.config.linearized_steps))
            runner = PaperSystemRunner(
                solver_backend=solver_backend,
                approximation=approximation,
                **runner_kwargs,
            )
            result = runner.run(mission=mission, steps=run_steps)
            data_file = runner.save_result(result, data_family_dir)
            if self._requires_success(approximation, solver_backend) and not result.success:
                raise RuntimeError(
                    f"{mission}/{approximation}/{solver_backend} did not complete: "
                    f"{result.message or 'mission_complete=false'}"
                )
        return PaperSystemRenderer(family_dir, fps=self.config.fps).render(
            result,
            data_file=data_file,
        )

    def _requires_success(self, approximation: str, solver_backend: str) -> bool:
        return (
            self.config.require_primary_success
            and approximation == "nonlinear"
            and solver_backend == self.config.primary_solver
        )


def clean_generated_outputs(output_dir: Path, data_dir: Path) -> None:
    """Remove generated artifacts and old demo outputs by explicit request."""
    candidates = [
        Path(output_dir),
        Path(data_dir),
        Path("results") / "paper_demo",
        Path("plots") / "demo",
        Path("plots") / "paper_demo",
        Path("plots") / "paper_system",
        Path("plots") / "from_data_smoke",
        Path("data") / "mpc",
        Path("data") / "tube",
        Path("data") / "adtmpc",
        Path("data") / "paper_demo",
        Path(".pytest_cache"),
        Path(".mypy_cache"),
        Path(".ruff_cache"),
    ]
    for candidate in candidates:
        if candidate.exists():
            if candidate.is_dir():
                shutil.rmtree(candidate)
            else:
                candidate.unlink()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate ORBEX-A paper-system artifacts")
    parser.add_argument("--output", default="results/paper_system")
    parser.add_argument("--data-output", default="data/paper_system")
    parser.add_argument(
        "--steps",
        type=int,
        default=450,
        help="Maximum number of receding-horizon MPC updates",
    )
    parser.add_argument("--mission", default="all", choices=["single", "multi", "all"])
    parser.add_argument("--primary-solver", default="gekko", choices=["gekko", "scipy", "casadi"])
    parser.add_argument("--secondary-solver", default="scipy", choices=["gekko", "scipy", "casadi"])
    parser.add_argument("--run-linearized", action="store_true")
    parser.add_argument(
        "--linearized-steps",
        type=int,
        default=20,
        help="Maximum updates for optional linearized comparison artifacts",
    )
    parser.add_argument("--clean-generated", action="store_true")
    parser.add_argument(
        "--from-data",
        action="store_true",
        help="Regenerate plots from mission_data.json files in the output folders",
    )
    args = parser.parse_args()

    config = DemoConfig(
        output_dir=Path(args.output),
        data_dir=Path(args.data_output),
        steps=args.steps,
        mission=args.mission,
        primary_solver=args.primary_solver,
        secondary_solver=args.secondary_solver,
        run_linearized=args.run_linearized,
        linearized_steps=args.linearized_steps,
        from_data=args.from_data,
        clean_generated=args.clean_generated,
    )
    manifests = OrbexaDemo(config).run()
    for manifest in manifests:
        print(f"Paper system written to {manifest.output_dir}")
        for artifact in manifest.artifacts:
            print(artifact)


PaperDemoRenderer = PaperSystemRenderer
DemoManifest = PaperSystemManifest
MissionDemo = PaperSystemResult


if __name__ == "__main__":
    main()
