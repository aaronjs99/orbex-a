"""Independent static artifacts derived from available saved mission fields."""
from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
import matplotlib.pyplot as plt
from orbexa.visualization.plot_canvas import artifact_figure, artifact_subplots
import numpy as np

from orbexa.visualization.geometry import MissionGeometry
from orbexa.visualization.samples import MissionSamples, recorded_array
from orbexa.visualization.style import chaser_color


class MissionPlots:
    """Render each static diagnostic only when its own recorded fields exist."""

    REQUIRED_FIELDS = {
        "actual_nominal_trajectories.png": ("anom_history", "actual_trajectories"),
        "tube_geometry.png": ("anom_history", "tube_radius_history or tube_diagnostics or tube_profiles"),
        "control_effort.png": ("anom_history", "controls"),
        "rendezvous_margin.png": ("anom_history", "rendezvous_margins"),
        "docking_cylinder_margin.png": ("anom_history", "docking_cylinder_margins"),
        "tightened_rendezvous_margin.png": ("anom_history", "tightened_rendezvous_margins"),
        "tightened_docking_cylinder_margin.png": ("anom_history", "tightened_docking_cylinder_margins"),
        "active_target_margin.png": ("anom_history", "active_target_margins"),
        "smid_fss_widths.png": ("anom_history", "feasible_set_history.eccentricity/alpha/beta"),
        "parameter_estimates_vs_truth.png": ("anom_history", "parameter_estimate_history", "feasible_set_history", "truth"),
        "target_attitude.png": ("anom_history", "target_attitude_history or saved target orientation model"),
        "target_angular_velocity.png": ("anom_history", "target_kinematics.body_angular_velocity or target_angular_velocity_history or saved target angular_velocity"),
        "multi_chaser_spacing.png": ("anom_history", "pairwise_spacing"),
    }

    def __init__(self, result, output_dir: Path, *, samples=None, geometry=None):
        self.result, self.output_dir = result, Path(output_dir)
        self.samples = samples if samples is not None else MissionSamples(result)
        self.q = self.samples.recorded_q
        self.geometry = geometry if geometry is not None else MissionGeometry(self.samples)
        self.skipped_artifacts, self.artifacts = {}, []

    def render(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts, self.skipped_artifacts = [], {}
        self._trajectory()
        self._line_plot("tube_geometry.png", "tube position radius", self.tube_values())
        controls = {}
        for identifier, values in self.result.controls.items():
            arr = recorded_array(values)
            if arr.ndim == 2 and arr.shape[1] >= 3:
                controls[identifier] = np.linalg.norm(arr[:, :3], axis=1)
        self._line_plot("control_effort.png", "control norm", controls)
        for filename, field, label in (
            ("rendezvous_margin.png", "rendezvous_margins", "physical rendezvous margin"),
            ("docking_cylinder_margin.png", "docking_cylinder_margins", "physical docking cylinder union margin"),
            ("tightened_rendezvous_margin.png", "tightened_rendezvous_margins", "tube-tightened rendezvous margin"),
            ("tightened_docking_cylinder_margin.png", "tightened_docking_cylinder_margins", "tube-tightened docking cylinder union margin"),
            ("active_target_margin.png", "active_target_margins", "active target safety margin"),
        ):
            self._line_plot(filename, label, getattr(self.result, field), zero_line=True, phase_markers=True)
        self._smid_widths()
        self._parameter_estimates()
        self._target_plot("target_attitude.png", self.samples.recorded_orientation, "Euler angle (rad)", ("roll", "pitch", "yaw"))
        kinematics = self.result.metadata.get("target_kinematics", {})
        kinematics = kinematics if isinstance(kinematics, Mapping) else {}
        spin = recorded_array(kinematics.get("body_angular_velocity", []))
        spin_q = recorded_array(kinematics.get("anomaly", []))
        if (spin.shape == (len(self.q), 3) and np.all(np.isfinite(spin))
                and spin_q.shape == self.q.shape and np.array_equal(spin_q, self.q)):
            self._target_plot("target_angular_velocity.png", spin, "Body angular velocity (rad/s)", ("body x", "body y", "body z"))
        else:
            self._target_plot("target_angular_velocity.png", self.samples.recorded_rates, "Euler angle rate (rad/rad anomaly)", ("roll rate", "pitch rate", "yaw rate"))
        self._spacing()
        return list(self.artifacts)

    def _skip(self, filename, reason):
        self.skipped_artifacts[filename] = reason

    def _save(self, fig, filename, label=None):
        path = self.output_dir / filename
        try:
            fig.tight_layout()
            fig.savefig(path, dpi=150)
        finally:
            plt.close(fig)
        self.artifacts.append((path, label))
        return path

    def _phase_lines(self, ax):
        for index in self.samples.phase_switch_indices():
            ax.axvline(self.samples.q[index], color="#D55E00", linestyle=":", linewidth=1.0, alpha=0.8)

    def _trajectory(self):
        if not len(self.samples.q):
            self._skip("actual_nominal_trajectories.png", "No complete usable recorded state trajectories; see saved-data insight notes.")
            return
        fig = artifact_figure(figsize=(8.5, 7))
        ax = fig.add_subplot(111, projection="3d")
        for index, identifier in enumerate(self.samples.ids):
            positions, color = self.samples.states[:, index, :3], chaser_color(index)
            ax.plot(*positions.T, color=color, label=f"{identifier} actual")
            ax.scatter(*positions[0], s=18, color=color)
            for number, nominal in enumerate(self.samples.nominal_horizons(identifier)):
                ax.plot(*nominal.T, linestyle=":", color=color, alpha=0.25,
                        label=f"{identifier} nominal horizons" if number == 0 else None)
            for number, switch in enumerate(self.samples.phase_switch_indices()):
                ax.scatter(*positions[switch], color="#D55E00", marker="x", s=50,
                           label="rendezvous-to-docking switch" if index == 0 and number == 0 else None)
        final = len(self.samples.q) - 1
        self.geometry.plot_sphere(ax)
        self.geometry.plot_target(ax, final)
        self.geometry.plot_docking_points(ax, final)
        ax.set(xlabel="x", ylabel="y", zlabel="z")
        ax.legend(fontsize=8)
        if self.geometry.cylinder_mesh(final) is None:
            ax.set_title("Recorded trajectories; saved target geometry or pose unavailable")
        self._save(fig, "actual_nominal_trajectories.png", "Actual/nominal trajectories with rotating cylinder target, rendezvous sphere, docking points, and phase switch markers.")

    def tube_values(self):
        values = {}
        if recorded_array(self.result.tube_radius_history).size:
            values["max_tube"] = self.result.tube_radius_history
        for identifier in self.samples.ids:
            matched = self.samples.matched_tube(identifier)
            if matched is not None:
                values[identifier] = np.linalg.norm(matched[1], axis=1)
                continue
            profiles = (self.result.tube_profiles or {}).get(identifier, [])
            if profiles and all(isinstance(profile, dict) and "max_position_radius" in profile for profile in profiles):
                values[identifier] = [profile["max_position_radius"] for profile in profiles]
        return values

    def _line_plot(self, filename, ylabel, values_by_label, *, zero_line=False, phase_markers=False):
        series = {}
        for label, values in values_by_label.items():
            values = recorded_array(values)
            if values.ndim == 1:
                count = min(len(values), len(self.q))
                if count and np.any(np.isfinite(values[:count])):
                    series[label] = values[:count]
        if not series:
            self._skip(filename, "No usable aligned recorded values for " + ", ".join(self.REQUIRED_FIELDS[filename][1:]) + ".")
            return
        fig, ax = artifact_subplots(figsize=(9, 4.8))
        for label, values in series.items():
            color = chaser_color(self.samples.ids.index(label)) if label in self.samples.ids else "#596780"
            ax.plot(self.q[:len(values)], values, label=label, color=color, linestyle="--" if label == "max_tube" else "-")
        if zero_line:
            ax.axhline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.75)
        if phase_markers:
            self._phase_lines(ax)
        ax.set(xlabel="true anomaly", ylabel=ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        self._save(fig, filename)

    def _parameter_histories(self):
        keys = ("eccentricity", "alpha", "beta")
        history = self.result.feasible_set_history or []
        count = min(len(history), len(self.q))
        if not count:
            return None
        bands = {}
        for key in keys:
            values = recorded_array([item.get(key, []) for item in history[:count] if isinstance(item, dict)])
            if values.shape != (count, 2) or not np.all(np.isfinite(values)) or np.any(values[:, 1] < values[:, 0]):
                return None
            bands[key] = values
        return bands

    def _smid_widths(self):
        bands = self._parameter_histories()
        if bands is None:
            self._skip("smid_fss_widths.png", "Saved feasible-set history requires finite eccentricity, alpha and beta interval pairs.")
            return
        fig, axes = artifact_subplots(3, 1, figsize=(9, 8), sharex=True)
        for ax, (key, values) in zip(axes, bands.items()):
            ax.plot(self.q[:len(values)], values[:, 1] - values[:, 0], label=key)
            ax.set_ylabel(f"{key} width")
            ax.grid(True, alpha=0.25)
        axes[-1].set_xlabel("true anomaly")
        self._save(fig, "smid_fss_widths.png")

    def _parameter_estimates(self):
        bands = self._parameter_histories()
        history = self.result.parameter_estimate_history or []
        count = min(len(history), len(self.q), len(next(iter(bands.values()))) if bands else 0)
        estimates = {key: recorded_array([item.get(key, np.nan) for item in history[:count] if isinstance(item, dict)]) for key in ("eccentricity", "alpha", "beta")}
        truth = recorded_array([(self.result.truth or {}).get(key, np.nan) for key in estimates])
        if not count or any(value.shape != (count,) or not np.all(np.isfinite(value)) for value in estimates.values()) or not np.all(np.isfinite(truth)):
            self._skip("parameter_estimates_vs_truth.png", "Saved point beliefs, finite feasible-set intervals and simulated truth for eccentricity, alpha and beta are required.")
            return
        fig, axes = artifact_subplots(3, 1, figsize=(9, 8), sharex=True)
        for ax, (key, values), true in zip(axes, estimates.items(), truth):
            times, intervals = self.q[:count], bands[key][:count]
            ax.fill_between(times, intervals[:, 0], intervals[:, 1], alpha=0.16, label="feasible set")
            ax.plot(times, values, label="belief")
            ax.axhline(true, color="black", linestyle="--", linewidth=1.0, label="truth")
            ax.set_ylabel(key)
            ax.grid(True, alpha=0.25)
        axes[-1].set_xlabel("true anomaly")
        axes[0].legend(fontsize=8)
        self._save(fig, "parameter_estimates_vs_truth.png")

    def _target_plot(self, filename, values, ylabel, labels):
        if values is None or not len(values):
            self._skip(filename, "Saved target samples or the target model stored in this result are required.")
            return
        fig, ax = artifact_subplots(figsize=(9, 4.8))
        for index, label in enumerate(labels):
            ax.plot(self.q, values[:, index], label=label)
        ax.set(xlabel="true anomaly", ylabel=ylabel)
        if "prescribed" in str(self.result.metadata.get("target_attitude_model", "")).lower():
            ax.set_title("Prescribed Euler-angle target motion in true anomaly")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        self._save(fig, filename)

    def _spacing(self):
        history = self.result.pairwise_spacing or []
        keys = sorted({key for item in history if isinstance(item, dict) for key in item})
        values = {key: [item.get(key, np.nan) if isinstance(item, dict) else np.nan for item in history] for key in keys}
        self._line_plot("multi_chaser_spacing.png", "pairwise separation", values)
