"""Additional diagnostics derived exclusively from saved mission samples."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from orbexa.visualization.plots.canvas import artifact_figure, artifact_subplots
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
import numpy as np

from orbexa.simulation.approach.records.results import ADTMPCMissionResult
from orbexa.visualization.data.samples import MissionSamples
from orbexa.visualization.style import _MARKERS, _LINES, _INK, _MUTED, _STYLE, chaser_color, chaser_label


class MissionInsights:
    """Render optional saved-data insights and report unsupported views honestly."""

    REQUIRED_FIELDS = {
        "docking_errors.png": ("anom_history", "actual_trajectories", "docking_points", "saved target pose"),
        "observable_drag.png": ("anom_history", "parameter_estimate_history.alpha/beta"),
        "smid_point_errors.png": ("anom_history", "smid_records emitted-point metrics"),
        "docking_body_closeup.png": ("anom_history", "actual_trajectories", "docking phase", "saved target pose and cylinder"),
        "docking_closeup.gif": ("at least two docking samples", "saved target pose and cylinder"),
        "minimum_pairwise_spacing.png": ("anom_history", "at least two actual_trajectories"),
    }

    def __init__(self, result: ADTMPCMissionResult, output_dir: Path, fps: int = 8, *, samples=None):
        if isinstance(fps, bool) or not isinstance(fps, int) or fps < 1:
            raise ValueError("fps must be a positive integer")
        self.result, self.output_dir, self.fps = result, Path(output_dir), fps
        self.samples = samples if samples is not None else MissionSamples(result)
        self.skipped_artifacts: Dict[str, str] = {}
        self.artifacts: List[Tuple[Path, str]] = []
        self.provenance: Dict[str, object] = {}

    def render(self) -> List[Tuple[Path, str]]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts, self.skipped_artifacts, self.provenance = [], {}, {}
        with plt.rc_context(_STYLE):
            self._docking_errors()
            self._observable_drag()
            self._smid_errors()
            self._body_closeup()
            self._pairwise_spacing()
        self.write_manifest()
        return list(self.artifacts)

    def write_manifest(self) -> Path:
        payload = {
            "schema": "orbexa.mission_insights.v1", "mission": self.result.mission,
            "approximation": self.result.approximation, "solver_backend": self.result.solver_backend,
            "recorded_samples": len(self.samples.q), "declared_samples": len(self.result.anom_history),
            "chaser_ids": self.samples.ids, "mission_success": bool(self.result.success),
            "notes": self.samples.notes, "artifacts": {path.name: label for path, label in self.artifacts if path.name != "insights_manifest.json"},
            "skipped_artifacts": self.skipped_artifacts, "provenance": self.provenance,
        }
        path = self.output_dir / "insights_manifest.json"
        path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        if not any(item[0] == path for item in self.artifacts):
            self.artifacts.append((path, "Saved-sample provenance and reasons for unavailable insight views."))
        return path

    def _save(self, fig, filename: str, label: str) -> Path:
        path = self.output_dir / filename
        try:
            fig.savefig(path, dpi=160)
        finally:
            plt.close(fig)
        self.artifacts.append((path, label))
        return path

    def _skip(self, filename: str, reason: str) -> None:
        self.skipped_artifacts[filename] = reason

    def _identity_handles(self):
        return [Line2D([], [], color=chaser_color(index), marker=_MARKERS[index % len(_MARKERS)],
                       linestyle=_LINES[index % len(_LINES)], linewidth=1.6, markersize=5,
                       label=chaser_label(identifier)) for index, identifier in enumerate(self.samples.ids)]

    def _axis(self, ax, q=None) -> None:
        q = self.samples.q if q is None else np.asarray(q)
        if len(q):
            left, right = float(q[0]), float(q[-1])
            pad = 0.02 * max(abs(left), 1.0) if left == right else 0.0
            ax.set_xlim(left - pad, right + pad)
        ax.grid(True)
        if self.samples.docking_start is not None:
            ax.axvline(self.samples.q[self.samples.docking_start], color="#89949C", linestyle=(0, (3, 4)), linewidth=0.9)

    @staticmethod
    def _error_scale(ax, arrays) -> None:
        finite = np.concatenate([np.asarray(values).ravel() for values in arrays])
        finite = finite[np.isfinite(finite)]
        if len(finite) and np.all(finite > 0.0):
            ax.set_yscale("log")
        else:
            positive = finite[finite > 0]
            ax.set_yscale("symlog", linthresh=max(float(np.min(positive)) * 0.5, 1e-10) if len(positive) else 1e-8)

    def _docking_errors(self) -> None:
        s = self.samples
        if s.position_error is None:
            self._skip("docking_errors.png", "Recorded states, target orientation, and assigned docking points are required.")
            return
        fig, axes = artifact_subplots(2, 1, figsize=(12, 8), sharex=True)
        rows = math.ceil(len(s.ids) / 4)
        fig.subplots_adjust(left=0.11, right=0.97, bottom=0.16, top=max(0.55, 0.82 - 0.035 * (rows - 1)), hspace=0.25)
        fig.suptitle("Error to the rotating docking goal | " + s.title, y=0.975, fontsize=17)
        fig.text(0.5, 0.93, "{} recorded simulation states · dashed vertical line marks docking when recorded".format(len(s.q)), ha="center", color=_MUTED)
        fig.legend(handles=self._identity_handles(), loc="upper center", bbox_to_anchor=(0.5, 0.90), ncol=min(4, len(s.ids)), frameon=False)
        for ax, values, key, label in zip(axes, (s.position_error, s.velocity_error),
                                        ("docking_position_tolerance", "docking_velocity_tolerance"),
                                        ("Position error (position units)", "Velocity-state error (position units/rad)")):
            if values is None:
                ax.text(0.5, 0.5, "Target rate history unavailable; velocity error cannot be reconstructed.", transform=ax.transAxes, ha="center", wrap=True)
            else:
                for index in range(len(s.ids)):
                    ax.plot(s.q, values[:, index], color=chaser_color(index), linestyle=_LINES[index % len(_LINES)], linewidth=1.6,
                            marker=_MARKERS[index % len(_MARKERS)] if len(s.q) == 1 else None)
                    ax.plot(s.q[-1], values[-1, index], marker=_MARKERS[index % len(_MARKERS)], color=chaser_color(index), markersize=5)
                tolerance = self.result.metadata.get(key)
                source = "saved tolerance"
                if tolerance is None:
                    tolerance, source = 0.08, "current-default fallback; absent from saved data"
                tolerance = float(tolerance)
                if np.isfinite(tolerance) and tolerance > 0:
                    ax.axhline(tolerance, color=_INK, linestyle=(0, (5, 3)), linewidth=1.1, label="{} ({})".format(tolerance, source))
                    ax.legend(loc="upper right", frameon=False, fontsize=9)
                self._error_scale(ax, [values])
                self.provenance[key] = {"value": tolerance if np.isfinite(tolerance) else None, "source": source}
            ax.set_ylabel(label)
            self._axis(ax)
        axes[-1].set_xlabel("True anomaly (rad)")
        final_position = float(np.nanmax(s.position_error[-1]))
        final_velocity = None if s.velocity_error is None else float(np.nanmax(s.velocity_error[-1]))
        final_text = "Final largest position error: {:.5g}".format(final_position)
        if final_velocity is not None:
            final_text += "   |   Final largest velocity-state error: {:.5g}".format(final_velocity)
        fig.text(0.5, 0.085, final_text, ha="center")
        fig.text(0.5, 0.045, "Velocity states use true anomaly, not seconds. Errors are sampled simulation diagnostics.", ha="center", fontsize=9, color=_MUTED)
        self._save(fig, "docking_errors.png", "Recorded position and anomaly-velocity errors to assigned rotating docking goals; any 0.08 fallback is labeled.")

    def _observable_drag(self) -> None:
        history = self.result.parameter_estimate_history or []
        count = min(len(history), len(self.samples.recorded_q))
        if not count or not all("alpha" in item and "beta" in item for item in history[:count]):
            self._skip("observable_drag.png", "Saved point-belief history does not contain both alpha and beta.")
            return
        q = self.samples.recorded_q[:count]
        alpha = np.asarray([item["alpha"] for item in history[:count]], dtype=float)
        beta = np.asarray([item["beta"] for item in history[:count]], dtype=float)
        if not np.any(np.isfinite(alpha) & np.isfinite(beta)):
            self._skip("observable_drag.png", "No finite paired alpha/beta point beliefs were recorded.")
            return
        finite = np.concatenate((np.abs(alpha[np.isfinite(alpha)]), np.abs(beta[np.isfinite(beta)])))
        maximum = float(np.max(finite))
        exponent = int(np.floor(np.log10(maximum))) if maximum > 0 else 0
        factor = 10.0 ** -exponent
        fig, axes = artifact_subplots(3, 1, figsize=(12, 9), sharex=True)
        fig.subplots_adjust(left=0.11, right=0.97, bottom=0.14, top=0.86, hspace=0.3)
        fig.suptitle("Drag combination & marginal uncertainty | " + self.samples.title, y=0.975, fontsize=17)
        fig.text(0.5, 0.93, "Shared controller point belief; shaded marginal sets are not confidence intervals", ha="center", color=_MUTED)
        axes[0].plot(q, (beta - alpha) * factor, color=chaser_color(0), linewidth=1.9, label="Point-belief difference", marker="o" if count == 1 else None)
        truth = self.result.truth or {}
        if all(key in truth for key in ("alpha", "beta")):
            axes[0].axhline((float(truth["beta"]) - float(truth["alpha"])) * factor, color=_INK, linestyle="--", linewidth=1.2, label="Simulated truth")
        axes[0].set_ylabel(r"$\beta-\alpha$ ($10^{%d}$)" % exponent)
        for ax, key, values, index in zip(axes[1:], ("alpha", "beta"), (alpha, beta), (0, 1)):
            bands = (self.result.feasible_set_history or [])[:count]
            if len(bands) == count and all(key in item and len(item[key]) == 2 for item in bands):
                bounds = np.asarray([item[key] for item in bands], dtype=float) * factor
                if count == 1:
                    ax.vlines(q[0], bounds[0, 0], bounds[0, 1], color=chaser_color(index), alpha=0.25, linewidth=9, label="Marginal feasible set")
                else:
                    ax.fill_between(q, bounds[:, 0], bounds[:, 1], color=chaser_color(index), alpha=0.14, label="Marginal feasible set")
            ax.plot(q, values * factor, color=chaser_color(index), linewidth=1.8, label="Point belief", marker="o" if count == 1 else None)
            if key in truth:
                ax.axhline(float(truth[key]) * factor, color=_INK, linestyle="--", linewidth=1.2, label="Simulated truth")
            ax.set_ylabel(r"$%s$ ($10^{%d}$)" % ("\\alpha" if key == "alpha" else "\\beta", exponent))
        for index, ax in enumerate(axes):
            ax.legend(loc="lower right" if index == 0 else "upper right", frameon=False, ncol=3, fontsize=9)
            self._axis(ax, q)
        axes[-1].set_xlabel("True anomaly (rad)")
        fig.text(0.5, 0.065, "Matching a drag difference does not establish separate alpha / beta identification.", ha="center")
        fig.text(0.5, 0.03, "Only saved beliefs, marginal sets, and available simulated truth are shown.", ha="center", fontsize=9, color=_MUTED)
        self._save(fig, "observable_drag.png", "Saved beta-alpha point-belief difference and individual marginal feasible sets; separate identification is not inferred.")

    def _smid_errors(self) -> None:
        records = self.result.smid_records or []
        if not records or not any(item.get("point_prediction_error") is not None or item.get("point_fit_error") is not None for item in records):
            self._skip("smid_point_errors.png", "No emitted-point SMID metrics were recorded; a set witness is not a substitute.")
            return
        rows = []
        for record in records:
            try:
                start, window = float(record["start_anom"]), int(record["window_size"])
                matches = np.flatnonzero(np.isclose(self.samples.recorded_q, start, rtol=0, atol=1e-8))
                if len(matches) and matches[0] + window - 1 < len(self.samples.recorded_q):
                    endpoint = self.samples.recorded_q[matches[0] + window - 1]
                elif self.result.metadata.get("anom_step") is not None:
                    endpoint = start + (window - 1) * float(self.result.metadata["anom_step"])
                else:
                    continue
                rows.append((endpoint, record))
            except (KeyError, TypeError, ValueError):
                continue
        if not rows:
            self._skip("smid_point_errors.png", "SMID window endpoints cannot be located in the saved anomaly history.")
            return
        q = np.asarray([item[0] for item in rows])
        def metric(key):
            return np.asarray([np.nan if item.get(key) is None else float(item[key]) for _, item in rows])
        series = [metric(key) for key in ("prediction_error", "point_prediction_error", "point_fit_error", "verification_error")]
        fig, ax = artifact_subplots(figsize=(12, 6.8))
        fig.subplots_adjust(left=0.11, right=0.97, bottom=0.2, top=0.77)
        fig.suptitle("SMID errors at the emitted point belief | " + self.samples.title, y=0.975, fontsize=17)
        fig.text(0.5, 0.925, "Recorded window endpoints; the candidate-set witness is a separate estimate", ha="center", color=_MUTED)
        specifications = (("Before update: window error", "#7B848C", "--"), ("Emitted point: window error", chaser_color(0), "-"),
                          ("Emitted point: weighted fit RMS", chaser_color(2), "-."), ("Set witness: GEKKO terminal error", "#A87400", "none"))
        for values, (label, color, line) in zip(series, specifications):
            if np.any(np.isfinite(values)):
                ax.plot(q, values, color=color, linestyle=line, linewidth=1.6, marker="^" if line == "none" else "o" if len(q) == 1 else None, markersize=4, label=label)
        bound = self.result.metadata.get("smid_error_bound")
        if bound is not None and np.isfinite(float(bound)):
            ax.axhline(float(bound), color=_INK, linestyle=(0, (5, 3)), linewidth=1.1, label="Saved error bound {:g}".format(float(bound)))
        self._error_scale(ax, series)
        self._axis(ax, q)
        ax.set(xlabel="True anomaly (rad)", ylabel="State-window error / weighted fit RMS")
        fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.89), ncol=2, frameon=False, fontsize=9)
        fig.text(0.5, 0.095, "Missing values remain absent; skipped verification is never plotted as zero.", ha="center", fontsize=10)
        fig.text(0.5, 0.045, "Bound unavailable in saved data." if bound is None else "Fit RMS and terminal error use different summaries of the recorded state window.", ha="center", fontsize=9, color=_MUTED)
        self._save(fig, "smid_point_errors.png", "Recorded before-update, emitted-point prediction/fit, and separate set-witness errors; missing metrics remain absent.")

    def _body_figure(self, start: int):
        s = self.samples
        radius, half_length = s.cylinder()
        positions = s.body[start:]
        goal_values = list(s.goals_body.values())
        xy = max(radius, float(np.nanmax(np.abs(positions[:, :, :2]))), max([float(np.max(np.abs(goal[:2]))) for goal in goal_values] + [0.0])) * 1.15
        z = max(half_length, float(np.nanmax(np.abs(positions[:, :, 2]))), max([abs(float(goal[2])) for goal in goal_values] + [0.0])) * 1.2
        fig = artifact_figure(figsize=(13, 8.2))
        grid = fig.add_gridspec(2, 2, left=0.025, right=0.97, bottom=0.26, top=0.85, width_ratios=(1.25, 1.0), hspace=0.35, wspace=0.16)
        ax3, axy, axz = fig.add_subplot(grid[:, 0], projection="3d"), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1])
        fig.suptitle("Docking close-up in the target body frame | " + s.title, y=0.975, fontsize=17)
        fig.text(0.5, 0.925, "Fixed saved exclusion boundary · diamonds mark available assigned docking points", ha="center", color=_MUTED)
        angle, heights = np.meshgrid(np.linspace(0, 2 * np.pi, 48), [-half_length, half_length])
        ax3.plot_surface(radius * np.cos(angle), radius * np.sin(angle), heights, color="#9BA7AF", alpha=0.18, linewidth=0, shade=False)
        axy.add_patch(Circle((0, 0), radius, facecolor="#EEF1F3", edgecolor="#8E9AA3"))
        axz.add_patch(Rectangle((-radius, -half_length), 2 * radius, 2 * half_length, facecolor="#EEF1F3", edgecolor="#8E9AA3"))
        ax3.set(xlim=(-xy, xy), ylim=(-xy, xy), zlim=(-z, z), xlabel="Body x", ylabel="Body y", zlabel="Body z")
        ax3.set_box_aspect((xy, xy, z))
        ax3.view_init(elev=25, azim=-58)
        for ax, vertical, limit in ((axy, 1, xy), (axz, 2, z)):
            ax.set(xlim=(-xy, xy), ylim=(-limit, limit), xlabel="Body x", ylabel="Body " + ("y" if vertical == 1 else "z"))
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True)
        axy.set_title("Radial view")
        axz.set_title("Axial side view")
        artists = []
        for index, identifier in enumerate(s.ids):
            color, marker, line = chaser_color(index), _MARKERS[index % len(_MARKERS)], _LINES[index % len(_LINES)]
            line3, = ax3.plot([], [], [], color=color, linewidth=1.8, linestyle=line)
            point3, = ax3.plot([], [], [], color=color, marker=marker, markersize=6, linestyle="none")
            line_xy, = axy.plot([], [], color=color, linewidth=1.7, linestyle=line)
            point_xy, = axy.plot([], [], color=color, marker=marker, markersize=6, linestyle="none")
            line_xz, = axz.plot([], [], color=color, linewidth=1.7, linestyle=line)
            point_xz, = axz.plot([], [], color=color, marker=marker, markersize=6, linestyle="none")
            if identifier in s.goals_body:
                goal = s.goals_body[identifier]
                ax3.scatter(*goal, color=color, marker="D", s=32, depthshade=False)
                axy.plot(goal[0], goal[1], marker="D", color=color, markersize=5)
                axz.plot(goal[0], goal[2], marker="D", color=color, markersize=5)
            artists.append((line3, point3, line_xy, point_xy, line_xz, point_xz))
        status = fig.text(0.5, 0.19, "", ha="center", fontsize=10)
        fig.legend(handles=self._identity_handles(), loc="upper center", bbox_to_anchor=(0.5, 0.15), ncol=min(5, len(s.ids)), frameon=False)
        fig.text(0.5, 0.025, "Recorded final-approach samples; no interpolation or continuous-time safety certification.", ha="center", fontsize=9, color=_MUTED)
        def update(frame):
            for index, group in enumerate(artists):
                line3, point3, line_xy, point_xy, line_xz, point_xz = group
                trail, point = s.body[start:frame + 1, index], s.body[frame, index]
                line3.set_data(trail[:, 0], trail[:, 1])
                line3.set_3d_properties(trail[:, 2])
                point3.set_data([point[0]], [point[1]])
                point3.set_3d_properties([point[2]])
                line_xy.set_data(trail[:, 0], trail[:, 1])
                point_xy.set_data([point[0]], [point[1]])
                line_xz.set_data(trail[:, 0], trail[:, 2])
                point_xz.set_data([point[0]], [point[2]])
            status.set_text("Recorded sample {} / {}   |   True anomaly {:.4f} rad".format(frame + 1, len(s.q), s.q[frame]))
            return [item for group in artists for item in group] + [status]
        return fig, update

    def _body_closeup(self) -> None:
        s = self.samples
        reason = None
        if s.docking_start is None:
            reason = "No docking phase was recorded."
        elif s.body is None or s.cylinder() is None:
            reason = "Saved target orientation and cylinder dimensions are required."
        if reason:
            for name in ("docking_body_closeup.png", "docking_closeup.gif"):
                self._skip(name, reason)
            return
        radius, half_length = s.cylinder()
        extent = max(radius, half_length, max([float(np.linalg.norm(goal)) for goal in s.goals_body.values()] + [0.0]))
        nearby = np.flatnonzero(np.max(np.linalg.norm(s.body, axis=2), axis=1) <= 1.8 * extent)
        start = max(s.docking_start, int(nearby[0]) if len(nearby) else max(0, len(s.q) - 80))
        frames = s.frame_indices(96, start=start)
        fig, update = self._body_figure(start)
        update(frames[-1])
        self._save(fig, "docking_body_closeup.png", "Recorded final approach in the saved target body frame, with radial and axial projections.")
        self.provenance["body_closeup"] = {"first_sample_index": start, "source_sample_indices": frames, "playback_frames_per_second": self.fps, "interpolated": False}
        if len(frames) < 2:
            self._skip("docking_closeup.gif", "Only one docking sample exists; no docking motion was recorded.")
            return
        fig, update = self._body_figure(start)
        animation = FuncAnimation(fig, update, frames=frames, interval=1000 / self.fps, blit=False, repeat=True)
        path = self.output_dir / "docking_closeup.gif"
        try:
            animation.save(path, writer=PillowWriter(fps=self.fps), dpi=80)
        finally:
            plt.close(fig)
        self.artifacts.append((path, "Body-frame docking replay of {} selected recorded samples at {} fps; no interpolated states.".format(len(frames), self.fps)))

    def _pairwise_spacing(self) -> None:
        s = self.samples
        if len(s.ids) < 2 or not len(s.q):
            self._skip("minimum_pairwise_spacing.png", "At least two recorded chaser trajectories are required.")
            return
        minimum = np.full((len(s.ids), len(s.ids)), np.nan)
        for row in range(len(s.ids)):
            for col in range(row + 1, len(s.ids)):
                distances = np.linalg.norm(s.states[:, row, :3] - s.states[:, col, :3], axis=1)
                finite = distances[np.isfinite(distances)]
                if len(finite):
                    minimum[row, col] = minimum[col, row] = np.min(finite)
        finite = minimum[np.isfinite(minimum)]
        if not len(finite):
            self._skip("minimum_pairwise_spacing.png", "No finite recorded pairwise distances are available.")
            return
        low, high = float(np.min(finite)), float(np.max(finite))
        limit = self.result.metadata.get("min_chaser_separation")
        fig, ax = artifact_subplots(figsize=(max(8.5, len(s.ids) * 0.45 + 4), max(7, len(s.ids) * 0.42 + 3)))
        fig.subplots_adjust(left=0.16, right=0.85, top=0.84, bottom=0.17)
        fig.suptitle("Minimum pairwise spacing over recorded samples", y=0.975, fontsize=17)
        subtitle = "{} states · smallest center spacing {:.5g}".format(len(s.q), low)
        subtitle += " · saved minimum {:g}".format(float(limit)) if limit is not None else " · configured minimum unavailable"
        fig.text(0.5, 0.925, subtitle, ha="center", color=_MUTED)
        cmap = plt.get_cmap("cividis").copy()
        cmap.set_bad("#F0F2F4")
        image = ax.imshow(np.ma.masked_invalid(minimum), cmap=cmap, vmin=low, vmax=high if high > low else low + max(abs(low) * 0.01, 1e-6))
        labels = [chaser_label(identifier) for identifier in s.ids]
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        for row in range(len(s.ids)):
            for col in range(len(s.ids)):
                value = minimum[row, col]
                if len(s.ids) <= 16:
                    ax.text(col, row, "—" if not np.isfinite(value) else "{:.3f}".format(value), ha="center", va="center",
                            color="#9BA4AD" if not np.isfinite(value) else "white" if value <= (low + high) / 2 else _INK, fontsize=9)
                if np.isfinite(value) and np.isclose(value, low, atol=1e-12, rtol=0):
                    ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, edgecolor="#D59B19", linewidth=2))
                if np.isfinite(value) and limit is not None and value < float(limit):
                    ax.plot(col, row, marker="x", color="#9B2D52", markersize=10)
        fig.colorbar(image, ax=ax, fraction=0.045, pad=0.035).set_label("Minimum recorded center spacing (position units)")
        fig.text(0.5, 0.065, "Gold outlines: smallest pair · crosses: below the saved limit · diagonal: not applicable", ha="center", fontsize=9)
        fig.text(0.5, 0.025, "Every finite recorded position is considered; sampled minima do not certify continuous-time separation.", ha="center", fontsize=9, color=_MUTED)
        self.provenance["minimum_recorded_pairwise_spacing"] = low
        self._save(fig, "minimum_pairwise_spacing.png", "Each chaser pair's minimum center spacing over all aligned recorded samples; diagonal is masked.")
