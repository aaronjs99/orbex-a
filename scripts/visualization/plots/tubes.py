"""Scientific plots of each chaser's own matched dynamic-tube samples.

Only recorded measurements are displayed. Empty legacy diagnostics are skipped;
interval diagnostics use interval endpoints and are never inferred from averaged
controls. A small norm error alone does not establish componentwise containment.
"""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
import re

import matplotlib.pyplot as plt
from orbexa.visualization.plots.canvas import artifact_figure, artifact_subplots
import numpy as np

from orbexa.visualization.data.samples import MissionSamples
from orbexa.visualization.style import _INK, chaser_color


_BLUE = "#2467A3"
_TEAL = "#008B81"
_AMBER = "#CB8B2A"
_RED = "#BA4D53"
_GRAY = "#76879A"
_COMPONENTS = (_BLUE, _TEAL, _AMBER)


def _get(source, key, default=None):
    return source.get(key, default) if isinstance(source, Mapping) else getattr(source, key, default)


def _array(value):
    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return np.empty(0)


def _ratio(error, bound):
    """Zero error in a zero-width tube is contained; nonzero error is not."""
    result = np.full_like(error, np.inf)
    np.divide(np.abs(error), bound, out=result, where=bound > 0.0)
    result[(bound == 0.0) & (error == 0.0)] = 0.0
    return np.max(result, axis=1)


class MatchedTubeDiagnostics:
    """Render per-chaser executive and component plots from saved results.

    ``render()`` returns ``[(path, human_readable_label), ...]`` for an artifact
    manifest. ``skipped_artifacts`` explains missing or incomplete legacy data.
    Each position/sliding error is paired only with its own chaser and node.
    """

    REQUIRED_FIELDS = {"matched_tube_diagnostics": ("anom_history", "tube_diagnostics.<chaser>.nominal_center/position_error/sliding_error/omega/phi/bandwidth")}

    def __init__(self, result, output_dir: Path, *, dpi: int = 180, samples=None):
        self.result = result
        self.samples = samples if samples is not None else MissionSamples(result)
        self.output_dir = Path(output_dir)
        self.dpi = int(dpi)
        self.metadata = _get(result, "metadata", {}) or {}
        self.q = _array(_get(result, "anom_history", []))
        elapsed = _array(self.metadata.get("elapsed_time_seconds", []))
        self.x = elapsed if elapsed.shape == self.q.shape and np.all(np.isfinite(elapsed)) else self.q
        self.xlabel = "Elapsed time (s)" if self.x is elapsed else "True anomaly (rad)"
        self.skipped_artifacts = {}

    def render(self):
        self.skipped_artifacts = {}
        diagnostics = _get(self.result, "tube_diagnostics", {}) or {}
        if not diagnostics:
            self.skipped_artifacts["matched_tube_diagnostics"] = "No matched tube diagnostics were recorded in this result."
            return []
        self.output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        style = {
            "font.family": "DejaVu Sans", "font.size": 9,
            "axes.labelcolor": _INK, "text.color": _INK,
            "axes.edgecolor": "#D5DDE5", "axes.titleweight": "semibold",
            "axes.spines.top": False, "axes.spines.right": False,
            "grid.color": "#DCE3EA", "grid.alpha": 0.6,
            "figure.facecolor": "#F8FAFC", "axes.facecolor": "white",
            "savefig.facecolor": "#F8FAFC", "legend.frameon": False,
        }
        with plt.rc_context(style):
            for chaser, entry in diagnostics.items():
                if not isinstance(entry, Mapping):
                    slug = re.sub(r"[^A-Za-z0-9_-]", "_", str(chaser))
                    self.skipped_artifacts[f"matched_tube_{slug}"] = "Matched node diagnostics are missing, nonfinite, or misaligned with the recorded anomaly history."
                    continue
                matched = {key: _array(entry.get(key, [])) for key in
                           ("nominal_center", "position_error", "sliding_error", "omega", "phi", "bandwidth")}
                valid = len(self.q) > 0 and all(
                    value.ndim == 2 and len(value) == len(self.q) and value.shape[1] >= 3
                    and np.all(np.isfinite(value)) for value in matched.values()
                )
                slug = re.sub(r"[^A-Za-z0-9_-]", "_", str(chaser))
                if not valid:
                    self.skipped_artifacts[f"matched_tube_{slug}"] = "Matched node diagnostics are missing, nonfinite, or misaligned with the recorded anomaly history."
                    continue
                paths.append((self._executive(chaser, slug, entry, matched),
                              f"{chaser}: matched tube, terminal approach, applied input and physical safety diagnostics."))
                paths.append((self._components(chaser, slug, matched),
                              f"{chaser}: signed component errors inside their own matched position and sliding tubes."))
        return paths

    def _outcome(self):
        complete = bool(self.metadata.get("mission_complete", _get(self.result, "success", False)))
        if complete and _get(self.result, "success", False):
            return "DOCKING COMPLETED", _TEAL
        if not _get(self.result, "success", False):
            return "RUN STOPPED / NOT COMPLETED", _RED
        return "RECORDED RUN / NOT COMPLETED", _AMBER

    def _decorate(self, fig, axes, chaser, title):
        outcome, color = self._outcome()
        metadata_y = 0.925 if fig.get_figheight() < 10 else 0.951
        fig.suptitle(title, x=0.055, y=0.985, ha="left", fontsize=18, fontweight="bold")
        fig.text(0.055, metadata_y, f"{chaser}  /  {_get(self.result, 'solver_backend', 'saved run')}  /  {len(self.q)} recorded samples",
                 fontsize=10, color=_GRAY)
        fig.text(0.945, metadata_y + 0.003, outcome, ha="right", fontsize=9, fontweight="bold", color=color)
        phases = _get(self.result, "sample_phase_history", [])
        for ax in np.asarray(axes).flat:
            ax.grid(True, linewidth=0.7)
            ax.tick_params(colors=_GRAY, labelsize=8)
            ax.set_xlabel(self.xlabel)
            if len(self.x) > 1:
                ax.set_xlim(self.x[0], self.x[-1])
            if len(phases) == len(self.x):
                for index in range(1, len(phases)):
                    if phases[index] != phases[index - 1]:
                        ax.axvline(self.x[index], color=_GRAY, linewidth=0.9, linestyle=":", alpha=0.7)
        message_lines = str(_get(self.result, "message", "")).splitlines()
        message = message_lines[0] if message_lines else (
            "No recorded controller interruptions." if _get(self.result, "success", False)
            else "No outcome explanation was recorded.")
        fig.text(0.055, 0.027, message[:150], color=color, fontsize=8)
        attitude_note = (
            " Target attitude is prescribed in anomaly."
            if "prescribed" in str(self.metadata.get("target_attitude_model", "")).lower() else ""
        )
        fig.text(0.055, 0.012,
                 "Matched samples and dense substep diagnostics; conditional model enclosure, not a continuous-time safety certificate." + attitude_note,
                 color=_GRAY, fontsize=7.5)

    def _interval(self, entry, key):
        value = _array(entry.get(key, []))
        if value.ndim >= 1 and len(value) == max(0, len(self.x) - 1) and np.all(np.isfinite(value)):
            return value
        return np.empty(0)

    @staticmethod
    def _legend(ax, *, loc="best"):
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc=loc, fontsize=8)

    @staticmethod
    def _missing(ax, message):
        ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center", color=_GRAY, fontsize=9)

    def _pose_error(self, chaser, matched):
        actual = _array((_get(self.result, "actual_trajectories", {}) or {}).get(chaser, []))
        attitudes = self.samples.orientation
        body = _array((_get(self.result, "docking_points", {}) or {}).get(chaser, []))
        if actual.ndim != 2 or len(actual) != len(self.x) or attitudes is None or attitudes.shape != (len(self.x), 3) or body.shape != (3,):
            return None, None
        goals = np.einsum("tij,j->ti", self.samples.rotations, body)
        return np.linalg.norm(actual[:, :3] - goals, axis=1), np.linalg.norm(matched["nominal_center"][:, :3] - goals, axis=1)

    def _executive(self, chaser, slug, entry, matched):
        fig, axes = artifact_subplots(3, 2, figsize=(14, 12.3))
        fig.subplots_adjust(left=0.075, right=0.91, top=0.91, bottom=0.09, hspace=0.48, wspace=0.4)
        self._decorate(fig, axes, chaser, "Dynamic tube / mission readout")
        error, omega = matched["position_error"][:, :3], matched["omega"][:, :3]
        sliding, phi = matched["sliding_error"][:, :3], matched["phi"][:, :3]
        ax = axes[0, 0]
        ax.set_title("01  Position tracking against its own tube", loc="left", fontsize=11)
        ax.plot(self.x, 1000 * np.linalg.norm(error, axis=1), color=_BLUE, linewidth=1.8, label="Actual − matched nominal")
        ax.plot(self.x, 1000 * np.linalg.norm(omega, axis=1), color=_TEAL, linewidth=1.5, linestyle="--", label="Own position-tube radius ‖Ω‖")
        ax.set_ylabel("Position norm (mm)")
        ax.set_ylim(bottom=0.0)
        self._legend(ax)

        ax = axes[0, 1]
        ax.set_title("02  Componentwise containment", loc="left", fontsize=11)
        for values, color, label in ((_ratio(error, omega), _BLUE, "max |position error| / Ω"),
                                     (_ratio(sliding, phi), _TEAL, "max |sliding error| / Φ")):
            finite = np.isfinite(values)
            ax.plot(self.x, np.where(finite, values, np.nan), color=color, linewidth=1.5, label=label)
            if not np.all(finite):
                ax.text(0.02, 0.97, "Nonzero error with zero bound recorded", transform=ax.transAxes, va="top", color=_RED, fontsize=8)
        ax.axhline(1.0, color=_RED, linewidth=1.1, linestyle="--", label="Containment limit")
        dense_margin = self._interval(entry, "dense_containment_margin")
        if len(dense_margin):
            ax.text(0.02, 0.03, f"Dense minimum margin: {np.min(dense_margin):.3g}", transform=ax.transAxes,
                    color=_TEAL if np.min(dense_margin) >= 0 else _RED, fontsize=8)
        ax.set_ylabel("Fraction of own component bound")
        self._legend(ax)

        ax = axes[1, 0]
        ax.set_title("03  Terminal position set and actual pose error", loc="left", fontsize=11)
        terminal = self._interval(entry, "terminal_position_bound")
        if len(terminal):
            ax.plot(self.x[1:], terminal * 1000, color=_TEAL, linewidth=1.6, label="Nominal goal distance + own tube radius")
        actual_pose, nominal_pose = self._pose_error(chaser, matched)
        if actual_pose is not None:
            ax.plot(self.x, actual_pose * 1000, color=_BLUE, linewidth=1.4, label="Actual distance to rotating docking goal")
            ax.plot(self.x, nominal_pose * 1000, color=_GRAY, linewidth=1.0, linestyle=":", label="Matched nominal goal distance")
        tolerance = self.metadata.get("docking_position_tolerance")
        if tolerance is not None and np.isfinite(tolerance) and tolerance > 0:
            ax.axhline(1000 * tolerance, color=_RED, linestyle="--", linewidth=1.1, label=f"Position tolerance: {1000*tolerance:g} mm")
        ax.set_yscale("symlog", linthresh=0.2)
        ax.set_ylim(bottom=0.0)
        ax.set_ylabel("Goal distance / bound (mm)")
        self._legend(ax, loc="lower left")
        if len(terminal) > 2 and actual_pose is not None:
            zoom = ax.inset_axes([0.57, 0.53, 0.4, 0.4])
            tail = min(len(terminal), max(8, len(terminal) // 6))
            zoom.plot(self.x[-tail:], terminal[-tail:] * 1000, color=_TEAL, linewidth=1.2)
            zoom.plot(self.x[-tail:], actual_pose[-tail:] * 1000, color=_BLUE, linewidth=1.1)
            if tolerance is not None and np.isfinite(tolerance) and tolerance > 0:
                zoom.axhline(1000 * tolerance, color=_RED, linestyle="--", linewidth=0.9)
            zoom.set_ylim(bottom=0.0)
            zoom.set_title("Final recorded approach (mm)", fontsize=7, loc="left")
            zoom.tick_params(labelsize=6, colors=_GRAY)
            zoom.grid(True, alpha=0.4, linewidth=0.5)

        ax = axes[1, 1]
        ax.set_title("04  Optimized tube bandwidth", loc="left", fontsize=11)
        for component, color in enumerate(_COMPONENTS):
            ax.plot(self.x, matched["bandwidth"][:, component], color=color, linewidth=1.4, label="xyz"[component])
        ax.set_ylabel("Bandwidth (1/rad anomaly)")
        self._legend(ax)

        ax = axes[2, 0]
        ax.set_title("05  Dense applied input and reserve use", loc="left", fontsize=11)
        effort = self._interval(entry, "applied_control_effort")
        if len(effort):
            ax.plot(self.x[1:], np.cumsum(effort), color=_BLUE, linewidth=1.7, label="Cumulative dense applied effort")
        else:
            self._missing(ax, "Dense applied effort was not recorded")
        ax.set_ylabel("∫ ‖applied input‖² dq  [m²/rad³]")
        reserve = self._interval(entry, "max_dense_input_reserve_ratio")
        reserve_axis = ax.twinx()
        reserve_axis.spines["right"].set_visible(True)
        reserve_axis.set_ylabel("Ancillary reserve used", color=_AMBER)
        reserve_axis.tick_params(axis="y", colors=_AMBER, labelsize=8)
        reserve_axis.axhline(1.0, color=_RED, linewidth=1.0, linestyle="--")
        if len(reserve):
            reserve_axis.plot(self.x[1:], reserve, color=_AMBER, linewidth=1.1, label="Maximum dense reserve ratio")
            reserve_axis.set_ylim(bottom=0.0, top=max(1.1, float(np.nanmax(reserve)) * 1.1))
        applied_rms, ancillary_rms = self._interval(entry, "applied_control_rms"), self._interval(entry, "ancillary_control_rms")
        if applied_rms.ndim == 2 and ancillary_rms.ndim == 2 and len(applied_rms) and len(ancillary_rms):
            ax.text(0.02, 0.03,
                    f"Final applied / ancillary RMS norm: {np.linalg.norm(applied_rms[-1]):.3g} / {np.linalg.norm(ancillary_rms[-1]):.3g} m/rad²",
                    transform=ax.transAxes, fontsize=7.5, color=_GRAY)
        handles, labels = ax.get_legend_handles_labels()
        extra_handles, extra_labels = reserve_axis.get_legend_handles_labels()
        ax.legend(handles + extra_handles, labels + extra_labels, fontsize=7.5, loc="upper left")

        ax = axes[2, 1]
        ax.set_title("06  Dense physical safety and chaser spacing", loc="left", fontsize=11)
        clearance = self._interval(entry, "dense_physical_clearance")
        if len(clearance):
            ax.plot(self.x[1:], clearance, color=_TEAL, linewidth=1.5, label="Minimum dense physical constraint margin")
        else:
            self._missing(ax, "Dense physical margins were not recorded")
        ax.axhline(0.0, color=_RED, linewidth=1.0, linestyle="--")
        ax.set_ylabel("Physical constraint margin (model units)")
        spacing = _get(self.result, "pairwise_spacing", [])
        if len(spacing) == len(self.x) and any(spacing):
            minimum = np.array([min(row.values()) if row else np.nan for row in spacing])
            spacing_axis = ax.twinx()
            spacing_axis.spines["right"].set_visible(True)
            spacing_axis.plot(self.x, minimum, color=_BLUE, linewidth=1.1, label="Minimum endpoint chaser spacing")
            spacing_axis.set_ylabel("Endpoint spacing (m)", color=_BLUE)
            separation = self.metadata.get("min_chaser_separation")
            if separation is not None:
                spacing_axis.axhline(separation, color=_BLUE, linestyle=":", linewidth=0.9)
            self._legend(spacing_axis, loc="upper right")
        dense_spacing = self.metadata.get("dense_min_spacing")
        if dense_spacing is not None and np.isfinite(dense_spacing):
            ax.text(0.02, 0.03, f"Whole-run dense minimum spacing: {dense_spacing:.4g} m", transform=ax.transAxes, color=_GRAY, fontsize=8)
        self._legend(ax, loc="upper left")
        path = self.output_dir / f"matched_tube_{slug}.png"
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        return path

    def _components(self, chaser, slug, matched):
        fig, axes = artifact_subplots(2, 3, figsize=(14, 8))
        fig.subplots_adjust(left=0.075, right=0.96, top=0.87, bottom=0.12, hspace=0.48, wspace=0.32)
        self._decorate(fig, axes, chaser, "Matched component containment")
        for row, (error_key, bound_key, quantity, unit, scale) in enumerate((
            ("position_error", "omega", "Position error", "mm", 1000.0),
            ("sliding_error", "phi", "Sliding error", "m/rad anomaly", 1.0),
        )):
            for component, color in enumerate(_COMPONENTS):
                ax = axes[row, component]
                bound = matched[bound_key][:, component] * scale
                error = matched[error_key][:, component] * scale
                ax.fill_between(self.x, -bound, bound, color=color, alpha=0.12, linewidth=0)
                ax.plot(self.x, bound, color=color, linestyle="--", linewidth=1.1, label=f"± own {'Ω' if row == 0 else 'Φ'}")
                ax.plot(self.x, -bound, color=color, linestyle="--", linewidth=1.1)
                ax.plot(self.x, error, color=_INK, linewidth=1.2, label="Signed matched error")
                violations = np.abs(error) > bound
                if np.any(violations):
                    ax.scatter(self.x[violations], error[violations], color=_RED, s=12, zorder=5, label="Outside recorded bound")
                ax.axhline(0.0, color=_GRAY, linewidth=0.6)
                ax.set_title(f"{'xyz'[component]}  /  {quantity}", loc="left", fontsize=11)
                ax.set_ylabel(f"{quantity} ({unit})")
                self._legend(ax)
        path = self.output_dir / f"matched_tube_components_{slug}.png"
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        return path
