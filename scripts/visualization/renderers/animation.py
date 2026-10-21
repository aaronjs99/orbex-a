"""One recorded-sample animation scene shared by MP4 and GIF encoders."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from orbexa.visualization.plots.canvas import artifact_figure, artifact_subplots
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
import numpy as np

from orbexa.visualization.data.geometry import MissionGeometry
from orbexa.visualization.data.samples import MissionSamples
from orbexa.visualization.style import chaser_color

logger = logging.getLogger(__name__)


class MissionAnimationScene:
    """Draw a common recorded state and target pose with all assigned goals."""

    def __init__(self, samples: MissionSamples, geometry: MissionGeometry, *, figsize=(8, 7)):
        if not len(samples.q):
            raise ValueError("A recorded animation scene requires aligned state samples")
        self.samples, self.geometry = samples, geometry
        self.figure = artifact_figure(figsize=figsize)
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.lines, self.points = {}, {}
        for index, identifier in enumerate(samples.ids):
            color = chaser_color(index)
            self.lines[identifier], = self.ax.plot([], [], [], label=identifier, color=color)
            self.points[identifier], = self.ax.plot([], [], [], marker="o", color=color)
        geometry.plot_sphere(self.ax)
        self.target = geometry.plot_target(self.ax, 0)
        self.goals, = self.ax.plot([], [], [], linestyle="", marker="D", color="#D55E00", label="docking points")
        center, extent = geometry.limits()
        self.ax.set(xlim=(center[0] - extent, center[0] + extent), ylim=(center[1] - extent, center[1] + extent),
                    zlim=(center[2] - extent, center[2] + extent), xlabel="x", ylabel="y", zlabel="z")
        self.ax.set_box_aspect((1, 1, 1))
        self.ax.legend(fontsize=8)
        notes = list(samples.notes)
        if self.target is None:
            notes.append("Saved target geometry or pose unavailable; target omitted.")
        if notes:
            self.figure.text(0.5, 0.025, " ".join(notes), ha="center", fontsize=7, wrap=True)
        self.update(0)

    def update(self, index: int):
        if index < 0 or index >= len(self.samples.q):
            raise IndexError("Animation index must refer to a common recorded sample")
        if self.target is not None:
            self.target.remove()
        self.target = self.geometry.plot_target(self.ax, index, label=False)
        goals = self.geometry.docking_points(index)
        values = np.asarray(list(goals.values())) if goals else np.empty((0, 3))
        self.goals.set_data(values[:, 0], values[:, 1])
        self.goals.set_3d_properties(values[:, 2])
        for number, identifier in enumerate(self.samples.ids):
            trail = self.samples.states[:index + 1, number, :3]
            point = trail[-1]
            self.lines[identifier].set_data(trail[:, 0], trail[:, 1])
            self.lines[identifier].set_3d_properties(trail[:, 2])
            self.points[identifier].set_data([point[0]], [point[1]])
            self.points[identifier].set_3d_properties([point[2]])
        phase = self.samples.phases[index]
        self.ax.set_title(f"Recorded sample {index + 1}/{len(self.samples.q)} · {phase} · q={self.samples.q[index]:.4f}")
        return list(self.lines.values()) + list(self.points.values()) + [self.goals] + ([self.target] if self.target is not None else [])

    def close(self):
        plt.close(self.figure)


class MissionAnimation:
    REQUIRED_FIELDS = {name: ("anom_history", "actual_trajectories") for name in ("trajectory.mp4", "trajectory.gif")}

    def __init__(self, result, output_dir: Path, *, fps: int = 8, samples=None, geometry=None):
        if isinstance(fps, bool) or not isinstance(fps, int) or fps < 1:
            raise ValueError("fps must be a positive integer")
        self.samples = samples if samples is not None else MissionSamples(result)
        self.geometry = geometry if geometry is not None else MissionGeometry(self.samples)
        self.output_dir, self.fps = Path(output_dir), fps
        self.artifacts, self.skipped_artifacts, self.provenance = [], {}, {}

    def render(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts, self.skipped_artifacts, self.provenance = [], {}, {}
        if not len(self.samples.q):
            for filename in self.REQUIRED_FIELDS:
                self.skipped_artifacts[filename] = "No recorded trajectory samples are available."
            return []
        self.video()
        self.gif(speed_multiplier=2.0)
        return list(self.artifacts)

    def _encode(self, filename, frames, writer, *, fps, dpi, figsize, label=None):
        scene = MissionAnimationScene(self.samples, self.geometry, figsize=figsize)
        path = self.output_dir / filename
        animation = FuncAnimation(scene.figure, scene.update, frames=frames, interval=1000 / fps, blit=False)
        try:
            animation.save(path, writer=writer, dpi=dpi)
        except Exception as exc:
            # Optional encoders may fail independently of valid state artifacts.
            path.unlink(missing_ok=True)
            self.skipped_artifacts[filename] = f"Animation encoding failed ({type(exc).__name__}): {exc}"
            logger.warning("%s: %s", filename, self.skipped_artifacts[filename])
        else:
            self.artifacts.append((path, label))
            self.provenance[filename] = {"source_sample_indices": frames, "recorded_common_samples": len(self.samples.q),
                                         "playback_frames_per_second": fps, "interpolated": False}
        finally:
            scene.close()

    def video(self):
        encoder = shutil.which("ffmpeg")
        if encoder is None:
            try:
                from imageio_ffmpeg import get_ffmpeg_exe
                encoder = get_ffmpeg_exe()
            except (ImportError, RuntimeError, OSError):
                encoder = None
        if encoder is None:
            reason = "FFmpeg is unavailable; no current MP4 was generated. The trajectory GIF remains available."
            self.skipped_artifacts["trajectory.mp4"] = reason
            logger.warning(reason)
            return
        with mpl.rc_context({"animation.ffmpeg_path": encoder}):
            self._encode("trajectory.mp4", self.samples.frame_indices(), FFMpegWriter(fps=self.fps), fps=self.fps,
                         dpi=130, figsize=(8, 7))

    def gif(self, *, speed_multiplier: float = 2.0):
        if not np.isfinite(speed_multiplier) or speed_multiplier <= 0:
            raise ValueError("speed_multiplier must be finite and positive")
        fps = max(1, int(round(self.fps * speed_multiplier)))
        self._encode("trajectory.gif", self.samples.frame_indices(80), PillowWriter(fps=fps), fps=fps,
                     dpi=100, figsize=(6.5, 5.6), label=f"Trajectory GIF preview at {speed_multiplier:g}x playback speed.")
