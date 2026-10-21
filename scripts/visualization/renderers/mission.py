"""Compose independent artifact renderers from a plain saved mission record."""
from __future__ import annotations

from pathlib import Path

from orbexa.simulation.approach.records.results import ADTMPCMissionResult
from orbexa.visualization.data.geometry import MissionGeometry
from orbexa.visualization.data.manifest import ADTMPCMissionManifest, _raw_data_reference
from orbexa.visualization.renderers.animation import MissionAnimation
from orbexa.visualization.publishing.gallery import MissionGallery
from orbexa.visualization.plots.insights import MissionInsights
from orbexa.visualization.renderers.interactive import MissionInteractive
from orbexa.visualization.plots.mission import MissionPlots
from orbexa.visualization.renderers.replay import MissionReplay
from orbexa.visualization.data.samples import MissionSamples
from orbexa.visualization.plots.tubes import MatchedTubeDiagnostics


class ADTMPCMissionRenderer:
    """Create all supported artifacts, recording missing optional data per view."""

    def __init__(self, output_dir: Path, *, fps: int = 8):
        if isinstance(fps, bool) or not isinstance(fps, int) or fps < 1:
            raise ValueError("fps must be a positive integer")
        self.output_dir, self.fps = Path(output_dir), fps

    def render(self, result: ADTMPCMissionResult, *, data_file=None) -> ADTMPCMissionManifest:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = ADTMPCMissionManifest(
            output_dir=self.output_dir, mission=result.mission, approximation=result.approximation,
            solver_backend=result.solver_backend, data_file=None if data_file is None else Path(data_file),
            success=result.success, mission_complete=bool(result.metadata.get("mission_complete", result.success)),
            sample_count=len(result.anom_history), chaser_count=len(result.actual_trajectories), message=result.message,
            model_version=result.metadata.get("model_version", ""), controller=result.metadata.get("controller", ""),
        )
        if data_file is not None:
            manifest.add(Path(data_file))
        samples = MissionSamples(result)
        geometry = MissionGeometry(samples)
        animations = MissionAnimation(result, self.output_dir, fps=self.fps, samples=samples, geometry=geometry)
        insights = MissionInsights(result, self.output_dir, fps=self.fps, samples=samples)
        for renderer in (
            MissionPlots(result, self.output_dir, samples=samples, geometry=geometry),
            MissionInteractive(result, self.output_dir, samples=samples, geometry=geometry),
            animations, insights, MatchedTubeDiagnostics(result, self.output_dir, samples=samples),
        ):
            for path, label in renderer.render():
                manifest.add(path, label)
            manifest.skipped_artifacts.update(renderer.skipped_artifacts)
        # Keep encoder provenance alongside existing insight provenance, without
        # changing the saved result or artifact-manifest schema.
        if animations.provenance:
            insights.provenance["trajectory_animation"] = animations.provenance
            insights.write_manifest()
        manifest.add(MissionReplay(result, samples=samples, geometry=geometry).write(self.output_dir / "replay.html"),
                     "Animated replay of recorded states with target rotation and update slider.")
        self._render_index(result, manifest)
        manifest.write()
        return manifest

    def _render_index(self, result, manifest):
        entry = {
            "mission": result.mission, "approximation": result.approximation,
            "backend": result.solver_backend, "success": result.success,
            "mission_complete": manifest.mission_complete, "sample_count": manifest.sample_count,
            "chaser_count": manifest.chaser_count, "message": result.message, "relative_dir": ".",
            "artifacts": [{"path": path.name, "label": manifest.artifact_labels.get(path.name, path.stem.replace("_", " "))}
                          for path in manifest.artifacts if path.parent == self.output_dir],
            "skipped_artifacts": manifest.skipped_artifacts, "notes": [],
            **_raw_data_reference(manifest.data_file, self.output_dir),
        }
        manifest.add(MissionGallery(self.output_dir).render([entry]))
