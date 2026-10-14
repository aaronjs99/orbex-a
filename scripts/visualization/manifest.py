"""Stable artifact manifest schema and raw-data references."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from orbexa.utils.io_utils import save_data

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
    model_version: str = ""
    controller: str = ""

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
            "model_version": self.model_version,
            "controller": self.controller,
        }
        payload["required_entries"] = [name for name in payload["required_entries"] if name not in self.skipped_artifacts]
        if not save_data(path, payload):
            raise OSError(f"Could not save artifact manifest to {path}")
        self.add(path)
        return path
