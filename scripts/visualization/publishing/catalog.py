"""Qualification and migration of current-model mission gallery families."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Optional
from orbexa.sessions import MissionSession
from orbexa.contracts import PHYSICAL_MODEL_ID, COUPLED_CONTROLLER_ID
from orbexa.simulation.approach.records.results import ADTMPCMissionResult, load_adtmpc_mission_result
from orbexa.visualization.data.manifest import ADTMPCMissionManifest, _raw_data_reference

def require_current_result(result: ADTMPCMissionResult) -> None:
    if (result.approximation != "nonlinear" or result.solver_backend != "gekko" or
            result.metadata.get("model_version") != PHYSICAL_MODEL_ID or
            result.metadata.get("controller") != COUPLED_CONTROLLER_ID):
        raise ValueError("Saved mission data must use the current physical LVLH coupled ADTMPC controller")


class MissionCatalog:
    """Retain only qualified nonlinear GEKKO physical coupled ADTMPC families."""

    @staticmethod
    def read(
        output_session_dir: Path, data_session_dir: Optional[Path] = None,
    ) -> Dict[str, Dict]:
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
        families = ("single/nonlinear", "multi/nonlinear")
        rejected = set()

        def qualify(record, relative, backend_key):
            mission, approximation = relative.split("/")
            identity = {
                "mission": mission, "approximation": approximation, backend_key: "gekko",
                "model_version": PHYSICAL_MODEL_ID,
                "controller": COUPLED_CONTROLLER_ID,
            }
            # Missing fields may be recovered from raw evidence; contradictory
            # fields must never be overwritten merely because raw data exists.
            if (any(record.get(key) not in (None, "", value) for key, value in identity.items()) or
                    any(record.get(key) not in (None, "", "gekko") for key in ("backend", "solver_backend"))):
                return None, None
            raw_file = None
            if any(record.get(key) != value for key, value in identity.items()):
                if data_session_dir is None:
                    return None, None
                raw_file = data_session_dir / relative / "mission_data.json"
                MissionSession._check_path(raw_file, data_session_dir)
                if not raw_file.is_file():
                    return None, None
                try:
                    result = load_adtmpc_mission_result(raw_file.parent)
                    require_current_result(result)
                except (OSError, ValueError, KeyError, TypeError):
                    return None, None
                if result.mission != mission:
                    return None, None
            return {**record, **identity}, raw_file

        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"Invalid mission gallery entry in {path}")
            relative = entry.get("relative_dir")
            if relative not in families:
                continue
            MissionSession._check_path(output_session_dir / relative, output_session_dir)
            qualified, raw_file = qualify(entry, relative, "backend")
            if qualified is None:
                rejected.add(relative)
                catalog.pop(relative, None)
                continue
            if relative in rejected:
                continue
            if raw_file is not None:
                qualified.update(_raw_data_reference(raw_file, output_session_dir))
            catalog[relative] = qualified
        # Preserve only current qualified families during a partial render.
        for relative in families:
            if relative in catalog or relative in rejected:
                continue
            family = output_session_dir / relative
            manifest_path = family / "manifest.json"
            MissionSession._check_path(manifest_path, output_session_dir)
            if not manifest_path.is_file():
                continue
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("schema") != "orbexa.adtmpc_mission.manifest.v1":
                continue
            payload, raw_file = qualify(payload, relative, "solver_backend")
            if payload is None:
                continue
            artifacts = []
            for value in payload.get("artifacts", []):
                # Family artifacts are flat; bind their filenames to this session.
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
                "message": payload.get("message", ""),
                "relative_dir": relative, "artifacts": artifacts,
                "skipped_artifacts": payload.get("skipped_artifacts", {}),
                "model_version": payload["model_version"], "controller": payload["controller"],
                **({} if raw_file is None else _raw_data_reference(raw_file, output_session_dir)),
            }
        return catalog


    @staticmethod
    def entry(manifest: ADTMPCMissionManifest, root: Path) -> Dict:
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
            "model_version": manifest.model_version, "controller": manifest.controller,
            **_raw_data_reference(manifest.data_file, root),
        }
