"""Compose session selection, result loading/running and artifact publication."""
from __future__ import annotations
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote
from orbexa.contracts import PHYSICAL_MODEL_ID, COUPLED_CONTROLLER_ID
from orbexa.layout import PACKAGE_ROOT, project_root
from orbexa.sessions import MissionSession
from orbexa.simulation.mission_result import load_adtmpc_mission_result
from orbexa.utils.io_utils import save_data
from orbexa.visualization.demo_config import DemoConfig
from orbexa.visualization.manifest import ADTMPCMissionManifest
from orbexa.visualization.mission_catalog import MissionCatalog, require_current_result
from orbexa.visualization.mission_gallery import MissionGallery
from orbexa.visualization.mission_renderer import ADTMPCMissionRenderer

logger = logging.getLogger(__name__)


class OrbexaDemo:
    """Public entrypoint used by the ORBEX-A CLI."""

    _read_gallery_catalog = staticmethod(MissionCatalog.read)
    _gallery_entry = staticmethod(MissionCatalog.entry)
    _require_current_result = staticmethod(require_current_result)

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
        session.preflight([(mission, "nonlinear") for mission in self._missions()])
        gallery_entries = self._read_gallery_catalog(output_session_dir, data_session_dir)
        if self.config.from_data:
            # Read every requested family before changing any rendered artifacts or notes.
            self._saved_results = {}
            for mission in self._missions():
                result = load_adtmpc_mission_result(data_session_dir / mission / "nonlinear")
                if result.mission != mission:
                    raise ValueError(f"Saved data does not match {mission}/nonlinear")
                self._require_current_result(result)
                self._saved_results[(mission, "nonlinear")] = result
        else:
            from orbexa.core.config import SimulationConfig
            self._mission_config = SimulationConfig.load(self.config.config_path)
            if self.config.clean_generated:
                clean_generated_outputs(session.output_root, session.data_root, scenario_path=self.config.config_path)
            session.create()

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
        for manifest in manifests:
            entry = self._gallery_entry(manifest, output_session_dir)
            gallery_entries[entry["relative_dir"]] = entry
        entries = list(gallery_entries.values())
        MissionGallery(output_session_dir).render(entries)
        if not save_data(output_session_dir / "gallery.json", {
            "schema": "orbexa.mission.gallery.v1", "entries": entries,
        }):
            raise OSError("Could not save mission gallery catalog")
        preview_mission = self._link_session_demo_gif(output_session_dir, entries)
        self._write_run_readme(output_session_dir, data_session_dir, session.name, preview_mission=preview_mission)
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


    def _write_run_readme(
        self, output_session_dir: Path, data_session_dir: Path, session_id: str,
        *, preview_mission: Optional[str] = None,
    ) -> None:
        """Create editable notes, linking the preview from each README's own root."""
        for directory in (output_session_dir, data_session_dir):
            directory.mkdir(parents=True, exist_ok=True)
            preview_href = None
            if preview_mission is not None:
                try:
                    preview_href = Path(os.path.relpath(output_session_dir / "demo.gif", directory)).as_posix()
                except ValueError:
                    pass
            content = self._run_readme_template(session_id, preview_href=preview_href)
            readme = directory / "README.md"
            if not readme.exists():
                readme.write_text(content, encoding="utf-8")
                continue
            previous = readme.read_text(encoding="utf-8")
            revised = []
            for line in previous.splitlines(keepends=True):
                if (line.strip() == "![Multi-agent ADTMPC demo](demo.gif)" or
                        line.startswith("![ADTMPC trajectory preview]") or
                        line.strip() == "No trajectory preview was generated for the available mission families."):
                    line = self._preview_line(preview_href) + "\n"
                revised.append(line)
            updated = "".join(revised)
            if updated != previous:
                readme.write_text(updated, encoding="utf-8")

    @staticmethod
    def _preview_line(preview_href: Optional[str]) -> str:
        return (f"![ADTMPC trajectory preview]({quote(preview_href, safe='/%')})" if preview_href else
                "No trajectory preview was generated for the available mission families.")

    def _run_readme_template(self, session_id: str, *, preview_href: Optional[str] = None) -> str:
        missions = ", ".join(self._missions())
        return f"""# ADTMPC Run Notes

{self._preview_line(preview_href)}

Session: `{session_id}`
Created: `{datetime.now().isoformat(timespec="seconds")}`

## Run Summary

- Mission(s): `{missions}`
- Primary solver: `{self.config.primary_solver}`
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


    def _link_session_demo_gif(self, output_session_dir: Path, entries=None) -> Optional[str]:
        """Expose an available qualified nonlinear family's trajectory preview."""
        if entries is None:
            entries = self._read_gallery_catalog(output_session_dir).values()
        available = {entry["mission"]: entry for entry in entries
                     if entry.get("mission") in ("single", "multi") and
                     entry.get("approximation") == "nonlinear" and entry.get("backend") == "gekko" and
                     entry.get("model_version") == PHYSICAL_MODEL_ID and
                     entry.get("controller") == COUPLED_CONTROLLER_ID}
        source, selected = None, None
        for mission in ("multi", "single"):
            entry = available.get(mission)
            candidate = output_session_dir / mission / "nonlinear" / "trajectory.gif"
            MissionSession._check_path(candidate, output_session_dir)
            if entry is not None and "trajectory.gif" not in entry.get("skipped_artifacts", {}) and candidate.is_file():
                source, selected = candidate, mission
                break
        if source is None:
            return None
        link = output_session_dir / "demo.gif"
        if link.is_symlink() or link.is_file():
            link.unlink()
        elif link.exists():
            logger.warning("Preserving existing directory at %s", link)
            return None
        try:
            link.symlink_to(source.relative_to(output_session_dir))
        except OSError:
            shutil.copy2(source, link)
        return selected

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
        if approximation != "nonlinear" or solver_backend != "gekko":
            raise ValueError("ADTMPC mission generation requires nonlinear GEKKO")
        if self.config.from_data:
            result = self._saved_results[(mission, approximation)]
            data_file = data_family_dir / "mission_data.json"
        else:
            from orbexa.simulation.adtmpc_mission import ADTMPCMissionRunner
            runner = ADTMPCMissionRunner(
                config=self._mission_config,
                solver_backend=solver_backend,
                approximation=approximation,
            )
            result = runner.run(mission=mission, steps=self.config.steps)
            data_file = runner.save_result(result, data_family_dir)
        self._require_current_result(result)
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
    checkout = project_root()
    protected = [PACKAGE_ROOT] + [checkout / name for name in ("config", "paper", "assets", ".git")]
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
