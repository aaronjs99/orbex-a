"""Portable, non-overwriting paths and latest references for mission sessions."""

from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
import re
from typing import Optional

logger = logging.getLogger(__name__)
LATEST_SCHEMA = "orbexa.session.latest.v1"


def validate_session_id(value: str, *, allow_latest: bool = False) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}", value):
        raise ValueError("session-id must be a single folder name using letters, digits, dots, underscores, or hyphens")
    stem = value.split(".", 1)[0].upper()
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if value.endswith(".") or stem in reserved or value.lower() == "latest.json" or (value.lower() == "latest" and not allow_latest):
        raise ValueError("session-id is reserved; choose another folder name")
    return value


@dataclass(frozen=True)
class MissionSession:
    output_root: Path
    data_root: Path
    name: str

    @property
    def output_dir(self) -> Path:
        return self.output_root / self.name

    @property
    def data_dir(self) -> Path:
        return self.data_root / self.name

    @classmethod
    def select(cls, output_root: Path, data_root: Path, *, session_id: Optional[str],
               from_data: bool, mission: str, solver: str) -> "MissionSession":
        output_root = Path(output_root).expanduser().resolve()
        data_root = Path(data_root).expanduser().resolve()
        if output_root == data_root or output_root in data_root.parents or data_root in output_root.parents:
            raise ValueError("output and data-output must be separate, non-overlapping folders")
        for root in (output_root, data_root):
            if root.exists() and not root.is_dir():
                raise ValueError(f"Session root must be a directory: {root}")
            if not from_data:
                cls._check_reference(root)
        if from_data and (session_id is None or session_id.lower() == "latest"):
            name = cls._latest_name(data_root)
        elif session_id is not None:
            name = validate_session_id(session_id)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            name = f"{timestamp}_{mission}_{solver}_adtmpc"
        return cls(output_root, data_root, name)

    @staticmethod
    def _check_reference(root: Path) -> None:
        reference = root / "latest.json"
        if reference.resolve() != reference:
            raise ValueError(f"Latest session reference must not be an alias: {reference}")
        if reference.exists():
            if not reference.is_file():
                raise ValueError(f"Latest session reference must be a file: {reference}")
            previous = json.loads(reference.read_text(encoding="utf-8"))
            if not isinstance(previous, dict) or previous.get("schema") != LATEST_SCHEMA:
                raise ValueError(f"Refusing to replace an unrecognized latest-session file: {reference}")

    @staticmethod
    def _latest_name(root: Path) -> str:
        reference = root / "latest.json"
        if reference.resolve() != reference:
            raise ValueError(f"Latest session reference must not be an alias: {reference}")
        if reference.is_file():
            payload = json.loads(reference.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("schema") != LATEST_SCHEMA:
                raise ValueError(f"Unrecognized latest session reference: {reference}; use --session-id NAME")
            name = validate_session_id(payload.get("session_id"))
            if not (root / name).is_dir():
                raise FileNotFoundError(f"Latest saved session is missing: {root / name}")
            return name
        legacy = root / "latest"
        if legacy.is_symlink():
            target = legacy.resolve()
            if target.parent != root or not target.is_dir():
                raise ValueError(f"Latest session link must point to a session inside {root}")
            return validate_session_id(target.name)
        if legacy.is_dir():
            return "latest"
        raise FileNotFoundError(f"No latest saved session in {root}; use --session-id NAME")

    @staticmethod
    def _check_path(path: Path, root: Path) -> None:
        resolved = path.resolve()
        if resolved != path or (resolved != root and root not in resolved.parents):
            raise ValueError(f"Session paths must stay inside {root} without symlinks or junctions: {path}")

    def preflight(self, families) -> None:
        """Reject directory and file aliases before any load, cleanup, or rendering."""
        for directory, root in ((self.output_dir, self.output_root), (self.data_dir, self.data_root)):
            self._check_path(directory, root)
        for mission, approximation in families:
            rendered = self.output_dir / mission / approximation
            raw = self.data_dir / mission / approximation
            self._check_path(rendered, self.output_root)
            self._check_path(raw, self.data_root)
            self._check_path(raw / "mission_data.json", self.data_root)
            if rendered.is_dir():
                for path in rendered.rglob("*"):
                    self._check_path(path, self.output_root)
        for directory in (self.output_dir, self.data_dir):
            self._check_path(directory / "README.md", directory.parent)
        preview = self.output_dir / "demo.gif"
        expected_preview = self.output_dir / "multi" / "nonlinear" / "trajectory.gif"
        if preview.is_symlink() and preview.resolve() == expected_preview.resolve():
            self._check_path(expected_preview, self.output_root)
        else:
            self._check_path(preview, self.output_root)

    def create(self) -> None:
        for directory in (self.output_dir, self.data_dir):
            if directory.exists() or directory.is_symlink():
                raise FileExistsError(f"Session already exists: {directory}; choose a new --session-id")
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.data_dir.mkdir(parents=True, exist_ok=False)

    def publish(self) -> None:
        from orbexa.utils.io_utils import save_data

        for root in (self.output_root, self.data_root):
            self._check_reference(root)
            if not save_data(root / "latest.json", {"schema": LATEST_SCHEMA, "session_id": self.name}):
                raise OSError(f"Could not record latest session in {root}")
            link = root / "latest"
            try:
                if link.is_symlink():
                    link.unlink()
                elif link.exists():
                    logger.warning("Preserving %s; newest session is recorded in latest.json", link)
                    continue
                link.symlink_to(self.name, target_is_directory=True)
            except OSError:
                logger.info("Latest session recorded in %s (directory links unavailable)", root / "latest.json")
