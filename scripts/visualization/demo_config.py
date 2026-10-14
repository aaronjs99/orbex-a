"""Validated configuration for fresh mission runs or saved-data rendering."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from orbexa.sessions import validate_session_id

@dataclass(frozen=True)
class DemoConfig:
    """Configuration for ADTMPC mission artifact generation."""

    output_dir: Path = Path("results")
    data_dir: Path = Path("data")
    session_id: Optional[str] = None
    config_path: Optional[Path] = None
    steps: int = 450
    """Maximum number of receding-horizon MPC updates."""
    mission: str = "all"
    primary_solver: str = "gekko"
    from_data: bool = False
    clean_generated: bool = False
    require_primary_success: bool = True
    fps: int = 8

    def __post_init__(self) -> None:
        for name in ("steps", "fps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.mission not in ("single", "multi", "all"):
            raise ValueError(f"Unknown mission: {self.mission}")
        if self.primary_solver != "gekko":
            raise ValueError("ADTMPC mission generation requires the nonlinear GEKKO controller")
        if self.session_id is not None:
            validate_session_id(self.session_id, allow_latest=self.from_data)
