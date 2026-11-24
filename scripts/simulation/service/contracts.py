"""Explicit live protocol, parameter limits and worker input contracts."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

from orbexa.layout import ProjectLayout

LIVE_SERVICE = "orbexa-live-v2"


LIVE_PROTOCOL_VERSION = 2


ALLOWED_ORIGINS = frozenset(
    (
        "http://localhost:8879",
        "http://127.0.0.1:8879",
        "http://localhost:8881",
        "http://127.0.0.1:8881",
        "https://orbex-a.aaronjs.me",
        "https://orbex-a-mission-studio.aaronjs99.chatgpt.site",
    )
)


PARAMETER_LIMITS = {
    "release_distance_m": (100.0, 1000.0),
    "cluster_spacing_m": (1.4, 3.0),
    "target_spin_deg_s": (0.2, 3.0),
}


@dataclass(frozen=True)
class LiveParameters:
    release_distance_m: float = 300.0
    cluster_spacing_m: float = 1.8
    target_spin_deg_s: float = 2.4

    def __post_init__(self):
        for name, (lower, upper) in PARAMETER_LIMITS.items():
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not lower <= value <= upper
            ):
                raise ValueError(
                    f"{name} must be a finite number in [{lower}, {upper}]"
                )
            object.__setattr__(self, name, float(value))

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) - set(PARAMETER_LIMITS):
            raise ValueError(
                "Supply only release_distance_m, cluster_spacing_m and target_spin_deg_s"
            )
        return cls(**value)

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class LiveServiceSettings:
    output_dir: Path
    config_path: Path
    spacecraft_path: Path
    maximum_updates: int = 1200

    def __post_init__(self):
        for name in (
            "output_dir",
            "config_path",
            "spacecraft_path",
        ):
            path = Path(getattr(self, name)).expanduser().resolve()
            if name != "output_dir" and not path.is_file():
                raise ValueError(f"Missing explicit {name}: {path}")
            object.__setattr__(self, name, path)
        if (
            isinstance(self.maximum_updates, bool)
            or not isinstance(self.maximum_updates, int)
            or not 1 <= self.maximum_updates <= 1200
        ):
            raise ValueError("maximum_updates must be an integer in [1, 1200]")
        ProjectLayout.current().require_output(self.output_dir, "Live outputs")
