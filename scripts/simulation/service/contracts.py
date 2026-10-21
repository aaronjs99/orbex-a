"""Explicit live protocol, parameter limits and worker input contracts."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from orbexa.layout import PACKAGE_ROOT


LIVE_SERVICE = 'orbexa-live-v2'


LIVE_PROTOCOL_VERSION = 2


ALLOWED_ORIGINS = frozenset((
    "http://localhost:8879", "http://127.0.0.1:8879",
    "http://localhost:8881", "http://127.0.0.1:8881",
    "https://orbex-a.aaronjs.me",
    "https://orbex-a-mission-studio.aaronjs99.chatgpt.site",
))


PARAMETER_LIMITS = dict(release_distance_m=(100., 1000.),
    cluster_spacing_m=(1.4, 3.), target_spin_deg_s=(.2, 3.))


@dataclass(frozen=True)
class LiveParameters:
    release_distance_m: float = 300.
    cluster_spacing_m: float = 1.8
    target_spin_deg_s: float = 2.4

    def __post_init__(self):
        for name, (lower, upper) in PARAMETER_LIMITS.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"{name} must be a finite number in [{lower}, {upper}]")
            object.__setattr__(self, name, float(value))

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) - set(PARAMETER_LIMITS):
            raise ValueError("Supply only release_distance_m, cluster_spacing_m and target_spin_deg_s")
        return cls(**value)

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class LiveServiceSettings:
    output_dir: Path
    config_path: Path
    prior_path: Path
    admission_path: Path
    deployment_path: Path
    spacecraft_path: Path
    maximum_updates: int = 1200
    resume_path: Path | None = None
    historical_analytical_resume: bool = False

    def __post_init__(self):
        for name in ("output_dir", "config_path", "prior_path", "admission_path", "deployment_path", "spacecraft_path"):
            path = Path(getattr(self, name)).expanduser().resolve()
            if name != "output_dir" and not path.is_file():
                raise ValueError(f"Missing explicit {name}: {path}")
            object.__setattr__(self, name, path)
        if self.resume_path is not None:
            path = Path(self.resume_path).expanduser().resolve()
            if not path.is_file():
                raise ValueError(f"Missing explicit resume record: {path}")
            object.__setattr__(self, 'resume_path', path)
        if not isinstance(self.historical_analytical_resume, bool) or bool(self.resume_path) != self.historical_analytical_resume:
            raise ValueError('Saved analytical resume requires the explicit historical_analytical_resume flag and a resume path')
        if isinstance(self.maximum_updates, bool) or not isinstance(self.maximum_updates, int) or not 1 <= self.maximum_updates <= 1200:
            raise ValueError("maximum_updates must be an integer in [1, 1200]")
        checkout = PACKAGE_ROOT.parent
        if (checkout / "pyproject.toml").is_file() and self.output_dir.is_relative_to(checkout):
            raise ValueError("Live outputs must be outside the maintained checkout")
