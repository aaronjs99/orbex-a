"""layout.py: dependency-free paths for the checkout, package and generated outputs.

Raw run records go under ``data/`` and rendered artifacts under ``results/``.
Both are gitignored roots of the checkout, or of the working directory for an
installed wheel, and the ``ORBEXA_DATA`` and ``ORBEXA_RESULTS`` environment
variables override them.

Copyright: Aaron John Sabu
"""

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent


def checkout_root() -> Path | None:
    """Return this package's checkout, or None for an installed wheel."""
    candidate = PACKAGE_ROOT.parent
    if (candidate / "pyproject.toml").is_file() and (candidate / "run.py").is_file():
        return candidate
    return None


def project_root() -> Path:
    """Use the checkout for default output roots, or the caller's directory."""
    return checkout_root() or Path.cwd()


def default_config_path() -> Path:
    """Find the checkout scenario without relying on package nesting depth."""
    root = checkout_root()
    if root is not None and (root / "config" / "default.yaml").is_file():
        return root / "config" / "default.yaml"
    raise FileNotFoundError(
        "Default scenario is unavailable; supply a scenario YAML with --config PATH"
    )


@dataclass(frozen=True)
class ProjectLayout:
    """Resolved generated-output roots for one checkout or working directory."""

    root: Path
    data_root: Path
    results_root: Path

    @classmethod
    def current(cls) -> "ProjectLayout":
        """Resolve the roots from the environment, then from the project root.

        Outputs: absolute ``data_root`` and ``results_root`` paths; neither is
        created.
        """
        root = project_root().resolve()
        data = os.environ.get("ORBEXA_DATA") or root / "data"
        results = os.environ.get("ORBEXA_RESULTS") or root / "results"
        return cls(
            root,
            Path(data).expanduser().resolve(),
            Path(results).expanduser().resolve(),
        )

    def run_directory(self, *parts: str) -> Path:
        """Name a new raw-record directory ``data/<parts>/<UTC run id>``.

        Outputs: a path that is not created here; callers create it with
        ``exist_ok=False`` so two runs never share a directory.
        """
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        return self.data_root.joinpath(*parts, run_id)

    def require_output(self, path: Path, description: str) -> None:
        """Refuse an output inside the checkout unless it is under a generated root.

        Inputs: ``path`` is resolved; ``description`` names the output in the
        error.
        Outputs: raises ``ValueError`` for checkout paths outside ``data/`` and
        ``results/``, such as ``scripts/`` or ``config/``.
        """
        checkout = checkout_root()
        if checkout is None or not path.is_relative_to(checkout):
            return
        if path.is_relative_to(self.data_root) or path.is_relative_to(
            self.results_root
        ):
            return
        raise ValueError(
            f"{description} inside the checkout must be under {self.data_root} "
            f"or {self.results_root}"
        )
