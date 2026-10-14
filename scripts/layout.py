"""Dependency-free paths for the physical scripts layout and installed package."""

from pathlib import Path
from typing import Optional


PACKAGE_ROOT = Path(__file__).resolve().parent


def checkout_root() -> Optional[Path]:
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
    raise FileNotFoundError("Default scenario is unavailable; supply a scenario YAML with --config PATH")
