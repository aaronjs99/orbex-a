#!/usr/bin/env python3
"""Run ORBEX-A from a source checkout, including after moving the folder."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from orbexa.cli.generate_demo import main


if __name__ == "__main__":
    raise SystemExit(main(default_root=ROOT))
