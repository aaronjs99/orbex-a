#!/usr/bin/env python3
"""Run ORBEX-A from a source checkout, including after moving the folder."""

from pathlib import Path
from importlib.util import module_from_spec, spec_from_file_location
import sys

ROOT = Path(__file__).resolve().parent
package_root = ROOT / "scripts"
package_spec = spec_from_file_location(
    "orbexa", package_root / "__init__.py", submodule_search_locations=[str(package_root)]
)
if package_spec is None or package_spec.loader is None:
    raise ImportError("Could not load the ORBEX-A scripts package")
package = module_from_spec(package_spec)
sys.modules["orbexa"] = package
package_spec.loader.exec_module(package)

from orbexa.cli.generate_demo import main


if __name__ == "__main__":
    raise SystemExit(main(default_root=ROOT))
