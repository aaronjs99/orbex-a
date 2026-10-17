#!/usr/bin/env python3
"""Run one declared sensed nonlinear mission from a movable source checkout."""

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

from orbexa.cli.run_nonlinear import main


if __name__ == "__main__":
    defaults=[]
    for option,path in (('--config',ROOT/'config/envisat.yaml'),
                        ('--spacecraft',ROOT/'config/spacecraft/envisat.json'),
                        ('--study',ROOT/'config/nonlinear-study.json')):
        defaults.extend((option,str(path)))
    raise SystemExit(main([*defaults,*sys.argv[1:]]))
