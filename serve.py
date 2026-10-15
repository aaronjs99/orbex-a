#!/usr/bin/env python3
"""Start the current Envisat Studio service from this movable checkout."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
package_root = ROOT / 'scripts'
spec = spec_from_file_location('orbexa', package_root / '__init__.py',
                               submodule_search_locations=[str(package_root)])
if spec is None or spec.loader is None:
    raise ImportError('Could not load the ORBEX-A scripts package')
package = module_from_spec(spec)
sys.modules['orbexa'] = package
spec.loader.exec_module(package)

if __name__ == '__main__':
    from orbexa.cli.serve_studio import main
    defaults = []
    for option, path in (
        ('--config', ROOT / 'config/envisat.yaml'),
        ('--prior', ROOT / 'config/envisat-prior.json'),
        ('--admission', ROOT / 'config/envisat-admission.json'),
        ('--deployment', ROOT / 'config/envisat-deployment.json'),
        ('--spacecraft', ROOT / 'config/spacecraft/envisat.json'),
        ('--output-dir', ROOT.parent / 'orbexa-output/live-jobs'),
    ):
        defaults.extend((option, str(path)))
    raise SystemExit(main([*defaults, *sys.argv[1:]]))
