"""Serve a bounded real ORBEX-A worker API on127.0.0.1:8882."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from orbexa.simulation.live_service import LiveServiceSettings, LiveSimulationService, make_server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='Explicit Envisat scenario YAML')
    parser.add_argument('--prior', type=Path, required=True, help='Independent initial-belief JSON')
    parser.add_argument('--admission', type=Path, required=True, help='Declared sensor/model/pose envelopes JSON')
    parser.add_argument('--deployment', type=Path, required=True, help='Close-in component-route options JSON')
    parser.add_argument('--spacecraft', type=Path, required=True, help='Explicit complete component geometry JSON')
    parser.add_argument('--output-dir', type=Path, required=True, help='Retained unique job directories outside the checkout')
    parser.add_argument('--max-updates', type=int, default=1200, help='Positive outer-update cap≤1200; a cap is not completion')
    parser.add_argument('--proof-deps', type=Path, help='Optional explicit directory containing the python-flint dependency')
    args = parser.parse_args(argv)
    service = server = None
    try:
        if args.proof_deps is not None:
            dependency = args.proof_deps.expanduser().resolve()
            if not dependency.is_dir():
                raise ValueError('Missing explicit proof dependency directory')
            sys.path.insert(0, str(dependency))
        settings = LiveServiceSettings(output_dir=args.output_dir, config_path=args.config,
            prior_path=args.prior, admission_path=args.admission, deployment_path=args.deployment,
            spacecraft_path=args.spacecraft, maximum_updates=args.max_updates)
        service = LiveSimulationService(settings)
        server = make_server(service)
        print(json.dumps(dict(service='orbexa-live-v1', url='http://127.0.0.1:8882',
            output_dir=str(settings.output_dir), maximum_updates=settings.maximum_updates)), flush=True)
        server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    finally:
        if service is not None:
            service.close()
        if server is not None:
            server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
