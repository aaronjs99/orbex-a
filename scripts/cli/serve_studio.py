"""Serve the local Studio at127.0.0.1:8881/studio/ and real API at127.0.0.1:8882."""
from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading

from orbexa.simulation.live_service import LiveServiceSettings, LiveSimulationService, make_server


THEORY_DOCUMENTS = (
    ('Adaptive dynamic tube MPC', 'theory.md'),
    ('Carrier deployment', 'deployment.md'),
    ('Envisat geometry', 'envisat.md'),
    ('Uncontrolled target dynamics', 'uncontrolled-target.md'),
    ('Component-aware approach', 'composite-approach.md'),
    ('Continuous geometry safety', 'composite-continuous-safety.md'),
    ('Fixed-deadline arrival', 'arrival-policy.md'),
    ('Terminal bounds', 'physical-terminal.md'),
    ('Attachment handoff', 'terminal-handoff.md'),
    ('Chaser attitude', 'chaser-attitude.md'),
    ('Attachment, detumbling and deorbit', 'mission-lifecycle.md'),
    ('Orbital model fidelity', 'dynamics-fidelity.md'),
)


def _export_live_viewer(settings):
    """Export a fresh empty live catalog and available source theory docs."""
    from orbexa.visualization.mission_studio import MissionStudio
    root = settings.output_dir.parent / 'studio'
    docs = settings.config_path.parent.parent / 'docs'
    documents = {title: docs / name for title, name in THEORY_DOCUMENTS if (docs / name).is_file()}
    index = MissionStudio(root).export([], live=True, theory_documents=documents)
    (root / 'index.html').write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta http-equiv="refresh" content="0;url=studio/">'
        '<title>ORBEX-A Studio</title><a href="studio/">Open Mission Studio</a></html>', encoding='utf-8')
    return root, index, len(documents)


def _viewer_server(root, *, port=8881):
    """Reserve only loopback; an occupied port never displaces its owner."""
    if not Path(root).is_dir():
        raise ValueError('Export the local Studio before serving it')
    try:
        server = ThreadingHTTPServer(('127.0.0.1', port),
            partial(SimpleHTTPRequestHandler, directory=str(root)))
    except OSError as error:
        raise OSError(f'Cannot start local Studio on127.0.0.1:{port}: {error}; an existing server was not replaced') from error
    server.daemon_threads = True
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='Explicit Envisat scenario YAML')
    parser.add_argument('--prior', type=Path, required=True, help='Independent initial-belief JSON')
    parser.add_argument('--admission', type=Path, required=True, help='Declared sensor/model/pose envelopes JSON')
    parser.add_argument('--deployment', type=Path, required=True, help='Close-in component-route options JSON')
    parser.add_argument('--spacecraft', type=Path, required=True, help='Explicit complete component geometry JSON')
    parser.add_argument('--output-dir', type=Path, required=True,
        help='Retained job directories outside the checkout; the live viewer is exported to the sibling studio directory')
    parser.add_argument('--max-updates', type=int, default=1200, help='Positive outer-update cap <=1200; a cap is not completion')
    parser.add_argument('--resume', type=Path,
        help='One-use first Run: resume this unfinished raw mission_data.json and its adjacent guard journal with identical knobs; new updates remain within the original total cap/domain')
    parser.add_argument('--proof-deps', type=Path, help='Optional explicit directory containing the python-flint dependency')
    args = parser.parse_args(argv)
    service = server = ui_server = ui_thread = None
    try:
        if args.proof_deps is not None:
            dependency = args.proof_deps.expanduser().resolve()
            if not dependency.is_dir():
                raise ValueError('Missing explicit proof dependency directory')
            sys.path.insert(0, str(dependency))
        settings = LiveServiceSettings(output_dir=args.output_dir, config_path=args.config,
            prior_path=args.prior, admission_path=args.admission, deployment_path=args.deployment,
            spacecraft_path=args.spacecraft, maximum_updates=args.max_updates, resume_path=args.resume)
        ui_root, _index, theory_count = _export_live_viewer(settings)
        # Reserve8881 before constructing/binding the API. A failed UI bind
        # must not leave a hidden API service or replace an unowned listener.
        ui_server = _viewer_server(ui_root)
        service = LiveSimulationService(settings)
        server = make_server(service)
        ui_thread = threading.Thread(target=ui_server.serve_forever,
            name='orbexa-local-studio', daemon=True)
        ui_thread.start()
        print(json.dumps(dict(service='orbexa-live-v1', url='http://127.0.0.1:8882',
            ui_url='http://127.0.0.1:8881/studio/', ui_output_dir=str(ui_root),
            theory_documents=theory_count, output_dir=str(settings.output_dir),
            maximum_updates=settings.maximum_updates,
            resume_source=None if settings.resume_path is None else str(settings.resume_path))), flush=True)
        server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError, TypeError, ImportError) as error:
        parser.error(str(error))
    finally:
        if service is not None:
            service.close()
        if server is not None:
            server.server_close()
        if ui_server is not None:
            if ui_thread is not None:
                ui_server.shutdown()
                ui_thread.join(3)
            ui_server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
