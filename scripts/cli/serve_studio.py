"""Serve the local Studio at127.0.0.1:8881/studio/ and real API at127.0.0.1:8882."""
from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import shutil
import sys
import threading

from orbexa.simulation.live_service import LiveServiceSettings, LiveSimulationService, make_server, LIVE_SERVICE


THEORY_DOCUMENTS = (
    ('Nonlinear mission formulation', 'nonlinear-mission.md'),
    ('Sensed adaptive tube MPC', 'sensed-tube.md'),
    ('Navigation and physical bounds', 'nonlinear-navigation.md'),
    ('Safe fleet fallback', 'fleet-fallback.md'),
    ('Finite contact and attachment', 'finite-contact.md'),
    ('Envisat geometry', 'envisat.md'),
    ('Matched comparison results', 'nonlinear-study-results.md'),
)


def _export_live_viewer(settings, root):
    """Export a fresh empty live catalog and available source theory docs."""
    from orbexa.visualization.mission_studio import MissionStudio
    root = Path(root)
    docs = settings.config_path.parent.parent / 'docs'
    documents = {title: docs / name for title, name in THEORY_DOCUMENTS if (docs / name).is_file()}
    index = MissionStudio(root).export([], live=True, theory_documents=documents)
    # Source Markdown links to published generated figures. Prefer local copies
    # when available, so a local export does not depend on the hosted release.
    public = 'https://orbex-a.aaronjs.me/theory/figures/'
    previous_public = 'https://orbex-a-mission-studio.aaronjs99.chatgpt.site/theory/figures/'
    folder = (docs/'figures').resolve()
    for document in (root/'studio').glob('theory-*.html'):
        text = document.read_text(encoding='utf-8').replace(previous_public, public)
        for name in set(re.findall(re.escape(public)+r'([A-Za-z0-9_.-]+\.(?:svg|png|json|jpg|webp))',text)):
            source = (folder/name).resolve()
            if source.is_file() and source.is_relative_to(folder):
                destination = root/'theory'/'figures'/name
                destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(source,destination)
                text = text.replace(public+name,'../theory/figures/'+name)
        document.write_text(text,encoding='utf-8')
    (root / 'index.html').write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta http-equiv="refresh" content="0;url=studio/">'
        '<title>ORBEX-A Studio</title><a href="studio/">Open Mission Studio</a></html>', encoding='utf-8')
    return root, index, len(documents)


def _viewer_server(root, *, port=8881):
    """Reserve only loopback; an occupied port never displaces its owner."""
    # Bind first. The handler is not started until its owned export is ready.
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
    parser.add_argument('--prior', type=Path, required=True, help='Retained analytical prior for historical resume; fresh runs use the shared nonlinear study prior')
    parser.add_argument('--admission', type=Path, required=True, help='Retained analytical admission settings for historical resume; fresh runs use the shared nonlinear sensing/safety profile')
    parser.add_argument('--deployment', type=Path, required=True, help='Retained analytical route settings for historical resume; fresh runs use the shared nonlinear deployment profile')
    parser.add_argument('--spacecraft', type=Path, required=True, help='Explicit complete component geometry JSON')
    parser.add_argument('--output-dir', type=Path, required=True,
        help='Retained jobs outside the checkout; each UI port gets a distinct sibling viewer directory')
    parser.add_argument('--max-updates', type=int, default=1200, help='Positive outer-update cap <=1200; a cap is not completion')
    parser.add_argument('--ui-port', type=int, default=8881, help='Loopback Studio port; use8893 to coexist with an older running service')
    parser.add_argument('--api-port', type=int, default=8882, help='Loopback simulator API port; the hosted viewer also checks8894')
    parser.add_argument('--resume', type=Path,
        help='One-use historical analytical resume source; requires --historical-analytical-resume and its adjacent guard journal')
    parser.add_argument('--historical-analytical-resume', action='store_true',
        help='Explicitly run the saved analytical model on first Run; fresh jobs use the sensed nonlinear profile')
    parser.add_argument('--proof-deps', type=Path, help='Optional explicit directory containing the python-flint dependency')
    args = parser.parse_args(argv)
    service = server = ui_server = ui_thread = None
    try:
        if (not 0 <= args.ui_port <= 65535 or not 0 <= args.api_port <= 65535
                or args.ui_port == args.api_port and args.ui_port != 0):
            raise ValueError('Declare distinct UI/API loopback ports in[0,65535]; zero requests an available port')
        if args.proof_deps is not None:
            dependency = args.proof_deps.expanduser().resolve()
            if not dependency.is_dir():
                raise ValueError('Missing explicit proof dependency directory')
            sys.path.insert(0, str(dependency))
        settings = LiveServiceSettings(output_dir=args.output_dir, config_path=args.config,
            prior_path=args.prior, admission_path=args.admission, deployment_path=args.deployment,
            spacecraft_path=args.spacecraft, maximum_updates=args.max_updates, resume_path=args.resume,
            historical_analytical_resume=args.historical_analytical_resume)
        # Reserve the UI before constructing/binding the API. A failed UI bind
        # must not leave a hidden API service or replace an unowned listener.
        ui_server = _viewer_server(settings.output_dir.parent/'unpublished-studio',port=args.ui_port)
        ui_root = settings.output_dir.parent/f'{settings.output_dir.name}-studio-{ui_server.server_port}'
        ui_server.RequestHandlerClass = partial(SimpleHTTPRequestHandler,directory=str(ui_root))
        service = LiveSimulationService(settings)
        server = make_server(service,port=args.api_port,viewer_port=ui_server.server_port)
        ui_root, _index, theory_count = _export_live_viewer(settings,ui_root)
        index = Path(_index)
        content = index.read_text(encoding='utf-8')
        if '<body>' not in content:
            raise ValueError('The current Studio template lacks its explicit body entry')
        index.write_text(content.replace('<body>',f'<body data-live-api-base="http://127.0.0.1:{server.server_port}">',1),encoding='utf-8')
        ui_thread = threading.Thread(target=ui_server.serve_forever,
            name='orbexa-local-studio', daemon=True)
        ui_thread.start()
        print(json.dumps(dict(service=LIVE_SERVICE, url=f'http://127.0.0.1:{server.server_port}',
            ui_url=f'http://127.0.0.1:{ui_server.server_port}/studio/', ui_output_dir=str(ui_root),
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
