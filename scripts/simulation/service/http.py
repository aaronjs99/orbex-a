"""Bounded loopback HTTP adapter for the injected live session owner."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit
from orbexa.simulation.service.contracts import ALLOWED_ORIGINS, LIVE_PROTOCOL_VERSION, LIVE_SERVICE, LiveParameters, PARAMETER_LIMITS
from orbexa.simulation.service.session import LiveSimulationService


class _LiveHandler(BaseHTTPRequestHandler):
    server_version = "ORBEXA-Loopback/1"
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass

    def _origin(self):
        origin = self.headers.get("Origin")
        host = self.headers.get("Host", "")
        if host not in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"):
            raise PermissionError("Only the literal loopback host is accepted")
        if origin is not None and origin not in self.server.allowed_origins:
            raise PermissionError("This browser origin is not allowed")
        return origin

    def _reply(self, status, payload=None, *, preflight=False):
        raw = b"" if status == 204 else json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        origin = self.headers.get("Origin")
        if origin in self.server.allowed_origins:
            self.send_header("Access-Control-Allow-Origin", origin)
            if preflight:
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Max-Age", "600")
                if self.headers.get("Access-Control-Request-Private-Network") == "true":
                    self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Vary", "Origin, Access-Control-Request-Headers, Access-Control-Request-Private-Network")
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if raw:
            self.wfile.write(raw)

    def _body(self):
        if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
            raise ValueError("POST requests require application/json")
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("A valid Content-Length is required") from error
        if not 0 < size <= 16384 or self.headers.get("Transfer-Encoding"):
            raise ValueError("JSON request size must be between1 and16384 bytes")
        self.connection.settimeout(5)
        def pairs(rows):
            result = {}
            for key, value in rows:
                if key in result:
                    raise ValueError("Duplicate JSON field")
                result[key] = value
            return result
        def invalid(_value):
            raise ValueError("JSON numbers must be finite")
        try:
            value = json.loads(self.rfile.read(size), object_pairs_hook=pairs, parse_constant=invalid)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("Malformed JSON request") from error
        if not isinstance(value, dict):
            raise ValueError("JSON request must be an object")
        return value

    def do_OPTIONS(self):
        try:
            self._origin()
            if self.headers.get("Origin") is None:
                raise PermissionError("Browser preflight requires an allowed origin")
            if urlsplit(self.path).path not in ("/run", "/pause", "/reset", "/state", "/health"):
                return self._reply(404, dict(error="Unknown endpoint"))
            requested = {x.strip().lower() for x in self.headers.get("Access-Control-Request-Headers", "").split(",") if x.strip()}
            if requested - {"content-type"} or self.headers.get("Access-Control-Request-Method", "GET") not in ("GET", "POST"):
                raise PermissionError("Unsupported preflight method or headers")
            self._reply(204, preflight=True)
        except PermissionError as error:
            self._reply(403, dict(error=str(error)))

    def do_GET(self):
        try:
            self._origin()
            url = urlsplit(self.path)
            if url.path == "/health":
                state = self.server.service.state()
                return self._reply(200, dict(service=LIVE_SERVICE, status="ready",
                    protocol_version=LIVE_PROTOCOL_VERSION, execution_model=state['execution_model'],
                    bind="127.0.0.1", port=self.server.server_port, worker_alive=self.server.service.worker_alive,
                    session_id=state['session_id'],
                    parameter_limits=PARAMETER_LIMITS, maximum_updates=self.server.service.settings.maximum_updates,
                    source="Maintained sensed nonlinear GEKKO/IPOPT with finite fleet admission and contact; explicit historical analytical resume available; outcomes may refuse"))
            if url.path != "/state":
                return self._reply(404, dict(error="Unknown endpoint"))
            query = parse_qs(url.query, keep_blank_values=True)
            if set(query) - {"after", "timeout"} or any(len(v) != 1 for v in query.values()):
                raise ValueError("Supply only one after and timeout query value")
            after = int(query['after'][0]) if 'after' in query else None
            timeout = float(query.get('timeout', ['20' if after is not None else '0'])[0])
            self._reply(200, self.server.service.state(after=after, timeout=timeout))
        except PermissionError as error:
            self._reply(403, dict(error=str(error)))
        except (ValueError, TypeError) as error:
            self._reply(400, dict(error=str(error)))
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        try:
            self._origin()
            envelope = self._body()
            if (set(envelope) != {'protocol_version','session_id','payload'}
                    or type(envelope.get('protocol_version')) is not int
                    or envelope['protocol_version'] != LIVE_PROTOCOL_VERSION
                    or envelope.get('session_id') != self.server.service.state()['session_id']):
                raise RuntimeError('Simulator session or protocol changed; reconnect before a command')
            value = envelope['payload']
            if not isinstance(value,dict):
                raise ValueError('Command payload must be a JSON object')
            if self.path == "/run":
                return self._reply(202, self.server.service.start(LiveParameters.from_dict(value)))
            if self.path == "/pause":
                if set(value) - {"paused"}:
                    raise ValueError("Supply only paused")
                return self._reply(200, self.server.service.pause(value.get('paused')))
            if self.path == "/reset":
                if value:
                    raise ValueError("Reset does not accept parameters")
                return self._reply(200, self.server.service.reset())
            self._reply(404, dict(error="Unknown endpoint"))
        except PermissionError as error:
            self._reply(403, dict(error=str(error)))
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (ValueError, TypeError, OSError) as error:
            self._reply(400, dict(error=str(error)))
        except RuntimeError as error:
            self._reply(409, dict(error=str(error)))


def make_server(service, *, port=8882, viewer_port=None):
    """Bind only IPv4 loopback; port0 is useful for isolated caller probes."""
    if not isinstance(service, LiveSimulationService):
        raise TypeError("Inject LiveSimulationService")
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("Invalid loopback port")
    if viewer_port is not None and (isinstance(viewer_port,bool) or not isinstance(viewer_port,int)
            or not 1 <= viewer_port <= 65535):
        raise ValueError('Invalid loopback viewer port')
    server = ThreadingHTTPServer(("127.0.0.1", port), _LiveHandler)
    server.daemon_threads = True
    server.service = service
    server.allowed_origins = ALLOWED_ORIGINS | (frozenset() if viewer_port is None else frozenset((
        f'http://127.0.0.1:{viewer_port}',f'http://localhost:{viewer_port}')))
    return server
