"""Read authoritative per-attempt IPOPT iterations without zero defaults."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re


NATIVE_IPOPT_LOG = 'native-ipopt.txt'


def native_ipopt_iteration_metadata(path, *, unavailable_reason=None):
    """Require one final integer summary followed by one native exit record.

    The solve owner must isolate/reset the file before invoking IPOPT. This
    reader also accepts immutable retained logs for outside-run inspection.
    Counts describe optimizer work, including failed solves, not feasibility.
    """
    path = Path(path)
    source = dict(method='native-ipopt-final-summary-v1', log_file=path.name,
                  status='UNAVAILABLE', sha256=None, native_exit=None,
                  semantics='Native iterations in this solve attempt; not accumulated retries or an acceptance decision')
    result = dict(iterations=None, iteration_count_provenance=source)
    if unavailable_reason is not None:
        source['reason'] = unavailable_reason
        return result
    try:
        payload = path.read_bytes()
    except OSError as error:
        source['reason'] = type(error).__name__+': '+str(error)
        return result
    source['sha256'] = hashlib.sha256(payload).hexdigest()
    lines = payload.decode('utf-8', errors='replace').splitlines()
    summaries = [(index, line) for index, line in enumerate(lines)
                 if line.lstrip().startswith('Number of Iterations')]
    exits = [(index, line.strip()) for index, line in enumerate(lines)
             if line.lstrip().startswith('EXIT:')]
    if len(exits) == 1:
        source['native_exit'] = exits[0][1]
    if len(summaries) > 1 or len(exits) > 1:
        source.update(status='AMBIGUOUS', reason='Multiple native summaries or exit records')
    elif len(summaries) != 1 or len(exits) != 1 or not exits[0][1][5:].strip():
        source.update(status='INCOMPLETE', reason='A final iteration summary and native exit are both required')
    elif exits[0][0] <= summaries[0][0]:
        source.update(status='INCOMPLETE', reason='The native exit precedes its iteration summary')
    else:
        match = re.fullmatch(r'\s*Number of Iterations[.\s]*:\s*([0-9]+)\s*', summaries[0][1])
        if match is None:
            source.update(status='MALFORMED', reason='The final summary is not a nonnegative integer count')
        else:
            try:
                result['iterations'] = int(match.group(1))
            except ValueError:
                source.update(status='MALFORMED', reason='The final count exceeds the supported integer format')
            else:
                source['status'] = 'REPORTED'
    return result
