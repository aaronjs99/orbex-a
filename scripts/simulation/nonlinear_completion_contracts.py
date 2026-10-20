"""Shared exact-value contracts and implementation identity for completion.

These helpers do not import the lifecycle, journal, or physical adapters.
"""
from __future__ import annotations

import copy
from fractions import Fraction
import hashlib
import json
from math import isqrt
from pathlib import Path
import re

from orbexa.control.reachability import IntervalBox
from orbexa.control.refuge import fingerprint as exact_fingerprint
from orbexa.simulation.mission_checkpoint import digest


_COMPLETION_SOURCE_FILES = (
    'nonlinear_completion.py',
    'nonlinear_completion_boundary.py',
    'nonlinear_completion_bridge.py',
    'contact_boundary.py',
    'contact_rotation.py',
    'contact_strain.py',
    'contact_strain_observation.py',
    'contact_wrench.py',
    'contact_observation.py',
    'finite_lock.py',
    'finite_lock_chart.py',
    'assembly_enclosure.py',
    'assembly_observation.py',
    'deorbit_entry.py',
    'nonlinear_completion_contracts.py',
    'nonlinear_completion_environment.py',
    'nonlinear_environment_enclosure.py',
    'nonlinear_completion_journal.py',
    'nonlinear_completion_observers.py',
    'nonlinear_completion_propellant.py',
)


def nonlinear_completion_source_identity():
    """Bind every completion component, including the compatibility facade."""
    directory = Path(__file__).parent
    files = {name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
             for name in _COMPLETION_SOURCE_FILES}
    manifest = dict(schema='orbexa.nonlinear_completion_source_bundle.v1', files=files)
    encoded = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return dict(schema=manifest['schema'], sha256=hashlib.sha256(encoded).hexdigest(), files=files)


def _require(condition, message):
    if not condition:
        raise ValueError('Nonlinear completion: '+message)



def _identity(value, label):
    _require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None,
             label+' must be a retained SHA-256 identity')
    return value



def _pairs(value, ids, label):
    items = tuple(value.items()) if isinstance(value, dict) else tuple(value)
    _require(len(items) == len(ids) and {key for key, _ in items} == set(ids), label+' must name the exact fleet')
    return dict(items)



def _box(value, dimension):
    _require(isinstance(value, dict), 'an exact saved interval box is required')
    result = IntervalBox(value.get('exact_lower', value.get('lower')),
                         value.get('exact_upper', value.get('upper')))
    _require(result.dimension == dimension, 'saved interval has the wrong dimension')
    if 'exact_lower' in value:
        _require(digest(result.as_dict()) == digest(value), 'saved exact and display interval bounds disagree')
    return result



def _sqrt_upper(value, bits=128):
    scale = 1 << bits
    numerator = isqrt(value.numerator*scale*scale//value.denominator)
    if numerator*numerator*value.denominator < value.numerator*scale*scale:
        numerator += 1
    return Fraction(numerator, scale)



def _dyadic_upper(value, bits=128):
    scale = 1 << bits
    return Fraction(-(-value.numerator*scale//value.denominator), scale)



def _compact_hold(record):
    result = {key: copy.deepcopy(value) for key, value in record.items()
              if key not in ('decision', 'runtime_commit')}
    decision, commit = record['decision'], record['runtime_commit']
    result['decision'] = {key: copy.deepcopy(value) for key, value in decision.items() if key != 'geometry'}
    result['geometry_id'] = None if decision.get('geometry') is None else exact_fingerprint(decision['geometry'])
    result['runtime_commit_status'], result['remaining_fuel'] = commit['status'], copy.deepcopy(commit['remaining_fuel'])
    return result
