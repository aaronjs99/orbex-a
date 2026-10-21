"""Whole-set descending entry-interface brackets from contiguous held flows."""
from __future__ import annotations

from fractions import Fraction
import copy

from orbexa.core.dynamics.reference_frame import fingerprint


def radial_status(box, radius_m):
    radius_squared = Fraction(float(radius_m))**2
    lower = sum(0 if lo <= 0 <= hi else min(lo*lo, hi*hi) for lo, hi in zip(box.lower[:3], box.upper[:3]))
    upper = sum(max(lo*lo, hi*hi) for lo, hi in zip(box.lower[:3], box.upper[:3]))
    radial_velocity_upper = sum(max(lo*vlow, lo*vhigh, hi*vlow, hi*vhigh)
        for lo, hi, vlow, vhigh in zip(box.lower[:3], box.upper[:3], box.lower[3:6], box.upper[3:6]))
    return dict(entirely_above=lower > radius_squared, entirely_at_or_below=upper <= radius_squared,
        position_norm_squared_lower=str(lower), position_norm_squared_upper=str(upper),
        radial_product_upper_m2_s=str(radial_velocity_upper), descending=radial_velocity_upper < 0)


class EntryInterfaceMonitor:
    """A positive above-to-below bracket, with every intervening flow descending."""
    def __init__(self, radius_m):
        self.radius_m = float(radius_m)
        self.anchor = None
        self.intervals = []
        self.last_end = None
        self.last_box = None
        self.parameter_fingerprint = None
        self.certificate = None

    def reset(self):
        """A fuel-split COM coordinate jump cannot support a continuous bracket."""
        self.anchor, self.intervals, self.last_end, self.last_box = None, [], None, None
        self.parameter_fingerprint, self.certificate = None, None

    def interval(self, start, end, *, fuel_flow_kg_s):
        if len(fuel_flow_kg_s) != len(start.fuel_kg) or any(value != 0 for value in fuel_flow_kg_s):
            raise ValueError('A continuous entry-interface bracket requires explicitly zero fuel depletion')
        if start.enclosure is None or end.enclosure is None or end.enclosure.last_swept is None:
            raise ValueError('A whole-set entry bracket requires a complete carried held-flow interval')
        q0, q1 = start.enclosure.exact_time_s, end.enclosure.exact_time_s
        if q1 <= q0 or self.last_end is not None and q0 != self.last_end:
            raise ValueError('Entry-interface held-flow clocks are not contiguous and increasing')
        if start.enclosure.parameter_fingerprint != end.enclosure.parameter_fingerprint:
            raise ValueError('Entry-interface model/geometry changed across a held flow')
        if self.parameter_fingerprint is not None and start.enclosure.parameter_fingerprint != self.parameter_fingerprint:
            raise ValueError('Entry-interface model/geometry changed inside a bracket')
        if self.last_box is not None and not all(lo >= old_lo and hi <= old_hi for lo, hi, old_lo, old_hi
                in zip(start.enclosure.box.lower, start.enclosure.box.upper, self.last_box.lower, self.last_box.upper)):
            raise ValueError('Entry-interface continuation is not contained in the preceding admitted endpoint')
        begin = radial_status(start.enclosure.box, self.radius_m)
        finish = radial_status(end.enclosure.box, self.radius_m)
        swept = radial_status(end.enclosure.last_swept, self.radius_m)
        if begin['entirely_above']:
            self.anchor = dict(exact_time_s=str(q0), state=start.enclosure.box.as_dict(), radial_status=begin,
                carried_chain_sha256=start.enclosure.proof_chain_sha256)
            self.intervals = []
        if not swept['descending']:
            self.anchor, self.intervals = None, []
        elif self.anchor is not None:
            self.intervals.append(dict(exact_start_time_s=str(q0), exact_end_time_s=str(q1),
                radial_product_upper_m2_s=swept['radial_product_upper_m2_s'],
                end_state_fingerprint=fingerprint(end.enclosure.box.as_dict()),
                carried_chain_sha256=end.enclosure.proof_chain_sha256))
        self.last_end = q1
        self.last_box = end.enclosure.box
        self.parameter_fingerprint = end.enclosure.parameter_fingerprint
        if self.anchor is not None and finish['entirely_at_or_below'] and self.intervals:
            result = dict(method='continuous-whole-set-descending-interface-bracket-v1', status='PROVEN',
                entry_radius_m=self.radius_m, above=self.anchor,
                below=dict(exact_time_s=str(q1), state=end.enclosure.box.as_dict(), radial_status=finish),
                intervals=copy.deepcopy(self.intervals), checked_holds=len(self.intervals),
                parameter_fingerprint=end.enclosure.parameter_fingerprint,
                fuel_depletion='zero throughout every bracketed held flow; no split COM coordinate jump',
                scope='Every realization consistent with the declared carried dynamics and bounded observations crosses the interface downward inside this finite bracket; below-interface propagation is only a numerical event bracket, not atmospheric flight')
            result['identity'] = fingerprint(result)
            self.certificate = result
        return self.certificate
