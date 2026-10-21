"""Exact acceleration-history moment indexing for navigation queries."""
from __future__ import annotations

from bisect import bisect_left
from collections import deque
from fractions import Fraction
import math

from orbexa.control.certification.reachability import IntervalBox, FlowCertificate, SweptSegment


class _AccelerationMomentLedger:
    """Bounded exact acceleration moments guarded by current source scalars.

    Lower and upper acceleration supports have separate scalar moments M0 and
    M1. Their query integral is delta(M1)-q0*delta(M0), exactly the old
    nonnegative segment weights. No interval subtraction or proof identity is
    cached. Unsupported types or incomplete topology defer to the old scan.
    """
    def __init__(self):
        self.owner, self.keys, self.sources, self.valid = None, (), (), False
        self._clear()

    def _clear(self):
        self.starts, self.ends, self.coefficients = [], [], []
        self.prefix = [((Fraction(0),)*3,)*4]
        self.counts, self.head, self.dead_holds = [], 0, 0

    @staticmethod
    def _key(hold, agent):
        if type(hold) is not dict or type(hold.get('flows')) is not dict:
            return None
        start, end = hold.get('start'), hold.get('end')
        if type(start) is not Fraction or type(end) is not Fraction:
            return None
        flow = hold['flows'].get(agent)
        if type(flow) is not FlowCertificate or type(flow.segments) is not tuple:
            return None
        segments = []
        for segment in flow.segments:
            if (type(segment) is not SweptSegment or type(segment.offset) is not tuple or len(segment.offset) != 2
                    or any(type(value) is not Fraction for value in segment.offset)
                    or type(segment.derivative) is not IntervalBox
                    or len(segment.derivative.lower) != 6 or len(segment.derivative.upper) != 6):
                return None
            lo, hi = segment.offset
            lower, upper = segment.derivative.lower[3:], segment.derivative.upper[3:]
            if any(type(value) is not Fraction for value in lower+upper):
                return None
            # Copy primitive integers: tuple aliases or object identities alone
            # miss forced same-object IntervalBox/Fraction mutations.
            scalars = tuple(part for value in lower+upper for part in (value.numerator, value.denominator))
            segments.append((id(segment), id(segment.derivative), lo.numerator, lo.denominator,
                             hi.numerator, hi.denominator, scalars))
        return (id(hold), start.numerator, start.denominator, end.numerator, end.denominator,
                id(flow), id(flow.segments), tuple(segments))

    def _append(self, key):
        scalars = key[1:5]+tuple(value for segment in key[7] for value in segment[2:6]+segment[6])
        # Fraction's usual normalization can itself be broken by forced
        # mutation. Such data keeps the original scan's arithmetic semantics.
        if (any(type(value) is not int for value in scalars)
                or any(d <= 0 or math.gcd(n, d) != 1 for n, d in zip(scalars[::2], scalars[1::2]))):
            return False
        start, end = Fraction(key[1], key[2]), Fraction(key[3], key[4])
        if end <= start or self.ends and start != self.ends[-1]:
            return False
        cursor = Fraction(0)
        for segment in key[7]:
            lo, hi = Fraction(segment[2], segment[3]), Fraction(segment[4], segment[5])
            if lo != cursor or hi <= lo:
                return False
            a, b = start+lo, start+hi
            values = segment[6]
            lower = tuple(Fraction(values[i], values[i+1]) for i in (0, 2, 4))
            upper = tuple(Fraction(values[i], values[i+1]) for i in (6, 8, 10))
            if any(lo > hi for lo, hi in zip(lower, upper)):
                return False
            dt, moment, previous = b-a, (b*b-a*a)/2, self.prefix[-1]
            self.prefix.append(tuple(tuple(p+weight*v for p, v in zip(row, bound))
                for row, bound, weight in ((previous[0], lower, dt), (previous[1], lower, moment),
                                          (previous[2], upper, dt), (previous[3], upper, moment))))
            self.starts.append(a)
            self.ends.append(b)
            self.coefficients.append((lower, upper))
            cursor = hi
        if start+cursor != end:
            return False
        self.counts.append(len(key[7]))
        return True

    def _rebuild(self, keys):
        self._clear()
        return all(self._append(key) for key in keys)

    def _prepare(self, history, agent):
        if type(history) is not deque or history.maxlen is None or not history:
            return False
        sources = tuple(history)
        try:
            keys = tuple(self._key(hold, agent) for hold in sources)
        except (KeyError, IndexError, AttributeError, TypeError, ValueError):
            return False
        if any(key is None for key in keys):
            return False
        if self.owner is history and keys == self.keys:
            return self.valid
        incremental = False
        if self.owner is history and self.valid:
            index = next((i for i, key in enumerate(self.keys) if key[0] == keys[0][0]), None)
            if index is not None:
                overlap = len(self.keys)-index
                if len(keys) >= overlap and self.keys[index:] == keys[:overlap]:
                    self.head += sum(self.counts[:index])
                    self.dead_holds += index
                    self.counts = self.counts[index:]
                    incremental = all(self._append(key) for key in keys[overlap:])
        self.valid = incremental or self._rebuild(keys)
        self.owner, self.keys, self.sources = history, keys, sources
        live = len(self.starts)-self.head
        if self.valid and (self.head >= max(1, live) or self.dead_holds >= history.maxlen):
            self.valid = self._rebuild(keys)
        return self.valid

    def _at(self, q):
        index = bisect_left(self.ends, q, self.head)
        if index >= len(self.ends) or q < self.starts[index]:
            return None
        if q == self.starts[index]:
            return self.prefix[index]
        if q == self.ends[index]:
            return self.prefix[index+1]
        dt, moment = q-self.starts[index], (q*q-self.starts[index]*self.starts[index])/2
        previous, (lower, upper) = self.prefix[index], self.coefficients[index]
        return tuple(tuple(p+weight*v for p, v in zip(row, bound))
            for row, bound, weight in ((previous[0], lower, dt), (previous[1], lower, moment),
                                      (previous[2], upper, dt), (previous[3], upper, moment)))

    def integral(self, history, agent, q0, q1):
        result = self.integrals(history, agent, ((q0, q1),))
        return result if result is NotImplemented else result[0]

    def integrals(self, history, agent, queries):
        """Check the actual source once for a finite synchronous query batch."""
        queries = tuple(queries)
        if (any(type(q) not in (Fraction, int) or type(q) is Fraction and
                    (type(q.numerator) is not int or type(q.denominator) is not int
                     or q.denominator <= 0 or math.gcd(q.numerator, q.denominator) != 1)
                for query in queries for q in query)
                or not self._prepare(history, agent)):
            return NotImplemented
        return tuple(self._integral_prepared(q0, q1) for q0, q1 in queries)

    def _integral_prepared(self, q0, q1):
        if not self.starts[self.head] <= q0 < q1 <= self.ends[-1]:
            return None
        first, last = self._at(q0), self._at(q1)
        if first is None or last is None:
            return None
        lower = tuple(b-a-q0*(d-c) for a, b, c, d in zip(first[1], last[1], first[0], last[0]))
        upper = tuple(b-a-q0*(d-c) for a, b, c, d in zip(first[3], last[3], first[2], last[2]))
        return lower, upper
