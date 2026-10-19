"""Versioned physical-time common random field for synthetic sensor studies.

The field is constant within declared time bins. It is not iid white noise and
does not schedule observations or change their actual acquisition timestamps.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math

import numpy as np


class PhysicalSlotNoise:
    """Branch-independent bounded innovations indexed by physical time/channel."""

    def __init__(self, seed, period_s, epoch_s, quantum_s):
        self.seed = seed
        self.quantum = Fraction(str(quantum_s))
        period = Fraction(str(period_s))
        epoch = Fraction(str(epoch_s))
        if self.quantum <= 0 or period <= 0:
            raise ValueError('Keyed sensor noise requires a positive period and time quantum')
        period_ticks, epoch_ticks = period/self.quantum, epoch/self.quantum
        if period_ticks.denominator != 1 or epoch_ticks.denominator != 1:
            raise ValueError('Keyed sensor period and epoch must be exact multiples of the time quantum')
        self.period_ticks, self.epoch_ticks = int(period_ticks), int(epoch_ticks)

    def slot_at(self, time_s):
        """Quantize nearest tick (ties upward), then use a half-open time bin."""
        time = Fraction(float(time_s))/self.quantum
        ticks = (time+Fraction(1, 2)).numerator//(time+Fraction(1, 2)).denominator
        return (ticks-self.epoch_ticks)//self.period_ticks

    def uniform(self, slot, agent, channel, low=0., high=1., size=None):
        """Stable SHA-256 counter draws; ``slot=None`` denotes a fixed bias."""
        event = ('persistent' if slot is None else
                 str((self.epoch_ticks+slot*self.period_ticks)*self.quantum))
        lower, upper = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
        shape = np.broadcast_shapes(lower.shape, upper.shape) if size is None else (size,)
        draws = []
        for component in range(math.prod(shape)):
            key = ['orbexa.sensor-uniform.sha256-53.v1', self.seed, event,
                   str(agent), channel, component]
            digest = hashlib.sha256(json.dumps(key, separators=(',', ':'), ensure_ascii=True).encode('ascii')).digest()
            draws.append((int.from_bytes(digest[:8], 'big') >> 11)/(1 << 53))
        value = lower+(upper-lower)*np.asarray(draws).reshape(shape)
        return float(value) if not shape else value
