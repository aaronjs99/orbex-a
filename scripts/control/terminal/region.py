"""Declared reference-centered region shared by arrival planning and entry screens.

This object is not a terminal-controller certificate. Its error bounds and
finite reference domain need their own source; an independently qualified
physical controller and its actual endpoint admission remain necessary.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import json
import math
from numbers import Integral, Real
from typing import Any, Sequence


def exact_scalar(value: Any) -> Fraction:
    """Finite, binary-exact numeric conversion; rational JSON strings are explicit."""
    if isinstance(value, bool):
        raise ValueError("Boolean values are not physical bounds")
    if isinstance(value, Fraction):
        result = value
    elif isinstance(value, Integral):
        result = Fraction(int(value))
    elif isinstance(value, str):
        if len(value) > 8192:
            raise ValueError("A physical scalar declaration is too long")
        try:
            if '/' not in value:
                decimal = Decimal(value)
                number = float(decimal)
                if (not decimal.is_finite() or not math.isfinite(number)
                        or number == 0. and not decimal.is_zero()):
                    raise ValueError("A physical scalar must have a finite non-underflowing binary64 representation")
            result = Fraction(value)
        except (ValueError, ZeroDivisionError, InvalidOperation) as error:
            raise ValueError("Use a finite numeric bound or explicit rational string") from error
    elif isinstance(value, Real):
        if not math.isfinite(float(value)):
            raise ValueError("Physical bounds must be finite")
        result = Fraction.from_float(float(value))
    else:
        raise TypeError("Use real scalars or explicit rational strings for physical bounds")
    try:
        number = float(result)
        finite = math.isfinite(number) and (number != 0. or result == 0)
    except OverflowError as error:
        raise ValueError("Physical bounds must be representable as finite binary64 values") from error
    if not finite:
        raise ValueError("Physical bounds must be finite")
    return result


def _vector(values, length, name, *, positive=False):
    result = tuple(exact_scalar(value) for value in values)
    if len(result) != length or any(value <= 0 if positive else value < 0 for value in result):
        raise ValueError(f"{name} requires {length} finite {'positive' if positive else 'nonnegative'} components")
    return result


@dataclass(frozen=True)
class TerminalRegion:
    """Required position/sliding box about an explicitly bounded moving reference.

    ``reference_state_error_abs`` is the complete six-state evaluation error
    declaration, including any query-clock displacement. It is never inferred
    from an orientation error or numerical integration tolerance.
    """

    omega: Sequence[Any]
    phi: Sequence[Any]
    lambda_gain: Sequence[Any]
    reference_state_error_abs: Sequence[Any]
    anomaly_domain: Sequence[Any]
    error_source: str
    frame: str
    maximum_query_displacement: Any

    def __post_init__(self):
        for name in ("omega", "phi", "lambda_gain"):
            object.__setattr__(self, name, _vector(getattr(self, name), 3, name, positive=True))
        object.__setattr__(self, "reference_state_error_abs",
                           _vector(self.reference_state_error_abs, 6, "reference_state_error_abs"))
        domain = tuple(exact_scalar(value) for value in self.anomaly_domain)
        if len(domain) != 2 or not domain[0] < domain[1]:
            raise ValueError("Terminal reference errors require an ordered finite anomaly domain")
        object.__setattr__(self, "anomaly_domain", domain)
        displacement = exact_scalar(self.maximum_query_displacement)
        if displacement < 0:
            raise ValueError("Reference query displacement must be nonnegative")
        object.__setattr__(self, "maximum_query_displacement", displacement)
        for name in ("error_source", "frame"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"Terminal region {name} must be explicitly identified")

    def as_dict(self):
        return {name: [str(value) for value in getattr(self, name)]
                for name in ("omega", "phi", "lambda_gain", "reference_state_error_abs", "anomaly_domain")} | {
                    "error_source": self.error_source, "frame": self.frame,
                    "maximum_query_displacement": str(self.maximum_query_displacement)}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise TypeError("A terminal region must be an explicit mapping")
        return cls(**value)

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def scaled(self, fraction):
        fraction = exact_scalar(fraction)
        if not 0 < fraction <= 1:
            raise ValueError("A planning region fraction must lie in (0,1]")
        return TerminalRegion(tuple(fraction * value for value in self.omega),
            tuple(fraction * value for value in self.phi), self.lambda_gain,
            self.reference_state_error_abs, self.anomaly_domain, self.error_source, self.frame,
            self.maximum_query_displacement)

    def contains_anomaly(self, anomaly):
        return self.anomaly_domain[0] <= exact_scalar(anomaly) <= self.anomaly_domain[1]

    def supports_query(self, canonical_anomaly, query_anomaly):
        canonical, query = exact_scalar(canonical_anomaly), exact_scalar(query_anomaly)
        return (self.contains_anomaly(canonical) and self.contains_anomaly(query)
                and abs(canonical - query) <= self.maximum_query_displacement)

    def tube_margins(self, center, reference_state, phi, omega, lambda_gain):
        """Exact support of the correlated MPC tube in this reference region."""
        z, g = tuple(map(exact_scalar, center)), tuple(map(exact_scalar, reference_state))
        if len(z) != 6 or len(g) != 6:
            raise ValueError("Terminal region states have six components")
        phi = _vector(phi, 3, "MPC phi")
        omega = _vector(omega, 3, "MPC omega")
        lam = _vector(lambda_gain, 3, "MPC lambda_gain", positive=True)
        ep = tuple(z[i] - g[i] for i in range(3))
        ev = tuple(z[i + 3] - g[i + 3] for i in range(3))
        position = tuple(self.omega[i] - abs(ep[i]) - omega[i] - self.reference_state_error_abs[i]
                         for i in range(3))
        sliding = tuple(self.phi[i] - abs(ev[i] + self.lambda_gain[i] * ep[i]) - phi[i]
            - abs(self.lambda_gain[i] - lam[i]) * omega[i] - self.reference_state_error_abs[i + 3]
            - self.lambda_gain[i] * self.reference_state_error_abs[i] for i in range(3))
        return position, sliding

    def contains_endpoint(self, state_box, reference_state, anomaly, *, query_anomaly=None):
        """Screen the entire carried interval endpoint, with reference errors.

        Exact box arithmetic prevents a printed point replacing the independently
        enclosed state. This membership screen does not establish the physical
        terminal controller's global inequalities or the validity of its errors.
        """
        query = anomaly if query_anomaly is None else query_anomaly
        if state_box.dimension != 6 or not self.supports_query(anomaly, query):
            return False
        reference = tuple(map(exact_scalar, reference_state))
        if len(reference) != 6:
            raise ValueError("Terminal reference state has six components")
        for i in range(3):
            ep_lo = state_box.lower[i] - reference[i] - self.reference_state_error_abs[i]
            ep_hi = state_box.upper[i] - reference[i] + self.reference_state_error_abs[i]
            ev_lo = state_box.lower[i + 3] - reference[i + 3] - self.reference_state_error_abs[i + 3]
            ev_hi = state_box.upper[i + 3] - reference[i + 3] + self.reference_state_error_abs[i + 3]
            if (ep_lo < -self.omega[i] or ep_hi > self.omega[i]
                    or ev_lo + self.lambda_gain[i] * ep_lo < -self.phi[i]
                    or ev_hi + self.lambda_gain[i] * ep_hi > self.phi[i]):
                return False
        return True
