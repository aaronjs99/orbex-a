"""relative.py: Declared Cartesian forces and exact motion in commanded orbit coordinates.

The algebra functions are shared by numerical truth and the separate Arb
inclusion service. Only the latter supplies validated continuous enclosures.

Copyright: Aaron John Sabu
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path

import numpy as np
from orbexa.models.dynamics.reference_frame import (
    FloatArithmetic,
    bounds,
    exact,
    fingerprint,
)

_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def dot(a, b):
    """Return the inner product of two sequences."""
    return sum(x * y for x, y in zip(a, b))


def cross(a, b):
    """Return the cross product ``a x b`` of two 3-vectors as a tuple."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def matvec(matrix, vector):
    """Return the product of a row-major matrix and a vector as a tuple."""
    return tuple(dot(row, vector) for row in matrix)


def transpose(matrix):
    """Return the transpose of a row-major matrix as a tuple of tuples."""
    return tuple(zip(*matrix))


def z(vector):
    """Return ``(-y, x, 0)`` for a 3-vector, the z axis crossed with it.

    The third component is ``0 * vector[2]``, a zero of the component type."""
    return (-vector[1], vector[0], 0 * vector[2])


@dataclass(frozen=True)
class ExponentialAtmosphere:
    """Declared radial exponential atmosphere, rigidly co-rotating, with wind.

    Fields: ``rho0_kg_m3`` (kg/m^3, density at ``reference_radius_m``),
    ``reference_radius_m`` (m), ``scale_height_m`` (m), ``rotation_rad_s`` (rad/s,
    three components), ``wind_m_s`` (m/s, three components) and ``declaration``
    (provenance text). Scalars and vector components may be given as a value or a
    ``(lo, hi)`` interval."""

    rho0_kg_m3: object = "2e-11"
    reference_radius_m: object = 6_678_137
    scale_height_m: object = 50_000
    rotation_rad_s: tuple = (0, 0, "7.292115e-5")
    wind_m_s: tuple = (0, 0, 0)
    declaration: str = "Illustrative co-rotating radial exponential study atmosphere"

    def __post_init__(self):
        """Convert the fields to exact intervals and validate the atmosphere.

        State changes: replaces ``rho0_kg_m3`` and ``scale_height_m`` with ``(lo, hi)``
        Fraction pairs, ``reference_radius_m`` with a Fraction and ``rotation_rad_s``
        and ``wind_m_s`` with tuples of ``(lo, hi)`` pairs.
        Outputs: raises ValueError when the density lower bound is negative, the scale
        height lower bound or reference radius is not positive, a vector does not have
        three components, or ``declaration`` is not a non-blank string."""
        for name in ("rho0_kg_m3", "scale_height_m"):
            object.__setattr__(self, name, bounds(getattr(self, name)))
        object.__setattr__(self, "reference_radius_m", exact(self.reference_radius_m))
        if (
            self.rho0_kg_m3[0] < 0
            or self.scale_height_m[0] <= 0
            or self.reference_radius_m <= 0
        ):
            raise ValueError(
                "Atmosphere needs nonnegative density and positive radii/height"
            )
        for name in ("rotation_rad_s", "wind_m_s"):
            values = tuple(bounds(v) for v in getattr(self, name))
            if len(values) != 3:
                raise ValueError("Atmosphere vectors have three components")
            object.__setattr__(self, name, values)
        if not isinstance(self.declaration, str) or not self.declaration.strip():
            raise ValueError("Atmosphere provenance declaration is required")

    def as_dict(self):
        """Return a JSON-serializable record of the atmosphere, numbers as strings."""
        return {
            "law": "radial-exponential-rigid-corotation-constant-wind-v1",
            "rho0_kg_m3": tuple(map(str, self.rho0_kg_m3)),
            "reference_radius_m": str(self.reference_radius_m),
            "scale_height_m": tuple(map(str, self.scale_height_m)),
            "rotation_rad_s": tuple(tuple(map(str, v)) for v in self.rotation_rad_s),
            "wind_m_s": tuple(tuple(map(str, v)) for v in self.wind_m_s),
            "declaration": self.declaration,
        }


@dataclass(frozen=True)
class NonlinearModelBounds:
    """Declared finite-domain plant: central gravity, J2 and vector drag.

    Fields: ``mu_m3_s2`` (m^3/s^2), ``j2``, ``body_radius_m`` and
    ``gravity_reference_radius_m`` (m), ``j2_axis`` (direction of the J2 axis),
    ``atmosphere``, ``sigma_target`` and ``sigma_chaser`` (drag parameters that
    multiply ``-rho * |v_rel| * v_rel / 2``), ``radius_domain_m`` (``(lo, hi)`` m
    interval of allowed distance from the central body), ``relative_position_limit_m``
    (m) and ``relative_velocity_limit_m_per_rad`` (m/rad) componentwise limits on
    the relative state, and ``declaration`` (provenance text). Parameters given as
    ``(lo, hi)`` intervals are the physical uncertainty bounds."""

    mu_m3_s2: object = "398600441800000"
    j2: object = "1.08262668e-3"
    body_radius_m: object = 6_378_137
    gravity_reference_radius_m: object = 6_378_137
    j2_axis: tuple = (0, 0, 1)
    atmosphere: ExponentialAtmosphere = ExponentialAtmosphere()
    sigma_target: object = "0.02"
    sigma_chaser: object = "0.045"
    radius_domain_m: tuple = (6_400_000, 12_000_000)
    relative_position_limit_m: object = 500
    relative_velocity_limit_m_per_rad: object = 10_000
    declaration: str = "Finite synthetic central/J2/isotropic-drag study plant"

    def __post_init__(self):
        """Convert the fields to exact values and validate the plant declaration.

        State changes: replaces ``mu_m3_s2``, ``j2``, ``sigma_target`` and
        ``sigma_chaser`` with ``(lo, hi)`` Fraction pairs, ``radius_domain_m`` likewise,
        the length and limit fields with Fractions and ``j2_axis`` with a Fraction
        tuple.
        Outputs: raises ValueError when ``j2_axis`` is not a nonzero 3-vector, when
        ``mu_m3_s2`` is not positive, a sigma or ``body_radius_m`` lower bound is
        negative, ``gravity_reference_radius_m`` or a relative limit is not positive,
        the lower ``radius_domain_m`` bound does not exceed ``body_radius_m``,
        ``atmosphere`` is not an ``ExponentialAtmosphere``, or ``declaration`` is not a
        non-blank string."""
        for name in ("mu_m3_s2", "j2", "sigma_target", "sigma_chaser"):
            object.__setattr__(self, name, bounds(getattr(self, name)))
        for name in (
            "body_radius_m",
            "gravity_reference_radius_m",
            "relative_position_limit_m",
            "relative_velocity_limit_m_per_rad",
        ):
            object.__setattr__(self, name, exact(getattr(self, name)))
        object.__setattr__(self, "radius_domain_m", bounds(self.radius_domain_m))
        axis = tuple(map(exact, self.j2_axis))
        if len(axis) != 3 or dot(axis, axis) <= 0:
            raise ValueError("J2 axis needs a nonzero declared direction")
        object.__setattr__(self, "j2_axis", axis)
        if (
            self.mu_m3_s2[0] <= 0
            or self.sigma_target[0] < 0
            or self.sigma_chaser[0] < 0
            or self.body_radius_m < 0
            or self.gravity_reference_radius_m <= 0
            or self.radius_domain_m[0] <= self.body_radius_m
            or self.relative_position_limit_m <= 0
            or self.relative_velocity_limit_m_per_rad <= 0
            or not isinstance(self.atmosphere, ExponentialAtmosphere)
            or not isinstance(self.declaration, str)
            or not self.declaration.strip()
        ):
            raise ValueError("Invalid finite nonlinear plant declaration")

    def as_dict(self):
        """Return a JSON-serializable record of the plant with numbers as strings.

        The record holds the model name, the SHA-256 of this source file, every
        declared field and the nested atmosphere record."""
        return dict(
            model="central-j2-exponential-vector-drag-v1",
            source_sha256=_SOURCE_SHA256,
            **{
                name: tuple(map(str, getattr(self, name)))
                for name in (
                    "mu_m3_s2",
                    "j2",
                    "sigma_target",
                    "sigma_chaser",
                    "radius_domain_m",
                )
            },
            **{
                name: str(getattr(self, name))
                for name in (
                    "body_radius_m",
                    "gravity_reference_radius_m",
                    "relative_position_limit_m",
                    "relative_velocity_limit_m_per_rad",
                )
            },
            j2_axis=tuple(map(str, self.j2_axis)),
            atmosphere=self.atmosphere.as_dict(),
            declaration=self.declaration,
        )

    @cached_property
    def fingerprint(self):
        """Return the SHA-256 hex digest of ``as_dict()``."""
        return fingerprint(self.as_dict())

    @cached_property
    def target_fingerprint(self):
        """Return the SHA-256 hex digest of the target-only part of ``as_dict()``.

        ``sigma_chaser``, ``relative_position_limit_m`` and
        ``relative_velocity_limit_m_per_rad`` are removed before hashing."""
        record = self.as_dict()
        for name in (
            "sigma_chaser",
            "relative_position_limit_m",
            "relative_velocity_limit_m_per_rad",
        ):
            record.pop(name)
        return fingerprint(record)

    def midpoint_model(self):
        """Explicit synthetic realization; never an implicit safety-bound choice."""
        midpoint = lambda pair: (pair[0] + pair[1]) / 2
        atmosphere = replace(
            self.atmosphere,
            rho0_kg_m3=midpoint(self.atmosphere.rho0_kg_m3),
            scale_height_m=midpoint(self.atmosphere.scale_height_m),
            rotation_rad_s=tuple(map(midpoint, self.atmosphere.rotation_rad_s)),
            wind_m_s=tuple(map(midpoint, self.atmosphere.wind_m_s)),
        )
        return replace(
            self,
            atmosphere=atmosphere,
            **{
                name: midpoint(getattr(self, name))
                for name in ("mu_m3_s2", "j2", "sigma_target", "sigma_chaser")
            },
        )


def parameters(model, arithmetic):
    """Arithmetic supplies an interval conversion; floats require point laws."""
    s = arithmetic.scalar
    interval = getattr(arithmetic, "interval", None)

    def convert(pair):
        """Return ``pair`` as an arithmetic interval, or as a scalar without one.

        Outputs: raises ValueError when the arithmetic has no interval constructor and
        the pair is not a single value (``lo != hi``)."""
        if interval is not None:
            return interval(pair)
        if pair[0] != pair[1]:
            raise ValueError("Numerical truth requires a declared point realization")
        return s(pair[0])

    atmosphere = model.atmosphere
    axis = tuple(map(s, model.j2_axis))
    norm = arithmetic.sqrt(dot(axis, axis))
    return {
        "mu": convert(model.mu_m3_s2),
        "j2": convert(model.j2),
        "re": s(model.gravity_reference_radius_m),
        "axis": tuple(v / norm for v in axis),
        "sigma_target": convert(model.sigma_target),
        "sigma_chaser": convert(model.sigma_chaser),
        "rho0": convert(atmosphere.rho0_kg_m3),
        "h": convert(atmosphere.scale_height_m),
        "r0": s(atmosphere.reference_radius_m),
        "omega": tuple(map(convert, atmosphere.rotation_rad_s)),
        "wind": tuple(map(convert, atmosphere.wind_m_s)),
    }


def radial(position, model, arithmetic):
    """Return the distance of ``position`` from the central body, in m.

    Raises ValueError when the radius is not fully inside ``radius_domain_m``."""
    radius = arithmetic.sqrt(sum(arithmetic.square(value) for value in position))
    if not (
        radius >= arithmetic.scalar(model.radius_domain_m[0])
        and radius <= arithmetic.scalar(model.radius_domain_m[1])
    ):
        raise ValueError("Full radius enclosure escapes the declared physical domain")
    return radius


def gravity(position, radius, par, arithmetic):
    """Return the central plus J2 gravitational acceleration at ``position``, m/s^2.

    Inputs: ``position`` (m) and its distance ``radius`` (m); ``par`` is the
    dictionary from ``parameters``."""
    mu, axis = par["mu"], par["axis"]
    projected = dot(position, axis)
    factor = 3 * par["j2"] * mu * par["re"] ** 2 / 2
    return tuple(
        -mu * x / radius**3
        + factor
        * ((5 * arithmetic.square(projected) / radius**2 - 1) * x - 2 * projected * k)
        / radius**5
        for x, k in zip(position, axis)
    )


def drag(state, radius, par, arithmetic, sigma):
    """Return the drag acceleration of a spacecraft, m/s^2.

    The acceleration is ``-sigma * rho * |v_rel| * v_rel / 2``. ``v_rel`` is the
    state velocity minus the co-rotating air velocity ``omega x r`` and the wind,
    and ``rho`` decays exponentially with ``radius`` above ``r0``.
    Inputs: ``state`` is position (m) followed by velocity (m/s); ``sigma`` is the
    drag parameter of the spacecraft."""
    air_rotation = cross(par["omega"], state[:3])
    relative = tuple(v - a - b for v, a, b in zip(state[3:], air_rotation, par["wind"]))
    speed = arithmetic.sqrt(sum(arithmetic.square(value) for value in relative))
    density = par["rho0"] * arithmetic.exp(-(radius - par["r0"]) / par["h"])
    return tuple(-sigma * density * speed * v / 2 for v in relative)


def acceleration(state, model, par, arithmetic, sigma):
    """Return the gravity plus drag acceleration at ``state``, m/s^2.

    Raises ValueError when the radius is outside ``radius_domain_m``."""
    radius = radial(state[:3], model, arithmetic)
    return tuple(
        a + b
        for a, b in zip(
            gravity(state[:3], radius, par, arithmetic),
            drag(state, radius, par, arithmetic, sigma),
        )
    )


def differential_acceleration(target, delta_r, delta_v, model, par, arithmetic):
    """Exact canceled force difference, including unequal spacecraft drag.

    S-R=(2*r.d+d.d)/(S+R) and inverse-power identities avoid subtracting
    unrelated absolute gravity intervals. The J2 polynomial is canceled too.
    """
    r, velocity = target[:3], target[3:]
    chaser_r = tuple(a + b for a, b in zip(r, delta_r))
    radius, other = radial(r, model, arithmetic), radial(chaser_r, model, arithmetic)
    radius_delta = (2 * dot(r, delta_r) + dot(delta_r, delta_r)) / (other + radius)

    def inverse_difference(power):
        """Return ``1/other**power - 1/radius**power`` without subtracting them."""
        polynomial = sum(other ** (power - 1 - k) * radius**k for k in range(power))
        return -radius_delta * polynomial / (other**power * radius**power)

    central = tuple(
        -par["mu"] * (d / other**3 + x * inverse_difference(3))
        for x, d in zip(r, delta_r)
    )
    axis = par["axis"]
    projected, dp = dot(r, axis), dot(delta_r, axis)
    next_projected = projected + dp
    factor = 3 * par["j2"] * par["mu"] * par["re"] ** 2 / 2
    j2 = tuple(
        factor
        * (
            5
            * (
                (
                    (2 * projected * dp + dp * dp) * x
                    + arithmetic.square(next_projected) * d
                )
                / other**7
                + arithmetic.square(projected) * x * inverse_difference(7)
            )
            - (d / other**5 + x * inverse_difference(5))
            - 2 * k * (dp / other**5 + projected * inverse_difference(5))
        )
        for x, d, k in zip(r, delta_r, axis)
    )
    air = cross(par["omega"], r)
    wt = tuple(v - a - b for v, a, b in zip(velocity, air, par["wind"]))
    delta_air = cross(par["omega"], delta_r)
    dw = tuple(v - a for v, a in zip(delta_v, delta_air))
    wc = tuple(a + b for a, b in zip(wt, dw))
    st, sc = (
        arithmetic.sqrt(sum(arithmetic.square(value) for value in wt)),
        arithmetic.sqrt(sum(arithmetic.square(value) for value in wc)),
    )
    density = par["rho0"] * arithmetic.exp(-(radius - par["r0"]) / par["h"])
    if st + sc > 0:
        ds = (2 * dot(wt, dw) + dot(dw, dw)) / (st + sc)
        density_difference = density * arithmetic.expm1(-radius_delta / par["h"])
        next_density = density + density_difference
        difference = tuple(
            density_difference * st * v + next_density * (ds * v + sc * d)
            for v, d in zip(wt, dw)
        )
        drag_delta = tuple(
            -(
                (par["sigma_chaser"] - par["sigma_target"]) * density * st * v
                + par["sigma_chaser"] * d
            )
            / 2
            for v, d in zip(wt, difference)
        )
    else:
        chaser = chaser_r + tuple(v + d for v, d in zip(velocity, delta_v))
        drag_delta = tuple(
            a - b
            for a, b in zip(
                drag(chaser, other, par, arithmetic, par["sigma_chaser"]),
                drag(target, radius, par, arithmetic, par["sigma_target"]),
            )
        )
    return tuple(a + b + c for a, b, c in zip(central, j2, drag_delta))


def target_field(q, state, model, clock, arithmetic, par=None):
    """Return the derivative of the target state with respect to q.

    Inputs: ``q`` is the unwrapped anomaly in rad; ``state`` is target position
    (m) followed by velocity (m/s); ``par`` optionally supplies precomputed
    ``parameters``.
    Outputs: tuple of six components, velocity / rate then acceleration / rate,
    where rate is dq/dt in 1/s from ``clock``."""
    par = parameters(model, arithmetic) if par is None else par
    _, rate, _ = clock.evaluate(q, arithmetic)
    force = acceleration(state, model, par, arithmetic, par["sigma_target"])
    return tuple(v / rate for v in state[3:]) + tuple(a / rate for a in force)


def relative_field(q, state, target, command, model, clock, arithmetic, par=None):
    """Return the derivative of the relative state with respect to q.

    The relative state is position ``p`` (m) and ``w``, the derivative of ``p``
    with respect to q (m/rad), in the commanded orbit frame. The result is
    ``(w, w')`` with ``w'`` combining the rotated differential acceleration divided
    by rate squared, the rotating-frame terms, the ``kappa`` damping term and the
    command ``u``, which is added unchanged.
    Inputs: ``q`` in rad; ``state`` the six relative components; ``target`` the
    target position (m) and velocity (m/s); ``command`` three components."""
    par = parameters(model, arithmetic) if par is None else par
    rotation, rate, kappa = clock.evaluate(q, arithmetic)
    p, w = state[:3], state[3:]
    zp, zw, zzp = z(p), z(w), z(z(p))
    delta_r = matvec(rotation, p)
    delta_v = matvec(rotation, tuple(rate * (a + b) for a, b in zip(w, zp)))
    delta_a = differential_acceleration(
        target, delta_r, delta_v, model, par, arithmetic
    )
    normalized = matvec(transpose(rotation), delta_a)
    return tuple(w) + tuple(
        a / rate**2 - 2 * b - c - kappa * (d + e) + u
        for a, b, c, d, e, u in zip(normalized, zw, zzp, w, zp, command)
    )


class NonlinearRelativeDynamics:
    """Binary64 study truth for an explicitly point-valued physical model."""

    def __init__(self, model, clock):
        """Store the model, clock, float arithmetic and point parameters.

        Raises TypeError when ``model`` is not ``NonlinearModelBounds``. Raises
        ValueError when a model parameter is an interval rather than a single value."""
        if not isinstance(model, NonlinearModelBounds):
            raise TypeError("Supply NonlinearModelBounds")
        self.model, self.clock, self.arithmetic = model, clock, FloatArithmetic()
        self.parameters = parameters(model, self.arithmetic)

    @staticmethod
    def _state(value, count):
        """Return ``value`` as a float array of shape ``(count,)``.

        Raises ValueError when the shape differs or an entry is not finite."""
        value = np.asarray(value, dtype=float)
        if value.shape != (count,) or not np.all(np.isfinite(value)):
            raise ValueError(f"Expected finite {count}-vector")
        return value

    def target_rhs(self, q, target):
        """Return the derivative of the 6-component target state with respect to q.

        Inputs: ``q`` is the unwrapped anomaly in rad; ``target`` is position (m) and
        velocity (m/s). Raises ValueError when ``q`` is outside the clock domain, the
        target is not a finite 6-vector or its radius is outside the model domain."""
        self.clock.check_domain((q, q))
        return np.asarray(
            target_field(
                float(q),
                self._state(target, 6),
                self.model,
                self.clock,
                self.arithmetic,
                self.parameters,
            )
        )

    def relative_rhs(self, q, relative, target, u):
        """Return the derivative of the relative state with respect to q.

        Inputs: ``q`` in rad; ``relative`` is position (m) and its q derivative (m/rad)
        in the commanded orbit frame; ``target`` is position (m) and velocity (m/s);
        ``u`` is the three-component command.
        Outputs: array of six components. Raises ValueError when ``q`` is outside the
        clock domain, an input has the wrong shape or is not finite, or a relative
        component exceeds ``relative_position_limit_m`` or
        ``relative_velocity_limit_m_per_rad`` in magnitude."""
        self.clock.check_domain((q, q))
        state = self._state(relative, 6)
        if np.max(np.abs(state[:3])) > float(
            self.model.relative_position_limit_m
        ) or np.max(np.abs(state[3:])) > float(
            self.model.relative_velocity_limit_m_per_rad
        ):
            raise ValueError(
                "Relative state escapes its declared componentwise regional domain"
            )
        return np.asarray(
            relative_field(
                float(q),
                state,
                self._state(target, 6),
                self._state(u, 3),
                self.model,
                self.clock,
                self.arithmetic,
                self.parameters,
            )
        )
