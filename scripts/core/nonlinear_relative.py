"""Declared Cartesian forces and exact motion in commanded orbit coordinates.

The algebra functions are shared by numerical truth and the separate Arb
inclusion service. Only the latter supplies validated continuous enclosures.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
import hashlib
from pathlib import Path
import numpy as np
from .reference_frame import bounds, exact, fingerprint, FloatArithmetic

_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def matvec(matrix, vector):
    return tuple(dot(row, vector) for row in matrix)


def transpose(matrix):
    return tuple(zip(*matrix))


def z(vector):
    return (-vector[1], vector[0], 0*vector[2])


@dataclass(frozen=True)
class ExponentialAtmosphere:
    rho0_kg_m3: object = '2e-11'
    reference_radius_m: object = 6_678_137
    scale_height_m: object = 50_000
    rotation_rad_s: tuple = (0, 0, '7.292115e-5')
    wind_m_s: tuple = (0, 0, 0)
    declaration: str = 'Illustrative co-rotating radial exponential study atmosphere'

    def __post_init__(self):
        for name in ('rho0_kg_m3', 'scale_height_m'):
            object.__setattr__(self, name, bounds(getattr(self, name)))
        object.__setattr__(self, 'reference_radius_m', exact(self.reference_radius_m))
        if self.rho0_kg_m3[0] < 0 or self.scale_height_m[0] <= 0 or self.reference_radius_m <= 0:
            raise ValueError('Atmosphere needs nonnegative density and positive radii/height')
        for name in ('rotation_rad_s', 'wind_m_s'):
            values = tuple(bounds(v) for v in getattr(self, name))
            if len(values) != 3:
                raise ValueError('Atmosphere vectors have three components')
            object.__setattr__(self, name, values)
        if not isinstance(self.declaration, str) or not self.declaration.strip():
            raise ValueError('Atmosphere provenance declaration is required')

    def as_dict(self):
        return dict(law='radial-exponential-rigid-corotation-constant-wind-v1',
            rho0_kg_m3=tuple(map(str, self.rho0_kg_m3)),
            reference_radius_m=str(self.reference_radius_m),
            scale_height_m=tuple(map(str, self.scale_height_m)),
            rotation_rad_s=tuple(tuple(map(str, v)) for v in self.rotation_rad_s),
            wind_m_s=tuple(tuple(map(str, v)) for v in self.wind_m_s), declaration=self.declaration)


@dataclass(frozen=True)
class NonlinearModelBounds:
    mu_m3_s2: object = '398600441800000'
    j2: object = '1.08262668e-3'
    body_radius_m: object = 6_378_137
    gravity_reference_radius_m: object = 6_378_137
    j2_axis: tuple = (0, 0, 1)
    atmosphere: ExponentialAtmosphere = ExponentialAtmosphere()
    sigma_target: object = '0.02'
    sigma_chaser: object = '0.045'
    radius_domain_m: tuple = (6_400_000, 12_000_000)
    relative_position_limit_m: object = 500
    relative_velocity_limit_m_per_rad: object = 10_000
    declaration: str = 'Finite synthetic central/J2/isotropic-drag study plant'

    def __post_init__(self):
        for name in ('mu_m3_s2', 'j2', 'sigma_target', 'sigma_chaser'):
            object.__setattr__(self, name, bounds(getattr(self, name)))
        for name in ('body_radius_m', 'gravity_reference_radius_m',
                     'relative_position_limit_m', 'relative_velocity_limit_m_per_rad'):
            object.__setattr__(self, name, exact(getattr(self, name)))
        object.__setattr__(self, 'radius_domain_m', bounds(self.radius_domain_m))
        axis = tuple(map(exact, self.j2_axis))
        if len(axis) != 3 or dot(axis, axis) <= 0:
            raise ValueError('J2 axis needs a nonzero declared direction')
        object.__setattr__(self, 'j2_axis', axis)
        if (self.mu_m3_s2[0] <= 0 or self.sigma_target[0] < 0 or self.sigma_chaser[0] < 0
                or self.body_radius_m < 0 or self.gravity_reference_radius_m <= 0
                or self.radius_domain_m[0] <= self.body_radius_m
                or self.relative_position_limit_m <= 0 or self.relative_velocity_limit_m_per_rad <= 0
                or not isinstance(self.atmosphere, ExponentialAtmosphere)
                or not isinstance(self.declaration, str) or not self.declaration.strip()):
            raise ValueError('Invalid finite nonlinear plant declaration')

    def as_dict(self):
        return dict(model='central-j2-exponential-vector-drag-v1',
            source_sha256=_SOURCE_SHA256,
            **{name: tuple(map(str, getattr(self, name))) for name in
               ('mu_m3_s2', 'j2', 'sigma_target', 'sigma_chaser', 'radius_domain_m')},
            **{name: str(getattr(self, name)) for name in ('body_radius_m',
                'gravity_reference_radius_m', 'relative_position_limit_m', 'relative_velocity_limit_m_per_rad')},
            j2_axis=tuple(map(str, self.j2_axis)), atmosphere=self.atmosphere.as_dict(),
            declaration=self.declaration)

    @cached_property
    def fingerprint(self):
        return fingerprint(self.as_dict())

    @cached_property
    def target_fingerprint(self):
        record = self.as_dict()
        for name in ('sigma_chaser', 'relative_position_limit_m', 'relative_velocity_limit_m_per_rad'):
            record.pop(name)
        return fingerprint(record)

    def midpoint_model(self):
        """Explicit synthetic realization; never an implicit safety-bound choice."""
        midpoint = lambda pair: (pair[0]+pair[1])/2
        atmosphere = replace(self.atmosphere, rho0_kg_m3=midpoint(self.atmosphere.rho0_kg_m3),
            scale_height_m=midpoint(self.atmosphere.scale_height_m),
            rotation_rad_s=tuple(map(midpoint, self.atmosphere.rotation_rad_s)),
            wind_m_s=tuple(map(midpoint, self.atmosphere.wind_m_s)))
        return replace(self, atmosphere=atmosphere,
            **{name: midpoint(getattr(self, name)) for name in
               ('mu_m3_s2', 'j2', 'sigma_target', 'sigma_chaser')})


def parameters(model, arithmetic):
    """Arithmetic supplies an interval conversion; floats require point laws."""
    s = arithmetic.scalar
    interval = getattr(arithmetic, 'interval', None)
    def convert(pair):
        if interval is not None:
            return interval(pair)
        if pair[0] != pair[1]:
            raise ValueError('Numerical truth requires a declared point realization')
        return s(pair[0])
    atmosphere = model.atmosphere
    axis = tuple(map(s, model.j2_axis))
    norm = arithmetic.sqrt(dot(axis, axis))
    return dict(mu=convert(model.mu_m3_s2), j2=convert(model.j2),
        re=s(model.gravity_reference_radius_m), axis=tuple(v/norm for v in axis),
        sigma_target=convert(model.sigma_target), sigma_chaser=convert(model.sigma_chaser),
        rho0=convert(atmosphere.rho0_kg_m3), h=convert(atmosphere.scale_height_m),
        r0=s(atmosphere.reference_radius_m),
        omega=tuple(map(convert, atmosphere.rotation_rad_s)), wind=tuple(map(convert, atmosphere.wind_m_s)))


def radial(position, model, arithmetic):
    radius = arithmetic.sqrt(sum(arithmetic.square(value) for value in position))
    if not (radius >= arithmetic.scalar(model.radius_domain_m[0]) and
            radius <= arithmetic.scalar(model.radius_domain_m[1])):
        raise ValueError('Full radius enclosure escapes the declared physical domain')
    return radius


def gravity(position, radius, par, arithmetic):
    mu, axis = par['mu'], par['axis']
    projected = dot(position, axis)
    factor = 3*par['j2']*mu*par['re']**2/2
    return tuple(-mu*x/radius**3 + factor*((5*arithmetic.square(projected)/radius**2-1)*x
                  -2*projected*k)/radius**5 for x, k in zip(position, axis))


def drag(state, radius, par, arithmetic, sigma):
    air_rotation = cross(par['omega'], state[:3])
    relative = tuple(v-a-b for v, a, b in zip(state[3:], air_rotation, par['wind']))
    speed = arithmetic.sqrt(sum(arithmetic.square(value) for value in relative))
    density = par['rho0']*arithmetic.exp(-(radius-par['r0'])/par['h'])
    return tuple(-sigma*density*speed*v/2 for v in relative)


def acceleration(state, model, par, arithmetic, sigma):
    radius = radial(state[:3], model, arithmetic)
    return tuple(a+b for a, b in zip(gravity(state[:3], radius, par, arithmetic),
                 drag(state, radius, par, arithmetic, sigma)))


def differential_acceleration(target, delta_r, delta_v, model, par, arithmetic):
    """Exact canceled force difference, including unequal spacecraft drag.

    S-R=(2*r.d+d.d)/(S+R) and inverse-power identities avoid subtracting
    unrelated absolute gravity intervals. The J2 polynomial is canceled too.
    """
    r, velocity = target[:3], target[3:]
    chaser_r = tuple(a+b for a, b in zip(r, delta_r))
    radius, other = radial(r, model, arithmetic), radial(chaser_r, model, arithmetic)
    radius_delta = (2*dot(r, delta_r)+dot(delta_r, delta_r))/(other+radius)
    def inverse_difference(power):
        polynomial = sum(other**(power-1-k)*radius**k for k in range(power))
        return -radius_delta*polynomial/(other**power*radius**power)
    central = tuple(-par['mu']*(d/other**3+x*inverse_difference(3)) for x, d in zip(r, delta_r))
    axis = par['axis']
    projected, dp = dot(r, axis), dot(delta_r, axis)
    next_projected = projected+dp
    factor = 3*par['j2']*par['mu']*par['re']**2/2
    j2 = tuple(factor*(5*(((2*projected*dp+dp*dp)*x+arithmetic.square(next_projected)*d)/other**7
                    +arithmetic.square(projected)*x*inverse_difference(7))
                    -(d/other**5+x*inverse_difference(5))
                    -2*k*(dp/other**5+projected*inverse_difference(5)))
               for x, d, k in zip(r, delta_r, axis))
    air = cross(par['omega'], r)
    wt = tuple(v-a-b for v, a, b in zip(velocity, air, par['wind']))
    delta_air = cross(par['omega'], delta_r)
    dw = tuple(v-a for v, a in zip(delta_v, delta_air))
    wc = tuple(a+b for a, b in zip(wt, dw))
    st, sc = arithmetic.sqrt(sum(arithmetic.square(value) for value in wt)), arithmetic.sqrt(sum(arithmetic.square(value) for value in wc))
    density = par['rho0']*arithmetic.exp(-(radius-par['r0'])/par['h'])
    if st+sc > 0:
        ds = (2*dot(wt, dw)+dot(dw, dw))/(st+sc)
        density_difference = density*arithmetic.expm1(-radius_delta/par['h'])
        next_density = density+ density_difference
        difference = tuple(density_difference*st*v+next_density*(ds*v+sc*d) for v, d in zip(wt, dw))
        drag_delta = tuple(-((par['sigma_chaser']-par['sigma_target'])*density*st*v
                           +par['sigma_chaser']*d)/2 for v, d in zip(wt, difference))
    else:
        chaser = chaser_r+tuple(v+d for v, d in zip(velocity, delta_v))
        drag_delta = tuple(a-b for a, b in zip(drag(chaser, other, par, arithmetic, par['sigma_chaser']),
                                             drag(target, radius, par, arithmetic, par['sigma_target'])))
    return tuple(a+b+c for a, b, c in zip(central, j2, drag_delta))


def target_field(q, state, model, clock, arithmetic, par=None):
    par = parameters(model, arithmetic) if par is None else par
    _, rate, _ = clock.evaluate(q, arithmetic)
    force = acceleration(state, model, par, arithmetic, par['sigma_target'])
    return tuple(v/rate for v in state[3:])+tuple(a/rate for a in force)


def relative_field(q, state, target, command, model, clock, arithmetic, par=None):
    par = parameters(model, arithmetic) if par is None else par
    rotation, rate, kappa = clock.evaluate(q, arithmetic)
    p, w = state[:3], state[3:]
    zp, zw, zzp = z(p), z(w), z(z(p))
    delta_r = matvec(rotation, p)
    delta_v = matvec(rotation, tuple(rate*(a+b) for a, b in zip(w, zp)))
    delta_a = differential_acceleration(target, delta_r, delta_v, model, par, arithmetic)
    normalized = matvec(transpose(rotation), delta_a)
    return tuple(w)+tuple(a/rate**2-2*b-c-kappa*(d+e)+u for a,b,c,d,e,u
        in zip(normalized, zw, zzp, w, zp, command))


class NonlinearRelativeDynamics:
    """Binary64 study truth for an explicitly point-valued physical model."""
    def __init__(self, model, clock):
        if not isinstance(model, NonlinearModelBounds):
            raise TypeError('Supply NonlinearModelBounds')
        self.model, self.clock, self.arithmetic = model, clock, FloatArithmetic()
        self.parameters = parameters(model, self.arithmetic)

    @staticmethod
    def _state(value, count):
        value = np.asarray(value, dtype=float)
        if value.shape != (count,) or not np.all(np.isfinite(value)):
            raise ValueError(f'Expected finite {count}-vector')
        return value

    def target_rhs(self, q, target):
        self.clock.check_domain((q, q))
        return np.asarray(target_field(float(q), self._state(target, 6), self.model,
                          self.clock, self.arithmetic, self.parameters))

    def relative_rhs(self, q, relative, target, u):
        self.clock.check_domain((q, q))
        state = self._state(relative, 6)
        if (np.max(np.abs(state[:3])) > float(self.model.relative_position_limit_m) or
                np.max(np.abs(state[3:])) > float(self.model.relative_velocity_limit_m_per_rad)):
            raise ValueError('Relative state escapes its declared componentwise regional domain')
        return np.asarray(relative_field(float(q), state, self._state(target, 6), self._state(u, 3),
            self.model, self.clock, self.arithmetic, self.parameters))

    def fleet_rhs(self, q, target, relative_states, commands):
        if len(relative_states) != len(commands):
            raise ValueError('Every relative spacecraft needs one command')
        target_derivative = self.target_rhs(q, target)
        relative_derivatives = [self.relative_rhs(q, state, target, u) for state, u in zip(relative_states, commands)]
        return target_derivative, np.asarray(relative_derivatives), 1/self.clock.point(q).rate_s_inv
