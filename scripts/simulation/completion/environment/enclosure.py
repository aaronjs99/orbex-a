"""Typed interval counterpart of the maintained completion J2/drag adapter.

Only the exact built-in adapter is accepted. Its cached point parameters must
still equal its declaration; a generic numeric callback is not an interval law.
The central term is canceled analytically, retaining the declared J2 axis,
co-rotating atmosphere, wind, density, drag semantics and physical ECI frame.
"""
from __future__ import annotations
from orbexa.layout import PACKAGE_ROOT

from dataclasses import dataclass
from fractions import Fraction
import copy
import hashlib
from pathlib import Path

from orbexa.control.certification.reachability import _ProofUnresolved
from orbexa.core.dynamics.relative import parameters, radial, drag
from orbexa.core.dynamics.reference_frame import FloatArithmetic, fingerprint
from orbexa.simulation.numerics.rigid_interval import norm_upper, square


_SOURCE = {name: hashlib.sha256((PACKAGE_ROOT/name).read_bytes()).hexdigest()
    for name in (
                    'simulation/completion/environment/enclosure.py',
                    'simulation/completion/environment/point.py',
                    'simulation/numerics/rigid_interval.py',
                    'layout.py',
                )}


class _ForceArithmetic:
    def __init__(self, arithmetic):
        self.arithmetic = arithmetic
        self.scalar, self.interval = arithmetic.scalar, arithmetic.interval

    def sqrt(self, value):
        # All sqrt calls in this maintained force algebra are norms or the
        # fixed axis norm. Restrict their hull to the known nonnegative range
        # before taking endpoint roots, including exactly co-rotating motion.
        ar = self.arithmetic
        box = ar.box((value,))
        if box.upper[0] < 0:
            raise _ProofUnresolved('nonnegative force norm has no nonnegative enclosure')
        lo = ar.box((ar.scalar(max(0, box.lower[0])).sqrt(),)).lower[0]
        hi = ar.box((ar.scalar(box.upper[0]).sqrt(),)).upper[0]
        return ar.interval((lo, hi))

    @staticmethod
    def exp(value):
        return value.exp()

    def square(self, value):
        return square(value, self.arithmetic)


@dataclass(frozen=True)
class NonlinearEnvironmentEnclosure:
    model: object
    dry_mass: tuple
    drag_area: tuple | None
    point_environment_fingerprint: str

    @classmethod
    def from_environment(cls, environment):
        from orbexa.simulation.completion.environment.point import NonlinearEnvironment
        if type(environment) is not NonlinearEnvironment or type(environment.arithmetic) is not FloatArithmetic:
            raise _ProofUnresolved('extra environment is not the maintained typed nonlinear adapter')
        if environment.par != parameters(environment.model, FloatArithmetic()):
            raise _ProofUnresolved('nonlinear environment cached parameters differ from its declaration')
        names = set(environment.dry_mass)
        if 'target' not in names or (environment.drag_area is not None and set(environment.drag_area) != names):
            raise _ProofUnresolved('nonlinear environment fleet or drag-area declaration changed')
        if any(value <= 0 for value in environment.dry_mass.values()) or (environment.drag_area is not None
                and any(value < 0 for value in environment.drag_area.values())):
            raise _ProofUnresolved('nonlinear environment mass or drag-area declaration is not physical')
        return cls(copy.deepcopy(environment.model), tuple(sorted((key, Fraction(value))
            for key, value in environment.dry_mass.items())), None if environment.drag_area is None
            else tuple(sorted((key, Fraction(value)) for key, value in environment.drag_area.items())),
            fingerprint(environment.as_dict()))

    def as_dict(self):
        return dict(method='typed-arb-j2-vector-drag-extra-acceleration-v1', source_files=dict(_SOURCE),
            point_environment_fingerprint=self.point_environment_fingerprint,
            model=self.model.as_dict(), dry_mass_kg={key: str(value) for key, value in self.dry_mass},
            drag_area_m2_by_name=None if self.drag_area is None else {key: str(value) for key, value in self.drag_area},
            intrinsic_extra_torque='exactly zero; central gravity gradient remains separate')

    def force_parameters(self, ar):
        arithmetic = _ForceArithmetic(ar)
        return parameters(self.model, arithmetic), arithmetic

    def evaluate(self, name, position, velocity, fuel, ar, prepared=None):
        if name not in dict(self.dry_mass):
            raise _ProofUnresolved('interval environment body is outside the declared fleet')
        par, arithmetic = self.force_parameters(ar) if prepared is None else prepared
        radius = radial(position, self.model, arithmetic)
        projected = sum(x*y for x, y in zip(position, par['axis']))
        factor = 3*par['j2']*par['mu']*par['re']**2/2
        j2 = tuple(factor*((5*square(projected, ar)/radius**2-1)*x-2*projected*k)/radius**5
            for x, k in zip(position, par['axis']))
        sigma = par['sigma_target' if name == 'target' else 'sigma_chaser']
        if self.drag_area is not None:
            mass = ar.scalar(dict(self.dry_mass)[name])+fuel
            if not mass > 0:
                raise _ProofUnresolved('interval drag mass is not proved positive')
            sigma = ar.scalar(dict(self.drag_area)[name])/mass
        drag_force = drag(tuple(position)+tuple(velocity), radius, par, arithmetic, sigma)
        return tuple(a+b for a, b in zip(j2, drag_force))

    def norm_upper_over_domain(self, velocity_norm_upper, minimum_mass, ar):
        """Uniform extra acceleration over the declared radial/velocity domain.

        The J2 vector triangle bound is 9|J2| mu Re^2/r^4. The density
        maximum and air-relative speed bound preserve every declared support.
        This bound bootstraps a local pose/velocity domain; it is not a sensor
        or physical speed restriction imposed on a mission.
        """
        par, _ = self.force_parameters(ar)
        rlo, rhi = self.model.radius_domain_m
        radius = ar.scalar(rlo)
        density = par['rho0']*(-(radius-par['r0'])/par['h']).exp()
        density_upper = ar.box((density,)).upper[0]
        omega = norm_upper(par['omega'], ar)
        wind = norm_upper(par['wind'], ar)
        speed = Fraction(velocity_norm_upper)+omega*rhi+wind
        mu = self.model.mu_m3_s2[1]
        j2 = max(abs(x) for x in self.model.j2)
        j2_upper = 9*j2*mu*self.model.gravity_reference_radius_m**2/rlo**4
        result = {}
        for name, dry in self.dry_mass:
            sigma = self.model.sigma_target[1] if name == 'target' else self.model.sigma_chaser[1]
            if self.drag_area is not None:
                mass = Fraction(minimum_mass[name])
                if mass <= 0:
                    raise _ProofUnresolved('uniform interval drag mass is not positive')
                sigma = dict(self.drag_area)[name]/mass
            result[name] = j2_upper+sigma*density_upper*speed*speed/2
        return result
