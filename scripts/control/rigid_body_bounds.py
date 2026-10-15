"""Global physical spin and body-to-LVLH rotation bounds on a finite orbit.

The initial spin enclosure, fixed inertia and environmental torque model are
explicit inputs. Orbital radius/clock and initial energy are bounded by Arb;
the energy comparison is analytic and never uses dense-trajectory samples.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .physical_terminal import (
    OrbitalClockBounds, _encode, _inertia_bounds, bound_orbital_clock,
)
from .reachability import IntervalBox, OrbitalModelBounds, _ARB_LOCK, _ArbArithmetic


RIGID_BODY_ROTATION_METHOD = "arb-rigid-energy-orbital-rotation-v1"


@dataclass(frozen=True)
class RigidBodyRotationBounds:
    """Conditional analytical certificate, independent of numerical pose error.

    ``initial_spin`` encloses inertial-relative body angular velocity in rad/s
    at ``clock.anomaly[0]``. The relative rate is per anomaly for a planar LVLH
    basis rotating by one radian per anomaly radian. A numerical evaluated pose
    still needs its own declared operator-error contract.
    """

    clock: OrbitalClockBounds
    initial_spin: IntervalBox
    inertia: tuple[tuple[Fraction, ...], ...]
    gravity_gradient: bool
    inertia_eigenvalue_lower: Fraction
    inertia_eigenvalue_upper: Fraction
    entry_twice_energy_upper: Fraction
    entry_energy_speed_upper: Fraction
    torque_norm_upper: Fraction
    energy_speed_rate_upper: Fraction
    spin_norm_upper: Fraction
    relative_angular_rate_upper: Fraction
    precision_bits: int
    backend_version: str
    method: str = RIGID_BODY_ROTATION_METHOD

    def as_dict(self):
        return _encode(vars(self))


def bound_rigid_body_rotation(start_anomaly, end_anomaly, parameters: OrbitalModelBounds,
                             *, inertia, initial_spin: IntervalBox,
                             gravity_gradient: bool = True,
                             precision_bits: int = 160) -> RigidBodyRotationBounds:
    """Bound spin and relative angular speed over the complete finite interval.

    The model is a fixed physically realizable SPD inertia, Euler's equation,
    and either zero torque or central spherical gravity-gradient torque only.
    Given ``V=sqrt(omega.T J omega/j_min)``, energy balance implies
    ``|omega|<=V`` and the absolute comparison ``V(t)<=V(0)+tau_max*t/j_min``.
    Positive regularization establishes the same bound at zero energy. With
    the certified clock minimum ``nu_min``, ``||R'||<=1+V_max/nu_min``.

    Invalid/unresolved input raises; it never produces a bound. Floats denote
    their exact binary rational values. This function neither samples target
    motion nor verifies a numerical attitude integration or sensor accuracy.
    """
    if not isinstance(parameters, OrbitalModelBounds):
        raise TypeError("Supply explicit OrbitalModelBounds for the physical target clock")
    if not isinstance(initial_spin, IntervalBox) or initial_spin.dimension != 3:
        raise TypeError("Supply a three-dimensional initial inertial-relative body-spin enclosure")
    if not isinstance(gravity_gradient, bool):
        raise TypeError("gravity_gradient must be a boolean; additional torques need another model")
    tensor, jmin, jmax = _inertia_bounds(inertia)
    clock = bound_orbital_clock(start_anomaly, end_anomaly, parameters,
                               precision_bits=precision_bits)
    import flint
    with _ARB_LOCK, flint.ctx.workprec(clock.precision_bits):
        ar = _ArbArithmetic(flint)
        spin = ar.vector(initial_spin)
        twice_energy = sum((spin[i] * ar.scalar(tensor[i][j]) * spin[j]
                            for i in range(3) for j in range(3)), ar.arb(0))
        energy_upper = ar.box((twice_energy,)).upper[0]
        if energy_upper < 0:
            raise ValueError("Initial rotational energy contradicts physical inertia")
        entry_speed = ar.box((ar.scalar(energy_upper / jmin).sqrt(),)).upper[0]
    torque = (3 * parameters.mu * (jmax - jmin) / (2 * clock.radius.lower[0] ** 3)
              if gravity_gradient else Fraction(0))
    energy_rate = torque / jmin
    spin_upper = entry_speed + energy_rate * clock.physical_duration_upper
    relative = 1 + spin_upper / clock.anomaly_rate.lower[0]
    return RigidBodyRotationBounds(clock, initial_spin, tensor, gravity_gradient,
        jmin, jmax, energy_upper, entry_speed, torque, energy_rate, spin_upper,
        relative, clock.precision_bits, getattr(flint, "__version__", "unknown"))
