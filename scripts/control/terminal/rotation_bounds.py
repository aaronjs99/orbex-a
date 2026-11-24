"""Global physical spin and body-to-LVLH rotation bounds on a finite orbit.

The initial spin enclosure, fixed inertia and environmental torque model are
explicit inputs. Orbital radius/clock and initial energy are bounded by Arb;
the energy comparison is analytic and never uses dense-trajectory samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from orbexa.control.terminal.physical import (
    OrbitalClockBounds,
    _encode,
)
from orbexa.proofs.reachability import (
    IntervalBox,
)

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
