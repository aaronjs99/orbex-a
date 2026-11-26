"""resources.py: Study propulsion envelopes for finite refuge commands.

Copyright: Aaron John Sabu
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from orbexa.common.intervals import _exact
from orbexa.control.fallback.records import CommandResourceBound
from orbexa.control.fallback.refuge import _absbox, _identity, fingerprint
from orbexa.proofs.attitude import _sqrt_upper


@dataclass(frozen=True)
class StudyPropulsionBounds:
    """Declare prospective study propulsion limits used to debit force and fuel.

    Fields are the mass upper bound (kg), installed force (N), specific impulse lower
    bound (s), available fuel (kg), attitude force allocation (N), standard gravity
    (m/s^2) and a hardware declaration string. The declaration is not a thruster
    qualification.
    """

    mass_upper_kg: Any = 130
    installed_force_n: Any = 100
    specific_impulse_lower_s: Any = 300
    available_fuel_kg: Any = 10
    attitude_force_allocation_n: Any = 0
    g0_m_s2: Any = "9.80665"
    declaration: str = "Prospective study hardware; no physical thruster qualification"

    def __post_init__(self):
        """Convert the numeric fields to exact values and validate them.

        Mutates the fields to exact values. Raises ValueError unless mass, force,
        specific impulse and gravity are positive, fuel and attitude allocation are
        nonnegative, and the attitude allocation is below the installed force.
        """
        for name in (
            "mass_upper_kg",
            "installed_force_n",
            "specific_impulse_lower_s",
            "available_fuel_kg",
            "attitude_force_allocation_n",
            "g0_m_s2",
        ):
            value = _exact(getattr(self, name))
            if value < 0 or (
                name not in ("available_fuel_kg", "attitude_force_allocation_n")
                and value == 0
            ):
                raise ValueError(
                    "Physical resource bounds need positive mass/force/Isp/gravity and nonnegative fuel"
                )
            object.__setattr__(self, name, value)
        if self.attitude_force_allocation_n >= self.installed_force_n:
            raise ValueError(
                "Attitude allocation exhausts the installed study authority"
            )
        _identity(self.declaration, "study hardware declaration")

    @property
    def fingerprint(self):
        """Return the fingerprint of this declaration."""
        return fingerprint(self)

    def input_resource(self, key, input_box, q, h, clock):
        """Bound the force margin and fuel debit of a command box over one window.

        key is not used in the computation. The force bound scales the largest applied
        input by mass and the squared reference-clock rate upper bound, plus the
        attitude allocation. The fuel debit is rounded up to a common denominator of
        2**80 per unit of q, so it only increases.

        Inputs: input_box applied-input IntervalBox; q window start and h window width
            in the unwrapped-anomaly reference clock; clock supplying the rate bounds.
        Outputs: CommandResourceBound with the fuel upper bound (kg) and the force
            margin, installed force minus the force upper bound (N). Raises ValueError
            if the clock rate lower bound is not positive.
        """
        q, h = _exact(q), _exact(h)
        rate = clock.interval_bounds((q, q + h))["rate_s_inv"]
        lower, upper = rate.lower[0], rate.upper[0]
        if lower <= 0:
            raise ValueError("Resource proof requires a positive reference-clock rate")
        # For every proper rotation C and every member of the applied box,
        # ||C*u||1 <= sqrt(3)||u||2 <= sqrt(3 sum_i max|u_i|^2).
        # Keep the old valid bound when outward sqrt rounding is larger.
        # This tightens an upper debit; it does not predict actual fuel use.
        magnitudes = _absbox(input_box)
        input_upper = min(
            3 * max(magnitudes),
            _sqrt_upper(3 * sum(value * value for value in magnitudes)),
        )
        force_upper = (
            self.mass_upper_kg * upper**2 * input_upper
            + self.attitude_force_allocation_n
        )
        fuel_per_q = (
            self.mass_upper_kg * upper * input_upper
            + self.attitude_force_allocation_n / lower
        ) / (self.specific_impulse_lower_s * self.g0_m_s2)
        # Sum a common-denominator upper bound. Repeated exact reciprocals of
        # independent Arb rate endpoints otherwise accumulate huge coprime
        # denominators in the finite tank ledger. The ceiling increases every
        # debit, preserving its conservative physical meaning.
        quantum = 1 << 80
        scaled = fuel_per_q * quantum
        fuel_per_q = Fraction(
            (scaled.numerator + scaled.denominator - 1) // scaled.denominator, quantum
        )
        return CommandResourceBound(
            q,
            h,
            input_box,
            h * fuel_per_q,
            self.installed_force_n - force_upper,
            self.fingerprint,
            "analytic-nu-squared-sharp-SO3-body-L1-force-and-Isp-dyadic-upper80:"
            + clock.fingerprint,
        )

    def callback(self, clock):
        """Return a function (key, input_box, q, h) that calls input_resource."""
        return lambda key, input_box, q, h: self.input_resource(
            key, input_box, q, h, clock
        )
