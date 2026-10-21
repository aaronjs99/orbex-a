"""Study propulsion envelopes for finite refuge commands."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any
from orbexa.control.certification.reachability import _exact
from orbexa.control.fallback.refuge import _absbox, _identity, fingerprint
from orbexa.control.fallback.records import CommandResourceBound
from orbexa.control.certification.attitude import _sqrt_upper


@dataclass(frozen=True)
class StudyPropulsionBounds:
    mass_upper_kg: Any = 130
    installed_force_n: Any = 100
    specific_impulse_lower_s: Any = 300
    available_fuel_kg: Any = 10
    attitude_force_allocation_n: Any = 0
    g0_m_s2: Any = "9.80665"
    declaration: str = "Prospective study hardware; no physical thruster qualification"

    def __post_init__(self):
        for name in ("mass_upper_kg","installed_force_n","specific_impulse_lower_s","available_fuel_kg",
                     "attitude_force_allocation_n","g0_m_s2"):
            value = _exact(getattr(self,name))
            if value < 0 or (name not in ("available_fuel_kg","attitude_force_allocation_n") and value == 0):
                raise ValueError("Physical resource bounds need positive mass/force/Isp/gravity and nonnegative fuel")
            object.__setattr__(self,name,value)
        if self.attitude_force_allocation_n >= self.installed_force_n:
            raise ValueError("Attitude allocation exhausts the installed study authority")
        _identity(self.declaration,"study hardware declaration")

    @property
    def fingerprint(self):
        return fingerprint(self)

    def input_resource(self,key,input_box,q,h,clock):
        q,h = _exact(q),_exact(h)
        rate = clock.interval_bounds((q,q+h))["rate_s_inv"]
        lower,upper = rate.lower[0],rate.upper[0]
        if lower <= 0:
            raise ValueError("Resource proof requires a positive reference-clock rate")
        # For every proper rotation C and every member of the applied box,
        # ||C*u||1 <= sqrt(3)||u||2 <= sqrt(3 sum_i max|u_i|^2).
        # Keep the old valid bound when outward sqrt rounding is larger.
        # This tightens an upper debit; it does not predict actual fuel use.
        magnitudes = _absbox(input_box)
        input_upper = min(3*max(magnitudes),_sqrt_upper(3*sum(value*value for value in magnitudes)))
        force_upper = self.mass_upper_kg*upper**2*input_upper+self.attitude_force_allocation_n
        fuel_per_q = (self.mass_upper_kg*upper*input_upper+self.attitude_force_allocation_n/lower)/(
            self.specific_impulse_lower_s*self.g0_m_s2)
        # Sum a common-denominator upper bound. Repeated exact reciprocals of
        # independent Arb rate endpoints otherwise accumulate huge coprime
        # denominators in the finite tank ledger. The ceiling increases every
        # debit, preserving its conservative physical meaning.
        quantum=1<<80
        scaled=fuel_per_q*quantum
        fuel_per_q=Fraction((scaled.numerator+scaled.denominator-1)//scaled.denominator,quantum)
        return CommandResourceBound(q,h,input_box,h*fuel_per_q,self.installed_force_n-force_upper,
            self.fingerprint,"analytic-nu-squared-sharp-SO3-body-L1-force-and-Isp-dyadic-upper80:"+clock.fingerprint)

    def callback(self,clock):
        return lambda key,input_box,q,h: self.input_resource(key,input_box,q,h,clock)
