"""A numerical sensed-state request schedule; safety admission stays separate."""
from dataclasses import dataclass, replace
import math
from numbers import Real
import numpy as np
from orbexa.core.attitude import finite_vector

@dataclass(frozen=True)
class ApoapsisBurnWindow:
    half_angle_rad: float | None = None
    minimum_eccentricity: float = 1e-8

    def __post_init__(self):
        if self.half_angle_rad is not None:
            if isinstance(self.half_angle_rad, (bool, np.bool_)) or not isinstance(self.half_angle_rad, Real):
                raise ValueError('Apogee window angle must be a finite real value')
            angle = float(self.half_angle_rad)
            if not math.isfinite(angle) or not 0 < angle <= math.pi:
                raise ValueError('Apogee window half angle must be in (0, pi]')
            object.__setattr__(self, 'half_angle_rad', angle)
        if isinstance(self.minimum_eccentricity, (bool, np.bool_)) or not isinstance(self.minimum_eccentricity, Real):
            raise ValueError('Circular-orbit threshold must be a finite real value')
        value = float(self.minimum_eccentricity)
        if not math.isfinite(value) or not 0 < value < 1:
            raise ValueError('Circular-orbit threshold must lie strictly between zero and one')
        object.__setattr__(self, 'minimum_eccentricity', value)

    def evaluate(self, position, velocity, mu):
        if self.half_angle_rad is None:
            return dict(active=True, rule='continuous_requested_burn')
        r, v = finite_vector(position,'Window position',3), finite_vector(velocity,'Window velocity',3)
        mu = float(mu)
        radius = float(np.linalg.norm(r))
        if not math.isfinite(mu) or mu <= 0 or radius <= 0:
            raise ValueError('Apogee request window needs a valid radius and gravitational parameter')
        angular_momentum = np.cross(r,v)
        if float(np.dot(v,v)/2-mu/radius) >= 0 or np.linalg.norm(angular_momentum) == 0:
            raise ValueError('An osculating apogee request window requires a bound nonradial orbit')
        eccentricity_vector = np.cross(v,angular_momentum)/mu-r/radius
        eccentricity = float(np.linalg.norm(eccentricity_vector))
        if eccentricity <= self.minimum_eccentricity:
            return dict(active=True, rule='no_unique_apogee_for_near_circular_sensed_orbit', eccentricity=eccentricity)
        cosine = float(np.clip(np.dot(r/radius,eccentricity_vector/eccentricity),-1.,1.))
        threshold = -math.cos(self.half_angle_rad)
        return dict(active=cosine <= threshold, eccentricity=eccentricity,
            true_anomaly_cosine=cosine, maximum_cosine=threshold,
            rule='cos(nu) <= -cos(half_angle) about the sensed osculating apogee')

    def as_dict(self):
        return dict(half_angle_rad=self.half_angle_rad, minimum_eccentricity=self.minimum_eccentricity,
            scope='Numerical request schedule from the supplied sensed/planning state; each actual held wrench, load, fuel, state flow and entry event still requires independent admission')


def supported_planning_state(physical, sensed):
    """Use a sensed center while retaining the entire admitted physical set."""
    if physical.time_s != sensed.time_s:
        raise ValueError('Burn planning needs the exact current sensed/physical clock')
    if physical.enclosure is None:
        return sensed
    from orbexa.control.reachability import IntervalBox
    from orbexa.simulation.assembly_enclosure import packed_state
    result = replace(sensed, enclosure=physical.enclosure)
    if not physical.enclosure.box.contains(IntervalBox.point(packed_state(result))):
        raise ValueError('Sensed burn planning center is outside the retained whole-state support')
    return result


def bounded_hold_duration(requested_s, remaining_s, *, current_time_s=None, deadline_s=None):
    """Bound a hold and, when supplied, its rounded point-clock endpoint."""
    from fractions import Fraction
    requested = float(requested_s)
    if not math.isfinite(requested) or requested <= 0:
        raise ValueError('Requested burn hold must be finite and positive')
    remaining = Fraction(remaining_s)
    if remaining <= 0:
        return 0.
    admitted = min(Fraction(requested), remaining)
    def rounded_down(value):
        result = float(value)
        return math.nextafter(result, 0.) if Fraction(result) > value else result
    duration = rounded_down(admitted)
    if (current_time_s is None) != (deadline_s is None):
        raise ValueError('Point-clock checking needs both current time and exact deadline')
    if current_time_s is not None:
        current = float(current_time_s)
        deadline = Fraction(deadline_s)
        if not math.isfinite(current):
            raise ValueError('Point-clock time must be finite')
        if Fraction(current+duration) > deadline:
            endpoint = float(deadline)
            if not math.isfinite(endpoint):
                raise ValueError('Point deadline needs a finite representable endpoint')
            if Fraction(endpoint) > deadline:
                endpoint = math.nextafter(endpoint, -math.inf)
            available = Fraction(endpoint)-Fraction(current)
            if available <= 0:
                return 0.
            admitted = min(admitted, available)
            duration = rounded_down(admitted)
        if Fraction(current+duration) > deadline:
            raise ValueError('Rounded point-clock endpoint exceeds the burn deadline')
        if duration > 0 and current+duration <= current:
            return 0.
    if not math.isfinite(duration) or duration < 0 or Fraction(duration) > admitted:
        raise ValueError('Remaining burn hold has no supported finite duration')
    return duration
