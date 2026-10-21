"""Model-local CLF/CBF proposals; physical admission remains a separate layer.

Distance/halfspace derivatives use physical-time double-integrator acceleration
coordinates. They must not be applied directly to anomaly-coordinate commands.
Derived from the MIT Swarm CLF-CBF Playground; see THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve_continuous_are


def vector(value, size, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f'{name} must be a finite {size}-vector')
    return result


@dataclass(frozen=True)
class BarrierEvaluation:
    """One hard affine proposal row, row @ command <= bound.

    Initial h >= 0 and psi1 >= 0 are required by the stated HOCBF comparison.
    A floating-point row is not a continuous/sampled nonlinear certificate.
    """
    row: np.ndarray
    bound: float
    h: float
    h_dot: float
    psi1: float
    uncertainty_support: float

    @property
    def initial_set_satisfied(self):
        return self.h >= 0 and self.psi1 >= 0


@dataclass(frozen=True)
class DistanceBarrier:
    safe_distance: float
    kappa1: float
    kappa2: float

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in (self.safe_distance, self.kappa1, self.kappa2)):
            raise ValueError('Distance and both HOCBF rates must be positive and finite')

    def evaluate(self, position, velocity, other_position, other_velocity, *,
                 other_acceleration=(0., 0., 0.), disturbance_radius=(0., 0., 0.),
                 controlled_other=False):
        r = vector(position, 3, 'position') - vector(other_position, 3, 'other position')
        v = vector(velocity, 3, 'velocity') - vector(other_velocity, 3, 'other velocity')
        a = vector(other_acceleration, 3, 'other acceleration')
        radius = vector(disturbance_radius, 3, 'relative acceleration support')
        if np.any(radius < 0):
            raise ValueError('Relative disturbance radii cannot be negative')
        if controlled_other and np.any(a):
            raise ValueError('A controlled neighbor uses its own command columns, not a fixed acceleration')
        h = float(r @ r - self.safe_distance**2)
        hd = float(2 * r @ v)
        drift = float(2 * v @ v - 2 * r @ a)
        support = float(2 * np.abs(r) @ radius)
        row = np.r_[-2*r, 2*r] if controlled_other else -2*r
        return BarrierEvaluation(row, drift + (self.kappa1+self.kappa2)*hd
            + self.kappa1*self.kappa2*h - support, h, hd,
            hd+self.kappa1*h, support)


@dataclass(frozen=True)
class HalfspaceBarrier:
    normal: tuple[float, float, float]
    offset: float
    kappa1: float
    kappa2: float

    def __post_init__(self):
        n = vector(self.normal, 3, 'halfspace unit normal')
        if not np.isclose(n @ n, 1., rtol=0., atol=1e-12):
            raise ValueError('Halfspace normal must have unit length')
        if not np.isfinite(self.offset) or self.offset < 0 or any(
                not np.isfinite(k) or k <= 0 for k in (self.kappa1, self.kappa2)):
            raise ValueError('Use a finite nonnegative offset and positive HOCBF rates')

    def evaluate(self, position, velocity, point, *, point_velocity=(0.,0.,0.),
                 point_acceleration=(0.,0.,0.), disturbance_radius=(0.,0.,0.)):
        n = np.asarray(self.normal)
        h = float(n @ (vector(position,3,'position')-vector(point,3,'point'))-self.offset)
        hd = float(n @ (vector(velocity,3,'velocity')-vector(point_velocity,3,'point velocity')))
        radius = vector(disturbance_radius,3,'acceleration support')
        if np.any(radius < 0):
            raise ValueError('Acceleration support must be nonnegative')
        support = float(np.abs(n) @ radius)
        drift = -float(n @ vector(point_acceleration,3,'point acceleration'))
        return BarrierEvaluation(-n, drift+(self.kappa1+self.kappa2)*hd
            +self.kappa1*self.kappa2*h-support, h, hd, hd+self.kappa1*h, support)


@dataclass(frozen=True)
class BrakingResult:
    margin_m: float | None
    required_gap_m: float | None
    net_braking_m_s2: float
    model: str = 'one-dimensional-constant-adverse-acceleration-with-latency'


class BrakingEnvelope:
    """Diagnostic stopping envelope, not a 3D spacecraft viability theorem."""
    @staticmethod
    def evaluate(gap_m, closing_speed_m_s, available_away_acceleration_m_s2, *,
                 adverse_acceleration_m_s2=0., latency_s=0.):
        values = (gap_m, closing_speed_m_s, available_away_acceleration_m_s2,
                  adverse_acceleration_m_s2, latency_s)
        if not np.all(np.isfinite(values)) or any(v < 0 for v in values[1:]):
            raise ValueError('Use a finite gap and nonnegative speed, acceleration and latency')
        net = available_away_acceleration_m_s2-adverse_acceleration_m_s2
        if net <= 0:
            return BrakingResult(None, None, net)
        delayed_speed = closing_speed_m_s+adverse_acceleration_m_s2*latency_s
        required = closing_speed_m_s*latency_s+.5*adverse_acceleration_m_s2*latency_s**2
        required += delayed_speed**2/(2*net)
        return BrakingResult(float(gap_m-required), float(required), float(net))


class QuadraticTrackingCLF:
    """True quadratic CLF row for a physical-time acceleration model.

    The row may be relaxed for progress. Collision and actuator rows must not
    inherit that slack. Saturation/obstacles can prevent V from decreasing.
    """
    def __init__(self, state_weight=None, input_weight=None):
        self.A = np.block([[np.zeros((3,3)), np.eye(3)], [np.zeros((3,6))]])
        self.B = np.vstack((np.zeros((3,3)), np.eye(3)))
        q = np.eye(6) if state_weight is None else np.asarray(state_weight,dtype=float)
        r = np.eye(3) if input_weight is None else np.asarray(input_weight,dtype=float)
        for value, size in ((q,6),(r,3)):
            if (value.shape!=(size,size) or not np.all(np.isfinite(value)) or
                    not np.allclose(value,value.T) or np.min(np.linalg.eigvalsh(value))<=0):
                raise ValueError('CLF design weights must be symmetric positive definite')
        self.P = solve_continuous_are(self.A, self.B, q, r)

    def evaluate(self, state, reference, *, reference_acceleration=(0.,0.,0.),
                 drift_acceleration=(0.,0.,0.), decay=1.):
        if not np.isfinite(decay) or decay <= 0:
            raise ValueError('CLF decay must be positive')
        e = vector(state,6,'state')-vector(reference,6,'reference')
        drift = np.r_[np.zeros(3),vector(drift_acceleration,3,'drift')
                      -vector(reference_acceleration,3,'reference acceleration')]
        value = float(e @ self.P @ e)
        row = 2*e @ self.P @ self.B
        bound = -decay*value-float(2*e @ self.P @ (self.A @ e+drift))
        return row, bound, value
