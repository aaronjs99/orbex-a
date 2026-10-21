"""A parallel carrier-centred transfer into rotating proximity references."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from orbexa.planning.rendezvous.routes import _point, _positive


@dataclass(frozen=True)
class CarrierTransferReference:
    """Blend a fixed LVLH release centre into a small rotating close cluster.

    Only the finite dispenser offsets share the target's rotation. The distant
    carrier centre has zero initial LVLH velocity; it is never multiplied by
    the target rotation or angular rate. This declares post-release states,
    rather than simulating launch, ejection or a powered carrier trajectory.
    """

    center_lvlh_m: tuple[float, float, float]
    close_center_body_m: tuple[float, float, float]
    duration_rad: float

    def __post_init__(self):
        for name in ('center_lvlh_m', 'close_center_body_m'):
            object.__setattr__(self, name, tuple(_point(getattr(self, name), name)))
        object.__setattr__(self, 'duration_rad', _positive(self.duration_rad, 'Carrier transfer duration'))

    def kinematics(self, anomaly, offset_body_m, rotation, derivative, second):
        if not np.isfinite(anomaly):
            raise ValueError('Carrier transfer anomaly must be finite')
        offset = _point(offset_body_m, 'Dispenser offset')
        matrices = [np.asarray(value, dtype=float) for value in (rotation, derivative, second)]
        if any(value.shape != (3, 3) or not np.all(np.isfinite(value)) for value in matrices):
            raise ValueError('Carrier transfer needs finite rotation jets')
        r, dr, ddr = matrices
        if not np.allclose(r.T @ r, np.eye(3), rtol=0, atol=1e-9) or abs(np.linalg.det(r)-1.) > 1e-9:
            raise ValueError('Carrier transfer requires a proper rotation')
        t = float(np.clip(anomaly / self.duration_rad, 0., 1.))
        s = 10*t**3 - 15*t**4 + 6*t**5
        ds = (30*t**2 - 60*t**3 + 30*t**4) / self.duration_rad
        dds = (60*t - 180*t**2 + 120*t**3) / self.duration_rad**2
        center, close = np.asarray(self.center_lvlh_m), np.asarray(self.close_center_body_m)
        local = s*close + offset
        displacement = r @ close - center
        position = (1-s)*center + r @ local
        velocity = dr @ local + ds*displacement
        acceleration = ddr @ local + 2*ds*(dr @ close) + dds*displacement
        return np.r_[position, velocity], np.r_[velocity, acceleration]
