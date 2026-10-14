# /***********************************************************
# *                                                         *
# * Copyright (c) 2026                                      *
# *                                                         *
# * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
# * University of California, Los Angeles                   *
# *                                                         *
# * Authors: Aaron John Sabu                                *
# * Contact: aaronjs@ucla.edu                               *
# *                                                         *
# ***********************************************************/

import numpy as np
from typing import Any, Optional, Tuple

# Constants
TWO_PI = 2.0 * np.pi


def orbital_radius_terms(
    q,
    *,
    eccentricity=0.0,
    alpha=0.0,
    mu: float = 3.986004418e14,
    semi_major_axis: Optional[float] = None,
    mean_motion: float = 0.001,
    specific_angular_momentum=None,
    theta0: float = 0.0,
    solver=None,
) -> Tuple[Any, Any, Any]:
    """Return ``(R, dR/dq, h)`` for the inverse-radius quadratic-drag orbit.

    ``q`` is unwrapped orbital angle in radians, ``R`` is meters, and ``h``
    is the angular-momentum integration constant in m^2/s (not its square).
    The instantaneous angular momentum is ``h * exp(-alpha*q)``. ``alpha``
    is dimensionless; this analytical orbit assumes small radial speed.

    An explicit semi-major axis is authoritative: h is recomputed using the
    current eccentricity, including a symbolic estimated eccentricity. A
    supplied h is used only when no semi-major axis is supplied. Otherwise
    the semi-major axis is inferred from the positive Keplerian mean motion.
    NumPy scalars/arrays and solver-native symbolic expressions are supported.
    The orbit exists only where its radius denominator is strictly positive;
    symbolic callers must enforce that domain in their formulation.
    """
    backend = np if solver is None else solver
    numeric = backend is np
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("mu must be positive")
    if numeric:
        eccentricity_values = np.asarray(eccentricity, dtype=float)
        if np.any(~np.isfinite(eccentricity_values)) or np.any(
            (eccentricity_values < 0.0) | (eccentricity_values >= 1.0)
        ):
            raise ValueError("elliptical eccentricity must satisfy 0 <= e < 1")
    if semi_major_axis is not None:
        if not np.isfinite(semi_major_axis) or semi_major_axis <= 0.0:
            raise ValueError("semi_major_axis must be positive")
        h = backend.sqrt(mu * semi_major_axis * (1.0 - eccentricity**2))
    elif specific_angular_momentum is not None:
        h = specific_angular_momentum
    else:
        if not np.isfinite(mean_motion) or mean_motion <= 0.0:
            raise ValueError("mean_motion must be positive when deriving the orbit")
        orbit_axis = (mu / mean_motion**2) ** (1.0 / 3.0)
        h = backend.sqrt(mu * orbit_axis * (1.0 - eccentricity**2))
    if numeric and (
        np.any(~np.isfinite(np.asarray(h))) or np.any(np.asarray(h) <= 0.0)
    ):
        raise ValueError("specific angular momentum must be positive")

    exponential = backend.exp(2.0 * alpha * q)
    denominator = exponential + eccentricity * backend.cos(q - theta0)
    if numeric and (
        np.any(~np.isfinite(np.asarray(denominator)))
        or np.any(np.asarray(denominator) <= 0.0)
    ):
        raise ValueError("orbital radius denominator must be positive")
    scale = h**2 * (1.0 + 4.0 * alpha**2) / mu
    radius = scale / denominator
    denominator_prime = (
        2.0 * alpha * exponential - eccentricity * backend.sin(q - theta0)
    )
    radius_prime = -scale * denominator_prime / denominator**2
    return radius, radius_prime, h


def orbital_anomaly_rate(
    q,
    *,
    eccentricity=0.0,
    alpha=0.0,
    mu: float = 3.986004418e14,
    semi_major_axis: Optional[float] = None,
    mean_motion: float = 0.001,
    specific_angular_momentum=None,
    theta0: float = 0.0,
    solver=None,
):
    """Return dq/dt in rad/s for the same orbit as ``orbital_radius_terms``.

    This includes target drag; ``dq_dt`` below is the fixed Keplerian helper.
    """
    radius, _, h = orbital_radius_terms(
        q,
        eccentricity=eccentricity,
        alpha=alpha,
        mu=mu,
        semi_major_axis=semi_major_axis,
        mean_motion=mean_motion,
        specific_angular_momentum=specific_angular_momentum,
        theta0=theta0,
        solver=solver,
    )
    backend = np if solver is None else solver
    return h * backend.exp(-alpha * q) / radius**2


def true_anomaly_to_time(
    q: float,
    *,
    eccentricity: float = 0.0,
    mean_motion: float = 0.001,
    t_periapsis: float = 0.0,
    return_m: bool = False,
    return_e: bool = False,
    solver=None,
) -> float:
    """
    Convert unwrapped Keplerian true anomaly to elapsed time in seconds.

    Every revolution is retained, including negative orbital angles. This
    fixed Keplerian clock excludes drag; use ``orbital_anomaly_rate`` and
    integrate its reciprocal for the decaying analytical orbit.

    Args:
        q: True anomaly (radians).
        eccentricity: Orbital eccentricity.
        mean_motion: Mean motion (rad/s).
        t_periapsis: Time of periapsis passage (seconds).
        return_m: Whether to return mean anomaly.
        return_e: Whether to return eccentric anomaly.
        m: Modeling object (e.g., GEKKO) for symbolic math.

    Returns:
        Time (seconds) or tuple (time, mean_anomaly, eccentric_anomaly).
    """
    backend = np if solver is None else solver
    if not np.isfinite(mean_motion) or mean_motion <= 0.0:
        raise ValueError("mean_motion must be positive")
    if backend is np and np.any(
        ~np.isfinite(np.asarray(eccentricity))
        | (np.asarray(eccentricity) < 0.0)
        | (np.asarray(eccentricity) >= 1.0)
    ):
        raise ValueError("elliptical eccentricity must satisfy 0 <= e < 1")
    # E-q is a bounded periodic correction. Its denominator stays positive
    # for e<1, avoiding both tangent singularities and revolution wrapping.
    ratio = eccentricity / (1.0 + backend.sqrt(1.0 - eccentricity**2))
    arctangent = backend.arctan if backend is np else backend.atan
    eccentric_anomaly = q - 2.0 * arctangent(
        ratio * backend.sin(q) / (1.0 + ratio * backend.cos(q))
    )
    mean_anomaly = eccentric_anomaly - eccentricity * backend.sin(eccentric_anomaly)

    time = t_periapsis + mean_anomaly / mean_motion

    outputs = [time]
    if return_m:
        outputs.append(mean_anomaly)
    if return_e:
        outputs.append(eccentric_anomaly)

    if len(outputs) == 1:
        return time
    return tuple(outputs)


def dq_dt(
    q: float,
    *,
    eccentricity: float = 0.0,
    mean_motion: float = 0.001,
    t_periapsis: float = 0.0,
    solver=None,
) -> float:
    """
    Calculate the time derivative of true anomaly (anom_step/dt).

    Args:
        q: True anomaly (radians).
        eccentricity: Orbital eccentricity.
        mean_motion: Mean motion (rad/s).
        t_periapsis: Time of periapsis passage (seconds).
        solver: Modeling object for symbolic math.
    """
    backend = np if solver is None else solver
    if not np.isfinite(mean_motion) or mean_motion <= 0.0:
        raise ValueError("mean_motion must be positive")
    if backend is np and np.any(
        ~np.isfinite(np.asarray(eccentricity))
        | (np.asarray(eccentricity) < 0.0)
        | (np.asarray(eccentricity) >= 1.0)
    ):
        raise ValueError("elliptical eccentricity must satisfy 0 <= e < 1")
    # Equivalent to differentiating Kepler's equation, without the removable
    # cos(q/2)/cos(E/2) singularity at apogee.
    return mean_motion * (1.0 + eccentricity * backend.cos(q)) ** 2 / (
        1.0 - eccentricity**2
    ) ** 1.5


def dt_dq(
    q: float,
    *,
    eccentricity: float = 0.0,
    mean_motion: float = 0.001,
    t_periapsis: float = 0.0,
    solver=None,
) -> float:
    """Calculate the true anomaly derivative of time (dt/anom_step)."""
    return 1.0 / dq_dt(
        q,
        eccentricity=eccentricity,
        mean_motion=mean_motion,
        t_periapsis=t_periapsis,
        solver=solver,
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    q_start = -1.5 * TWO_PI
    q_end = 1.5 * TWO_PI
    num_samples = 2000
    test_eccentricity = 0.1

    q_grid = np.linspace(q_start, q_end, num_samples)
    time_grid = np.array(
        [true_anomaly_to_time(th, eccentricity=test_eccentricity) for th in q_grid]
    )
    q_dot_grid = np.array([dq_dt(th, eccentricity=test_eccentricity) for th in q_grid])
    dt_dq_grid = np.array([dt_dq(th, eccentricity=test_eccentricity) for th in q_grid])

    plt.figure(figsize=(10, 8))

    plt.subplot(3, 1, 1)
    plt.plot(q_grid / TWO_PI, time_grid)
    plt.ylabel("Time (s)")
    plt.title("True Anomaly vs Time")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(q_grid / TWO_PI, q_dot_grid)
    plt.ylabel("anom_step/dt (rad/s)")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(q_grid / TWO_PI, dt_dq_grid)
    plt.xlabel("Revolutions")
    plt.ylabel("dt/anom_step (s/rad)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
