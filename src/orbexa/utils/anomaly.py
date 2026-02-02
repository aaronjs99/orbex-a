# /***********************************************************
# *                                                         *
# * Copyright (c) 2026                                      *
# *                                                         *
# * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
# * University of California, Los Angeles                   *
# *                                                         *
# * Authors: Aaron John Sabu, Brett T. Lopez                *
# * Contact: {aaronjs, btlopez}@ucla.edu                    *
# *                                                         *
# ***********************************************************/

import numpy as np
import matplotlib.pyplot as plt

# Constants
TWO_PI = 2.0 * np.pi


def true_anomaly_to_time(
    theta: float,
    *,
    eccentricity: float = 0.0,
    mean_motion: float = 0.001,
    t_periapsis: float = 0.0,
    return_m: bool = False,
    return_e: bool = False,
    solver=None,
) -> float:
    """
    Convert true anomaly to time since periapsis.

    Args:
        theta: True anomaly (radians).
        eccentricity: Orbital eccentricity.
        mean_motion: Mean motion (rad/s).
        t_periapsis: Time of periapsis passage (seconds).
        return_m: Whether to return mean anomaly.
        return_e: Whether to return eccentric anomaly.
        m: Modeling object (e.g., GEKKO) for symbolic math.

    Returns:
        Time (seconds) or tuple (time, mean_anomaly, eccentric_anomaly).
    """
    if eccentricity == 0.0:
        mean_anomaly = theta
        eccentric_anomaly = theta
    elif solver is None:
        # Numpy path (handles scalars and arrays)
        sin_half_theta = np.sin(theta / 2)
        cos_half_theta = np.cos(theta / 2)

        # Using arctan2 for stability and to avoid quadrant issues
        term = np.sqrt((1 - eccentricity) / (1 + eccentricity)) * sin_half_theta
        eccentric_anomaly = 2 * np.arctan2(term, cos_half_theta)

        mean_anomaly = eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly)
    else:
        # Modeling path (GEKKO, etc.)
        sin_half_theta = solver.sin(theta / 2)
        cos_half_theta = solver.cos(theta / 2)

        term = solver.sqrt((1 - eccentricity) / (1 + eccentricity)) * sin_half_theta
        eccentric_anomaly = 2 * solver.atan(term / cos_half_theta)
        mean_anomaly = eccentric_anomaly - eccentricity * solver.sin(eccentric_anomaly)

    time = t_periapsis + mean_anomaly / mean_motion

    outputs = [time]
    if return_m:
        outputs.append(mean_anomaly)
    if return_e:
        outputs.append(eccentric_anomaly)

    if len(outputs) == 1:
        return time
    return tuple(outputs)


def dtheta_dt(
    theta: float,
    *,
    eccentricity: float = 0.0,
    mean_motion: float = 0.001,
    t_periapsis: float = 0.0,
    solver=None,
) -> float:
    """
    Calculate the time derivative of true anomaly (d_theta/dt).

    Args:
        theta: True anomaly (radians).
        eccentricity: Orbital eccentricity.
        mean_motion: Mean motion (rad/s).
        t_periapsis: Time of periapsis passage (seconds).
        solver: Modeling object for symbolic math.
    """
    # Get time and anomalies
    _, mean_anomaly, eccentric_anomaly = true_anomaly_to_time(
        theta,
        eccentricity=eccentricity,
        mean_motion=mean_motion,
        t_periapsis=t_periapsis,
        return_m=True,
        return_e=True,
        solver=solver,
    )

    if eccentricity == 0.0:
        return mean_motion

    if solver is None:
        # dE/dt = n / (1 - e*cos(E))
        de_dt = mean_motion / (1 - eccentricity * np.cos(eccentric_anomaly))

        # dtheta/dt = de_dt * sqrt((1+e)/(1-e)) * (cos(theta/2) / cos(E/2))^2
        # (This is a standard relationship between dtheta and dE)
        theta_dot = (
            de_dt
            * np.sqrt((1 + eccentricity) / (1 - eccentricity))
            * ((np.cos(theta / 2) / np.cos(eccentric_anomaly / 2)) ** 2)
        )
    else:
        de_dt = mean_motion / (1 - eccentricity * solver.cos(eccentric_anomaly))
        # dtheta/dt = de_dt * sqrt((1+e)/(1-e)) * (cos(theta/2) / cos(E/2))^2
        theta_dot = (
            de_dt
            * solver.sqrt((1 + eccentricity) / (1 - eccentricity))
            * ((solver.cos(theta / 2) / solver.cos(eccentric_anomaly / 2)) ** 2)
        )

    return theta_dot


def dt_dtheta(
    theta: float,
    *,
    eccentricity: float = 0.0,
    mean_motion: float = 0.001,
    t_periapsis: float = 0.0,
    solver=None,
) -> float:
    """Calculate the true anomaly derivative of time (dt/d_theta)."""
    return 1.0 / dtheta_dt(
        theta,
        eccentricity=eccentricity,
        mean_motion=mean_motion,
        t_periapsis=t_periapsis,
        solver=solver,
    )


if __name__ == "__main__":
    theta_start = -1.5 * TWO_PI
    theta_end = 1.5 * TWO_PI
    num_samples = 2000
    test_eccentricity = 0.1

    theta_grid = np.linspace(theta_start, theta_end, num_samples)
    time_grid = np.array(
        [true_anomaly_to_time(th, eccentricity=test_eccentricity) for th in theta_grid]
    )
    theta_dot_grid = np.array(
        [dtheta_dt(th, eccentricity=test_eccentricity) for th in theta_grid]
    )
    dt_dtheta_grid = np.array(
        [dt_dtheta(th, eccentricity=test_eccentricity) for th in theta_grid]
    )

    plt.figure(figsize=(10, 8))

    plt.subplot(3, 1, 1)
    plt.plot(theta_grid / TWO_PI, time_grid)
    plt.ylabel("Time (s)")
    plt.title("True Anomaly vs Time")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(theta_grid / TWO_PI, theta_dot_grid)
    plt.ylabel("dTheta/dt (rad/s)")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(theta_grid / TWO_PI, dt_dtheta_grid)
    plt.xlabel("Revolutions")
    plt.ylabel("dt/dTheta (s/rad)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
