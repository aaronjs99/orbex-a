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

"""Geometric sampling using the established global NumPy random stream."""

import numpy as np


def random_unit_vectors(n: int) -> np.ndarray:
    """Generate n random unit vectors uniformly distributed on the sphere."""
    v = np.random.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v

def gen_init_state(num_chasers: int, rX: float, rV: float) -> np.ndarray:
    """
    Generate random initial positions and velocities for chasers.

    Args:
        num_chasers (int): Number of chasers.
        rX (float): Maximum initial position radius.
        rV (float): Maximum initial velocity magnitude.

    Returns:
        np.ndarray: Initial state vector of shape (num_chasers * 6,)
    """
    pos_dir = random_unit_vectors(num_chasers)
    vel_dir = random_unit_vectors(num_chasers)

    pos = rX * pos_dir
    vel = rV * vel_dir

    # Interleave pos and vel: [x1, y1, z1, vx1, vy1, vz1, x2, ...]
    # The original code interleaved them: x_0[chaser * 6 + 0:3] = pos, + 3:6 = vel
    x0 = np.hstack([pos, vel]).reshape((num_chasers, 6))
    return x0.flatten()

def gen_shape_data(shape, shapeParams, numPoints=100):
    if shape == "cylinder":
        center, radius, height = shapeParams
        z_data = np.linspace(
            center[2] - height / 2.0, center[2] + height / 2.0, numPoints
        )
        theta_data = np.linspace(0, 2 * np.pi, numPoints)
        theta, z = np.meshgrid(theta_data, z_data)
        x_pos = center[0] + radius * np.cos(theta)
        y_pos = center[1] + radius * np.sin(theta)
        return x_pos, y_pos, z
    elif shape == "sphere":
        center, radius = shapeParams
        return gen_shape_data(
            "ellipsoid", (center, (radius, radius, radius)), numPoints=100
        )
    elif shape == "ellipsoid":
        center, radii = shapeParams
        theta_vals = np.linspace(0, 2 * np.pi, numPoints)
        phi_vals = np.linspace(0, np.pi, numPoints)
        x_pos = center[0] + radii[0] * np.outer(np.cos(theta_vals), np.sin(phi_vals))
        y_pos = center[1] + radii[1] * np.outer(np.sin(theta_vals), np.sin(phi_vals))
        z_pos = center[2] + radii[2] * np.outer(
            np.ones(np.size(theta_vals)), np.cos(phi_vals)
        )
        return x_pos, y_pos, z_pos
    else:
        raise ValueError("Shape not recognized")

def gen_fib_lattice(sphere_params, num_points, **kwargs):
    sphere_center, sphere_radius = sphere_params
    if "theta_0" not in kwargs.keys() and "phi_0" not in kwargs.keys():
        theta_0, phi_0 = np.random.uniform(0, 2 * np.pi), np.random.uniform(0, np.pi)
    else:
        theta_0, phi_0 = kwargs["theta_0"], kwargs["phi_0"]
    # Generate Fibonacci Lattice
    fib_lattice = np.zeros((num_points, 3))
    goldenRatio = (1 + np.sqrt(5)) / 2

    for i in range(num_points):
        theta = 2 * np.pi * i / goldenRatio + theta_0
        phi = np.arccos(1 - (2 * i + 1) / num_points) + phi_0
        fib_lattice[i, 0] = sphere_center[0] + sphere_radius * np.cos(theta) * np.sin(
            phi
        )
        fib_lattice[i, 1] = sphere_center[1] + sphere_radius * np.sin(theta) * np.sin(
            phi
        )
        fib_lattice[i, 2] = sphere_center[2] + sphere_radius * np.cos(phi)
    return fib_lattice
