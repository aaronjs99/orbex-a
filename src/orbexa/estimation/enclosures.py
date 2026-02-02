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

"""
ORBEX-A Geometric Enclosures.

This module provides tools to calculate minimum enclosing and maximum inscribed
ellipsoids for a set of 3D points using GEKKO optimizations.
"""

import numpy as np
from typing import List, Tuple
from gekko import GEKKO


def min_enclosing_ellipsoid(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Calculate the Minimum Enclosing Ellipsoid (MEE) of a set of points.

    Args:
        points: N x 3 array of points.

    Returns:
        (radii, volume): Tuple of (3,) array of radii and float volume.
    """
    m = GEKKO(remote=False)

    R = [m.Var(value=1.0) for _ in range(3)]
    for dim in range(3):
        m.Equation(R[dim] >= 0.0)

    for point in range(len(points)):
        m.Equation(
            np.sum([(points[point][dim] / R[dim]) ** 2 for dim in range(3)]) <= 1.0
        )

    m.Minimize(4.0 * np.pi * R[0] * R[1] * R[2] / 3.0)

    m.options.IMODE = 3
    m.options.SOLVER = 3
    m.solve(disp=False)

    radii = np.array([R[dim].value[-1] for dim in range(3)])
    volume = m.options.OBJFCNVAL

    m.cleanup()
    return radii, volume


def max_inscribed_ellipsoid(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Calculate the Maximum Inscribed Ellipsoid (MIE) within a set of points.

    Args:
        points: N x 3 array of points defining the boundary?
               Wait, max inscribed usually means inside a polytope defined by points?
               Or is it "points MUST be outside"?
               The equation is sum((X/R)^2) >= 1.0.
               This means all points must be OUTSIDE or ON boundary of ellipsoid.
               So finding largest ellipsoid that excludes all points?
               If points describe obstacles?
    """
    m = GEKKO(remote=False)

    R = [m.Var(value=1.0) for _ in range(3)]
    for dim in range(3):
        m.Equation(R[dim] >= 0.0)

    for point in range(len(points)):
        # Constraint: Point outside ellipsoid
        m.Equation(
            np.sum([(points[point][dim] / R[dim]) ** 2 for dim in range(3)]) >= 1.0
        )

    # Maximize Volume
    m.Maximize(4.0 * np.pi * R[0] * R[1] * R[2] / 3.0)

    m.options.IMODE = 3
    m.options.SOLVER = 3
    m.solve(disp=False)

    radii = np.array([R[dim].value[-1] for dim in range(3)])
    volume = -m.options.OBJFCNVAL

    m.cleanup()
    return radii, volume


# Aliases


if __name__ == "__main__":
    X = np.array(
        [
            [0.0, 0.0, 0.6],
            [0.8, 0.0, 0.6],
            [0.0, 0.8, 0.6],
            [-0.8, 0.0, 0.6],
            [0.0, -0.8, 0.6],
            [0.8, 0.0, 0.0],
            [0.0, 0.8, 0.0],
            [-0.8, 0.0, 0.0],
            [0.0, -0.8, 0.0],
            [0.8, 0.0, -0.6],
            [0.0, 0.8, -0.6],
            [-0.8, 0.0, -0.6],
            [0.0, -0.8, -0.6],
            [0.0, 0.0, -0.6],
        ]
    )

    R1, V1 = min_enclosing_ellipsoid(X)
    R2, V2 = max_inscribed_ellipsoid(X)

    print("Minimum Enclosing Ellipsoid Radii  : ", R1)
    print("Minimum Enclosing Ellipsoid Volume : ", V1)
    print("Maximum Inscribed Ellipsoid Radii  : ", R2)
    print("Maximum Inscribed Ellipsoid Volume : ", V2)
