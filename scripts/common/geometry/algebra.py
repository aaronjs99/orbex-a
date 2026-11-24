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

"""algebra.py: Array geometry, attitude algebra and observed target coordinates.

Copyright: Aaron John Sabu
"""

import numpy as np


def calc_distance(p1, p2) -> float:
    """
    Compute Euclidean distance between two points.

    Args:
        p1, p2 (np.ndarray): Points in 3D space.

    Returns:
        float: Distance.
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    return float(np.linalg.norm(p1 - p2))


def gen_skew_sym_mat(val: np.ndarray) -> np.ndarray:
    """
    Generate a 3x3 skew-symmetric matrix from a 3-element vector.

    Args:
        val (list or np.ndarray): 3-element vector.

    Returns:
        np.ndarray: 3x3 skew-symmetric matrix.
    """
    v = np.asarray(val, dtype=float).reshape(3)
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def tait_bryan_to_rotation_matrix(angles: np.ndarray) -> np.ndarray:
    """
    Compute a rotation matrix from Tait-Bryan angles.

    Args:
        angles (list or np.ndarray): [alpha, beta, gamma] angles in radians.

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    ### Extract individual angles ###
    alpha, beta, gamma = angles

    ### Compute sine and cosine values ###
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    cb = np.cos(beta)
    sb = np.sin(beta)
    cg = np.cos(gamma)
    sg = np.sin(gamma)

    ### Compute the rotation matrix ###
    return np.array(
        [
            [cb * cg, -cb * sg, sb],
            [ca * sg + sa * sb * cg, ca * cg - sa * sb * sg, -sa * cb],
            [sa * sg - ca * sb * cg, sa * cg + ca * sb * sg, ca * cb],
        ]
    )
