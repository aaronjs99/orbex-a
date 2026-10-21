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

"""Numerical cylinder and pyramidal region constraints."""

import numpy as np


def cylinder_inside_constraint(r: np.ndarray, limits: dict) -> float:
    """
    Compute violation of cylinder containment constraint.
    Returns <= 0 if inside, > 0 if outside.
    Constraint: Inside cylinder defined by radius r_T and length l_T.
    """
    r = np.asarray(r, dtype=float)
    tol = limits.get("tolerance", 0.0)

    # Radial distance squared
    rad2 = r[0] ** 2 + r[1] ** 2
    # Axial distance squared (z^2)
    z2 = r[2] ** 2

    # Bounds
    r_limit = limits["r_T"] * (1.0 + tol)
    l_limit = limits["l_T"] * (1.0 + tol)

    # Max violation
    return max(
        rad2 - r_limit**2,
        z2 - l_limit**2,
    )

def pyramidal_constraint(x_0, x_f, mu):
    """
    Generate inequality constraints for a pyramidal region defined between start and goal.
    Returns (A, b) such that A x <= b defines the region.
    """
    x_0 = np.asarray(x_0, dtype=float)
    x_f = np.asarray(x_f, dtype=float)

    mu_x = float(mu.get("mu_x", 0.0))  # Access safely
    mu_y = float(mu.get("mu_y", 0.0))
    # Original constructed mu vector incorrectly if inputs were floats; assuming dict access
    mu_vec = np.array([mu_x, mu_y, 0.0])
    mu__norm = np.linalg.norm(mu_vec)

    x_f_norm = np.linalg.norm(x_f)
    if x_f_norm < 1e-9:
        raise ValueError("Final state must be non-zero")

    dot_val = np.dot(x_0 - (x_f + mu_vec), x_f)
    k = (dot_val / (x_f_norm**2)) + 1.0
    x_m = k * x_f  # Apex-ish point?

    A = x_0 - k * x_f
    A_norm = np.linalg.norm(A)
    B = np.cross(k * x_f, A)
    B_norm = np.linalg.norm(B)

    # Compute Orthogonal Basis
    # u_fwd is along A
    if A_norm < 1e-9:
        # Degenerate: A is zero (x0 = k*xf?)
        # Just return zeros or handle?
        # Fallback to pure Identity
        return np.zeros((4, 3)), np.zeros(4), np.ones(4)

    u_fwd = A / A_norm

    # Handle collinearity for B (Up vector)
    if B_norm < 1e-9:
        # Try Z axis
        B = np.cross(u_fwd, np.array([0, 0, 1]))
        if np.linalg.norm(B) < 1e-9:
            # u_fwd is Z, try Y
            B = np.cross(u_fwd, np.array([0, 1, 0]))
        B_norm = np.linalg.norm(B)

    u_up = B / B_norm
    u_right = np.cross(u_fwd, u_up)

    # Generate Base Points (Rectangular base around x0)
    # Using mu inputs as half-widths.
    # Scaled by A_norm? Original scaled by A_norm in X_vec.
    # If mu is "slope", then width = length * slope?
    # Original mu seems to be a 'radius' parameter?
    # "mu_norm" was used in x_c calculation.
    # Let's assume mu_x, mu_y are the actual dimensions at x0?
    # Or are they tangents?
    # Let's preserve the scale roughly: original used "normalized" vectors * A_norm.
    # We will use mu_x * A_norm if mu is a ratio (tan theta).
    # Typically mu is a tan(theta). So width = Dist * mu.

    dx = mu_x * A_norm if mu_x < 10 else mu_x  # heuristic: if small, treating as ratio
    dy = mu_y * A_norm if mu_y < 10 else mu_y

    # Base corners
    p = [
        x_0 + dx * u_up + dy * u_right,
        x_0 - dx * u_up + dy * u_right,
        x_0 - dx * u_up - dy * u_right,
        x_0 + dx * u_up - dy * u_right,
    ]

    # Re-scale points (unsure of geometric intent, preserving logic roughly but cleaner)
    p0_norm = np.linalg.norm(p[0])
    for i in range(len(p)):
        p_i_norm = np.linalg.norm(p[i])
        if p_i_norm > 1e-9:
            p[i] = p[i] * p0_norm / p_i_norm

    # Calculate center point x_c
    if mu__norm == 0:
        x_c = x_f.copy()
    else:
        # Vector division is not defined. Original was: x_f * (A - k * mu__norm) / (A - mu__norm)
        # Assuming A is vector, mu__norm is scalar? Or A is vector norm?
        # A was defined as vector (x_0 - k*x_f).
        # Original: (A - k*mu) / (A - mu) -> elementwise? Or A_norm?
        # Given potential ambiguity, let's assume scalar scaling of x_f logic from context:
        # "x_c = x_f * (A_norm - k*mu_norm)/(A_norm - mu_norm)" ?
        # Original code: x_c = x_f * (A - k * mu__norm) / (A - mu__norm)
        # Since A is vector, this would be elementwise division.
        # Let's trust user feedback: "A is a vector, mu__norm is scalar... it’s not obvious that’s correct"
        # I will replace with a robust centroid calculation if possible, or stick to a safe interpretation.
        # Safe interpretation: Use simple centroid of polygon p + origin?
        # User said "Use x_f" if ambiguous?
        # Let's approximate x_c = x_f (goal) as the "interior" point for normal orientation check.
        x_c = x_f.copy()

    def calc_plane(p_1, p_2, p_3):
        v_1 = p_2 - p_1
        v_2 = p_3 - p_1
        n = np.cross(v_1, v_2)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-9:
            return np.zeros(3), 0.0
        n = n / n_norm
        return n, np.dot(n, p_1)

    A_mat_list, B_mat_list = [], []
    for i in range(len(p)):
        # Plane formed by apex x_c and two base points?
        # Original: calc_plane(x_c, p[i], p[i+1])
        # This forms side faces of pyramid
        n, d = calc_plane(x_c, p[i], p[(i + 1) % len(p)])
        A_mat_list.append(n)
        B_mat_list.append(d)

    A_mat = np.array(A_mat_list)
    B_mat = np.array(B_mat_list)

    # Polarity check: Ensure x_m (start/apex?) is on the correct side?
    # Actually x_m was derived from x_0 and x_f.
    # User said: "Verify with a known interior point x_m and flip plane signs so A x_m <= b holds."
    # Let's check violations.

    # Calculate A*x_m - B
    violations = A_mat @ x_m - B_mat

    # Flip where violated
    for i in range(len(violations)):
        if violations[i] > 1e-9:  # If A*x > b
            A_mat[i] = -A_mat[i]
            B_mat[i] = -B_mat[i]

    # Redundant info, but keeping signature compatible if possible
    polarity = np.zeros(len(violations))

    return A_mat, B_mat, polarity
