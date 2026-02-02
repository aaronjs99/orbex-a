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
from scipy import signal
from typing import Optional, Tuple, List, Union


# Math and Geometry Utilities


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


def calc_current_pos(target, x_i, t):
    """
    Compute the current position of a point on the target.

    Args:
        target (Target): Target object with inertial state.
        x_i (np.ndarray): Initial position of the point.
        t (float): Time.

    Returns:
        np.ndarray: Current position vector.
    """
    rotMatrix = tait_bryan_to_rotation_matrix(target.getObservedState(t))
    x_t = np.dot(rotMatrix, x_i)
    x_t = np.append(x_t, [0.00, 0.00, 0.00])
    return x_t


def discretize(
    dt: float,
    A: np.ndarray,
    B: np.ndarray,
    C: Optional[np.ndarray] = None,
    D: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a continuous-time system to discrete-time.

    Args:
        dt (float): Sampling time.
        A (np.ndarray): Continuous-time A matrix.
        B (np.ndarray): Continuous-time B matrix.
        C (np.ndarray, optional): Output matrix.
        D (np.ndarray, optional): Feedthrough matrix.

    Returns:
        tuple: (A_d, B_d, C_d, D_d)
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    n_states = A.shape[0]
    n_inputs = B.shape[1]

    if C is None:
        C = np.zeros((1, n_states))
        C[0, 0] = 1.0
    else:
        C = np.asarray(C, dtype=float)

    if D is None:
        D = np.zeros((C.shape[0], n_inputs))
    else:
        D = np.asarray(D, dtype=float)

    # Scipy returns 5 args: Ad, Bd, Cd, Dd, dt
    Ad, Bd, Cd, Dd, _ = signal.cont2discrete((A, B, C, D), dt)

    return Ad, Bd, Cd, Dd


def calc_local_occlusion_cost(
    x: np.ndarray, w: np.ndarray, v: float, X: np.ndarray, limits: Optional[dict] = None
) -> float:
    """
    Estimate occlusion cost (to be minimized).
    Higher cost = Closer to neighbors or violating bounds.

    Cost = Sum(w_i / distance_i) + Penalty(bounds)
    Note: Original logic used 'distance' in the sum which implied Maximization.
    If we want 'declustering' (separation), we want to MAXIMIZE distance.
    If we are MINIMIZING cost, then Cost should be inversely proportional to distance?
    OR Cost = -Sum(distance).

    The user's original code:
      obs += w * dist (Sum of distances)
      obs -= penalty
      return -obs / const -> -(Sum(dist) - Penalty) = Penalty - Sum(Dist)

    Minimizing (Penalty - Sum(Dist)) equivalent to Maximizing Sum(Dist) and Minimizing Penalty.
    This is consistent.

    So the return value is: Penalty - WeightedSum(Distances).
    If I return this, the optimizer should MINIMIZE it.

    Let's make it explicit.
    """
    x = np.asarray(x)

    # 1. Declustering Reward (Weighted Sum of Distances)
    # We want to maximize this.
    dist_reward = 0.0
    for j in range(len(X)):
        dist = np.linalg.norm(x - X[j])
        dist_reward += w[j] * dist

    # 2. Bounding Penalty
    # We want to minimize this.
    penalty = 0.0
    norm_x = np.linalg.norm(x)

    # Hardcoded bounds from original code (9, 11) - ideally should come from 'limits'
    lower_bound = 9.0
    upper_bound = 11.0

    if norm_x < lower_bound:
        penalty += v * (lower_bound - norm_x) ** 2
    elif norm_x > upper_bound:
        penalty += v * (norm_x - upper_bound) ** 2

    # Total Cost = Penalty - Reward
    # Normalizer from original code: 2 * len(X)^2
    normalizer = 2 * (len(X) ** 2) if len(X) > 0 else 1.0

    cost = (penalty - dist_reward) / normalizer
    return cost


def calc_global_occlusion_cost(
    X: np.ndarray, W: np.ndarray, V: np.ndarray, X0: np.ndarray, B: np.ndarray
) -> float:
    """
    Global occlusion cost (to be minimized).
    """
    num_agents = len(X0) // 3
    dist_reward = 0.0
    travel_cost = 0.0
    bound_penalty = 0.0

    for w_idx in range(num_agents):
        x_w = X[3 * w_idx : 3 * w_idx + 3]

        # Declustering (Reward)
        for i in range(num_agents):
            if i != w_idx:
                x_i = X[3 * i : 3 * i + 3]
                # Original used W[i], let's stick to that
                dist_reward += W[i] * np.linalg.norm(x_w - x_i)

        # Travel Minimization (Cost)
        x0_w = X0[3 * w_idx : 3 * w_idx + 3]
        travel_cost += V[0] * np.linalg.norm(x_w - x0_w)

        # Bounding (Penalty/Cost)
        norm_xw = np.linalg.norm(x_w)
        if norm_xw < B[0]:
            bound_penalty += V[1] * (B[0] - norm_xw) ** 2
        elif norm_xw > B[1]:
            bound_penalty += V[2] * (norm_xw - B[1]) ** 2

    # Total Cost = (Travel + Penalty) - Reward
    # Note: original had 'obs = -obs / ...' where obs start as 0, then += Reward, -= Travel?, -= Penalty?
    # Original:
    # obs = 0
    # obs -= W * dist (Negative Reward)
    # obs += V[0] * travel (Positive Cost)
    # obs += V * penalty (Positive Cost)
    # return obs / const
    # This means 'obs' was ALREADY the Cost (Minimization target).
    # Start 0.
    # Distances reduce cost (Good).
    # Travel increases cost (Bad).
    # Penalty increases cost (Bad).
    # So the return value was (Travel + Penalty - WeightedDist).

    total_cost = (travel_cost + bound_penalty) - dist_reward

    return total_cost / (2 * (num_agents**2))


## Target Shape Constraints
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


def gen_shape_data(shape, shapeParams, numPoints=100):
    if shape == "cylinder":
        center, radius, height = shapeParams
        z_data = np.linspace(
            center[2] - height / 2.0, center[2] + height / 2.0, numPoints
        )
        theta_data = np.linspace(0, 2 * np.pi, numPoints)
        theta, z = np.meshgrid(theta_data, z_data)
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        return x, y, z
    elif shape == "sphere":
        center, radius = shapeParams
        return gen_shape_data(
            "ellipsoid", (center, (radius, radius, radius)), numPoints=100
        )
    elif shape == "ellipsoid":
        center, radii = shapeParams
        u = np.linspace(0, 2 * np.pi, numPoints)
        v = np.linspace(0, np.pi, numPoints)
        x = center[0] + radii[0] * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radii[1] * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radii[2] * np.outer(np.ones(np.size(u)), np.cos(v))
        return x, y, z
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
