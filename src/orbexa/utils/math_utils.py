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

import scipy
import numpy as np
import scipy.signal
# import matplotlib.pyplot as plt

from orbexa.core.params import *

np.random.seed(0)

# Math and Geometry Utilities

def calcDistance(p1, p2):
    """
    Compute Euclidean distance between two points.

    Args:
        p1, p2 (np.ndarray): Points in 3D space.

    Returns:
        float: Distance.
    """
    return np.linalg.norm(np.add(p1, -p2))


def genInitState(numChasers, rX, rV, *args, **kwargs):
    """
    Generate random initial positions and velocities for chasers.

    Args:
        numChasers (int): Number of chasers.
        rX (float): Maximum initial position radius.
        rV (float): Maximum initial velocity magnitude.

    Returns:
        np.ndarray: Initial state vector of shape (numChasers * 6,)
    """
    x_0 = np.zeros((numChasers * 6))
    for chaser in range(numChasers):
        phi, theta = np.random.uniform(0, 2 * np.pi), np.random.uniform(0, np.pi)
        x_0[chaser * 6 + 0] = rX * np.cos(phi) * np.sin(theta)
        x_0[chaser * 6 + 1] = rX * np.sin(phi) * np.sin(theta)
        x_0[chaser * 6 + 2] = rX * np.cos(theta)
        x_0[chaser * 6 + 3] = rV * np.cos(phi) * np.sin(theta)
        x_0[chaser * 6 + 4] = rV * np.sin(phi) * np.sin(theta)
        x_0[chaser * 6 + 5] = rV * np.cos(theta)
    return x_0


def genSkewSymMat(val):
    """
    Generate a 3x3 skew-symmetric matrix from a 3-element vector.

    Args:
        val (list or np.ndarray): 3-element vector.

    Returns:
        np.ndarray: 3x3 skew-symmetric matrix.
    """
    try:
        return np.array(
            [
                [0.0, -val[2], val[1]],
                [val[2], 0.0, -val[0]],
                [-val[1], val[0], 0.0],
            ]
        )
    except:
        return [
            [0.0, -val[2], val[1]],
            [val[2], 0.0, -val[0]],
            [-val[1], val[0], 0.0],
        ]


def tait_bryan_to_rotation_matrix(angles, *args, **kwargs):
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
    try:
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        cb = np.cos(beta)
        sb = np.sin(beta)
        cg = np.cos(gamma)
        sg = np.sin(gamma)
    except:
        m = kwargs["m"]
        ca = m.cos(alpha)
        sa = m.sin(alpha)
        cb = m.cos(beta)
        sb = m.sin(beta)
        cg = m.cos(gamma)
        sg = m.sin(gamma)

    ### Compute the rotation matrix ###
    rotation_matrix = [
        [cb * cg, -cb * sg, sb],
        [ca * sg + sa * sb * cg, ca * cg - sa * sb * sg, -sa * cb],
        [sa * sg - ca * sb * cg, sa * cg + ca * sb * sg, ca * cb],
    ]
    try:
        return np.array(rotation_matrix)
    except:
        return rotation_matrix


def calcCurrentPos(target, x_i, t):
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


def discretize(dt, A, B, *args):
    """
    Convert a continuous-time system to discrete-time.

    Args:
        dt (float): Sampling time.
        A (np.ndarray): Continuous-time A matrix.
        B (np.ndarray): Continuous-time B matrix.

    Returns:
        tuple: Discretized (A_d, B_d) or (A_d, B_d, C_d, D_d)
    """
    if len(args) == 0:
        try:
            C = np.zeros((1, A.shape[0]))
        except:
            A_val = A(0, 0)
            C = np.zeros((1, A_val.shape[0]))
        C[0][0] = 1
        D = np.zeros((B.shape[1], B.shape[1]))
    else:
        C = args[0]
        D = args[1]

    sys = scipy.signal.cont2discrete((A, B, C, D), dt)

    A_d = sys[0]
    B_d = sys[1]
    C_d = sys[2]
    D_d = sys[3]

    if len(args) == 0:
        return A_d, B_d
    else:
        return A_d, B_d, C_d, D_d


def calcLocalOcclusion(x, w, v, X):
    """
    Estimate occlusion cost based on distance to neighbors.

    Args:
        x (np.ndarray): Ego agent state.
        w (list): Neighbor weights.
        v (float): Bounding penalty.
        X (list of np.ndarray): Neighbor states.

    Returns:
        float: Occlusion cost.
    """
    obs = 0
    # Declustering
    for j in range(len(X)):
        obs += w[j] * np.linalg.norm(np.subtract(x, X[j]))
    # Bounding
    if np.linalg.norm(x) < 9:
        obs -= v * (9 - np.linalg.norm(x)) ** 2
    elif np.linalg.norm(x) > 11:
        obs -= v * (np.linalg.norm(x) - 11) ** 2
    # Normalization
    obs = -obs / (2 * (len(X) ** 2))
    return obs


def calcGlobalOcclusion(X, W, V, X0, B):
    """
    Global occlusion cost considering all agents and penalties.

    Args:
        X (np.ndarray): Current states.
        W (list): Weight vector.
        V (list): Bounding penalty coefficients.
        X0 (np.ndarray): Initial states.
        B (list): Bounding distance limits.

    Returns:
        float: Total occlusion cost.
    """
    obs = 0
    numAgents = int(len(X0) / 3)
    for w in range(1, numAgents + 1):
        x_w = X[3 * w - 3 : 3 * w]
        # Declustering
        for i in range(1, numAgents + 1):
            if i != w:
                x_i = X[3 * i - 3 : 3 * i]
                obs -= W[i - 1] * np.linalg.norm(np.subtract(x_w, x_i))
        # Travel Minimization
        x0_w = X0[3 * w - 3 : 3 * w]
        obs += V[0] * np.linalg.norm(np.subtract(x_w, x0_w))
        # Bounding
        if np.linalg.norm(x_w) < B[0]:
            obs += V[1] * (B[0] - np.linalg.norm(x_w)) ** 2
        elif np.linalg.norm(x_w) > B[1]:
            obs += V[2] * (np.linalg.norm(x_w) - B[1]) ** 2
    # Normalization
    obs = obs / (2 * (numAgents**2))
    return obs


## Target Shape Constraints
def cylinderRadialUpperConstraint(r):
    """
    Constraint for upper radial boundary of a cylinder.

    Args:
        r (np.ndarray): Point in space.

    Returns:
        float: Constraint value.
    """
    return (
        np.sum([r[j] ** 2 for j in range(0, 2)])
        - (targetLimit["r_T"] * (1.00 + 0.001)) ** 2
    )


def cylinderRadialLowerConstraint(r):
    """
    Constraint for lower radial boundary of a cylinder.

    Args:
        r (np.ndarray): Point in space.

    Returns:
        float: Constraint value.
    """
    return (
        -np.sum([r[j] ** 2 for j in range(0, 2)])
        + (targetLimit["r_T"] * (1.00 - 0.001)) ** 2
    )


def cylinderAxialUpperConstraint(r):
    """
    Constraint for upper axial boundary of a cylinder.

    Args:
        r (np.ndarray): Point in space.

    Returns:
        float: Constraint value.
    """
    return np.sum([r[j] ** 2 for j in range(2, 3)]) - targetLimit["l_T"] ** 2


def cylinderAxialLowerConstraint(r):
    """
    Constraint for lower axial boundary of a cylinder.

    Args:
        r (np.ndarray): Point in space.

    Returns:
        float: Constraint value.
    """
    return -np.sum([r[j] ** 2 for j in range(2, 3)]) - targetLimit["l_T"] ** 2


def genShapeData(shape, shapeParams, numPoints=100):
    """
    Generate 3D mesh grid data for a given shape.

    Args:
        shape (str): Shape type ("cylinder", "sphere", "ellipsoid").
        shapeParams (tuple): Parameters defining the shape.
        numPoints (int): Number of discretization points.

    Returns:
        tuple: 3D arrays (x, y, z) for plotting.
    """
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
        return genShapeData(
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


def genFibLattice(sphereParams, numPoints, **kwargs):
    """
    Generate a uniformly distributed set of points over a sphere using Fibonacci lattice.

    Args:
        sphereParams (tuple): (center, radius) of the sphere.
        numPoints (int): Number of points to generate.
        kwargs (dict): Optional keys: 'theta_0', 'phi_0'.

    Returns:
        np.ndarray: Array of shape (numPoints, 3) representing points on the sphere.
    """
    sphere_center, sphere_radius = sphereParams
    if "theta_0" not in kwargs.keys() and "phi_0" not in kwargs.keys():
        theta_0, phi_0 = np.random.uniform(0, 2 * np.pi), np.random.uniform(0, np.pi)
    else:
        theta_0, phi_0 = kwargs["theta_0"], kwargs["phi_0"]
    # Generate Fibonacci Lattice
    fib_lattice = np.zeros((numPoints, 3))
    goldenRatio = (1 + np.sqrt(5)) / 2

    for i in range(numPoints):
        theta = 2 * np.pi * i / goldenRatio + theta_0
        phi = np.arccos(1 - (2 * i + 1) / numPoints) + phi_0
        fib_lattice[i, 0] = sphere_center[0] + sphere_radius * np.cos(theta) * np.sin(
            phi
        )
        fib_lattice[i, 1] = sphere_center[1] + sphere_radius * np.sin(theta) * np.sin(
            phi
        )
        fib_lattice[i, 2] = sphere_center[2] + sphere_radius * np.cos(phi)
    return fib_lattice


def pyramidalConstraint(x_0, x_f, mu):
    """
    Generate inequality constraints for a pyramidal region defined between start and goal.

    Args:
        x_0 (array-like): Initial position vector.
        x_f (array-like): Final position vector.
        mu (dict): Dictionary with 'mu_x' and 'mu_y' offset parameters.

    Returns:
        tuple: (A matrix, B vector, polarity array) defining half-space inequalities.
    """
    x_0 = np.array(x_0)
    x_f = np.array(x_f)
    mu_x, mu_y = mu["mu_x"], mu["mu_y"]
    mu = np.array([mu_x, mu_y, 0.0])
    mu__norm = np.linalg.norm(mu)
    x_0_norm = np.linalg.norm(x_0)
    x_f_norm = np.linalg.norm(
        x_f
    )  # should be 1.0 if the target location is on the unit sphere

    if x_f_norm == 0:
        raise ValueError("Final state must be non-zero")

    k = (np.dot(x_0 - (x_f + mu), x_f) / x_f_norm**2) + 1
    x_m = k * x_f

    A = x_0 - k * x_f
    B = np.cross(k * x_f, A)
    A_norm = np.linalg.norm(A)
    B_norm = np.linalg.norm(B)
    X_vec = (-A / A_norm + B / B_norm) * A_norm
    Y_vec = (-A / A_norm - B / B_norm) * A_norm

    p = [x_0, x_0 + X_vec, x_0 + X_vec + Y_vec, x_0 + Y_vec]

    if mu__norm == 0:
        x_c = x_f.copy()
    else:
        x_c = x_f * (A - k * mu__norm) / (A - mu__norm)

    for i, p_i in enumerate(p):
        if not (all(p_i == 0.0)):
            p[i] = p_i * np.linalg.norm(p[0]) / np.linalg.norm(p_i)

    def calcPlane(p_1, p_2, p_3):
        v_1 = p_2 - p_1
        v_2 = p_3 - p_1
        n = np.cross(v_1, v_2)
        n = n / np.linalg.norm(n)
        return np.array([n[0], n[1], n[2]]), np.dot(n, p_1)

    A_mat, B_mat = [], []
    for i in range(len(p)):
        A_i, B_i = calcPlane(x_c, p[i], p[(i + 1) % len(p)])
        A_mat.append(A_i)
        B_mat.append(B_i)
    A_mat = np.vstack(A_mat)
    B_mat = np.array(B_mat)

    polarity = np.sign(np.dot(A_mat, x_m) - B_mat)
    for i, pol_i in enumerate(polarity):
        if pol_i == -1:
            pol_i = 1
            A_mat[i, :] = -A_mat[i, :]
            B_mat[i] = -B_mat[i]
    return A_mat, B_mat, polarity


def trajopt_target(timeParams, orbitParams, solverParams, *args, **kwargs):
    """
    Solve for the time-varying orientation of a target using trajectory optimization.

    Args:
        timeParams (dict): Includes 't_s', 'timeSeq', 'numMPCSteps', 'numActSteps'.
        orbitParams (dict): Includes 'eccentricity' for the orbital model.
        solverParams (dict): Contains GEKKO solver settings and target params.

    Returns:
        tuple: (t, q, rotMatrices) — time, anomaly, and rotation matrices over horizon.
    """
    ## Unpack Parameters ##
    t_s = timeParams["t_s"]
    timeSeq = timeParams["timeSeq"]
    numMPCSteps = timeParams["numMPCSteps"]
    numActSteps = timeParams["numActSteps"]
    eccentricity = orbitParams["eccentricity"]

    from gekko import GEKKO
    import orbexa.core.params as p

    ## Initialize MPC ##
    m = GEKKO(remote=solverParams["remote"])
    m.time = timeSeq
    w = np.ones(numMPCSteps)
    final = np.zeros(numMPCSteps)
    final[-1] = 1
    target_thetas = []

    ## Start Time Anomaly ##
    t_s = timeSeq[0]
    ## Final Time Anomaly ##
    t_f = timeSeq[-1]

    ## Initialize Variables ##
    if True:
        t = m.Var(value=0)
        q = m.Var(value=0, fixed_initial=False)
        W = m.Param(value=w)
        final = m.Param(value=final)

    ## Constraint Equations ##
    eqs = []
    ### Time and Anomaly Update ###
    if True:
        eqs.append(t.dt() == 1)
        E = m.Intermediate(
            2 * m.atan(np.sqrt((1 - eccentricity) / (1 + eccentricity)) * m.tan(t / 2))
        )
        M = m.Intermediate(E - eccentricity * m.sin(E))
        eqs.append(q == p.t_p + t_s + M / p.n)
    ### Target Dynamics ###
    targetParams = solverParams["targetParams"]
    target_theta_0 = targetParams["theta_0"]
    target_omega_0 = targetParams["omega_0"]
    momInertia = targetParams["momInertia"]
    target_theta = [
        m.Var(value=target_theta_0[i], fixed_initial=True)
        for i in range(len(target_theta_0))
    ]
    target_omega = [
        m.Var(value=target_omega_0[i], fixed_initial=True)
        for i in range(len(target_omega_0))
    ]
    for i in range(len(target_theta)):
        eqs.append(target_theta[i].dt() == target_omega[i])
        #  (n*np.sqrt((1+eccentricity)/(1-eccentricity))*((m.cos(t/2)/m.cos(E/2))**2)/(1-eccentricity*m.cos(E))))
        eqs.append(
            target_omega[i].dt()
            == (
                np.matmul(
                    np.matmul(np.linalg.inv(momInertia), genSkewSymMat(target_omega)),
                    np.matmul(momInertia, target_omega),
                )[i]
            )
        )
        #  (n*np.sqrt((1+eccentricity)/(1-eccentricity))*((m.cos(t/2)/m.cos(E/2))**2)/(1-eccentricity*m.cos(E))))
    rotMatrix = m.Array(m.Var, (3, 3), fixed_initial=False)
    for i in range(3):
        for j in range(3):
            eqs.append(
                rotMatrix[i][j]
                == tait_bryan_to_rotation_matrix(target_theta, m=m)[i][j]
            )

    eqs = m.Equations(eqs)
    m.options.IMODE = 6
    m.options.REDUCE = 3
    m.options.SOLVER = 3
    m.options.MAX_ITER = 3000
    m.options.MAX_MEMORY = 512

    m.solve(disp=solverParams["disp"], debug=2)

    ## Extract Solution ##
    rotMatrices = [[None for j in range(3)] for i in range(3)]
    if True:
        t = np.array(t.value)[:numActSteps]
        q = np.array(q.value)[:numActSteps]
        for i in range(3):
            for j in range(3):
                rotMatrices[i][j] = np.array(rotMatrix[i][j].value)[:numActSteps]
        rotMatrices = np.transpose(rotMatrices)

    return t, q, rotMatrices
