"""attitude.py: Standalone rigid-body attitude dynamics, separate from prescribed mission motion.

Orientations use the existing body-to-inertial rotation ``Rx(roll) Ry(pitch) Rz(yaw)``.
Body angular velocity and torque are in rad/s and body-frame units respectively.
Quaternions are scalar-last ``[x, y, z, w]`` and remain the propagation state;
reported Euler angles are a canonical chart and may jump at chart boundaries.

Copyright: Aaron John Sabu
"""

import warnings

import numpy as np
from scipy.spatial.transform import Rotation


def finite_vector(value, name: str, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {size} finite values")
    return array.copy()


def inertia_matrix(value) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if (
        matrix.shape != (3, 3)
        or not np.all(np.isfinite(matrix))
        or not np.allclose(matrix, matrix.T, rtol=0, atol=1e-12)
        or np.min(np.linalg.eigvalsh(matrix)) <= 0
    ):
        raise ValueError("Inertia must be finite, symmetric and positive definite")
    return matrix.copy()


def euler_to_quaternion(angles) -> np.ndarray:
    """Convert roll, pitch, yaw in radians to a body-to-inertial quaternion."""
    return Rotation.from_euler(
        "XYZ", finite_vector(angles, "Euler angles", 3)
    ).as_quat()


def quaternion_to_euler(quaternion) -> np.ndarray:
    """Report a canonical Euler chart without using that chart for integration."""
    quaternion = finite_vector(quaternion, "Quaternion", 4)
    if np.linalg.norm(quaternion) == 0:
        raise ValueError("Quaternion must have nonzero norm")
    with warnings.catch_warnings():
        # At gimbal lock SciPy chooses zero yaw; the quaternion stays authoritative.
        warnings.filterwarnings(
            "ignore", message="Gimbal lock detected.*", category=UserWarning
        )
        return Rotation.from_quat(quaternion).as_euler("XYZ")


def body_rates_to_euler_rates(angles, body_rates, *, solver=None):
    """Map body rates to Euler derivatives in the shared Rx Ry Rz convention.

    Numeric calls reject a singular chart. Symbolic callers must constrain pitch
    to one nonsingular chart before using this mapping in their formulation.
    """
    if solver is None:
        _, pitch, yaw = finite_vector(angles, "Euler angles", 3)
        p, q, r = finite_vector(body_rates, "Body angular velocity", 3)
        backend = np
    else:
        if len(angles) != 3 or len(body_rates) != 3:
            raise ValueError(
                "Symbolic angles and body rates must each have three components"
            )
        _, pitch, yaw = angles
        p, q, r = body_rates
        backend = solver
    cosine_pitch = backend.cos(pitch)
    if solver is None and abs(cosine_pitch) <= 1e-8:
        raise ValueError(
            "Euler-rate chart is singular at pitch +/- pi/2; propagate quaternions"
        )
    roll_component = backend.cos(yaw) * p - backend.sin(yaw) * q
    result = [
        roll_component / cosine_pitch,
        backend.sin(yaw) * p + backend.cos(yaw) * q,
        r - backend.sin(pitch) / cosine_pitch * roll_component,
    ]
    return np.asarray(result) if solver is None else result


def body_rate_derivative(body_rates, inertia, torque) -> np.ndarray:
    """Euler's rigid-body equation, using the current body rate at each call."""
    return np.linalg.solve(inertia, torque - np.cross(body_rates, inertia @ body_rates))
