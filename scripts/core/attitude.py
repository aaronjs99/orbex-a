"""Standalone rigid-body attitude dynamics, separate from prescribed mission motion.

Orientations use the existing body-to-inertial rotation ``Rx(roll) Ry(pitch) Rz(yaw)``.
Body angular velocity and torque are in rad/s and body-frame units respectively.
Quaternions are scalar-last ``[x, y, z, w]`` and remain the propagation state;
reported Euler angles are a canonical chart and may jump at chart boundaries.
"""

from dataclasses import dataclass
from typing import Callable
import warnings

import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation


def finite_vector(value, name: str, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {size} finite values")
    return array.copy()


def inertia_matrix(value) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (3, 3) or not np.all(np.isfinite(matrix))
            or not np.allclose(matrix, matrix.T, rtol=0, atol=1e-12)
            or np.min(np.linalg.eigvalsh(matrix)) <= 0):
        raise ValueError("Inertia must be finite, symmetric and positive definite")
    return matrix.copy()


def euler_to_quaternion(angles) -> np.ndarray:
    """Convert roll, pitch, yaw in radians to a body-to-inertial quaternion."""
    return Rotation.from_euler("XYZ", finite_vector(angles, "Euler angles", 3)).as_quat()


def quaternion_to_euler(quaternion) -> np.ndarray:
    """Report a canonical Euler chart without using that chart for integration."""
    quaternion = finite_vector(quaternion, "Quaternion", 4)
    if np.linalg.norm(quaternion) == 0:
        raise ValueError("Quaternion must have nonzero norm")
    with warnings.catch_warnings():
        # At gimbal lock SciPy chooses zero yaw; the quaternion stays authoritative.
        warnings.filterwarnings("ignore", message="Gimbal lock detected.*", category=UserWarning)
        return Rotation.from_quat(quaternion).as_euler("XYZ")


def euler_rates_to_body_rates(angles, euler_rates) -> np.ndarray:
    """Map roll/pitch/yaw derivatives per second to body rates in rad/s."""
    _, pitch, yaw = finite_vector(angles, "Euler angles", 3)
    roll_rate, pitch_rate, yaw_rate = finite_vector(euler_rates, "Euler rates", 3)
    return np.array([
        np.cos(pitch) * np.cos(yaw) * roll_rate + np.sin(yaw) * pitch_rate,
        -np.cos(pitch) * np.sin(yaw) * roll_rate + np.cos(yaw) * pitch_rate,
        np.sin(pitch) * roll_rate + yaw_rate,
    ])


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
            raise ValueError("Symbolic angles and body rates must each have three components")
        _, pitch, yaw = angles
        p, q, r = body_rates
        backend = solver
    cosine_pitch = backend.cos(pitch)
    if solver is None and abs(cosine_pitch) <= 1e-8:
        raise ValueError("Euler-rate chart is singular at pitch +/- pi/2; propagate quaternions")
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


def evaluate_torque(t, state, dt, inertia, torque_value, torque_type) -> np.ndarray:
    """Resolve the legacy Target torque interface into a body-frame torque."""
    if torque_type == "zero":
        return np.zeros(3)
    if torque_type == "given":
        values = np.asarray(torque_value, dtype=float)
        if values.ndim != 2 or values.shape[1] != 3 or len(values) == 0 or not np.all(np.isfinite(values)):
            raise ValueError("Given torque must be a nonempty sequence of finite three-vectors")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Torque sampling dt must be finite and positive")
        index = min(max(int(np.floor(t / dt)), 0), len(values) - 1)
        return values[index].copy()
    if torque_type == "function":
        if not callable(torque_value):
            raise ValueError("Function torque requires a callable accepting (attitude_state, inertia)")
        result = torque_value(state.copy(), inertia.copy())
        if not isinstance(result, dict) or "torque" not in result:
            raise ValueError("Torque callable must return a dictionary with a 'torque' three-vector")
        return finite_vector(result["torque"], "Torque", 3)
    raise ValueError("torqueType must be 'zero', 'given' or 'function'")


@dataclass(frozen=True)
class AttitudeTrajectory:
    """Endpoint samples from quaternion integration, excluding the initial state."""

    times: np.ndarray
    quaternions: np.ndarray
    body_rates: np.ndarray

    @property
    def euler_angles(self) -> np.ndarray:
        return np.array([quaternion_to_euler(value) for value in self.quaternions])


class RigidBodyAttitude:
    """Numerical attitude propagator with an explicit inertia and tolerance policy."""

    def __init__(self, inertia, *, rtol: float = 1e-10, atol: float = 1e-12):
        self.inertia = inertia_matrix(inertia)
        if any(not np.isfinite(value) or value <= 0 for value in (rtol, atol)):
            raise ValueError("Integration tolerances must be finite and positive")
        self.rtol, self.atol = float(rtol), float(atol)

    def propagate(self, quaternion, body_rates, times, *, torque: Callable | None = None) -> AttitudeTrajectory:
        """Integrate from local time zero to positive, strictly increasing endpoints.

        The optional torque callable accepts ``(time_seconds, quaternion, body_rate)``.
        Integration failure raises; no frozen-state replacement is manufactured.
        """
        quaternion = finite_vector(quaternion, "Quaternion", 4)
        norm = np.linalg.norm(quaternion)
        if norm == 0:
            raise ValueError("Quaternion must have nonzero norm")
        quaternion /= norm
        body_rates = finite_vector(body_rates, "Body angular velocity", 3)
        times = np.asarray(times, dtype=float)
        if (times.ndim != 1 or len(times) == 0 or not np.all(np.isfinite(times))
                or times[0] <= 0 or np.any(np.diff(times) <= 0)):
            raise ValueError("Attitude endpoints must be finite, positive and strictly increasing")

        def rhs(t, state):
            vector, scalar, omega = state[:3], state[3], state[4:]
            quaternion_dot = 0.5 * np.concatenate((scalar * omega + np.cross(vector, omega), [-np.dot(vector, omega)]))
            applied = np.zeros(3) if torque is None else finite_vector(torque(t, state[:4].copy(), omega.copy()), "Torque", 3)
            return np.concatenate((quaternion_dot, body_rate_derivative(omega, self.inertia, applied)))

        solution = solve_ivp(rhs, (0.0, float(times[-1])), np.concatenate((quaternion, body_rates)),
                             t_eval=times, method="DOP853", rtol=self.rtol, atol=self.atol)
        if not solution.success or solution.y.shape[1] != len(times) or not np.all(np.isfinite(solution.y)):
            raise RuntimeError(f"Rigid-body attitude integration failed: {solution.message}")
        quaternions = solution.y[:4].T.copy()
        quaternions /= np.linalg.norm(quaternions, axis=1)[:, None]
        return AttitudeTrajectory(times.copy(), quaternions, solution.y[4:].T.copy())
