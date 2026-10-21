"""Shape-independent unactuated quaternion/Euler motion on a declared orbital clock.

Implementations supply inertia, spin, initial quaternion and integration settings.
Collision geometry never enters these equations.
"""
from __future__ import annotations
from typing import Dict, Sequence, Tuple
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation
from orbexa.core.dynamics.attitude import body_rate_derivative, body_rates_to_euler_rates, finite_vector, inertia_matrix, quaternion_to_euler
from orbexa.core.dynamics.anomaly import orbital_radius_terms
from orbexa.utils.geometry.algebra import gen_skew_sym_mat


class RigidBodyMotion:
    """Shared dense physical motion; subclasses provide immutable configuration."""

    def _initialize_motion(self) -> None:
        quaternion = finite_vector(self.initial_quaternion, "Initial quaternion", 4)
        scale = float(np.max(np.abs(quaternion)))
        if scale == 0.0:
            raise ValueError("Initial quaternion must have nonzero norm")
        quaternion /= scale
        quaternion /= np.linalg.norm(quaternion)
        omega = finite_vector(self.initial_body_angular_velocity, "Initial body angular velocity (rad/s)", 3)
        inertia = inertia_matrix(self.inertia)
        principal = np.linalg.eigvalsh(inertia)
        # Scaling the triangle avoids overflow and allows only eigensolver
        # roundoff at the physical planar-body boundary Imax = I1 + I2.
        if 1.0 - np.sum(principal[:2] / principal[-1]) > 64.0 * np.finfo(float).eps:
            raise ValueError("Physical inertia principal moments must satisfy Imax <= I1 + I2")
        if not isinstance(self.gravity_gradient, (bool, np.bool_)):
            raise ValueError("gravity_gradient must be a boolean")
        object.__setattr__(self, "gravity_gradient", bool(self.gravity_gradient))
        domain = np.asarray(self.anomaly_domain, dtype=float)
        if domain.shape != (2,) or not np.all(np.isfinite(domain)) or domain[0] >= domain[1]:
            raise ValueError("anomaly_domain must contain two finite, strictly increasing bounds")
        if not np.isfinite(self.eccentricity) or not 0.0 <= self.eccentricity < 1.0:
            raise ValueError("Target clock eccentricity must belong to [0, 1)")
        if not np.isfinite(self.alpha) or self.alpha < 0.0:
            raise ValueError("Target clock drag alpha must be finite and nonnegative")
        for name in ("mu", "semi_major_axis", "integration_rtol", "integration_atol", "integration_max_step"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in (("initial_quaternion", quaternion),
                            ("initial_body_angular_velocity", omega), ("inertia", inertia)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "anomaly_domain", (float(domain[0]), float(domain[1])))
        def rhs(anom, state):
            rate, _ = self.anomaly_clock(anom)
            vector, scalar, body_rate = state[1:4], state[4], state[5:8]
            quaternion_dot = 0.5 * np.concatenate((
                scalar * body_rate + np.cross(vector, body_rate), [-np.dot(vector, body_rate)]))
            omega_dot = body_rate_derivative(body_rate, inertia, self._body_torque(anom, state[1:5]))
            return np.concatenate(([1.0], quaternion_dot, omega_dot)) / rate

        initial = np.concatenate(([0.0], quaternion, omega))
        solution = solve_ivp(rhs, self.anomaly_domain, initial, method="DOP853", dense_output=True,
                             rtol=self.integration_rtol, atol=self.integration_atol,
                             max_step=self.integration_max_step)
        if not solution.success or solution.sol is None or not np.all(np.isfinite(solution.y)):
            raise RuntimeError(f"Rigid-body target integration failed: {solution.message}")
        object.__setattr__(self, "_solution", solution.sol)


    def _query(self, anom: float) -> float:
        value = np.asarray(anom, dtype=float)
        if (value.ndim != 0 or not np.isfinite(value)
                or not self.anomaly_domain[0] <= value <= self.anomaly_domain[1]):
            raise ValueError(f"Target anomaly must be a finite scalar in {self.anomaly_domain}")
        return float(value)


    def anomaly_clock(self, anom: float) -> Tuple[float, float]:
        """Return ``(dq/dt, d(dq/dt)/dq)`` for the analytical drag orbit."""
        q = self._query(anom)
        radius, radius_prime, h = orbital_radius_terms(
            q, eccentricity=self.eccentricity, alpha=self.alpha,
            mu=self.mu, semi_major_axis=self.semi_major_axis,
        )
        rate = h * np.exp(-self.alpha * q) / radius**2
        rate_prime = rate * (-self.alpha - 2.0 * radius_prime / radius)
        if not np.isfinite(rate) or rate <= 0.0 or not np.isfinite(rate_prime):
            raise ValueError("Target orbital clock must remain finite and strictly increasing")
        return float(rate), float(rate_prime)


    def _state_at(self, anom: float) -> Tuple[float, np.ndarray, np.ndarray]:
        values = self._solution(self._query(anom))
        quaternion, omega = values[1:5].copy(), values[5:8].copy()
        norm = np.linalg.norm(quaternion)
        if not np.all(np.isfinite(values)) or not np.isfinite(norm) or norm == 0.0:
            raise RuntimeError("Target dense attitude trajectory is non-finite or singular")
        return float(values[0]), quaternion / norm, omega


    def physical_time_at(self, anom: float) -> float:
        """Elapsed physical seconds since the initial anomaly."""
        return self._state_at(anom)[0]


    def inertial_quaternion_at(self, anom: float) -> np.ndarray:
        """Continuous scalar-last body-to-initial-inertial quaternion."""
        return self._state_at(anom)[1]


    def quaternion_at(self, anom: float) -> np.ndarray:
        """Continuous scalar-last body-to-LVLH quaternion, without sign folding."""
        q = self._query(anom)
        inertial = self._state_at(q)[1]
        half = -0.5 * (q - self.anomaly_domain[0])
        vector, scalar = np.array([0.0, 0.0, np.sin(half)]), np.cos(half)
        result = np.concatenate((scalar * inertial[:3] + inertial[3] * vector
                                 + np.cross(vector, inertial[:3]),
                                 [scalar * inertial[3] - np.dot(vector, inertial[:3])]))
        return result / np.linalg.norm(result)


    def body_angular_velocity_at(self, anom: float) -> np.ndarray:
        """Body components of angular velocity relative to inertial space, rad/s."""
        return self._state_at(anom)[2]


    @property
    def attitude_model(self) -> str:
        return "gravity_gradient_rigid_body" if self.gravity_gradient else "torque_free_rigid_body"


    def orbital_position_inertial_at(self, anom: float) -> np.ndarray:
        """Central-body-to-target position in the fixed initial inertial basis, m.

        At the initial anomaly the radial axis is positive LVLH y. This is an
        inertial orbital-plane basis; spherical central-field torque does not
        require an arbitrary absolute node/inclination rotation to GCRF axes.
        """
        q = self._query(anom)
        radius, _, _ = orbital_radius_terms(q, eccentricity=self.eccentricity,
            alpha=self.alpha, mu=self.mu, semi_major_axis=self.semi_major_axis)
        angle = q - self.anomaly_domain[0]
        return radius * np.array([-np.sin(angle), np.cos(angle), 0.0])


    def _body_torque(self, anom: float, inertial_quaternion: np.ndarray) -> np.ndarray:
        if not self.gravity_gradient:
            return np.zeros(3)
        position = self.orbital_position_inertial_at(anom)
        radius = np.linalg.norm(position)
        radial_body = Rotation.from_quat(inertial_quaternion).inv().apply(position / radius)
        return (3.0 * self.mu / radius**3) * np.cross(radial_body, self.inertia @ radial_body)


    def body_torque_at(self, anom: float) -> np.ndarray:
        """Applied environmental torque in body coordinates, N m; zero by default."""
        q = self._query(anom)
        return self._body_torque(q, self._state_at(q)[1])


    def inertial_angular_momentum_at(self, anom: float) -> np.ndarray:
        """Spin angular momentum about the target COM in inertial components, kg m²/s."""
        _, quaternion, omega = self._state_at(anom)
        return Rotation.from_quat(quaternion).apply(self.inertia @ omega)


    def rotation_matrix_at(self, anom: float) -> np.ndarray:
        """Proper body-to-LVLH rotation without an Euler-chart conversion."""
        return Rotation.from_quat(self.quaternion_at(anom)).as_matrix()


    def rotation_matrices(self, anom: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return exact ``(R, R', R'')`` mapping body to LVLH; primes are per q.

        Derivatives use the propagated physical body rate and Euler's equation,
        the anomaly-clock derivative, and the rotating LVLH basis. No Euler
        finite differences or constant-Euler-rate approximation is used.
        """
        q = self._query(anom)
        rotation = self.rotation_matrix_at(q)
        omega = self._state_at(q)[2]
        rate, rate_prime = self.anomaly_clock(q)
        omega_dot = body_rate_derivative(omega, self.inertia, self.body_torque_at(q))
        body_anomaly_rate = omega / rate
        body_anomaly_prime = (omega_dot - omega * rate_prime) / rate**2
        k = gen_skew_sym_mat(body_anomaly_rate)
        kp = gen_skew_sym_mat(body_anomaly_prime)
        z = gen_skew_sym_mat(np.array([0.0, 0.0, 1.0]))
        prime = -z @ rotation + rotation @ k
        double_prime = z @ z @ rotation - 2.0 * z @ rotation @ k + rotation @ (k @ k + kp)
        return rotation, prime, double_prime


    def orientation_at(self, anom: float) -> np.ndarray:
        """Canonical Rx Ry Rz Euler chart of body-to-LVLH attitude, in radians."""
        return quaternion_to_euler(self.quaternion_at(anom))


    def angular_velocity_at(self, anom: float) -> np.ndarray:
        """Euler/anomaly derivatives for the legacy target protocol, not rad/s.

        A singular Euler chart raises explicitly. Quaternion/rotation and
        docking kinematics remain defined through such a chart singularity.
        """
        q = self._query(anom)
        rotation = Rotation.from_quat(self.quaternion_at(q)).as_matrix()
        rate, _ = self.anomaly_clock(q)
        relative_body_rate = self.body_angular_velocity_at(q) / rate - rotation.T @ np.array([0.0, 0.0, 1.0])
        return body_rates_to_euler_rates(self.orientation_at(q), relative_body_rate)


    def docking_state(self, docking_point_body: np.ndarray, anom: float) -> np.ndarray:
        point = finite_vector(docking_point_body, "Body-fixed docking point", 3)
        rotation, prime, _ = self.rotation_matrices(anom)
        return np.concatenate((rotation @ point, prime @ point))


    def docking_acceleration(self, docking_point_body: np.ndarray, anom: float) -> np.ndarray:
        point = finite_vector(docking_point_body, "Body-fixed docking point", 3)
        return self.rotation_matrices(anom)[2] @ point


    def recorded_kinematics(self, anomalies: Sequence[float]) -> Dict[str, np.ndarray]:
        """Sample the same dense target used by controller/reference queries.

        Samples preserve caller order. Returned arrays are independent copies;
        quaternion histories are continuous even when Euler charts wrap. Both
        rotations and both quaternion frames are named explicitly.
        """
        queries = np.asarray(anomalies, dtype=float)
        if queries.ndim != 1 or len(queries) == 0:
            raise ValueError("Target kinematic samples require a nonempty one-dimensional anomaly sequence")
        queries = np.array([self._query(q) for q in queries])
        states = [self._state_at(q) for q in queries]
        rotations = [self.rotation_matrices(q) for q in queries]
        clock = np.array([self.anomaly_clock(q) for q in queries])
        inertial_quaternions = np.array([state[1] for state in states])
        inertial_rotations = Rotation.from_quat(inertial_quaternions).as_matrix()
        body_rates = np.array([state[2] for state in states])
        body_torques = np.array([self._body_torque(q, state[1]) for q, state in zip(queries, states)])
        positions = np.array([self.orbital_position_inertial_at(q) for q in queries])
        orbital_radii = np.linalg.norm(positions, axis=1)
        body_momenta = body_rates @ self.inertia.T
        return dict(anomaly=queries, physical_time=np.array([state[0] for state in states]),
                    quaternions=np.array([self.quaternion_at(q) for q in queries]),
                    inertial_quaternions=inertial_quaternions,
                    inertial_rotations=inertial_rotations,
                    body_angular_velocity=body_rates,
                    body_angular_acceleration=np.array([body_rate_derivative(omega, self.inertia, torque)
                        for omega, torque in zip(body_rates, body_torques)]),
                    body_torque=body_torques,
                    inertial_torque=np.einsum("nij,nj->ni", inertial_rotations, body_torques),
                    inertial_angular_momentum=np.einsum("nij,nj->ni", inertial_rotations, body_momenta),
                    rotational_kinetic_energy=0.5 * np.einsum("ni,ni->n", body_rates, body_momenta),
                    rotational_power=np.einsum("ni,ni->n", body_rates, body_torques),
                    orbital_position_inertial=positions, orbital_radius=orbital_radii,
                    radial_unit_body=np.einsum("nji,nj->ni", inertial_rotations, positions / orbital_radii[:, None]),
                    rotations=np.array([value[0] for value in rotations]),
                    rotation_primes=np.array([value[1] for value in rotations]),
                    rotation_double_primes=np.array([value[2] for value in rotations]),
                    anomaly_rate=clock[:, 0], anomaly_rate_prime=clock[:, 1])
