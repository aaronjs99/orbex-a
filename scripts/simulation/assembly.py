"""Physical post-capture rigid assembly, in SI units and a fixed inertial frame.

The body basis stays fixed in the target structure. Its origin need not equal
the assembly center of mass. Fuel is distributed with its owning component;
depletion updates mass, center of mass and inertia. Removed fuel is assumed to
leave with the local co-rotation velocity in addition to its exhaust velocity.
Small symmetric depletion splits preserve the material structure while moving
the center-of-mass coordinate. This is a finite-step variable-mass model, not a
sloshing, elastic-contact or resolved plume model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Sequence
import numpy as np
from scipy.spatial.transform import Rotation

from orbexa.core.attitude import finite_vector, inertia_matrix

STANDARD_GRAVITY = 9.80665


def finite_scalar(value: float, name: str) -> float:
    """Normalize a real SI scalar; strings, booleans and arrays are not scalars.

    Python/NumPy real numbers and Fraction are accepted and stored as binary64
    for numerical physical propagation. Implicit string or __float__ coercion
    must not leave a nonnumeric object inside an otherwise validated model.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real numeric scalar")
    try:
        value = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be representable as a finite physical scalar") from error
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real numeric scalar")
    return value


def positive(value: float, name: str, *, allow_zero: bool = False) -> float:
    value = finite_scalar(value, name)
    if (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return value


def physical_inertia(value) -> np.ndarray:
    """Require SPD inertia and the rigid-body principal-moment triangle rule."""
    matrix = inertia_matrix(value)
    principal = np.linalg.eigvalsh(matrix)
    if principal[-1] > principal[0] + principal[1] + 1e-12 * principal[-1]:
        raise ValueError("Inertia principal moments must satisfy the physical triangle inequality")
    return matrix


def parallel_axis(offset) -> np.ndarray:
    offset = np.asarray(offset, dtype=float)
    return np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset)


@dataclass(frozen=True)
class MassElement:
    """One attached body; inertia per kg and centroid are in assembly axes.

    Constant specific inertia approximates fuel as distributed like dry mass.
    Both dry and wet inertia therefore remain physically realizable.
    """
    name: str
    dry_mass_kg: float
    initial_fuel_kg: float
    inertia_per_kg_m2: np.ndarray
    position_body_m: np.ndarray
    rotation_element_to_assembly: np.ndarray = field(default_factory=lambda: np.eye(3))

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Mass element name must be nonempty")
        object.__setattr__(self, "dry_mass_kg", positive(self.dry_mass_kg, "Dry mass"))
        object.__setattr__(self, "initial_fuel_kg", positive(self.initial_fuel_kg, "Fuel mass", allow_zero=True))
        for name, value in (("inertia_per_kg_m2", physical_inertia(self.inertia_per_kg_m2)),
                            ("position_body_m", finite_vector(self.position_body_m, "Component position", 3))):
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        rotation = np.asarray(self.rotation_element_to_assembly, dtype=float)
        if (rotation.shape != (3, 3) or np.any(~np.isfinite(rotation))
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0)
                or not np.isclose(np.linalg.det(rotation), 1., atol=1e-10, rtol=0)):
            raise ValueError("Component mounting rotation must be proper orthogonal")
        rotation = rotation.copy()
        rotation.setflags(write=False)
        object.__setattr__(self, "rotation_element_to_assembly", rotation)

    @property
    def initial_mass_kg(self):
        return self.dry_mass_kg + self.initial_fuel_kg

    def to_dict(self):
        return dict(name=self.name, dry_mass_kg=float(self.dry_mass_kg), initial_fuel_kg=float(self.initial_fuel_kg),
                    inertia_per_kg_m2=self.inertia_per_kg_m2.tolist(), position_body_m=self.position_body_m.tolist(),
                    rotation_element_to_assembly=self.rotation_element_to_assembly.tolist())


@dataclass(frozen=True)
class AssemblyState:
    time_s: float
    position_eci_m: np.ndarray
    velocity_eci_m_s: np.ndarray
    quaternion_body_to_eci: np.ndarray
    angular_velocity_body_rad_s: np.ndarray
    fuel_kg: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "time_s", finite_scalar(self.time_s, "Assembly time"))
        for name, size in (("position_eci_m", 3), ("velocity_eci_m_s", 3),
                           ("quaternion_body_to_eci", 4), ("angular_velocity_body_rad_s", 3)):
            value = finite_vector(getattr(self, name), name, size)
            if name == "quaternion_body_to_eci":
                norm = np.linalg.norm(value)
                if norm == 0:
                    raise ValueError("Assembly quaternion must have nonzero norm")
                value /= norm
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        fuel = np.asarray(self.fuel_kg, dtype=float)
        if fuel.ndim != 1 or np.any(~np.isfinite(fuel)) or np.any(fuel < 0):
            raise ValueError("Assembly fuel must be a finite nonnegative vector")
        fuel = fuel.copy()
        fuel.setflags(write=False)
        object.__setattr__(self, "fuel_kg", fuel)

    def to_dict(self):
        return {name: float(value) if name == "time_s" else value.tolist() for name, value in vars(self).items()}


@dataclass(frozen=True)
class AttachedAssembly:
    elements: Sequence[MassElement]
    mu_m3_s2: float = 3.986004418e14
    earth_radius_m: float = 6378137.0
    gravity_gradient: bool = False
    max_integration_step_s: float = 1.0

    def __post_init__(self):
        elements = tuple(self.elements)
        if not elements or len({element.name for element in elements}) != len(elements):
            raise ValueError("Assembly needs uniquely named mass elements")
        object.__setattr__(self, "elements", elements)
        for name in ("mu_m3_s2", "earth_radius_m", "max_integration_step_s"):
            object.__setattr__(self, name, positive(getattr(self, name), name))

    @property
    def initial_mass_kg(self):
        return sum(element.initial_mass_kg for element in self.elements)

    @property
    def initial_fuel_kg(self):
        return np.array([element.initial_fuel_kg for element in self.elements])

    def mass_properties(self, fuel_kg):
        fuel = np.asarray(fuel_kg, dtype=float)
        if (fuel.shape != (len(self.elements),) or np.any(~np.isfinite(fuel)) or np.any(fuel < -1e-12)
                or np.any(fuel > self.initial_fuel_kg + 1e-10)):
            raise ValueError("Fuel must match assembly components and remain within installed capacity")
        masses = np.array([element.dry_mass_kg for element in self.elements]) + np.maximum(fuel, 0)
        positions = np.array([element.position_body_m for element in self.elements])
        mass = float(np.sum(masses))
        center = masses @ positions / mass
        inertia = sum(m * (element.inertia_per_kg_m2 + parallel_axis(element.position_body_m - center))
                      for m, element in zip(masses, self.elements))
        return mass, center, inertia

    def angular_momentum_eci(self, state):
        inertia = self.mass_properties(state.fuel_kg)[2]
        return Rotation.from_quat(state.quaternion_body_to_eci).apply(inertia @ state.angular_velocity_body_rad_s)

    def state_record(self, state):
        mass, center, inertia = self.mass_properties(state.fuel_kg)
        result = state.to_dict()
        result.update(mass_kg=mass, total_fuel_kg=float(np.sum(state.fuel_kg)), center_of_mass_body_m=center.tolist(),
                      inertia_body_kg_m2=inertia.tolist(), angular_momentum_eci_kg_m2_s=self.angular_momentum_eci(state).tolist(),
                      rotational_energy_j=float(state.angular_velocity_body_rad_s @ inertia @ state.angular_velocity_body_rad_s / 2))
        return result

    def component_states(self, state):
        """Physical component poses for renderers, relative to the current COM."""
        rotation = Rotation.from_quat(state.quaternion_body_to_eci)
        center = self.mass_properties(state.fuel_kg)[1]
        result = {}
        for element in self.elements:
            offset = element.position_body_m - center
            result[element.name] = dict(
                position_eci_m=(state.position_eci_m + rotation.apply(offset)).tolist(),
                velocity_eci_m_s=(state.velocity_eci_m_s + rotation.apply(np.cross(state.angular_velocity_body_rad_s, offset))).tolist(),
                quaternion_body_to_eci=(rotation * Rotation.from_matrix(element.rotation_element_to_assembly)).as_quat().tolist())
        return result

    def gravity_gradient_torque_body(self, state):
        """Known central-field torque at the actual state and current fuel load."""
        if not self.gravity_gradient:
            return np.zeros(3)
        inertia = self.mass_properties(state.fuel_kg)[2]
        rotation = Rotation.from_quat(state.quaternion_body_to_eci).as_matrix()
        return self._gravity_gradient_torque_body(state.position_eci_m, rotation, inertia)

    def _gravity_gradient_torque_body(self, position, rotation, inertia):
        radius = np.linalg.norm(position)
        if radius <= 0:
            raise ValueError("Inertial orbital position must be nonzero")
        radial_body = rotation.T @ (position / radius)
        return 3 * self.mu_m3_s2 / radius**3 * np.cross(radial_body, inertia @ radial_body)

    def derivative(self, values, mass, inertia, force_body, torque_body):
        position, velocity, q, omega = values[:3], values[3:6], values[6:10], values[10:13]
        radius = np.linalg.norm(position)
        if radius <= 0:
            raise ValueError("Inertial orbital position must be nonzero")
        rotation = Rotation.from_quat(q).as_matrix()
        torque = np.asarray(torque_body, dtype=float).copy()
        if self.gravity_gradient:
            torque += self._gravity_gradient_torque_body(position, rotation, inertia)
        qdot = .5 * np.r_[q[3] * omega + np.cross(q[:3], omega), -np.dot(q[:3], omega)]
        result = np.r_[velocity, -self.mu_m3_s2 * position / radius**3 + rotation @ force_body / mass,
                       qdot, np.linalg.solve(inertia, torque - np.cross(omega, inertia @ omega))]
        return result, rotation @ torque

    def advance(self, state, duration_s, force_body_n, torque_body_nm, fuel_flow_kg_s):
        return self.advance_with_audit(state, duration_s, force_body_n, torque_body_nm, fuel_flow_kg_s)[0]

    def advance_with_audit(self, state, duration_s, force_body_n, torque_body_nm, fuel_flow_kg_s):
        """Integrate an actual held wrench, without instantaneous attitude resets.

        The angular audit separates external torque impulse from angular
        momentum carried away by the assumed co-rotating fuel distribution.
        """
        duration = positive(duration_s, "Propagation duration")
        force = finite_vector(force_body_n, "Applied body force (N)", 3)
        torque = finite_vector(torque_body_nm, "Applied body torque (N m)", 3)
        flow = np.asarray(fuel_flow_kg_s, dtype=float)
        self.mass_properties(state.fuel_kg)
        if flow.shape != state.fuel_kg.shape or np.any(~np.isfinite(flow)) or np.any(flow < 0):
            raise ValueError("Fuel flow must match the assembly and be finite and nonnegative")
        if np.any(flow * duration > state.fuel_kg + 1e-12):
            raise ValueError("Held propulsion command would consume unavailable fuel")
        count = max(1, int(np.ceil(duration / self.max_integration_step_s)))
        dt = duration / count
        values = np.r_[state.position_eci_m, state.velocity_eci_m_s, state.quaternion_body_to_eci,
                       state.angular_velocity_body_rad_s]
        fuel = state.fuel_kg.copy()
        control_center = self.mass_properties(fuel)[1]
        impulse, flux = np.zeros(3), np.zeros(3)

        def depletion(new_fuel):
            nonlocal values, fuel, flux
            _, old_center, old_inertia = self.mass_properties(fuel)
            _, new_center, new_inertia = self.mass_properties(new_fuel)
            rotation = Rotation.from_quat(values[6:10]).as_matrix()
            shift = new_center - old_center
            values[:3] += rotation @ shift
            values[3:6] += rotation @ np.cross(values[10:13], shift)
            flux += rotation @ ((new_inertia - old_inertia) @ values[10:13])
            fuel = new_fuel

        for _ in range(count):
            depletion(np.maximum(0, fuel - .5 * dt * flow))
            mass, center, inertia = self.mass_properties(fuel)
            held_torque = torque - np.cross(center - control_center, force)
            k1, t1 = self.derivative(values, mass, inertia, force, held_torque)
            k2, t2 = self.derivative(values + .5 * dt * k1, mass, inertia, force, held_torque)
            k3, t3 = self.derivative(values + .5 * dt * k2, mass, inertia, force, held_torque)
            k4, t4 = self.derivative(values + dt * k3, mass, inertia, force, held_torque)
            values += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            values[6:10] /= np.linalg.norm(values[6:10])
            impulse += dt / 6 * (t1 + 2 * t2 + 2 * t3 + t4)
            depletion(np.maximum(0, fuel - .5 * dt * flow))
        if not np.all(np.isfinite(values)):
            raise RuntimeError("Assembly integration produced a non-finite state")
        final = AssemblyState(state.time_s + duration, values[:3], values[3:6], values[6:10], values[10:13], fuel)
        residual = self.angular_momentum_eci(final) - self.angular_momentum_eci(state) - impulse - flux
        return final, dict(external_angular_impulse_eci_kg_m2_s=impulse.tolist(),
                           fuel_corotation_angular_impulse_eci_kg_m2_s=flux.tolist(),
                           angular_momentum_balance_residual_kg_m2_s=float(np.linalg.norm(residual)))

    def to_dict(self):
        return dict(elements=[element.to_dict() for element in self.elements], mu_m3_s2=self.mu_m3_s2,
                    earth_radius_m=self.earth_radius_m, gravity_gradient=self.gravity_gradient,
                    max_integration_step_s=self.max_integration_step_s,
                    mass_model="component-proportional fuel distribution; symmetric depletion splitting")
