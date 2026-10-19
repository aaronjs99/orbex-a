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
    attachment_limits: Sequence[dict] = ()
    environment: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        elements = tuple(self.elements)
        if not elements or len({element.name for element in elements}) != len(elements):
            raise ValueError("Assembly needs uniquely named mass elements")
        object.__setattr__(self, "elements", elements)
        for name in ("mu_m3_s2", "earth_radius_m", "max_integration_step_s"):
            object.__setattr__(self, name, positive(getattr(self, name), name))
        limits = tuple(dict(value) for value in self.attachment_limits)
        if len({value['name'] for value in limits}) != len(limits) or any(value['name'] not in {e.name for e in elements} for value in limits):
            raise ValueError('Attachment load limits must name unique assembly elements')
        for value in limits:
            normal = finite_vector(value['normal_body'], 'Attachment load normal', 3)
            if not np.isclose(np.linalg.norm(normal), 1., atol=1e-9, rtol=0):
                raise ValueError('Attachment load normal must be unit length')
            value['normal_body'] = normal.tolist()
            value['surface_anchor_body_m'] = finite_vector(value['surface_anchor_body_m'], 'Attachment load point', 3).tolist()
            for name in ('tensile_limit_n', 'compression_limit_n', 'shear_limit_n', 'moment_limit_nm'):
                value[name] = positive(value[name], name)
        object.__setattr__(self, 'attachment_limits', limits)
        if self.environment is not None and not callable(self.environment):
            raise TypeError('Assembly environment must be an explicit callable')

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

    def environment_wrench(self, values, fuel_kg, time_s):
        """Extra acceleration beyond central gravity and torque beyond GG.

        The callback receives the same flat name-to-ContactBodyState mapping
        as finite contact, preserving the physical ECI basis across the latch.
        Returned intrinsic torques use each individual component body frame.
        """
        if self.environment is None:
            return np.zeros(3), np.zeros(3), {}
        from orbexa.simulation.contact import ContactBodyState
        fuel = np.asarray(fuel_kg, dtype=float)
        _, center, _ = self.mass_properties(fuel)
        rotation = Rotation.from_quat(values[6:10]).as_matrix()
        omega = values[10:13]
        states = {}
        for index, element in enumerate(self.elements):
            offset = element.position_body_m-center
            local = element.rotation_element_to_assembly
            states[element.name] = ContactBodyState(time_s, values[:3]+rotation@offset,
                values[3:6]+rotation@np.cross(omega, offset), Rotation.from_matrix(rotation@local).as_quat(),
                local.T@omega, fuel[index])
        additional = self.environment(time_s, states)
        if set(additional) != set(states):
            raise ValueError('Assembly environment must declare every component extra wrench')
        force, torque = np.zeros(3), np.zeros(3)
        normalized = {}
        for index, element in enumerate(self.elements):
            extra = additional[element.name]
            acceleration = finite_vector(extra['acceleration_eci_m_s2'], 'Extra environment acceleration', 3)
            moment = finite_vector(extra.get('torque_body_nm', np.zeros(3)), 'Extra environment torque', 3)
            local_force = (element.dry_mass_kg+fuel[index])*(rotation.T@acceleration)
            force += local_force
            torque += np.cross(element.position_body_m-center, local_force)+element.rotation_element_to_assembly@moment
            normalized[element.name] = dict(acceleration_eci_m_s2=acceleration, torque_body_nm=moment)
        return force, torque, normalized

    def derivative(self, values, mass, inertia, force_body, torque_body, *, time_s=None, fuel_kg=None):
        position, velocity, q, omega = values[:3], values[3:6], values[6:10], values[10:13]
        radius = np.linalg.norm(position)
        if radius <= 0:
            raise ValueError("Inertial orbital position must be nonzero")
        rotation = Rotation.from_quat(q).as_matrix()
        torque = np.asarray(torque_body, dtype=float).copy()
        force = np.asarray(force_body, dtype=float).copy()
        if self.environment is not None:
            if time_s is None or fuel_kg is None:
                raise ValueError('Assembly environment propagation requires physical time and fuel')
            extra_force, extra_torque, _ = self.environment_wrench(values, fuel_kg, time_s)
            force += extra_force
            torque += extra_torque
        if self.gravity_gradient:
            torque += self._gravity_gradient_torque_body(position, rotation, inertia)
        qdot = .5 * np.r_[q[3] * omega + np.cross(q[:3], omega), -np.dot(q[:3], omega)]
        result = np.r_[velocity, -self.mu_m3_s2 * position / radius**3 + rotation @ force / mass,
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

        for step in range(count):
            depletion(np.maximum(0, fuel - .5 * dt * flow))
            mass, center, inertia = self.mass_properties(fuel)
            held_torque = torque - np.cross(center - control_center, force)
            clock = state.time_s+step*dt
            k1, t1 = self.derivative(values, mass, inertia, force, held_torque, time_s=clock, fuel_kg=fuel)
            k2, t2 = self.derivative(values + .5 * dt * k1, mass, inertia, force, held_torque, time_s=clock+.5*dt, fuel_kg=fuel)
            k3, t3 = self.derivative(values + .5 * dt * k2, mass, inertia, force, held_torque, time_s=clock+.5*dt, fuel_kg=fuel)
            k4, t4 = self.derivative(values + dt * k3, mass, inertia, force, held_torque, time_s=clock+dt, fuel_kg=fuel)
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
                    attachment_limits=[dict(value) for value in self.attachment_limits],
                    environment='injected per-component extra ECI acceleration and intrinsic body torque' if self.environment else 'central gravity and configured gravity gradient only',
                    environment_contract=self.environment.as_dict() if callable(getattr(self.environment, 'as_dict', None)) else None,
                    mass_model="component-proportional fuel distribution; symmetric depletion splitting")

    def attachment_load_audit(self, state, propulsion, commands_n):
        """Reconstruct joint wrenches from actual allocated nozzle commands.

        This uses each component's Newton-Euler balance at the supplied state.
        No bond strength is inferred from the target's collision geometry.
        """
        if not self.attachment_limits:
            return dict(passed=True, maximum_capacity_ratio=0., members={})
        commands = np.asarray(commands_n, dtype=float)
        if commands.shape != (len(propulsion.thrusters),) or np.any(~np.isfinite(commands)):
            raise ValueError('Attachment loads require actual allocated thruster commands')
        mass, center, inertia = self.mass_properties(state.fuel_kg)
        force, torque = np.zeros((len(self.elements), 3)), np.zeros((len(self.elements), 3))
        for command, thruster in zip(commands, propulsion.thrusters):
            local_force = command*thruster.direction_body
            force[thruster.owner_index] += local_force
            torque[thruster.owner_index] += np.cross(thruster.position_body_m-
                self.elements[thruster.owner_index].position_body_m, local_force)
        allocated = propulsion.matrix(self, state.fuel_kg)@commands
        if self.environment is not None:
            values = np.r_[state.position_eci_m, state.velocity_eci_m_s, state.quaternion_body_to_eci,
                           state.angular_velocity_body_rad_s]
            extra_force, extra_torque, per_component = self.environment_wrench(values, state.fuel_kg, state.time_s)
            allocated += np.r_[extra_force, extra_torque]
            rotation = Rotation.from_quat(state.quaternion_body_to_eci).as_matrix()
            for index, element in enumerate(self.elements):
                value = per_component[element.name]
                force[index] += (element.dry_mass_kg+state.fuel_kg[index])*(rotation.T@value['acceleration_eci_m_s2'])
                torque[index] += element.rotation_element_to_assembly@value['torque_body_nm']
        w = state.angular_velocity_body_rad_s
        alpha = np.linalg.solve(inertia, allocated[3:]+self.gravity_gradient_torque_body(state)-np.cross(w, inertia@w))
        rotation = Rotation.from_quat(state.quaternion_body_to_eci).as_matrix()
        gravity_com = -self.mu_m3_s2*state.position_eci_m/np.linalg.norm(state.position_eci_m)**3
        result = {}
        by_name = {element.name: index for index, element in enumerate(self.elements)}
        for limit in self.attachment_limits:
            index = by_name[limit['name']]
            element = self.elements[index]
            wet_mass = element.dry_mass_kg+state.fuel_kg[index]
            local_inertia = wet_mass*element.inertia_per_kg_m2
            offset = element.position_body_m-center
            position = state.position_eci_m+rotation@offset
            gravity_element = -self.mu_m3_s2*position/np.linalg.norm(position)**3
            acceleration = allocated[:3]/mass+np.cross(alpha, offset)+np.cross(w, np.cross(w, offset))
            acceleration += rotation.T@(gravity_com-gravity_element)
            reaction = wet_mass*acceleration-force[index]
            local_gg = self._gravity_gradient_torque_body(position, rotation, local_inertia) if self.gravity_gradient else np.zeros(3)
            moment_com = local_inertia@alpha+np.cross(w, local_inertia@w)-torque[index]-local_gg
            arm = np.asarray(limit['surface_anchor_body_m'])-element.position_body_m
            moment = moment_com-np.cross(arm, reaction)
            normal = np.asarray(limit['normal_body'])
            fn = float(reaction@normal)
            shear = float(np.linalg.norm(reaction-fn*normal))
            ratios = dict(tensile=max(0., -fn)/limit['tensile_limit_n'],
                          compression=max(0., fn)/limit['compression_limit_n'],
                          shear=shear/limit['shear_limit_n'], moment=float(np.linalg.norm(moment))/limit['moment_limit_nm'])
            result[element.name] = dict(force_body_n=reaction.tolist(), moment_at_pad_body_nm=moment.tolist(),
                                       capacity_ratios=ratios, maximum_capacity_ratio=max(ratios.values()))
        maximum = max([0.]+[value['maximum_capacity_ratio'] for value in result.values()])
        return dict(passed=maximum <= 1., maximum_capacity_ratio=maximum, members=result,
                    scope='Newton-Euler attachment loads at recorded controlled states; representative declared bond capacities')

    def attachment_load_enclosure(self, state, propulsion, commands_n, duration_s, **initial_sets):
        """Bound joint loads throughout an exact held interval before motion.

        A default point start is a conditional local obligation. Supplying a
        certified start set makes its scope explicit; this method does not
        infer accumulated integration or navigation uncertainty from a point.
        """
        from orbexa.simulation.attachment_enclosure import AttachmentLoadEncloser
        return AttachmentLoadEncloser().bound(assembly=self, state=state,
            propulsion=propulsion, commands_n=commands_n, duration_s=duration_s, **initial_sets)

    def attachment_history_audit(self, propulsion, controls, history):
        """Check actual component loads at every recorded control endpoint."""
        if not self.attachment_limits:
            return dict(passed=True, maximum_capacity_ratio=0., checked_states=0,
                        scope='No physical bond capacities declared by the explicit idealized reproduction model')
        by_time = {float(record['time_s']): record for record in history}
        maximum, checked, first = 0., 0, None
        checked_times = set()
        member_maxima = {value['name']: 0. for value in self.attachment_limits}
        for control in controls:
            for time_s in (control['time_s'], control['time_s']+control['duration_s']):
                key = min(by_time, key=lambda value: abs(value-time_s))
                if abs(key-time_s) > 1e-6:
                    raise ValueError('Attachment load audit is missing a control boundary state')
                record = by_time[key]
                state = AssemblyState(record['time_s'], record['position_eci_m'], record['velocity_eci_m_s'],
                    record['quaternion_body_to_eci'], record['angular_velocity_body_rad_s'], record['fuel_kg'])
                audit = self.attachment_load_audit(state, propulsion, control['allocation']['commands_n'])
                maximum = max(maximum, audit['maximum_capacity_ratio'])
                checked += 1
                checked_times.add(key)
                for name, value in audit['members'].items():
                    member_maxima[name] = max(member_maxima[name], value['maximum_capacity_ratio'])
                if not audit['passed'] and first is None:
                    first = dict(time_s=time_s, **audit)
        for record in history:
            time_s = float(record['time_s'])
            if time_s in checked_times and not record.get('phase', '').startswith('coast'):
                continue
            state = AssemblyState(time_s, record['position_eci_m'], record['velocity_eci_m_s'],
                record['quaternion_body_to_eci'], record['angular_velocity_body_rad_s'], record['fuel_kg'])
            audit = self.attachment_load_audit(state, propulsion, np.zeros(len(propulsion.thrusters)))
            maximum = max(maximum, audit['maximum_capacity_ratio'])
            checked += 1
            for name, value in audit['members'].items():
                member_maxima[name] = max(member_maxima[name], value['maximum_capacity_ratio'])
            if not audit['passed'] and (first is None or time_s < first['time_s']):
                first = dict(time_s=time_s, **audit)
        return dict(passed=first is None, maximum_capacity_ratio=maximum, member_maximum_capacity_ratios=member_maxima,
                    checked_states=checked, first_violation=first,
                    scope='Every actual control start/end and recorded zero-thrust coast state; no continuous load enclosure between recorded states')
