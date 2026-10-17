"""Finite compliant capture of a rigid target by independently moving buses.

This is a declared representative adhesive-pad study model. The finite-bandwidth
positioner has negligible moving mass; its reaction work is explicitly counted.
It is not a manufactured gripper specification or a flexible spacecraft model.
The six-DOF vehicle/contact-wrench formulation follows the modeling scope of
NASA-CR-179147: https://ntrs.nasa.gov/citations/19870016368 . Adhesion strengths,
pad properties and actuator parameters below are independent study assumptions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Optional
import numpy as np
from scipy.spatial.transform import Rotation

from orbexa.core.attitude import finite_vector
from orbexa.simulation.assembly import STANDARD_GRAVITY, physical_inertia, positive


def _limit(vector, maximum):
    norm = float(np.linalg.norm(vector))
    return np.asarray(vector) * min(1., maximum / max(norm, 1e-30))


@dataclass(frozen=True)
class FiniteContactOptions:
    max_duration_s: float = 60.
    max_step_s: float = .005
    history_step_s: float = .05
    extension_speed_m_s: float = .020
    extension_acceleration_m_s2: float = .020
    positioner_time_constant_s: float = .15
    touch_speed_m_s: float = .001
    slow_touch_distance_m: float = .002
    surface_search_depth_m: float = .020
    range_error_bound_m: float = .002
    pad_radius_m: float = .025
    positioner_radius_m: float = .005
    compression_stiffness_n_m: float = 2000.
    compression_damping_n_s_m: float = 700.
    max_compression_m: float = .003
    preload_n: float = 1.
    preload_command_factor: float = 1.5
    preload_dwell_s: float = .5
    friction_coefficient: float = .5
    friction_regularization_m_s: float = .001
    bond_stiffness_n_m: float = 20000.
    bond_damping_n_s_m: float = 1000.
    bond_rotational_stiffness_nm_rad: float = 500.
    bond_rotational_damping_nm_s_rad: float = 300.
    bond_engagement_s: float = 1.
    tensile_limit_n: float = 250.
    shear_limit_n: float = 250.
    moment_limit_nm: float = 5.
    positioner_force_limit_n: float = 400.
    positioner_power_limit_w: float = 20.
    translation_force_n: float = 100.
    attitude_nozzle_force_n: float = 3.
    attitude_moment_arm_m: float = .4
    specific_impulse_s: float = 300.
    stationkeeping_frequency_rad_s: float = .5
    attitude_frequency_rad_s: float = 1.
    settling_dwell_s: float = 2.
    settling_speed_m_s: float = .0001
    settling_spin_rad_s: float = .00001
    max_lock_energy_j: float = .0001
    minimum_bus_clearance_m: float = .001
    momentum_absolute_tolerance: float = 1e-7
    momentum_relative_tolerance: float = 1e-7
    energy_absolute_tolerance_j: float = 1e-5
    energy_relative_tolerance: float = .001

    def __post_init__(self):
        for name, value in vars(self).items():
            object.__setattr__(self, name, positive(value, name, allow_zero=name in (
                'friction_coefficient', 'minimum_bus_clearance_m')))
        if self.touch_speed_m_s > self.extension_speed_m_s:
            raise ValueError('Touch speed cannot exceed free extension speed')
        if self.settling_dwell_s + self.preload_dwell_s + self.bond_engagement_s >= self.max_duration_s:
            raise ValueError('Finite contact dwell must fit the contact duration')
        if self.preload_n >= self.compression_stiffness_n_m * self.max_compression_m:
            raise ValueError('Preload must fit the allowed pad compression')


@dataclass(frozen=True)
class ContactBodyState:
    time_s: float
    position_eci_m: np.ndarray
    velocity_eci_m_s: np.ndarray
    quaternion_body_to_eci: np.ndarray
    angular_velocity_body_rad_s: np.ndarray
    fuel_kg: float = 0.

    def __post_init__(self):
        object.__setattr__(self, 'time_s', float(self.time_s))
        if not np.isfinite(self.time_s):
            raise ValueError('Contact time must be finite')
        for name, size in (('position_eci_m', 3), ('velocity_eci_m_s', 3),
                           ('quaternion_body_to_eci', 4), ('angular_velocity_body_rad_s', 3)):
            value = finite_vector(getattr(self, name), name, size)
            if size == 4:
                norm = np.linalg.norm(value)
                if norm == 0:
                    raise ValueError('Contact quaternion cannot be zero')
                value /= norm
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, 'fuel_kg', positive(self.fuel_kg, 'Contact fuel', allow_zero=True))

    def to_dict(self):
        return {name: value.tolist() if isinstance(value, np.ndarray) else value
                for name, value in vars(self).items()}


@dataclass(frozen=True)
class ContactMember:
    id: str
    properties: Any
    initial_state: ContactBodyState
    surface_anchor_body_m: np.ndarray
    normal_body: np.ndarray
    tangent_u_body: np.ndarray
    tangent_v_body: np.ndarray
    arm_reach_m: float = .1
    patch_id: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id:
            raise ValueError('Contact member requires an id')
        for name in ('surface_anchor_body_m', 'normal_body', 'tangent_u_body', 'tangent_v_body'):
            value = finite_vector(getattr(self, name), name, 3)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        axes = np.column_stack((self.tangent_u_body, self.tangent_v_body, self.normal_body))
        if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-9, rtol=0) or np.linalg.det(axes) <= 0:
            raise ValueError('Contact patch requires right-handed orthonormal axes')
        object.__setattr__(self, 'arm_reach_m', positive(self.arm_reach_m, 'Finite tip reach'))


@dataclass(frozen=True)
class FiniteContactResult:
    success: bool
    message: str
    final_target: ContactBodyState
    final_chasers: Mapping[str, ContactBodyState]
    history: tuple[dict, ...]
    surface_contacts: Mapping[str, dict]
    audit: dict

    def to_dict(self):
        return dict(success=self.success, message=self.message,
            final_target=self.final_target.to_dict(),
            final_chasers={name: state.to_dict() for name, state in self.final_chasers.items()},
            history=list(self.history), surface_contacts=dict(self.surface_contacts), audit=self.audit)


class FiniteContactSimulator:
    """Simultaneous rigid dynamics with objective compliant-pad wrenches.

    ``feedback(time_s, snapshot)`` may return estimated ContactBodyState values
    under the same target/chasers keys. It affects commands only. Without it,
    feedback is explicitly exact. All audits always use the physical plant.
    The snapshot also contains ``contact_gaps_m[id]`` (positive outward) and
    ``contact_wrenches_eci[id]`` with force_n/couple_nm three-vectors. An
    injected observer must supply its own measured versions to use these
    channels; omitted wrench channels mean zero measured force, and omitted
    range channels force a slow search over the declared uncertainty depth.
    ``environment(time_s, bodies_by_name)`` receives a flat mapping containing
    target and every chaser id, and must return an extra acceleration_eci_m_s2
    and optional torque_body_nm for every body. These are beyond central
    gravity and configured gravity-gradient torque, in the supplied ECI basis.
    """

    def __init__(self, options: Optional[FiniteContactOptions] = None):
        self.options = options or FiniteContactOptions()

    def run(self, *, target_properties, target_state: ContactBodyState,
            members, geometry, mu_m3_s2, gravity_gradient=False,
            feedback: Optional[Callable] = None, environment: Optional[Callable] = None) -> FiniteContactResult:
        o, members = self.options, tuple(members)
        if not members or len({m.id for m in members}) != len(members):
            raise ValueError('Finite contact requires a nonempty uniquely identified fleet')
        if any(m.initial_state.time_s != target_state.time_s for m in members):
            raise ValueError('All finite contact bodies must share the exact initial physical clock')
        mu = positive(mu_m3_s2, 'Contact gravitational parameter')
        n, count = len(members) + 1, len(members)
        initial = (target_state,) + tuple(m.initial_state for m in members)
        dry = np.array([target_properties.mass_kg] + [m.properties.dry_mass_kg for m in members])
        specific = np.array([physical_inertia(target_properties.inertia_kg_m2) / dry[0]] +
                            [m.properties.inertia_per_kg_m2 for m in members])
        mounts = np.array([[-m.properties.gripper_offset_m, 0., 0.] for m in members])
        anchors = np.array([m.surface_anchor_body_m for m in members])
        normals = np.array([m.normal_body for m in members])
        # A translating, nonrotating origin avoids subtracting orbital-scale
        # kinetic energies and momenta to resolve millimeter contact effects.
        body_slice, tip_start = slice(6, 6 + 14*n), 6 + 14*n
        tip_slice, ledger_start = slice(tip_start, tip_start + 6*count), tip_start + 6*count
        y = np.zeros(ledger_start + 19)
        y[:6] = np.r_[target_state.position_eci_m, target_state.velocity_eci_m_s]
        bodies = y[body_slice].reshape(n, 14)
        for index, state in enumerate(initial):
            bodies[index] = np.r_[state.position_eci_m - y[:3], state.velocity_eci_m_s - y[3:6],
                                  state.quaternion_body_to_eci, state.angular_velocity_body_rad_s, state.fuel_kg]
        started = target_state.time_s
        engagement = np.full(count, np.inf)
        preload = np.zeros(count)
        bond_rest = np.zeros((count, 3))
        rotation_rest = np.zeros((count, 3, 3))
        frozen_extension = np.full((count, 3), np.nan)
        first_touch = np.full(count, np.nan)
        first_touch_speed = np.full(count, np.nan)
        controls = np.zeros((n, 6))
        extension_acceleration = np.zeros((count, 3))
        sensed_normal_force = np.zeros(count)
        sensed_gap = np.full(count, np.inf)
        saturation_count = 0
        history, contact_records = [], {}
        maxima = dict(force_n=0., moment_nm=0., compression_m=0., extension_m=0.,
                      positioner_power_w=0., internal_force_residual_n=0., internal_torque_residual_nm=0.,
                      quaternion_norm_error=0.)
        minima = dict(bus_component_clearance_m=np.inf, peer_bus_clearance_m=np.inf,
                      positioner_component_clearance_m=np.inf, positioner_peer_clearance_m=np.inf,
                      patch_edge_margin_m=np.inf)

        def gravity(position):
            radius = np.linalg.norm(position)
            if not np.isfinite(radius) or radius <= 0:
                raise ValueError('Contact orbital position must be finite and nonzero')
            return -mu * position / radius**3

        def unpack(values):
            return values[body_slice].reshape(n, 14), values[tip_slice].reshape(count, 6)

        def snapshot(t, values, diag=None):
            b, _ = unpack(values)
            states = tuple(ContactBodyState(started+t, values[:3]+row[:3], values[3:6]+row[3:6],
                row[6:10], row[10:13], max(0., row[13])) for row in b)
            gaps = {}
            if diag is None:
                rotations = Rotation.from_quat(b[:, 6:10]).as_matrix()
                tips = unpack(values)[1]
                for i, member in enumerate(members):
                    tip = b[i+1, :3]+rotations[i+1]@(mounts[i]+tips[i, :3])
                    patch = b[0, :3]+rotations[0]@anchors[i]
                    gaps[member.id] = float((tip-patch)@(rotations[0]@normals[i]))
            else:
                gaps = {m.id: diag['rows'][i]['gap'] for i, m in enumerate(members)}
            return dict(target=states[0], chasers={m.id: states[i+1] for i, m in enumerate(members)},
                contact_wrenches_eci={m.id: dict(force_n=(np.zeros(3) if diag is None else diag['rows'][i]['force']).tolist(),
                    couple_nm=(np.zeros(3) if diag is None else diag['rows'][i]['couple']).tolist())
                    for i, m in enumerate(members)},
                contact_gaps_m=gaps)

        def check_observation(observation, clock):
            if set(observation['chasers']) != {member.id for member in members}:
                raise ValueError('Contact observations must name the exact fleet')
            for state in (observation['target'], *observation['chasers'].values()):
                if not isinstance(state, ContactBodyState) or abs(state.time_s-clock) > 1e-8:
                    raise ValueError('Contact observations require current-clock ContactBodyState estimates')

        first_observation = snapshot(0., y)
        if feedback is not None:
            first_observation = feedback(started, first_observation)
        check_observation(first_observation, started)
        rt_est = Rotation.from_quat(first_observation['target'].quaternion_body_to_eci).as_matrix()
        reference_offsets, reference_rotations = [], []
        for member in members:
            state = first_observation['chasers'][member.id]
            reference_offsets.append(rt_est.T @ (state.position_eci_m-first_observation['target'].position_eci_m))
            reference_rotations.append(rt_est.T @ Rotation.from_quat(state.quaternion_body_to_eci).as_matrix())

        def evaluate(t, values):
            b, tips = unpack(values)
            r = Rotation.from_quat(b[:, 6:10]).as_matrix()
            w = np.einsum('nij,nj->ni', r, b[:, 10:13])
            mass = dry + b[:, 13]
            inertia = mass[:, None, None] * specific
            force = np.einsum('nij,nj->ni', r, controls[:, :3])
            torque = controls[:, 3:].copy()
            external_force = force.copy()
            external_torque = np.einsum('nij,nj->ni', r, torque)
            reference_acceleration = gravity(values[:3])
            additional = {}
            if environment is not None:
                view = snapshot(t, values)
                body_states = dict(target=view['target'], **view['chasers'])
                additional = environment(started+t, body_states)
                if set(additional) != set(body_states):
                    raise ValueError('Contact environment must declare every target/chaser extra wrench')
            for k in range(n):
                differential_g = gravity(values[:3]+b[k, :3]) - reference_acceleration
                force[k] += mass[k] * differential_g
                external_force[k] += mass[k] * differential_g
                if gravity_gradient:
                    absolute = values[:3]+b[k, :3]
                    radial_body = r[k].T @ (absolute/np.linalg.norm(absolute))
                    gg = 3*mu/np.linalg.norm(absolute)**3*np.cross(radial_body, inertia[k]@radial_body)
                    torque[k] += gg
                    external_torque[k] += r[k] @ gg
                if additional:
                    extra = additional['target' if k == 0 else members[k-1].id]
                    acceleration = finite_vector(extra['acceleration_eci_m_s2'], 'Extra environment acceleration', 3)
                    moment = finite_vector(extra.get('torque_body_nm', np.zeros(3)), 'Extra environment torque', 3)
                    force[k] += mass[k]*acceleration
                    external_force[k] += mass[k]*acceleration
                    torque[k] += moment
                    external_torque[k] += r[k]@moment
            internal_force, internal_torque = np.zeros_like(force), np.zeros_like(force)
            potential, powers = 0., np.zeros(7)  # external, flux, positioner, engagement, compression, friction, bond
            rows = []
            for k, member in enumerate(members):
                j, tip_offset = k+1, r[k+1] @ (mounts[k]+tips[k, :3])
                tip = b[j, :3]+tip_offset
                patch = b[0, :3]+r[0]@anchors[k]
                e = r[0].T@(tip-patch)
                tip_velocity = b[j, 3:6]+np.cross(w[j], tip_offset)+r[j]@tips[k, 3:]
                u = r[0].T@(tip_velocity-b[0, 3:6]-np.cross(w[0], tip-b[0, :3]))
                gap, normal_speed = float(e@normals[k]), float(u@normals[k])
                compression = max(0., -gap)
                elastic = o.compression_stiffness_n_m*compression
                damping = o.compression_damping_n_s_m*max(0., -normal_speed) if compression > 0 else 0.
                normal_force = elastic+damping
                slip = u-normal_speed*normals[k]
                friction = -o.friction_coefficient*normal_force*slip / np.sqrt(slip@slip+o.friction_regularization_m_s**2)
                pad_force = normals[k]*normal_force+friction
                pad_couple = np.zeros(3)
                bond_force, angle, relative_w = np.zeros(3), np.zeros(3), r[0].T@(w[j]-w[0])
                activation = float(np.clip((t-engagement[k])/o.bond_engagement_s, 0., 1.))
                rate = 1/o.bond_engagement_s if 0 <= t-engagement[k] < o.bond_engagement_s else 0.
                potential += .5*o.compression_stiffness_n_m*compression**2
                powers[4] += damping*max(0., -normal_speed)
                powers[5] -= float(friction@slip)
                if np.isfinite(engagement[k]):
                    deformation = e-bond_rest[k]
                    angle = Rotation.from_matrix(r[0].T@r[j]@rotation_rest[k].T).as_rotvec()
                    bond_force = -activation*(o.bond_stiffness_n_m*deformation+o.bond_damping_n_s_m*u)
                    pad_force += bond_force
                    pad_couple = -activation*(o.bond_rotational_stiffness_nm_rad*angle+
                                              o.bond_rotational_damping_nm_s_rad*relative_w)
                    stored = .5*(o.bond_stiffness_n_m*(deformation@deformation)+
                                 o.bond_rotational_stiffness_nm_rad*(angle@angle))
                    potential += activation*stored
                    powers[3] += rate*stored
                    powers[6] += activation*(o.bond_damping_n_s_m*(u@u)+
                                             o.bond_rotational_damping_nm_s_rad*(relative_w@relative_w))
                f, couple = r[0]@pad_force, r[0]@pad_couple
                internal_force[j] += f
                internal_force[0] -= f
                internal_torque[j] += np.cross(tip_offset, f)+couple
                # Objective target-frame deformation requires the force lever
                # at the actual tip, including its compliance displacement.
                internal_torque[0] -= np.cross(tip-b[0, :3], f)+couple
                positioner_power = -float(f@(r[j]@tips[k, 3:]))
                powers[2] += positioner_power
                rows.append(dict(gap=gap, compression=compression, normal_force=normal_force,
                    force=f, force_patch=pad_force, bond_force=bond_force, couple=couple,
                    e=e, u=u, angle=angle, spin=relative_w, activation=activation,
                    positioner_power=positioner_power, tip=tip, patch=patch, rotation=r[j],
                    tip_velocity=tip_velocity, relative_rotation=r[0].T@r[j]))
            force += internal_force
            torque += np.einsum('nji,nj->ni', r, internal_torque)
            flow = np.zeros(n)
            flow[1:] = (np.sum(np.abs(controls[1:, :3]), axis=1)+
                np.sum(np.abs(controls[1:, 3:]), axis=1)/o.attitude_moment_arm_m)/(o.specific_impulse_s*STANDARD_GRAVITY)
            derivative = np.zeros_like(values)
            derivative[:6] = np.r_[values[3:6], reference_acceleration]
            db, ds = unpack(derivative)
            db[:, :3], db[:, 3:6], db[:, 13] = b[:, 3:6], force/mass[:, None], -flow
            q, wb = b[:, 6:10], b[:, 10:13]
            db[:, 6:10] = .5*np.column_stack((q[:, 3, None]*wb+np.cross(q[:, :3], wb),
                                               -np.sum(q[:, :3]*wb, axis=1)))
            for k in range(n):
                db[k, 10:13] = np.linalg.solve(inertia[k], torque[k]-np.cross(wb[k], inertia[k]@wb[k]))
            ds[:, :3], ds[:, 3:] = tips[:, 3:], extension_acceleration
            external_h = np.sum(np.cross(b[:, :3], external_force)+external_torque, axis=0)
            fuel_p = -flow[:, None]*b[:, 3:6]
            fuel_h = np.cross(b[:, :3], fuel_p)-flow[:, None]*np.einsum('nij,nj->ni', r, np.einsum('nij,nj->ni', specific, wb))
            powers[0] = float(np.sum(external_force*b[:, 3:6])+np.sum(external_torque*w))
            powers[1] = -.5*float(np.sum(flow*(np.sum(b[:, 3:6]**2, axis=1)+
                                                np.einsum('ni,nij,nj->n', wb, specific, wb))))
            derivative[ledger_start:] = np.r_[np.sum(external_force, axis=0), np.sum(fuel_p, axis=0),
                                              external_h, np.sum(fuel_h, axis=0), powers]
            linear = np.sum(mass[:, None]*b[:, 3:6], axis=0)
            angular = np.sum(np.cross(b[:, :3], mass[:, None]*b[:, 3:6])+
                np.einsum('nij,nj->ni', r, np.einsum('nij,nj->ni', inertia, wb)), axis=0)
            kinetic = .5*float(np.sum(mass*np.sum(b[:, 3:6]**2, axis=1))+
                               np.einsum('ni,nij,nj->', wb, inertia, wb))
            internal_h = np.sum(np.cross(b[:, :3], internal_force)+internal_torque, axis=0)
            return derivative, dict(rows=rows, rotations=r, spins=w, masses=mass, inertias=inertia,
                potential=potential, kinetic=kinetic, linear=linear, angular=angular,
                internal_force=float(np.linalg.norm(np.sum(internal_force, axis=0))),
                internal_torque=float(np.linalg.norm(internal_h)), fuel_flow=flow,
                acceleration=db[:, 3:6].copy(), angular_acceleration=db[:, 10:13].copy())

        def command(t, values, diag):
            nonlocal saturation_count
            b, tips = unpack(values)
            observation = snapshot(t, values, diag)
            if feedback is not None:
                observation = feedback(started+t, observation)
            check_observation(observation, started+t)
            target = observation['target']
            rt = Rotation.from_quat(target.quaternion_body_to_eci).as_matrix()
            wt = rt@target.angular_velocity_body_rad_s
            measured = observation.get('contact_wrenches_eci', {})
            measured_forces, measured_couples = [], []
            target_torque = np.zeros(3)
            for k, member in enumerate(members):
                channel = measured.get(member.id, {})
                measured_forces.append(finite_vector(channel.get('force_n', np.zeros(3)), 'Measured contact force', 3))
                measured_couples.append(finite_vector(channel.get('couple_nm', np.zeros(3)), 'Measured pad couple', 3))
                estimate = observation['chasers'][member.id]
                ri = Rotation.from_quat(estimate.quaternion_body_to_eci).as_matrix()
                tip = estimate.position_eci_m+ri@(mounts[k]+tips[k, :3])
                target_torque -= np.cross(tip-target.position_eci_m, measured_forces[-1])+measured_couples[-1]
            target_inertia = target_properties.inertia_kg_m2
            target_torque_body = rt.T@target_torque
            if gravity_gradient:
                radial = rt.T@(target.position_eci_m/np.linalg.norm(target.position_eci_m))
                target_torque_body += 3*mu/np.linalg.norm(target.position_eci_m)**3*np.cross(radial, target_inertia@radial)
            target_alpha = rt@np.linalg.solve(target_inertia, target_torque_body-
                np.cross(target.angular_velocity_body_rad_s, target_inertia@target.angular_velocity_body_rad_s))
            target_reaction_acceleration = -np.sum(measured_forces, axis=0)/target_properties.mass_kg
            for k, member in enumerate(members):
                j, state = k+1, observation['chasers'][member.id]
                ri = Rotation.from_quat(state.quaternion_body_to_eci).as_matrix()
                wi = ri@state.angular_velocity_body_rad_s
                offset = rt@reference_offsets[k]
                position_error = target.position_eci_m+offset-state.position_eci_m
                velocity_error = target.velocity_eci_m_s+np.cross(wt, offset)-state.velocity_eci_m_s
                a = gravity(target.position_eci_m)-gravity(state.position_eci_m)+target_reaction_acceleration
                a += np.cross(target_alpha, offset)+np.cross(wt, np.cross(wt, offset))
                a += o.stationkeeping_frequency_rad_s**2*position_error+2*o.stationkeeping_frequency_rad_s*velocity_error
                force_requested = ri.T@(diag['masses'][j]*a-measured_forces[k])
                expected = rt@reference_rotations[k]
                angle = Rotation.from_matrix(ri.T@expected).as_rotvec()
                inertia = diag['inertias'][j]
                torque_requested = inertia@(ri.T@target_alpha+o.attitude_frequency_rad_s**2*angle+
                    2*o.attitude_frequency_rad_s*(ri.T@(wt-wi)))+np.cross(b[j, 10:13], inertia@b[j, 10:13])
                lever = ri@(mounts[k]+tips[k, :3])
                torque_requested -= ri.T@(np.cross(lever, measured_forces[k])+measured_couples[k])
                if gravity_gradient:
                    radial = ri.T@(state.position_eci_m/np.linalg.norm(state.position_eci_m))
                    torque_requested -= 3*mu/np.linalg.norm(state.position_eci_m)**3*np.cross(radial, inertia@radial)
                # Once the compliant bond can carry load, release free-flight
                # stationkeeping continuously. Cancelling measured bond damping
                # indefinitely would create a delayed positive feedback loop.
                free_flight_fraction = 1.-diag['rows'][k]['activation']
                force_requested *= free_flight_fraction
                torque_requested *= free_flight_fraction
                force = np.clip(force_requested, -o.translation_force_n, o.translation_force_n)
                torque_limit = 2*o.attitude_moment_arm_m*o.attitude_nozzle_force_n
                torque = np.clip(torque_requested, -torque_limit, torque_limit)
                saturation_count += int(np.any(force != force_requested) or np.any(torque != torque_requested))
                controls[j] = np.r_[force, torque]
                if b[j, 13] <= 1e-12:
                    controls[j] = 0.
                patch = target.position_eci_m+rt@anchors[k]
                normal = rt@normals[k]
                cmd = ri.T@(patch-state.position_eci_m)-mounts[k]-ri.T@normal*(
                    o.surface_search_depth_m+o.preload_command_factor*o.preload_n/o.compression_stiffness_n_m)
                if np.isfinite(engagement[k]) and not np.all(np.isfinite(frozen_extension[k])):
                    frozen_extension[k] = tips[k, :3]
                if np.all(np.isfinite(frozen_extension[k])):
                    cmd = frozen_extension[k]
                observed_tip = state.position_eci_m+ri@(mounts[k]+tips[k, :3])
                gap_est = float((observed_tip-patch)@normal)
                ranges = observation.get('contact_gaps_m', {})
                if member.id in ranges:
                    gap_est = float(ranges[member.id])
                    if not np.isfinite(gap_est):
                        raise ValueError('Contact gap measurement must be finite')
                    slow_distance = o.slow_touch_distance_m+o.range_error_bound_m
                else:
                    # Without a local range channel, reserve the entire
                    # declared surface-search uncertainty before fast motion.
                    slow_distance = o.slow_touch_distance_m+o.surface_search_depth_m
                sensed_gap[k] = gap_est
                sensed_normal_force[k] = float(measured_forces[k]@normal)
                speed = min(o.extension_speed_m_s, o.touch_speed_m_s+np.sqrt(
                    2*o.extension_acceleration_m_s2*max(0., gap_est-slow_distance)))
                desired_velocity = _limit((cmd-tips[k, :3])/o.positioner_time_constant_s, speed)
                acceleration = _limit((desired_velocity-tips[k, 3:])/o.positioner_time_constant_s,
                                      o.extension_acceleration_m_s2)
                observed_tip_velocity = state.velocity_eci_m_s+np.cross(wi, ri@(mounts[k]+tips[k, :3]))+ri@tips[k, 3:]
                closing = -float((observed_tip_velocity-target.velocity_eci_m_s-
                    np.cross(wt, observed_tip-target.position_eci_m))@normal)
                braking_distance = max(0., closing**2-o.touch_speed_m_s**2)/(2*o.extension_acceleration_m_s2)
                braking_distance += max(0., closing-o.touch_speed_m_s)*o.positioner_time_constant_s
                if closing > o.touch_speed_m_s and gap_est-slow_distance <= braking_distance:
                    outward = ri.T@normal
                    acceleration += outward*(o.extension_acceleration_m_s2-acceleration@outward)
                extension_acceleration[k] = _limit(acceleration, o.extension_acceleration_m_s2)

        def cylinder_distance(point):
            radial = np.linalg.norm(point[:2])-float(geometry['radius'])
            axial = abs(point[2])-float(geometry['half_length'])
            return float(np.linalg.norm(np.maximum([radial, axial], 0.))+min(max(radial, axial), 0.))

        def guard(t, values, diag, dt):
            b, tips = unpack(values)
            rt, wt = diag['rotations'][0], diag['spins'][0]
            segments = []
            for k, member in enumerate(members):
                j, row = k+1, diag['rows'][k]
                extension = float(np.linalg.norm(tips[k, :3]))
                maxima['extension_m'] = max(maxima['extension_m'], extension)
                maxima['force_n'] = max(maxima['force_n'], float(np.linalg.norm(row['force'])))
                maxima['moment_nm'] = max(maxima['moment_nm'], float(np.linalg.norm(row['couple'])))
                maxima['compression_m'] = max(maxima['compression_m'], row['compression'])
                maxima['positioner_power_w'] = max(maxima['positioner_power_w'], abs(row['positioner_power']))
                if extension > member.arm_reach_m+1e-10:
                    return f'{member.id}: positioner_reach_exceeded'
                if np.linalg.norm(tips[k, 3:]) > o.extension_speed_m_s+o.extension_acceleration_m_s2*dt+1e-10:
                    return f'{member.id}: positioner_speed_exceeded'
                if row['compression'] > o.max_compression_m:
                    return f'{member.id}: pad_compression_exceeded'
                if np.linalg.norm(row['force']) > o.positioner_force_limit_n or abs(row['positioner_power']) > o.positioner_power_limit_w:
                    return f'{member.id}: positioner_force_or_power_exceeded'
                if row['activation'] > 0:
                    fn = float(row['force_patch']@normals[k])
                    shear = np.linalg.norm(row['force_patch']-fn*normals[k])
                    if (max(0., -fn) > row['activation']*o.tensile_limit_n or
                        shear > row['activation']*o.shear_limit_n+o.friction_coefficient*row['normal_force'] or
                        np.linalg.norm(row['couple']) > row['activation']*o.moment_limit_nm):
                        return f'{member.id}: adhesive_capacity_exceeded'
                com = rt.T@(b[j, :3]-b[0, :3])
                relative_speed = np.linalg.norm(b[j, 3:6]-b[0, 3:6]-np.cross(wt, b[j, :3]-b[0, :3]))
                # Both endpoint and an integration-step travel reserve are
                # checked. This is a declared velocity/acceleration sweep bound.
                travel = relative_speed*dt+.5*dt**2*(np.linalg.norm(diag['acceleration'][j]-diag['acceleration'][0])+
                                                   2*np.linalg.norm(wt)*relative_speed)
                clearance = (geometry.signed_distance(com) if hasattr(geometry, 'components') else cylinder_distance(com))-member.properties.body_radius_m-travel
                minima['bus_component_clearance_m'] = min(minima['bus_component_clearance_m'], clearance)
                if clearance < o.minimum_bus_clearance_m:
                    return f'{member.id}: bus_component_clearance_failed'
                tip_body = rt.T@(row['tip']-b[0, :3])
                contact_point = tip_body-row['gap']*normals[k]
                if hasattr(geometry, 'components'):
                    patch = geometry.attachment_patch(member.patch_id)
                    delta = contact_point-np.asarray(patch.point_body_m)
                    margin = min(patch.half_extents_m[0]-patch.margin_m-abs(delta@patch.tangent_u_body),
                                 patch.half_extents_m[1]-patch.margin_m-abs(delta@patch.tangent_v_body))-o.pad_radius_m-travel
                    minima['patch_edge_margin_m'] = min(minima['patch_edge_margin_m'], float(margin))
                    if margin < 0:
                        return f'{member.id}: pad_footprint_left_patch'
                arm_vector = row['tip']-b[j, :3]
                start = b[j, :3]+arm_vector/np.linalg.norm(arm_vector)*member.properties.body_radius_m
                # The thin pad is the only selected-face contact exception.
                end = row['tip']+rt@normals[k]*o.positioner_radius_m
                segments.append((start, end))
                mid_body = rt.T@((start+end)/2-b[0, :3])
                half = np.linalg.norm(end-start)/2
                if hasattr(geometry, 'components'):
                    patch = geometry.attachment_patch(member.patch_id)
                    for component in geometry.components:
                        if component.id == patch.component_id:
                            continue
                        margin = component.signed_distance(mid_body)-half-o.positioner_radius_m-travel
                        pad_margin = component.signed_distance(tip_body)-o.pad_radius_m-travel
                        margin = min(margin, pad_margin)
                        minima['positioner_component_clearance_m'] = min(minima['positioner_component_clearance_m'], margin)
                        if margin <= 0:
                            return f'{member.id}: positioner_obstructed_by_{component.id}'
            for k, member in enumerate(members):
                j = k+1
                for other in range(k+1, count):
                    right = other+1
                    margin = np.linalg.norm(b[j, :3]-b[right, :3])-member.properties.body_radius_m-members[other].properties.body_radius_m
                    margin -= np.linalg.norm(b[j, 3:6]-b[right, 3:6])*dt+.5*dt**2*np.linalg.norm(diag['acceleration'][j]-diag['acceleration'][right])
                    minima['peer_bus_clearance_m'] = min(minima['peer_bus_clearance_m'], float(margin))
                    if margin < o.minimum_bus_clearance_m:
                        return f'{member.id}|{members[other].id}: peer_bus_clearance_failed'
                start, end = segments[k]
                direction = end-start
                for other, peer in enumerate(members):
                    if other == k:
                        continue
                    p = b[other+1, :3]
                    closest = start+np.clip((p-start)@direction/(direction@direction), 0., 1.)*direction
                    travel = (np.linalg.norm(diag['rows'][k]['tip_velocity']-b[other+1, 3:6])+o.extension_speed_m_s)*dt
                    margin = np.linalg.norm(p-closest)-peer.properties.body_radius_m-max(o.pad_radius_m, o.positioner_radius_m)-travel
                    minima['positioner_peer_clearance_m'] = min(minima['positioner_peer_clearance_m'], float(margin))
                    if margin <= 0:
                        return f'{member.id}: positioner_obstructed_by_peer_{peer.id}'
            return None

        def record(t, values, diag, force=False):
            if history and not force and started+t-history[-1]['time_s'] < o.history_step_s-1e-10:
                return
            view, tips = snapshot(t, values), unpack(values)[1]
            history.append(dict(time_s=started+t, target=view['target'].to_dict(),
                chasers={key: state.to_dict() for key, state in view['chasers'].items()},
                pads={member.id: dict(extension_body_m=tips[k, :3].tolist(), extension_velocity_body_m_s=tips[k, 3:].tolist(),
                    gap_m=row['gap'], force_eci_n=row['force'].tolist(), moment_eci_nm=row['couple'].tolist(),
                    activation=row['activation'], relative_speed_m_s=float(np.linalg.norm(row['u'])),
                    relative_spin_rad_s=float(np.linalg.norm(row['spin'])),
                    tip_position_eci_m=(values[:3]+row['tip']).tolist())
                    for k, (member, row) in enumerate(zip(members, diag['rows']))},
                controls_body={member.id: controls[k+1].tolist() for k, member in enumerate(members)},
                ledger=values[ledger_start:].tolist(), kinetic_energy_j=diag['kinetic'], contact_potential_j=diag['potential']))

        _, original = evaluate(0., y)
        t, dwell, lock_energy = 0., 0., np.inf

        def finish(success, message, diag):
            record(t, y, diag, force=not history or history[-1]['time_s'] != started+t)
            view, ledger = snapshot(t, y), y[ledger_start:]
            p_residual = diag['linear']-original['linear']-ledger[:3]-ledger[3:6]
            h_residual = diag['angular']-original['angular']-ledger[6:9]-ledger[9:12]
            expected_energy = ledger[12]+ledger[13]+ledger[14]+ledger[15]-sum(ledger[16:19])
            energy_residual = diag['kinetic']+diag['potential']-original['kinetic']-original['potential']-expected_energy
            scale = max(1e-12, abs(diag['kinetic']-original['kinetic']), abs(expected_energy), sum(abs(ledger[12:])))
            momentum_ok = (np.linalg.norm(p_residual) <= o.momentum_absolute_tolerance+o.momentum_relative_tolerance*max(np.linalg.norm(original['linear']), np.linalg.norm(ledger[:3])) and
                np.linalg.norm(h_residual) <= o.momentum_absolute_tolerance+o.momentum_relative_tolerance*max(np.linalg.norm(original['angular']), np.linalg.norm(ledger[6:9])))
            energy_ok = abs(energy_residual) <= max(o.energy_absolute_tolerance_j, o.energy_relative_tolerance*scale)
            if success and not (momentum_ok and energy_ok):
                success, message = False, 'finite_contact_balance_tolerance_failed'
            for k, member in enumerate(members):
                row = diag['rows'][k]
                contact_records[member.id] = dict(event='finite_compliant_adhesive_capture',
                    time_s=started+t, first_contact_time_s=None if np.isnan(first_touch[k]) else started+first_touch[k],
                    first_contact_relative_speed_m_s=None if np.isnan(first_touch_speed[k]) else float(first_touch_speed[k]),
                    contact_event_time_resolution_s=o.max_step_s,
                    engagement_start_time_s=None if not np.isfinite(engagement[k]) else started+engagement[k],
                    latch_time_s=started+t if success else None, target_docking_fixture_required=False,
                    surface_point_body_m=(anchors[k]+row['e']-(row['e']@normals[k])*normals[k]).tolist(),
                    tip_at_contact_body_m=(anchors[k]+row['e']).tolist(),
                    contact_residual_m=abs(row['gap']), available_tip_reach_m=member.arm_reach_m,
                    target_interface='representative adhesive pad on unprepared declared structural patch',
                    tensile_limit_n=o.tensile_limit_n, shear_limit_n=o.shear_limit_n, moment_limit_nm=o.moment_limit_nm)
            finite_minima = {key: None if not np.isfinite(value) else float(value) for key, value in minima.items()}
            audit = dict(model='finite_coupled_rigid_body_compliant_adhesive.v1', options=asdict(o),
                elapsed_s=t, integrated_body_count=n, chaser_count=count, peer_pair_count=count*(count-1)//2,
                occupied_component_count=len(geometry.components) if hasattr(geometry, 'components') else 1,
                geometry_scope='All occupied components, finite buses, pads/positioners and peer buses checked at internal steps with velocity/acceleration travel reserves; no interval continuous-contact enclosure',
                observed_settling_dwell_s=dwell, final_rigidization_energy_j=float(lock_energy) if np.isfinite(lock_energy) else None,
                extrema=maxima, minimum_clearances=finite_minima,
                linear_momentum_balance_residual_kg_m_s=float(np.linalg.norm(p_residual)),
                angular_momentum_balance_residual_kg_m2_s=float(np.linalg.norm(h_residual)),
                energy_balance_residual_j=float(energy_residual), energy_balance_scale_j=float(scale),
                momentum_balance_passed=bool(momentum_ok), energy_balance_passed=bool(energy_ok),
                external_linear_impulse_kg_m_s=ledger[:3].tolist(), fuel_linear_flux_kg_m_s=ledger[3:6].tolist(),
                external_angular_impulse_kg_m2_s=ledger[6:9].tolist(), fuel_angular_flux_kg_m2_s=ledger[9:12].tolist(),
                external_work_j=float(ledger[12]), fuel_energy_flux_j=float(ledger[13]),
                positioner_work_j=float(ledger[14]), engagement_work_j=float(ledger[15]),
                compression_dissipation_j=float(ledger[16]), friction_dissipation_j=float(ledger[17]),
                bond_dissipation_j=float(ledger[18]), initial_kinetic_energy_j=original['kinetic'],
                final_kinetic_energy_j=diag['kinetic'], final_contact_potential_j=diag['potential'],
                fuel_used_kg={member.id: float(member.initial_state.fuel_kg-view['chasers'][member.id].fuel_kg) for member in members},
                saturated_control_steps=saturation_count,
                balance_frame='translating nonrotating central-gravity reference; differential gravity includes the frame acceleration; fuel carries local velocity and specific inertia',
                environment_scope='injected extra accelerations beyond central gravity and extra body torques beyond gravity gradient; included in external wrench/work ledgers' if environment else 'central gravity and gravity gradient only',
                environment_contract=environment.as_dict() if callable(getattr(environment, 'as_dict', None)) else None,
                feedback_scope='injected estimated relative navigation; physical plant retained' if feedback else 'exact physical feedback assumption',
                scope='Representative finite compliant pads and massless finite-bandwidth positioners, rigid hull and bus inertia; no mechanism material, flexible appendage, plume or flight qualification',
                reference='https://ntrs.nasa.gov/citations/19870016368')
            return FiniteContactResult(success, message, view['target'], view['chasers'], tuple(history), contact_records, audit)

        record(0., y, original)
        try:
            while t < o.max_duration_s-1e-12:
                dt = min(o.max_step_s, o.max_duration_s-t)
                _, diag = evaluate(t, y)
                reason = guard(t, y, diag, dt)
                if reason:
                    return finish(False, reason, diag)
                command(t, y, diag)
                for k, row in enumerate(diag['rows']):
                    if row['compression'] > 0 and np.isnan(first_touch[k]):
                        first_touch[k] = t
                        first_touch_speed[k] = np.linalg.norm(row['u'])
                    if not np.isfinite(engagement[k]):
                        preload[k] = preload[k]+dt if (sensed_normal_force[k] >= .8*o.preload_n and
                            sensed_gap[k] <= o.range_error_bound_m and row['compression'] > 0) else 0.
                        if preload[k] >= o.preload_dwell_s:
                            engagement[k], bond_rest[k], rotation_rest[k] = t, row['e'], row['relative_rotation']
                committed_values, committed_time = y.copy(), t
                a, _ = evaluate(t, y)
                b, _ = evaluate(t+dt/2, y+dt*a/2)
                c, _ = evaluate(t+dt/2, y+dt*b/2)
                d, _ = evaluate(t+dt, y+dt*c)
                y += dt*(a+2*b+2*c+d)/6
                t += dt
                body, tips = unpack(y)
                if not np.all(np.isfinite(y)) or np.any(body[:, 13] < -1e-12):
                    y, t = committed_values, committed_time
                    _, diag = evaluate(t, y)
                    return finish(False, 'finite_contact_nonfinite_state_or_fuel_exhausted', diag)
                norms = np.linalg.norm(body[:, 6:10], axis=1)
                maxima['quaternion_norm_error'] = max(maxima['quaternion_norm_error'], float(np.max(abs(norms-1))))
                body[:, 6:10] /= norms[:, None]
                _, diag = evaluate(t, y)
                maxima['internal_force_residual_n'] = max(maxima['internal_force_residual_n'], diag['internal_force'])
                maxima['internal_torque_residual_nm'] = max(maxima['internal_torque_residual_nm'], diag['internal_torque'])
                reason = guard(t, y, diag, dt)
                if reason:
                    return finish(False, reason, diag)
                mass, rotations = diag['masses'], diag['rotations']
                center = mass@body[:, :3]/sum(mass)
                velocity = mass@body[:, 3:6]/sum(mass)
                offsets = body[:, :3]-center
                inertia_world = sum(rotations[k]@diag['inertias'][k]@rotations[k].T+
                    mass[k]*((offsets[k]@offsets[k])*np.eye(3)-np.outer(offsets[k], offsets[k])) for k in range(n))
                angular_com = diag['angular']-np.cross(center, diag['linear'])
                rigid_spin = np.linalg.solve(inertia_world, angular_com)
                residual_velocity = body[:, 3:6]-velocity-np.cross(rigid_spin, offsets)
                residual_spin = diag['spins']-rigid_spin
                lock_energy = .5*float(np.sum(mass*np.sum(residual_velocity**2, axis=1))+
                    sum(residual_spin[k]@(rotations[k]@diag['inertias'][k]@rotations[k].T)@residual_spin[k] for k in range(n)))
                quiet = (all(row['activation'] >= 1. and np.linalg.norm(row['u']) <= o.settling_speed_m_s and
                    np.linalg.norm(row['spin']) <= o.settling_spin_rad_s for row in diag['rows']) and
                    np.max(np.linalg.norm(tips[:, 3:], axis=1)) <= o.settling_speed_m_s/4 and
                    np.max(np.linalg.norm(residual_velocity, axis=1)) <= o.settling_speed_m_s and
                    np.max(np.linalg.norm(residual_spin, axis=1)) <= o.settling_spin_rad_s and lock_energy <= o.max_lock_energy_j)
                dwell = dwell+dt if quiet else 0.
                record(t, y, diag)
                if dwell >= o.settling_dwell_s:
                    return finish(True, 'finite_contact_settled_latch_completed', diag)
            return finish(False, 'finite_contact_duration_limit_before_settled_latch', diag)
        except (ValueError, RuntimeError, ArithmeticError) as error:
            _, diag = evaluate(t, y)
            return finish(False, f'finite_contact_rejected: {type(error).__name__}: {error}', diag)
