"""Coupled rigid-body derivatives, compliant pad laws and balance ledgers."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from orbexa.core.attitude import finite_vector
from orbexa.simulation.assembly import STANDARD_GRAVITY


class ContactMechanics:
    """Physical plant depending only on fleet, pad options, bond and actuator state."""

    def __init__(self, *, fleet, options, bonds, actuation, environment=None):
        self.fleet = fleet
        self.options = options
        self.bonds = bonds
        self.actuation = actuation
        self.environment = environment

    def evaluate(self, t, values):
        fleet, o = self.fleet, self.options
        members, n = fleet.members, fleet.body_count
        dry, specific = fleet.dry, fleet.specific
        mounts, anchors, normals = fleet.mounts, fleet.anchors, fleet.normals
        mu, gravity_gradient = fleet.mu, fleet.gravity_gradient
        started, ledger_start = fleet.started, fleet.ledger_start
        unpack, snapshot, gravity = fleet.unpack, fleet.snapshot, fleet.gravity
        controls = self.actuation.controls
        extension_acceleration = self.actuation.extension_acceleration
        engagement = self.bonds.engagement
        bond_rest, rotation_rest = self.bonds.bond_rest, self.bonds.rotation_rest
        environment = self.environment
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
        # Follow the actual target translation. Its common inertial position
        # and velocity live in the six leading coordinates; target-local
        # translation stays zero by definition. This retains shared target
        # motion instead of independently boxing it a second time in every
        # contact gap. Account for the accelerating origin in all local
        # momentum, angular-momentum and energy ledgers.
        frame_increment = force[0]/mass[0]
        reference_acceleration = reference_acceleration+frame_increment
        external_force -= mass[:, None]*frame_increment
        force -= mass[:, None]*frame_increment
        force[0] = 0.
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

    def settling_state(self, values, diag):
        """Return rigidization energy and the original all-body quiet condition."""
        o, n = self.options, self.fleet.body_count
        body, tips = self.fleet.unpack(values)
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
        return lock_energy, quiet
