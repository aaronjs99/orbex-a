"""Packed physical coordinates and explicit mutable contact/actuator state.

The six leading values track the actual target COM in nonrotating ECI axes.
Each body then stores local position/velocity, quaternion, body spin and fuel;
each tip stores extension/velocity. The final 19 values are balance ledgers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
from scipy.spatial.transform import Rotation

from orbexa.simulation.attached.assembly.model import physical_inertia, positive
from orbexa.simulation.contact.contracts import ContactBodyState


@dataclass(frozen=True)
class ContactFleet:
    """Fixed physical inputs and packed-state layout for one contact interval."""
    target_properties: Any
    members: tuple
    mu: float
    gravity_gradient: bool
    dry: np.ndarray
    specific: np.ndarray
    mounts: np.ndarray
    anchors: np.ndarray
    normals: np.ndarray
    started: float
    body_slice: slice
    tip_slice: slice
    ledger_start: int
    initial_values: np.ndarray

    @classmethod
    def create(cls, *, target_properties, target_state, members, mu_m3_s2,
               gravity_gradient=False):
        members = tuple(members)
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
        # The target-following nonrotating origin avoids subtracting orbital-scale
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
        return cls(target_properties, members, mu, gravity_gradient, dry, specific,
                   mounts, anchors, normals, started, body_slice, tip_slice, ledger_start, y)

    @property
    def body_count(self):
        return len(self.members) + 1

    @property
    def chaser_count(self):
        return len(self.members)

    def gravity(self, position):
        mu = self.mu
        radius = np.linalg.norm(position)
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError('Contact orbital position must be finite and nonzero')
        return -mu * position / radius**3

    def unpack(self, values):
        return (values[self.body_slice].reshape(self.body_count, 14),
                values[self.tip_slice].reshape(self.chaser_count, 6))

    def snapshot(self, t, values, diag=None):
        started, members = self.started, self.members
        mounts, anchors, normals = self.mounts, self.anchors, self.normals
        unpack = self.unpack
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
            positioner_encoders={m.id: dict(extension_body_m=unpack(values)[1][i, :3].tolist(),
                extension_velocity_body_m_s=unpack(values)[1][i, 3:].tolist()) for i, m in enumerate(members)},
            contact_wrenches_eci={m.id: dict(force_n=(np.zeros(3) if diag is None else diag['rows'][i]['force']).tolist(),
                couple_nm=(np.zeros(3) if diag is None else diag['rows'][i]['couple']).tolist())
                for i, m in enumerate(members)},
            contact_gaps_m=gaps)

    def check_observation(self, observation, clock):
        members = self.members
        if set(observation['chasers']) != {member.id for member in members}:
            raise ValueError('Contact observations must name the exact fleet')
        for state in (observation['target'], *observation['chasers'].values()):
            if not isinstance(state, ContactBodyState) or abs(state.time_s-clock) > 1e-8:
                raise ValueError('Contact observations require current-clock ContactBodyState estimates')


@dataclass
class ContactActuation:
    """Commands and measured channels held constant during an RK4 step."""
    controls: np.ndarray
    extension_acceleration: np.ndarray
    sensed_normal_force: np.ndarray
    sensed_gap: np.ndarray
    frozen_extension: np.ndarray

    @classmethod
    def create(cls, body_count, chaser_count):
        return cls(np.zeros((body_count, 6)), np.zeros((chaser_count, 3)),
                   np.zeros(chaser_count), np.full(chaser_count, np.inf),
                   np.full((chaser_count, 3), np.nan))


@dataclass
class ContactBondState:
    """Touch/preload events and rest configurations of the growing bonds."""
    engagement: np.ndarray
    preload: np.ndarray
    bond_rest: np.ndarray
    rotation_rest: np.ndarray
    first_touch: np.ndarray
    first_touch_speed: np.ndarray

    @classmethod
    def create(cls, count):
        return cls(np.full(count, np.inf), np.zeros(count), np.zeros((count, 3)),
                   np.zeros((count, 3, 3)), np.full(count, np.nan), np.full(count, np.nan))

    def advance_engagement(self, t, dt, diag, actuation, options):
        o = options
        engagement, preload = self.engagement, self.preload
        bond_rest, rotation_rest = self.bond_rest, self.rotation_rest
        first_touch, first_touch_speed = self.first_touch, self.first_touch_speed
        sensed_normal_force, sensed_gap = actuation.sensed_normal_force, actuation.sensed_gap
        for k, row in enumerate(diag['rows']):
            if row['compression'] > 0 and np.isnan(first_touch[k]):
                first_touch[k] = t
                first_touch_speed[k] = np.linalg.norm(row['u'])
            if not np.isfinite(engagement[k]):
                preload[k] = preload[k]+dt if (sensed_normal_force[k] >= .8*o.preload_n and
                    sensed_gap[k] <= o.range_error_bound_m and row['compression'] > 0) else 0.
                if preload[k] >= o.preload_dwell_s:
                    engagement[k], bond_rest[k], rotation_rest[k] = t, row['e'], row['relative_rotation']


def rk4_contact_step(evaluate, t, values, dt):
    """Advance the packed state with commands fixed at the step boundary."""
    a, _ = evaluate(t, values)
    b, _ = evaluate(t+dt/2, values+dt*a/2)
    c, _ = evaluate(t+dt/2, values+dt*b/2)
    d, _ = evaluate(t+dt, values+dt*c)
    values += dt*(a+2*b+2*c+d)/6
