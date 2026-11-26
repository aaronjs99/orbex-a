"""state.py: Packed physical coordinates and explicit mutable contact/actuator state.

The six leading values track the actual target COM in nonrotating ECI axes.
Each body then stores local position/velocity, quaternion, body spin and fuel;
each tip stores extension/velocity. The final 19 values are balance ledgers.

Copyright: Aaron John Sabu
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from orbexa.mission.attached.assembly.model import physical_inertia, positive
from orbexa.mission.contact.contracts import ContactBodyState
from scipy.spatial.transform import Rotation


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
    def create(
        cls,
        *,
        target_properties,
        target_state,
        members,
        mu_m3_s2,
        gravity_gradient=False,
    ):
        """Build the fleet layout and packed initial state.

        Inputs: `mu_m3_s2` is the gravitational parameter (m^3/s^2). The packed
        state starts with the target COM position and velocity in ECI, then
        per-body local position, velocity, quaternion, spin and fuel, then tip
        extension and velocity, then 19 zeroed ledger values.
        Raises ValueError for an empty or duplicate-id fleet, for initial clocks
        that differ from the target clock, or for a nonpositive `mu_m3_s2`."""
        members = tuple(members)
        if not members or len({m.id for m in members}) != len(members):
            raise ValueError(
                "Finite contact requires a nonempty uniquely identified fleet"
            )
        if any(m.initial_state.time_s != target_state.time_s for m in members):
            raise ValueError(
                "All finite contact bodies must share the exact initial physical clock"
            )
        mu = positive(mu_m3_s2, "Contact gravitational parameter")
        n, count = len(members) + 1, len(members)
        initial = (target_state,) + tuple(m.initial_state for m in members)
        dry = np.array(
            [target_properties.mass_kg] + [m.properties.dry_mass_kg for m in members]
        )
        specific = np.array(
            [physical_inertia(target_properties.inertia_kg_m2) / dry[0]]
            + [m.properties.inertia_per_kg_m2 for m in members]
        )
        mounts = np.array([[-m.properties.gripper_offset_m, 0.0, 0.0] for m in members])
        anchors = np.array([m.surface_anchor_body_m for m in members])
        normals = np.array([m.normal_body for m in members])
        # The target-following nonrotating origin avoids subtracting orbital-scale
        # kinetic energies and momenta to resolve millimeter contact effects.
        body_slice, tip_start = slice(6, 6 + 14 * n), 6 + 14 * n
        tip_slice, ledger_start = (
            slice(tip_start, tip_start + 6 * count),
            tip_start + 6 * count,
        )
        y = np.zeros(ledger_start + 19)
        y[:6] = np.r_[target_state.position_eci_m, target_state.velocity_eci_m_s]
        bodies = y[body_slice].reshape(n, 14)
        for index, state in enumerate(initial):
            bodies[index] = np.r_[
                state.position_eci_m - y[:3],
                state.velocity_eci_m_s - y[3:6],
                state.quaternion_body_to_eci,
                state.angular_velocity_body_rad_s,
                state.fuel_kg,
            ]
        started = target_state.time_s
        return cls(
            target_properties,
            members,
            mu,
            gravity_gradient,
            dry,
            specific,
            mounts,
            anchors,
            normals,
            started,
            body_slice,
            tip_slice,
            ledger_start,
            y,
        )

    @property
    def body_count(self):
        """Return the number of integrated bodies: the target plus every chaser."""
        return len(self.members) + 1

    @property
    def chaser_count(self):
        """Return the number of chasers."""
        return len(self.members)

    def gravity(self, position):
        """Return the central-gravity acceleration at an ECI position.

        Inputs: `position` in m. Outputs: acceleration in m/s^2.
        Raises ValueError if the radius is not finite and positive."""
        mu = self.mu
        radius = np.linalg.norm(position)
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("Contact orbital position must be finite and nonzero")
        return -mu * position / radius**3

    def unpack(self, values):
        """Return writable views of the body and tip blocks of a packed state.

        Outputs: `(bodies, tips)` shaped (body_count, 14) and (chaser_count, 6)."""
        return (
            values[self.body_slice].reshape(self.body_count, 14),
            values[self.tip_slice].reshape(self.chaser_count, 6),
        )

    def snapshot(self, t, values, diag=None):
        """Return the physical observation dictionary for a packed state.

        Inputs: `t` is elapsed time (s); `diag`, if given, supplies pad gaps and
        contact wrenches, otherwise gaps are computed from geometry and wrenches
        are zero.
        Outputs: a dictionary with target and chaser `ContactBodyState` values in
        ECI at clock `started + t`, positioner encoder products, per-chaser
        contact wrenches in ECI and contact gaps (m)."""
        started, members = self.started, self.members
        mounts, anchors, normals = self.mounts, self.anchors, self.normals
        unpack = self.unpack
        b, _ = unpack(values)
        states = tuple(
            ContactBodyState(
                started + t,
                values[:3] + row[:3],
                values[3:6] + row[3:6],
                row[6:10],
                row[10:13],
                max(0.0, row[13]),
            )
            for row in b
        )
        gaps = {}
        if diag is None:
            rotations = Rotation.from_quat(b[:, 6:10]).as_matrix()
            tips = unpack(values)[1]
            for i, member in enumerate(members):
                tip = b[i + 1, :3] + rotations[i + 1] @ (mounts[i] + tips[i, :3])
                patch = b[0, :3] + rotations[0] @ anchors[i]
                gaps[member.id] = float((tip - patch) @ (rotations[0] @ normals[i]))
        else:
            gaps = {m.id: diag["rows"][i]["gap"] for i, m in enumerate(members)}
        return {
            "target": states[0],
            "chasers": {m.id: states[i + 1] for i, m in enumerate(members)},
            "positioner_encoders": {
                m.id: {
                    "extension_body_m": unpack(values)[1][i, :3].tolist(),
                    "extension_velocity_body_m_s": unpack(values)[1][i, 3:].tolist(),
                }
                for i, m in enumerate(members)
            },
            "contact_wrenches_eci": {
                m.id: {
                    "force_n": (
                        np.zeros(3) if diag is None else diag["rows"][i]["force"]
                    ).tolist(),
                    "couple_nm": (
                        np.zeros(3) if diag is None else diag["rows"][i]["couple"]
                    ).tolist(),
                }
                for i, m in enumerate(members)
            },
            "contact_gaps_m": gaps,
        }

    def check_observation(self, observation, clock):
        """Validate an observation against the fleet and a physical clock.

        Raises ValueError if the chaser ids differ from the fleet ids, or if any
        state is not a `ContactBodyState` stamped within 1e-8 s of `clock`."""
        members = self.members
        if set(observation["chasers"]) != {member.id for member in members}:
            raise ValueError("Contact observations must name the exact fleet")
        for state in (observation["target"], *observation["chasers"].values()):
            if (
                not isinstance(state, ContactBodyState)
                or abs(state.time_s - clock) > 1e-8
            ):
                raise ValueError(
                    "Contact observations require current-clock ContactBodyState estimates"
                )


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
        """Return zeroed commands with unset gap and frozen-extension channels.

        The sensed gap is initialised to infinity and the frozen extension to NaN."""
        return cls(
            np.zeros((body_count, 6)),
            np.zeros((chaser_count, 3)),
            np.zeros(chaser_count),
            np.full(chaser_count, np.inf),
            np.full((chaser_count, 3), np.nan),
        )


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
        """Return bond state for `count` chasers with no engagement.

        Engagement times start at infinity and touch records at NaN."""
        return cls(
            np.full(count, np.inf),
            np.zeros(count),
            np.zeros((count, 3)),
            np.zeros((count, 3, 3)),
            np.full(count, np.nan),
            np.full(count, np.nan),
        )

    def advance_engagement(self, t, dt, diag, actuation, options):
        """Update first-touch records and start bond engagement after preload dwell.

        Inputs: `t` is elapsed time (s), `dt` the step (s), `diag` the mechanics
        diagnostics, `actuation` the sensed channels and `options` the contact
        options. Mutates `first_touch`, `first_touch_speed`, `preload`,
        `engagement`, `bond_rest` and `rotation_rest`. The preload timer counts
        steps where sensed normal force is at least 0.8 of `preload_n`, sensed gap
        is within `range_error_bound_m` and the pad is compressed; it resets
        otherwise. Engagement starts when it reaches `preload_dwell_s`."""
        o = options
        engagement, preload = self.engagement, self.preload
        bond_rest, rotation_rest = self.bond_rest, self.rotation_rest
        first_touch, first_touch_speed = self.first_touch, self.first_touch_speed
        sensed_normal_force, sensed_gap = (
            actuation.sensed_normal_force,
            actuation.sensed_gap,
        )
        for k, row in enumerate(diag["rows"]):
            if row["compression"] > 0 and np.isnan(first_touch[k]):
                first_touch[k] = t
                first_touch_speed[k] = np.linalg.norm(row["u"])
            if not np.isfinite(engagement[k]):
                preload[k] = (
                    preload[k] + dt
                    if (
                        sensed_normal_force[k] >= 0.8 * o.preload_n
                        and sensed_gap[k] <= o.range_error_bound_m
                        and row["compression"] > 0
                    )
                    else 0.0
                )
                if preload[k] >= o.preload_dwell_s:
                    engagement[k], bond_rest[k], rotation_rest[k] = (
                        t,
                        row["e"],
                        row["relative_rotation"],
                    )


def rk4_contact_step(evaluate, t, values, dt):
    """Advance the packed state with commands fixed at the step boundary."""
    a, _ = evaluate(t, values)
    b, _ = evaluate(t + dt / 2, values + dt * a / 2)
    c, _ = evaluate(t + dt / 2, values + dt * b / 2)
    d, _ = evaluate(t + dt, values + dt * c)
    values += dt * (a + 2 * b + 2 * c + d) / 6
