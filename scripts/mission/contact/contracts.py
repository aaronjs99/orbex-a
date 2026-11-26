"""contracts.py: Public SI contracts for finite compliant contact, independent of execution.

Copyright: Aaron John Sabu
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from orbexa.mission.attached.assembly.model import positive
from orbexa.models.dynamics.attitude import finite_vector


@dataclass(frozen=True)
class FiniteContactOptions:
    """Frozen SI options for the finite compliant contact stage.

    Inputs: lengths in m, speeds in m/s, accelerations in m/s^2, times in s,
    forces in N, moments in N m, stiffnesses in N/m or N m/rad, dampings in
    N s/m or N m s/rad, energy in J, power in W, frequencies in rad/s.
    """

    handoff_rule: str = "load_bounded_finite_lock"
    max_duration_s: float = 60.0
    max_step_s: float = 0.005
    history_step_s: float = 0.05
    extension_speed_m_s: float = 0.020
    extension_acceleration_m_s2: float = 0.020
    positioner_time_constant_s: float = 0.15
    touch_speed_m_s: float = 0.001
    slow_touch_distance_m: float = 0.002
    surface_search_depth_m: float = 0.020
    range_error_bound_m: float = 0.002
    pad_radius_m: float = 0.025
    positioner_radius_m: float = 0.005
    compression_stiffness_n_m: float = 2000.0
    compression_damping_n_s_m: float = 700.0
    max_compression_m: float = 0.003
    preload_n: float = 1.0
    preload_command_factor: float = 1.5
    preload_dwell_s: float = 0.5
    friction_coefficient: float = 0.5
    friction_regularization_m_s: float = 0.001
    bond_stiffness_n_m: float = 20000.0
    bond_damping_n_s_m: float = 1000.0
    bond_rotational_stiffness_nm_rad: float = 500.0
    bond_rotational_damping_nm_s_rad: float = 300.0
    bond_engagement_s: float = 1.0
    tensile_limit_n: float = 250.0
    shear_limit_n: float = 250.0
    moment_limit_nm: float = 5.0
    positioner_force_limit_n: float = 400.0
    positioner_power_limit_w: float = 20.0
    translation_force_n: float = 100.0
    attitude_nozzle_force_n: float = 3.0
    attitude_moment_arm_m: float = 0.4
    specific_impulse_s: float = 300.0
    stationkeeping_frequency_rad_s: float = 0.5
    attitude_frequency_rad_s: float = 1.0
    settling_dwell_s: float = 2.0
    settling_speed_m_s: float = 0.0001
    settling_spin_rad_s: float = 0.00001
    max_lock_energy_j: float = 0.0001
    lock_duration_s: float = 0.25
    minimum_bus_clearance_m: float = 0.001
    momentum_absolute_tolerance: float = 1e-7
    momentum_relative_tolerance: float = 1e-7
    energy_absolute_tolerance_j: float = 1e-5
    energy_relative_tolerance: float = 0.001

    def __post_init__(self):
        """Validate the options and coerce every numeric field to a float.

        Outputs: raises ValueError if handoff_rule is not load_bounded_finite_lock or
        settled_passive, a field is not finite and positive (friction_coefficient and
        minimum_bus_clearance_m may be zero), touch speed exceeds extension speed, the
        settling, preload and bond dwells do not fit the maximum duration, or preload
        does not fit within the stiffness times the maximum compression.
        """
        if self.handoff_rule not in ("load_bounded_finite_lock", "settled_passive"):
            raise ValueError(
                "Declare load_bounded_finite_lock or settled_passive contact handoff"
            )
        for name, value in vars(self).items():
            if name == "handoff_rule":
                continue
            object.__setattr__(
                self,
                name,
                positive(
                    value,
                    name,
                    allow_zero=name
                    in ("friction_coefficient", "minimum_bus_clearance_m"),
                ),
            )
        if self.touch_speed_m_s > self.extension_speed_m_s:
            raise ValueError("Touch speed cannot exceed free extension speed")
        if (
            self.settling_dwell_s + self.preload_dwell_s + self.bond_engagement_s
            >= self.max_duration_s
        ):
            raise ValueError("Finite contact dwell must fit the contact duration")
        if self.preload_n >= self.compression_stiffness_n_m * self.max_compression_m:
            raise ValueError("Preload must fit the allowed pad compression")


@dataclass(frozen=True)
class ContactBodyState:
    """Frozen rigid-body state at one contact time, expressed in ECI.

    Inputs: time_s in s, position_eci_m in m, velocity_eci_m_s in m/s,
    quaternion_body_to_eci unitless, angular_velocity_body_rad_s in rad/s in the
    body frame, fuel_kg in kg.
    """

    time_s: float
    position_eci_m: np.ndarray
    velocity_eci_m_s: np.ndarray
    quaternion_body_to_eci: np.ndarray
    angular_velocity_body_rad_s: np.ndarray
    fuel_kg: float = 0.0

    def __post_init__(self):
        """Validate the state, normalize the quaternion and freeze the arrays.

        Outputs: raises ValueError if the time is not finite, a vector has the wrong
        size or a non-finite entry, the quaternion is zero, or fuel is negative.
        """
        object.__setattr__(self, "time_s", float(self.time_s))
        if not np.isfinite(self.time_s):
            raise ValueError("Contact time must be finite")
        for name, size in (
            ("position_eci_m", 3),
            ("velocity_eci_m_s", 3),
            ("quaternion_body_to_eci", 4),
            ("angular_velocity_body_rad_s", 3),
        ):
            value = finite_vector(getattr(self, name), name, size)
            if size == 4:
                norm = np.linalg.norm(value)
                if norm == 0:
                    raise ValueError("Contact quaternion cannot be zero")
                value /= norm
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(
            self, "fuel_kg", positive(self.fuel_kg, "Contact fuel", allow_zero=True)
        )

    def to_dict(self):
        """Return the fields as a dict with arrays converted to lists."""
        return {
            name: value.tolist() if isinstance(value, np.ndarray) else value
            for name, value in vars(self).items()
        }


@dataclass(frozen=True)
class ContactMember:
    """Frozen description of one chaser taking part in finite contact.

    Inputs: surface_anchor_body_m in m and the unit axes normal_body, tangent_u_body
    and tangent_v_body are expressed in the member body frame; arm_reach_m in m.
    """

    id: str
    properties: Any
    initial_state: ContactBodyState
    surface_anchor_body_m: np.ndarray
    normal_body: np.ndarray
    tangent_u_body: np.ndarray
    tangent_v_body: np.ndarray
    arm_reach_m: float = 0.1
    patch_id: str | None = None

    def __post_init__(self):
        """Validate the member and freeze its body-frame vectors.

        Outputs: raises ValueError if id is empty or not a string, a vector is not
        three finite values, the tangent and normal axes are not right-handed
        orthonormal, or arm_reach_m is not finite and positive.
        """
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("Contact member requires an id")
        for name in (
            "surface_anchor_body_m",
            "normal_body",
            "tangent_u_body",
            "tangent_v_body",
        ):
            value = finite_vector(getattr(self, name), name, 3)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        axes = np.column_stack(
            (self.tangent_u_body, self.tangent_v_body, self.normal_body)
        )
        if (
            not np.allclose(axes.T @ axes, np.eye(3), atol=1e-9, rtol=0)
            or np.linalg.det(axes) <= 0
        ):
            raise ValueError("Contact patch requires right-handed orthonormal axes")
        object.__setattr__(
            self, "arm_reach_m", positive(self.arm_reach_m, "Finite tip reach")
        )


@dataclass(frozen=True)
class FiniteContactResult:
    """Frozen outcome of a finite contact run.

    Inputs: success and message report the outcome; final_target and final_chasers
    hold the final body states; history, surface_contacts and audit hold the
    recorded data.
    """

    success: bool
    message: str
    final_target: ContactBodyState
    final_chasers: Mapping[str, ContactBodyState]
    history: tuple[dict, ...]
    surface_contacts: Mapping[str, dict]
    audit: dict

    def to_dict(self):
        """Return the result as a dict with body states converted to dicts."""
        return {
            "success": self.success,
            "message": self.message,
            "final_target": self.final_target.to_dict(),
            "final_chasers": {
                name: state.to_dict() for name, state in self.final_chasers.items()
            },
            "history": list(self.history),
            "surface_contacts": dict(self.surface_contacts),
            "audit": self.audit,
        }
