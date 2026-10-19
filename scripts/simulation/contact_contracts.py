"""Public SI contracts for finite compliant contact, independent of execution."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional
import numpy as np

from orbexa.core.attitude import finite_vector
from orbexa.simulation.assembly import positive


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
