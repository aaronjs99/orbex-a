"""Unactuated rigid-body attitude with the supported cylinder collision envelope.

Physical body rates describe rotation relative to an inertial frame. The public
target rotation maps body coordinates into the rotating mission LVLH frame.
Quaternions are scalar-last; Euler angles are only a compatibility chart.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple, TYPE_CHECKING

import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

from orbexa.control.constraints import CylinderConstraint
from orbexa.core.attitude import (
    body_rate_derivative, body_rates_to_euler_rates, euler_to_quaternion,
    finite_vector, inertia_matrix, quaternion_to_euler,
)
from orbexa.core.target_geometry import TumblingCylinderTarget
from orbexa.core.rigid_body_motion import RigidBodyMotion
from orbexa.utils.anomaly import orbital_radius_terms
from orbexa.utils.math_utils import gen_skew_sym_mat

if TYPE_CHECKING:
    from orbexa.core.config import SimulationConfig


@dataclass(frozen=True)
class RigidBodyCylinderTarget(RigidBodyMotion):
    """Unactuated rigid body, queried on one declared anomaly interval.

    ``initial_quaternion`` maps body to LVLH at ``anomaly_domain[0]``. An
    inertial basis is fixed coincident to LVLH at that instant. Initial body
    angular velocity is relative to that inertial basis, in rad/s; it is never
    inferred from the legacy prescribed Euler/anomaly rates. Inertia is a
    body-frame SPD matrix in kg m^2 with physically realizable principal
    moments, independent of the cylinder envelope.

    One DOP853 dense trajectory is built per instance. Queries never extend,
    restart or mutate it. The default is torque-free. ``gravity_gradient=True``
    includes central-field gravity-gradient torque using the current orbital
    radius and attitude. Neither mode models attitude-control, aerodynamic,
    magnetic or contact torque. The planar orbital frame rotates by the
    unwrapped anomaly increment about its positive orbit-normal axis.
    """

    radius: float
    half_length: float
    initial_quaternion: np.ndarray
    initial_body_angular_velocity: np.ndarray
    inertia: np.ndarray
    eccentricity: float
    alpha: float
    mu: float
    semi_major_axis: float
    anomaly_domain: Tuple[float, float] = (0.0, 2.0 * np.pi)
    tolerance: float = 0.0
    docking_standoff: float = 0.12
    integration_rtol: float = 1.0e-11
    integration_atol: float = 1.0e-13
    integration_max_step: float = 0.05
    gravity_gradient: bool = False
    _geometry: TumblingCylinderTarget = field(init=False, repr=False, compare=False)
    _solution: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._initialize_motion()
        geometry = TumblingCylinderTarget(
            radius=self.radius, half_length=self.half_length,
            initial_orientation=quaternion_to_euler(self.initial_quaternion), angular_velocity=np.zeros(3),
            tolerance=self.tolerance, docking_standoff=self.docking_standoff,
        )
        geometry.initial_orientation.setflags(write=False)
        geometry.angular_velocity.setflags(write=False)
        object.__setattr__(self, "_geometry", geometry)

    @classmethod
    def from_config(cls, config: SimulationConfig, *, initial_body_angular_velocity,
                    initial_quaternion=None, inertia=None, anomaly_domain=None,
                    gravity_gradient: bool = False,
                    **integration_options) -> "RigidBodyCylinderTarget":
        """Use configured geometry/clock and an explicitly supplied rad/s spin.

        The default domain includes every configured update plus its longest
        prediction horizon. Pass a larger domain for a longer run. The legacy
        ``target.initial_angular_velocity`` is deliberately not converted.
        """
        geometry = TumblingCylinderTarget.from_config(config)
        if anomaly_domain is None:
            end = max(config.total_anom, config.iter_anomaly) + max(config.mpc.horizon_steps.values()) * config.anom_step
            anomaly_domain = (0.0, end)
        return cls(
            radius=geometry.radius, half_length=geometry.half_length,
            initial_quaternion=(euler_to_quaternion(geometry.initial_orientation)
                                if initial_quaternion is None else initial_quaternion),
            initial_body_angular_velocity=initial_body_angular_velocity,
            inertia=config.target.inertia if inertia is None else inertia,
            eccentricity=config.orbit.eccentricity, alpha=config.orbit.target_drag,
            mu=config.orbit.mu, semi_major_axis=config.orbit.semi_major_axis,
            anomaly_domain=anomaly_domain, tolerance=geometry.tolerance,
            docking_standoff=geometry.docking_standoff, gravity_gradient=gravity_gradient,
            **integration_options,
        )

















    @property
    def initial_orientation(self) -> np.ndarray:
        return self._geometry.initial_orientation.copy()

    @property
    def height(self) -> float:
        return self._geometry.height

    @property
    def bounding_sphere_radius(self) -> float:
        return self._geometry.bounding_sphere_radius

    @property
    def rendezvous_radius(self) -> float:
        return self._geometry.rendezvous_radius

    def approach_azimuth(self, initial_state: np.ndarray) -> float:
        return self._geometry.approach_azimuth(initial_state)

    def approach_position_body(self, initial_state: np.ndarray) -> np.ndarray:
        return self._geometry.approach_position_body(initial_state)

    def approach_axial_coordinate(self, initial_state: np.ndarray) -> float:
        return self._geometry.approach_axial_coordinate(initial_state)

    def side_docking_geometry(self, azimuth: float, axial: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        return self._geometry.side_docking_geometry(azimuth, axial)

    def docking_points_body(self, count: int, start_azimuth: float = 0.0) -> List[np.ndarray]:
        return self._geometry.docking_points_body(count, start_azimuth)

    def assign_docking_targets(self, initial_states: Sequence[np.ndarray]) -> List[Dict[str, Any]]:
        return self._geometry.assign_docking_targets(initial_states)



    def collision_params(self, operation: str, anom: float) -> Dict[str, Any]:
        values = self._geometry.collision_params(operation, 0.0)
        values.pop("angular_velocity")
        values.update(orientation=self.orientation_at(anom).tolist(),
                      rotation_matrix=self.rotation_matrix_at(anom).tolist(),
                      attitude_model=self.attitude_model)
        return values

    def cylinder_constraint(self, anom: float, tube_radius: float = 0.0) -> CylinderConstraint:
        return CylinderConstraint(radius=self.radius * (1.0 + self.tolerance),
                                  half_length=self.half_length * (1.0 + self.tolerance),
                                  orientation=self.orientation_at(anom), tube_radius=tube_radius)


    def as_dict(self) -> Dict[str, Any]:
        values = self._geometry.as_dict()
        values.pop("angular_velocity")
        values.update(
            attitude_model=self.attitude_model, motion_model=self.attitude_model,
            gravity_gradient=self.gravity_gradient,
            external_torque_model=("central_spherical_gravity_gradient" if self.gravity_gradient else "zero"),
            collision_geometry="cylinder_envelope", attitude_reference_frame="body_to_LVLH",
            quaternion_order="xyzw", initial_quaternion=self.initial_quaternion.tolist(),
            initial_body_angular_velocity_rad_s=self.initial_body_angular_velocity.tolist(),
            inertia_kg_m2=self.inertia.tolist(), anomaly_domain=list(self.anomaly_domain),
            principal_moments_kg_m2=np.linalg.eigvalsh(self.inertia).tolist(),
            inertia_triangle_relative_roundoff_tolerance=64.0 * np.finfo(float).eps,
            clock_model="analytical_inverse_radius_quadratic_drag",
            clock_parameters=dict(eccentricity=float(self.eccentricity), alpha=float(self.alpha),
                                  mu=float(self.mu), semi_major_axis=float(self.semi_major_axis), theta0=0.0),
            inertial_reference="fixed coincident to LVLH at the first anomaly",
            lvlh_frame_rotation="positive anomaly increment about orbit normal",
            integration=dict(method="DOP853", rtol=float(self.integration_rtol),
                             atol=float(self.integration_atol), max_anomaly_step=float(self.integration_max_step)),
            assumptions=["rigid body with constant physically realizable body-frame inertia",
                         ("central-field gravity-gradient torque only; no target attitude control"
                          if self.gravity_gradient else "zero net external torque"),
                         "planar analytical orbit; no plane precession",
                         "no aerodynamic, magnetic, solar-radiation or contact torque",
                         "known target attitude and physical spin; no attitude-estimation uncertainty",
                         "cylinder collision envelope, not a validated spacecraft asset"],
        )
        return values
