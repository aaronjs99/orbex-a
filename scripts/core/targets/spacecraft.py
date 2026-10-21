"""Rigid spacecraft motion with explicit component solids and surface patches."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence
import numpy as np

from orbexa.core.dynamics.attitude import euler_to_quaternion, quaternion_to_euler
from orbexa.core.dynamics.rigid_motion import RigidBodyMotion
from orbexa.core.geometry.composite import SpacecraftGeometry


@dataclass(frozen=True)
class RigidBodySpacecraftTarget(RigidBodyMotion):
    """Unactuated quaternion/Euler target independent of a collision primitive.

    Every occupied component participates in exclusion. The bounding radius is
    only a camera/deployment scale; it never replaces collision geometry.
    Mass and inertia are explicit study inputs, not identified flight truth.
    """

    geometry: SpacecraftGeometry
    mass_kg: float
    initial_quaternion: np.ndarray
    initial_body_angular_velocity: np.ndarray
    inertia: np.ndarray
    eccentricity: float
    alpha: float
    mu: float
    semi_major_axis: float
    anomaly_domain: tuple[float, float] = (0., 2.*np.pi)
    docking_standoff: float = .30
    integration_rtol: float = 1e-11
    integration_atol: float = 1e-13
    integration_max_step: float = .05
    gravity_gradient: bool = True
    shape: str = field(default='composite_spacecraft', init=False)

    def __post_init__(self):
        if not isinstance(self.geometry, SpacecraftGeometry):
            raise TypeError('Spacecraft target requires explicit SpacecraftGeometry')
        if self.geometry.frame != 'target_body_COM':
            raise ValueError('Dynamics and geometry must share the target-body COM frame')
        for name in ('mass_kg', 'docking_standoff'):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
            object.__setattr__(self, name, float(value))
        self._initialize_motion()
        if np.trace(self.inertia) > 2*self.mass_kg*self.bounding_sphere_radius**2*(1+1e-12):
            raise ValueError('Target inertia exceeds its declared occupied-envelope moment bound')

    @property
    def initial_orientation(self):
        return quaternion_to_euler(self.initial_quaternion)

    @property
    def bounding_sphere_radius(self):
        return max(float(np.linalg.norm([float(x) for x in p]))
                   for c in self.geometry.components for p in c.vertices_body_m)

    @property
    def rendezvous_radius(self):
        """Deployment scale only; exclusion always uses the occupied union."""
        return self.bounding_sphere_radius

    def collision_params(self, operation: str, anom: float) -> dict[str, Any]:
        return dict(operation=operation, shape='composite', active_safety_model='component_obb_union',
            center=[0.,0.,0.], rotation_matrix=self.rotation_matrix_at(anom).tolist(),
            geometry=self.geometry.as_dict(), geometry_fingerprint=self.geometry.fingerprint,
            docking_standoff=self.docking_standoff, tube_radius=0., attitude_model=self.attitude_model)

    def assign_docking_targets(self, initial_states: Sequence[np.ndarray]):
        from orbexa.planning.rendezvous.composite import assign_composite_docking_targets
        return assign_composite_docking_targets(self, initial_states)

    @classmethod
    def from_config(cls, config, *, geometry: SpacecraftGeometry, mass_kg: float,
                    initial_body_angular_velocity, initial_quaternion=None,
                    inertia=None, anomaly_domain=None, gravity_gradient=True,
                    docking_standoff=None,
                    **integration_options):
        end = max(config.total_anom, config.iter_anomaly)+max(config.mpc.horizon_steps.values())*config.anom_step
        return cls(geometry=geometry, mass_kg=mass_kg,
            initial_quaternion=(euler_to_quaternion(config.target.initial_orientation)
                                if initial_quaternion is None else initial_quaternion),
            initial_body_angular_velocity=initial_body_angular_velocity,
            inertia=config.target.inertia if inertia is None else inertia,
            eccentricity=config.orbit.eccentricity, alpha=config.orbit.target_drag,
            mu=config.orbit.mu, semi_major_axis=config.orbit.semi_major_axis,
            anomaly_domain=(0.,end) if anomaly_domain is None else anomaly_domain,
            docking_standoff=config.target.docking_standoff if docking_standoff is None else docking_standoff,
            gravity_gradient=gravity_gradient,
            **integration_options)

    def as_dict(self):
        return dict(shape=self.shape, geometry=self.geometry.as_dict(),
            geometry_fingerprint=self.geometry.fingerprint, mass_kg=self.mass_kg,
            bounding_sphere_radius=self.bounding_sphere_radius,
            rendezvous_sphere_radius=self.rendezvous_radius,
            initial_orientation=self.initial_orientation.tolist(), docking_standoff=self.docking_standoff,
            docking_clearance=self.docking_standoff, assignment_strategy='composite_accessible_patch_hungarian',
            attitude_model=self.attitude_model, motion_model=self.attitude_model,
            gravity_gradient=self.gravity_gradient,
            external_torque_model='central_spherical_gravity_gradient' if self.gravity_gradient else 'zero',
            collision_geometry='explicit_component_obb_union', attitude_reference_frame='body_to_LVLH',
            quaternion_order='xyzw', initial_quaternion=self.initial_quaternion.tolist(),
            initial_body_angular_velocity_rad_s=self.initial_body_angular_velocity.tolist(),
            inertia_kg_m2=self.inertia.tolist(), anomaly_domain=list(self.anomaly_domain),
            principal_moments_kg_m2=np.linalg.eigvalsh(self.inertia).tolist(),
            clock_model='analytical_inverse_radius_quadratic_drag',
            clock_parameters=dict(eccentricity=float(self.eccentricity),alpha=float(self.alpha),
                mu=float(self.mu),semi_major_axis=float(self.semi_major_axis),theta0=0.),
            inertial_reference='fixed coincident to LVLH at the first anomaly',
            lvlh_frame_rotation='positive anomaly increment about orbit normal',
            integration=dict(method='DOP853',rtol=self.integration_rtol,atol=self.integration_atol,
                             max_anomaly_step=self.integration_max_step),
            assumptions=['rigid body with constant physically realizable inertia',
                'known target attitude and spin under caller-declared pose bounds',
                'planar analytical orbit with central-field gravity-gradient torque' if self.gravity_gradient
                    else 'planar analytical orbit and torque-free attitude',
                'component dimensions/layout and mass properties have the provenance and uncertainty in geometry metadata',
                'unprepared structural-face candidates; contact strength and flexible appendages are not qualified'])

    @classmethod
    def from_dict(cls, value):
        if value.get('shape') != 'composite_spacecraft':
            raise ValueError('Unsupported physical spacecraft target shape')
        geometry = SpacecraftGeometry.from_dict(value['geometry'])
        if value.get('geometry_fingerprint') != geometry.fingerprint:
            raise ValueError('Target geometry fingerprint differs from its occupied components')
        clock, integration = value['clock_parameters'], value['integration']
        if value.get('clock_model') != 'analytical_inverse_radius_quadratic_drag' or integration.get('method') != 'DOP853':
            raise ValueError('Unsupported target clock or attitude integrator')
        return cls(geometry=geometry,mass_kg=value['mass_kg'],
            initial_quaternion=value['initial_quaternion'],
            initial_body_angular_velocity=value['initial_body_angular_velocity_rad_s'],
            inertia=value['inertia_kg_m2'],eccentricity=clock['eccentricity'],alpha=clock['alpha'],
            mu=clock['mu'],semi_major_axis=clock['semi_major_axis'],anomaly_domain=tuple(value['anomaly_domain']),
            docking_standoff=value['docking_standoff'],gravity_gradient=value['gravity_gradient'],
            integration_rtol=integration['rtol'],integration_atol=integration['atol'],
            integration_max_step=integration['max_anomaly_step'])
