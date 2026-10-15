"""Prospective finite-spacecraft ADTMPC approach scenario composition."""
from __future__ import annotations

from dataclasses import dataclass
import copy
import numpy as np

from orbexa.estimation.belief import ParameterBelief
from orbexa.simulation.deployment import CompactArcDeployment, CompactDeploymentOptions
from orbexa.simulation.mission_safety import MissionSafety


@dataclass(frozen=True)
class PhysicalApproachOptions:
    """Representative injectable geometry; these values are not flight hardware."""

    body_radius_m: float = .150
    com_standoff_m: float = .165
    initial_body_spin_rad_s: tuple[float, float, float] = (.0002, .0003, .00015)
    gravity_gradient: bool = True

    def __post_init__(self):
        if (not np.isfinite(self.body_radius_m) or self.body_radius_m <= 0 or
                not np.isfinite(self.com_standoff_m) or self.com_standoff_m <= self.body_radius_m):
            raise ValueError('The COM standoff must exceed the positive body radius')
        spin = np.asarray(self.initial_body_spin_rad_s, dtype=float)
        if spin.shape != (3,) or not np.all(np.isfinite(spin)):
            raise ValueError('Initial physical spin must have three finite components')
        if not isinstance(self.gravity_gradient, bool):
            raise TypeError('gravity_gradient must be a boolean')


@dataclass(frozen=True)
class ContinuousAdmissionOptions:
    """Declared sensing/model envelopes required by a continuous-flow guard.

    They are assumptions supplied by the caller; this class does not infer a
    sensor accuracy or validate the numerical target's attitude integration.
    The angular rate is the full relative angular-velocity norm per anomaly for
    the cylinder path. Composite physical targets derive that bound from their
    explicit initial spin, inertia, torque model and finite orbital domain;
    their numerical pose operator error remains a caller assumption.
    """

    measurement_radius: tuple[float, ...]
    acceleration_radius: tuple[float, ...]
    target_rotation_operator_error: float
    target_relative_angular_rate_bound: float

    def __post_init__(self):
        for name, size in (('measurement_radius', 6), ('acceleration_radius', 3)):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (size,) or not np.all(np.isfinite(value)) or np.any(value < 0):
                raise ValueError(f'{name} must be a finite nonnegative {size}-vector')
        for name in ('target_rotation_operator_error', 'target_relative_angular_rate_bound'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative')


class PhysicalApproachScenario:
    """Compose compact deployment, physical target and finite-body safety.

    A supplied controller belief is copied and never centered on plant truth.
    The target quaternion/rate propagation uses physical seconds and its
    declared environmental torque. The maintained nonlinear ADTMPC is unchanged.
    """

    def __init__(self, config, *, initial_belief: ParameterBelief,
                 maximum_updates: int = 550, options: PhysicalApproachOptions | None = None,
                 deployment_options: CompactDeploymentOptions | None = None,
                 admission: ContinuousAdmissionOptions | None = None,
                 certificate_sink=None, adaptation_mode='adaptive', controller_factory=None,
                 target_anomaly_domain=None, target_bound_scope=None, target=None,
                 vehicle_design=None):
        from orbexa.core.rigid_body_target import RigidBodyCylinderTarget
        if isinstance(maximum_updates, bool) or not isinstance(maximum_updates, (int, np.integer)) or maximum_updates < 1:
            raise ValueError('maximum_updates must be a positive integer')
        self.options = options if options is not None else PhysicalApproachOptions()
        self.config = copy.deepcopy(config)
        self.config.target.docking_standoff = self.options.com_standoff_m
        self.config.__post_init__()
        self.prior = initial_belief.copy()
        end = (maximum_updates + max(self.config.mpc.horizon_steps.values()) + 2) * self.config.anom_step
        domain = (0., end) if target_anomaly_domain is None else target_anomaly_domain
        self.target = (RigidBodyCylinderTarget.from_config(self.config,
            initial_body_angular_velocity=self.options.initial_body_spin_rad_s,
            gravity_gradient=self.options.gravity_gradient, anomaly_domain=domain) if target is None else target)
        from orbexa.core.spacecraft_target import RigidBodySpacecraftTarget
        composite = isinstance(self.target,RigidBodySpacecraftTarget)
        if not composite and not isinstance(self.target,RigidBodyCylinderTarget):
            raise TypeError('Physical approach requires an explicitly supported rigid-body target')
        if self.config.orbit.time_periapsis != 0:
            raise ValueError('The physical target clock requires the supported zero periapsis phase origin')
        for name, expected in (('eccentricity', self.config.orbit.eccentricity),
                               ('alpha', self.config.orbit.target_drag),
                               ('mu', self.config.orbit.mu),
                               ('semi_major_axis', self.config.orbit.semi_major_axis)):
            if getattr(self.target, name) != expected:
                raise ValueError(f'Injected target clock {name} must equal the physical approach configuration')
        if not self.target.anomaly_domain[0] <= 0 <= self.target.anomaly_domain[1]:
            raise ValueError('The injected target preview domain must include the initial approach anomaly zero')
        if not np.array_equal(self.target.initial_body_angular_velocity, self.options.initial_body_spin_rad_s):
            raise ValueError('Injected target initial body spin must equal the explicit physical approach options')
        if self.target.gravity_gradient != self.options.gravity_gradient:
            raise ValueError('Injected target environmental torque model must equal the physical approach options')
        if not np.isclose(self.target.docking_standoff,self.options.com_standoff_m,rtol=0,atol=1e-12):
            raise ValueError('Injected target and physical COM standoff must agree')
        if composite:
            from orbexa.simulation.composite_deployment import CompositePatchDeployment
            self.config.target.inertia = self.target.inertia.copy()
            self.deployment = CompositePatchDeployment(self.config,target=self.target,
                body_radius_m=self.options.body_radius_m,options=deployment_options)
        else:
            self.deployment = CompactArcDeployment(self.config, target=self.target, options=deployment_options)
        self.safety = MissionSafety(self.config, self.target, chaser_body_radius=self.options.body_radius_m)
        guard = None
        self.rotation_bounds = None
        if admission is not None:
            from orbexa.control.reachability import IntervalBox, OrbitalModelBounds, RotatingCylinderBounds
            from orbexa.simulation.certified_execution import CertifiedExecutionGuard
            orbit = self.config.orbit
            scope = ('Caller-declared valid target pose/operator-error and full relative angular-rate bounds for this hold'
                     if target_bound_scope is None else target_bound_scope)
            if not isinstance(scope, str) or not scope.strip():
                raise ValueError('Identify the declared target-bound assumptions')
            if composite:
                from orbexa.control.rigid_body_bounds import bound_rigid_body_rotation
                self.rotation_bounds = bound_rigid_body_rotation(*self.target.anomaly_domain,
                    OrbitalModelBounds(eccentricity=(self.target.eccentricity,) * 2,
                        alpha=(self.target.alpha,) * 2, beta=(0, 0), mu=self.target.mu,
                        semi_major_axis=self.target.semi_major_axis),
                    inertia=self.target.inertia,
                    initial_spin=IntervalBox.point(self.target.initial_body_angular_velocity),
                    gravity_gradient=self.target.gravity_gradient)
                if target_bound_scope is None:
                    scope = ('Analytical fixed-inertia energy and orbital-clock rotation-rate bound over the '
                        'declared target preview domain; caller-declared numerical pose/operator error. '
                        'Known initial body spin and central gravity-gradient-only or zero torque model.')
            def geometry(q, width):
                if composite:
                    from orbexa.control.composite_geometry import RotatingCompositeBounds
                    return RotatingCompositeBounds(geometry=self.target.geometry,
                        orientation_matrix_at_start=self.target.rotation_matrix_at(q),
                        orientation_matrix_error=admission.target_rotation_operator_error,
                        angular_rate_per_anomaly=self.rotation_bounds.relative_angular_rate_upper,scope=scope,
                        rotation_rate_bounds=self.rotation_bounds)
                return RotatingCylinderBounds(radius=self.target.radius * (1 + self.target.tolerance),
                    half_length=self.target.half_length * (1 + self.target.tolerance),
                    orientation_matrix_at_start=self.target.rotation_matrix_at(q),
                    orientation_matrix_error=admission.target_rotation_operator_error,
                    angular_rate_per_anomaly=admission.target_relative_angular_rate_bound,
                    scope=scope)
            guard = CertifiedExecutionGuard(model=OrbitalModelBounds(
                    eccentricity=self.prior.feasible_sets['eccentricity'],
                    alpha=self.prior.feasible_sets['alpha'], beta=self.prior.feasible_sets['beta'],
                    semi_major_axis=orbit.semi_major_axis, mu=orbit.mu),
                measurement_radius=admission.measurement_radius,
                acceleration_radius=admission.acceleration_radius,
                body_radius=self.options.body_radius_m,
                input_lower=[float(b['lower']) for b in self.config.mpc.input_bounds],
                input_upper=[float(b['upper']) for b in self.config.mpc.input_bounds],
                minimum_pair_separation=self.safety.min_chaser_separation,
                target_bounds=geometry, record_sink=certificate_sink)
        dependencies = dict(safety=self.safety, initial_belief=self.prior,
                            adaptation_mode=adaptation_mode, execution_guard=guard)
        if controller_factory is not None:
            dependencies['controller_factory'] = controller_factory
        if vehicle_design is not None:
            dependencies['vehicle_design'] = vehicle_design
        self.scenario = self.deployment.create_scenario(**dependencies)
        self.scenario.require_coupled_execution()
