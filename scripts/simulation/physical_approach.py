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
    The angular rate is the full relative angular-velocity norm per anomaly.
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
                 target_anomaly_domain=None, target_bound_scope=None):
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
        self.target = RigidBodyCylinderTarget.from_config(self.config,
            initial_body_angular_velocity=self.options.initial_body_spin_rad_s,
            gravity_gradient=self.options.gravity_gradient, anomaly_domain=domain)
        self.deployment = CompactArcDeployment(self.config, target=self.target, options=deployment_options)
        self.safety = MissionSafety(self.config, self.target, chaser_body_radius=self.options.body_radius_m)
        guard = None
        if admission is not None:
            from orbexa.control.reachability import OrbitalModelBounds, RotatingCylinderBounds
            from orbexa.simulation.certified_execution import CertifiedExecutionGuard
            orbit = self.config.orbit
            scope = ('Caller-declared valid target pose/operator-error and full relative angular-rate bounds for this hold'
                     if target_bound_scope is None else target_bound_scope)
            if not isinstance(scope, str) or not scope.strip():
                raise ValueError('Identify the declared target-bound assumptions')
            def geometry(q, width):
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
        self.scenario = self.deployment.create_scenario(**dependencies)
        self.scenario.require_coupled_execution()
