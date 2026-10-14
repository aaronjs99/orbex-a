"""Independent Cartesian SI propagation for orbital model-discrepancy replay.

This numerical service uses full vector gravity/drag, without relative-motion
linearization or an anomaly clock. It is not connected to the mission's
analytical controller, contractor or Arb flow proof. Density, atmosphere
velocity, spacecraft drag area per mass, J2 and applied accelerations are
explicit caller declarations. No real atmosphere or spacecraft is inferred.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Callable, Sequence
import numpy as np


def _scalar(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real numeric scalar")
    try:
        value = float(value)
    except (OverflowError, ValueError, TypeError) as error:
        raise ValueError(f"{name} must be representable as a finite scalar") from error
    if not np.isfinite(value) or (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError(f"{name} has an invalid finite value or sign")
    return value


def _array(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" and (raw.dtype.kind != "O" or any(
            isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in raw.flat)):
        raise ValueError(f"{name} must contain real numeric scalars")
    try:
        result = np.array(raw, dtype=float, copy=True)
    except (OverflowError, ValueError, TypeError) as error:
        raise ValueError(f"{name} must be representable as finite binary64 values") from error
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _vector(value, name, width):
    result = _array(value, name)
    if result.shape != (width,):
        raise ValueError(f"{name} must have {width} components")
    return result


def _states(value):
    result = _array(value, "Cartesian states")
    if result.shape == (6,):
        result = result[None, :]
    if result.ndim != 2 or result.shape[1] != 6 or len(result) < 1:
        raise ValueError("Use [position m, velocity m/s] with shape (N,6) or (6,)")
    return result


@dataclass(frozen=True)
class InertialGravity:
    """Central gravity plus optional fixed-axis J2, entirely in one inertial frame.

    J2 is the unnormalized zonal coefficient, with sign opposite C20. Its
    reference radius and inertial symmetry axis must describe the same body.
    minimum_radius_m is a declared model-domain boundary, not an impact model.
    """

    mu_m3_s2: float
    j2: float = 0.
    reference_radius_m: float | None = None
    symmetry_axis: Sequence[float] = (0., 0., 1.)
    minimum_radius_m: float = 0.

    def __post_init__(self):
        for name, options in (("mu_m3_s2", dict(positive=True)), ("j2", {}),
                              ("minimum_radius_m", dict(nonnegative=True))):
            object.__setattr__(self, name, _scalar(getattr(self, name), name, **options))
        if self.reference_radius_m is not None:
            object.__setattr__(self, "reference_radius_m", _scalar(self.reference_radius_m, "J2 reference radius", positive=True))
        if self.j2 and self.reference_radius_m is None:
            raise ValueError("Nonzero J2 requires an explicit reference radius")
        axis = _vector(self.symmetry_axis, "J2 symmetry axis", 3)
        norm = np.linalg.norm(axis)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("J2 symmetry axis must have a finite nonzero norm")
        axis /= norm
        axis.setflags(write=False)
        object.__setattr__(self, "symmetry_axis", axis)

    def _radial(self, positions):
        positions = _array(positions, "Inertial positions")
        if positions.ndim < 1 or positions.shape[-1] != 3:
            raise ValueError("Inertial positions need a final dimension of three")
        radius = np.linalg.norm(positions, axis=-1)
        if np.any(~np.isfinite(radius)) or np.any(radius <= self.minimum_radius_m):
            raise ValueError("Position lies outside the declared positive-radius gravity domain")
        return radius, positions / radius[..., None]

    def acceleration(self, positions):
        radius, radial = self._radial(positions)
        acceleration = -(self.mu_m3_s2 / radius / radius)[..., None] * radial
        if self.j2:
            latitude_sine = radial @ self.symmetry_axis
            factor = 1.5 * self.j2 * (self.reference_radius_m / radius) ** 2 * self.mu_m3_s2 / radius / radius
            acceleration += factor[..., None] * ((5 * latitude_sine ** 2 - 1)[..., None] * radial
                                                   - 2 * latitude_sine[..., None] * self.symmetry_axis)
        if not np.all(np.isfinite(acceleration)):
            raise ValueError("Gravity acceleration is not finite")
        return acceleration

    def potential(self, positions):
        """Specific potential (J/kg), useful for independent conservation checks."""
        radius, radial = self._radial(positions)
        value = -self.mu_m3_s2 / radius
        if self.j2:
            value += self.mu_m3_s2 / radius * self.j2 * (self.reference_radius_m / radius) ** 2 * (
                3 * (radial @ self.symmetry_axis) ** 2 - 1) / 2
        if not np.all(np.isfinite(value)):
            raise ValueError("Gravity potential is not finite")
        return value

    def as_dict(self):
        return dict(mu_m3_s2=self.mu_m3_s2, j2=self.j2, reference_radius_m=self.reference_radius_m,
                    symmetry_axis=self.symmetry_axis.tolist(), minimum_radius_m=self.minimum_radius_m)


@dataclass(frozen=True)
class QuadraticDrag:
    """Isotropic drag: -rho*(C_D A/m)*|v-v_air|*(v-v_air)/2.

    Density receives (seconds, inertial position m). The optional atmosphere
    velocity callback uses the same arguments and returns inertial m/s; None
    explicitly means a stationary atmosphere in this frame. Cross-section,
    coefficient and mass are combined in cda_over_mass_m2_kg. An attitude-
    dependent surface/flow model must be supplied separately, not inferred.
    """

    cda_over_mass_m2_kg: float
    density_kg_m3: Callable
    description: str
    atmosphere_velocity_m_s: Callable | None = None

    def __post_init__(self):
        object.__setattr__(self, "cda_over_mass_m2_kg", _scalar(
            self.cda_over_mass_m2_kg, "C_D A/m", nonnegative=True))
        if not callable(self.density_kg_m3) or (self.atmosphere_velocity_m_s is not None and not callable(self.atmosphere_velocity_m_s)):
            raise ValueError("Density and optional atmosphere velocity must be explicit callables")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("Describe the declared density, atmosphere and drag-parameter source")

    def acceleration(self, time_s, state):
        time_s = _scalar(time_s, "Drag time")
        state = _vector(state, "Drag Cartesian state", 6)
        if self.cda_over_mass_m2_kg == 0:
            return np.zeros(3)
        density = _scalar(self.density_kg_m3(time_s, state[:3].copy()), "Atmospheric density", nonnegative=True)
        air = np.zeros(3) if self.atmosphere_velocity_m_s is None else _vector(
            self.atmosphere_velocity_m_s(time_s, state[:3].copy()), "Atmosphere inertial velocity", 3)
        relative = state[3:] - air
        result = -.5 * density * self.cda_over_mass_m2_kg * np.linalg.norm(relative) * relative
        if not np.all(np.isfinite(result)):
            raise ValueError("Drag acceleration is not finite")
        return result


@dataclass(frozen=True)
class LVLHFrame:
    """Instantaneous x=-transverse, y=radial, z=orbit-normal frame.

    target_acceleration_m_s2 is needed for the frame derivative, including
    out-of-plane acceleration. Relative velocities in this API are m/s,
    not dp/dq. Both transformations include the rotating-frame velocity term.
    """

    target_state: np.ndarray
    target_acceleration_m_s2: np.ndarray
    rotation: np.ndarray = field(init=False)
    rotation_derivative: np.ndarray = field(init=False)

    def __post_init__(self):
        state = _vector(self.target_state, "Target Cartesian state", 6)
        acceleration = _vector(self.target_acceleration_m_s2, "Target acceleration", 3)
        position, velocity = state[:3], state[3:]
        radius = np.linalg.norm(position)
        momentum = np.cross(position, velocity)
        momentum_norm = np.linalg.norm(momentum)
        if not np.isfinite(radius) or not np.isfinite(momentum_norm) or radius == 0 or momentum_norm == 0:
            raise ValueError("A local orbital frame needs nonzero finite radius and angular momentum")
        radial, normal = position / radius, momentum / momentum_norm
        radial_dot = (velocity - radial * np.dot(radial, velocity)) / radius
        momentum_dot = np.cross(position, acceleration)
        normal_dot = (momentum_dot - normal * np.dot(normal, momentum_dot)) / momentum_norm
        transverse = np.cross(normal, radial)
        transverse_dot = np.cross(normal_dot, radial) + np.cross(normal, radial_dot)
        rotation = np.column_stack((-transverse, radial, normal))
        derivative = np.column_stack((-transverse_dot, radial_dot, normal_dot))
        for name, value in (("target_state", state), ("target_acceleration_m_s2", acceleration),
                            ("rotation", rotation), ("rotation_derivative", derivative)):
            if not np.all(np.isfinite(value)):
                raise ValueError("The local orbital frame is not finite")
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def to_inertial(self, relative_states):
        relative = _states(relative_states)
        positions = relative[:, :3] @ self.rotation.T
        velocities = relative[:, 3:] @ self.rotation.T + relative[:, :3] @ self.rotation_derivative.T
        return np.concatenate((positions + self.target_state[:3], velocities + self.target_state[3:]), axis=1)

    def from_inertial(self, states):
        states = _states(states)
        positions = (states[:, :3] - self.target_state[:3]) @ self.rotation
        velocities = (states[:, 3:] - self.target_state[3:]) @ self.rotation - positions @ self.rotation_derivative.T @ self.rotation
        return np.concatenate((positions, velocities), axis=1)


@dataclass(frozen=True)
class InertialPropagationResult:
    status: str
    reason: str
    times_s: np.ndarray
    states: np.ndarray
    requested_samples_completed: int
    accepted_steps: int
    evaluations: int
    context: dict

    def __post_init__(self):
        for name in ("times_s", "states"):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def success(self):
        return self.status == "COMPLETED"

    @property
    def final_states(self):
        return self.states[-1]

    def as_dict(self):
        return dict(status=self.status, reason=self.reason, success=self.success,
                    times_s=self.times_s.tolist(), states=self.states.tolist(),
                    requested_samples_completed=self.requested_samples_completed,
                    accepted_steps=self.accepted_steps, evaluations=self.evaluations, context=self.context)


class InertialDynamics:
    """Shared-time full nonlinear Cartesian propagation of independent spacecraft.

    There is no mutual gravity, contact, attitude or mass-depletion model.
    Applied accelerations are inertial m/s². Callbacks must be pure functions.
    A bounded numerical failure retains all requested samples already reached
    and the last accepted integrator point; it never reports a complete span.
    """

    def __init__(self, gravity: InertialGravity, drag_models: Sequence[QuadraticDrag | None] = ()):
        if not isinstance(gravity, InertialGravity):
            raise TypeError("Supply explicit InertialGravity parameters")
        self.gravity, self.drag_models = gravity, tuple(drag_models)
        if any(item is not None and not isinstance(item, QuadraticDrag) for item in self.drag_models):
            raise TypeError("Use one QuadraticDrag or None for every propagated spacecraft")

    def accelerations(self, time_s, states, applied_acceleration=None):
        time_s, states = _scalar(time_s, "Physical time"), _states(states)
        if self.drag_models and len(self.drag_models) != len(states):
            raise ValueError("Drag-model count must match the complete propagated fleet")
        result = self.gravity.acceleration(states[:, :3])
        for index, model in enumerate(self.drag_models):
            if model is not None:
                result[index] += model.acceleration(time_s, states[index])
        if applied_acceleration is not None:
            applied = _array(applied_acceleration, "Applied inertial acceleration")
            if applied.shape != result.shape:
                raise ValueError("Applied inertial acceleration must have shape (N,3)")
            result += applied
        if not np.all(np.isfinite(result)):
            raise ValueError("The complete inertial acceleration is not finite")
        return result

    def propagate(self, initial_states, sample_times_s, *, max_step_s,
                  rtol=1e-11, atol=(1e-6, 1e-6, 1e-6, 1e-9, 1e-9, 1e-9),
                  max_steps=100000, applied_acceleration: Callable | None = None,
                  control_description=None) -> InertialPropagationResult:
        """Propagate a finite increasing SI sample grid with explicit resources.

        The first/last requested times define the integration span. On failure,
        an unscheduled final accepted point may be appended to preserve the
        numerical prefix. No error enclosure or physical safety proof follows.
        """
        from scipy.integrate import DOP853
        import scipy
        states, times = _states(initial_states), _array(sample_times_s, "SI sample times")
        if times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0):
            raise ValueError("Supply at least two finite strictly increasing SI times")
        max_step_s, rtol = _scalar(max_step_s, "Maximum SI step", positive=True), _scalar(rtol, "Relative tolerance", positive=True)
        if rtol < 100 * np.finfo(float).eps:
            raise ValueError("DOP853 relative tolerance must be at least 100 times binary64 machine epsilon")
        tolerance = _vector(atol, "SI absolute tolerance", 6)
        if np.any(tolerance <= 0) or isinstance(max_steps, bool) or not isinstance(max_steps, Integral) or max_steps < 1:
            raise ValueError("Use positive component tolerances and a positive integer step budget")
        if applied_acceleration is not None and not callable(applied_acceleration):
            raise ValueError("Applied acceleration must be an explicit callable")
        if control_description is None and applied_acceleration is None:
            control_description = "No applied acceleration"
        if not isinstance(control_description, str) or not control_description.strip():
            raise ValueError("Describe the physical applied-acceleration source")
        self.accelerations(times[0], states)  # Validate model/fleet before any propagation.
        context = dict(gravity=self.gravity.as_dict(), drags=[None if item is None else dict(
            cda_over_mass_m2_kg=item.cda_over_mass_m2_kg, declaration=item.description,
            atmosphere="stationary inertial" if item.atmosphere_velocity_m_s is None else "caller-supplied inertial velocity")
            for item in self.drag_models], control=control_description,
            integration=dict(method="DOP853", scipy_version=scipy.__version__, rtol=rtol, atol=tolerance.tolist(),
                             max_step_s=max_step_s, max_steps=int(max_steps), requested_span_s=[float(times[0]), float(times[-1])]),
            scope="Numerical Cartesian gravity/drag replay; no validated flow, collision certificate or closed-loop mission claim")
        output_times, output_states, index, steps, evaluations = [float(times[0])], [states.copy()], 1, 0, 0
        accepted_time, accepted_state = float(times[0]), states.copy()

        def rhs(time, flat):
            nonlocal evaluations
            evaluations += 1
            current = flat.reshape(states.shape)
            applied = None if applied_acceleration is None else applied_acceleration(float(time), current.copy())
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                return np.concatenate((current[:, 3:], self.accelerations(time, current, applied)), axis=1).ravel()

        status, reason = "COMPLETED", "Requested numerical SI interval completed"
        try:
            solver = DOP853(rhs, accepted_time, states.ravel(), float(times[-1]),
                            max_step=max_step_s, rtol=rtol, atol=np.tile(tolerance, len(states)))
            while solver.status == "running" and steps < max_steps:
                message = solver.step()
                if solver.status == "failed":
                    status, reason = "NUMERICAL_FAILURE", str(message)
                    break
                accepted_time, accepted_state = float(solver.t), _states(solver.y.reshape(states.shape))
                steps += 1
                dense = solver.dense_output()
                while index < len(times) and times[index] <= accepted_time:
                    output_times.append(float(times[index]))
                    output_states.append(_states(dense(times[index]).reshape(states.shape)))
                    index += 1
            if solver.status == "running":
                status, reason = "STEP_BUDGET_EXHAUSTED", "Finite integration step budget exhausted before the requested end"
        except (ValueError, RuntimeError, ArithmeticError, TypeError) as error:
            status, reason = "DYNAMICS_REFUSED", f"{type(error).__name__}: {error}"
        if status != "COMPLETED" and accepted_time > output_times[-1]:
            output_times.append(accepted_time)
            output_states.append(accepted_state)
        return InertialPropagationResult(status, reason, np.asarray(output_times), np.asarray(output_states),
                                         index, steps, evaluations, context)
