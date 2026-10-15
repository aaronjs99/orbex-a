"""Bounded free-flying bus attitude tracking during a saved approach.

This is a numerical attitude replay in physical seconds. It does not alter
the translational mission, certify its collision geometry, or infer a bus
pose from point-mass translation. A fuel allowance for that translation is
required explicitly. Quaternions are body-to-ECI, scalar last (xyzw).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from orbexa.core.attitude import body_rate_derivative, finite_vector
from orbexa.simulation.assembly import STANDARD_GRAVITY, physical_inertia, positive
from orbexa.simulation.capture import ChaserPhysicalProperties


def _rotation(quaternion):
    """Body-to-ECI matrix for a scalar-last quaternion, normalizing stages."""
    vector = np.asarray(quaternion, dtype=float)
    vector = vector / np.linalg.norm(vector)
    x, y, z, w = vector
    return np.array([[1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
                     [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
                     [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]])


def _readonly_vector(value, name, size):
    vector = finite_vector(value, name, size)
    vector.setflags(write=False)
    return vector


@dataclass(frozen=True)
class ChaserAttitudeInitialState:
    quaternion_body_to_eci: np.ndarray
    angular_velocity_body_rad_s: np.ndarray

    def __post_init__(self):
        quaternion = finite_vector(self.quaternion_body_to_eci, "Initial bus quaternion", 4)
        norm = float(np.linalg.norm(quaternion))
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("Initial bus quaternion must have finite nonzero norm")
        quaternion /= norm
        quaternion.setflags(write=False)
        object.__setattr__(self, "quaternion_body_to_eci", quaternion)
        object.__setattr__(self, "angular_velocity_body_rad_s", _readonly_vector(
            self.angular_velocity_body_rad_s, "Initial bus angular velocity", 3))


@dataclass(frozen=True)
class ChaserAttitudeActuators:
    """Three orthogonal torque couples, each made of two signed nozzles.

    A signed channel denotes a colocated pair of opposed physical nozzles;
    only the nozzle in the commanded direction fires. The two locations in
    each torque couple fire equal magnitudes, cancelling net force exactly.
    """
    moment_arm_m: float = .1
    nozzle_force_limit_n: float = .2
    specific_impulse_s: float = 300.

    def __post_init__(self):
        for name in ("moment_arm_m", "nozzle_force_limit_n", "specific_impulse_s"):
            object.__setattr__(self, name, positive(getattr(self, name), name))

    @property
    def torque_limit_nm(self):
        return 2 * self.moment_arm_m * self.nozzle_force_limit_n

    @property
    def positions_body_m(self):
        axes = np.eye(3)
        return np.array([sign * self.moment_arm_m * axes[(axis + 1) % 3]
                         for axis in range(3) for sign in (-1, 1)])

    @property
    def directions_body(self):
        axes = np.eye(3)
        return np.array([sign * axes[(axis + 2) % 3]
                         for axis in range(3) for sign in (-1, 1)])

    def allocate(self, requested_torque_nm):
        requested = finite_vector(requested_torque_nm, "Requested bus torque", 3)
        requested_commands = requested / (2 * self.moment_arm_m)
        commands = np.repeat(np.clip(requested_commands, -self.nozzle_force_limit_n, self.nozzle_force_limit_n), 2)
        forces = commands[:, None] * self.directions_body
        torque = np.sum(np.cross(self.positions_body_m, forces), axis=0)
        flow = float(np.sum(np.abs(commands)) / (STANDARD_GRAVITY * self.specific_impulse_s))
        return dict(requested_torque_body_nm=requested, commands_n=commands, force_body_n=np.sum(forces, axis=0),
                    torque_body_nm=torque, fuel_flow_kg_s=flow,
                    saturated=bool(np.any(np.abs(requested_commands) > self.nozzle_force_limit_n)))

    def to_dict(self):
        return dict(moment_arm_m=float(self.moment_arm_m), nozzle_force_limit_n=float(self.nozzle_force_limit_n),
                    specific_impulse_s=float(self.specific_impulse_s), torque_limit_nm=float(self.torque_limit_nm),
                    positions_body_m=self.positions_body_m.tolist(), directions_body=self.directions_body.tolist(),
                    channel_semantics="six signed channels; each sign selects an opposed physical nozzle")


@dataclass(frozen=True)
class ChaserAttitudeOptions:
    control_period_s: float = .5
    max_integration_step_s: float = .25
    natural_frequency_rad_s: float = .25
    damping_ratio: float = .9
    initial_misalignment_rotvec_rad: np.ndarray = field(default_factory=lambda: np.array([.4, -.3, .2]))
    max_final_orientation_error_rad: float = .01
    max_final_spin_error_rad_s: float = 1e-5
    max_constant_inertia_mass_change_fraction: float = .02
    max_samples: int = 500000

    def __post_init__(self):
        for name in ("control_period_s", "max_integration_step_s", "natural_frequency_rad_s", "damping_ratio",
                     "max_final_orientation_error_rad", "max_final_spin_error_rad_s"):
            object.__setattr__(self, name, positive(getattr(self, name), name))
        fraction = positive(self.max_constant_inertia_mass_change_fraction, "Constant inertia mass fraction")
        if fraction >= 1:
            raise ValueError("Constant inertia mass fraction must be less than one")
        object.__setattr__(self, "max_constant_inertia_mass_change_fraction", fraction)
        if isinstance(self.max_samples, bool) or not isinstance(self.max_samples, int) or self.max_samples < 2:
            raise ValueError("Maximum attitude samples must be an integer of at least two")
        offset = _readonly_vector(self.initial_misalignment_rotvec_rad, "Initial attitude offset", 3)
        if np.linalg.norm(offset) >= np.pi:
            raise ValueError("Default initial offset must be strictly inside the pi rotation chart")
        object.__setattr__(self, "initial_misalignment_rotvec_rad", offset)


class RecordedTargetAttitudeReference:
    """Shortest-arc SLERP pose and independently interpolated known rates.

    Pose interpolation is exact at recorded nodes. Its between-node derivative
    need not equal the interpolated physical spin; that defect is reported.
    This reference is numerical data, not a validated target enclosure.
    """

    def __init__(self, approach):
        kinematics = approach.metadata.get("target_kinematics", {})
        anomaly = np.asarray(approach.anom_history, dtype=float)
        saved_anomaly = np.asarray(kinematics.get("anomaly", []), dtype=float)
        self.times_s = np.asarray(kinematics.get("physical_time", []), dtype=float)
        count = len(anomaly) if anomaly.ndim == 1 else 0
        if (count < 2 or np.any(~np.isfinite(anomaly)) or np.any(np.diff(anomaly) <= 0)
                or saved_anomaly.shape != (count,) or not np.allclose(saved_anomaly, anomaly, rtol=0, atol=1e-12)
                or self.times_s.shape != (count,) or np.any(~np.isfinite(self.times_s))
                or np.any(~np.isfinite(np.diff(self.times_s))) or np.any(np.diff(self.times_s) <= 0)):
            raise ValueError("Recorded attitude reference needs aligned, increasing anomaly and physical time")
        if approach.target.get("quaternion_order", "xyzw") != "xyzw":
            raise ValueError("Recorded attitude quaternions must use xyzw order")
        quaternions = np.asarray(kinematics.get("inertial_quaternions", []), dtype=float)
        self.body_rates = np.asarray(kinematics.get("body_angular_velocity", []), dtype=float)
        if (quaternions.shape != (count, 4) or np.any(~np.isfinite(quaternions))
                or np.any(np.abs(np.linalg.norm(quaternions, axis=1) - 1) > 1e-8)
                or self.body_rates.shape != (count, 3) or np.any(~np.isfinite(self.body_rates))):
            raise ValueError("Recorded target quaternion and physical body rates must match every node")
        self.rotations = Rotation.from_quat(quaternions)
        relative = (self.rotations[:-1].inv() * self.rotations[1:]).as_rotvec()
        if np.any(np.linalg.norm(relative, axis=1) >= np.pi - 1e-8):
            raise ValueError("Recorded pose spacing has an ambiguous shortest rotation arc")
        self.slerp = Slerp(self.times_s, self.rotations)
        if "body_angular_acceleration" in kinematics:
            self.body_acceleration = np.asarray(kinematics["body_angular_acceleration"], dtype=float)
            self.acceleration_source = "recorded physical body angular acceleration"
        elif approach.target.get("attitude_model") == "torque_free_rigid_body":
            inertia = physical_inertia(approach.target.get("inertia_kg_m2"))
            self.body_acceleration = np.array([body_rate_derivative(omega, inertia, np.zeros(3))
                                               for omega in self.body_rates])
            self.acceleration_source = "Euler torque-free derivative from recorded spin and declared target inertia"
        else:
            raise ValueError("Non-torque-free target data must explicitly record body angular acceleration")
        if self.body_acceleration.shape != (count, 3) or np.any(~np.isfinite(self.body_acceleration)):
            raise ValueError("Recorded target body angular acceleration must match every node")
        arc_rates = relative / np.diff(self.times_s)[:, None]
        self.rate_interpolation_defect_rad_s = float(max(
            np.max(np.linalg.norm(arc_rates - self.body_rates[:-1], axis=1)),
            np.max(np.linalg.norm(arc_rates - self.body_rates[1:], axis=1))))
        self.times_s = self.times_s.copy()
        self.body_rates = self.body_rates.copy()
        self.body_acceleration = self.body_acceleration.copy()

    def sample(self, times_s):
        queries = np.asarray(times_s, dtype=float)
        if (queries.ndim != 1 or len(queries) == 0 or np.any(~np.isfinite(queries))
                or np.any(queries < self.times_s[0]) or np.any(queries > self.times_s[-1])):
            raise ValueError("Attitude reference queries must be finite and stay inside the recorded interval")
        rates = np.column_stack([np.interp(queries, self.times_s, self.body_rates[:, axis]) for axis in range(3)])
        acceleration = np.column_stack([np.interp(queries, self.times_s, self.body_acceleration[:, axis]) for axis in range(3)])
        return self.slerp(queries).as_matrix(), rates, acceleration

    def to_dict(self):
        return dict(pose_interpolation="shortest-arc SLERP of recorded body-to-ECI quaternions",
                    rate_interpolation="componentwise linear interpolation of recorded physical body rates",
                    acceleration_interpolation="componentwise linear interpolation of physical body acceleration",
                    acceleration_source=self.acceleration_source,
                    max_slerp_vs_known_rate_defect_rad_s=self.rate_interpolation_defect_rad_s,
                    certified_target_truth=False)


@dataclass(frozen=True)
class ChaserAttitudeResult:
    success: bool
    message: str
    times_s: np.ndarray
    histories: dict[str, dict[str, Any]]
    audit: dict[str, Any]
    assumptions: tuple[str, ...]
    capture_chasers: dict[str, ChaserPhysicalProperties]

    @property
    def final_attitudes(self):
        """CaptureModel-compatible physical quaternion/body-spin pairs."""
        return {key: (np.array(history["quaternion_body_to_eci"][-1], dtype=float),
                      np.array(history["angular_velocity_body_rad_s"][-1], dtype=float))
                for key, history in self.histories.items()}

    @property
    def fuel_remaining_kg(self):
        return {key: value.fuel_kg for key, value in self.capture_chasers.items()}

    def to_dict(self):
        return dict(success=bool(self.success), message=self.message, times_s=self.times_s.tolist(),
                    histories=self.histories, audit=self.audit, assumptions=list(self.assumptions),
                    final_attitudes={key: [q.tolist(), omega.tolist()] for key, (q, omega) in self.final_attitudes.items()},
                    fuel_remaining_kg=self.fuel_remaining_kg,
                    capture_chasers={key: dict(dry_mass_kg=float(value.dry_mass_kg), fuel_kg=float(value.fuel_kg),
                        body_radius_m=float(value.body_radius_m), gripper_offset_m=float(value.gripper_offset_m))
                        for key, value in self.capture_chasers.items()})


class ChaserAttitudeTracker:
    """Held geometric attitude feedback, bounded nozzles, explicit fuel debit."""

    def __init__(self, *, options: Optional[ChaserAttitudeOptions] = None,
                 actuators: Optional[ChaserAttitudeActuators] = None,
                 chasers: Optional[Mapping[str, ChaserPhysicalProperties]] = None,
                 default_chaser: Optional[ChaserPhysicalProperties] = None):
        self.options = options or ChaserAttitudeOptions()
        self.actuators = actuators or ChaserAttitudeActuators()
        self.chasers = dict(chasers or {})
        self.default_chaser = default_chaser or ChaserPhysicalProperties()

    @staticmethod
    def advance_held_torque(quaternion, body_rates, inertia, torque_body_nm, duration_s,
                            *, max_step_s=.25):
        """RK4 quaternion/Euler propagation plus inertial impulse and work.

        The applied body torque stays fixed over this hold. Inertia stays
        constant under the documented small-depletion approximation. This
        public kernel also permits independent replay of saved commands.
        """
        quaternion = finite_vector(quaternion, "Bus quaternion", 4)
        norm = float(np.linalg.norm(quaternion))
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("Bus quaternion must have finite nonzero norm")
        quaternion /= norm
        omega = finite_vector(body_rates, "Bus body rates", 3)
        matrix = physical_inertia(inertia)
        inverse = np.linalg.inv(matrix)
        torque = finite_vector(torque_body_nm, "Held bus torque", 3)
        duration = positive(duration_s, "Attitude hold duration")
        maximum = positive(max_step_s, "Attitude integration step")
        required = duration / maximum
        if not np.isfinite(required) or required > 1000000:
            raise ValueError("Attitude hold exceeds one million integration substeps")
        count = int(np.ceil(required))
        step = duration / count
        state = np.r_[quaternion, omega, np.zeros(4)]

        def rhs(value):
            vector, scalar, rate = value[:3], value[3], value[4:7]
            quaternion_dot = .5 * np.r_[scalar * rate + np.cross(vector, rate), -np.dot(vector, rate)]
            acceleration = inverse @ (torque - np.cross(rate, matrix @ rate))
            return np.r_[quaternion_dot, acceleration, _rotation(value[:4]) @ torque, np.dot(torque, rate)]

        for _ in range(count):
            k1 = rhs(state)
            k2 = rhs(state + step * k1 / 2)
            k3 = rhs(state + step * k2 / 2)
            k4 = rhs(state + step * k3)
            state += step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            norm = float(np.linalg.norm(state[:4]))
            if not np.all(np.isfinite(state)) or norm == 0:
                raise RuntimeError("Chaser attitude propagation became nonfinite")
            state[:4] /= norm
        return state[:4].copy(), state[4:7].copy(), state[7:10].copy(), float(state[10])

    def track(self, approach, *, translation_fuel_debit_kg: Mapping[str, float],
              initial_states: Optional[Mapping[str, ChaserAttitudeInitialState]] = None) -> ChaserAttitudeResult:
        """Track mounting frames over the complete saved approach timeline.

        The supplied translational allowance is debited once from available
        fuel; it is not claimed to be a measured translation fuel history.
        It must name every bus. Insufficient fuel or an exceeded constant-
        inertia allowance stops before the unexecutable hold and returns a
        failed result containing the common committed fleet prefix.
        """
        reference = RecordedTargetAttitudeReference(approach)
        ids = list(approach.actual_trajectories)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Attitude tracking needs a nonempty fleet")
        if set(translation_fuel_debit_kg) != set(ids):
            raise ValueError("An explicit translational fuel debit is required for every bus")
        if set(self.chasers) - set(ids):
            raise ValueError("Physical bus properties name an unrecorded chaser")
        if initial_states is not None and set(initial_states) != set(ids):
            raise ValueError("Explicit initial attitude states must name every recorded bus")
        assignments = approach.metadata.get("chaser_assignments", [])
        mapping = {value["chaser_id"]: value for value in assignments}
        if len(mapping) != len(assignments) or set(mapping) != set(ids):
            raise ValueError("Recorded assignments must name every bus exactly once")
        count = len(reference.times_s)
        for key in ids:
            states = np.asarray(approach.actual_trajectories[key], dtype=float)
            if states.shape != (count, 6) or np.any(~np.isfinite(states)):
                raise ValueError(f"{key}: saved translation must align with the attitude timeline")
        duration = float(reference.times_s[-1] - reference.times_s[0])
        estimated = duration / self.options.control_period_s
        if not np.isfinite(estimated) or estimated > self.options.max_samples:
            raise ValueError("Recorded approach exceeds the configured attitude sample budget")
        regular_count = int(np.ceil(estimated))
        if regular_count + count > self.options.max_samples:
            raise ValueError("Recorded approach exceeds the configured attitude sample budget")
        regular = reference.times_s[0] + np.arange(regular_count) * self.options.control_period_s
        # Every actual approach node remains an exact attitude output node.
        times = np.unique(np.r_[regular, reference.times_s])
        times = times[(times >= reference.times_s[0]) & (times <= reference.times_s[-1])]
        rotations, target_rates, target_acceleration = reference.sample(times)
        properties, inertias, fuel, initial_fuel, translation, mounts, states = {}, {}, {}, {}, {}, {}, {}
        histories, impulse, work, initial_momentum, initial_energy = {}, {}, {}, {}, {}

        def errors(key, index, quaternion, rate):
            desired_rotation = rotations[index] @ mounts[key]
            actual_rotation = _rotation(quaternion)
            desired_rate = mounts[key].T @ target_rates[index]
            angle = float(Rotation.from_matrix(desired_rotation.T @ actual_rotation).magnitude())
            spin = float(np.linalg.norm(actual_rotation @ rate - rotations[index] @ target_rates[index]))
            return desired_rotation, desired_rate, angle, spin

        def append_state(key, index):
            quaternion, rate = states[key]
            desired_rotation, desired_rate, angle, spin = errors(key, index, quaternion, rate)
            history = histories[key]
            for name, value in (("quaternion_body_to_eci", quaternion.tolist()),
                                ("angular_velocity_body_rad_s", rate.tolist()),
                                ("desired_quaternion_body_to_eci", Rotation.from_matrix(desired_rotation).as_quat().tolist()),
                                ("desired_angular_velocity_body_rad_s", desired_rate.tolist()),
                                ("orientation_error_rad", angle), ("spin_error_rad_s", spin),
                                ("fuel_remaining_kg", float(fuel[key]))):
                history[name].append(value)

        for key in ids:
            normal = finite_vector(mapping[key]["docking_normal_body"], "Assigned outward normal", 3)
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm == 0:
                raise ValueError(f"{key}: docking normal cannot be zero")
            normal /= normal_norm
            from orbexa.core.mounting import mounting_frame
            mounts[key] = mounting_frame(normal)
            prop = properties[key] = self.chasers.get(key, self.default_chaser)
            if not isinstance(prop, ChaserPhysicalProperties):
                raise TypeError("Physical bus properties must be ChaserPhysicalProperties instances")
            if self.actuators.moment_arm_m > prop.body_radius_m:
                raise ValueError(f"{key}: nozzle locations exceed the spherical bus envelope")
            debit = translation[key] = positive(translation_fuel_debit_kg[key], "Translational fuel debit", allow_zero=True)
            if debit > prop.fuel_kg:
                raise ValueError(f"{key}: translational fuel allowance exceeds installed fuel")
            if debit / prop.mass_kg > self.options.max_constant_inertia_mass_change_fraction:
                raise ValueError(f"{key}: translational fuel debit exceeds the constant-inertia approximation scope")
            inertias[key] = physical_inertia(prop.mass_kg * prop.inertia_per_kg_m2)
            fuel[key] = initial_fuel[key] = prop.fuel_kg - debit
            if initial_states is None:
                initial = ChaserAttitudeInitialState((Rotation.from_matrix(rotations[0] @ mounts[key])
                    * Rotation.from_rotvec(self.options.initial_misalignment_rotvec_rad.copy())).as_quat(), np.zeros(3))
            else:
                initial = initial_states[key]
                if not isinstance(initial, ChaserAttitudeInitialState):
                    raise TypeError("Explicit bus initial states must be ChaserAttitudeInitialState instances")
            states[key] = (initial.quaternion_body_to_eci.copy(), initial.angular_velocity_body_rad_s.copy())
            impulse[key], work[key] = np.zeros(3), 0.
            initial_momentum[key] = _rotation(states[key][0]) @ inertias[key] @ states[key][1]
            initial_energy[key] = float(states[key][1] @ inertias[key] @ states[key][1] / 2)
            histories[key] = {name: [] for name in ("quaternion_body_to_eci", "angular_velocity_body_rad_s",
                "desired_quaternion_body_to_eci", "desired_angular_velocity_body_rad_s", "orientation_error_rad",
                "spin_error_rad_s", "fuel_remaining_kg", "requested_torque_body_nm", "applied_torque_body_nm",
                "nozzle_commands_n", "net_force_body_n", "fuel_flow_kg_s", "saturated")}
            append_state(key, 0)
        message, committed = "approach_attitude_tracking_completed", 1
        for index, hold in enumerate(np.diff(times)):
            allocations = {}
            for key in ids:
                quaternion, omega = states[key]
                actual = _rotation(quaternion)
                desired = rotations[index] @ mounts[key]
                transport = actual.T @ desired
                desired_rate = mounts[key].T @ target_rates[index]
                desired_acceleration = mounts[key].T @ target_acceleration[index]
                difference = desired.T @ actual - actual.T @ desired
                attitude_error = .5 * np.array([difference[2, 1], difference[0, 2], difference[1, 0]])
                transported_rate = transport @ desired_rate
                rate_error = omega - transported_rate
                acceleration = (-self.options.natural_frequency_rad_s**2 * attitude_error
                    - 2 * self.options.damping_ratio * self.options.natural_frequency_rad_s * rate_error
                    - np.cross(omega, transported_rate) + transport @ desired_acceleration)
                requested = inertias[key] @ acceleration + np.cross(omega, inertias[key] @ omega)
                allocation = allocations[key] = self.actuators.allocate(requested)
                used = allocation["fuel_flow_kg_s"] * hold
                if used > fuel[key] + 1e-14:
                    message = f"{key}: insufficient_attitude_propellant"
                    break
                debit = translation[key] + initial_fuel[key] - fuel[key] + used
                if debit / properties[key].mass_kg > self.options.max_constant_inertia_mass_change_fraction:
                    message = f"{key}: constant_inertia_mass_change_scope_exceeded"
                    break
            if message != "approach_attitude_tracking_completed":
                break
            for key in ids:
                allocation, history = allocations[key], histories[key]
                q, omega, delta_impulse, delta_work = self.advance_held_torque(
                    *states[key], inertias[key], allocation["torque_body_nm"], hold,
                    max_step_s=self.options.max_integration_step_s)
                states[key] = q, omega
                impulse[key] += delta_impulse
                work[key] += delta_work
                fuel[key] = max(0., fuel[key] - allocation["fuel_flow_kg_s"] * hold)
                # Both the requested feedback and the physically allocated
                # commands are retained, including every saturated hold.
                history["requested_torque_body_nm"].append(allocation["requested_torque_body_nm"].tolist())
                for name, value in (("applied_torque_body_nm", allocation["torque_body_nm"].tolist()),
                                    ("nozzle_commands_n", allocation["commands_n"].tolist()),
                                    ("net_force_body_n", allocation["force_body_n"].tolist()),
                                    ("fuel_flow_kg_s", allocation["fuel_flow_kg_s"]),
                                    ("saturated", allocation["saturated"])):
                    history[name].append(value)
                append_state(key, index + 1)
            committed = index + 2
        complete = committed == len(times)
        final_properties = {key: replace(properties[key], fuel_kg=float(fuel[key])) for key in ids}
        bus_audit = {}
        for key in ids:
            q, omega = states[key]
            history = histories[key]
            final_energy = float(omega @ inertias[key] @ omega / 2)
            consumed = initial_fuel[key] - fuel[key]
            integrated_flow = float(np.dot(history["fuel_flow_kg_s"], np.diff(times[:committed])))
            passed = bool(complete and history["orientation_error_rad"][-1] <= self.options.max_final_orientation_error_rad
                          and history["spin_error_rad_s"][-1] <= self.options.max_final_spin_error_rad_s)
            bus_audit[key] = dict(passed=passed, final_orientation_error_rad=history["orientation_error_rad"][-1],
                final_spin_error_rad_s=history["spin_error_rad_s"][-1], initial_orientation_error_rad=history["orientation_error_rad"][0],
                initial_spin_error_rad_s=history["spin_error_rad_s"][0], translation_fuel_debit_kg=translation[key],
                attitude_fuel_consumed_kg=consumed, fuel_remaining_kg=float(fuel[key]),
                initial_wet_mass_kg=float(properties[key].mass_kg), inertia_body_kg_m2=inertias[key].tolist(),
                total_mass_debit_fraction=float((translation[key] + consumed) / properties[key].mass_kg),
                fuel_balance_residual_kg=float(consumed - integrated_flow),
                angular_impulse_eci_nm_s=impulse[key].tolist(),
                angular_momentum_balance_residual_kg_m2_s=float(np.linalg.norm(
                    _rotation(q) @ inertias[key] @ omega - initial_momentum[key] - impulse[key])),
                torque_work_j=float(work[key]), rotational_energy_balance_residual_j=float(final_energy - initial_energy[key] - work[key]),
                max_nozzle_force_n=float(np.max(np.abs(history["nozzle_commands_n"]))) if committed > 1 else 0.,
                max_net_force_n=float(np.max(np.linalg.norm(history["net_force_body_n"], axis=1))) if committed > 1 else 0.,
                saturated_holds=int(np.sum(history["saturated"])),
                bus_orientation_independent_spherical_envelope_radius_m=float(properties[key].body_radius_m))
        success = bool(complete and all(value["passed"] for value in bus_audit.values()))
        if complete and not success:
            message = "final_chaser_attitude_or_spin_gate_failed"
        assumptions = (
            "decoupled rotational replay; saved translational trajectory is not recomputed with attitude or fuel dynamics",
            "constant initial wet-bus inertia; total mass debit is bounded by the configured small-depletion fraction",
            "exact bus attitude/rate feedback, synchronized target preview, continuous throttling and ideal paired nozzles",
            "no plume interaction, external disturbance torque, slosh, flexible structure, sensor or thruster errors",
            "known spherical bus envelope is orientation independent; gripper is stowed during approach and its deployment sweep is not resolved",
            "translation propellant is a caller-supplied conservative debit; no translational actuator/fuel reconstruction is claimed",
            "SLERP pose and separately interpolated target rates form a numerical tracking reference, not certified physical target truth",
            "numerical RK4 with normalized quaternions; no interval integration or general saturated tracking theorem")
        audit = dict(model="held_geometric_bus_attitude_with_nozzle_couples_v1", reference=reference.to_dict(),
                     actuators=self.actuators.to_dict(), buses=bus_audit, timeline_complete=complete,
                     requested_end_time_s=float(times[-1]), committed_end_time_s=float(times[committed - 1]),
                     control_period_s=float(self.options.control_period_s),
                     max_integration_step_s=float(self.options.max_integration_step_s),
                     natural_frequency_rad_s=float(self.options.natural_frequency_rad_s),
                     damping_ratio=float(self.options.damping_ratio),
                     default_initial_misalignment_rotvec_rad=self.options.initial_misalignment_rotvec_rad.tolist(),
                     explicit_initial_states=initial_states is not None,
                     inertia_model="constant initial wet-bus inertia with enforced maximum total mass debit",
                     max_final_orientation_error_rad=float(self.options.max_final_orientation_error_rad),
                     max_final_spin_error_rad_s=float(self.options.max_final_spin_error_rad_s),
                     max_constant_inertia_mass_change_fraction=float(self.options.max_constant_inertia_mass_change_fraction),
                     approach_node_indices=np.searchsorted(times[:committed], reference.times_s[reference.times_s <= times[committed - 1]]).tolist(),
                     capture_requires_successful_attitude_result=True)
        return ChaserAttitudeResult(success, message, times[:committed].copy(), histories, audit, assumptions, final_properties)
