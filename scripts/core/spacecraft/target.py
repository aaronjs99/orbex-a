# /***********************************************************
# *                                                         *
# * Copyright (c) 2026                                      *
# *                                                         *
# * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
# * University of California, Los Angeles                   *
# *                                                         *
# * Authors: Aaron John Sabu                                *
# * Contact: aaronjs@ucla.edu                               *
# *                                                         *
# ***********************************************************/


"""Standalone target rigid-body propagation; prescribed mission motion is separate."""
from __future__ import annotations

from itertools import count
from typing import TYPE_CHECKING

import numpy as np
from orbexa.core.spacecraft.base import Spacecraft
from orbexa.core.dynamics.attitude import RigidBodyAttitude, euler_rates_to_body_rates, euler_to_quaternion, evaluate_torque, finite_vector, quaternion_to_euler

if TYPE_CHECKING:
    from orbexa.core.configuration.model import SimulationConfig


class Target(Spacecraft):
    """Standalone rigid body, independent of ADTMPC's prescribed target motion.

    ``curr_state`` remains three roll/pitch/yaw angles for deflection consumers;
    ``angular_velocity`` is three body rates in rad/s. ``attitude_state`` combines
    these into six values. A scalar-last quaternion is the integration state.

    Configured Euler derivatives per anomaly are converted at initial anomaly
    zero to Euler derivatives per second, then to body rates. Default ``dt`` is
    config.anom_step divided by that initial anomaly rate, in seconds; it remains
    a fixed standalone sample period. ``angularVelocity`` overrides body rad/s,
    and ``initState`` accepts either three angles or six angles/body-rate values.
    """

    _tids = count(0)

    def __init__(self, config: "SimulationConfig", **kwargs):
        from orbexa.core.dynamics.anomaly import orbital_anomaly_rate

        orbit = config.orbit
        initial_anomaly_rate = float(orbital_anomaly_rate(
            0.0, eccentricity=orbit.eccentricity, alpha=orbit.target_drag,
            mu=orbit.mu, semi_major_axis=orbit.semi_major_axis,
            mean_motion=orbit.mean_motion, specific_angular_momentum=orbit.specific_angular_momentum))
        initial = np.asarray(kwargs.get("initState", config.target.initial_orientation), dtype=float)
        if initial.shape not in ((3,), (6,)) or not np.all(np.isfinite(initial)):
            raise ValueError("Target initState must contain three Euler angles or six angles/body-rate values")
        orientation = initial[:3].copy()
        body_rates = kwargs.get("angularVelocity", initial[3:] if len(initial) == 6 else
                                euler_rates_to_body_rates(orientation, config.target.initial_angular_velocity * initial_anomaly_rate))
        if kwargs.get("numStates", 3) not in (3, 6):
            raise ValueError("Target has three Euler angles and three body angular rates")
        kwargs = dict(kwargs, initState=orientation, numStates=3,
                      dt=kwargs.get("dt", config.anom_step / initial_anomaly_rate))
        self.tid = next(self._tids) + 1
        super().__init__(config, **kwargs)

        self.observation_error_ang_pos = config.target.observation_error[
            "angular_position"
        ]
        self.observation_error_ang_vel = config.target.observation_error[
            "angular_velocity"
        ]

        self.angular_velocity = finite_vector(body_rates, "Body angular velocity", 3)
        self.attitude_dynamics = RigidBodyAttitude(kwargs.get("momInertia", config.target.inertia),
                                                  rtol=kwargs.get("rtol", 1e-10), atol=kwargs.get("atol", 1e-12))
        self.mom_inertia = self.attitude_dynamics.inertia.copy()
        self.geometry = kwargs.get("geometry", {"Ineqs": [], "Eqs": []})
        self.initial_anomaly_rate = initial_anomaly_rate
        self.quaternion = euler_to_quaternion(self.curr_state)
        self.quaternion_history = [self.quaternion.copy()]
        self.angular_velocity_history = [self.angular_velocity.copy()]
        self.time_history = [0.0]

    @property
    def attitude_state(self):
        """Six-value attitude observation: Euler angles followed by body rates."""
        return np.concatenate((self.curr_state, self.angular_velocity))

    @property
    def attitude_state_history(self):
        return np.column_stack((self.state_history, self.angular_velocity_history))

    def update_geometry(self, **kwargs):
        if "geometryIneqs" in kwargs:
            self.geometry["Ineqs"] = kwargs["geometryIneqs"]
        if "geometryEqs" in kwargs:
            self.geometry["Eqs"] = kwargs["geometryEqs"]
        return self.geometry

    def update_state(self, **kwargs):
        """Advance endpoint samples or append an explicit angles/body-rate state.

        For N intervals, given torque accepts N held interval rows, or N+1 node
        rows including time zero. Node row k+1 drives interval k; the initial
        node has no preceding interval. This aligns deflection's ending-node
        force samples with the standalone held-torque propagation contract.
        """
        allowed = {"newAngularPos", "newAngularVel", "numSteps", "torqueVal", "torqueType"}
        if set(kwargs) - allowed:
            raise TypeError(f"Unsupported Target update arguments: {sorted(set(kwargs) - allowed)}")
        if "newAngularPos" in kwargs or "newAngularVel" in kwargs:
            if not {"newAngularPos", "newAngularVel"}.issubset(kwargs) or len(kwargs) != 2:
                raise ValueError("Explicit state updates require only newAngularPos and newAngularVel together")
            angles = finite_vector(kwargs["newAngularPos"], "Euler angles", 3)
            rates = finite_vector(kwargs["newAngularVel"], "Body angular velocity", 3)
            quaternions = np.array([euler_to_quaternion(angles)])
            angles = np.array([angles])
            rates = np.array([rates])
            times = np.array([self.dt])
        else:
            num_steps = kwargs.get("numSteps", 1)
            if isinstance(num_steps, bool) or not isinstance(num_steps, (int, np.integer)) or num_steps < 0:
                raise ValueError("numSteps must be a nonnegative integer")
            if num_steps == 0:
                return self.state_history
            torque_type = kwargs.get("torqueType", "zero")
            torque_val = kwargs.get("torqueVal")
            if torque_type == "given":
                torques = np.asarray(torque_val, dtype=float)
                if (torques.shape not in ((num_steps, 3), (num_steps + 1, 3))
                        or not np.all(np.isfinite(torques))):
                    raise ValueError("Given torque requires N interval rows or N+1 ending-node rows of finite body-frame three-vectors")
                interval_torques = torques[1:] if len(torques) == num_steps + 1 else torques
                # Resolve each constant-torque interval separately so solver stages
                # cannot interpolate or skip a discontinuity in the torque sequence.
                pieces = []
                quaternion, rate = self.quaternion.copy(), self.angular_velocity.copy()
                for torque in interval_torques:
                    piece = self.attitude_dynamics.propagate(quaternion, rate, [self.dt],
                                                            torque=lambda t, q, w, value=torque: value)
                    pieces.append(piece)
                    quaternion, rate = piece.quaternions[-1], piece.body_rates[-1]
                quaternions = np.vstack([piece.quaternions for piece in pieces])
                rates = np.vstack([piece.body_rates for piece in pieces])
                times = self.dt * np.arange(1, num_steps + 1)
                angles = np.array([quaternion_to_euler(value) for value in quaternions])
            else:
                if torque_type not in {"zero", "function"}:
                    raise ValueError("torqueType must be 'zero', 'given' or 'function'")
                if torque_type == "function" and not callable(torque_val):
                    raise ValueError("Function torque requires a callable")

                def torque(t, quaternion, body_rate):
                    state = np.concatenate((quaternion_to_euler(quaternion), body_rate))
                    return evaluate_torque(t, state, self.dt, self.mom_inertia, torque_val, torque_type)

                piece = self.attitude_dynamics.propagate(self.quaternion, self.angular_velocity,
                                                        self.dt * np.arange(1, num_steps + 1),
                                                        torque=None if torque_type == "zero" else torque)
                quaternions, rates, times, angles = piece.quaternions, piece.body_rates, piece.times, piece.euler_angles
        start_time = self.time_history[-1]
        self.quaternion_history.extend(value.copy() for value in quaternions)
        self.angular_velocity_history.extend(value.copy() for value in rates)
        self.state_history.extend(value.copy() for value in angles)
        self.time_history.extend(float(start_time + value) for value in times)
        self.quaternion = self.quaternion_history[-1].copy()
        self.angular_velocity = self.angular_velocity_history[-1].copy()
        self.curr_state = self.state_history[-1].copy()
        return self.state_history

    def _observation_index(self, t):
        if not np.isfinite(t) or t < 0:
            raise ValueError("Observation time must be finite and nonnegative")
        index = int(np.floor(t / self.dt + 1e-12))
        if index >= len(self.state_history):
            self.update_state(numSteps=index - len(self.state_history) + 1)
        return index

    def get_observed_state(self, t: float = 0.0):
        """Observe Euler angles at the preceding sample of elapsed seconds."""
        state = self.state_history[self._observation_index(t)]
        return state * self.rng.gauss(1.00, self.observation_error_ang_pos)

    def get_observed_ang_vel(self, t: float = 0.0):
        """Observe body angular velocity in rad/s at an elapsed-second sample."""
        ang_vel = self.angular_velocity_history[self._observation_index(t)]
        return ang_vel * self.rng.gauss(1.00, self.observation_error_ang_vel)

    def get_mom_inertia(self):
        return self.mom_inertia.copy()
