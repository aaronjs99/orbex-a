"""Shared recorded-sample alignment and saved target kinematics."""
from __future__ import annotations
from typing import List, Optional, Tuple
from collections.abc import Mapping
from types import SimpleNamespace
import numpy as np
from orbexa.simulation.approach.records.results import ADTMPCMissionResult
from orbexa.utils.math_utils import tait_bryan_to_rotation_matrix


def recorded_array(value) -> np.ndarray:
    """Read a numeric history without turning malformed optional data into states."""
    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return np.empty(0)

class MissionSamples:
    """Aligned recorded states and saved target kinematics shared by renderers.

    Missing histories are never extrapolated from a new scenario. A saved target's
    own constant orientation-rate model may reconstruct missing target samples;
    this fallback is recorded in ``notes``.
    """

    def __init__(self, result: ADTMPCMissionResult):
        if isinstance(result, Mapping):
            result = SimpleNamespace(**{
                "metadata": {}, "anom_history": [], "actual_trajectories": {},
                "sample_phase_history": [], "target_attitude_history": [],
                "target_angular_velocity_history": [], "target": {},
                "docking_points": {}, "nominal_trajectories": {}, "tube_diagnostics": {},
                **result,
            })
        self.result = result
        self.target = result.target or {}
        self.metadata = result.metadata or {}
        self.notes: List[str] = []
        q = recorded_array(result.anom_history).reshape(-1)
        if len(q) and not np.all(np.isfinite(q)):
            q = q[:int(np.flatnonzero(~np.isfinite(q))[0])]
            self.notes.append("Views stop before the first nonfinite recorded anomaly sample.")
        self.recorded_q = q
        trajectories = {}
        for identifier, values in result.actual_trajectories.items():
            values = recorded_array(values)
            if values.ndim == 2 and values.shape[1] >= 6 and len(values):
                values = values[:, :6]
                finite = np.all(np.isfinite(values), axis=1)
                if not np.all(finite):
                    values = values[:int(np.flatnonzero(~finite)[0])]
                    self.notes.append("{} stops before its first nonfinite recorded state.".format(identifier))
                if len(values):
                    trajectories[identifier] = values
                else:
                    self.notes.append("No usable recorded state trajectory for {}.".format(identifier))
            else:
                self.notes.append("No usable recorded state trajectory for {}.".format(identifier))
        count = min([len(q)] + [len(values) for values in trajectories.values()]) if trajectories else 0
        self.ids = list(trajectories)
        self.q = q[:count]
        self.states = np.stack([values[:count] for values in trajectories.values()], axis=1) if count else np.empty((0, len(self.ids), 6))
        if count != len(q) or any(len(values) != count for values in trajectories.values()):
            self.notes.append("Views use the {} state samples shared by the saved anomaly and trajectory histories.".format(count))
        self.title = "Single chaser" if len(self.ids) == 1 else "{} chasers".format(len(self.ids))
        phases = list(result.sample_phase_history or self.metadata.get("sample_phase_history", []))
        self.phases = (phases + ["unknown"] * count)[:count]
        docking = [index for index, phase in enumerate(self.phases) if phase == "docking"]
        self.docking_start = docking[0] if docking else None
        self.recorded_orientation = self._target_history(result.target_attitude_history, "initial_orientation", "orientation", q=self.recorded_q)
        self.recorded_rates = self._target_history(result.target_angular_velocity_history, "angular_velocity", "rate", q=self.recorded_q)
        self.orientation = None if self.recorded_orientation is None or not count else self.recorded_orientation[:count]
        self.rates = None if self.recorded_rates is None or not count else self.recorded_rates[:count]
        if count and self.orientation is None:
            self.orientation = self._target_history(result.target_attitude_history, "initial_orientation", "orientation")
        if count and self.rates is None:
            self.rates = self._target_history(result.target_angular_velocity_history, "angular_velocity", "rate")
        self.rotations = None if self.orientation is None else np.stack([tait_bryan_to_rotation_matrix(value) for value in self.orientation])
        self.rotation_primes = None
        kinematics = self.metadata.get("target_kinematics", {})
        if isinstance(kinematics, Mapping) and kinematics:
            recorded_anomaly = recorded_array(kinematics.get("anomaly", []))
            rotations = recorded_array(kinematics.get("rotations", []))
            primes = recorded_array(kinematics.get("rotation_primes", []))
            aligned = (recorded_anomaly.ndim == 1 and len(recorded_anomaly) >= count and
                       np.array_equal(recorded_anomaly[:count], self.q))
            proper = (recorded_anomaly.ndim == 1 and rotations.shape == (len(recorded_anomaly), 3, 3) and
                      np.all(np.isfinite(rotations[:count])) and
                      np.allclose(rotations[:count].transpose(0, 2, 1) @ rotations[:count], np.eye(3), atol=1e-8) and
                      np.allclose(np.linalg.det(rotations[:count]), 1.0, atol=1e-8))
            if aligned and proper:
                self.rotations = rotations[:count]
                if primes.shape == rotations.shape and np.all(np.isfinite(primes[:count])):
                    self.rotation_primes = primes[:count]
            else:
                self.notes.append("Stored target kinematics are invalid or unaligned; only usable saved Euler geometry is shown.")
        if self.rotations is None and count:
            self.notes.append("Saved target orientation is unavailable; views omit the rotating target and docking geometry.")
        self.body = None if self.rotations is None else np.einsum("tji,tcj->tci", self.rotations, self.states[:, :, :3])
        self.goals_body = {}
        for identifier in self.ids:
            value = recorded_array((result.docking_points or {}).get(identifier, [])).reshape(-1)
            if value.shape == (3,) and np.all(np.isfinite(value)):
                self.goals_body[identifier] = value
        self.position_error = self.velocity_error = None
        if count and self.rotations is not None and len(self.goals_body) == len(self.ids):
            goals = np.stack([self.goals_body[identifier] for identifier in self.ids])
            positions = np.einsum("tij,cj->tci", self.rotations, goals)
            self.position_error = np.linalg.norm(self.states[:, :, :3] - positions, axis=2)
            if self.rotation_primes is not None:
                velocities = np.einsum("tij,cj->tci", self.rotation_primes, goals)
                self.velocity_error = np.linalg.norm(self.states[:, :, 3:] - velocities, axis=2)
            elif self.rates is not None:
                from orbexa.control.mpc.constraints import rotating_body_point_velocity
                velocities = np.array([[rotating_body_point_velocity(goal, orientation, rate) for goal in goals]
                                       for orientation, rate in zip(self.orientation, self.rates)])
                self.velocity_error = np.linalg.norm(self.states[:, :, 3:] - velocities, axis=2)

    def _target_history(self, history, saved_key: str, kind: str, *, q=None) -> Optional[np.ndarray]:
        q = self.q if q is None else q
        if not len(q):
            return None
        values = recorded_array(history)
        if values.ndim == 2 and values.shape[1] == 3 and len(values) >= len(q) and np.all(np.isfinite(values[:len(q)])):
            return values[:len(q)]
        saved = recorded_array(self.target.get(saved_key, []))
        if saved.shape != (3,) or not np.all(np.isfinite(saved)):
            return None
        if kind == "orientation":
            rate = recorded_array(self.target.get("angular_velocity", []))
            if rate.shape != (3,) or not np.all(np.isfinite(rate)):
                return None
            reconstructed = saved + q[:, None] * rate
        else:
            reconstructed = np.repeat(saved[None, :], len(q), axis=0)
        # Preserve finite recorded rows when a saved model completes a partial
        # history. Never extend a final recorded pose by holding it constant.
        if values.ndim == 2 and values.shape[1] == 3:
            count = min(len(values), len(q))
            finite = np.all(np.isfinite(values[:count]), axis=1)
            reconstructed[np.flatnonzero(finite)] = values[:count][finite]
        if not np.all(np.isfinite(reconstructed)):
            return None
        note = "Target {} samples reconstructed from the model stored in this result.".format(kind)
        if note not in self.notes:
            self.notes.append(note)
        return reconstructed

    def phase_switch_indices(self) -> List[int]:
        return [index for index in range(1, len(self.q))
                if self.phases[index] != self.phases[index - 1]]

    def frame_indices(self, limit: Optional[int] = None, *, start: int = 0) -> List[int]:
        """Select recorded stops, always retaining transitions and the final row."""
        count = len(self.q)
        if start < 0 or start >= count:
            return []
        number = count - start if limit is None else min(limit, count - start)
        indices = np.linspace(start, count - 1, max(number, 1)).astype(int).tolist()
        return sorted(set(indices + [start, count - 1] +
                          [index for index in self.phase_switch_indices() if index >= start]))

    def matched_tube(self, identifier: str):
        """Return this chaser's aligned nominal center and component tube bounds."""
        entry = (self.result.tube_diagnostics or {}).get(identifier, {})
        if not isinstance(entry, Mapping):
            return None
        center, omega = (recorded_array(entry.get(key, [])) for key in ("nominal_center", "omega"))
        count = len(self.q)
        valid = count > 0 and center.ndim == 2 and omega.ndim == 2 and center.shape[1] >= 3 and omega.shape[1] >= 3
        if not valid or len(center) < count or len(omega) < count:
            return None
        center, omega = center[:count, :3], omega[:count, :3]
        if not np.all(np.isfinite(center)) or not np.all(np.isfinite(omega)) or np.any(omega < 0):
            return None
        return center, omega

    def nominal_horizons(self, identifier: str):
        for history in (self.result.nominal_trajectories or {}).get(identifier, []) or []:
            values = recorded_array(history)
            if values.ndim == 2 and values.shape[1] >= 3 and len(values) and np.all(np.isfinite(values[:, :3])):
                yield values[:, :3]

    def cylinder(self) -> Optional[Tuple[float, float]]:
        target = self.target
        if target.get("shape", "cylinder") != "cylinder":
            return None
        try:
            radius = float(target["radius"]) * (1.0 + float(target.get("tolerance", 0.0)))
            half_length = float(target.get("half_length", float(target.get("height", 0.0)) / 2.0)) * (1.0 + float(target.get("tolerance", 0.0)))
            if radius > 0 and half_length > 0 and np.all(np.isfinite((radius, half_length))):
                return radius, half_length
        except (KeyError, TypeError, ValueError):
            pass
        return None
