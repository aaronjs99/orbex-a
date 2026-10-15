"""Portable, sample-aligned projection of a recorded mission for browser replay."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from orbexa.visualization.samples import MissionSamples, recorded_array

STUDIO_DATA_SCHEMA = "orbexa.mission.studio.data.v1"
COLORS = ("#5bd4bd", "#74b9ff", "#f5b567", "#ab99ee", "#9fd783", "#f48f9b", "#63d5e6", "#e3d77e")


def _history(entry, key, count, *, interval=False):
    """Missing diagnostics remain null; interval values belong to their END node."""
    values = recorded_array(entry.get(key, []))
    expected = count - 1 if interval else count
    if values.ndim != 1 or len(values) < expected:
        return [None] * count
    rows = [float(value) if np.isfinite(value) else None for value in values[:expected]]
    return [None] + rows if interval else rows


def _occupancy(error, bound):
    ratios = np.zeros_like(error)
    np.divide(np.abs(error), bound, out=ratios, where=bound > 0)
    ratios[(bound == 0) & (error != 0)] = np.inf
    return [float(value) if np.isfinite(value) else None for value in np.max(ratios, axis=1)]


@dataclass
class StudioDataset:
    """Exact recorded nodes plus explicitly identified derived display metrics.

    No optimizer, configuration, truth propagation or plotting library is loaded.
    ``null`` denotes unavailable measurements rather than a fabricated zero.
    """

    result: Any

    def to_dict(self):
        samples = MissionSamples(self.result)
        count = len(samples.q)
        if count < 1 or not samples.ids:
            raise ValueError("Mission Studio requires at least one recorded state sample")
        if np.any(np.diff(samples.q) <= 0):
            raise ValueError("Recorded anomaly samples must be strictly increasing")
        result, metadata = samples.result, samples.metadata
        elapsed = recorded_array(metadata.get("elapsed_time_seconds", []))
        clock = (elapsed.ndim == 1 and len(elapsed) >= count and
                 np.all(np.isfinite(elapsed[:count])) and np.all(np.diff(elapsed[:count]) > 0))
        tracks = []
        gain = recorded_array(metadata.get("lambda_gain", []))
        assignments = {item.get('chaser_id'): item for item in metadata.get('chaser_assignments', [])
                       if isinstance(item, dict)}
        for index, identifier in enumerate(samples.ids):
            diagnostics = (result.tube_diagnostics or {}).get(identifier, {})
            matched = samples.matched_tube(identifier)
            center = omega = phi = None
            occupancy = sliding_occupancy = [None] * count
            if matched is not None:
                center, omega = matched
                occupancy = _occupancy(samples.states[:, index, :3] - center, omega)
                raw_phi = recorded_array(diagnostics.get("phi", []))
                full_center = recorded_array(diagnostics.get("nominal_center", []))
                if (raw_phi.ndim == 2 and raw_phi.shape[1] == 3 and len(raw_phi) >= count
                        and np.all(np.isfinite(raw_phi[:count])) and np.all(raw_phi[:count] >= 0)):
                    phi = raw_phi[:count]
                    if full_center.ndim == 2 and full_center.shape[1] == 6 and len(full_center) >= count and gain.shape == (3,):
                        error = samples.states[:, index] - full_center[:count]
                        sliding_occupancy = _occupancy(error[:, 3:] + gain * error[:, :3], phi)
            assignment = assignments.get(identifier, {})
            point = recorded_array(assignment.get('docking_point_body', []))
            normal = recorded_array(assignment.get('docking_normal_body', []))
            standoff = assignment.get('docking_standoff')
            surface = None
            if (point.shape == (3,) and normal.shape == (3,) and np.all(np.isfinite(point))
                    and np.all(np.isfinite(normal)) and np.linalg.norm(normal) > 0
                    and isinstance(standoff, (int, float)) and not isinstance(standoff, bool)
                    and np.isfinite(standoff) and standoff >= 0):
                surface = point - standoff * normal / np.linalg.norm(normal)
            tracks.append({
                "id": identifier, "label": identifier.replace("_", " ").title(),
                "color": COLORS[index % len(COLORS)], "states": samples.states[:, index].tolist(),
                "center": None if center is None else center.tolist(),
                "omega": None if omega is None else omega.tolist(),
                "phi": None if phi is None else phi.tolist(),
                "goal_body": samples.goals_body.get(identifier, np.array([])).tolist() or None,
                "surface_point_body_m": None if surface is None else surface.tolist(),
                "goal_distance_m": [None] * count if samples.position_error is None else samples.position_error[:, index].tolist(),
                "tube_radius_m": [None] * count if omega is None else np.linalg.norm(omega, axis=1).tolist(),
                "position_occupancy": occupancy, "sliding_occupancy": sliding_occupancy,
                "input_reserve_ratio": _history(diagnostics, "max_dense_input_reserve_ratio", count, interval=True),
                "clearance_m": _history(diagnostics, "dense_physical_clearance", count, interval=True),
                "terminal_position_bound_m": _history(diagnostics, "terminal_position_bound", count, interval=True),
            })
        complete = bool(metadata.get("mission_complete", False))
        successful = bool(getattr(result, "success", False))
        synthetic = 'SYNTHETIC' in str(metadata.get('qualification_provenance', '')).upper()
        message = getattr(result, "message", "")
        update_limit = not complete and message == "max_mpc_updates_reached_before_terminal_tube_dwell"
        checkpoint = not complete and message == (
            "bounded_continuation_checkpoint; max_mpc_updates_reached_before_terminal_tube_dwell")
        return {
            "schema": STUDIO_DATA_SCHEMA, "q": samples.q.tolist(),
            "elapsed_s": elapsed[:count].tolist() if clock else None,
            "phases": samples.phases, "tracks": tracks,
            "rotations": None if samples.rotations is None else samples.rotations.tolist(),
            "target": samples.target, "notes": samples.notes,
            "chaser_body_radius_m": metadata.get('scenario', {}).get('chaser_body_bounding_radius_m', 0.0),
            "collision_envelope": None if samples.cylinder() is None else
                {"shape": "cylinder", "radius": samples.cylinder()[0], "half_length": samples.cylinder()[1],
                 "semantics": "Recorded target dimensions inflated by the saved geometric tolerance"},
            "outcome": {"success": successful, "complete": complete,
                        "label": "Synthetic physics fixture" if synthetic else "Recorded criteria passed" if complete and successful else
                                 "Recorded checkpoint" if checkpoint else
                                 "Recorded update limit reached" if update_limit else
                                 "Recorded failure" if not successful else "Recorded mission incomplete",
                        "message": message},
            "model": metadata.get("model_version", "unspecified"),
            "dynamics_scope": ("Analytical Carter–Humi relative translation; its declared uncertainty bounds do not automatically cover full nonlinear gravity, vector drag or J2. See Orbital model fidelity."
                               if metadata.get("model_version") == "physical-lvlh-v1" else
                               "Recorded model and assumptions apply to this saved case."),
            "controller": metadata.get("controller", "unspecified"),
            "synthetic": synthetic,
            "scope": metadata.get("guarantee_scope", "Saved model evidence; no physical certification"),
            "position_tolerance_m": metadata.get("docking_position_tolerance"),
            "min_separation_m": metadata.get("min_chaser_separation"),
            "metric_semantics": {
                "position_occupancy": "max component |actual position - own nominal center| / own Omega; zero over zero is zero; undefined unbounded ratios are null",
                "sliding_occupancy": "max component |e_v + Lambda e_p| / own Phi",
                "input_reserve_ratio": "maximum recorded dense ancillary/input-reserve ratio over the preceding interval; unavailable at initial node",
                "clearance_m": "minimum recorded dense physical clearance over the preceding interval; this is not a continuous-time minimum",
                "playback": "Only recorded nodes are displayed. Playback speed multiplies recorded elapsed seconds when present; otherwise anomaly is normalized to a 30 second presentation.",
                "surface_points": "Derived selected hull points from recorded COM standoff assignments; planning coordinates, not target docking hardware",
            },
        }
