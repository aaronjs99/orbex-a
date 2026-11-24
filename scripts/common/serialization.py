"""Source-qualified numerical continuation of a committed mission record.

This is a numerical restart, not the admissible safe-backup theorem. Actual
states, persistent nominal centers, tube widths/gains, belief, clock, phase and
assignments are retained. Unavailable full control warm starts and dense SMID
measurements are explicitly reset; neither is reconstructed from mean inputs.
No callables or pickle payloads are serialized.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

CHECKPOINT_SCHEMA = "orbexa.mission_checkpoint.v1"


def plain(value):
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def digest(value):
    return hashlib.sha256(
        json.dumps(
            plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _target_signature(metadata):
    names = (
        "shape",
        "radius",
        "height",
        "half_length",
        "tolerance",
        "docking_standoff",
        "initial_orientation",
        "angular_velocity",
        "attitude_model",
        "motion_model",
        "gravity_gradient",
        "initial_quaternion",
        "initial_body_angular_velocity_rad_s",
        "inertia_kg_m2",
        "clock_parameters",
        "anomaly_domain",
        "integration",
        "geometry",
        "geometry_fingerprint",
        "geometry_model",
        "mass_kg",
    )
    return {name: metadata[name] for name in names if name in metadata}


def physical_target_from_snapshot(snapshot):
    """Reconstruct an explicitly identified physical shape without substitution.

    Composite geometry is loaded by its own validated target adapter.  The
    cylinder branch is retained solely for previously declared cylinder runs.
    Missing or unsupported shapes never select that branch implicitly.
    """
    shape = snapshot.get("shape")
    if shape == "composite_spacecraft":
        from orbexa.models.targets.spacecraft import RigidBodySpacecraftTarget

        return RigidBodySpacecraftTarget.from_dict(snapshot)
    if shape != "cylinder":
        raise ValueError(
            "Unsupported physical target snapshot geometry; no cylinder fallback is available"
        )
    from orbexa.models.targets.cylinder import RigidBodyCylinderTarget

    clock, integration = snapshot["clock_parameters"], snapshot["integration"]
    return RigidBodyCylinderTarget(
        radius=snapshot["radius"],
        half_length=snapshot["half_length"],
        initial_quaternion=snapshot["initial_quaternion"],
        initial_body_angular_velocity=snapshot["initial_body_angular_velocity_rad_s"],
        inertia=snapshot["inertia_kg_m2"],
        eccentricity=clock["eccentricity"],
        alpha=clock["alpha"],
        mu=clock["mu"],
        semi_major_axis=clock["semi_major_axis"],
        anomaly_domain=snapshot["anomaly_domain"],
        tolerance=snapshot["tolerance"],
        docking_standoff=snapshot["docking_standoff"],
        gravity_gradient=snapshot["gravity_gradient"],
        integration_rtol=integration["rtol"],
        integration_atol=integration["atol"],
        integration_max_step=integration["max_anomaly_step"],
    )
