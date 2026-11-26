"""snapshot.py: Physical target reconstruction from recorded snapshots and target signatures.

Copyright: Aaron John Sabu
"""

from __future__ import annotations


def _target_signature(metadata):
    """Return the subset of target metadata keys that identify the physical target.

    Only the named keys present in metadata are copied.
    """
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
