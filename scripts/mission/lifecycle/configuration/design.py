"""design.py: Injectable vehicle sizing for composite-target mission studies.

Copyright: Aaron John Sabu
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
from orbexa.mission.approach.physical.attitude import (
    ChaserAttitudeActuators,
    ChaserAttitudeTracker,
)
from orbexa.mission.attached.assembly.model import positive
from orbexa.mission.attached.detumbling import DetumblingSimulator, RateDampingPolicy
from orbexa.mission.attached.disposal.axis import AxialBurnPlanner
from orbexa.mission.attached.disposal.preview import FiniteBurnThrottlePlanner
from orbexa.mission.attached.disposal.simulator import (
    DeorbitSimulator,
    ThrustPointingPolicy,
)
from orbexa.mission.contact.capture import (
    CaptureOptions,
    ChaserPhysicalProperties,
    TargetPhysicalProperties,
)
from orbexa.mission.lifecycle.coordinator import (
    CompleteMissionOptions,
    CompleteMissionRunner,
)
from orbexa.models.geometry.composite import SpacecraftGeometry
from orbexa.models.targets.spacecraft import RigidBodySpacecraftTarget


@dataclass(frozen=True)
class SpacecraftMissionDesign:
    """Explicit composite geometry, target mass/inertia and chaser hardware.

    Defaults retain the existing Envisat study concept. A mission-wrapper JSON
    declares every vehicle field so another target never silently inherits it.
    No resizing or strength is inferred from geometry or after a failed run.
    """

    geometry: SpacecraftGeometry
    chaser: ChaserPhysicalProperties = field(
        default_factory=lambda: ChaserPhysicalProperties(
            dry_mass_kg=30.0, fuel_kg=100.0, body_radius_m=0.55, gripper_offset_m=0.8
        )
    )
    com_standoff_m: float = 0.85
    arm_reach_m: float = 0.10
    translation_force_n: float = 100.0
    attitude_nozzle_force_n: float = 3.0
    attitude_moment_arm_m: float = 0.4
    specific_impulse_s: float = 300.0
    detumbling_gain_nm_s: float = 2000.0
    pointing_attitude_gain_nm: float = 500.0
    pointing_rate_gain_nm_s: float = 8000.0

    def __post_init__(self):
        """Validate the vehicle design.

        Raises TypeError for a non-explicit geometry or chaser, and ValueError when a
        sizing value is not positive, the COM standoff does not clear the bus, the
        nozzle arm exceeds the bus radius, the carried tip is outside the patch or its
        reach, or the geometry metadata lacks target mass and inertia.
        """
        if not isinstance(self.geometry, SpacecraftGeometry):
            raise TypeError("Vehicle design requires explicit SpacecraftGeometry")
        if not isinstance(self.chaser, ChaserPhysicalProperties):
            raise TypeError("Vehicle design requires explicit ChaserPhysicalProperties")
        for name in (
            "com_standoff_m",
            "arm_reach_m",
            "translation_force_n",
            "attitude_nozzle_force_n",
            "attitude_moment_arm_m",
            "specific_impulse_s",
            "detumbling_gain_nm_s",
            "pointing_attitude_gain_nm",
            "pointing_rate_gain_nm_s",
        ):
            object.__setattr__(self, name, positive(getattr(self, name), name))
        if self.com_standoff_m <= self.chaser.body_radius_m:
            raise ValueError("Free-flight COM standoff must clear the finite bus")
        if self.attitude_moment_arm_m > self.chaser.body_radius_m:
            raise ValueError("Attitude nozzle positions must fit the bus envelope")
        if (
            not 0
            <= self.com_standoff_m - self.chaser.gripper_offset_m
            <= self.arm_reach_m
        ):
            raise ValueError(
                "Nominal COM standoff must put the carried tip outside the patch and within its declared reach"
            )
        properties = self.geometry.metadata.get("mass_properties", {})
        if not all(key in properties for key in ("nominal_mass_kg", "inertia_kg_m2")):
            raise ValueError(
                "Declare source-qualified target mass and inertia in geometry metadata"
            )
        TargetPhysicalProperties(
            properties["nominal_mass_kg"], properties["inertia_kg_m2"]
        )

    @classmethod
    def from_json(cls, path: Path, **design_options):
        """Load a vehicle design from a spacecraft JSON file.

        A mission-wrapper record (schema orbexa.spacecraft-mission.v1) must declare
        every vehicle_design field and chaser property and cannot be combined with
        keyword overrides. Any other object is read as bare geometry, with
        ``design_options`` supplying the vehicle fields.

        Inputs:
            path: JSON file.
            design_options: vehicle field values for the bare-geometry form.

        Outputs:
            SpacecraftMissionDesign. Raises ValueError for malformed or incomplete
            declarations.
        """
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise ValueError("Spacecraft JSON must declare an object")
        if record.get("schema") == "orbexa.spacecraft-mission.v1":
            if (
                set(record) - {"schema", "geometry", "vehicle_design", "description"}
                or not isinstance(record.get("geometry"), dict)
                or not isinstance(record.get("vehicle_design"), dict)
            ):
                raise ValueError(
                    "Mission spacecraft JSON requires geometry and explicit vehicle_design"
                )
            names = {item.name for item in fields(cls)} - {"geometry"}
            declared = dict(record["vehicle_design"])
            if set(declared) != names:
                raise ValueError(
                    "Mission vehicle_design must explicitly declare: "
                    + ", ".join(sorted(names))
                )
            if set(declared) & set(design_options):
                raise ValueError(
                    "Vehicle fields cannot be declared both in a mission JSON and keyword overrides"
                )
            if not isinstance(declared["chaser"], Mapping):
                raise ValueError(
                    "Mission vehicle_design.chaser requires explicit mass, fuel and bus geometry"
                )
            chaser_names = {item.name for item in fields(ChaserPhysicalProperties)}
            if set(declared["chaser"]) != chaser_names:
                raise ValueError(
                    "Mission vehicle_design.chaser must explicitly declare: "
                    + ", ".join(sorted(chaser_names))
                )
            declared["chaser"] = ChaserPhysicalProperties(**declared["chaser"])
            return cls(
                geometry=SpacecraftGeometry.from_dict(record["geometry"]), **declared
            )
        return cls(geometry=SpacecraftGeometry.from_dict(record), **design_options)

    def validate_fleet(self, config, *, fallback_fuel_reserve_kg=None):
        """Validate declared sizing; geometric/dynamic proof may still refuse."""
        count = config.num_chasers
        if (
            isinstance(count, bool)
            or not isinstance(count, (int, np.integer))
            or count < 1
        ):
            raise ValueError("Declare a positive integer fleet size")
        if len(self.geometry.attachment_patches) < count:
            raise ValueError(
                f"Target {self.geometry.geometry_id} declares {len(self.geometry.attachment_patches)} attachment patches for {count} chasers"
            )
        if (
            fallback_fuel_reserve_kg is not None
            and positive(fallback_fuel_reserve_kg, "Fallback fuel reserve")
            > self.chaser.fuel_kg
        ):
            raise ValueError(
                "The declared fallback reserve exceeds each installed chaser tank"
            )
        return int(count)

    @classmethod
    def from_dict(cls, geometry, record):
        """Rebuild a design from its recorded dict for a given geometry.

        Raises ValueError when the record's geometry fingerprint or id differs from
        ``geometry``, or when the rebuilt target mass differs from the recorded one.
        """
        if (
            record.get("geometry_fingerprint") != geometry.fingerprint
            or record.get("geometry_id") != geometry.geometry_id
        ):
            raise ValueError(
                "Saved vehicle design does not match the recorded target geometry"
            )
        names = (
            "com_standoff_m",
            "arm_reach_m",
            "translation_force_n",
            "attitude_nozzle_force_n",
            "attitude_moment_arm_m",
            "specific_impulse_s",
            "detumbling_gain_nm_s",
            "pointing_attitude_gain_nm",
            "pointing_rate_gain_nm_s",
        )
        design = cls(
            geometry=geometry,
            chaser=ChaserPhysicalProperties(**record["chaser"]),
            **{name: record[name] for name in names},
        )
        if design.target_mass_kg != record.get("target_mass_kg"):
            raise ValueError("Saved design changes the source-qualified target mass")
        return design

    @property
    def target_mass_kg(self):
        """Return the source-qualified nominal target mass in kg from the geometry
        metadata.
        """
        return float(self.geometry.metadata["mass_properties"]["nominal_mass_kg"])

    def target(
        self, config, *, initial_body_spin_rad_s, anomaly_domain, gravity_gradient=True
    ):
        """Return the rigid-body target built from ``config`` and this design.

        Inputs:
            config: mission configuration; its fleet size is validated first.
            initial_body_spin_rad_s: initial body angular velocity in rad/s.
            anomaly_domain: anomaly domain passed to the target.
            gravity_gradient: whether gravity-gradient torque is modeled.

        Outputs:
            RigidBodySpacecraftTarget using the geometry, mass, inertia and the COM
            standoff as docking standoff.
        """
        self.validate_fleet(config)
        return RigidBodySpacecraftTarget.from_config(
            config,
            geometry=self.geometry,
            mass_kg=self.target_mass_kg,
            inertia=np.asarray(
                self.geometry.metadata["mass_properties"]["inertia_kg_m2"]
            ),
            initial_body_angular_velocity=initial_body_spin_rad_s,
            anomaly_domain=anomaly_domain,
            gravity_gradient=gravity_gradient,
            docking_standoff=self.com_standoff_m,
        )

    def completion_runner(
        self,
        *,
        input_component_bound,
        approach_runner=None,
        stage_sink=None,
        translation_accountant=None,
        completion_profile=None,
    ):
        """Return a CompleteMissionRunner configured from this design.

        Builds the attitude tracker, capture, detumbling and deorbit models from the
        declared hardware and gains, and takes the contact, detumbling and deorbit
        settings from the completion profile.

        Inputs:
            input_component_bound: held translational command bound in m/rad**2.
            approach_runner: optional approach runner for ``run``.
            stage_sink: optional callback receiving each stage result.
            translation_accountant: optional accountant replacing the default
                translation budget.
            completion_profile: None, profile or mapping for
                ``MissionCompletionProfile.from_declaration``.
        """
        from orbexa.mission.lifecycle.configuration.profile import (
            MissionCompletionProfile,
        )

        profile = MissionCompletionProfile.from_declaration(completion_profile)
        tracker = ChaserAttitudeTracker(
            default_chaser=self.chaser,
            actuators=ChaserAttitudeActuators(
                moment_arm_m=self.attitude_moment_arm_m,
                nozzle_force_limit_n=self.attitude_nozzle_force_n,
                specific_impulse_s=self.specific_impulse_s,
            ),
        )
        return CompleteMissionRunner(
            options=CompleteMissionOptions(
                input_component_bound=input_component_bound,
                target_mass_kg=self.target_mass_kg,
                translation_force_n=self.translation_force_n,
                specific_impulse_s=self.specific_impulse_s,
                finite_contact=profile.contact,
            ),
            approach_runner=approach_runner,
            attitude_tracker=tracker,
            capture_options=CaptureOptions(arm_reach_m=self.arm_reach_m),
            detumbling=DetumblingSimulator(
                options=profile.detumbling,
                policy=RateDampingPolicy(
                    self.detumbling_gain_nm_s, compensate_gravity_gradient=True
                ),
            ),
            deorbit=DeorbitSimulator(
                options=profile.deorbit,
                pointing_policy=ThrustPointingPolicy(
                    self.pointing_attitude_gain_nm,
                    self.pointing_rate_gain_nm_s,
                    compensate_gravity_gradient=True,
                ),
                burn_axis_planner=AxialBurnPlanner(),
                burn_throttle_planner=FiniteBurnThrottlePlanner(),
            ),
            stage_sink=stage_sink,
            translation_accountant=translation_accountant,
        )

    def as_dict(self):
        """Return the declared vehicle design as a dict with its scope statement."""
        return dict(
            geometry_id=self.geometry.geometry_id,
            geometry_fingerprint=self.geometry.fingerprint,
            target_mass_kg=self.target_mass_kg,
            chaser={
                "dry_mass_kg": self.chaser.dry_mass_kg,
                "fuel_kg": self.chaser.fuel_kg,
                "body_radius_m": self.chaser.body_radius_m,
                "gripper_offset_m": self.chaser.gripper_offset_m,
            },
            **{
                name: getattr(self, name)
                for name in (
                    "com_standoff_m",
                    "arm_reach_m",
                    "translation_force_n",
                    "attitude_nozzle_force_n",
                    "attitude_moment_arm_m",
                    "specific_impulse_s",
                    "detumbling_gain_nm_s",
                    "pointing_attitude_gain_nm",
                    "pointing_rate_gain_nm_s",
                )
            },
            scope="Declared study vehicle concept and actuator limits; no manufactured spacecraft or propulsion qualification",
        )
