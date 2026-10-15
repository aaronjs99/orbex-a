"""Injectable vehicle sizing for composite-target mission studies."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from orbexa.core.spacecraft_geometry import SpacecraftGeometry
from orbexa.core.spacecraft_target import RigidBodySpacecraftTarget
from orbexa.simulation.capture import CaptureOptions, ChaserPhysicalProperties, TargetPhysicalProperties
from orbexa.simulation.chaser_attitude import ChaserAttitudeActuators, ChaserAttitudeTracker
from orbexa.simulation.complete_mission import CompleteMissionOptions, CompleteMissionRunner
from orbexa.simulation.deorbit import DeorbitSimulator, ThrustPointingPolicy
from orbexa.simulation.detumbling import DetumblingSimulator, RateDampingPolicy


@dataclass(frozen=True)
class SpacecraftMissionDesign:
    """Declared chaser concept; published target identity does not qualify it.

    Eight small12kg buses cannot dispose of an8t spacecraft with4kg of fuel
    each. These larger chemical-propulsion buses provide a reviewable sizing
    case. No unrecorded resizing occurs after a failed simulation.
    """
    geometry: SpacecraftGeometry
    chaser: ChaserPhysicalProperties = field(default_factory=lambda: ChaserPhysicalProperties(
        dry_mass_kg=30.,fuel_kg=100.,body_radius_m=.55,gripper_offset_m=.8))
    com_standoff_m: float = .85
    arm_reach_m: float = .10
    translation_force_n: float = 100.
    attitude_nozzle_force_n: float = 3.
    attitude_moment_arm_m: float = .4
    specific_impulse_s: float = 300.
    detumbling_gain_nm_s: float = 2000.
    pointing_attitude_gain_nm: float = 500.
    pointing_rate_gain_nm_s: float = 8000.

    def __post_init__(self):
        if not isinstance(self.geometry,SpacecraftGeometry):
            raise TypeError('Vehicle design requires explicit SpacecraftGeometry')
        if not isinstance(self.chaser,ChaserPhysicalProperties):
            raise TypeError('Vehicle design requires explicit ChaserPhysicalProperties')
        for name in ('com_standoff_m','arm_reach_m','translation_force_n','attitude_nozzle_force_n',
                     'attitude_moment_arm_m','specific_impulse_s','detumbling_gain_nm_s',
                     'pointing_attitude_gain_nm','pointing_rate_gain_nm_s'):
            value=getattr(self,name)
            if isinstance(value,bool) or not np.isfinite(value) or value<=0:
                raise ValueError(f'{name} must be finite and positive')
        if self.com_standoff_m<=self.chaser.body_radius_m:
            raise ValueError('Free-flight COM standoff must clear the finite bus')
        if self.attitude_moment_arm_m>self.chaser.body_radius_m:
            raise ValueError('Attitude nozzle positions must fit the bus envelope')
        properties=self.geometry.metadata.get('mass_properties',{})
        if not all(key in properties for key in ('nominal_mass_kg','inertia_kg_m2')):
            raise ValueError('Declare source-qualified target mass and inertia in geometry metadata')
        TargetPhysicalProperties(properties['nominal_mass_kg'],properties['inertia_kg_m2'])

    @classmethod
    def from_json(cls,path:Path,**design_options):
        return cls(geometry=SpacecraftGeometry.from_json(path),**design_options)

    @classmethod
    def from_dict(cls,geometry,record):
        if (record.get('geometry_fingerprint')!=geometry.fingerprint
                or record.get('geometry_id')!=geometry.geometry_id):
            raise ValueError('Saved vehicle design does not match the recorded target geometry')
        names=('com_standoff_m','arm_reach_m','translation_force_n','attitude_nozzle_force_n',
            'attitude_moment_arm_m','specific_impulse_s','detumbling_gain_nm_s',
            'pointing_attitude_gain_nm','pointing_rate_gain_nm_s')
        design=cls(geometry=geometry,chaser=ChaserPhysicalProperties(**record['chaser']),
                   **{name:record[name] for name in names})
        if design.target_mass_kg!=record.get('target_mass_kg'):
            raise ValueError('Saved design changes the source-qualified target mass')
        return design

    @property
    def target_mass_kg(self):
        return float(self.geometry.metadata['mass_properties']['nominal_mass_kg'])

    def target(self,config,*,initial_body_spin_rad_s,anomaly_domain,gravity_gradient=True):
        return RigidBodySpacecraftTarget.from_config(config,geometry=self.geometry,mass_kg=self.target_mass_kg,
            inertia=np.asarray(self.geometry.metadata['mass_properties']['inertia_kg_m2']),
            initial_body_angular_velocity=initial_body_spin_rad_s,anomaly_domain=anomaly_domain,
            gravity_gradient=gravity_gradient,docking_standoff=self.com_standoff_m)

    def completion_runner(self,*,input_component_bound,approach_runner=None,stage_sink=None):
        tracker=ChaserAttitudeTracker(default_chaser=self.chaser,actuators=ChaserAttitudeActuators(
            moment_arm_m=self.attitude_moment_arm_m,nozzle_force_limit_n=self.attitude_nozzle_force_n,
            specific_impulse_s=self.specific_impulse_s))
        return CompleteMissionRunner(options=CompleteMissionOptions(input_component_bound=input_component_bound,
            target_mass_kg=self.target_mass_kg,translation_force_n=self.translation_force_n,
            specific_impulse_s=self.specific_impulse_s),approach_runner=approach_runner,attitude_tracker=tracker,
            capture_options=CaptureOptions(arm_reach_m=self.arm_reach_m),
            detumbling=DetumblingSimulator(policy=RateDampingPolicy(self.detumbling_gain_nm_s)),
            deorbit=DeorbitSimulator(pointing_policy=ThrustPointingPolicy(
                self.pointing_attitude_gain_nm,self.pointing_rate_gain_nm_s)),stage_sink=stage_sink)

    def as_dict(self):
        return dict(geometry_id=self.geometry.geometry_id,geometry_fingerprint=self.geometry.fingerprint,
            target_mass_kg=self.target_mass_kg,chaser=dict(dry_mass_kg=self.chaser.dry_mass_kg,
                fuel_kg=self.chaser.fuel_kg,body_radius_m=self.chaser.body_radius_m,
                gripper_offset_m=self.chaser.gripper_offset_m),
            **{name:getattr(self,name) for name in ('com_standoff_m','arm_reach_m','translation_force_n',
                'attitude_nozzle_force_n','attitude_moment_arm_m','specific_impulse_s','detumbling_gain_nm_s',
                'pointing_attitude_gain_nm','pointing_rate_gain_nm_s')},
            scope='Declared study vehicle concept and actuator limits; no manufactured spacecraft or propulsion qualification')
