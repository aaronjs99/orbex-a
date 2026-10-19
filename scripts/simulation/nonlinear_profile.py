"""Explicit prospective cases for sensed, nonlinear multi-spacecraft missions.

These are simulated hardware/environment declarations, not Envisat telemetry
or a sensor specification. Case construction has no access to future truth.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from fractions import Fraction
import copy
import numpy as np

from orbexa.core.reference_frame import ReferenceClockFrame, fingerprint
from orbexa.core.nonlinear_relative import NonlinearModelBounds, ExponentialAtmosphere
from orbexa.estimation.belief import ParameterBelief
from orbexa.estimation.navigation import StudySensorProfile
from orbexa.simulation.spacecraft_design import SpacecraftMissionDesign
from orbexa.simulation.composite_deployment import CompositeDeploymentOptions, CompositePatchDeployment
from orbexa.simulation.mission_safety import MissionSafety


@dataclass(frozen=True)
class NonlinearStudyOptions:
    release_distance_m: float = 300.
    release_pitch_m: float = 1.8
    nominal_spin_deg_s: float = 2.4
    initial_position_radius_m: float = .005
    initial_velocity_radius_m_s: float = .005
    lambda_per_rad: float = 300.
    # Common initial/fixed baseline selected from the sensed terminal budget;
    # every comparison arm starts at this same declared bandwidth.
    initial_bandwidth_per_rad: float = 1000.
    bandwidth_bounds_per_rad: tuple = (160.,1200.)
    bandwidth_rate_per_rad2: float = 6000.
    precontact_position_tolerance_m: float = .04
    # At the profile's rate ceiling, nu*(9+.04) remains below the
    # separately enforced 0.01 m/s physical contact-speed gate.
    precontact_velocity_tolerance_m_per_rad: float = 9.
    observation_position_floor_m: float = .0015
    observation_velocity_floor_m_per_rad: float = .5
    nominal_position_transfer_budget_m: float = 1e-9
    nominal_velocity_transfer_budget_m_per_rad: float = 1e-8
    sliding_peak_padding_m_per_rad: float = 1e-6
    position_peak_padding_m: float = 1e-7
    actuator_gain_error: float = .001
    actuator_axis_error_rad: float = .001
    actuator_bias_m_per_rad2: float = .2
    maximum_updates: int = 1200
    planning_workers: int = 8
    fallback_fuel_reserve_kg: float = 10.
    sensor: StudySensorProfile = StudySensorProfile()
    # Legacy keeps saved adaptive/frozen declarations unchanged. Enabled lets
    # both arms share initial orbit conditioning before online membership.
    initial_orbit_conditioning: str = 'legacy'

    def __post_init__(self):
        for key,value in vars(self).items():
            if key in ('sensor','bandwidth_bounds_per_rad','maximum_updates','planning_workers',
                       'initial_orbit_conditioning'):
                continue
            if isinstance(value,bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(key+' must be positive and finite')
        if (isinstance(self.maximum_updates,bool) or not isinstance(self.maximum_updates,int)
                or self.maximum_updates<1 or not isinstance(self.sensor,StudySensorProfile)):
            raise ValueError('Explicit sensor profile and update limit are required')
        if isinstance(self.planning_workers,bool) or not isinstance(self.planning_workers,int) or not 1<=self.planning_workers<=8:
            raise ValueError('Declare one to eight native GEKKO workers')
        if (not isinstance(self.initial_orbit_conditioning,str) or
                self.initial_orbit_conditioning not in ('legacy','enabled','disabled')):
            raise ValueError('Initial orbit conditioning must be legacy, enabled or disabled')
        lo,hi = self.bandwidth_bounds_per_rad
        if not 0 < lo <= self.initial_bandwidth_per_rad <= hi:
            raise ValueError('Initial bandwidth must belong to its prospective range')

    def as_dict(self):
        return asdict(self) | dict(sensor=self.sensor.as_dict(),
            scope='Prospective bounded synthetic model, sensor products and vehicle study; no flight qualification')

    @property
    def fingerprint(self):
        return fingerprint(self.as_dict())

    def controller_config(self,base):
        cfg=copy.deepcopy(base)
        cfg.orbit.initial_conditions=np.array((0.,0.,self.release_distance_m,0.,0.,0.))
        cfg.tube.sliding_gains=np.full(3,self.lambda_per_rad)
        cfg.tube.bandwidth_0=np.full(3,self.initial_bandwidth_per_rad)
        cfg.tube.bandwidth_range=dict(lower=self.bandwidth_bounds_per_rad[0],upper=self.bandwidth_bounds_per_rad[1])
        cfg.tube.bandwidth_rate_limit=self.bandwidth_rate_per_rad2
        cfg.mpc.docking_position_tolerance=self.precontact_position_tolerance_m
        cfg.mpc.docking_velocity_tolerance=self.precontact_velocity_tolerance_m_per_rad
        cfg.mpc.horizon_steps=dict(rendezvous=8,docking=6)
        cfg.mission.terminal_activation_distance=.3
        cfg.__post_init__()
        return cfg


@dataclass(frozen=True)
class IndependentTruthCase:
    case_id: str
    initial_eccentricity: float
    target_sigma_m2_kg: float
    chaser_sigma_m2_kg: float
    target_spin_deg_s: float
    seed: int

    def __post_init__(self):
        if (not self.case_id or not 0<=self.initial_eccentricity<1 or
                min(self.target_sigma_m2_kg,self.chaser_sigma_m2_kg,self.target_spin_deg_s)<=0
                or isinstance(self.seed,bool) or not isinstance(self.seed,int)):
            raise ValueError('Invalid independently selected physical truth case')


def reference_environment(config,*,sigma_target=(.008,.04),sigma_chaser=(.018,.09),relative_position_limit_m=500):
    """Earth axis is inclined in the fixed initial-reference inertial basis."""
    t=Fraction(1161600,1000000)
    axis=(0,2*t/(1+t*t),(1-t*t)/(1+t*t))
    atmosphere=ExponentialAtmosphere(rho0_kg_m3='2.786238816920409e-14',
        reference_radius_m=config.orbit.semi_major_axis,scale_height_m=80000,
        rotation_rad_s=tuple(Fraction('7.292115e-5')*v for v in axis),
        declaration='Declared radial exponential atmosphere near800km with rigid Earth corotation; fixed study density, no space-weather forecast')
    return NonlinearModelBounds(mu_m3_s2=config.orbit.mu,j2_axis=axis,atmosphere=atmosphere,
        sigma_target=sigma_target,sigma_chaser=sigma_chaser,
        radius_domain_m=(config.orbit.central_body_radius+120000,config.orbit.semi_major_axis+50000),
        relative_position_limit_m=relative_position_limit_m,relative_velocity_limit_m_per_rad=10000)


def reference_clock(config,options):
    return ReferenceClockFrame(semi_major_axis_m=config.orbit.semi_major_axis,
        eccentricity=config.orbit.eccentricity,alpha=config.orbit.target_drag,mu_m3_s2=config.orbit.mu,
        q_domain=(0,(options.maximum_updates+100)*config.anom_step))


def initial_target_state(config,eccentricity):
    a,mu=config.orbit.semi_major_axis,config.orbit.mu
    r=a*(1-eccentricity)
    return np.array((0.,r,0.,-np.sqrt(mu*(1+eccentricity)/r),0.,0.))


def controller_prior(config):
    ranges={'eccentricity':(.0003,.0017),'alpha':(.8e-9,4e-9),'beta':(1.8e-9,9e-9)}
    return ParameterBelief(ranges,{key:float(np.mean(value)) for key,value in ranges.items()})


def compose_deployment(config,design,preview,options):
    deployment=CompositeDeploymentOptions(carrier_center_lvlh_m=(0.,0.,options.release_distance_m),
        carrier_transfer_duration_rad=.18,release_clearance_m=4.,separation_buffer_m=.1,
        staging_offset_m=2.,route_buffer_m=.08,reference_tube_radius_m=.025,
        reference_speed_m_per_rad=2000.,reference_acceleration_m_per_rad2=50000.,
        minimum_segment_duration_rad=.008,staging_position_tolerance_m=.2,
        staging_velocity_tolerance_m_per_rad=15.)
    # Preserve the requested physical pitch after the prospective tube buffer.
    deployment=replace(deployment,separation_buffer_m=options.release_pitch_m-
        config.mpc.min_chaser_separation-2*deployment.reference_tube_radius_m)
    composed=CompositePatchDeployment(config,preview,body_radius_m=design.chaser.body_radius_m,options=deployment)
    safety=MissionSafety(config,preview,chaser_body_radius=design.chaser.body_radius_m)
    scenario=composed.create_scenario(initial_belief=controller_prior(config),safety=safety,
        vehicle_design=design.as_dict(),adaptation_mode='adaptive')
    return composed,scenario
