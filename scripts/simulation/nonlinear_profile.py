"""Explicit prospective cases for sensed, nonlinear multi-spacecraft missions.

These are simulated hardware/environment declarations, not Envisat telemetry
or a sensor specification. Case construction has no access to future truth.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from fractions import Fraction
import copy
import math
import numpy as np

from orbexa.core.reference_frame import ReferenceClockFrame, fingerprint
from orbexa.core.nonlinear_relative import NonlinearModelBounds, ExponentialAtmosphere
from orbexa.estimation.belief import ParameterBelief
from orbexa.estimation.navigation import StudySensorProfile
from orbexa.simulation.spacecraft_design import SpacecraftMissionDesign
from orbexa.simulation.composite_deployment import CompositeDeploymentOptions, CompositePatchDeployment
from orbexa.simulation.mission_safety import MissionSafety
from orbexa.simulation.completion_profile import MissionCompletionProfile
from orbexa.simulation.contingency_reference import AccelerationContingencyOptions


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
    membership_policy: str = 'independent-holds-v1'
    membership_window_holds: int = 8
    membership_max_nuisance_cells: int = 2
    membership_max_flow_evaluations_per_candidate: int = 256
    rendezvous_horizon_steps: int = 8
    docking_horizon_steps: int = 6
    near_feedback_subdivisions: int = 1
    near_target_radius_m: float = 30.
    terminal_activation_distance_m: float = .3
    initial_target_orientation_error: float = .02
    target_rate_bound_rad_s: float = .08
    target_acceleration_bound_rad_s2: float = .005
    controller_parameter_prior: tuple = ()
    target_sigma_prior_m2_kg: tuple = (.008,.04)
    chaser_sigma_prior_m2_kg: tuple = (.018,.09)
    nominal_body_spin_rad_s: tuple | None = None
    deployment: CompositeDeploymentOptions | None = None
    carrier_transfer_duration_rad: float | None = None
    near_refuge_extra_holds: tuple = ()
    completion: MissionCompletionProfile | None = None
    far_contingency: AccelerationContingencyOptions | None = None

    def __post_init__(self):
        for key,value in vars(self).items():
            if key in ('sensor','bandwidth_bounds_per_rad','maximum_updates','planning_workers',
                       'initial_orbit_conditioning','membership_policy','membership_window_holds',
                       'membership_max_nuisance_cells','membership_max_flow_evaluations_per_candidate',
                       'rendezvous_horizon_steps','docking_horizon_steps','controller_parameter_prior',
                       'near_feedback_subdivisions','near_target_radius_m',
                       'target_sigma_prior_m2_kg','chaser_sigma_prior_m2_kg','nominal_body_spin_rad_s','deployment',
                       'completion','far_contingency','carrier_transfer_duration_rad','near_refuge_extra_holds'):
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
        if self.membership_policy not in ('independent-holds-v1','linked-affine-window-v1'):
            raise ValueError('Select an explicit nonlinear membership policy')
        for name in ('membership_window_holds','membership_max_nuisance_cells',
                     'membership_max_flow_evaluations_per_candidate'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int) or value<1:
                raise ValueError(name+' must be a positive integer')
        for name in ('rendezvous_horizon_steps','docking_horizon_steps'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int) or value<2:
                raise ValueError(name+' must contain at least two nodes')
        if (isinstance(self.near_feedback_subdivisions,bool)
                or not isinstance(self.near_feedback_subdivisions,int)
                or self.near_feedback_subdivisions<1):
            raise ValueError('near_feedback_subdivisions must be a positive integer')
        value=self.near_target_radius_m
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            raise ValueError('near_target_radius_m must be positive and finite')
        object.__setattr__(self,'near_target_radius_m',float(value))
        for name in ('target_sigma_prior_m2_kg','chaser_sigma_prior_m2_kg'):
            pair=tuple(map(float,getattr(self,name)))
            if len(pair)!=2 or any(not np.isfinite(v) or v<0 for v in pair) or pair[0]>pair[1]:
                raise ValueError(name+' must be a finite ordered nonnegative interval')
            object.__setattr__(self,name,pair)
        prior=self.controller_parameter_prior
        pairs=tuple(prior.items()) if hasattr(prior,'items') else tuple(prior)
        if pairs:
            if len(pairs)!=3 or {key for key,_ in pairs}!={'eccentricity','alpha','beta'}:
                raise ValueError('Controller prior needs eccentricity, alpha and beta intervals')
            normalized=[]
            for key,pair in pairs:
                low,high=map(float,pair)
                if not np.isfinite(low+high) or not 0<=low<=high or key=='eccentricity' and high>=1:
                    raise ValueError('Invalid controller parameter interval: '+key)
                normalized.append((key,(low,high)))
            pairs=tuple(sorted(normalized))
        object.__setattr__(self,'controller_parameter_prior',pairs)
        if self.nominal_body_spin_rad_s is not None:
            vector=tuple(map(float,self.nominal_body_spin_rad_s))
            if len(vector)!=3 or not all(np.isfinite(value) for value in vector):
                raise ValueError('Nominal body spin must have three finite rad/s components')
            object.__setattr__(self,'nominal_body_spin_rad_s',vector)
        duration=self.carrier_transfer_duration_rad
        if duration is not None:
            if isinstance(duration,bool) or not isinstance(duration,(int,float)) or not math.isfinite(duration) or duration<=0:
                raise ValueError('carrier_transfer_duration_rad must be positive and finite')
            object.__setattr__(self,'carrier_transfer_duration_rad',float(duration))
        extra=tuple(self.near_refuge_extra_holds)
        if any(type(value) is not int or value<=1024 for value in extra) or tuple(sorted(set(extra)))!=extra:
            raise ValueError('Extra near-refuge holds must be distinct increasing integers above 1024')
        object.__setattr__(self,'near_refuge_extra_holds',extra)
        if self.deployment is not None:
            declared=self.deployment
            if hasattr(declared,'items'):
                declared=CompositeDeploymentOptions(**dict(declared))
            if not isinstance(declared,CompositeDeploymentOptions):
                raise TypeError('Deployment must declare CompositeDeploymentOptions or its named fields')
            if declared.carrier_center_lvlh_m is None:
                raise ValueError('A nonlinear deployment requires an explicit carrier LVLH release centre')
            if duration is not None and float(duration)!=declared.carrier_transfer_duration_rad:
                raise ValueError('Carrier transfer duration conflicts with the explicit deployment')
            object.__setattr__(self,'deployment',declared)
        if self.completion is not None:
            object.__setattr__(self,'completion',MissionCompletionProfile.from_declaration(self.completion))
        if self.far_contingency is not None:
            value=self.far_contingency
            if hasattr(value,'items'):
                value=AccelerationContingencyOptions(**dict(value))
            if not isinstance(value,AccelerationContingencyOptions):
                raise TypeError('far_contingency requires explicit AccelerationContingencyOptions')
            object.__setattr__(self,'far_contingency',value)
        lo,hi = self.bandwidth_bounds_per_rad
        if not 0 < lo <= self.initial_bandwidth_per_rad <= hi:
            raise ValueError('Initial bandwidth must belong to its prospective range')

    def as_dict(self):
        result=asdict(self) | dict(sensor=self.sensor.as_dict(),
            scope='Prospective bounded synthetic model, sensor products and vehicle study; no flight qualification')
        if self.completion is None:
            result.pop('completion')
        else:
            result['completion']=self.completion.as_dict()
        if self.far_contingency is None:
            result.pop('far_contingency')
        if self.carrier_transfer_duration_rad is None:
            result.pop('carrier_transfer_duration_rad')
        if not self.near_refuge_extra_holds:
            result.pop('near_refuge_extra_holds')
        if self.near_target_radius_m==30.:
            result.pop('near_target_radius_m')
        # Old saved cohorts keep their exact declarations and identities.
        if self.membership_policy=='independent-holds-v1':
            for name in ('membership_policy','membership_window_holds','membership_max_nuisance_cells',
                         'membership_max_flow_evaluations_per_candidate'):
                result.pop(name)
            defaults=dict(rendezvous_horizon_steps=8,docking_horizon_steps=6,terminal_activation_distance_m=.3,
                near_feedback_subdivisions=1,
                initial_target_orientation_error=.02,target_rate_bound_rad_s=.08,target_acceleration_bound_rad_s2=.005,
                controller_parameter_prior=(),target_sigma_prior_m2_kg=(.008,.04),chaser_sigma_prior_m2_kg=(.018,.09),
                nominal_body_spin_rad_s=None,deployment=None)
            for name,value in defaults.items():
                if getattr(self,name)==value:result.pop(name)
        return result

    @property
    def fingerprint(self):
        return fingerprint(self.as_dict())

    @property
    def nominal_spin_vector(self):
        return ((0.,0.,math.radians(self.nominal_spin_deg_s))
            if self.nominal_body_spin_rad_s is None else self.nominal_body_spin_rad_s)

    def controller_config(self,base):
        cfg=copy.deepcopy(base)
        cfg.orbit.initial_conditions=np.array((0.,0.,self.release_distance_m,0.,0.,0.))
        cfg.tube.sliding_gains=np.full(3,self.lambda_per_rad)
        cfg.tube.bandwidth_0=np.full(3,self.initial_bandwidth_per_rad)
        cfg.tube.bandwidth_range=dict(lower=self.bandwidth_bounds_per_rad[0],upper=self.bandwidth_bounds_per_rad[1])
        cfg.tube.bandwidth_rate_limit=self.bandwidth_rate_per_rad2
        cfg.mpc.docking_position_tolerance=self.precontact_position_tolerance_m
        cfg.mpc.docking_velocity_tolerance=self.precontact_velocity_tolerance_m_per_rad
        cfg.mpc.horizon_steps=dict(rendezvous=self.rendezvous_horizon_steps,docking=self.docking_horizon_steps)
        cfg.mission.terminal_activation_distance=self.terminal_activation_distance_m
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
    body_spin_rad_s: tuple | None = None

    def __post_init__(self):
        if (not self.case_id or not 0<=self.initial_eccentricity<1 or
                min(self.target_sigma_m2_kg,self.chaser_sigma_m2_kg,self.target_spin_deg_s)<=0
                or isinstance(self.seed,bool) or not isinstance(self.seed,int)):
            raise ValueError('Invalid independently selected physical truth case')
        if self.body_spin_rad_s is not None:
            vector=tuple(map(float,self.body_spin_rad_s))
            if len(vector)!=3 or not all(np.isfinite(value) for value in vector):
                raise ValueError('Physical body spin must have three finite rad/s components')
            object.__setattr__(self,'body_spin_rad_s',vector)

    @property
    def spin_vector(self):
        return ((0.,0.,math.radians(self.target_spin_deg_s))
            if self.body_spin_rad_s is None else self.body_spin_rad_s)


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


def controller_prior(config,options=None):
    ranges=(dict(options.controller_parameter_prior) if options is not None and options.controller_parameter_prior else
        {'eccentricity':(.0003,.0017),'alpha':(.8e-9,4e-9),'beta':(1.8e-9,9e-9)})
    return ParameterBelief(ranges,{key:float(np.mean(value)) for key,value in ranges.items()})


def compose_deployment(config,design,preview,options):
    deployment=options.deployment
    if deployment is None:
        deployment=CompositeDeploymentOptions(carrier_center_lvlh_m=(0.,0.,options.release_distance_m),
        carrier_transfer_duration_rad=(.18 if options.carrier_transfer_duration_rad is None
            else options.carrier_transfer_duration_rad),release_clearance_m=4.,separation_buffer_m=.1,
        staging_offset_m=2.,route_buffer_m=.08,reference_tube_radius_m=.025,
        reference_speed_m_per_rad=2000.,reference_acceleration_m_per_rad2=50000.,
        minimum_segment_duration_rad=.008,staging_position_tolerance_m=.2,
        staging_velocity_tolerance_m_per_rad=15.)
        # Preserve the requested physical pitch after the prospective tube buffer.
        deployment=replace(deployment,separation_buffer_m=options.release_pitch_m-
            config.mpc.min_chaser_separation-2*deployment.reference_tube_radius_m)
    composed=CompositePatchDeployment(config,preview,body_radius_m=design.chaser.body_radius_m,options=deployment)
    safety=MissionSafety(config,preview,chaser_body_radius=design.chaser.body_radius_m)
    scenario=composed.create_scenario(initial_belief=controller_prior(config,options),safety=safety,
        vehicle_design=design.as_dict(),adaptation_mode='adaptive')
    return composed,scenario
