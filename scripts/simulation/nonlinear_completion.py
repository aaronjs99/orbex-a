"""Physical continuation of an authenticated sensed nonlinear approach.

The adapter uses the executed Cartesian endpoint and the commanded reference
clock. It never reconstructs a Clohessy-Hill ephemeris or fuel history. Sensor
callbacks are declared synthetic realizations; rotational replay stays decoupled.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from collections.abc import Mapping
from fractions import Fraction
import copy
import hashlib
from itertools import combinations
from itertools import zip_longest
from math import isqrt
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

from orbexa.control.reachability import IntervalBox
from orbexa.control.refuge import fingerprint as exact_fingerprint, exact_record
from orbexa.control.contact_admission import (ContactAdmissionInputs, ContactAdmissionSource,
    ContactReachPolicy, contact_reach_admission, contact_speed_admission)
from orbexa.core.attitude import body_rate_derivative
from orbexa.core.nonlinear_relative import ExponentialAtmosphere, NonlinearModelBounds, acceleration, parameters
from orbexa.core.reference_frame import FloatArithmetic, ReferenceClockFrame, exact, fingerprint
from orbexa.simulation.assembly import STANDARD_GRAVITY, positive
from orbexa.simulation.capture import CaptureTruth
from orbexa.simulation.complete_mission import ApproachPropellantBudget, CompleteMissionResult, _outward_float
from orbexa.simulation.contact import ContactBodyState
from orbexa.simulation.mission_checkpoint import digest, plain
from orbexa.simulation.nonlinear_archive import HOLD_INDEX_SCHEMA, NonlinearHoldArchive, hold_index, semantic_sha256
from orbexa.simulation.nonlinear_target_bounds import (observed_point_goal_bounds, exact_norm_upper,
    _SOURCE_SHA256 as _GOAL_BOUNDS_SOURCE_SHA256)

from orbexa.simulation.nonlinear_completion_contracts import (
    _require, _identity, _pairs, _box, _sqrt_upper, _dyadic_upper, _compact_hold,
    nonlinear_completion_source_identity)
from orbexa.simulation.nonlinear_completion_boundary import (
    _CONTACT_ADAPTER_SOURCE_SHA256, terminal_contact_speed_record,
    verify_nonlinear_arrival, capture_truth_from_nonlinear)
from orbexa.simulation.nonlinear_completion_environment import NonlinearEnvironment
from orbexa.simulation.nonlinear_completion_propellant import (
    NonlinearPropellantBudget, NonlinearTranslationAccountant)
from orbexa.simulation.nonlinear_completion_observers import (
    PastNavigationAttitudeObserver, BiasedContactObserver)


def complete_nonlinear_mission(approach, nonlinear_runner=None, *, capture_truth=None, model=None, clock=None,
                               design=None, hold_records=None, completion_runner=None, stage_sink=None,
                               drag_area_m2_by_name=None, verified_approach_boundary=None):
    """Run actual saved approach -> observed attitude -> finite contact -> disposal.

    Supply the live nonlinear runner, or explicit typed model/clock/design and
    its full admitted hold journal. Incomplete approaches remain incomplete.
    """
    if not approach.success or not approach.metadata.get('mission_complete', False):
        return CompleteMissionResult(False, 'nonlinear_approach_incomplete: '+approach.message, approach)
    _require(approach.metadata.get('model_version') == 'nonlinear-reference-relative-v1', 'a saved sensed nonlinear approach is required')
    if nonlinear_runner is not None:
        _require(all(value is None for value in (model, clock, design)), 'use a live runner or explicit boundary declarations')
        model, clock, design = nonlinear_runner.truth_model, nonlinear_runner.clock, nonlinear_runner.design
        if hold_records is None:
            hold_records = nonlinear_runner.executor.journal
    _require(model is not None and clock is not None and design is not None and hold_records is not None,
             'supply model, clock, design and the complete full nonlinear hold journal')
    if verified_approach_boundary is None:
        arrival_verification = verify_nonlinear_arrival(approach, clock=clock)
    else:
        from orbexa.simulation.nonlinear_completion_bridge import VerifiedClosedApproachBoundary
        _require(type(verified_approach_boundary) is VerifiedClosedApproachBoundary,
                 'an independently verified typed old-source boundary is required')
        arrival_verification = verified_approach_boundary.verify(approach, model=model, clock=clock,
            design=design, hold_records=hold_records)
    _require(digest(approach.metadata['scenario']['vehicle_design']) == digest(design.as_dict()), 'declared vehicle differs from the saved approach')
    saved_truth = capture_truth_from_nonlinear(approach, model=model, clock=clock)
    if capture_truth is None:
        capture_truth = saved_truth
    else:
        _require(isinstance(capture_truth, CaptureTruth) and digest(plain(capture_truth)) == digest(plain(saved_truth)),
                 'explicit physical boundary differs from retained actual nonlinear truth')
    accountant = NonlinearTranslationAccountant(approach, model=model, clock=clock, hold_records=hold_records,
        expected_actuator_id=None if nonlinear_runner is None else nonlinear_runner.executor.actuator.fingerprint,
        expected_sensor_id=None if nonlinear_runner is None else nonlinear_runner.options.sensor.fingerprint)
    certificate = approach.metadata['continuous_execution_certificate']
    bound = max(abs(float(value)) for value in (*certificate['input_lower'], *certificate['input_upper']))
    profile = (getattr(nonlinear_runner.options, 'completion', None) if nonlinear_runner is not None else
               approach.metadata.get('study_options', {}).get('completion'))
    complete = completion_runner or design.completion_runner(input_component_bound=float(np.nextafter(bound, np.inf)),
        stage_sink=stage_sink, translation_accountant=accountant, completion_profile=profile)
    _require(complete.options.target_mass_kg == design.target_mass_kg and
             complete.options.translation_force_n == design.translation_force_n and
             complete.options.specific_impulse_s == design.specific_impulse_s and
             complete.attitude_tracker.actuators.nozzle_force_limit_n == design.attitude_nozzle_force_n and
             complete.attitude_tracker.actuators.moment_arm_m == design.attitude_moment_arm_m and
             complete.attitude_tracker.actuators.specific_impulse_s == design.specific_impulse_s,
             'completion hardware differs from the recorded prospective vehicle')
    _require(all(complete.attitude_tracker.chasers.get(key, complete.attitude_tracker.default_chaser) == design.chaser
                 for key in approach.actual_trajectories), 'completion bus mass/envelope differs from the saved vehicle')
    complete.translation_accountant = accountant
    if stage_sink is not None:
        complete.stage_sink = stage_sink
    complete.capture_options = replace(complete.capture_options, arm_reach_m=design.arm_reach_m,
        max_relative_speed_m_s=.01, max_orientation_error_rad=.01, max_spin_error_rad_s=.001,
        idealized_instantaneous=False)
    complete.attitude_tracker.options = replace(complete.attitude_tracker.options,
        max_final_orientation_error_rad=.01, max_final_spin_error_rad_s=.001)
    _require(float(approach.metadata['docking_position_tolerance']) <= .04+1e-12,
             'saved whole-position endpoint enclosure exceeds the declared 40 mm contact profile')
    chasers = {key: complete.attitude_tracker.chasers.get(key, complete.attitude_tracker.default_chaser)
               for key in approach.actual_trajectories}
    environment = NonlinearEnvironment(model, target_mass_kg=design.target_mass_kg, chasers=chasers,
        drag_area_m2_by_name=drag_area_m2_by_name)
    attitude_observer = PastNavigationAttitudeObserver(approach, model=model, clock=clock)
    contact_observer = BiasedContactObserver()
    from orbexa.simulation.contact_boundary import TerminalContactBoundary
    contact_boundary = TerminalContactBoundary.from_approach(approach, clock,
        orientation_gate_rad=complete.capture_options.max_orientation_error_rad,
        spin_gate_rad_s=complete.capture_options.max_spin_error_rad_s)
    result = complete.complete(approach, capture_truth=capture_truth, contact_feedback=contact_observer,
        contact_environment=environment, attitude_observations=attitude_observer,
        contact_initial_enclosure=contact_boundary)
    audit = dict(method='nonlinear-actual-boundary-physical-completion-v1',
        source_approach_sha256=semantic_sha256(vars(approach)), environment=environment.as_dict(),
        arrival_verification=arrival_verification,
        environment_rhs_calls=environment.calls, attitude_observations=attitude_observer.as_dict(),
        contact_observations=contact_observer.as_dict(),
        capture_truth=plain(capture_truth), admission=dict(position_enclosure_m=.04, relative_speed_m_s=.01,
            orientation_rad=.01, spin_error_rad_s=.001, arm_reach_m=design.arm_reach_m),
        scope='Simulated nonlinear continuation with explicit sensors and retained physical gates; separate attitude replay and representative rigid compliant contact, no hardware or flight qualification')
    return replace(result, completion_audit=audit)
