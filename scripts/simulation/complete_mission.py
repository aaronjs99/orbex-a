"""Compose a recorded physical approach with attitude, latch and disposal stages.

The approach remains the nonlinear ADTMPC simulation. Rotational replay is a
separate bounded-actuator model. Translation propellant is conservatively
debited using a continuously enclosed orbital clock, rather than silently
starting capture with full tanks. Stage failures retain the preceding record.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from functools import partial
from pathlib import Path
from typing import Callable, Mapping, Optional, TYPE_CHECKING

import numpy as np

from orbexa.simulation.assembly import STANDARD_GRAVITY, positive
from orbexa.simulation.capture import CaptureModel, CaptureOptions, ChaserPhysicalProperties, TargetPhysicalProperties
from orbexa.simulation.chaser_attitude import ChaserAttitudeResult, ChaserAttitudeTracker
from orbexa.simulation.lifecycle import MissionLifecycle, MissionLifecycleResult, MissionLifecycleStore
from orbexa.simulation.mission_result import ADTMPCMissionResult, MissionResultStore
from orbexa.simulation.propulsion import ThrusterSystem
from orbexa.utils.io_utils import save_data

if TYPE_CHECKING:
    from orbexa.simulation.terminal_handoff import PhysicalTerminalHandoff


COMPLETE_MISSION_SCHEMA = "orbexa.complete-mission.v1"


def _outward_float(value: Fraction) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("Physical accounting bound cannot be represented by a finite float")
    return float(np.nextafter(number, np.inf)) if Fraction(number) < value else number


def bind_approach_body_geometry(approach, chasers, default_chaser, *, capture_reference=False,
                                terminal_handoff=None) -> dict:
    """Bind physical bus envelopes to the geometry used during free flight.

    A smaller known spherical body inherits exclusion of a larger enclosing
    sphere. A capture-endpoint check cannot retrospectively qualify a larger
    bus throughout an approach that used a smaller body or a point mass.
    """
    scenario = approach.metadata.get("scenario", {})
    if not isinstance(scenario, Mapping) or scenario.get("tracked_reference_point") != "chaser_center_of_mass":
        raise ValueError("Physical completion requires a recorded chaser center-of-mass approach reference")
    value = scenario.get("chaser_body_bounding_radius_m")
    if value is None:
        raise ValueError("Approach did not declare its free-flight chaser body envelope")
    envelopes = {"sampled_safety_body_radius_m": positive(value, "Recorded approach body radius", allow_zero=True)}
    continuous = approach.metadata.get("continuous_execution_certificate")
    if continuous is not None:
        if not isinstance(continuous, Mapping) or "body_radius_m" not in continuous:
            raise ValueError("Continuous approach safety record lacks its physical body envelope")
        envelopes["continuous_safety_body_radius_m"] = positive(
            continuous["body_radius_m"], "Continuous approach body radius", allow_zero=True)
    radius = min(envelopes.values())
    ids = tuple(approach.actual_trajectories)
    if not ids or set(chasers) - set(ids):
        raise ValueError("Physical bus properties must name the recorded fleet")
    actual, required = {}, {}
    terminal_radii = None
    if terminal_handoff is not None:
        terminal_handoff.verify_approach(approach)
        bounds = terminal_handoff.as_dict()["controller_bounds"]
        if set(bounds) != set(ids):
            raise ValueError("Terminal body envelopes must name the exact approach fleet")
        terminal_radii = {key: Fraction(bounds[key]["body_radius"]) for key in ids}
        if any(value < 0 for value in terminal_radii.values()):
            raise ValueError("Terminal body radii must be nonnegative")
    for key in ids:
        properties = chasers.get(key, default_chaser)
        if not isinstance(properties, ChaserPhysicalProperties):
            raise ValueError(f"{key}: physical bus properties must be ChaserPhysicalProperties")
        # An explicitly reinterpreted tip reference must enclose the COM
        # offset as well as the bus, for every possible attitude along flight.
        required[key] = properties.body_radius_m + (properties.gripper_offset_m if capture_reference else 0.)
        if required[key] > radius:
            raise ValueError(f"{key}: physical bus radius exceeds the body envelope used during approach")
        exact_required = Fraction(properties.body_radius_m) + (Fraction(properties.gripper_offset_m)
                                                              if capture_reference else 0)
        if terminal_radii is not None and exact_required > terminal_radii[key]:
            raise ValueError(f"{key}: physical bus radius exceeds the body envelope proved during terminal tracking")
        actual[key] = float(properties.body_radius_m)
    return dict(**envelopes, binding_body_radius_m=radius,
                terminal_body_radii_m=None if terminal_radii is None else {key: str(value) for key, value in terminal_radii.items()},
                physical_body_radii_m=actual,
                required_enclosing_radii_m=required,
                reference_interpretation="explicit_capture_reference" if capture_reference else "chaser_center_of_mass",
                scope="Known spherical bus bodies are enclosed by every recorded approach body envelope; capture gates still apply")


@dataclass(frozen=True)
class CompleteMissionOptions:
    """Representative vehicle limits, with an explicit approach input bound.

    ``input_component_bound`` bounds every actual held translational command
    in m/rad², including ancillary feedback. It is a caller declaration; a
    sparse history of nominal inputs cannot establish that bound.
    """

    input_component_bound: float
    translation_force_n: float = 10.
    specific_impulse_s: float = 300.
    target_mass_kg: float = 1000.

    def __post_init__(self):
        for name in ("input_component_bound", "translation_force_n", "specific_impulse_s", "target_mass_kg"):
            object.__setattr__(self, name, positive(getattr(self, name), name))


@dataclass(frozen=True)
class ApproachPropellantBudget:
    """Global upper allowances, not a reconstructed fuel history."""

    clock: dict
    force_l1_upper_n: dict[str, float]
    translation_fuel_debit_kg: dict[str, float]
    input_component_bound: float
    specific_impulse_s: float
    translation_force_limit_n: float

    def to_dict(self):
        return dict(clock=self.clock, force_l1_upper_n=self.force_l1_upper_n,
                    translation_fuel_debit_kg=self.translation_fuel_debit_kg,
                    input_component_bound_m_per_rad2=self.input_component_bound,
                    specific_impulse_s=self.specific_impulse_s,
                    translation_nozzle_limit_n=self.translation_force_limit_n,
                    method="For any proper bus rotation: ||F_body||_1 <= 3*m_wet*(dq/dt)_max^2*U; fuel <= force_bound*duration_upper/(g0*Isp)",
                    scope="Conservative approach translation debit under declared held-input, continuously throttleable nozzle and exact-actuation assumptions; not measured consumption")


@dataclass(frozen=True)
class CompleteMissionResult:
    success: bool
    message: str
    approach: ADTMPCMissionResult
    translation_budget: Optional[ApproachPropellantBudget] = None
    attitude: Optional[ChaserAttitudeResult] = None
    lifecycle: Optional[MissionLifecycleResult] = None
    assumptions: tuple[str, ...] = field(default_factory=lambda: (
        "Approach translation, free-bus attitude replay and post-latch assembly dynamics use their separately declared models.",
        "Free-bus translation and torque couples are decoupled; exhaust is unobstructed and the gripper is stowed until capture.",
        "Target properties and kinematics are known; physical release, uncertain navigation, contact elasticity and atmospheric demise are not qualified.",
    ))
    geometry_binding: Optional[dict] = None
    terminal_handoff: Optional[PhysicalTerminalHandoff] = None

    def to_dict(self):
        failed_stage = None if self.lifecycle is None else self.lifecycle.failure.get("stage")
        return dict(success=self.success, message=self.message,
                    stages=dict(approach=bool(self.approach.success and self.approach.metadata.get("mission_complete", False)),
                                terminal_tracking=None if self.terminal_handoff is None else True,
                                chaser_attitude=None if self.attitude is None else self.attitude.success,
                                attachment=None if self.lifecycle is None else self.lifecycle.capture.success,
                                detumbling=False if failed_stage == "detumbling" else None if self.lifecycle is None or self.lifecycle.detumbling is None else self.lifecycle.detumbling.success,
                                deorbit=False if failed_stage == "deorbit" else None if self.lifecycle is None or self.lifecycle.deorbit is None else self.lifecycle.deorbit.success),
                    translation_budget=None if self.translation_budget is None else self.translation_budget.to_dict(),
                    geometry_binding=self.geometry_binding,
                    terminal_handoff=None if self.terminal_handoff is None else self.terminal_handoff.as_dict(),
                    assumptions=list(self.assumptions),
                    records=dict(approach="approach/mission_data.json",
                                 chaser_attitude=None if self.attitude is None else "chaser_attitude.json",
                                 terminal_handoff=None if self.terminal_handoff is None else "terminal_handoff.json",
                                 lifecycle=None if self.lifecycle is None else "lifecycle_data.json"))


class CompleteMissionRunner:
    """One orchestration boundary with independently injectable stage models.

    ``complete`` also accepts a saved result, so changing a post-capture policy
    does not require repeating an expensive approach. It never replaces an
    unsuccessful approach or attitude gate with an idealized successful state.
    """

    def __init__(self, *, options: CompleteMissionOptions, approach_runner=None,
                 attitude_tracker: Optional[ChaserAttitudeTracker] = None,
                 capture_options: Optional[CaptureOptions] = None,
                 detumbling=None, deorbit=None, propulsion_factory: Optional[Callable] = None,
                 stage_sink: Optional[Callable[[str, object], None]] = None):
        self.options = options
        self.approach_runner = approach_runner
        self.attitude_tracker = attitude_tracker or ChaserAttitudeTracker()
        self.capture_options = capture_options or CaptureOptions()
        if self.capture_options.assume_attitude_matched:
            raise ValueError("CompleteMissionRunner uses actual attitude replay; perfect attitude matching cannot be enabled")
        if self.capture_options.assume_approach_point_is_capture_reference:
            raise ValueError("CompleteMissionRunner requires literal center-of-mass approach trajectories")
        self.detumbling, self.deorbit = detumbling, deorbit
        self.propulsion_factory = propulsion_factory or partial(ThrusterSystem.for_assembly,
            translation_force_n=options.translation_force_n,
            attitude_force_n=self.attitude_tracker.actuators.nozzle_force_limit_n,
            moment_arm_m=self.attitude_tracker.actuators.moment_arm_m,
            specific_impulse_s=options.specific_impulse_s)
        self.stage_sink = stage_sink

    def _record(self, stage, result):
        if self.stage_sink is not None:
            self.stage_sink(stage, result)

    def translation_budget(self, approach: ADTMPCMissionResult) -> ApproachPropellantBudget:
        from orbexa.control.physical_terminal import bound_orbital_clock
        from orbexa.control.reachability import OrbitalModelBounds
        anomalies = np.asarray(approach.anom_history, dtype=float)
        if anomalies.ndim != 1 or len(anomalies) < 2 or np.any(~np.isfinite(anomalies)) or np.any(np.diff(anomalies) <= 0):
            raise ValueError("Physical approach accounting requires an increasing anomaly history")
        for key in approach.actual_trajectories:
            recorded = approach.controls.get(key)
            if recorded is not None and len(recorded):
                means = np.asarray(recorded, dtype=float)
                if means.shape != (len(anomalies) - 1, 3) or np.any(~np.isfinite(means)):
                    raise ValueError(f"{key}: recorded applied-input means must align with the approach intervals")
                # A mean cannot establish the largest held command, but it
                # can falsify an already contradictory declared upper bound.
                if np.any(np.abs(means) > self.options.input_component_bound):
                    raise ValueError(f"{key}: declared held-input bound is contradicted by a recorded applied-input mean")
        truth = approach.truth
        start, end = Fraction(float(anomalies[0])), Fraction(float(anomalies[-1]))
        continuous = approach.metadata.get("continuous_execution_certificate")
        if continuous is not None:
            if not isinstance(continuous, Mapping) or not isinstance(continuous.get("exact_end_anomaly"), str):
                raise ValueError("Guarded physical accounting requires its exact committed end clock")
            try:
                start = Fraction(float(continuous["start_anomaly"]))
                end = Fraction(continuous["exact_end_anomaly"])
            except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError) as error:
                raise ValueError("Guarded physical accounting has an invalid committed clock") from error
            if (abs(float(start) - anomalies[0]) > 1e-10 or abs(float(end) - anomalies[-1]) > 1e-10
                    or end <= start):
                raise ValueError("Guarded accounting clock disagrees with the committed approach")
            declared_limits = (continuous.get("input_lower"), continuous.get("input_upper"))
            if any(value is not None for value in declared_limits):
                if any(not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 3
                       for value in declared_limits):
                    raise ValueError("Guarded accounting requires complete three-component actuator limits")
                try:
                    lower, upper = (tuple(Fraction(value) for value in row) for row in declared_limits)
                except (TypeError, ValueError, ZeroDivisionError) as error:
                    raise ValueError("Guarded accounting actuator limits must be finite scalar values") from error
                limit = Fraction(self.options.input_component_bound)
                if any(lo > hi or lo < -limit or hi > limit for lo, hi in zip(lower, upper)):
                    raise ValueError("Declared fuel-accounting input bound does not cover the proved actuator intervals")
        clock = bound_orbital_clock(start, end, OrbitalModelBounds(
            eccentricity=(truth['eccentricity'],) * 2, alpha=(truth['alpha'],) * 2,
            beta=(truth['beta'],) * 2, semi_major_axis=truth['semi_major_axis'], mu=truth['mu']))
        rate = clock.anomaly_rate.upper[0]
        force, debit = {}, {}
        for key in approach.actual_trajectories:
            prop = self.attitude_tracker.chasers.get(key, self.attitude_tracker.default_chaser)
            bound = 3 * Fraction(prop.mass_kg) * rate**2 * Fraction(self.options.input_component_bound)
            if bound > Fraction(self.options.translation_force_n):
                raise ValueError(f"{key}: approach force allowance exceeds the installed translation nozzle limit")
            fuel = bound * clock.physical_duration_upper / (Fraction(STANDARD_GRAVITY) * Fraction(self.options.specific_impulse_s))
            force[key], debit[key] = _outward_float(bound), _outward_float(fuel)
        return ApproachPropellantBudget(clock.as_dict(), force, debit, float(self.options.input_component_bound),
                                        float(self.options.specific_impulse_s), float(self.options.translation_force_n))

    def run(self, *, mission: str = "multi", steps: int = 550) -> CompleteMissionResult:
        if self.approach_runner is None:
            raise ValueError("run requires an injected approach runner; use complete for a saved approach")
        return self.complete(self.approach_runner.run(mission=mission, steps=steps))

    def complete(self, approach: ADTMPCMissionResult, *,
                 terminal_handoff: Optional[PhysicalTerminalHandoff] = None) -> CompleteMissionResult:
        self._record("approach", approach)
        if not approach.success or not approach.metadata.get("mission_complete", False):
            return CompleteMissionResult(False, f"approach_incomplete: {approach.message}", approach)
        budget = attitude = geometry = verified_handoff = None
        try:
            execution = approach
            if terminal_handoff is not None:
                from orbexa.simulation.terminal_handoff import PhysicalTerminalHandoff
                if not isinstance(terminal_handoff, PhysicalTerminalHandoff):
                    raise ValueError("Physical continuation requires a verified PhysicalTerminalHandoff")
                terminal_handoff.verify_approach(approach)
                verified_handoff = terminal_handoff
                execution = terminal_handoff.trajectory
                self._record("terminal_handoff", terminal_handoff)
            geometry = bind_approach_body_geometry(approach, self.attitude_tracker.chasers,
                                                  self.attitude_tracker.default_chaser, terminal_handoff=verified_handoff)
            budget = self.translation_budget(execution)
            self._record("translation_budget", budget)
            attitude = self.attitude_tracker.track(execution, translation_fuel_debit_kg=budget.translation_fuel_debit_kg)
            self._record("chaser_attitude", attitude)
            if not attitude.success:
                return CompleteMissionResult(False, f"chaser_attitude_rejected: {attitude.message}", approach, budget, attitude,
                                             geometry_binding=geometry, terminal_handoff=verified_handoff)
            capture_geometry = bind_approach_body_geometry(approach, attitude.capture_chasers,
                                                           self.attitude_tracker.default_chaser, terminal_handoff=verified_handoff)
            geometry["capture_body_radii_m"] = capture_geometry["physical_body_radii_m"]
            inertia = approach.target.get("inertia_kg_m2")
            if inertia is None:
                raise ValueError("Physical target inertia is required for attachment")
            lifecycle = MissionLifecycle(capture=CaptureModel(
                target=TargetPhysicalProperties(self.options.target_mass_kg, inertia),
                chasers=attitude.capture_chasers, default_chaser=self.attitude_tracker.default_chaser,
                options=self.capture_options), detumbling=self.detumbling, deorbit=self.deorbit,
                propulsion_factory=self.propulsion_factory).run(approach, chaser_attitudes=attitude.final_attitudes,
                                                               terminal_handoff=verified_handoff)
            self._record("lifecycle", lifecycle)
            return CompleteMissionResult(lifecycle.success, lifecycle.message, approach, budget, attitude, lifecycle,
                                         geometry_binding=geometry, terminal_handoff=verified_handoff)
        except (ValueError, RuntimeError, ArithmeticError, ImportError) as error:
            return CompleteMissionResult(False, f"physical_completion_rejected: {type(error).__name__}: {error}", approach, budget, attitude,
                                         geometry_binding=geometry, terminal_handoff=verified_handoff)


class CompleteMissionStore:
    """Atomically record completed stages into a newly reserved directory."""

    filename = "complete_mission.json"

    def __init__(self, output_dir: Path):
        self.directory = Path(output_dir)
        self.directory.mkdir(parents=True, exist_ok=False)

    def record(self, stage: str, result):
        if stage == "approach":
            MissionResultStore.save(result, self.directory / "approach")
        elif stage == "lifecycle":
            MissionLifecycleStore.save(result, self.directory)
        elif stage in ("translation_budget", "chaser_attitude"):
            if not save_data(self.directory / f"{stage}.json", result.to_dict()):
                raise OSError(f"Could not save {stage} record")
        elif stage == "terminal_handoff":
            if not save_data(self.directory / "terminal_handoff.json", result.as_dict()):
                raise OSError("Could not save physical terminal handoff")
        else:
            raise ValueError(f"Unknown mission stage: {stage}")

    def finish(self, result: CompleteMissionResult) -> Path:
        path = self.directory / self.filename
        if not save_data(path, dict(schema=COMPLETE_MISSION_SCHEMA, result=result.to_dict())):
            raise OSError("Could not save complete mission summary")
        return path

    @staticmethod
    def save(result: CompleteMissionResult, output_dir: Path) -> Path:
        store = CompleteMissionStore(output_dir)
        store.record("approach", result.approach)
        if result.terminal_handoff is not None:
            store.record("terminal_handoff", result.terminal_handoff)
        if result.translation_budget is not None:
            store.record("translation_budget", result.translation_budget)
        if result.attitude is not None:
            store.record("chaser_attitude", result.attitude)
        if result.lifecycle is not None:
            store.record("lifecycle", result.lifecycle)
        return store.finish(result)
