"""Source-qualified numerical continuation of a committed mission record.

This is a numerical restart, not the admissible safe-backup theorem. Actual
states, persistent nominal centers, tube widths/gains, belief, clock, phase and
assignments are retained. Unavailable full control warm starts and dense SMID
measurements are explicitly reset; neither is reconstructed from mean inputs.
No callables or pickle payloads are serialized.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, is_dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import numpy as np

from orbexa.control.sample_hold import TUBE_MODEL_VERSION
from orbexa.contracts import PHYSICAL_MODEL_ID, COUPLED_CONTROLLER_ID
from orbexa.estimation.belief import ParameterBelief
from orbexa.simulation.coupled_state import CoupledMissionState
from orbexa.simulation.mission_models import ChaserConfig, MissionPhase
from orbexa.simulation.mission_result import ADTMPCMissionResult


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
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def source_fingerprints():
    package = Path(__file__).resolve().parents[1]
    return {path.relative_to(package).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(package.rglob("*.py"))}


def scenario_checkpoint_context(scenario):
    return dict(configuration=plain(scenario.config), truth=scenario.truth.as_dict(),
        target=scenario.target.as_dict(), policy=scenario.scenario_metadata(),
        body_radius=scenario.safety.chaser_body_radius,
        adaptation_mode=scenario.adaptation_mode, source_fingerprints=source_fingerprints())


def _configuration_signature(configuration):
    result = copy.deepcopy(plain(configuration))
    # Solver settings and corrected solver source may change in a numerical
    # restart; every model/controller/mission setting remains authoritative.
    result.pop("solver", None)
    return result


def _policy_signature(metadata):
    result = {}
    deployment = metadata.get("deployment")
    if deployment:
        names = ("kind", "options", "count", "initial_anomaly", "launch_radius", "launch_azimuth",
                 "initial_pitch", "initial_spacing", "final_spacing", "arc_span", "assignment_strategy")
        if deployment.get("kind") in ("compact_component_aware_patch_routes", "carrier_cluster_to_component_aware_patch_routes"):
            composite = ("geometry_id", "geometry_fingerprint", "release_center_body_m",
                         "common_direction_position", "initial_pitch_m", "staging_anomaly",
                         "route_order", "routes")
            if any(name not in deployment for name in composite):
                raise ValueError("Composite numerical restart requires the complete route and geometry policy")
            names += composite
            if deployment.get('kind') == 'carrier_cluster_to_component_aware_patch_routes':
                carrier = ('carrier_center_lvlh_m', 'transfer_end_anomaly', 'initial_states_lvlh',
                           'close_route_start_body_m', 'carrier_context')
                if any(name not in deployment for name in carrier):
                    raise ValueError('Carrier restart requires the complete release/transfer policy')
                names += carrier
        result["deployment"] = {name: deployment[name] for name in names if name in deployment}
    for name in ("tracked_reference_point", "belief_update_mode", "chaser_body_bounding_radius_m", "vehicle_design", "arrival_policy"):
        if name in metadata:
            result[name] = metadata[name]
    if "arrival_policy" in result:
        from orbexa.simulation.arrival_policy import ArrivalPolicyConfig
        result["arrival_policy"] = ArrivalPolicyConfig.from_dict(result["arrival_policy"]).as_dict()
    # Persisted JSON turns tuples/NumPy scalars into lists/Python scalars.
    # Compare supported policy values in that same representation on both sides.
    return plain(result)


def _target_signature(metadata):
    names = ("shape", "radius", "height", "half_length", "tolerance", "docking_standoff",
             "initial_orientation", "angular_velocity", "attitude_model", "motion_model",
             "gravity_gradient", "initial_quaternion", "initial_body_angular_velocity_rad_s",
             "inertia_kg_m2", "clock_parameters", "anomaly_domain", "integration",
             "geometry", "geometry_fingerprint", "geometry_model", "mass_kg")
    return {name: metadata[name] for name in names if name in metadata}


def physical_target_from_snapshot(snapshot):
    """Reconstruct an explicitly identified physical shape without substitution.

    Composite geometry is loaded by its own validated target adapter.  The
    cylinder branch is retained solely for previously declared cylinder runs.
    Missing or unsupported shapes never select that branch implicitly.
    """
    shape = snapshot.get("shape")
    if shape == "composite_spacecraft":
        from orbexa.core.spacecraft_target import RigidBodySpacecraftTarget
        return RigidBodySpacecraftTarget.from_dict(snapshot)
    if shape != "cylinder":
        raise ValueError("Unsupported physical target snapshot geometry; no cylinder fallback is available")
    from orbexa.core.rigid_body_target import RigidBodyCylinderTarget
    clock, integration = snapshot["clock_parameters"], snapshot["integration"]
    return RigidBodyCylinderTarget(radius=snapshot["radius"], half_length=snapshot["half_length"],
        initial_quaternion=snapshot["initial_quaternion"],
        initial_body_angular_velocity=snapshot["initial_body_angular_velocity_rad_s"],
        inertia=snapshot["inertia_kg_m2"], eccentricity=clock["eccentricity"], alpha=clock["alpha"],
        mu=clock["mu"], semi_major_axis=clock["semi_major_axis"], anomaly_domain=snapshot["anomaly_domain"],
        tolerance=snapshot["tolerance"], docking_standoff=snapshot["docking_standoff"],
        gravity_gradient=snapshot["gravity_gradient"], integration_rtol=integration["rtol"],
        integration_atol=integration["atol"], integration_max_step=integration["max_anomaly_step"])


@dataclass(frozen=True)
class MissionCheckpoint:
    """A committed record plus its immutable identity and explicit source context."""

    identity: str
    source_record: dict
    source_record_sha256: str
    source_canonical_sha256: str
    source_context: dict
    schema: str = CHECKPOINT_SCHEMA

    @classmethod
    def from_result(cls, result: ADTMPCMissionResult, *, source_context: dict | None = None,
                    source_record_sha256: str | None = None):
        record = plain(result.to_dict())
        canonical = digest(record)
        context = plain(source_context or {})
        source_sha = source_record_sha256 or canonical
        payload = dict(schema=CHECKPOINT_SCHEMA, source_record_sha256=source_sha,
                       source_canonical_sha256=canonical, source_context=context)
        checkpoint = cls(digest(payload), record, source_sha, canonical, context)
        checkpoint._validate_record()
        return checkpoint

    @classmethod
    def from_file(cls, path: Path, *, source_context: dict | None = None):
        path = Path(path)
        raw = path.read_bytes()
        payload = json.loads(raw)
        result = ADTMPCMissionResult.from_dict(payload.get("result", payload))
        context = dict(source_context or {})
        context.setdefault("source_reference", str(path.resolve()))
        return cls.from_result(result, source_context=context,
                               source_record_sha256=hashlib.sha256(raw).hexdigest())

    @property
    def mission(self):
        return self.source_record["mission"]

    @property
    def anomaly(self):
        return float(self.source_record["anom_history"][-1])

    @property
    def committed_updates(self):
        return len(self.source_record["anom_history"]) - 1

    @property
    def result(self):
        return ADTMPCMissionResult.from_dict(copy.deepcopy(self.source_record))

    def _validate_record(self):
        if self.schema != CHECKPOINT_SCHEMA or self.source_canonical_sha256 != digest(self.source_record):
            raise ValueError("Checkpoint schema or source semantic digest does not match")
        payload = dict(schema=self.schema, source_record_sha256=self.source_record_sha256,
                       source_canonical_sha256=self.source_canonical_sha256, source_context=self.source_context)
        if self.identity != digest(payload):
            raise ValueError("Checkpoint identity does not match its source and context")
        record, metadata = self.source_record, self.source_record["metadata"]
        if record["target"].get("shape") == "composite_spacecraft":
            from orbexa.core.spacecraft_geometry import SpacecraftGeometry
            geometry = SpacecraftGeometry.from_dict(record["target"]["geometry"])
            if record["target"].get("geometry_fingerprint") != geometry.fingerprint:
                raise ValueError("Composite checkpoint target geometry identity is missing or inconsistent")
        if (metadata.get("model_version") != PHYSICAL_MODEL_ID
                or metadata.get("controller") != COUPLED_CONTROLLER_ID
                or metadata.get("tube_model_version") != TUBE_MODEL_VERSION):
            raise ValueError("Only the maintained physical held-tube mission can be numerically resumed")
        clock = np.asarray(record["anom_history"], dtype=float)
        if clock.ndim != 1 or len(clock) < 1 or not np.all(np.isfinite(clock)) or np.any(np.diff(clock) <= 0):
            raise ValueError("Checkpoint requires an ordered committed anomaly history")
        kinematics = metadata.get("target_kinematics")
        if kinematics is not None:
            if not isinstance(kinematics, dict) or not {"anomaly", "physical_time"} <= set(kinematics):
                raise ValueError("Saved target kinematics require anomaly and physical-time records")
            for name, values in kinematics.items():
                values = np.asarray(values, dtype=float)
                if values.ndim < 1 or values.shape[0] != len(clock) or not np.all(np.isfinite(values)):
                    raise ValueError(f"Complete finite numeric target kinematics {name} are required")
            if not np.array_equal(np.asarray(kinematics["anomaly"], dtype=float), clock):
                raise ValueError("Saved target kinematics do not use the committed event clock")
            if not np.array_equal(np.asarray(kinematics["physical_time"], dtype=float),
                                  np.asarray(metadata.get("elapsed_time_seconds", []), dtype=float)):
                raise ValueError("Saved target kinematics and elapsed-time record disagree")
        if record["mission"] not in ("single", "multi") or record["approximation"] != "nonlinear" or record["solver_backend"] != "gekko":
            raise ValueError("Only nonlinear GEKKO single/multi mission records are supported")
        nodes, intervals = len(clock), len(clock) - 1
        if len(record["sample_phase_history"]) != nodes or len(record["phase_history"]) < intervals:
            raise ValueError("Phase histories do not cover the committed fleet events")
        dwell = metadata.get("terminal_dwell_observed", 0)
        if isinstance(dwell, bool) or not isinstance(dwell, int) or not 0 <= dwell <= intervals:
            raise ValueError("Invalid saved terminal dwell count")
        for name, values in (("elapsed time", metadata.get("elapsed_time_seconds", [])),
                             ("tube radius", record["tube_radius_history"])):
            values = np.asarray(values, dtype=float)
            if values.shape != (nodes,) or not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError(f"Complete finite nonnegative {name} history is required")
        if len(record["feasible_set_history"]) != nodes or len(record["parameter_estimate_history"]) != nodes:
            raise ValueError("Belief histories do not cover the committed fleet events")
        final_belief = ParameterBelief(**copy.deepcopy(record["final_belief"]))
        # ParameterBelief normalizes its input dictionaries in place. Validation
        # must not convert the authoritative historical JSON lists into tuples.
        last_belief = ParameterBelief(copy.deepcopy(record["feasible_set_history"][-1]),
                                     copy.deepcopy(record["parameter_estimate_history"][-1]))
        if final_belief.as_metadata() != last_belief.as_metadata():
            raise ValueError("Final belief disagrees with the last committed belief record")
        ids = list(record["actual_trajectories"])
        assignments = metadata.get("chaser_assignments", [])
        if not ids or [item.get("chaser_id") for item in assignments] != ids:
            raise ValueError("A complete ordered original assignment record is required")
        for assignment in assignments:
            checked = ChaserConfig(**copy.deepcopy(assignment))
            if record["docking_points"].get(checked.chaser_id) != checked.docking_point_body.tolist():
                raise ValueError("Saved reference point and original assignment disagree")
        for name in ("controls", "nominal_trajectories", "tube_profiles", "solve_stats", "solver_costs", "tube_diagnostics"):
            if set(record[name]) != set(ids):
                raise ValueError(f"Incomplete saved {name} fleet")
        for key in ids:
            controls = np.asarray(record["controls"][key], dtype=float)
            if ((intervals and controls.shape != (intervals, 3)) or (not intervals and controls.size)
                    or not np.all(np.isfinite(controls))):
                raise ValueError("Applied outer input history does not match the committed events")
            if (len(record["nominal_trajectories"][key]) < intervals
                    or len(record["tube_profiles"][key]) != len(record["nominal_trajectories"][key])
                    or len(record["solve_stats"][key]) < intervals
                    or len(record["solver_costs"][key]) != len(record["solve_stats"][key])):
                raise ValueError("Incomplete original planning or solve-attempt history")
            values = np.asarray(record["actual_trajectories"][key], dtype=float)
            if values.shape != (len(clock), 6) or not np.all(np.isfinite(values)):
                raise ValueError("Actual state history does not match the committed clock")
            diagnostic = record["tube_diagnostics"][key]
            node_fields = {"nominal_center": 6, "phi": 3, "omega": 3, "bandwidth": 3,
                           "position_error": 3, "sliding_error": 3, "containment_margin": None}
            interval_fields = {"terminal_position_bound": None, "terminal_velocity_bound": None,
                "max_dense_input_reserve_ratio": None, "nominal_control": 3, "ancillary_control_rms": 3,
                "applied_control_rms": 3, "applied_control_effort": None, "dense_containment_margin": None,
                "dense_physical_clearance": None, "nominal_center_defect": 6}
            for name, width in {**node_fields, **interval_fields}.items():
                values = diagnostic.get(name)
                expected = nodes if name in node_fields else intervals
                if not isinstance(values, list) or len(values) != expected:
                    raise ValueError(f"Complete saved diagnostic {name} is required before restart")
                for value in values:
                    if name == "nominal_center_defect" and value is None:
                        continue
                    array = np.asarray(value, dtype=float)
                    if array.shape != (() if width is None else (width,)) or not np.all(np.isfinite(array)):
                        raise ValueError(f"Invalid saved diagnostic {name}")
            for name, width in (("nominal_center", 6), ("phi", 3), ("omega", 3), ("bandwidth", 3)):
                array = np.asarray(diagnostic[name], dtype=float)
                if array.shape != (len(clock), width) or not np.all(np.isfinite(array)):
                    raise ValueError(f"Complete persistent {name} history is required")
                if name != "nominal_center" and np.any(array < 0):
                    raise ValueError("Persistent tube data must be nonnegative")
        self.arrival_runtime()

    def arrival_runtime(self):
        """Reconstruct only the typed policy state, never a serialized backup."""
        metadata = self.source_record['metadata']
        policy = metadata.get('scenario', {}).get('arrival_policy')
        context = metadata.get('checkpoint_context') or self.source_context.get('checkpoint_context', {})
        context_policy = context.get('policy', {}).get('arrival_policy')
        if policy != context_policy:
            raise ValueError('Recorded arrival policy and source checkpoint context disagree')
        if policy is None:
            if 'arrival_state' in metadata or 'arrival_activation' in metadata:
                raise ValueError('Arrival state cannot be silently detached from its enabled policy')
            return None
        from orbexa.simulation.arrival_policy import ArrivalPolicyConfig, ArrivalRuntime
        runtime = ArrivalRuntime.from_metadata(ArrivalPolicyConfig.from_dict(policy), metadata.get('arrival_state'),
                                              activation_marker=metadata.get('arrival_activation'))
        if runtime.transition_identity is not None:
            ledger = metadata.get('policy_transition_history', [])
            matches = [item for item in ledger if item.get('identity') == runtime.transition_identity]
            if len(matches) != 1:
                raise ValueError('Arrival state lost its source policy-transition ledger')
            origin = matches[0]
            payload = {key: value for key, value in origin.items() if key != 'identity'}
            if (digest(payload) != runtime.transition_identity
                    or runtime.enabled_at_update != origin['committed_updates']
                    or runtime.exact_enabled_anomaly != Fraction(origin['exact_start_anomaly'])):
                raise ValueError('Arrival activation changed its exact source transition seam')
        elif runtime.enabled_at_update != 0 or runtime.exact_enabled_anomaly != 0:
            raise ValueError('A fresh arrival policy must retain its initial update0/exact0 origin')
        if runtime.enabled_at_update > self.committed_updates:
            raise ValueError('Arrival activation is later than the saved committed cursor')
        if runtime.commitment is None and 'docking' in self.source_record['phase_history'][
                runtime.enabled_at_update:self.committed_updates]:
            raise ValueError('Committed docking history cannot lose its accepted arrival deadline')
        if runtime.latest_candidate is not None and runtime.latest_candidate['payload']['step']['current_update'] > self.committed_updates:
            raise ValueError('The latest arrival candidate is later than the saved committed cursor')
        return runtime

    def validate_scenario(self, scenario, mission):
        self._validate_record()
        record, metadata = self.source_record, self.source_record["metadata"]
        if mission != self.mission or metadata.get("mission_complete"):
            raise ValueError("Resume the same unfinished mission")
        source = metadata.get("checkpoint_context") or self.source_context.get("checkpoint_context")
        if source is None:
            raise ValueError("Older records require explicit checkpoint_context from their frozen configuration/source")
        if not source.get("source_fingerprints") or not isinstance(source.get("configuration"), dict):
            raise ValueError("A numerical restart requires an identified source and complete frozen configuration")
        if source.get("truth") != record["truth"]:
            raise ValueError("Source configuration context disagrees with the saved physical truth")
        current = scenario_checkpoint_context(scenario)
        if metadata.get("horizon_steps") != scenario.reference.configured_horizon_metadata():
            raise ValueError("Numerical restart changed the planning horizon policy")
        if _configuration_signature(source["configuration"]) != _configuration_signature(current["configuration"]):
            raise ValueError("Numerical restart changed a model, mission, controller or adaptation configuration")
        if record["truth"] != scenario.truth.as_dict():
            raise ValueError("Numerical restart changed the physical orbital truth")
        old_target, new_target = _target_signature(record["target"]), _target_signature(current["target"])
        if any(name not in new_target or new_target[name] != value for name, value in old_target.items()):
            raise ValueError("Numerical restart changed the target motion, geometry, clock or epoch")
        old_policy = _policy_signature(source.get("policy", metadata.get("scenario", {})))
        new_policy = _policy_signature(current["policy"])
        if old_policy.get('arrival_policy') != new_policy.get('arrival_policy'):
            raise ValueError('Changing an arrival policy requires an explicit MissionPolicyTransition')
        policy_changed = any(name not in new_policy or new_policy[name] != value for name, value in old_policy.items())
        if record["target"].get("shape") == "composite_spacecraft":
            # A missing source policy field cannot inherit a newly generated
            # composite route, schedule or installed-vehicle declaration.
            policy_changed = old_policy != new_policy
        if policy_changed:
            raise ValueError("Numerical restart changed the reference/deployment policy")
        if source.get("body_radius") != current["body_radius"] or source.get("adaptation_mode", "adaptive") != current["adaptation_mode"]:
            raise ValueError("Numerical restart changed body geometry or belief-update mode")
        for name, value in (("anom_step", scenario.config.anom_step), ("feedback_substeps", scenario.config.tube.feedback_substeps)):
            if metadata.get(name) != value:
                raise ValueError("Numerical restart changed the execution hold schedule")
        if (metadata.get("continuous_execution_certificate") is not None) != (scenario.execution_guard is not None):
            raise ValueError("A numerical restart cannot silently add or remove the historical continuous guard")
        saved_kinematics = metadata.get("target_kinematics")
        if saved_kinematics is not None:
            if not callable(getattr(scenario.target, "recorded_kinematics", None)):
                raise ValueError("Numerical restart removed the recorded target kinematics interface")
            proposed = scenario.target.recorded_kinematics([self.anomaly])
            if set(saved_kinematics) != set(proposed):
                raise ValueError("Numerical restart changed the target kinematic record schema")
            for name, values in saved_kinematics.items():
                saved, following = np.asarray(values), np.asarray(proposed[name])
                if saved.shape[:1] != (len(record["anom_history"]),) or saved.shape[1:] != following.shape[1:]:
                    raise ValueError("Saved target kinematic history has an incompatible shape")
        runtime = self.arrival_runtime()
        if runtime is not None:
            from orbexa.simulation.arrival_policy import exact_outer_step
            guarded = metadata.get('continuous_execution_certificate') or {}
            exact_end = guarded.get('exact_end_anomaly')
            if exact_end is None and self.committed_updates == 0 and self.anomaly == 0.:
                exact_end = '0'
            if (exact_end is None or guarded.get('admitted_holds') !=
                    self.committed_updates * scenario.config.tube.feedback_substeps):
                raise ValueError('Arrival restart requires the original exact clock and complete outer-grid prefix')
            if runtime.commitment is not None:
                runtime.commitment.check_clock(self.committed_updates, exact_end, exact_outer_step(scenario.config))
        return current

    def restore(self, scenario):
        self.validate_scenario(scenario, self.mission)
        return self._restore_validated(scenario)

    def _restore_validated(self, scenario):
        """Shared restoration after ordinary validation or an explicit transition.

        The transition API separately checks every non-arrival configuration and
        policy field. Guard.restore below retains all original physical gates.
        """
        record, metadata = self.source_record, self.source_record["metadata"]
        ids = list(record["actual_trajectories"])
        chasers = [ChaserConfig(**copy.deepcopy(value)) for value in metadata["chaser_assignments"]]
        diagnostics = record["tube_diagnostics"]
        state = CoupledMissionState(anomaly=self.anomaly,
            states={key: np.asarray(record["actual_trajectories"][key][-1], dtype=float).copy() for key in ids},
            centers={key: np.asarray(diagnostics[key]["nominal_center"][-1], dtype=float).copy() for key in ids},
            phi={key: np.asarray(diagnostics[key]["phi"][-1], dtype=float).copy() for key in ids},
            omega={key: np.asarray(diagnostics[key]["omega"][-1], dtype=float).copy() for key in ids},
            bandwidth={key: np.asarray(diagnostics[key]["bandwidth"][-1], dtype=float).copy() for key in ids})
        arrival = self.arrival_runtime()
        if arrival is not None:
            state.arrival_commitment = arrival.commitment
        lam = np.asarray(scenario.config.tube.sliding_gains)
        for key in ids:
            error = state.states[key] - state.centers[key]
            if np.max(np.abs(error[:3]) - state.omega[key]) > 1e-8 or np.max(
                    np.abs(error[3:] + lam * error[:3]) - state.phi[key]) > 1e-8:
                raise ValueError("The saved actual state is outside its persistent tube")
        name = record["sample_phase_history"][-1]
        if name not in ("deployment", "staging", "rendezvous", "docking"):
            raise ValueError("Unknown phase operation requires an explicit phase restoration interface")
        phase = MissionPhase(name, "docking" if name == "docking" else "rendezvous")
        if scenario.execution_guard is not None:
            source_guard_context = copy.deepcopy(self.source_context.get("guard_context", {}))
            if "input_bounds" in metadata:
                source_guard_context.setdefault("input_lower", [entry["lower"] for entry in metadata["input_bounds"]])
                source_guard_context.setdefault("input_upper", [entry["upper"] for entry in metadata["input_bounds"]])
                source_guard_context.setdefault("source_reference", "Recorded metadata.input_bounds in " + self.source_record_sha256)
            scenario.execution_guard.restore(metadata["continuous_execution_certificate"],
                chaser_ids=ids, anomaly=state.anomaly,
                hold_width=scenario.config.anom_step / scenario.config.tube.feedback_substeps,
                source_context=source_guard_context,
                checkpoint_identity=self.identity)
        belief = ParameterBelief(**copy.deepcopy(record["final_belief"]))
        return chasers, state, belief, phase, int(metadata.get("terminal_dwell_observed", 0))

    def restart_record(self, scenario):
        old = self.source_record["metadata"].get("checkpoint_context") or self.source_context.get("checkpoint_context", {})
        return dict(checkpoint_schema=self.schema, checkpoint_identity=self.identity,
            source_record_sha256=self.source_record_sha256, source_reference=self.source_context.get("source_reference"),
            source_success=self.source_record["success"], source_message=self.source_record["message"],
            committed_updates=self.committed_updates, anomaly=self.anomaly,
            adaptation_window="Reset to the saved current primary state; mean outer inputs cannot recover the exact dense window",
            planning_warm_starts="Cold start; unavailable complete control horizons are not reconstructed",
            source_solver_options=plain(old.get("configuration", {}).get("solver", {})),
            resumed_solver_options=plain(scenario.config.solver),
            source_fingerprints=old.get("source_fingerprints", {}), resumed_source_fingerprints=source_fingerprints(),
            scope="Numerical continuation of a retained prefix; not a matched statistical trial or a formal safe-backup theorem")

    def as_dict(self):
        return plain(asdict(self))

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(self.as_dict(), allow_nan=False), encoding="utf-8")
        temporary.replace(path)
        return path

    @classmethod
    def load(cls, path: Path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        checkpoint = cls(**payload)
        checkpoint._validate_record()
        return checkpoint
