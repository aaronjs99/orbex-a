"""Verify a separate physical terminal record before using its actual endpoint.

The original MPC record remains untouched.  The combined trajectory is only a
numerical attitude/accounting view: it contains actual states and commands, not
invented MPC solutions, nominal plans, measurements, or adaptation histories.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Mapping
import zlib

import numpy as np

from orbexa.control.physical_terminal import PhysicalTerminalBoundsBuilder, PhysicalTerminalErrorBounds
from orbexa.control.reachability import (
    HeldInputRequest, IntervalBox, OrbitalModelBounds, ReachabilityConfig,
    RotatingCylinderBounds, _exact, _float_outward,
)
from orbexa.core.rigid_body_target import RigidBodyCylinderTarget
from orbexa.simulation.capture import CaptureModel, CaptureTruth, TargetPhysicalProperties
from orbexa.simulation.mission_checkpoint import digest, plain, _target_signature
from orbexa.simulation.mission_result import ADTMPCMissionResult
from orbexa.simulation.physical_terminal_execution import (
    EXECUTION_METHOD, PhysicalTerminalRunner, TerminalObservation,
)


HANDOFF_METHOD = "verified-physical-terminal-handoff-v1"


def _require(condition, reason):
    if not condition:
        raise ValueError("Physical terminal handoff: " + reason)


def _box(value):
    return IntervalBox(value["exact_lower"], value["exact_upper"])


def _observation(value):
    return TerminalObservation(value["measurements"], tuple(map(_box, value["state_enclosures"])))


def _builder(value):
    arguments = copy.deepcopy(value)
    for name in ("target_orbit", "plant_parameters", "nominal_parameters"):
        arguments[name] = OrbitalModelBounds(**arguments[name])
    for name in ("entry_spin", "input_limits"):
        arguments[name] = _box(arguments[name])
    arguments["errors"] = PhysicalTerminalErrorBounds(**arguments["errors"])
    return PhysicalTerminalBoundsBuilder(**arguments)


def _target(spec):
    clock, integration = spec["clock_parameters"], spec["integration"]
    return RigidBodyCylinderTarget(radius=spec["radius"], half_length=spec["half_length"],
        initial_quaternion=spec["initial_quaternion"],
        initial_body_angular_velocity=spec["initial_body_angular_velocity_rad_s"],
        inertia=spec["inertia_kg_m2"], eccentricity=clock["eccentricity"], alpha=clock["alpha"],
        mu=clock["mu"], semi_major_axis=clock["semi_major_axis"], anomaly_domain=spec["anomaly_domain"],
        tolerance=spec["tolerance"], docking_standoff=spec["docking_standoff"],
        gravity_gradient=spec["gravity_gradient"], integration_rtol=integration["rtol"],
        integration_atol=integration["atol"], integration_max_step=integration["max_anomaly_step"])


def _readonly(values):
    value = np.array(values, dtype=float, copy=True)
    _require(np.all(np.isfinite(value)), "endpoint arrays must be finite")
    value.setflags(write=False)
    return value


def _outward(values):
    return _readonly(tuple(_float_outward(value, True) for value in values))


@dataclass(frozen=True)
class PhysicalTerminalTrajectory:
    """Actual translation/target samples for attitude replay and fuel accounting.

    This deliberately does not implement ADTMPCMissionResult.  The controls are
    the source approach's reported applied means followed by actual held
    terminal commands, with their different semantics explicitly recorded.
    """

    anom_history: np.ndarray
    actual_trajectories: Mapping[str, np.ndarray]
    controls: Mapping[str, np.ndarray]
    target: dict
    truth: dict
    metadata: dict


@dataclass(frozen=True, init=False)
class PhysicalTerminalHandoff:
    """Source-bound, independently rechecked finite terminal endpoint.

    Construct with ``from_files``.  Status strings and hashes alone never
    establish admission: global bounds, commands, continuous flows and swept
    separation are recomputed from the declared contracts.  Sensor/reference
    error validity and the initial physical spin enclosure remain assumptions.
    """

    trajectory: PhysicalTerminalTrajectory
    capture_truth: CaptureTruth
    states: Mapping[str, np.ndarray]
    nominal_centers: Mapping[str, np.ndarray]
    omega: Mapping[str, np.ndarray]
    phi: Mapping[str, np.ndarray]
    lambda_gain: np.ndarray
    exact_end_anomaly: Fraction
    _approach_digest: str = field(repr=False)
    _summary: dict = field(repr=False)
    _integrity_digest: str = field(repr=False)

    def __init__(self, *args, **kwargs):
        raise TypeError("Construct a verified physical terminal handoff with from_files")

    @classmethod
    def from_files(cls, approach_path, terminal_result_path):
        """Load and verify a complete, guarded approach and its terminal record.

        This focused v1 accepts the producer's default Arb flow policy (with
        each saved precision) and a single common Lambda.  Unsupported records
        refuse instead of inferring missing execution or sensor assumptions.
        An adjacent continuous-certificates.jsonl.gz journal, when present,
        must match every stored hold exactly.
        """
        try:
            return cls._from_files(approach_path, terminal_result_path)
        except (KeyError, TypeError, IndexError, OverflowError, ZeroDivisionError, EOFError, zlib.error) as error:
            raise ValueError("Physical terminal handoff: malformed or missing source/proof field: " + str(error)) from error

    @classmethod
    def _from_files(cls, approach_path, terminal_result_path):
        approach_path, terminal_result_path = Path(approach_path), Path(terminal_result_path)
        approach_raw, terminal_raw = approach_path.read_bytes(), terminal_result_path.read_bytes()
        payload = json.loads(approach_raw)
        approach = ADTMPCMissionResult.from_dict(payload.get("result", payload))
        record = plain(approach.to_dict())
        execution = json.loads(gzip.decompress(terminal_raw) if terminal_result_path.suffix == ".gz" else terminal_raw)
        source_sha, terminal_sha = (hashlib.sha256(value).hexdigest() for value in (approach_raw, terminal_raw))
        _require(approach.success and approach.metadata.get("mission_complete") is True,
                 "the originating approach did not complete")
        _require(execution.get("method") == EXECUTION_METHOD and execution.get("success") is True
                 and execution.get("status") == "COMPLETED_FINITE_WINDOW"
                 and execution.get("reset_performed") is False, "the terminal execution is incomplete or reset")
        context = execution["source_context"]
        _require(context["caller"].get("approach_result_sha256") == source_sha,
                 "the terminal record names a different raw approach source")
        _require(_target_signature(context["target"]) == _target_signature(record["target"]),
                 "the physical target epoch, geometry, dynamics or numerical domain changed")
        guarded = record["metadata"].get("continuous_execution_certificate")
        _require(isinstance(guarded, dict) and isinstance(guarded.get("exact_end_anomaly"), str),
                 "a guarded source with its exact committed endpoint is required")
        clocks = tuple(map(_exact, execution["anomaly_history"]))
        _require(len(clocks) >= 2 and all(a < b for a, b in zip(clocks, clocks[1:])),
                 "the terminal clock must strictly increase")
        _require(clocks[0] == Fraction(guarded["exact_end_anomaly"]),
                 "the terminal entry is not the exact committed approach event")
        _require(_exact(execution["final_observation_anomaly"]) == clocks[-1],
                 "the terminal final observation is stale")
        ids = tuple(record["actual_trajectories"])
        count = len(ids)
        states = np.asarray(execution["actual_state_history"], dtype=float)
        _require(states.shape == (len(clocks), count, 6) and np.all(np.isfinite(states)),
                 "terminal actual state history has an invalid shape or value")
        initial = np.array([record["actual_trajectories"][key][-1] for key in ids])
        _require(np.array_equal(states[0], initial), "the actual approach endpoint was replaced")
        saved_certificates = execution["bound_certificates"]
        _require(len(saved_certificates) == count, "global certificate fleet differs from the source")
        builders = tuple(_builder(value["assumptions"]) for value in saved_certificates)
        source_body_radius = _exact(guarded["body_radius_m"])
        scenario_body_radius = _exact(record["metadata"]["scenario"]["chaser_body_bounding_radius_m"])
        _require(source_body_radius > 0 and scenario_body_radius > 0
                 and all(builder.body_radius >= max(source_body_radius, scenario_body_radius) for builder in builders),
                 "a terminal proof uses a smaller body envelope than the originating physical approach")
        source_inputs = IntervalBox(guarded["input_lower"], guarded["input_upper"])
        _require(source_inputs.dimension == 3 and all(source_inputs.contains(builder.input_limits) for builder in builders),
                 "terminal actuator limits exceed the originating declared input envelope")
        _require(all(builder.start_anomaly == clocks[0] and builder.end_anomaly == clocks[-1]
                     and builder.port_index == index for index, builder in enumerate(builders)),
                 "the completed finite window or body-point order differs from the certificates")
        _require(all(builder.lambda_gain == builders[0].lambda_gain for builder in builders),
                 "this handoff requires a common declared Lambda")
        ports = tuple(tuple(map(_exact, record["docking_points"][key])) for key in ids)
        _require(all(builder.body_ports == ports for builder in builders), "assigned body points changed")
        assignments = record["metadata"].get("chaser_assignments", [])
        _require({item["chaser_id"] for item in assignments} == set(ids)
                 and all(tuple(map(_exact, item["docking_point_body"])) == ports[ids.index(item["chaser_id"])]
                         for item in assignments), "source body-point assignments contradict their stored points")
        parameter_context = dict(mu=record["truth"]["mu"], semi_major_axis=record["truth"]["semi_major_axis"])
        truth = OrbitalModelBounds(**{name: (record["truth"][name],) * 2
                                    for name in ("eccentricity", "alpha", "beta")}, **parameter_context)
        _require(truth == OrbitalModelBounds(**context["truth"]), "numerical physical translation truth changed")
        prior = OrbitalModelBounds(**{name: record["initial_belief"]["feasible_sets"][name]
                                    for name in ("eccentricity", "alpha", "beta")}, **parameter_context)
        nominal = OrbitalModelBounds(**{name: (record["final_belief"]["estimates"][name],) * 2
                                      for name in ("eccentricity", "alpha", "beta")}, **parameter_context)
        _require(all(builder.plant_parameters == prior and builder.nominal_parameters == nominal for builder in builders),
                 "terminal prior or nominal parameter context differs from the actual source")
        discrepancy = abs(_exact(record["anom_history"][-1]) - clocks[0])
        _require(all(discrepancy <= builder.errors.anomaly_evaluation_error_abs for builder in builders),
                 "the source reporter/exact event discrepancy exceeds the declared query budget")
        first_observation, last_observation = (_observation(execution[name])
                                               for name in ("initial_observation", "final_observation"))
        _require(len(first_observation.measurements) == count and len(last_observation.measurements) == count,
                 "the endpoint observations have a different fleet")
        _require(set(guarded.get("endpoint_boxes", {})) == set(ids), "the source exact endpoint boxes are missing")
        for index, (key, builder) in enumerate(zip(ids, builders)):
            source_box = _box(guarded["endpoint_boxes"][key])
            sensor_box = IntervalBox.from_center_radius(first_observation.measurements[index], builder.errors.sensing_error_abs)
            intersection = IntervalBox(tuple(max(a, b) for a, b in zip(source_box.lower, sensor_box.lower)),
                                       tuple(min(a, b) for a, b in zip(source_box.upper, sensor_box.upper)))
            _require(first_observation.state_enclosures[index] == intersection,
                     "initial observation is not the actual sensor/source endpoint intersection")
        holds = execution["hold_records"]
        _require(len(holds) == len(clocks) - 1 and bool(holds), "the terminal execution has a missing or refused hold")
        geometry_error = holds[0]["geometry"]["target_bounds"]["orientation_matrix_error"]
        target = _target(context["target"])
        runner = PhysicalTerminalRunner(builders=builders, target=target, truth=truth,
            observation_source=lambda *args: None, orientation_matrix_error=geometry_error,
            minimum_pair_center_distance=holds[0]["geometry"]["minimum_pair_center_distance"])
        _require(not runner._observation_reason(first_observation, states[0]), "invalid actual entry observation")
        _require(runner.fleet.admit(first_observation.state_enclosures, runner.references_at(clocks[0])).admissible,
                 "global proof or robust actual entry admission failed")
        _require(digest(execution["admission"]) == digest(runner.fleet.admit(
            first_observation.state_enclosures, runner.references_at(clocks[0])).as_dict()), "saved admission differs")
        for saved, controller in zip(saved_certificates, runner.fleet.controllers):
            _require(digest(saved) == digest(controller.result.as_dict()),
                     "a saved global certificate differs from its freshly derived proof")
        command_rows, previous_endpoints = [], first_observation.state_enclosures
        flow_count = 0
        for index, hold in enumerate(holds):
            q, width = clocks[index], clocks[index + 1] - clocks[index]
            _require(hold.get("method") == EXECUTION_METHOD and hold.get("stage") == index
                     and hold.get("executed") is True and hold.get("decision") == "EXECUTED_PROVEN_HOLD"
                     and _exact(hold["start_anomaly"]) == q and _exact(hold["hold_width"]) == width
                     and _exact(hold["end_anomaly"]) == q + width,
                     f"hold {index} has an invalid exact execution event")
            _require(np.array_equal(np.asarray(hold["actual_endpoint"]), states[index + 1]),
                     f"hold {index} endpoint differs from the actual history")
            observation = _observation(hold["initial_observation"])
            if index == 0:
                _require(digest(observation.as_dict()) == digest(first_observation.as_dict()), "initial hold observation changed")
            _require(not runner._observation_reason(observation, states[index]), f"hold {index} observation is invalid")
            _require(all(a.contains(b) for a, b in zip(previous_endpoints, observation.state_enclosures)),
                     f"hold {index} observation does not refine the preceding proved endpoints")
            commands = runner.fleet.commands(observation.measurements, observation.state_enclosures,
                                             runner.references_at(q), hold_width=width)
            _require(digest([command.as_dict() for command in commands]) == digest(hold["commands"]),
                     f"hold {index} commands differ from the actual held law")
            _require(len(hold["flows"]) == count, f"hold {index} continuous fleet is incomplete")
            flows = []
            for member, (builder, command, box, saved) in enumerate(zip(builders, commands,
                    observation.state_enclosures, hold["flows"])):
                request = HeldInputRequest(initial_state=box, held_input=command.applied_input,
                    start_anomaly=(q, q), hold_width=(width, width),
                    acceleration_disturbance=IntervalBox.from_center_radius((0, 0, 0),
                        builder.errors.acceleration_disturbance_abs))
                _require(digest(saved.get("request")) == digest(request.as_dict())
                         and digest(saved.get("model_bounds")) == digest(builder.plant_parameters.as_dict()),
                         f"hold {index}, member {member} continuous proof request changed")
                # The record stores precision, but not arbitrary custom stepping
                # policies.  Reprove with the producer's documented default.
                from orbexa.control.reachability import ValidatedFlowEncloser
                flow = ValidatedFlowEncloser(ReachabilityConfig(precision_bits=saved["precision_bits"])).enclose_orbital(
                    request, builder.plant_parameters)
                _require(flow.proven and digest(flow.as_dict()) == digest(saved),
                         f"hold {index}, member {member} continuous Arb proof did not reproduce")
                _require(flow.endpoint.contains(IntervalBox.point(states[index + 1, member])),
                         f"hold {index}, member {member} actual endpoint leaves its proved enclosure")
                flows.append(flow)
            fingerprints = [flow.fingerprint for flow in flows]
            _require(fingerprints == hold["flow_fingerprints"]
                     and hold.get("numerical_endpoint_contained") == [True] * count,
                     f"hold {index} proof fingerprint or endpoint flags differ")
            first = builders[0]
            geometry = RotatingCylinderBounds(radius=first.target_radius * (1 + first.target_tolerance),
                half_length=first.target_half_length * (1 + first.target_tolerance),
                orientation_matrix_at_start=target.rotation_matrix_at(float(q)),
                orientation_matrix_error=runner.orientation_matrix_error,
                angular_rate_per_anomaly=max(controller.result.derived["rotation_derivative_operator_bounds"][0]
                    for controller in runner.fleet.controllers),
                scope="Derived global physical terminal rotation-rate bound and explicitly declared matrix evaluation error")
            collision = runner.guard.check(flows, tuple(builder.body_radius for builder in builders), target=geometry,
                clearance=first.clearance + max(builder.errors.target_solid_position_error_radius for builder in builders),
                minimum_pair_center_distance=runner.minimum_pair_center_distance)
            _require(collision.admissible and digest(collision.as_dict()) == digest(hold["geometry"]),
                     f"hold {index} whole-fleet swept geometry proof did not reproduce")
            linkage = dict(start_anomaly=str(q), hold_width=str(width), commands=hold["commands"],
                           flow_fingerprints=fingerprints, geometry=hold["geometry"])
            _require(digest(linkage) == hold["command_flow_geometry_sha256"]
                     and digest(linkage) == digest(hold["command_flow_geometry_linkage"]),
                     f"hold {index} command/flow/geometry linkage changed")
            command_rows.append([command.float_command() for command in commands])
            previous_endpoints = tuple(flow.endpoint for flow in flows)
            flow_count += len(flows)
        _require(not runner._observation_reason(last_observation, states[-1])
                 and all(a.contains(b) for a, b in zip(previous_endpoints, last_observation.state_enclosures))
                 and runner.fleet.admit(last_observation.state_enclosures, runner.references_at(clocks[-1])).admissible,
                 "final actual observation, preceding enclosure or robust terminal membership failed")
        duration = target.physical_time_at(float(clocks[-1])) - target.physical_time_at(float(clocks[0]))
        _require(np.isfinite(execution["physical_duration_seconds"])
                 and abs(duration - execution["physical_duration_seconds"]) <= 1e-9,
                 "reported terminal physical duration differs from the target clock")
        journal = terminal_result_path.with_name("continuous-certificates.jsonl.gz")
        journal_sha = None
        if journal.is_file():
            journal_sha = hashlib.sha256(journal.read_bytes()).hexdigest()
            with gzip.open(journal, "rt", encoding="utf-8") as stream:
                journal_count = 0
                for journal_count, line in enumerate(stream, 1):
                    _require(journal_count <= len(holds) and digest(json.loads(line)) == digest(holds[journal_count - 1]),
                             "the adjacent proof journal differs from the execution record")
            _require(journal_count == len(holds), "the adjacent proof journal is incomplete")
        return cls._create(approach, record, execution, runner, clocks, states, ids,
                           np.asarray(command_rows, dtype=float), source_sha, terminal_sha,
                           journal_sha, approach_path, terminal_result_path, flow_count, duration, discrepancy)

    @classmethod
    def _create(cls, approach, record, execution, runner, clocks, states, ids, commands,
                source_sha, terminal_sha, journal_sha, approach_path, terminal_path, flow_count, duration, discrepancy):
        references = runner.references_at(clocks[-1])
        endpoints, centers, omega, phi, bounds = {}, {}, {}, {}, {}
        for key, state, reference, controller in zip(ids, states[-1], references, runner.fleet.controllers):
            b = controller.certificate.bounds
            evaluation = controller.result.derived["effective_reference_state_error_abs"]
            endpoints[key], centers[key] = _readonly(state), _readonly(reference.state)
            padded_omega = tuple(o + e for o, e in zip(b.omega, evaluation[:3]))
            padded_phi = tuple(p + ev + lam * ep for p, ev, lam, ep in
                               zip(b.phi, evaluation[3:], b.lambda_gain, evaluation[:3]))
            omega[key], phi[key] = _outward(padded_omega), _outward(padded_phi)
            bounds[key] = dict(reference_id=b.reference_id, true_reference_omega=list(map(str, b.omega)),
                true_reference_phi=list(map(str, b.phi)), reference_state_error_abs=list(map(str, evaluation)),
                capture_center_omega=list(map(str, padded_omega)), capture_center_phi=list(map(str, padded_phi)),
                lambda_gain=list(map(str, b.lambda_gain)), bandwidth=list(map(str, b.bandwidth)),
                maximum_hold=str(b.maximum_hold), position_tolerance=str(b.position_tolerance),
                whole_tube_position_radius_squared=str(sum(v * v for v in b.omega)),
                input_limits=b.input_limits.as_dict(), body_radius=str(b.geometry.body_radius))
        last_kinematics = runner.target.recorded_kinematics([float(clocks[-1])])
        truth_view = SimpleNamespace(target=copy.deepcopy(record["target"]), truth=copy.deepcopy(record["truth"]),
            anom_history=np.array([float(clocks[-1])]), metadata={"target_kinematics": last_kinematics})
        capture_truth = CaptureModel(target=TargetPhysicalProperties(inertia_kg_m2=runner.target.inertia)).recorded_truth(truth_view)
        metadata = copy.deepcopy(record["metadata"])
        original_kinematics = metadata["target_kinematics"]
        terminal_kinematics = runner.target.recorded_kinematics(np.asarray([float(value) for value in clocks[1:]]))
        _require(set(original_kinematics) == set(terminal_kinematics), "target kinematics fields changed across stages")
        combined_kinematics = {name: np.concatenate((np.asarray(original_kinematics[name]),
                                                     np.asarray(terminal_kinematics[name])), axis=0).tolist()
                               for name in original_kinematics}
        metadata["target_kinematics"] = combined_kinematics
        metadata["elapsed_time_seconds"] = (np.asarray(combined_kinematics["physical_time"])
                                                - combined_kinematics["physical_time"][0]).tolist()
        metadata["sample_phase_history"] = list(record["sample_phase_history"]) + ["physical_terminal"] * len(commands)
        metadata["controls_semantics"] = ("Source approach reported applied-input means followed by exact held terminal "
            "command centers; the proof record retains actuator intervals. No MPC solve or adaptation history is extended.")
        source_guard = record["metadata"]["continuous_execution_certificate"]
        source_inputs = IntervalBox(source_guard["input_lower"], source_guard["input_upper"])
        metadata["input_bounds"] = [dict(lower=_float_outward(lo, False), upper=_float_outward(hi, True))
                                    for lo, hi in zip(source_inputs.lower, source_inputs.upper)]
        metadata["continuous_execution_certificate"] = dict(
            method="hybrid-source-and-terminal-accounting-clock-v1",
            start_anomaly=source_guard["start_anomaly"], exact_end_anomaly=str(clocks[-1]),
            body_radius_m=float(min(builder.body_radius for builder in runner.builders)),
            input_lower=list(map(str, source_inputs.lower)), input_upper=list(map(str, source_inputs.upper)),
            source_approach_exact_end_anomaly=str(clocks[0]), source_approach_result_sha256=source_sha,
            terminal_execution_sha256=terminal_sha,
            scope="Composite accounting clock only; original approach and terminal continuous proofs remain separate records")
        trajectory = PhysicalTerminalTrajectory(
            anom_history=_readonly(list(record["anom_history"]) + [float(value) for value in clocks[1:]]),
            actual_trajectories=MappingProxyType({key: _readonly(np.concatenate((np.asarray(
                record["actual_trajectories"][key]), states[1:, index]), axis=0)) for index, key in enumerate(ids)}),
            controls=MappingProxyType({key: _readonly(np.concatenate((np.asarray(record["controls"][key]),
                commands[:, index]), axis=0)) for index, key in enumerate(ids)}),
            target=copy.deepcopy(record["target"]), truth=copy.deepcopy(record["truth"]), metadata=metadata)
        summary = dict(method=HANDOFF_METHOD, source_approach=dict(path=str(approach_path.resolve()),
            raw_sha256=source_sha, semantic_sha256=digest(record), samples=len(record["anom_history"]),
            exact_end_anomaly=str(clocks[0]), reporter_end_anomaly=str(_exact(record["anom_history"][-1])),
            reporter_event_discrepancy=str(discrepancy)),
            terminal_execution=dict(path=str(terminal_path.resolve()), raw_sha256=terminal_sha,
                producer_source_snapshot_sha256=execution["source_context"]["caller"].get("source_snapshot_sha256"),
                journal_sha256=journal_sha, hold_records_sha256=digest(execution["hold_records"]),
                exact_start_anomaly=str(clocks[0]), exact_end_anomaly=str(clocks[-1]),
                physical_duration_seconds=float(duration), holds=len(commands), rederived_global_certificates=len(ids),
                reproved_continuous_flows=flow_count, reproved_swept_fleet_holds=len(commands)),
            endpoint=dict(states={key: value.tolist() for key, value in endpoints.items()},
                reference_centers={key: value.tolist() for key, value in centers.items()},
                omega={key: value.tolist() for key, value in omega.items()},
                phi={key: value.tolist() for key, value in phi.items()},
                lambda_gain=list(map(float, runner.builders[0].lambda_gain)),
                capture_truth=plain(vars(capture_truth))), controller_bounds=bounds,
            original_approach_preserved=True, reset_performed=False, fabricated_mpc_or_smid_history=False,
            scope="Finite conditional terminal continuation and numerical physical capture endpoint; sensor/reference/initial-spin contracts remain explicit assumptions",
            reference_center_semantics="Evaluated physical reference, not an optimized nominal center; capture Omega/Phi include outward reference-evaluation reserve",
            trajectory_semantics="Actual translation/target samples for attitude replay and conservative propellant accounting only")
        self = object.__new__(cls)
        for name, value in dict(trajectory=trajectory, capture_truth=capture_truth,
                states=MappingProxyType(endpoints), nominal_centers=MappingProxyType(centers),
                omega=MappingProxyType(omega), phi=MappingProxyType(phi),
                lambda_gain=_readonly(runner.builders[0].lambda_gain), exact_end_anomaly=clocks[-1],
                _approach_digest=digest(record), _summary=summary).items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_integrity_digest", self._current_integrity())
        self.verify_approach(approach)
        return self

    def _current_integrity(self):
        # Mappings used by consumers include nested metadata.  Detect any
        # accidental mutation before a downstream physical boundary is used.
        trajectory = dict(vars(self.trajectory), actual_trajectories=dict(self.trajectory.actual_trajectories),
                          controls=dict(self.trajectory.controls))
        return digest(dict(summary=self._summary, trajectory=trajectory,
            capture_truth=vars(self.capture_truth), states=dict(self.states),
            centers=dict(self.nominal_centers), omega=dict(self.omega), phi=dict(self.phi),
            lambda_gain=self.lambda_gain, exact_end=str(self.exact_end_anomaly)))

    def verify_approach(self, approach):
        """Reject a changed source or a modified already-verified handoff."""
        _require(isinstance(approach, ADTMPCMissionResult) and digest(plain(approach.to_dict())) == self._approach_digest,
                 "the supplied approach differs from the exact originating history")
        _require(self._current_integrity() == self._integrity_digest, "the verified handoff was modified")
        return True

    def as_dict(self):
        """Compact verified endpoint/provenance; raw proofs stay in their source file."""
        _require(self._current_integrity() == self._integrity_digest, "the verified handoff was modified")
        return copy.deepcopy(self._summary)
