"""Optional source-bound propellant accounting from complete admitted holds.

This replays retained interval proofs, never the controlled trajectory. Every
delivered actuator value must be enclosed by the saved held-input box. Process
disturbances are external forcing, not an unrecorded propellant allowance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import gzip
import hashlib
import importlib
import json
from math import isqrt
from pathlib import Path
import re
import zlib

import numpy as np

from orbexa.control import reachability
from orbexa.control.reachability import (
    HeldInputRequest, IntervalBox, OrbitalModelBounds, ReachabilityConfig,
    ValidatedFlowEncloser, RotatingCylinderBounds, SweptCollisionGuard,
)
from orbexa.control.physical_terminal import bound_orbital_clock
from orbexa.simulation.assembly import STANDARD_GRAVITY
from orbexa.simulation.complete_mission import ApproachPropellantBudget, _outward_float
from orbexa.simulation.mission_checkpoint import digest, plain
from orbexa.simulation.mission_result import ADTMPCMissionResult


HELD_PROPELLANT_METHOD = "source-bound-admitted-held-input-propellant-v1"
_PROOF_SOURCE = Path(reachability.__file__).resolve()
_LOADED_ACCOUNTANT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_PROOF_MODULES = {name:importlib.import_module("orbexa."+name[:-3].replace("/", ".")) for name in (
    "control/reachability.py", "control/physical_terminal.py", "control/rigid_body_bounds.py",
    "control/composite_geometry.py", "core/spacecraft_geometry.py")}
_LOADED_PROOF_SHA256 = {name:hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                        for name, module in _PROOF_MODULES.items()}


def _require(condition, reason):
    if not condition:
        raise ValueError("Held-input propellant accounting: " + reason)


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _hash(value):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             "explicit expected SHA-256 values are required")
    return value


def _box(value, dimension):
    _require(isinstance(value, dict), "a complete exact interval box is required")
    result = IntervalBox(value["exact_lower"], value["exact_upper"])
    _require(result.dimension == dimension and digest(result.as_dict()) == digest(value),
             "exact and exported interval bounds disagree")
    return result


def _subset(inner, outer):
    return inner.dimension == outer.dimension and all(
        a <= b <= c <= d for a, b, c, d in zip(outer.lower, inner.lower, inner.upper, outer.upper))


def _sqrt_upper(value, bits=128):
    """A dyadic upper square root, proved by an integer squared comparison."""
    _require(value >= 0, "a squared input norm must be nonnegative")
    scale = 1 << bits
    numerator = isqrt(value.numerator * scale * scale // value.denominator)
    if numerator * numerator * value.denominator < value.numerator * scale * scale:
        numerator += 1
    result = Fraction(numerator, scale)
    _require(result * result >= value, "outward square root failed")
    return result


def _dyadic_upper(value, bits=128):
    """Round a nonnegative term outward before exact bounded-size summation."""
    _require(value >= 0, "an impulse term must be nonnegative")
    scale = 1 << bits
    return Fraction(-(-value.numerator * scale // value.denominator), scale)


def _target(value, rotation_cache):
    _require(isinstance(value, dict), "complete occupied target bounds are required")
    if value.get("shape") == "composite_spacecraft":
        from orbexa.control.composite_geometry import RotatingCompositeBounds, CompositeSweptCollisionGuard
        from orbexa.core.spacecraft_geometry import SpacecraftGeometry
        geometry = SpacecraftGeometry.from_dict(value["geometry"])
        _require(geometry.fingerprint == value["geometry_fingerprint"], "occupied geometry fingerprint changed")
        names = ("orientation_matrix_at_start", "orientation_matrix_error", "angular_rate_per_anomaly",
                 "center_speed_per_anomaly", "frame", "scope")
        arguments = {name: value[name] for name in names}
        arguments["center_at_start"] = _box(value["center_at_start"], 3)
        # The optional rate proof remains part of the exact serialized identity.
        if "rotation_rate_bounds" in value:
            saved = value["rotation_rate_bounds"]
            identity = digest(saved)
            if identity not in rotation_cache:
                from orbexa.control.rigid_body_bounds import bound_rigid_body_rotation
                clock = saved["clock"]
                rebuilt = bound_rigid_body_rotation(*clock["anomaly"], OrbitalModelBounds(**clock["parameters"]),
                    inertia=saved["inertia"], initial_spin=_box(saved["initial_spin"], 3),
                    gravity_gradient=saved["gravity_gradient"], precision_bits=saved["precision_bits"])
                _require(digest(rebuilt.as_dict()) == identity, "analytical target rate proof did not reproduce")
                rotation_cache[identity] = rebuilt
            arguments["rotation_rate_bounds"] = rotation_cache[identity]
        return RotatingCompositeBounds(geometry=geometry, **arguments), CompositeSweptCollisionGuard()
    _require("shape" not in value, "unsupported occupied geometry")
    arguments = dict(value)
    for name in ("orientation_at_start", "center_at_start"):
        arguments[name] = _box(arguments[name], 3)
    return RotatingCylinderBounds(**arguments), SweptCollisionGuard()


def _verified_terminal_holds(handoff, approach, source_sha256, start_anomaly, chaser_ids):
    """Read actuator boxes only after the existing full terminal reproof gate."""
    from orbexa.simulation.terminal_handoff import PhysicalTerminalHandoff
    _require(isinstance(handoff, PhysicalTerminalHandoff), "a verified physical terminal handoff is required")
    handoff.verify_approach(approach)
    summary = handoff.as_dict()
    _require(summary["source_approach"]["raw_sha256"] == source_sha256,
             "terminal handoff names a different original approach")
    terminal = summary["terminal_execution"]
    path = Path(terminal["path"])
    raw = path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == terminal["raw_sha256"], "verified terminal raw source changed")
    record = json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)
    holds = record["hold_records"]
    _require(digest(holds) == terminal["hold_records_sha256"] and len(holds) == terminal["holds"]
             and Fraction(terminal["exact_start_anomaly"]) == start_anomaly,
             "verified terminal hold identity changed")
    ids, q, result = tuple(chaser_ids), start_anomaly, []
    _require(tuple(approach.actual_trajectories) == ids
             == tuple(handoff.trajectory.actual_trajectories), "terminal fleet or ordering changed")
    for index, hold in enumerate(holds):
        q0, q1 = Fraction(hold["start_anomaly"]), Fraction(hold["end_anomaly"])
        width = Fraction(hold["hold_width"])
        _require(hold.get("executed") is True and hold.get("decision") == "EXECUTED_PROVEN_HOLD"
                 and hold["stage"] == index and q0 == q and q1 == q0+width and width > 0
                 and len(hold["flows"]) == len(ids), "terminal tail has a missing or unexecuted shared hold")
        boxes = {}
        for key, flow in zip(ids, hold["flows"]):
            request = flow["request"]
            _require(tuple(map(Fraction, request["start_anomaly"])) == (q0, q0)
                     and tuple(map(Fraction, request["hold_width"])) == (width, width),
                     "terminal fleet request clocks disagree")
            boxes[key] = _box(request["held_input"], 3)
        result.append((q0, q1, boxes))
        q = q1
    _require(result and q == handoff.exact_end_anomaly == Fraction(terminal["exact_end_anomaly"]),
             "terminal final clock changed")
    handoff.verify_approach(approach)
    return summary, tuple(result)


@dataclass(frozen=True)
class HeldPropellantBudget(ApproachPropellantBudget):
    """Outward allowances with exact sums and authenticated source identities."""

    held_accounting: dict = field(default_factory=dict)

    def to_dict(self):
        result = super().to_dict()
        result.update(method=HELD_PROPELLANT_METHOD,
            scope="Conservative per-hold translation debit for the authenticated complete approach and optional verified terminal tail; arbitrary proper bus attitudes and initial wet mass; not measured fuel",
            held_input_accounting=plain(self.held_accounting))
        return result


class HeldInputPropellantAccountant:
    """Inject into CompleteMissionRunner after authenticating immutable inputs.

    Construct with from_files. Expected hashes are caller-selected provenance,
    not inferred evidence. Every producer Python file must match the original
    checkpoint source map. Original Picard and swept geometry witnesses are
    independently rebuilt before any smaller allowance is returned. Only the
    original default Picard policy (with each saved precision) is supported.
    Complete ordered journals are required, including every restored prefix.
    """

    def __init__(self, *args, **kwargs):
        raise TypeError("Construct a source-bound accountant with from_files")

    @classmethod
    def from_files(cls, approach_path, journal_paths, *, expected_approach_sha256,
                   expected_journal_sha256, producer_package_root):
        try:
            approach_path = Path(approach_path).resolve()
            paths = tuple(Path(path).resolve() for path in journal_paths)
            hashes = tuple(map(_hash, expected_journal_sha256))
            _require(paths and len(paths) == len(hashes) and len(set(paths)) == len(paths),
                     "a complete ordered set of uniquely named bound journals is required")
            _require(_sha(approach_path) == _hash(expected_approach_sha256), "raw approach identity changed")
            payload = json.loads(approach_path.read_bytes())
            _require(isinstance(payload, dict), "a typed original approach dictionary is required")
            approach = ADTMPCMissionResult.from_dict(payload.get("result", payload))
            record = plain(approach.to_dict())
            producer = Path(producer_package_root).resolve()
            sources = record["metadata"]["checkpoint_context"]["source_fingerprints"]
            _require(isinstance(sources, dict) and sources, "producer checkpoint source map is missing")
            _require(set(sources) == {p.relative_to(producer).as_posix() for p in producer.rglob("*.py")},
                     "the complete producer package source set must match its checkpoint")
            for name, expected in sources.items():
                path = (producer / name).resolve()
                _require(path.is_relative_to(producer) and path.suffix == ".py"
                         and _sha(path) == _hash(expected), "producer source bytes changed: " + name)
            _require(sources.get("control/reachability.py") == reachability._LOADED_SOURCE_SHA256
                     == _sha(_PROOF_SOURCE), "loaded orbital proof source differs from the producer")
            for path, expected in zip(paths, hashes):
                _require(_sha(path) == expected, "guard journal identity changed")
            binding = dict(approach_path=str(approach_path), approach_sha256=expected_approach_sha256,
                approach_semantic_sha256=digest(record), journals=[dict(path=str(p), sha256=h) for p, h in zip(paths, hashes)],
                producer_package_root=str(producer), producer_source_sha256=dict(sources))
            result = object.__new__(cls)
            result._binding, result._integrity = binding, digest(binding)
            return result
        except (OSError, KeyError, TypeError, ArithmeticError, EOFError, zlib.error) as error:
            raise ValueError("Held-input propellant accounting: malformed or unavailable bound source: " + str(error)) from error

    def _authenticate(self, approach):
        _require(digest(self._binding) == self._integrity, "accountant binding was modified")
        _require(isinstance(approach, ADTMPCMissionResult)
                 and digest(plain(approach.to_dict())) == self._binding["approach_semantic_sha256"],
                 "the supplied approach differs from its authenticated source")
        _require(_sha(self._binding["approach_path"]) == self._binding["approach_sha256"], "raw approach changed")
        _require(_sha(Path(__file__)) == _LOADED_ACCOUNTANT_SHA256
                 and _sha(_PROOF_SOURCE) == reachability._LOADED_SOURCE_SHA256
                 == self._binding["producer_source_sha256"]["control/reachability.py"],
                 "loaded accounting/proof source changed")
        producer = Path(self._binding["producer_package_root"])
        _require(set(self._binding["producer_source_sha256"])
                 == {p.relative_to(producer).as_posix() for p in producer.rglob("*.py")},
                 "producer package source set changed")
        for name, expected in self._binding["producer_source_sha256"].items():
            _require(_sha(producer/name) == expected, "producer source changed: " + name)
        for name, module in _PROOF_MODULES.items():
            _require(_sha(module.__file__) == _LOADED_PROOF_SHA256[name]
                     == self._binding["producer_source_sha256"].get(name),
                     "loaded proof dependency differs from the producer: " + name)
        for row in self._binding["journals"]:
            _require(_sha(row["path"]) == row["sha256"], "guard journal changed")

    def translation_budget(self, approach, *, options, attitude_tracker, terminal_handoff=None):
        try:
            return self._translation_budget(approach, options, attitude_tracker, terminal_handoff)
        except (OSError, KeyError, TypeError, IndexError, ArithmeticError, EOFError, zlib.error) as error:
            raise ValueError("Held-input propellant accounting: malformed or unavailable proof: " + str(error)) from error

    def _translation_budget(self, approach, options, tracker, terminal_handoff):
        self._authenticate(approach)
        _require(approach.success and approach.metadata.get("mission_complete") is True,
                 "the originating approach did not complete")
        guard = approach.metadata["continuous_execution_certificate"]
        _require(guard["method"] == "arb-picard-continuous-v1", "unsupported original execution guard")
        ids = tuple(approach.actual_trajectories)
        _require(ids and len(set(ids)) == len(ids), "the exact nonempty fleet is required")
        count = guard["admitted_holds"]
        _require(type(count) is int and count > 0 and type(guard["rejected_holds"]) is int
                 and guard["rejected_holds"] == 0, "only complete admitted-only journals are supported")
        model = OrbitalModelBounds(**guard["model_bounds"])
        truth = approach.truth
        for name in ("eccentricity", "alpha", "beta"):
            value = Fraction(truth[name])
            _require(getattr(model, name)[0] <= value <= getattr(model, name)[1], "physical truth leaves retained model")
        _require(model.mu == Fraction(truth["mu"]) and model.semi_major_axis == Fraction(truth["semi_major_axis"]),
                 "physical orbital constants changed")
        limits = IntervalBox(guard["input_lower"], guard["input_upper"])
        limit = Fraction(options.input_component_bound)
        _require(limits.dimension == 3 and all(-limit <= lo <= hi <= limit for lo, hi in zip(limits.lower, limits.upper)),
                 "declared input bound does not cover retained actuator limits")
        disturbance = IntervalBox.from_center_radius((0, 0, 0), guard["acceleration_radius"])
        start = q = Fraction(guard["start_anomaly"])
        end = Fraction(guard["exact_end_anomaly"])
        anomalies = np.asarray(approach.anom_history, dtype=float)
        _require(anomalies.ndim == 1 and len(anomalies) >= 2 and np.all(np.isfinite(anomalies))
                 and np.all(np.diff(anomalies) > 0) and abs(float(start)-anomalies[0]) <= 1e-10
                 and abs(float(end)-anomalies[-1]) <= 1e-10 and end > start,
                 "approach and exact committed clocks disagree")
        states = {key: np.asarray(approach.actual_trajectories[key], dtype=float) for key in ids}
        _require(all(value.shape == (len(anomalies), 6) and np.all(np.isfinite(value)) for value in states.values()),
                 "actual fleet history is incomplete")
        properties = {key: tracker.chasers.get(key, tracker.default_chaser) for key in ids}
        impulse, peak = {key: Fraction(0) for key in ids}, {key: Fraction(0) for key in ids}
        ledger = hashlib.sha256()
        previous, observed, rotation_cache, last_context = {}, 0, {}, None
        sample_index, last_width = 1, None

        def debit_hold(q0, q1, boxes):
            _require(q1 > q0 and set(boxes) == set(ids), "invalid or incomplete shared hold")
            clock = bound_orbital_clock(q0, q1, model)
            row = dict(start=str(q0), end=str(q1), clock=clock.as_dict(), inputs={}, force_upper={})
            for key in ids:
                box = boxes[key]
                _require(_subset(box, limits), "actual held input leaves the retained actuator limits")
                squared = 3 * sum(max(abs(lo), abs(hi))**2 for lo, hi in zip(box.lower, box.upper))
                norm = _sqrt_upper(squared)
                force = Fraction(properties[key].mass_kg) * clock.anomaly_rate.upper[0]**2 * norm
                _require(force <= Fraction(options.translation_force_n), "held force allowance exceeds installed nozzle limit")
                # Different clock minima have unrelated rational numerators.
                # Outward dyadic terms avoid an enormous product denominator;
                # their sum is still exact and conservative over every hold.
                term = _dyadic_upper(force * clock.physical_duration_upper)
                impulse[key] += term
                peak[key] = max(peak[key], force)
                row["inputs"][key] = box.as_dict()
                row["force_upper"][key] = str(force)
                row.setdefault("impulse_upper", {})[key] = str(term)
            ledger.update(json.dumps(plain(row), sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
            ledger.update(b"\n")

        for source in self._binding["journals"]:
            path = Path(source["path"])
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rt", encoding="utf-8") as stream:
                for line in stream:
                    _require(bool(line.strip()), "blank or missing hold record")
                    row = json.loads(line)
                    _require(isinstance(row, dict), "a complete hold dictionary is required")
                    _require(row.get("admitted") is True and tuple(row["chaser_ids"]) == ids
                             and set(row["flows"]) == set(ids), "refused or changed-fleet journal hold")
                    _require(isinstance(row["exact_start_anomaly"], str) and isinstance(row["exact_end_anomaly"], str)
                             and not isinstance(row["width"], bool) and not isinstance(row["anomaly"], bool),
                             "explicit exact clock strings and numeric reporter labels are required")
                    q0, q1 = Fraction(row["exact_start_anomaly"]), Fraction(row["exact_end_anomaly"])
                    width = Fraction(row["width"])
                    _require(q0 == q and q1 == q0 + width and width > 0 and q1 <= end
                             and abs(float(q0)-float(row["anomaly"])) <= 1e-10,
                             "journal hold clock is not the exact complete contiguous shared clock")
                    rebuilt, boxes = [], {}
                    for key in ids:
                        saved = row["flows"][key]
                        request = saved["request"]
                        initial = _box(request["initial_state"], 6)
                        boxes[key] = _box(request["held_input"], 3)
                        actual_disturbance = _box(request["acceleration_disturbance"], 3)
                        _require(actual_disturbance == disturbance
                                 and tuple(map(Fraction, request["start_anomaly"])) == (q0, q0)
                                 and tuple(map(Fraction, request["hold_width"])) == (width, width),
                                 "retained request clock or disturbance changed")
                        if previous:
                            _require(_subset(initial, previous[key]), "request does not refine the preceding endpoint")
                        else:
                            expected = IntervalBox.from_center_radius(states[key][0], guard["measurement_radius"])
                            _require(initial == expected, "initial request differs from the source release measurement box")
                        flow = ValidatedFlowEncloser(ReachabilityConfig(precision_bits=saved["precision_bits"])).enclose_orbital(
                            HeldInputRequest(initial, boxes[key], (q0, q0), (width, width), actual_disturbance), model)
                        _require(flow.proven and digest(flow.as_dict()) == digest(saved),
                                 "original complete Picard witness did not reproduce")
                        rebuilt.append(flow)
                    target, collision = _target(row["collision"]["target_bounds"], rotation_cache)
                    from orbexa.simulation.certified_execution import CertifiedExecutionGuard
                    context = plain(CertifiedExecutionGuard._target_context(target))
                    declared_context = dict(row["target_bound_context"])
                    declared_context["orientation_matrix_error"] = str(Fraction(declared_context["orientation_matrix_error"])
                        + target.angular_rate_per_anomaly*abs(Fraction(float(q0))-q0))
                    _require(digest(context) == digest(declared_context)
                             and CertifiedExecutionGuard._shape_identity(context)
                             == CertifiedExecutionGuard._shape_identity(guard["target_bound_context"]),
                             "whole-hold occupied target context changed")
                    proof = collision.check(rebuilt, [guard["body_radius_m"]]*len(ids), target=target,
                        minimum_pair_center_distance=guard["minimum_pair_separation_m"])
                    _require(proof.admissible and digest(proof.as_dict()) == digest(row["collision"]),
                             "original complete swept admission did not reproduce")
                    debit_hold(q0, q1, boxes)
                    previous = {key: flow.endpoint for key, flow in zip(ids, rebuilt)}
                    if sample_index < len(anomalies) and abs(float(q1)-anomalies[sample_index]) <= 1e-10:
                        _require(all(_subset(IntervalBox.point(states[key][sample_index]), previous[key]) for key in ids),
                                 "journal endpoint does not enclose the matching actual source sample")
                        sample_index += 1
                    elif sample_index < len(anomalies):
                        _require(float(q1) < anomalies[sample_index]-1e-10,
                                 "journal skipped an actual source sample event")
                    last_context = row["target_bound_context"]
                    last_width = width
                    q, observed = q1, observed+1
        _require(observed == count and q == end and sample_index == len(anomalies)
                 and set(guard["endpoint_boxes"]) == set(ids)
                 and Fraction(guard["last_hold_width"]) == last_width
                 and abs(float(guard["last_committed_anomaly"])-anomalies[-1]) <= 1e-10,
                 "journal hold count, exact endpoint or final fleet is incomplete")
        _require(digest(last_context) == digest(guard["target_bound_context"]), "final target-bound context changed")
        for key in ids:
            _require(previous[key] == _box(guard["endpoint_boxes"][key], 6)
                     and _subset(IntervalBox.point(states[key][-1]), previous[key]),
                     "journal final enclosure differs from the source endpoint")
        terminal_summary, terminal_count = None, 0
        if terminal_handoff is not None:
            terminal_summary, holds = _verified_terminal_holds(terminal_handoff, approach,
                self._binding["approach_sha256"], q, ids)
            for q0, q1, boxes in holds:
                debit_hold(q0, q1, boxes)
                q, terminal_count = q1, terminal_count+1
        self._authenticate(approach)
        denominator = Fraction(STANDARD_GRAVITY)*Fraction(options.specific_impulse_s)
        exact_fuel = {key:value/denominator for key, value in impulse.items()}
        binding = dict(self._binding, approach_holds=observed, terminal_holds=terminal_count,
            exact_start_anomaly=str(start), exact_end_anomaly=str(q), squared_norm_root_precision_bits=128,
            impulse_term_outward_precision_bits=128,
            exact_impulse_upper_n_s={key:str(value) for key,value in impulse.items()},
            exact_fuel_upper_kg={key:str(value) for key,value in exact_fuel.items()},
            exact_peak_force_l1_upper_n={key:str(value) for key,value in peak.items()},
            reproduced_approach_flows=observed*len(ids), reproduced_approach_swept_holds=observed,
            hold_terms_sha256=ledger.hexdigest(), terminal_handoff=terminal_summary,
            formula="F1 <= m_wet*nu_max^2*sqrt(3*sum(max(abs(u_lower),abs(u_upper))^2)); fuel <= sum(F1*duration_upper)/(g0*Isp)",
            assumptions=["Every actual delivered actuator command lies in its saved held-input box; no unrecorded actuator consumption",
                "The retained physical prior, clock, process and geometry/sensor assumptions remain valid",
                "Translation nozzles are continuously throttleable, exact and unobstructed; any proper bus attitude is allowed",
                "Initial wet mass bounds every later mass; the original attitude inertia and fuel gates remain mandatory"])
        return HeldPropellantBudget(bound_orbital_clock(start, q, model).as_dict(),
            {key:_outward_float(value) for key,value in peak.items()},
            {key:_outward_float(value) for key,value in exact_fuel.items()},
            float(options.input_component_bound), float(options.specific_impulse_s),
            float(options.translation_force_n), binding)
