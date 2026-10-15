"""Offline execution of a certified physical terminal stage from its real entry.

The caller supplies the actual approach endpoint and its measurement enclosure.
Refusal preserves that endpoint; no projection or synthetic reset is performed.
Each admitted held input is linked to its continuous Arb flow and whole-fleet
swept geometry certificate. This simulator does not provide a hardware scheduler.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import json
from typing import Any, Callable, Sequence

from orbexa.control.physical_terminal import (
    PhysicalTerminalBoundsBuilder, PhysicalTerminalFleet, _encode, _vector,
)
from orbexa.control.reachability import (
    HeldInputRequest, IntervalBox, OrbitalModelBounds, ReachabilityConfig,
    RotatingCylinderBounds, SweptCollisionGuard, ValidatedFlowEncloser, _exact,
)
from orbexa.control.terminal_tracking import (
    TerminalFeedforwardPolicy, TerminalHoldReferences, TerminalReferenceSample,
)
from orbexa.control.composite_geometry import CompositeSweptCollisionGuard, RotatingCompositeBounds


EXECUTION_METHOD = "physical-terminal-held-execution-v1"
MIDPOINT_EXECUTION_METHOD = "physical-terminal-midpoint-execution-v1"
MIDPOINT_EXECUTION_SCHEMA = "orbexa.physical_terminal_execution.midpoint.v1"


def _digest(value):
    return hashlib.sha256(json.dumps(_encode(value), sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def terminal_execution_policy(record):
    """Resolve only supported policy/schema pairs, including unchanged legacy v1."""
    if not isinstance(record, dict):
        raise ValueError("A typed physical terminal execution record is required")
    if record.get("method") == EXECUTION_METHOD:
        if "held_policy" in record or "schema" in record:
            raise ValueError("Legacy start execution cannot claim a different held policy/schema")
        return TerminalFeedforwardPolicy.START_V1
    if (record.get("method") == MIDPOINT_EXECUTION_METHOD
            and record.get("schema") == MIDPOINT_EXECUTION_SCHEMA
            and record.get("held_policy") == TerminalFeedforwardPolicy.MIDPOINT_V1.value):
        return TerminalFeedforwardPolicy.MIDPOINT_V1
    raise ValueError("Unsupported or inconsistent physical terminal execution method/policy/schema")


def terminal_hold_linkage(event):
    """Canonical command/reference/flow/geometry binding used by writer and reader."""
    policy = terminal_execution_policy(event)
    value = {key: event[key] for key in ("start_anomaly", "hold_width", "commands", "flow_fingerprints", "geometry")}
    if policy is TerminalFeedforwardPolicy.MIDPOINT_V1:
        value.update({key: event[key] for key in ("schema", "held_policy", "exact_midpoint_anomaly",
            "exact_end_anomaly", "target_context_sha256", "reference_samples", "evaluation_error_contracts",
            "source_context_sha256")})
    return value


def terminal_source_lineage(approach_record):
    """Hash the literal approach's arrival/region/transition facts for a new stage.

    These hashes preserve source identity; they grant no admission or motion.
    The handoff independently compares them with its exact raw approach file.
    """
    from orbexa.simulation.mission_checkpoint import plain
    record = plain(approach_record)
    metadata = record["metadata"]
    policy = metadata.get("scenario", {}).get("arrival_policy")
    state = metadata.get("arrival_state")
    region = None if policy is None else policy.get("required_terminal_region")
    return dict(arrival_policy_sha256=None if policy is None else _digest(policy),
        required_region_sha256=None if region is None else _digest(region),
        arrival_state_sha256=None if state is None else _digest(state),
        arrival_activation_sha256=None if metadata.get("arrival_activation") is None else _digest(metadata["arrival_activation"]),
        transition_history_sha256=_digest(metadata.get("policy_transition_history", [])),
        transition_identity=None if state is None else state.get("transition_identity"))


class _TerminalSourceFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class TerminalObservation:
    """An independently valid sensor observation at one actual sample clock."""

    measurements: Sequence[Sequence[Any]]
    state_enclosures: Sequence[IntervalBox]

    def __post_init__(self):
        measurements = tuple(_vector(value, 6, "Terminal measurement") for value in self.measurements)
        boxes = tuple(self.state_enclosures)
        if not measurements or len(measurements) != len(boxes) or any(box.dimension != 6 for box in boxes):
            raise ValueError("Supply one finite six-state measurement/enclosure per terminal member")
        object.__setattr__(self, "measurements", measurements)
        object.__setattr__(self, "state_enclosures", boxes)

    def as_dict(self):
        return _encode(vars(self))


@dataclass(frozen=True)
class PhysicalTerminalExecutionResult:
    status: str
    reason: str
    anomaly_history: tuple[Fraction, ...]
    actual_state_history: tuple[tuple[tuple[float, ...], ...], ...]
    initial_observation: TerminalObservation
    final_observation: TerminalObservation
    final_observation_anomaly: Fraction
    admission: dict
    bound_certificates: tuple[dict, ...]
    hold_records: tuple[dict, ...]
    physical_duration_seconds: float
    source_context: dict = field(default_factory=dict)
    method: str = EXECUTION_METHOD
    held_policy: str | None = None
    schema: str | None = None

    @property
    def success(self):
        return self.status == "COMPLETED_FINITE_WINDOW"

    @property
    def final_states(self):
        return self.actual_state_history[-1]

    def as_dict(self):
        record = dict(vars(self))
        for key in ("held_policy", "schema"):
            if record[key] is None:
                record.pop(key)
        return _encode(record) | {"success": self.success, "reset_performed": False}


class PhysicalTerminalRunner:
    """Numerical held plant execution with independent per-hold proof admission.

    The physical truth is used only by numerical propagation. The controller
    and each Arb flow retain the explicitly supplied parameter prior. A callable
    observation source must return valid observations; its sensor provenance
    remains an explicit experiment assumption. The initial observation is
    supplied separately so a saved approach endpoint is never replaced.
    orientation_matrix_error covers matrix evaluation/estimation error plus
    the derived rotation-rate bound times the complete query-clock error,
    including conversion of the exact sample anomaly to binary64.
    """

    def __init__(self, *, builders: Sequence[PhysicalTerminalBoundsBuilder], target,
                 truth: OrbitalModelBounds,
                 observation_source: Callable[[Fraction, tuple, tuple[IntervalBox, ...]], TerminalObservation],
                 orientation_matrix_error,
                 minimum_pair_center_distance="0.35",
                 reachability_config: ReachabilityConfig | None = None,
                 record_sink: Callable[[dict], None] | None = None,
                 acceleration_source: Callable[[float, Any], Any] | None = None,
                 source_context: dict | None = None):
        self.builders = tuple(builders)
        self.fleet = PhysicalTerminalFleet(self.builders)
        self.policy = self.fleet.controllers[0].certificate.policy
        self.target, self.truth = target, truth
        self.observation_source = observation_source
        self.orientation_matrix_error = _exact(orientation_matrix_error)
        self.minimum_pair_center_distance = _exact(minimum_pair_center_distance)
        if self.orientation_matrix_error < 0 or self.minimum_pair_center_distance < 0 or not callable(observation_source):
            raise ValueError("Supply nonnegative geometry bounds and an explicit observation source")
        if record_sink is not None and not callable(record_sink):
            raise ValueError("record_sink must be callable")
        if acceleration_source is not None and not callable(acceleration_source):
            raise ValueError("acceleration_source must be callable")
        for name in ("eccentricity", "alpha", "beta", "theta0"):
            if getattr(truth, name)[0] != getattr(truth, name)[1]:
                raise ValueError("Numerical execution needs a fixed physical truth point")
            point = getattr(truth, name)[0]
            for builder in self.builders:
                lo, hi = getattr(builder.plant_parameters, name)
                if not lo <= point <= hi:
                    raise ValueError("Execution truth lies outside the retained controller/flow prior")
        first = self.builders[0]
        if truth.mu != first.plant_parameters.mu or truth.semi_major_axis != first.plant_parameters.semi_major_axis:
            raise ValueError("Execution truth and controller need the same fixed mu and semi-major axis")
        for name, value in (("eccentricity", target.eccentricity), ("alpha", target.alpha)):
            lo, hi = getattr(first.target_orbit, name)
            if not lo <= _exact(value) <= hi:
                raise ValueError("The actual target clock lies outside its terminal bound context")
            if getattr(truth, name)[0] != _exact(value):
                raise ValueError("Numerical translation truth and the physical target need the same actual orbital clock")
        if not first.target_orbit.theta0[0] <= 0 <= first.target_orbit.theta0[1]:
            raise ValueError("The maintained physical target clock uses zero orbital phase")
        if truth.theta0 != (Fraction(0), Fraction(0)):
            raise ValueError("Numerical translation truth must share the physical target's zero orbital phase")
        if (_exact(target.mu) != first.target_orbit.mu
                or _exact(target.semi_major_axis) != first.target_orbit.semi_major_axis
                or bool(target.gravity_gradient) != first.gravity_gradient
                or tuple(tuple(map(_exact, row)) for row in target.inertia) != first.inertia):
            raise ValueError("The actual physical target contradicts the certified terminal context")
        shape = target.as_dict().get("shape")
        geometries = tuple(getattr(builder, "target_geometry", None) for builder in self.builders)
        self.composite_geometry = None
        if shape == "composite_spacecraft":
            from orbexa.core.spacecraft_geometry import SpacecraftGeometry
            geometry = getattr(target, "geometry", None)
            if (not isinstance(geometry, SpacecraftGeometry) or any(value is None for value in geometries)
                    or any(value.fingerprint != geometry.fingerprint for value in geometries)):
                raise ValueError("The actual composite target and every terminal builder must share one explicit complete geometry")
            self.composite_geometry = geometry
        elif shape == "cylinder":
            if (any(value is not None for value in geometries) or _exact(target.radius) != first.target_radius
                    or _exact(target.half_length) != first.target_half_length
                    or _exact(target.tolerance) != first.target_tolerance):
                raise ValueError("The actual cylinder contradicts its explicit terminal geometry")
        else:
            raise ValueError("Unsupported physical terminal target geometry; no cylinder fallback is available")
        if not target.anomaly_domain[0] <= float(first.start_anomaly) < float(first.end_anomaly) <= target.anomaly_domain[1]:
            raise ValueError("The complete terminal window must belong to the actual target's numerical domain")
        entry_spin = IntervalBox.point(target.body_angular_velocity_at(float(first.start_anomaly)))
        if not first.entry_spin.contains(entry_spin):
            raise ValueError("The declared entry spin box does not enclose the actual numerical target spin")
        self.encloser = ValidatedFlowEncloser(reachability_config)
        guard_type = CompositeSweptCollisionGuard if self.composite_geometry is not None else SweptCollisionGuard
        self.guard = guard_type(precision_bits=self.encloser.config.precision_bits)
        self.record_sink, self.acceleration_source = record_sink, acceleration_source
        self.source_context = dict(caller=dict(source_context or {}), truth=truth.as_dict(),
            target=target.as_dict(), observation_source="Explicit caller-supplied simulation observation contract",
            execution_scope="Offline numerical replay with exact scheduled anomaly holds; no hardware scheduler")
        if self.policy is TerminalFeedforwardPolicy.MIDPOINT_V1:
            caller = self.source_context["caller"]
            if caller.get("origin") not in ("actual_approach_endpoint", "initialized_terminal_fixture"):
                raise ValueError("Midpoint execution needs an explicit actual-approach or initialized-fixture origin")
            for key in ("source_snapshot_sha256", "error_source_sha256"):
                value = caller.get(key)
                if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                    raise ValueError("Midpoint execution needs its frozen source and error-declaration SHA256 identities")
            if caller["origin"] == "actual_approach_endpoint":
                value = caller.get("approach_result_sha256")
                if (not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
                        or not isinstance(caller.get("source_approach_lineage"), dict)):
                    raise ValueError("Actual midpoint execution must identify the original approach and arrival/region/transition lineage")
            self.source_context["terminal_stage"] = dict(schema="orbexa.physical_terminal_stage.v1",
                held_policy=self.policy.value, proof_reference=first._terminal_proof_reference(),
                exact_start_anomaly=str(first.start_anomaly), exact_end_anomaly=str(first.end_anomaly),
                maximum_hold_by_reference={builder.reference_id: str(builder.maximum_hold) for builder in self.builders},
                builder_contract_sha256=tuple(_digest(controller.result.assumptions) for controller in self.fleet.controllers),
                target_context_sha256=_digest(self.source_context["target"]),
                scope="Separate finite terminal stage; no approach state, clock, MPC or adaptation history is replaced")

    @staticmethod
    def _coefficients(anomaly, parameters):
        # Numerical replay only. All global bounds and continuous-flow proof
        # coefficients are independently evaluated by Arb in the proof layer.
        import numpy as np
        e, alpha, beta, phase = (float(getattr(parameters, name)[0])
                                  for name in ("eccentricity", "alpha", "beta", "theta0"))
        q, mu, axis = float(anomaly), float(parameters.mu), float(parameters.semi_major_axis)
        h2 = mu * axis * (1 - e * e)
        exponential = np.exp(2 * alpha * q)
        denominator = exponential + e * np.cos(q - phase)
        radius = h2 * (1 + 4 * alpha * alpha) / (mu * denominator)
        k = (e * np.sin(q - phase) - 2 * alpha * exponential) / denominator
        curvature = (1 + 4 * alpha * alpha) * exponential / denominator
        gamma = beta - alpha
        damping = 2 * k - gamma
        matrix = np.array(((1 - curvature, -damping, 0, damping, 2, 0),
                           (damping, 1 + 2 * curvature, 0, -2, damping, 0),
                           (0, 0, -curvature, 0, 0, damping)))
        affine = np.array((gamma * radius, -gamma * radius * k, 0))
        return matrix, affine

    def references_at(self, anomaly) -> tuple[TerminalReferenceSample, ...]:
        q = _exact(anomaly)
        result = []
        for builder in self.builders:
            if abs(_exact(float(q)) - q) > builder.errors.anomaly_evaluation_error_abs:
                raise ValueError("Binary64 reference query exceeds its declared anomaly-evaluation budget")
            port = tuple(map(float, builder.body_ports[builder.port_index]))
            matrix, affine = self._coefficients(q, builder.nominal_parameters)
            result.append(TerminalReferenceSample(q, self.target.docking_state(port, float(q)),
                self.target.docking_acceleration(port, float(q)), matrix, affine, builder.reference_id))
        return tuple(result)

    def hold_references_at(self, start_anomaly, hold_width) -> tuple[TerminalHoldReferences, ...]:
        """Evaluate feedback at the real start and feedforward at the actual midpoint."""
        q, hold = _exact(start_anomaly), _exact(hold_width)
        feedback = self.references_at(q)
        sample_q = q + self.policy.sample_fraction * hold
        feedforward = feedback if self.policy is TerminalFeedforwardPolicy.START_V1 else self.references_at(sample_q)
        pairs = tuple(TerminalHoldReferences(self.policy, start, preview, _exact(float(q)),
            _exact(float(sample_q)), builder.errors.anomaly_evaluation_error_abs)
            for builder, start, preview in zip(self.builders, feedback, feedforward))
        for pair, controller, builder in zip(pairs, self.fleet.controllers, self.builders):
            pair.validate(controller.certificate.bounds, hold,
                          maximum_query_displacement=builder.errors.anomaly_evaluation_error_abs)
            if abs(_exact(float(q + hold)) - q - hold) > builder.errors.anomaly_evaluation_error_abs:
                raise ValueError("Binary64 held endpoint exceeds its declared clock-evaluation budget")
        return pairs

    def _policy_event(self, q, hold, pairs):
        if self.policy is TerminalFeedforwardPolicy.START_V1:
            return {}
        return dict(schema=MIDPOINT_EXECUTION_SCHEMA, held_policy=self.policy.value,
            exact_midpoint_anomaly=str(q + hold / 2), exact_end_anomaly=str(q + hold),
            target_context_sha256=_digest(self.source_context["target"]),
            source_context_sha256=_digest(self.source_context),
            reference_samples=tuple(pair.as_dict() for pair in pairs),
            evaluation_error_contracts=tuple(builder.errors.as_dict() for builder in self.builders))

    def geometry_bounds_at(self, anomaly):
        """The exact declared target shape with its finite-window pose contract."""
        q = _exact(anomaly)
        first = self.builders[0]
        rotation_rate = max(controller.result.derived["rotation_derivative_operator_bounds"][0]
                            for controller in self.fleet.controllers)
        common = dict(orientation_matrix_at_start=self.target.rotation_matrix_at(float(q)),
            orientation_matrix_error=self.orientation_matrix_error, angular_rate_per_anomaly=rotation_rate,
            scope="Derived global physical terminal rotation-rate bound and explicitly declared matrix evaluation error")
        if self.composite_geometry is not None:
            return RotatingCompositeBounds(geometry=self.composite_geometry, **common)
        return RotatingCylinderBounds(radius=first.target_radius * (1 + first.target_tolerance),
                                      half_length=first.target_half_length * (1 + first.target_tolerance), **common)

    @staticmethod
    def _numerical_states(values, count):
        import numpy as np
        states = np.asarray(values, dtype=float)
        if states.shape != (count, 6) or not np.all(np.isfinite(states)):
            raise ValueError("The actual terminal endpoint needs one finite six-state per fleet member")
        return states.copy()

    @staticmethod
    def _tuple_states(states):
        return tuple(tuple(map(float, value)) for value in states)

    def _observation_reason(self, observation, states):
        if not isinstance(observation, TerminalObservation) or len(observation.measurements) != len(self.builders):
            return "Observation source did not return the complete fleet"
        for box, measured, state, builder in zip(observation.state_enclosures, observation.measurements,
                                                 states, self.builders):
            if not box.contains(IntervalBox.point(state)):
                return "The supplied observation does not enclose the actual numerical state"
            if any(max(abs(lo - value), abs(hi - value)) > error for lo, hi, value, error in
                   zip(box.lower, box.upper, measured, builder.errors.sensing_error_abs)):
                return "The supplied measurement/enclosure exceeds the physical sensor budget"
        return ""

    def run(self, *, initial_anomaly, initial_states, initial_observation: TerminalObservation,
            end_anomaly=None) -> PhysicalTerminalExecutionResult:
        import numpy as np
        from scipy.integrate import solve_ivp

        q = _exact(initial_anomaly)
        first = self.builders[0]
        end = first.end_anomaly if end_anomaly is None else _exact(end_anomaly)
        if q != first.start_anomaly or not q < end <= first.end_anomaly:
            raise ValueError("Execute from the actual certified entry and stop within its finite domain")
        states = self._numerical_states(initial_states, len(self.builders))
        if not isinstance(initial_observation, TerminalObservation) or len(initial_observation.measurements) != len(self.builders):
            raise ValueError("Supply the actual approach endpoint observation for every fleet member")
        observation = initial_observation
        observation_q = q
        history, clocks, records = [self._tuple_states(states)], [q], []
        admission = dict(admissible=False, reasons=("Reference evaluation has not been admitted",),
                         anomaly=str(q), members_contained=())
        initial_q = q
        certificates = tuple(controller.result.as_dict() for controller in self.fleet.controllers)

        def finish(status, reason):
            duration = self.target.physical_time_at(float(q)) - self.target.physical_time_at(float(initial_q))
            extra = {} if self.policy is TerminalFeedforwardPolicy.START_V1 else dict(
                method=MIDPOINT_EXECUTION_METHOD, held_policy=self.policy.value, schema=MIDPOINT_EXECUTION_SCHEMA)
            return PhysicalTerminalExecutionResult(status, reason, tuple(clocks), tuple(history),
                initial_observation, observation, observation_q, admission, certificates, tuple(records), float(duration),
                dict(self.source_context), **extra)

        try:
            admission = self.fleet.admit(observation.state_enclosures, self.references_at(q)).as_dict()
        except Exception as error:
            return finish("REFUSED_REFERENCE", f"Reference source {type(error).__name__}: {error}")

        observation_reason = self._observation_reason(observation, states)
        if observation_reason:
            return finish("REFUSED_OBSERVATION", observation_reason)
        if not admission["admissible"]:
            return finish("REFUSED_INITIAL", "; ".join(admission["reasons"]))
        while q < end:
            hold = min(end - q, *(builder.maximum_hold for builder in self.builders))
            if float(q + hold) <= float(q) or any(abs(_exact(float(q)) - q) > builder.errors.anomaly_evaluation_error_abs
                or abs(_exact(float(q + hold)) - q - hold) > builder.errors.anomaly_evaluation_error_abs for builder in self.builders):
                return finish("REFUSED_CLOCK_EVALUATION", "Binary64 sample-clock evaluation exceeds its declared bound")
            try:
                pairs = self.hold_references_at(q, hold)
            except Exception as error:
                return finish("REFUSED_REFERENCE", f"Reference source {type(error).__name__}: {error}")
            references = tuple(pair.feedback for pair in pairs)
            try:
                commands = self.fleet.commands(observation.measurements, observation.state_enclosures,
                    references, hold_width=hold,
                    hold_references=pairs if self.policy is TerminalFeedforwardPolicy.MIDPOINT_V1 else None)
            except ValueError as error:
                return finish("REFUSED_MEMBERSHIP", str(error))
            flows = tuple(self.encloser.enclose_orbital(HeldInputRequest(
                initial_state=box, held_input=command.applied_input,
                start_anomaly=(q, q), hold_width=(hold, hold),
                acceleration_disturbance=IntervalBox.from_center_radius((0, 0, 0),
                    builder.errors.acceleration_disturbance_abs)), builder.plant_parameters)
                for builder, box, command in zip(self.builders, observation.state_enclosures, commands))
            method = EXECUTION_METHOD if self.policy is TerminalFeedforwardPolicy.START_V1 else MIDPOINT_EXECUTION_METHOD
            event = dict(method=method, stage=len(records), start_anomaly=str(q), hold_width=str(hold),
                         initial_observation=observation.as_dict(),
                         commands=tuple(command.as_dict() for command in commands),
                         flows=tuple(flow.as_dict() for flow in flows),
                         flow_fingerprints=tuple(flow.fingerprint for flow in flows))
            event.update(self._policy_event(q, hold, pairs))
            if any(not flow.proven for flow in flows):
                event.update(executed=False, decision="REFUSED_FLOW")
                records.append(event)
                if self.record_sink:
                    self.record_sink(event)
                return finish("REFUSED_FLOW", "; ".join(flow.reason for flow in flows if not flow.proven))
            geometry = self.geometry_bounds_at(q)
            solid_error = max(builder.errors.target_solid_position_error_radius for builder in self.builders)
            collision = self.guard.check(flows, tuple(builder.body_radius for builder in self.builders),
                target=geometry, clearance=first.clearance + solid_error,
                minimum_pair_center_distance=self.minimum_pair_center_distance)
            event["geometry"] = collision.as_dict()
            linkage = terminal_hold_linkage(event)
            event["command_flow_geometry_linkage"] = linkage
            event["command_flow_geometry_sha256"] = _digest(linkage)
            if not collision.admissible:
                event.update(executed=False, decision="REFUSED_GEOMETRY")
                records.append(event)
                if self.record_sink:
                    self.record_sink(event)
                return finish("REFUSED_GEOMETRY", collision.reason)
            try:
                control = np.asarray(tuple(command.float_command() for command in commands))
            except ValueError as error:
                event.update(executed=False, decision="REFUSED_COMMAND_CONVERSION", message=str(error))
                records.append(event)
                if self.record_sink:
                    self.record_sink(event)
                return finish("REFUSED_COMMAND_CONVERSION", str(error))

            def rhs(anomaly, flat):
                actual = flat.reshape(len(self.builders), 6)
                matrix, affine = self._coefficients(anomaly, self.truth)
                disturbance = np.zeros((len(self.builders), 3))
                if self.acceleration_source is not None:
                    try:
                        disturbance = np.asarray(self.acceleration_source(anomaly, actual.copy()), dtype=float)
                    except Exception as error:
                        raise _TerminalSourceFailure(f"Acceleration source {type(error).__name__}: {error}") from error
                    if disturbance.shape != (len(self.builders), 3) or not np.all(np.isfinite(disturbance)) or any(
                            abs(_exact(value)) > allowance for row, builder in zip(disturbance, self.builders)
                            for value, allowance in zip(row, builder.errors.acceleration_disturbance_abs)):
                        raise ValueError("Numerical process source exceeds the certified acceleration budget")
                return np.concatenate((actual[:, 3:], actual @ matrix.T + affine + control + disturbance), axis=1).ravel()

            try:
                solution = solve_ivp(rhs, (float(q), float(q + hold)), states.ravel(), method="DOP853",
                                     rtol=1e-12, atol=1e-13, max_step=float(hold) / 2)
            except (ValueError, FloatingPointError, OverflowError, _TerminalSourceFailure) as error:
                event.update(executed=False, decision="NUMERICAL_PROPAGATION_REFUSED", message=str(error))
                records.append(event)
                if self.record_sink:
                    self.record_sink(event)
                return finish("NUMERICAL_PROPAGATION_REFUSED", str(error))
            if not solution.success or not np.all(np.isfinite(solution.y)):
                event.update(executed=False, decision="NUMERICAL_PROPAGATION_FAILED", message=solution.message)
                records.append(event)
                if self.record_sink:
                    self.record_sink(event)
                return finish("NUMERICAL_PROPAGATION_FAILED", solution.message)
            states = solution.y[:, -1].reshape(len(self.builders), 6)
            q += hold
            history.append(self._tuple_states(states))
            clocks.append(q)
            endpoint_contained = tuple(flow.endpoint.contains(IntervalBox.point(state)) for flow, state in zip(flows, states))
            event.update(executed=True, decision="EXECUTED_PROVEN_HOLD", end_anomaly=str(q),
                         actual_endpoint=self._tuple_states(states), numerical_endpoint_contained=endpoint_contained)
            records.append(event)
            if self.record_sink:
                self.record_sink(event)
            if not all(endpoint_contained):
                return finish("NUMERICAL_ENDPOINT_MISMATCH", "Numerical replay left a proved continuous-flow endpoint enclosure")
            try:
                next_observation = self.observation_source(q, self._tuple_states(states), tuple(flow.endpoint for flow in flows))
            except Exception as error:
                return finish("REFUSED_OBSERVATION", f"Observation source {type(error).__name__}: {error}")
            observation_reason = self._observation_reason(next_observation, states)
            if observation_reason:
                return finish("REFUSED_OBSERVATION", observation_reason)
            intersections = []
            for observed, flow in zip(next_observation.state_enclosures, flows):
                lower = tuple(max(a, b) for a, b in zip(observed.lower, flow.endpoint.lower))
                upper = tuple(min(a, b) for a, b in zip(observed.upper, flow.endpoint.upper))
                if any(lo > hi for lo, hi in zip(lower, upper)):
                    observation = next_observation
                    return finish("REFUSED_OBSERVATION", "Independent observation and preceding flow enclosures are disjoint")
                intersections.append(IntervalBox(lower, upper))
            observation = TerminalObservation(next_observation.measurements, tuple(intersections))
            observation_q = q
        try:
            final_admission = self.fleet.admit(observation.state_enclosures, self.references_at(q))
        except Exception as error:
            return finish("REFUSED_REFERENCE", f"Reference source {type(error).__name__}: {error}")
        if not final_admission.admissible:
            return finish("REFUSED_FINAL_MEMBERSHIP", "Final observation is outside the robust terminal set")
        return finish("COMPLETED_FINITE_WINDOW", "Every held command and whole-fleet swept flow was proved before numerical execution")
