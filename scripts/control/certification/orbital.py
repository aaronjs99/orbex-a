"""Authenticated analytical orbital holds with separate VoC endpoints.

This service owns its Picard proof dependency. Saved flags or caller-supplied
flows never grant admission: rebuild recomputes the entire supported proof.
The original Picard flow remains the swept witness, with its old fingerprint.
No mission, fleet supervisor or physical execution is performed here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from orbexa.control.certification import reachability as _reachability
from orbexa.control.certification.reachability import FRAME, METHOD as PICARD_METHOD, EnclosureStatus, FlowCertificate, HeldInputRequest, IntervalBox, OrbitalModelBounds, ReachabilityConfig, ValidatedFlowEncloser, _ARB_LOCK, _ArbArithmetic, _ProofUnresolved, _exact, _orbital_coefficients, _pair


ENDPOINT_METHOD = "arb-voc-affine-orbital-endpoint-v1"
HOLD_SCHEMA = "orbexa.orbital_hold.v1"
ENDPOINT_SCHEMA = "orbexa.orbital_endpoint.v1"
MODEL = "physical-lvlh-v1"
try:
    _LOADED_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
except OSError:
    _LOADED_SOURCE_SHA256 = None


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _sum_boxes(left, right):
    return IntervalBox(tuple(a + b for a, b in zip(left.lower, right.lower)),
                       tuple(a + b for a, b in zip(left.upper, right.upper)))


def _require_box(value, dimension, name):
    if type(value) is not IntervalBox or value.dimension != dimension:
        raise ValueError(f"{name} requires an explicit {dimension}-dimensional IntervalBox")


def _source_identity():
    files = {'control/certification/reachability.py': Path(_reachability.__file__),
             'control/certification/orbital.py': Path(__file__)}
    try:
        hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                  for name, path in files.items()}
    except OSError as error:
        raise ValueError("Readable supported source files are required for orbital proof identity") from error
    loaded = {'control/certification/reachability.py': _reachability._LOADED_SOURCE_SHA256,
              'control/certification/orbital.py': _LOADED_SOURCE_SHA256}
    if hashes != loaded:
        raise ValueError("Supported source bytes differ from the loaded orbital proof modules")
    return dict(model=MODEL, frame=FRAME, picard_method=PICARD_METHOD,
                endpoint_method=ENDPOINT_METHOD, source_files_sha256=hashes)


def _box_record(value, name):
    if not isinstance(value, dict) or set(value) != {
            "lower", "upper", "exact_lower", "exact_upper"}:
        raise ValueError(f"{name} requires the complete exact box record")
    result = IntervalBox(value["exact_lower"], value["exact_upper"])
    if _json(result.as_dict()) != _json(value):
        raise ValueError(f"{name} exact and exported bounds disagree")
    return result


@dataclass(frozen=True)
class OrbitalHoldContext:
    """Retained physical assumptions, exact domain and fixed proof policy.

    model_source_sha256 identifies the caller's immutable physical-model
    declaration. Its containment of the real plant is an explicit assumption;
    a hash does not prove that assumption. No fitted point or contracted prior
    is substituted for model by this service.
    """

    model: OrbitalModelBounds
    anomaly_domain: tuple[Any, Any]
    input_limits: IntervalBox
    acceleration_disturbance: IntervalBox
    model_source_sha256: str
    reachability_config: ReachabilityConfig

    def __post_init__(self):
        if type(self.model) is not OrbitalModelBounds:
            raise ValueError("Use the supported analytical OrbitalModelBounds directly")
        domain = _pair(self.anomaly_domain, "anomaly_domain")
        if domain[0] == domain[1]:
            raise ValueError("A nonempty finite anomaly domain is required")
        _require_box(self.input_limits, 3, "Physical input limits")
        _require_box(self.acceleration_disturbance, 3, "Original process disturbance")
        if (not isinstance(self.model_source_sha256, str) or
                re.fullmatch(r"[0-9a-f]{64}", self.model_source_sha256) is None):
            raise ValueError("Identify the original physical-model declaration by SHA-256")
        if (type(self.reachability_config) is not ReachabilityConfig or
                self.reachability_config.precision_bits < 128):
            raise ValueError("An explicit supported proof policy with at least 128 bits is required")
        object.__setattr__(self, "anomaly_domain", domain)
        _json(self.as_dict())

    def as_dict(self):
        policy = {name: str(value) if isinstance(value, Fraction) else value
                  for name, value in vars(self.reachability_config).items()}
        return dict(model=MODEL, frame=FRAME, model_bounds=self.model.as_dict(),
                    anomaly_domain=tuple(map(str, self.anomaly_domain)),
                    input_limits=self.input_limits.as_dict(),
                    acceleration_disturbance=self.acceleration_disturbance.as_dict(),
                    model_source_sha256=self.model_source_sha256, proof_policy=policy)


@dataclass(frozen=True)
class OrbitalHoldRequest:
    """A constant-input family and a separate varying delivery-error family.

    The clock and duration are exact. Input is in m/rad²; state is [m,m/rad].
    Every actual actuator value must lie in constant_input plus the varying
    error box. The original process error is supplied only by the context.
    """

    initial_state: IntervalBox
    constant_input: IntervalBox
    start_anomaly: Any
    hold_width: Any
    varying_actuator_error: IntervalBox = field(
        default_factory=lambda: IntervalBox.point((0, 0, 0)))

    def __post_init__(self):
        _require_box(self.initial_state, 6, "Initial state")
        _require_box(self.constant_input, 3, "Constant input")
        _require_box(self.varying_actuator_error, 3, "Varying actuator error")
        start, width = _exact(self.start_anomaly), _exact(self.hold_width)
        if width <= 0:
            raise ValueError("This endpoint method requires a positive exact hold width")
        object.__setattr__(self, "start_anomaly", start)
        object.__setattr__(self, "hold_width", width)
        _json(self.as_dict())

    def as_dict(self):
        return dict(initial_state=self.initial_state.as_dict(),
                    constant_input=self.constant_input.as_dict(),
                    varying_actuator_error=self.varying_actuator_error.as_dict(),
                    start_anomaly=str(self.start_anomaly), hold_width=str(self.hold_width))

    @classmethod
    def from_dict(cls, value):
        expected = {"initial_state", "constant_input", "varying_actuator_error",
                    "start_anomaly", "hold_width"}
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("An exact supported orbital hold request is required")
        try:
            result = cls(_box_record(value["initial_state"], "Initial state"),
                         _box_record(value["constant_input"], "Constant input"),
                         value["start_anomaly"], value["hold_width"],
                         _box_record(value["varying_actuator_error"], "Varying actuator error"))
        except (KeyError, TypeError, ArithmeticError) as error:
            raise ValueError("Malformed orbital hold request") from error
        if _json(result.as_dict()) != _json(value):
            raise ValueError("The hold request is not in its exact canonical form")
        return result


@dataclass(frozen=True)
class OrbitalEndpointCertificate:
    """Immutable evidence only; restore through OrbitalHoldProofService."""

    _record_json: str = field(repr=False)

    def as_dict(self):
        return json.loads(self._record_json)

    @property
    def fingerprint(self):
        return hashlib.sha256(self._record_json.encode("utf-8")).hexdigest()

    @property
    def endpoint(self):
        return _box_record(self.as_dict()["refined_intersection_endpoint"], "Refined endpoint")


@dataclass(frozen=True)
class OrbitalHeldFlow:
    """Original swept certificate plus a separately identified endpoint.

    proven reports this result's status, not authenticity of arbitrary
    user-constructed/deserialized instances. Use the service's fresh prove
    or authoritative rebuild boundary before relying on saved evidence.
    """

    picard_flow: FlowCertificate
    endpoint_certificate: OrbitalEndpointCertificate | None
    _record_json: str = field(repr=False)

    def as_dict(self):
        record = json.loads(self._record_json)
        if (_json(self.picard_flow.as_dict()) != _json(record["picard_flow"]) or
                (None if self.endpoint_certificate is None else self.endpoint_certificate.as_dict())
                != record["endpoint_certificate"]):
            raise ValueError("Orbital hold evidence changed after construction")
        return record

    @property
    def fingerprint(self):
        return _digest(self.as_dict())

    @property
    def proven(self):
        return self.as_dict()["status"] == EnclosureStatus.PROVEN.value

    @property
    def endpoint(self):
        return None if self.endpoint_certificate is None else self.endpoint_certificate.endpoint


@dataclass(frozen=True)
class OrbitalHoldProofService:
    """Own and authenticate the supported Picard-plus-VoC proof construction.

    No arbitrary flow/coefficients/actuator callback is accepted. Instances
    bind exact source bytes and refuse a source change before subsequent use.
    This service neither narrows a prior nor establishes real-plant validity.
    """

    context: OrbitalHoldContext
    _implementation_json: str = field(init=False, repr=False)

    def __post_init__(self):
        if type(self.context) is not OrbitalHoldContext:
            raise ValueError("An explicit OrbitalHoldContext is required")
        object.__setattr__(self, "_implementation_json", _json(_source_identity()))

    @property
    def _implementation(self):
        return json.loads(self._implementation_json)

    def _validate(self, request, swept_entry):
        if type(request) is not OrbitalHoldRequest:
            raise ValueError("Use an exact OrbitalHoldRequest, not a supplied flow certificate")
        if _source_identity() != self._implementation:
            raise ValueError("The bound orbital proof implementation changed")
        lower, upper = self.context.anomaly_domain
        if not lower <= request.start_anomaly < request.start_anomaly + request.hold_width <= upper:
            raise ValueError("The complete hold exceeds the exact retained anomaly domain")
        applied = _sum_boxes(request.constant_input, request.varying_actuator_error)
        if not self.context.input_limits.contains(applied):
            raise ValueError("Constant input plus varying actuator delivery exceeds physical authority")
        entry = request.initial_state if swept_entry is None else swept_entry
        _require_box(entry, 6, "Picard swept-proof entry")
        if not entry.contains(request.initial_state):
            raise ValueError("The original Picard request must contain the complete proposed entry")
        return entry

    def prove(self, request: OrbitalHoldRequest, *, swept_entry: IntervalBox | None = None):
        """Fresh proof; optional larger entry reuses only a proved flow family.

        Every noninitial field is constructed here and is identical for the
        original Picard family and the smaller endpoint-refinement family.
        """
        entry = self._validate(request, swept_entry)
        context = self.context
        total = _sum_boxes(context.acceleration_disturbance, request.varying_actuator_error)
        source_request = HeldInputRequest(entry, request.constant_input,
            (request.start_anomaly, request.start_anomaly),
            (request.hold_width, request.hold_width), total)
        flow = ValidatedFlowEncloser(context.reachability_config).enclose_orbital(source_request, context.model)
        if (flow.method != PICARD_METHOD or flow.model != MODEL or flow.frame != FRAME or
                flow.initial_state != source_request.initial_state or flow.request != source_request.as_dict() or
                flow.start_anomaly != source_request.start_anomaly or flow.hold_width != source_request.hold_width or
                flow.model_bounds != context.model.as_dict()):
            raise ValueError("The authoritative Picard service returned an inconsistent model/request")
        endpoint, status, reason = None, flow.status, flow.reason
        if flow.proven:
            if flow.endpoint is None or flow.swept_state is None or not flow.segments:
                raise ValueError("The authentic positive-width Picard proof is incomplete")
            try:
                endpoint = self._refine(flow, request)
                reason = "Original continuous Picard swept proof and separate constant-input VoC endpoint proved"
            except _ProofUnresolved as error:
                status, reason = EnclosureStatus.UNRESOLVED, str(error)
        record = dict(schema=HOLD_SCHEMA, status=status.value, reason=reason,
            context=context.as_dict(), implementation=self._implementation,
            request=request.as_dict(), picard_request=source_request.as_dict(),
            picard_flow=flow.as_dict(), picard_flow_fingerprint=flow.fingerprint,
            endpoint_certificate=None if endpoint is None else endpoint.as_dict(),
            endpoint_certificate_fingerprint=None if endpoint is None else endpoint.fingerprint,
            input_semantics="One unknown constant vector per exact hold plus separately bounded measurable varying actuator delivery",
            scope="Analytical orbital held-flow and endpoint proof only; no fleet, terminal, runtime fallback or hardware claim")
        return OrbitalHeldFlow(flow, endpoint, _json(record))

    def _refine(self, flow, request):
        import flint

        config, model = self.context.reachability_config, self.context.model
        q, width = request.start_anomaly, request.hold_width
        with _ARB_LOCK, flint.ctx.workprec(config.precision_bits):
            ar = _ArbArithmetic(flint)
            h = ar.scalar(width)
            matrix_start, affine_start, _ = _orbital_coefficients(ar.scalar(q), model, ar)
            f0 = tuple(tuple(ar.exact_point(value.mid()) for value in row) for row in matrix_start)
            b0 = tuple(ar.exact_point(value.mid()) for value in affine_start)
            matrix, affine, _ = _orbital_coefficients(ar.interval((q, q + width)), model, ar)
            swept = ar.vector(flow.swept_state)
            total = _sum_boxes(self.context.acceleration_disturbance, request.varying_actuator_error)
            process = (ar.arb(0),) * 3 + ar.vector(total)
            residual = tuple(sum(((matrix[i][j] - ar.scalar(f0[i][j])) * swept[j]
                                  for j in range(6)), affine[i] - ar.scalar(b0[i]) + process[i])
                             for i in range(6))
            residual_box = ar.box(residual)
            residual_abs = tuple(max(abs(lo), abs(hi))
                                 for lo, hi in zip(residual_box.lower, residual_box.upper))
            augmented = flint.arb_mat([
                [ar.scalar(f0[i][j]) * h if i < 6 and j < 6 else
                 h if i < 6 and j == i + 6 else ar.arb(0) for j in range(12)]
                for i in range(12)])
            transition = augmented.exp()
            positive = flint.arb_mat([[ar.scalar(abs(value)) * h for value in row] for row in f0]).exp()
            error = ar.box(tuple(h * sum((positive[i, j] * ar.scalar(residual_abs[j])
                                         for j in range(6)), ar.arb(0)) for i in range(6))).upper
            initial = ar.vector(request.initial_state)
            held = (ar.arb(0),) * 3 + ar.vector(request.constant_input)
            forcing = tuple(ar.scalar(value) + u for value, u in zip(b0, held))
            base = tuple(sum((transition[i, j] * initial[j] + transition[i, j + 6] * forcing[j]
                              for j in range(6)), ar.arb(0)) for i in range(6))
            raw = ar.box(tuple(value + ar.interval((-allowance, allowance))
                               for value, allowance in zip(base, error)))
            lo = tuple(max(a, b) for a, b in zip(raw.lower, flow.endpoint.lower))
            hi = tuple(min(a, b) for a, b in zip(raw.upper, flow.endpoint.upper))
            if any(a > b for a, b in zip(lo, hi)):
                raise _ProofUnresolved("Independent Picard and VoC endpoints have empty intersection")
            endpoint = IntervalBox(lo, hi)

            def matrix_box(rows):
                return [ar.box(tuple(row)).as_dict() for row in rows]

            record = dict(schema=ENDPOINT_SCHEMA, method=ENDPOINT_METHOD,
                precision_bits=config.precision_bits, backend_version=flint.__version__,
                frame=FRAME, model=MODEL, model_bounds=model.as_dict(),
                context_fingerprint=_digest(self.context.as_dict()),
                implementation_fingerprint=_digest(self._implementation),
                request=request.as_dict(), source_picard_request=flow.request,
                initial_state_subset_of_picard_entry=flow.initial_state.contains(request.initial_state),
                picard_flow_fingerprint=flow.fingerprint,
                validated_picard_swept_state=flow.swept_state.as_dict(),
                frozen_exact_F0=[[str(value) for value in row] for row in f0],
                frozen_exact_b0=[str(value) for value in b0],
                coefficient_matrix_enclosure=matrix_box(matrix), affine_enclosure=ar.box(affine).as_dict(),
                residual_enclosure=residual_box.as_dict(), residual_abs_upper=tuple(map(str, residual_abs)),
                matrix_E_enclosure=matrix_box([[transition[i, j] for j in range(6)] for i in range(6)]),
                matrix_G_enclosure=matrix_box([[transition[i, j + 6] for j in range(6)] for i in range(6)]),
                matrix_exp_abs_F0_h_enclosure=matrix_box([[positive[i, j] for j in range(6)] for i in range(6)]),
                error_radius_upper=tuple(map(str, error)), raw_voc_endpoint=raw.as_dict(),
                original_picard_endpoint=flow.endpoint.as_dict(), refined_intersection_endpoint=endpoint.as_dict(),
                constant_input=request.constant_input.as_dict(),
                varying_actuator_error=request.varying_actuator_error.as_dict(),
                original_process_disturbance=self.context.acceleration_disturbance.as_dict(),
                status="CONDITIONAL_PROVEN_ENDPOINT_ONLY",
                theorem="Constant held input uses G; varying delivery and process are included once in the residual; original genuine Picard swept family remains unchanged")
            return OrbitalEndpointCertificate(_json(record))

    def rebuild(self, evidence: Mapping[str, Any] | OrbitalHeldFlow):
        """Reprove complete saved evidence; status flags/digests alone cannot pass."""
        saved = evidence.as_dict() if type(evidence) is OrbitalHeldFlow else evidence
        if not isinstance(saved, dict):
            raise ValueError("A complete supported orbital hold evidence record is required")
        expected_keys = {"schema", "status", "reason", "context", "implementation", "request",
                         "picard_request", "picard_flow", "picard_flow_fingerprint",
                         "endpoint_certificate", "endpoint_certificate_fingerprint",
                         "input_semantics", "scope"}
        try:
            if (set(saved) != expected_keys or saved["schema"] != HOLD_SCHEMA or
                    saved["status"] != EnclosureStatus.PROVEN.value or
                    _json(saved["context"]) != _json(self.context.as_dict()) or
                    _json(saved["implementation"]) != _json(self._implementation) or
                    _source_identity() != self._implementation):
                raise ValueError("Saved hold method, retained context or implementation identity changed")
            endpoint = saved["endpoint_certificate"]
            if (not isinstance(endpoint, dict) or endpoint.get("schema") != ENDPOINT_SCHEMA or
                    endpoint.get("method") != ENDPOINT_METHOD or
                    saved["endpoint_certificate_fingerprint"] != _digest(endpoint)):
                raise ValueError("A complete supported distinct endpoint witness is required")
            request = OrbitalHoldRequest.from_dict(saved["request"])
            entry = _box_record(saved["picard_request"]["initial_state"], "Original Picard entry")
            rebuilt = self.prove(request, swept_entry=entry)
            if not rebuilt.proven or _json(rebuilt.as_dict()) != _json(saved):
                raise ValueError("Reconstructed orbital proof differs from saved evidence")
            return rebuilt
        except (KeyError, TypeError, ArithmeticError, RecursionError) as error:
            raise ValueError("Malformed orbital hold evidence") from error
