"""Transactional held-fleet admission, fallback dispatch and commit."""
from __future__ import annotations

from typing import Callable
from orbexa.control.certification.reachability import FlowCertificate, HeldInputRequest, IntervalBox, SweptCollisionGuard, _exact
from orbexa.control.fallback.refuge import _identity, exact_record
from orbexa.control.fallback.common import _same_plant
from orbexa.control.fallback.plan import FleetFallbackPlan
from orbexa.control.fallback.records import PerformanceAttempt, PerformanceAdmissionRefusal, PerformanceProposal, HeldFleetDecision, CommandResourceBound


def _intersect(left, right):
    lower = tuple(max(a, c) for a, c in zip(left.lower, right.lower))
    upper = tuple(min(a, c) for a, c in zip(left.upper, right.upper))
    if any(a > c for a, c in zip(lower, upper)):
        raise ValueError("Observation contradicts the proved continuous endpoint")
    return IntervalBox(lower, upper)


def _require_policy_supports(current, candidate, start, acceleration_disturbance):
    """Context names alone cannot authenticate tighter physical assumptions.

    A candidate may not silently increase hardware authority or shrink the
    original sensor/actuator support. A separately authenticated calibration
    interface would be needed to allow a justified reduction in these budgets.
    Every future segment, not only its entry controller, is checked.
    """
    original = dict(current.segment_at(start).controllers)
    disturbance = tuple(max(abs(lo), abs(hi)) for lo, hi in zip(
        acceleration_disturbance.lower, acceleration_disturbance.upper))
    for segment in candidate.segments:
        for key, controller in segment.controllers:
            a, b = original[key].certificate.spec, controller.certificate.spec
            if not _same_plant(a.context, b.context):
                raise ValueError('Successor changed actual model, sensing, actuator, geometry or fuel context')
            if not a.input_limits.contains(b.input_limits):
                raise ValueError('Successor increased the original physical hardware input authority')
            if any(new < old for new, old in zip(b.actuator_error_abs, a.actuator_error_abs)):
                raise ValueError('Successor reduced the original physical actuator-error support')
            if any(new < old for new, old in zip(b.sensing_error_abs, a.sensing_error_abs)):
                raise ValueError('Successor reduced the original physical sensing-error support')
            if any(disturbance):
                verifier = getattr(controller.builder.drift_bounds_provider, 'verify_certificate_source', None)
                if not callable(verifier) or verifier(controller.certificate, disturbance) is not True:
                    raise ValueError('Successor lacks a source-authenticated nonzero acceleration-disturbance bound')


class FleetAdmissionRuntime:
    """Own the admitted fallback, exact clock, fuel and pending fleet decision.

    flow_provider(chaser_id, HeldInputRequest) returns a genuine FlowCertificate.
    flow_identity(flow) supplies plant_fingerprint, target_model_fingerprint,
    clock_fingerprint and target_flow_fingerprint. Its default reads model_bounds.
    A scheduler must invoke the ready decision independently of a late planner;
    these classes do not implement physical actuator deadlines.
    """
    def __init__(self, active_fallback: FleetFallbackPlan, initial_observations,
                 *, flow_provider: Callable, remaining_fuel=None, flow_identity: Callable | None = None,
                 acceleration_disturbance: IntervalBox | None = None, performance_resource_bound: Callable | None = None):
        if not isinstance(active_fallback, FleetFallbackPlan) or not active_fallback.certificate.admissible:
            raise ValueError("A complete proved fleet fallback is required before motion")
        if not callable(flow_provider):
            raise TypeError("An actual nonlinear flow proof provider is required")
        active_fallback.verify_integrity()
        q = active_fallback.certificate.start
        if set(initial_observations) != set(active_fallback.ids):
            raise ValueError("Initial observed fleet differs from the complete fallback")
        boxes = {key: value.state_enclosure for key, value in initial_observations.items()}
        if not active_fallback.contains(boxes, q):
            raise ValueError("Complete initial state enclosure is outside the fallback entry region")
        for key, controller in active_fallback.segment_at(q).controllers:
            controller.observation_box(initial_observations[key], q)
        self.active_fallback, self.q = active_fallback, q
        self.flow_provider = flow_provider
        self.performance_resource_bound = performance_resource_bound
        self.flow_identity = (lambda flow: flow.model_bounds) if flow_identity is None else flow_identity
        self.acceleration_disturbance = acceleration_disturbance or IntervalBox.point((0, 0, 0))
        if self.acceleration_disturbance.dimension != 3:
            raise ValueError("Additional acceleration disturbance needs three components")
        first = active_fallback.segments[0]
        fuel = ({key: c.certificate.spec.available_fuel for key, c in first.controllers}
                if remaining_fuel is None else dict(remaining_fuel))
        if set(fuel) != set(active_fallback.ids):
            raise ValueError("Fuel fleet differs from fallback fleet")
        self.remaining_fuel = {key: _exact(value) for key, value in fuel.items()}
        needed = active_fallback.remaining_fuel(q)
        if any(self.remaining_fuel[key] < needed[key] for key in needed):
            raise ValueError("Reserved fuel does not cover the complete finite fallback")
        self._pending = None
        self.events = []

    def _fallback(self, observations, q, h, attempt, reason, *, refine_resources=True):
        plan = self.active_fallback
        plan.verify_integrity()
        if set(observations) != set(plan.ids):
            raise ValueError("Observed fleet changed")
        segment = plan.segment_at(q, h)
        commands, applied, charge, resources = [], [], [], []
        for key, controller in segment.controllers:
            value = controller.command(observations[key], q=q, hold_width=h, admitted_invariant=True)
            commands.append((key, tuple(_exact(x) for x in value.float_command())))
            applied.append((key, value.applied_input))
            b = controller.certificate.spec
            uniform = cost = h*b.fuel_rate_upper_per_q
            # The installed invariant already covers this complete command.
            # A refused optional refinement cannot interrupt its execution.
            if refine_resources and self.performance_resource_bound is not None:
                try:
                    resource = self.performance_resource_bound(key, value.applied_input, q, h)
                    if (isinstance(resource, CommandResourceBound) and resource.start == q
                            and resource.hold_width == h and resource.applied_input == value.applied_input
                            and resource.fuel_domain_id == b.context.fuel_domain_id
                            and resource.force_margin_n >= 0 and resource.fuel_upper_kg >= 0):
                        cost = min(uniform, resource.fuel_upper_kg)
                        if cost < uniform:
                            resources.append(dict(agent_id=key, uniform_fuel_upper_kg=uniform,
                                charged_fuel_kg=cost, command_resource_bound=exact_record(resource)))
                except Exception:
                    cost = uniform
            charge.append((key, cost))
        charge = tuple(charge)
        needed = plan.remaining_fuel(q+h)
        if any(dict(charge)[key]+needed[key] > self.remaining_fuel[key] for key in plan.ids):
            raise ValueError("Remaining finite fallback exceeds reserved fuel")
        if resources:
            attempt = PerformanceAttempt(attempt.status, attempt.message, attempt.solver_id,
                dict(original_attempt=exact_record(attempt), fallback_resources=resources))
        geometry = next(p for s,p in zip(plan.segments,plan.certificate.geometry) if s is segment)
        return HeldFleetDecision(q, h,
            "FALLBACK_HOLD" if q >= plan.certificate.terminal_holding_start else "FALLBACK_TRACK",
            reason, tuple(commands), tuple(applied), attempt, plan.fingerprint, plan.fingerprint,
            tuple((key, c.certificate.spec.context.fingerprint) for key, c in segment.controllers),
            tuple((key, observations[key].fingerprint) for key in plan.ids), charge,geometry=geometry)

    def decide(self, observations, *, q, hold_width,
               proposal: PerformanceProposal | None = None, failed_attempt: PerformanceAttempt | None = None,
               target_flow_fingerprint: str | None = None, refine_fallback_resources=True):
        q, h = _exact(q), _exact(hold_width)
        if not isinstance(refine_fallback_resources, bool):
            raise TypeError('Fallback resource refinement must be an explicit boolean')
        if self._pending is not None or q != self.q:
            raise ValueError("A decision is pending or the execution clock changed")
        attempt = (proposal.attempt if isinstance(proposal, PerformanceProposal) else failed_attempt
                   if failed_attempt is not None else PerformanceAttempt("NO_PROPOSAL", "No ready performance proposal"))
        if not isinstance(attempt, PerformanceAttempt):
            raise TypeError("Retain the actual typed performance attempt")
        # Existing feedback precedes consideration of a performance proof. No
        # proposal error can replace this queued command by zero or a pause.
        fallback = self._fallback(observations, q, h, attempt,
            "Stored finite feedback policy selected: " + attempt.status,
            refine_resources=refine_fallback_resources)
        if proposal is None:
            self._pending = (fallback, self.active_fallback, None)
            return fallback
        try:
            decision, endpoints = self._admit(proposal, observations, q, h, target_flow_fingerprint)
            self._pending = (decision, proposal.successor, endpoints)
            return decision
        except Exception as error:
            refused_attempt = PerformanceAttempt("ADMISSION_REFUSED", str(error), attempt.solver_id,
                dict(original_attempt=exact_record(attempt),
                     proposal_id=proposal.fingerprint if isinstance(proposal, PerformanceProposal) else None,
                     successor_id=proposal.successor.fingerprint if isinstance(proposal, PerformanceProposal) else None,
                     exception_type=type(error).__name__, q=q, hold_width=h,
                     admission_details=getattr(error, "admission_details", ())))
            fallback = self._fallback(observations, q, h, refused_attempt,
                f"Performance admission refused ({type(error).__name__}: {error}); stored finite feedback selected",
                refine_resources=refine_fallback_resources)
            self._pending = (fallback, self.active_fallback, None)
            return fallback

    def performance_fuel_charge(self,key,box,q,h,controller):
        """Same proved held debit for prospective search and final admission."""
        b=controller.certificate.spec
        if self.performance_resource_bound is None:
            if not controller.certificate.input_box.contains(box):
                raise ValueError("A wider performance command needs its own physical force/fuel proof")
            return h*b.fuel_rate_upper_per_q
        resource=self.performance_resource_bound(key,box,q,h)
        if (not isinstance(resource,CommandResourceBound) or resource.start!=q or resource.hold_width!=h
                or resource.applied_input!=box or resource.fuel_domain_id!=b.context.fuel_domain_id
                or resource.force_margin_n<0 or resource.fuel_upper_kg<0):
            raise ValueError("Performance command resource proof is stale or exceeds installed force authority")
        return resource.fuel_upper_kg

    def _admit(self, proposal, observations, q, h, target_flow_id):
        if not isinstance(proposal, PerformanceProposal) or proposal.start != q or proposal.hold_width != h:
            raise ValueError("Performance proposal is stale or has a different hold schedule")
        _identity(target_flow_id, "current trusted target-flow fingerprint")
        current, successor = self.active_fallback, proposal.successor
        successor.verify_integrity()
        if (not successor.certificate.admissible or successor.ids != current.ids
                or successor.certificate.start != q+h):
            raise ValueError("No complete admitted successor at the exact performance endpoint")
        if (successor.geometry_oracle.body_radii != current.geometry_oracle.body_radii
                or successor.geometry_oracle.component_ids != current.geometry_oracle.component_ids
                or successor.geometry_oracle.clearance < current.geometry_oracle.clearance
                or successor.geometry_oracle.minimum_pair_center_distance < current.geometry_oracle.minimum_pair_center_distance):
            raise ValueError("Successor changed physical bodies/components or weakened the safety clearance")
        _require_policy_supports(current, successor, q, self.acceleration_disturbance)
        segment, next_segment = current.segment_at(q, h), successor.segment_at(q+h)
        inputs = dict(proposal.applied_inputs)
        observed, contexts = dict(proposal.observation_ids), dict(proposal.context_ids)
        if any(set(value) != set(current.ids) for value in (inputs, observed, contexts)):
            raise ValueError("Proposal omitted a fleet input/observation/context")
        flows, commands, charge = {}, [], []
        for key, controller in segment.controllers:
            b, next_b = controller.certificate.spec, dict(next_segment.controllers)[key].certificate.spec
            if not _same_plant(b.context, next_b.context):
                raise ValueError("Successor changed actual model, geometry, sensing, actuator or fuel context")
            if observed[key] != observations[key].fingerprint or contexts[key] != b.context.fingerprint:
                raise ValueError("Proposal used a stale observation or controller context")
            box = inputs[key]
            if not b.input_limits.contains(box):
                raise ValueError("Performance input exceeds the hardware box")
            midpoint = tuple((lo+hi)/2 for lo, hi in zip(box.lower, box.upper))
            executable = tuple(_exact(float(value)) for value in midpoint)
            if any(abs(value-exact) > limit for value, exact, limit in zip(executable, midpoint, b.command_rounding_abs)):
                raise ValueError("Performance conversion exceeds its rounding budget")
            if not box.contains(IntervalBox.from_center_radius(executable, b.actuator_error_abs)):
                raise ValueError("Performance input box omits physical actuator error")
            initial = controller.observation_box(observations[key], q)
            request = HeldInputRequest(initial, box, (q, q), (h, h), self.acceleration_disturbance)
            flow = self.flow_provider(key, request)
            if (not isinstance(flow, FlowCertificate) or not flow.proven or flow.request != request.as_dict()
                    or flow.initial_state != initial or flow.start_anomaly != (q, q) or flow.hold_width != (h, h)
                    or flow.frame != b.context.frame or flow.model != b.context.model_id):
                raise ValueError("Performance flow is unresolved/incomplete or bound to another request/model/frame")
            SweptCollisionGuard._check_flow_structure(flow)
            identity = self.flow_identity(flow)
            if (identity.get("plant_fingerprint") != b.context.parameter_set_id
                    or identity.get("target_model_fingerprint") != b.context.target_model_id
                    or identity.get("clock_fingerprint") != b.context.clock_id
                    or identity.get("target_flow_fingerprint") != target_flow_id):
                raise ValueError("Flow has stale physical model, target ephemeris or clock identities")
            flows[key] = flow
            commands.append((key, executable))
            cost=self.performance_fuel_charge(key,box,q,h,controller)
            charge.append((key, cost))
        geometry = current.geometry_oracle.flows(flows)
        if not geometry.admissible:
            raise ValueError("Complete component/peer swept separation not proved: " + "; ".join(geometry.reasons))
        endpoints = {key: flows[key].endpoint for key in current.ids}
        if not successor.contains(endpoints, q+h):
            entry, summary = {}, []
            for key, controller in next_segment.controllers:
                b, box = controller.certificate.spec, endpoints[key]
                reference = controller.reference_at(q+h)
                position_required, sliding_required = [], []
                for i in range(3):
                    elo = box.lower[i]-reference.position[i]
                    ehi = box.upper[i]-reference.position[i]
                    position_required.append(max(abs(elo), abs(ehi))+b.reference_error_abs[i])
                    slo = box.lower[i+3]-reference.velocity[i]+b.lambda_gain[i]*elo
                    shi = box.upper[i+3]-reference.velocity[i]+b.lambda_gain[i]*ehi
                    sliding_required.append(max(abs(slo), abs(shi))+b.reference_error_abs[i+3]
                                            +b.lambda_gain[i]*b.reference_error_abs[i])
                position_margin = tuple(a-r for a, r in zip(b.omega, position_required))
                sliding_margin = tuple(a-r for a, r in zip(b.phi, sliding_required))
                entry[key] = dict(endpoint_box=exact_record(box), reference=exact_record(reference),
                    omega=b.omega, phi=b.phi, lambda_gain=b.lambda_gain,
                    reference_error_abs=b.reference_error_abs,
                    position_required=tuple(position_required), sliding_required=tuple(sliding_required),
                    position_margin=position_margin, sliding_margin=sliding_margin)
                summary.append(f"{key} p_margin={tuple(float(v) for v in position_margin)} "
                               f"s_margin={tuple(float(v) for v in sliding_margin)}")
            raise PerformanceAdmissionRefusal(
                "Complete performance endpoint is outside successor fallback entry; " + "; ".join(summary),
                details=dict(endpoint_q=q+h, per_agent=entry,
                             display_scope="Float margins are display only; exact rational evidence is retained"))
        needed = successor.remaining_fuel(q+h)
        if any(dict(charge)[key]+needed[key] > self.remaining_fuel[key] for key in current.ids):
            raise ValueError("Performance leaves insufficient fuel for its complete successor")
        return HeldFleetDecision(q, h, "PERFORMANCE", "Joint swept safety and complete finite feedback successors proved",
            tuple(commands), tuple((key, inputs[key]) for key in current.ids), proposal.attempt,
            current.fingerprint, successor.fingerprint,
            tuple((key, c.certificate.spec.context.fingerprint) for key, c in segment.controllers),
            tuple((key, observations[key].fingerprint) for key in current.ids), tuple(charge),
            proposal.fingerprint, tuple(flows[key].fingerprint for key in current.ids), geometry), endpoints

    def commit(self, decision: HeldFleetDecision, observations, *, q):
        """Commit after applying the exact simultaneous whole-fleet hold.

        Impossible observations are assumption violations. They cannot grant a
        safety claim or a physical pause. No proof clock changes on rejection.
        """
        if self._pending is None or not isinstance(decision, HeldFleetDecision):
            raise ValueError("No admitted fleet decision is pending")
        expected, successor, endpoints = self._pending
        q = _exact(q)
        if decision != expected or q != decision.start+decision.hold_width:
            raise ValueError("Execution decision/clock differs from the pending proof")
        successor.verify_integrity()
        if set(observations) != set(successor.ids):
            raise ValueError("Endpoint observation fleet changed")
        contracted = {}
        for key, controller in successor.segment_at(q).controllers:
            box = controller.observation_box(observations[key], q)
            contracted[key] = _intersect(box, endpoints[key]) if endpoints is not None else box
        remaining = {key: self.remaining_fuel[key]-value for key, value in decision.charged_fuel}
        if any(value < 0 for value in remaining.values()):
            raise ValueError("Committed command exceeds reserved fuel")
        self.active_fallback, self.q, self.remaining_fuel = successor, q, remaining
        self._pending = None
        record = dict(decision=decision.as_dict(), exact_end=str(q),
            endpoint_observations={key: exact_record(value) for key, value in observations.items()},
            contracted_beliefs={key: exact_record(value) for key, value in contracted.items()},
            remaining_fuel=exact_record(remaining), remaining_safe_q=str(successor.certificate.end-q),
            status="FINITE_SAFE_WINDOW_ENDED" if q == successor.certificate.end else "COMMITTED")
        self.events.append(record)
        return record
