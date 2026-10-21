"""Finite refuge construction, target preflight and bounded candidate search."""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from orbexa.control.certification.reachability import IntervalBox, _exact
from orbexa.control.fallback.refuge import (
    ControlContext,
    ReferenceBounds,
    RefugeTrackingSpec,
    RefugeBoundsBuilder,
    RefugeController,
    _vec,
    exact_record,
    fingerprint,
)
from orbexa.control.fallback.plan import FleetFallbackPlan, FleetFeedbackSegment
from orbexa.control.fallback.geometry import FleetGeometryOracle
from orbexa.control.fallback.fuel import FleetFuelBudget

from orbexa.simulation.nonlinear.safety.reference import C2RefugeReference
from orbexa.simulation.nonlinear.safety.drift import ZERO_NOMINAL_ID, NonlinearZeroDriftBounds
from orbexa.simulation.nonlinear.safety.resources import StudyPropulsionBounds


class RefugeTargetGeometryRefusal(ValueError):
    """A decisive target-component refusal before controller construction.

    The copied exact evidence only refuses a proposal. It never substitutes
    for a controller, complete peer proof, resource check or admission.
    """
    def __init__(self, message, evidence):
        super().__init__(message)
        value = exact_record(evidence)
        self.refusal_evidence = dict(identity=fingerprint(value), **value)


def _target_preflight_indices(count, budget):
    """A bounded selection of original whole windows; no admission shortcut."""
    indices, seen, maximum = [], set(), min(count,budget)
    def add(numerator,denominator):
        value = min(count-1,count*numerator//denominator)
        if value not in seen and len(indices) < maximum:
            seen.add(value)
            indices.append(value)
    for numerator,denominator in ((0,1),(1,1),(1,2),(1,3),(2,3)):
        add(numerator,denominator)
    level = 2
    while len(indices) < maximum:
        denominator = 1 << level
        for numerator in range(1,denominator,2):
            add(numerator,denominator)
            if len(indices) == maximum:
                break
        level += 1
    return tuple(indices)


def _refuse_target_geometry(references, contexts, geometry, omega, windows, *,
                            checked_controller=None,window_budget=16):
    """Run the final checker's target calculation on the same proposal boxes.

    Independent-box peer failures are deliberately ignored: the actual final
    checker can recover their same-time polynomial correlation. Passing this
    preflight grants no authority and the original construction still runs.
    """
    if type(geometry) is not FleetGeometryOracle:
        raise TypeError("Early target refusal requires the original complete geometry oracle")
    indices = _target_preflight_indices(len(windows),window_budget)
    for checked, index in enumerate(indices):
        start,end = windows[index]
        bounds, boxes = {}, {}
        for key, reference in references.items():
            ctx = contexts[key]
            value = reference.bounds(start, end)
            if (not isinstance(value, ReferenceBounds) or value.start > start or value.end < end
                    or value.reference_id != ctx.reference_id
                    or value.frame != ctx.frame or value.clock_id != ctx.clock_id
                    or ctx.geometry_id != geometry.geometry_id
                    or ctx.target_model_id != geometry.target_model_id
                    or ctx.frame != geometry.frame or ctx.clock_id != geometry.clock_id):
                raise ValueError("Early target query has stale reference/geometry/frame/clock context")
            bounds[key] = value
            boxes[key] = IntervalBox(tuple(p-o for p,o in zip(value.position.lower,omega)),
                                    tuple(p+o for p,o in zip(value.position.upper,omega)))
        occupied = geometry._occupied(start, end)
        # Freeze this one genuine query for the existing final target formula.
        # No provider is replaced on the actual plan/oracle.
        probe = FleetGeometryOracle(lambda *_: occupied,
            component_ids=geometry.component_ids,body_radii=dict(geometry.body_radii),
            geometry_id=geometry.geometry_id,target_model_id=geometry.target_model_id,
            frame=geometry.frame,clock_id=geometry.clock_id,clearance=geometry.clearance,
            minimum_pair_center_distance=geometry.minimum_pair_center_distance)
        certificate = probe.regions(boxes,start,end,region_fingerprints=tuple(
            fingerprint(dict(reference_bounds=bounds[key],omega=omega)) for key in references))
        decisive = next((item for item in certificate.records
            if item.kind == 'target_component' and item.distance_squared_lower <= item.required_squared), None)
        if decisive is not None:
            reason = f"No strict whole-set separation for {decisive.first_id} and component {decisive.second_id}"
            raise RefugeTargetGeometryRefusal(reason,dict(
                kind='TARGET_COMPONENT_EARLY_REFUSAL',decisive_target_record=decisive,
                geometry_probe=certificate,occupied_bounds=occupied,
                reference_bounds=tuple(bounds.items()),position_boxes=tuple(boxes.items()),omega=omega,
                geometry_oracle_id=geometry.fingerprint,evaluated_target_windows=checked+1,
                selected_original_window_indices=indices,decisive_original_window_index=index,
                preflight_window_budget=window_budget,
                already_checked_controller=checked_controller,
                independent_box_peer_results='IGNORED; polynomial correlation is checked by the original final constructor',
                remaining_checks=('REMAINING_NONLINEAR_CONTROLLER_CERTIFICATES_UNEVALUATED',
                    'COMPLETE_ROUTE_INPUT_AND_FUEL_UNEVALUATED','CORRELATED_PEER_PROOFS_UNEVALUATED',
                    'INITIAL_ENTRY_UNEVALUATED','OTHER_TARGET_WINDOWS_UNEVALUATED_BY_PREFLIGHT')))


def construct_refuge_plan(*, references, contexts, drift_adapter: NonlinearZeroDriftBounds,
                          geometry_oracle: FleetGeometryOracle, propulsion: StudyPropulsionBounds,
                          omega,phi,lambda_gain,beta,acceleration_bound,maximum_hold,input_limits,
                          sensing_error_abs,actuator_error_abs,command_rounding_abs,
                          maximum_window="0.002",maximum_windows=10000,
                          early_target_refusal=False,early_target_window_budget=16):
    """Build a real finite feedback route; mathematical/geometry refusal survives.

    All members share piece boundaries. A new route/reference is never admitted
    by a numerical predicted endpoint; caller must perform whole-box entry or
    the runtime's joint nonlinear successor transaction.
    """
    ids = tuple(references)
    if type(early_target_refusal) is not bool:
        raise TypeError("Early target refusal must be explicitly enabled or disabled")
    if early_target_refusal and (type(early_target_window_budget) is not int or early_target_window_budget < 1):
        raise ValueError("Declare a positive finite target-preflight window budget")
    if ids != geometry_oracle.ids or set(contexts) != set(ids):
        raise ValueError("Reference, geometry and model-context fleets differ")
    windows = references[ids[0]].windows(maximum_window,maximum_windows)
    if any(references[key].windows(maximum_window,maximum_windows) != windows for key in ids):
        raise ValueError("Constructive fleet routes require common finite reference windows")
    segments = []
    omega,phi = _vec(omega,3,"omega",positive=True),_vec(phi,3,"phi",positive=True)
    lam,force = _vec(lambda_gain,3,"lambda",positive=True),_vec(beta,3,"beta",positive=True)
    errors = _vec(sensing_error_abs,6,"sensor envelope",nonnegative=True)
    actuator = _vec(actuator_error_abs,3,"actuator error",nonnegative=True)
    rounding = _vec(command_rounding_abs,3,"command rounding",nonnegative=True)
    target_preflight_done = False
    for start,end in windows:
        controllers = {}
        for key in ids:
            reference,ctx = references[key],contexts[key]
            if (ctx.reference_id != reference.reference_id or ctx.fuel_domain_id != propulsion.fingerprint):
                raise ValueError("Constructive refuge reference/fuel identities differ from the requested context")
            bounds = reference.bounds(start,end)
            ve = tuple(p+l*o for p,l,o in zip(phi,lam,omega))
            pad = tuple(l*(v+e)+b+a+r for l,v,e,b,a,r in zip(lam,ve,errors[3:],force,actuator,rounding))
            command_range = IntervalBox(tuple(a-p for a,p in zip(bounds.acceleration.lower,pad)),
                                       tuple(a+p for a,p in zip(bounds.acceleration.upper,pad)))
            resources = propulsion.input_resource(key,command_range,start,end-start,drift_adapter.clock)
            if resources.force_margin_n < 0:
                raise ValueError(f"{key}: finite feedback/reference range exceeds installed physical study force")
            spec = RefugeTrackingSpec(start,end,ctx,omega,phi,lam,force,maximum_hold,acceleration_bound,
                input_limits,errors,actuator,rounding,(0,)*9,resources.fuel_upper_kg/(end-start),
                propulsion.available_fuel_kg,resources.proof_reference,
                (propulsion.declaration,"Mass upper bound and Isp lower bound remain valid throughout the finite route",
                 "Unknown proper body rotation uses the outward sqrt(3 sum max|u_i|^2) bound, capped by 3 max|u|, with explicit attitude allocation"))
            builder = RefugeBoundsBuilder(spec,reference,drift_adapter,drift_adapter.nominal_drift,
                                           nominal_model_id=ZERO_NOMINAL_ID)
            controllers[key] = RefugeController(builder)
            if early_target_refusal and not target_preflight_done:
                # Preserve the existing cheap first-controller refusal before
                # spending work on occupied geometry for an unusable profile.
                _refuse_target_geometry(references,contexts,geometry_oracle,omega,windows,
                    checked_controller=controllers[key].certificate,window_budget=early_target_window_budget)
                target_preflight_done = True
        segments.append(FleetFeedbackSegment(controllers))
    return FleetFallbackPlan(tuple(segments),geometry_oracle)


@dataclass(frozen=True)
class PatchRefugeCandidateAttempt:
    retreat_duration: Fraction
    holding_duration: Fraction
    tangent_offset: tuple[Fraction,Fraction]
    status: str
    reasons: tuple[str,...]
    plan_id: str | None = None


@dataclass(frozen=True)
class PatchRefugeSearchResult:
    plan: FleetFallbackPlan | None
    references: tuple[tuple[str,C2RefugeReference],...]
    contexts: tuple[tuple[str,ControlContext],...]
    attempts: tuple[PatchRefugeCandidateAttempt,...]

    @property
    def admissible(self):
        return self.plan is not None and self.plan.certificate.admissible

    @property
    def reference_map(self):
        return dict(self.references)

    @property
    def context_map(self):
        return dict(self.contexts)

    @property
    def early_refusals(self):
        """Copied refusal-only diagnostics, outside admitted identity fields."""
        return exact_record(getattr(self,"_early_refusals",()))


def _patch_search_result(plan, references, contexts, attempts, early_refusals):
    value = PatchRefugeSearchResult(plan,references,contexts,tuple(attempts))
    if early_refusals:
        object.__setattr__(value,"_early_refusals",tuple(early_refusals))
    return value


def construct_patch_refuge_candidates(*, initial_states, initial_enclosures,
        outward_directions, tangent_u, tangent_v, start, contexts,
        drift_adapter: NonlinearZeroDriftBounds, geometry_oracle: FleetGeometryOracle,
        propulsion: StudyPropulsionBounds, retreat_durations=("0.015","0.02"),
        holding_duration="0.003", outward_distance=2,
        tangent_offsets=(("0.5",0),(0,0),("-0.5",0),(0,"0.5"),(0,"-0.5"),
                         ("0.5","0.5"),("0.5","-0.5"),("-0.5","0.5"),("-0.5","-0.5")),
        maximum_candidates=18, fuel_budget=None, **tracking_options):
    """Prepare a finite jointly proved patch-to-refuge route, or retain refusal.

    All agents receive the same fixed reference-frame tangent displacement.
    Supplied outward vectors, initial jets and intervals are exact inputs;
    their interpretation as measured patch normals is a caller assumption.
    No numerical optimizer or truth pose is used. Every attempted complete
    route still needs the nonlinear invariant inequalities, all occupied
    components, peer separation, terminal holding, fuel and initial-box entry.
    This construction runs before installation, never at an execution failure.
    """
    ids = geometry_oracle.ids
    if (set(initial_states) != set(ids) or set(initial_enclosures) != set(ids)
            or set(outward_directions) != set(ids) or set(contexts) != set(ids)):
        raise ValueError("Patch candidate states, enclosures, directions, contexts and geometry fleets differ")
    if not isinstance(maximum_candidates,int) or isinstance(maximum_candidates,bool) or maximum_candidates < 1:
        raise ValueError("Declare a positive finite patch candidate budget")
    q,hold,distance = map(_exact,(start,holding_duration,outward_distance))
    if fuel_budget is not None:
        if type(fuel_budget) is not FleetFuelBudget:
            raise TypeError('Patch continuation search needs an exact FleetFuelBudget')
        fuel_budget.check_context(ids,q,propulsion.fingerprint)
    durations = tuple(_exact(value) for value in retreat_durations)
    offsets = tuple(_vec(value,2,"common tangent offset") for value in tangent_offsets)
    if not durations or not offsets or hold <= 0 or distance <= 0 or any(value <= 0 for value in durations):
        raise ValueError("Positive finite retreat, holding and outward-distance domains are required")
    u,v = _vec(tangent_u,3,"common tangent u"),_vec(tangent_v,3,"common tangent v")
    if not any(u) or not any(v) or not any(u[i]*v[j]-u[j]*v[i] for i in range(3) for j in range(i)):
        raise ValueError("Common patch tangent vectors must be nonzero and independent")
    states,directions = {},{}
    for key in ids:
        states[key] = _vec(initial_states[key],6,key+" reference initial jet")
        directions[key] = _vec(outward_directions[key],3,key+" fixed outward vector")
        box = initial_enclosures[key]
        if (not any(directions[key]) or not isinstance(box,IntervalBox) or box.dimension != 6
                or not box.contains(IntervalBox.point(states[key])) or not isinstance(contexts[key],ControlContext)):
            raise ValueError(key+": a nonzero outward vector and complete initial state enclosure are required")
        ctx = contexts[key]
        if ctx.frame != geometry_oracle.frame or ctx.clock_id != geometry_oracle.clock_id:
            raise ValueError(key+": candidate reference frame and clock differ from the fleet geometry")
    attempts,early_refusals = [],[]
    for duration in durations:
        for offset in offsets:
            if len(attempts) >= maximum_candidates:
                return _patch_search_result(None,(),(),attempts,early_refusals)
            references,rebound = {},{}
            try:
                displacement = tuple(offset[0]*a+offset[1]*b for a,b in zip(u,v))
                for key in ids:
                    ctx,state,direction = contexts[key],states[key],directions[key]
                    goal = tuple(p+distance*n+d for p,n,d in zip(state[:3],direction,displacement))
                    reference = C2RefugeReference.retreat(start=q,retreat_end=q+duration,hold_end=q+duration+hold,
                        initial_state=state,refuge_position=goal,frame=ctx.frame,clock_id=ctx.clock_id)
                    references[key] = reference
                    rebound[key] = replace(ctx,reference_id=reference.reference_id)
                plan = construct_refuge_plan(references=references,contexts=rebound,drift_adapter=drift_adapter,
                    geometry_oracle=geometry_oracle,propulsion=propulsion,**tracking_options)
                reasons = tuple(dict.fromkeys(plan.certificate.reasons))
                if not reasons and not plan.contains(initial_enclosures,q):
                    reasons = ("Complete supplied initial state boxes do not enter every candidate feedback tube",)
                if not reasons and fuel_budget is not None:
                    try:fuel_budget.require(plan.remaining_fuel(q))
                    except ValueError as error:reasons=(str(error),)
                if reasons:
                    attempts.append(PatchRefugeCandidateAttempt(duration,hold,offset,"REFUSED",reasons))
                    continue
                attempts.append(PatchRefugeCandidateAttempt(duration,hold,offset,"CONDITIONAL_PROVEN",(),plan.fingerprint))
                return _patch_search_result(plan,tuple(references.items()),tuple(rebound.items()),attempts,early_refusals)
            except Exception as error:
                if isinstance(error,RefugeTargetGeometryRefusal):
                    early_refusals.append(error.refusal_evidence)
                attempts.append(PatchRefugeCandidateAttempt(duration,hold,offset,"REFUSED",
                    (type(error).__name__+": "+str(error),)))
    return _patch_search_result(None,(),(),attempts,early_refusals)
