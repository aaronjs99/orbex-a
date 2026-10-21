"""Detached finite-backup renewal with complete new nonlinear proofs.

The supplied task contains a genuine declared navigation target/time prior and
mathematical policy data only.  It contains no actuator, random generator,
truth state, or owning execution runtime.  Holding extension preserves the
original route, then proves a longer final hold.  It is a fallback renewal,
not a performance/arrival planner or an infinite-safety construction.
"""
from __future__ import annotations

from dataclasses import dataclass,replace
from fractions import Fraction
import pickle

from orbexa.control.fallback.runtime import FleetAdmissionRuntime
from orbexa.control.fallback.plan import FleetFallbackPlan
from orbexa.control.fallback.replacement import prepare_fleet_replacement
from orbexa.control.certification.reachability import IntervalBox, _exact
from orbexa.control.fallback.refuge import fingerprint
from orbexa.estimation.navigation.records import NavigationEstimate
from orbexa.simulation.nonlinear.safety.reference import C2RefugeReference, PolynomialReferencePiece
from orbexa.simulation.nonlinear.safety.drift import NonlinearZeroDriftBounds
from orbexa.simulation.nonlinear.safety.geometry import AnyProperTargetPose
from orbexa.simulation.nonlinear.safety.candidates import construct_refuge_plan
from orbexa.simulation.nonlinear.safety.providers import FixedProofProvider


class _PreparationRuntime(FleetAdmissionRuntime):
    """Algebra-only view; deliberately has no authority to decide or commit."""
    def __init__(self,snapshot):
        snapshot.plan.verify_integrity()
        if (not snapshot.plan.certificate.admissible
                or not snapshot.plan.certificate.start <= snapshot.q < snapshot.plan.certificate.end
                or len(snapshot.remaining_fuel) != len(snapshot.plan.ids)
                or set(dict(snapshot.remaining_fuel)) != set(snapshot.plan.ids)
                or any(value < 0 for _,value in snapshot.remaining_fuel)):
            raise ValueError('Detached preparation snapshot lacks its complete finite plan/clock/resources')
        required=snapshot.plan.remaining_fuel(snapshot.q)
        if any(dict(snapshot.remaining_fuel)[key] < amount for key,amount in required.items()):
            raise ValueError('Detached preparation snapshot cannot afford its installed finite policy')
        self.active_fallback,self.q=snapshot.plan,snapshot.q
        self.remaining_fuel=dict(snapshot.remaining_fuel)
        self.acceleration_disturbance=snapshot.acceleration_disturbance
        self._pending=None

    def decide(self,*args,**kwargs):
        raise TypeError('A detached preparation view cannot issue an actuation decision')

    def commit(self,*args,**kwargs):
        raise TypeError('A detached preparation view cannot commit physical motion or fuel')


@dataclass(frozen=True)
class FeedbackPreparationSnapshot:
    plan: FleetFallbackPlan
    q: Fraction
    remaining_fuel: tuple
    acceleration_disturbance: IntervalBox
    source_id: str

    def __post_init__(self):
        object.__setattr__(self,'q',_exact(self.q))
        object.__setattr__(self,'remaining_fuel',tuple((key,_exact(value)) for key,value in self.remaining_fuel))
        if (not isinstance(self.plan,FleetFallbackPlan) or not isinstance(self.source_id,str) or not self.source_id
                or not isinstance(self.acceleration_disturbance,IntervalBox)
                or self.acceleration_disturbance.dimension != 3):
            raise ValueError('Bind a detached complete fleet policy and disturbance to an exact source identity')

    @classmethod
    def from_runtime(cls,runtime,*,source_id):
        if not isinstance(runtime,FleetAdmissionRuntime) or runtime._pending is not None:
            raise TypeError('Take a preparation snapshot only at an idle committed runtime boundary')
        return cls(runtime.active_fallback,runtime.q,tuple(runtime.remaining_fuel.items()),
                   runtime.acceleration_disturbance,source_id)

    @property
    def context_id(self):
        return fingerprint(dict(source_id=self.source_id,active_fallback_id=self.plan.fingerprint,
            acceleration_disturbance=self.acceleration_disturbance))

    def runtime_view(self):
        return _PreparationRuntime(self)


@dataclass(frozen=True)
class FeedbackHoldExtensionTask:
    snapshot: FeedbackPreparationSnapshot
    source_navigation: NavigationEstimate
    model: object
    clock: object
    propulsion: object
    enclosure_config: object
    extension_width: Fraction

    def __post_init__(self):
        object.__setattr__(self,'extension_width',_exact(self.extension_width))
        nav=self.source_navigation
        if not isinstance(self.snapshot,FeedbackPreparationSnapshot) or not isinstance(nav,NavigationEstimate):
            raise ValueError('Use a typed detached feedback snapshot and declared navigation prior')
        sensor_ids={controller.certificate.spec.context.sensor_id
            for _,controller in self.snapshot.plan.segments[0].controllers}
        if (nav.status != 'VALID' or _exact(nav.exact_event_q) > self.snapshot.q
                or nav.clock_fingerprint != self.clock.fingerprint or nav.model_fingerprint != self.model.fingerprint
                or nav.frame_id != self.snapshot.plan.geometry_oracle.frame
                or any(value not in sensor_ids for value in nav.sensor_fingerprints)
                or set(nav.relative_state_box_by_agent) != set(self.snapshot.plan.ids)
                or self.extension_width <= 0):
            raise ValueError('Hold renewal needs a genuine same-clock declared navigation prior and positive finite extension')
        # An initial declared prior may have no delivered packets. An older
        # navigation prior remains usable only through a newly proved
        # forecast over its entire age and the proposed future scope.  It is
        # never relabeled as a current acquisition or replaced by a truth point.
        self.clock.check_domain((_exact(nav.exact_event_q),self.snapshot.plan.certificate.end+self.extension_width))


def prepare_hold_extension(request):
    """Source-backed worker entry; rebuild every complete future proof."""
    task=pickle.loads(request.payload)
    if not isinstance(task,FeedbackHoldExtensionTask) or request.context_id != task.snapshot.context_id:
        raise TypeError('Use a typed same-source detached feedback holding-extension task')
    runtime=task.snapshot.runtime_view()
    activation,h=_exact(request.activation_q),_exact(request.hold_width)
    if not runtime.q <= activation < runtime.active_fallback.certificate.end:
        raise ValueError('The replacement must activate while the installed finite backup still covers motion')
    old=runtime.active_fallback
    maximum_window=max(segment.end-segment.start for segment in old.segments)
    if (task.extension_width/maximum_window).denominator != 1:
        raise ValueError('Declare the extension as an exact integer number of installed proof windows; decimal near-boundaries cannot split executed holds')
    # A local attitude proof may not be extrapolated.  General near-target
    # renewal needs a freshly extended attitude flow and its own factory.
    pose=getattr(old.geometry_oracle.occupied_provider,'pose_provider',None)
    if type(pose) is not AnyProperTargetPose:
        raise ValueError('This holding-extension factory requires the original analytical all-proper-attitude geometry envelope')
    controllers=dict(old.segments[0].controllers)
    first=next(iter(controllers.values())).certificate.spec
    fields=('omega','phi','lambda_gain','beta','maximum_hold','acceleration_bound','input_limits',
        'sensing_error_abs','actuator_error_abs','command_rounding_abs','reference_error_abs','available_fuel')
    if any(any(getattr(controller.certificate.spec,name) != getattr(controllers[key].certificate.spec,name)
                   for name in fields) for segment in old.segments for key,controller in segment.controllers):
        raise ValueError('This extension factory retains a uniform original tracking declaration')
    end=old.certificate.end+task.extension_width
    references,contexts={},{}
    for key,controller in controllers.items():
        original=controller.builder.reference_provider
        if type(original) is not C2RefugeReference or any(len(row) != 1 for row in original.pieces[-1].coefficients):
            raise ValueError('Extend only an exact C2 route with a genuinely constant terminal holding piece')
        last=original.pieces[-1]
        reference=C2RefugeReference((*original.pieces[:-1],PolynomialReferencePiece(last.start,end,last.coefficients)),
            frame=original.frame,clock_id=original.clock_id)
        references[key]=reference
        contexts[key]=replace(controller.certificate.spec.context,reference_id=reference.reference_id)
    from orbexa.control.certification.nonlinear import NonlinearFlowEncloser
    encloser=NonlinearFlowEncloser(task.enclosure_config)
    nav=task.source_navigation
    source_q=_exact(nav.exact_event_q)
    target=encloser.enclose_target(nav.target_ephemeris_box,(source_q,source_q),(end-source_q,)*2,
        task.model,task.clock,nav.physical_time_box)
    if not target.proven:
        raise ValueError('Extended nonlinear target forecast refused: '+target.reason)
    original=next(iter(controllers.values())).builder.drift_bounds_provider
    drift=NonlinearZeroDriftBounds(model=task.model,clock=task.clock,target_provider=FixedProofProvider(target),
        maximum_width=original.maximum_width,maximum_pieces=original.maximum_pieces,
        precision_bits=original.precision_bits,extra_acceleration_abs=original.extra_acceleration_abs)
    candidate=construct_refuge_plan(references=references,contexts=contexts,drift_adapter=drift,
        geometry_oracle=old.geometry_oracle,propulsion=task.propulsion,
        omega=first.omega,phi=first.phi,lambda_gain=first.lambda_gain,beta=first.beta,
        acceleration_bound=first.acceleration_bound,maximum_hold=first.maximum_hold,input_limits=first.input_limits,
        sensing_error_abs=first.sensing_error_abs,actuator_error_abs=first.actuator_error_abs,
        command_rounding_abs=first.command_rounding_abs,
        maximum_window=maximum_window)
    if not candidate.certificate.admissible or candidate.certificate.end <= old.certificate.end:
        raise ValueError('No complete genuinely extended feedback policy was proved')
    if any(((boundary-activation)/h).denominator != 1
            for segment in candidate.segments for boundary in (segment.start,segment.end)):
        raise ValueError('A regenerated proof boundary would split the declared exact execution hold schedule')
    return prepare_fleet_replacement(runtime,candidate.suffix(activation),hold_width=h)
