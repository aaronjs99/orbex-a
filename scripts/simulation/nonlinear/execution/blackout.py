"""Finite causal observation propagation during declared sensor blackouts."""
from __future__ import annotations

from orbexa.control.certification.reachability import IntervalBox, _exact
from orbexa.control.fallback.refuge import RefugeObservation

from orbexa.simulation.nonlinear.safety.drift import ZERO_NOMINAL_ID


class BlackoutObservationPropagator:
    """Finite sensor-blackout interval propagation, without a solver/flow service.

    Known applied controls plus globally proved actual-drift boxes propagate a
    valid state enclosure. Its midpoint becomes a virtual measurement only
    while the installed controller's sensing budget and blackout horizon hold.
    A horizon violation is an assumption failure, not an infinite-safe pause.
    """
    def __init__(self, observations, *, maximum_blackout_q):
        self.boxes = {key:value.state_enclosure for key,value in observations.items()}
        self.q = next(iter(observations.values())).q
        if any(value.q != self.q for value in observations.values()):
            raise ValueError("Blackout observer fleet must share one sample clock")
        self.last_sensor_q = self.q
        self.maximum_blackout_q = _exact(maximum_blackout_q)
        if self.maximum_blackout_q <= 0:
            raise ValueError("Declare a positive finite blackout horizon")

    def advance(self,decision,plan):
        if decision.source not in ("FALLBACK_TRACK","FALLBACK_HOLD"):
            raise ValueError("Performance navigation requires its actual flow proof; this propagator retains fallback invariance only")
        if decision.start != self.q:
            raise ValueError("Blackout propagation clock is noncontiguous")
        h = decision.hold_width
        if self.q+h-self.last_sensor_q > self.maximum_blackout_q:
            raise ValueError("Declared sensor-blackout horizon exhausted")
        segment,inputs = plan.segment_at(self.q,h),dict(decision.applied_inputs)
        updated,observations = {},{}
        for key,controller in segment.controllers:
            b,box = controller.certificate.spec,self.boxes[key]
            if b.context.nominal_model_id != ZERO_NOMINAL_ID:
                raise ValueError("Blackout acceleration envelope requires the actual-zero-nominal drift adapter")
            drift = controller.certificate.drift_bounds.mismatch_abs
            applied = inputs[key]
            al = tuple(u-w for u,w in zip(applied.lower,drift))
            ah = tuple(u+w for u,w in zip(applied.upper,drift))
            lo = tuple(p+h*v+h*h*a/2 for p,v,a in zip(box.lower[:3],box.lower[3:],al))+tuple(v+h*a for v,a in zip(box.lower[3:],al))
            hi = tuple(p+h*v+h*h*a/2 for p,v,a in zip(box.upper[:3],box.upper[3:],ah))+tuple(v+h*a for v,a in zip(box.upper[3:],ah))
            result = IntervalBox(lo,hi)
            measured = tuple((a+c)/2 for a,c in zip(lo,hi))
            observation = RefugeObservation(self.q+h,measured,result,b.context.sensor_id,b.context.frame,b.context.clock_id)
            # The next segment may change gains/tube, while its exact region
            # inclusion was already proved by the complete plan constructor.
            next_controller = dict(plan.segment_at(self.q+h).controllers)[key]
            next_controller.observation_box(observation,self.q+h)
            updated[key],observations[key] = result,observation
        self.boxes,self.q = updated,self.q+h
        return observations

    def correct(self,observations):
        if set(observations) != set(self.boxes) or any(value.q != self.q for value in observations.values()):
            raise ValueError("Recovered sensor data have a different fleet/clock")
        updated = {}
        for key,value in observations.items():
            old,new = self.boxes[key],value.state_enclosure
            lo,hi = tuple(max(a,b) for a,b in zip(old.lower,new.lower)),tuple(min(a,b) for a,b in zip(old.upper,new.upper))
            if any(a>b for a,b in zip(lo,hi)):
                raise ValueError("Recovered sensor contradicts propagated blackout enclosure")
            updated[key] = IntervalBox(lo,hi)
        self.boxes,self.last_sensor_q = updated,self.q
