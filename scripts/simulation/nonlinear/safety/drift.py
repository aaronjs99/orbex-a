"""Actual nonlinear drift enclosures for the exact-zero nominal refuge."""
from __future__ import annotations

from collections import OrderedDict
from fractions import Fraction
from orbexa.control.certification.reachability import IntervalBox, _exact, _ARB_LOCK
from orbexa.control.fallback.refuge import (
    NonlinearDriftBounds,
    _vec,
    _absbox,
    exact_record,
    fingerprint,
)
from orbexa.control.certification.nonlinear import (
    _Arithmetic,
    _region,
    _target_box,
    TargetEphemerisCertificate,
)
from orbexa.core.dynamics.relative import relative_field, parameters

from orbexa.simulation.nonlinear.safety.cache import _FrozenProofIdentityCache


ZERO_NOMINAL_ID = "exact-zero-drift-refuge-v1"


class NonlinearZeroDriftBounds:
    """Conservative actual nonlinear acceleration envelope, with F_nom=0.

    Zero nominal drift has exact zero Jacobians/evaluation error. Actual force,
    J2, vector drag, frame rotation and clock terms are enclosed with Arb.
    target_provider(start,end) must return a validated full target ephemeris,
    never a reset numerical pose. Limits and source identities are checked.
    """
    def __init__(self, *, model, clock, target_provider, maximum_width="0.0001",
                 maximum_pieces=10000, precision_bits=128, extra_acceleration_abs=(0,0,0)):
        self.model,self.clock,self.target_provider = model,clock,target_provider
        self.maximum_width = _exact(maximum_width)
        self.maximum_pieces,self.precision_bits = maximum_pieces,precision_bits
        self.extra_acceleration_abs = _vec(extra_acceleration_abs,3,"extra acceleration",nonnegative=True)
        self._identities = _FrozenProofIdentityCache()
        self._target_boxes = OrderedDict()
        if self.maximum_width <= 0 or maximum_pieces < 1 or precision_bits < 64:
            raise ValueError("Finite nonlinear bound budgets are required")

    @staticmethod
    def nominal_drift(q, measurement):
        return (Fraction(0),)*3

    def verify_certificate_source(self, certificate, required_extra):
        """Authenticate existing drift/support provenance without a new flow.

        A mutable provider tag cannot enlarge a previously proved disturbance
        budget. Recompute the original target/model/clock/region/scope/extra/
        precision digest and require its exact retained proof reference.
        """
        from orbexa.control.fallback.refuge import RefugeCertificate
        if type(self) is not NonlinearZeroDriftBounds or not isinstance(certificate, RefugeCertificate) or not certificate.admissible:
            raise ValueError('A source-backed nonlinear refuge certificate is required')
        required = _vec(required_extra, 3, 'required extra acceleration', nonnegative=True)
        extra = _vec(self.extra_acceleration_abs, 3, 'declared extra acceleration', nonnegative=True)
        if any(need > bound for need, bound in zip(required, extra)):
            raise ValueError('Existing nonlinear source does not cover the required extra acceleration')
        spec, ctx = certificate.spec, certificate.spec.context
        if (ctx.model_id != 'nonlinear-reference-relative-v1' or ctx.nominal_model_id != ZERO_NOMINAL_ID
                or ctx.parameter_set_id != self.model.fingerprint
                or ctx.target_model_id != self.model.target_fingerprint
                or ctx.clock_id != self.clock.fingerprint or ctx.frame != 'reference-orbit-frame-v1'):
            raise ValueError('Existing nonlinear drift certificate changed model/frame/clock context')
        target = self.target_provider(spec.start, spec.end)
        if (not isinstance(target, TargetEphemerisCertificate) or not target.proven
                or target.model_bounds.get('target_model_fingerprint') != self.model.target_fingerprint
                or target.model_bounds.get('clock_fingerprint') != self.clock.fingerprint):
            raise ValueError('Existing nonlinear drift source lacks a complete same-model target proof')
        source = fingerprint(dict(target=self._identities.identity(target), model=self.model.fingerprint,
            clock=self.clock.fingerprint, region=exact_record(certificate.state_box),
            scope=(str(spec.start), str(spec.end)), extra=exact_record(extra), precision=self.precision_bits))
        if certificate.drift_bounds.proof_reference != 'Arb-actual-nonlinear-drift:'+source:
            raise ValueError('Existing nonlinear drift/source declaration changed after proof')
        return True

    def __call__(self, spec, reference, state_box, sensing_box):
        import flint
        ctx,model,clock = spec.context,self.model,self.clock
        if (ctx.model_id != "nonlinear-reference-relative-v1" or ctx.nominal_model_id != ZERO_NOMINAL_ID
                or ctx.parameter_set_id != model.fingerprint or ctx.target_model_id != model.target_fingerprint
                or ctx.frame != "reference-orbit-frame-v1" or ctx.clock_id != clock.fingerprint):
            raise ValueError("Nonlinear refuge model/reference-clock identities differ")
        _region(state_box,model)
        clock.check_domain((spec.start,spec.end))
        target = self.target_provider(spec.start,spec.end)
        if (not isinstance(target,TargetEphemerisCertificate) or not target.proven
                or target.model_bounds.get("target_model_fingerprint") != model.target_fingerprint
                or target.model_bounds.get("clock_fingerprint") != clock.fingerprint):
            raise ValueError("A proved same-model/clock target ephemeris is required")
        target_id = self._identities.identity(target)
        ratio = (spec.end-spec.start)/self.maximum_width
        count = max(1,(ratio.numerator+ratio.denominator-1)//ratio.denominator)
        if count > self.maximum_pieces:
            raise ValueError("Finite nonlinear drift-bound budget exhausted")
        step,result = (spec.end-spec.start)/count,None
        with _ARB_LOCK, flint.ctx.workprec(self.precision_bits):
            ar = _Arithmetic(flint)
            par,state,zero = parameters(model,ar),ar.vector(state_box),(ar.scalar(0),)*3
            for index in range(count):
                interval = (spec.start+index*step,spec.start+(index+1)*step)
                cache_key = (target_id,interval)
                target_box = self._target_boxes.get(cache_key)
                if target_box is None:
                    target_box = _target_box(target,interval)
                    self._target_boxes[cache_key] = target_box
                    while len(self._target_boxes) > 1024:
                        self._target_boxes.popitem(last=False)
                value = relative_field(ar.interval(interval),state,ar.vector(target_box),
                    zero,model,clock,ar,par)[3:]
                local = ar.box(value)
                result = local if result is None else result.hull(local)
        mismatch = tuple(v+e for v,e in zip(_absbox(result),self.extra_acceleration_abs))
        source = fingerprint(dict(target=target_id, model=model.fingerprint, clock=clock.fingerprint,
            region=exact_record(state_box), scope=(str(spec.start),str(spec.end)),
            extra=exact_record(self.extra_acceleration_abs), precision=self.precision_bits))
        return NonlinearDriftBounds(spec.start,spec.end,state_box,sensing_box,spec.input_limits,
            IntervalBox.point((0,0,0)),mismatch,((0,0,0),)*3,((0,0,0),)*3,(0,0,0),(0,0,0),
            ctx.model_id,ctx.nominal_model_id,ctx.parameter_set_id,ctx.target_model_id,ctx.fuel_domain_id,
            ctx.frame,ctx.clock_id,"Arb-actual-nonlinear-drift:"+source,
            ("Declared full nonlinear gravity/J2/vector drag and fixed commanded reference clock",
             "Validated target ephemeris continuous coverage; no numerical target resets",
             "Exact zero nominal drift and global regional actual-drift forcing enclosure",
             "Explicit additional bounded anomaly-acceleration disturbance"))
