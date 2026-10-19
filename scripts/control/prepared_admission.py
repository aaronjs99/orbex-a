"""Prepare and independently activate one future certified feedback policy.

The complete old relational error tube must enter the new policy at its exact
activation clock. The final gate rechecks entry, physical supports and full
future fuel without a new nonlinear or geometry proof. No future measurement
or hard real-time guarantee is inferred.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction

from .fleet_fallback import (FleetAdmissionRuntime, FleetFallbackPlan,
    _same_plant, _require_policy_supports)
from .reachability import IntervalBox, _exact
from .refuge import fingerprint, exact_record


class AdmissionContextChanged(RuntimeError):
    """A proof callback changed authoritative execution state; do not continue."""


@dataclass(frozen=True)
class PreparedFleetReplacement:
    """Complete feedback policy with a proved relational future entry."""
    start: Fraction
    hold_width: Fraction
    parent_fallback_id: str
    acceleration_disturbance: IntervalBox
    successor: FleetFallbackPlan
    entry_proof: tuple
    required_fuel: tuple
    proof_identity: str

    def _record(self):
        return dict(start=self.start, hold_width=self.hold_width,
            parent_fallback_id=self.parent_fallback_id,
            acceleration_disturbance=self.acceleration_disturbance,
            successor_id=self.successor.fingerprint, entry_proof=self.entry_proof,
            required_fuel=self.required_fuel)

    def verify_integrity(self):
        if fingerprint(self._record()) != self.proof_identity:
            raise ValueError('Prepared feedback replacement changed after its entry proof')

    @property
    def fingerprint(self):
        self.verify_integrity()
        return self.proof_identity

    def as_dict(self):
        self.verify_integrity()
        return dict(schema='orbexa.prepared-fleet-replacement.v1',
            identity=self.proof_identity, **exact_record(self._record()),
            scope='Complete old relational tube enters a complete proved finite feedback policy at a future exact clock')


def prepare_fleet_replacement(runtime, candidate, *, hold_width):
    """Prove future entry in (position error, sliding error) coordinates.

    With exact reference differences dp=r_old-r_new, dv=v_old-v_new,
    e_new=e_old+dp and
    s_new=s_old+(Lambda_new-Lambda_old)e_old+dv+Lambda_new*dp.
    Componentwise interval bounds preserve this correlation. Reference sample
    errors are charged conservatively on both sides; no sensor acquisition at
    the future activation is invented. Every candidate's complete route,
    terminal holding, geometry, hardware and resource proof already exists.
    """
    if (not isinstance(runtime, FleetAdmissionRuntime) or runtime._pending is not None
            or not isinstance(candidate, FleetFallbackPlan) or not candidate.certificate.admissible):
        raise TypeError('Prepare from an idle detached runtime and a complete proved feedback policy')
    old = runtime.active_fallback
    old.verify_integrity(); candidate.verify_integrity()
    q, h = candidate.certificate.start, _exact(hold_width)
    if h <= 0 or not runtime.q <= q <= old.certificate.end or candidate.ids != old.ids:
        raise ValueError('Replacement must activate within the retained finite backup scope with the whole fleet')
    if (candidate.geometry_oracle.body_radii != old.geometry_oracle.body_radii
            or candidate.geometry_oracle.component_ids != old.geometry_oracle.component_ids
            or candidate.geometry_oracle.clearance < old.geometry_oracle.clearance
            or candidate.geometry_oracle.minimum_pair_center_distance < old.geometry_oracle.minimum_pair_center_distance):
        raise ValueError('Replacement changed physical bodies/components or weakened separation')
    _require_policy_supports(old, candidate, q, runtime.acceleration_disturbance)
    previous = dict(old.segment_at(q).controllers)
    entry = []
    for key, controller in candidate.segment_at(q, h).controllers:
        old_controller = previous[key]
        a, b = old_controller.certificate.spec, controller.certificate.spec
        if not _same_plant(a.context, b.context):
            raise ValueError('Replacement changed the actual plant, sensing, actuator, geometry or fuel context')
        left, right = old_controller.reference_at(q), controller.reference_at(q)
        dp = tuple(x-y for x, y in zip(left.position, right.position))
        dv = tuple(x-y for x, y in zip(left.velocity, right.velocity))
        position, sliding = [], []
        for i in range(3):
            # These conservative pads also cover nonzero declared reference
            # sample errors. Current exact polynomial providers have zero pads.
            old_e = a.omega[i]+a.reference_error_abs[i]
            old_s = a.phi[i]+a.reference_error_abs[i+3]+a.lambda_gain[i]*a.reference_error_abs[i]
            position.append(abs(dp[i])+old_e+b.reference_error_abs[i])
            sliding.append(old_s+abs(b.lambda_gain[i]-a.lambda_gain[i])*old_e+
                abs(dv[i]+b.lambda_gain[i]*dp[i])+b.reference_error_abs[i+3]+
                b.lambda_gain[i]*b.reference_error_abs[i])
        if any(need > limit for need, limit in zip(position, b.omega)) or any(
                need > limit for need, limit in zip(sliding, b.phi)):
            raise ValueError(key+': complete old relational tube does not enter replacement')
        entry.append((key, dict(old_controller_id=old_controller.certificate.fingerprint,
            new_controller_id=controller.certificate.fingerprint, old_reference=exact_record(left),
            new_reference=exact_record(right), old_omega=a.omega, old_phi=a.phi,
            old_lambda=a.lambda_gain, new_omega=b.omega, new_phi=b.phi, new_lambda=b.lambda_gain,
            old_reference_error_abs=a.reference_error_abs, new_reference_error_abs=b.reference_error_abs,
            position_required=tuple(position), sliding_required=tuple(sliding),
            method='exact-relational-error-and-sliding-entry-v1')))
    required = tuple(candidate.remaining_fuel(q).items())
    if any(runtime.remaining_fuel[key] < need for key, need in required):
        raise ValueError('Preparation snapshot cannot cover the complete replacement fuel reserve')
    # Freeze nested proof records into immutable tuples before publication.
    from .fleet_fallback import _freeze
    value = PreparedFleetReplacement(q, h, old.fingerprint, runtime.acceleration_disturbance,
        candidate, tuple((key, _freeze(record)) for key, record in entry), required, '')
    return replace(value, proof_identity=fingerprint(value._record()))


def activate_fleet_replacement(runtime, prepared, observations, *, q, hold_width):
    """Install a timely policy only after current support/context/fuel checks."""
    q, h = _exact(q), _exact(hold_width)
    if not isinstance(runtime, FleetAdmissionRuntime) or not isinstance(prepared, PreparedFleetReplacement):
        raise TypeError('Use the owning runtime and a typed prepared feedback replacement')
    if runtime._pending is not None or runtime.q != q or prepared.start != q or prepared.hold_width != h:
        raise ValueError('Feedback replacement is stale, early or on another hold schedule')
    old = runtime.active_fallback
    def state():
        return (id(runtime.active_fallback), runtime.active_fallback.fingerprint,
            runtime.q, dict(runtime.remaining_fuel), runtime._pending, runtime.acceleration_disturbance)
    original = state()
    try:
        prepared.verify_integrity()
        if (old.fingerprint != prepared.parent_fallback_id or
                runtime.acceleration_disturbance != prepared.acceleration_disturbance or
                set(observations) != set(old.ids)):
            raise ValueError('Feedback replacement has another active backup, disturbance or fleet')
        # Independently rederive exact entry and full future resources. A
        # producer's freely constructible receipt/hash grants no admission.
        checked = prepare_fleet_replacement(runtime, prepared.successor, hold_width=h)
        if checked.fingerprint != prepared.fingerprint:
            raise ValueError('Prepared feedback receipt differs from independently checked current admission')
        previous = dict(old.segment_at(q).controllers)
        for key, controller in prepared.successor.segment_at(q, h).controllers:
            previous[key].observation_box(observations[key], q)
            controller.observation_box(observations[key], q)
        record = checked.as_dict()
    finally:
        try:
            unchanged = state() == original
        except Exception:
            unchanged = False
        if not unchanged:
            raise AdmissionContextChanged('Feedback admission callback changed authoritative execution state')
    # The old relational invariant, not rectangular sensor-box membership,
    # grants entry. No fresh nonlinear/geometry proof or future sensing is used.
    runtime.active_fallback = prepared.successor
    return record
