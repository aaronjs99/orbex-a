"""Conservative physical resource accounting from authenticated nonlinear holds."""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from orbexa.core.reference_frame import exact
from orbexa.simulation.assembly import STANDARD_GRAVITY
from orbexa.simulation.complete_mission import ApproachPropellantBudget, _outward_float
from orbexa.simulation.mission_checkpoint import plain
from orbexa.simulation.nonlinear_archive import NonlinearHoldArchive
from orbexa.simulation.nonlinear_completion_contracts import _dyadic_upper, _require, _sqrt_upper
from orbexa.simulation.nonlinear_completion_journal import (NonlinearJournalBinding,
    bind_nonlinear_journal, verified_nonlinear_holds, verify_nonlinear_timeline)


@dataclass(frozen=True)
class NonlinearPropellantBudget(ApproachPropellantBudget):
    held_accounting: dict = field(default_factory=dict)

    def to_dict(self):
        record = super().to_dict()
        record.update(method='source-bound-nonlinear-admitted-hold-propellant-v1',
            scope='Conservative upper debit from every authenticated nonlinear applied-input support and exact reference-clock hold; not measured fuel',
            held_input_accounting=plain(self.held_accounting))
        return record



class NonlinearTranslationAccountant:
    """Convert a source-bound full journal into conservative force and fuel debits."""
    def __init__(self, approach, *, model, clock, hold_records,
                 expected_actuator_id=None, expected_sensor_id=None):
        binding = bind_nonlinear_journal(approach, model=model, clock=clock, hold_records=hold_records,
            expected_actuator_id=expected_actuator_id, expected_sensor_id=expected_sensor_id)
        self.model = binding.model
        self.clock = binding.clock
        self.records = binding.records
        self.source_hash = binding.source_hash
        self.journal_hash = binding.journal_hash
        self.actuator_id = binding.actuator_id
        self.sensor_id = binding.sensor_id

    def translation_budget(self, approach, *, options, attitude_tracker, terminal_handoff=None):
        _require(terminal_handoff is None, 'a CH terminal continuation cannot be mixed with this nonlinear journal')
        binding = NonlinearJournalBinding(self.model, self.clock, self.records, self.source_hash, self.journal_hash, self.actuator_id, self.sensor_id)
        timeline = verify_nonlinear_timeline(approach, binding=binding)
        ids, start, end = timeline.ids, timeline.start, timeline.end
        impulse, peak = dict.fromkeys(ids, Fraction(0)), dict.fromkeys(ids, Fraction(0))
        body_radii = {key: attitude_tracker.chasers.get(key, attitude_tracker.default_chaser).body_radius_m for key in ids}
        for hold in verified_nonlinear_holds(approach, binding=binding, timeline=timeline, body_radii=body_radii):
            q0, q1 = hold.exact_start, hold.exact_end
            duration = self.clock.elapsed_bounds(q0, q1)
            rate = self.clock.interval_bounds((q0, q1))['rate_s_inv'].upper[0]
            for key in ids:
                box = hold.input_box(key)
                u = tuple(max(abs(lo), abs(hi)) for lo, hi in zip(box.lower, box.upper))
                _require(max(u) <= exact(options.input_component_bound), key+': saved actuator support exceeds the supplied component bound')
                prop = attitude_tracker.chasers.get(key, attitude_tracker.default_chaser)
                force = exact(prop.mass_kg)*rate**2*_sqrt_upper(3*sum(value*value for value in u))
                _require(force <= exact(options.translation_force_n), key+': per-hold body-force allowance exceeds installed authority')
                impulse[key] += _dyadic_upper(force*duration.upper[0])
                peak[key] = max(peak[key], force)
        denominator = exact(STANDARD_GRAVITY)*exact(options.specific_impulse_s)
        fuel = {key: value/denominator for key, value in impulse.items()}
        binding = dict(source_approach_sha256=self.source_hash, full_journal_sha256=self.journal_hash,
            full_journal_archive=self.records.as_dict() if isinstance(self.records, NonlinearHoldArchive) else None,
            exact_start_anomaly=str(start), exact_end_anomaly=str(end), holds=len(self.records),
            actuator_realization_id=self.actuator_id, sensor_profile_id=self.sensor_id,
            clock_fingerprint=self.clock.fingerprint, physical_truth_fingerprint=self.model.fingerprint,
            decision_identities_verified=True, complete_committed_clock_verified=True,
            exact_impulse_upper_n_s={key: str(value) for key, value in impulse.items()},
            exact_fuel_upper_kg={key: str(value) for key, value in fuel.items()},
            formula='F1 <= m_initial_wet*nu_hold_upper^2*sqrt(3*sum(max(abs(u_lo),abs(u_hi))^2)); sum(F1*dt_hold_upper)/(g0*Isp)',
            proof_scope='Authenticated retained admission bookkeeping; independent interval witnesses are retained by the producer, not re-executed here')
        elapsed = self.clock.elapsed_bounds(start, end)
        return NonlinearPropellantBudget(dict(method='original-commanded-reference-clock-v1',
            physical_duration_s=elapsed.as_dict(), exact_start_anomaly=str(start), exact_end_anomaly=str(end),
            clock_fingerprint=self.clock.fingerprint), {key: _outward_float(value) for key, value in peak.items()},
            {key: _outward_float(value) for key, value in fuel.items()}, float(options.input_component_bound),
            float(options.specific_impulse_s), float(options.translation_force_n), binding)
