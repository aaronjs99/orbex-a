"""Finite compliant capture of a rigid target by independently moving buses.

This is a declared representative adhesive-pad study model. The finite-bandwidth
positioner has negligible moving mass; its reaction work is explicitly counted.
It is not a manufactured gripper specification or a flexible spacecraft model.
The six-DOF vehicle/contact-wrench formulation follows the modeling scope of
NASA-CR-179147: https://ntrs.nasa.gov/citations/19870016368 . Adhesion strengths,
pad properties and actuator parameters below are independent study assumptions.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Optional
import numpy as np

from orbexa.simulation.contact_contracts import (
    ContactBodyState, ContactMember, FiniteContactOptions, FiniteContactResult,
)
from orbexa.simulation.contact_control import ContactController, _limit
from orbexa.simulation.contact_guards import ContactGuards
from orbexa.simulation.contact_mechanics import ContactMechanics
from orbexa.simulation.contact_records import ContactRecorder
from orbexa.simulation.contact_state import (
    ContactActuation, ContactBondState, ContactFleet, rk4_contact_step,
)


_CONTACT_SOURCE_FILES = (
    'contact.py', 'contact_contracts.py', 'contact_control.py', 'contact_guards.py',
    'contact_mechanics.py', 'contact_records.py', 'contact_state.py',
)


def finite_contact_source_identity():
    """Bind all implementation files, including the public coordinator, by bytes."""
    directory = Path(__file__).parent
    files = {f'simulation/{name}': hashlib.sha256((directory / name).read_bytes()).hexdigest()
             for name in _CONTACT_SOURCE_FILES}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(schema='orbexa.finite_contact_source_bundle.v1', sha256=digest, files=files)


class FiniteContactSimulator:
    """Simultaneous rigid dynamics with objective compliant-pad wrenches.

    ``feedback(time_s, snapshot)`` may return estimated ContactBodyState values
    under the same target/chasers keys. It affects commands only. Without it,
    feedback is explicitly exact. All audits always use the physical plant.
    The snapshot also contains ``contact_gaps_m[id]`` (positive outward) and
    ``contact_wrenches_eci[id]`` with force_n/couple_nm three-vectors. An
    injected observer must supply its own measured versions to use these
    channels; omitted wrench channels mean zero measured force, and omitted
    range channels force a slow search over the declared uncertainty depth.
    ``environment(time_s, bodies_by_name)`` receives a flat mapping containing
    target and every chaser id, and must return an extra acceleration_eci_m_s2
    and optional torque_body_nm for every body. These are beyond central
    gravity and configured gravity-gradient torque, in the supplied ECI basis.
    """

    def __init__(self, options: Optional[FiniteContactOptions] = None):
        self.options = options or FiniteContactOptions()

    def run(self, *, target_properties, target_state: ContactBodyState,
            members, geometry, mu_m3_s2, gravity_gradient=False,
            feedback: Optional[Callable] = None, environment: Optional[Callable] = None) -> FiniteContactResult:
        o = self.options
        fleet = ContactFleet.create(target_properties=target_properties, target_state=target_state,
            members=members, mu_m3_s2=mu_m3_s2, gravity_gradient=gravity_gradient)
        y = fleet.initial_values
        bonds = ContactBondState.create(fleet.chaser_count)
        actuation = ContactActuation.create(fleet.body_count, fleet.chaser_count)
        controller = ContactController(fleet=fleet, options=o, bonds=bonds, actuation=actuation,
            initial_values=y, feedback=feedback)
        mechanics = ContactMechanics(fleet=fleet, options=o, bonds=bonds, actuation=actuation,
            environment=environment)
        guards = ContactGuards(fleet=fleet, options=o, geometry=geometry)
        _, original = mechanics.evaluate(0., y)
        t, dwell, lock_energy = 0., 0., np.inf
        recorder = ContactRecorder(fleet=fleet, options=o, geometry=geometry, bonds=bonds,
            actuation=actuation, extrema=guards.maxima, minimum_clearances=guards.minima,
            original=original, feedback=feedback, environment=environment)

        def finish(success, message, diag):
            return recorder.finish(success, message, t=t, y=y, diag=diag, dwell=dwell,
                lock_energy=lock_energy, saturation_count=controller.saturation_count)

        recorder.record(0., y, original)
        try:
            while t < o.max_duration_s-1e-12:
                dt = min(o.max_step_s, o.max_duration_s-t)
                _, diag = mechanics.evaluate(t, y)
                reason = guards.check(t, y, diag, dt)
                if reason:
                    return finish(False, reason, diag)
                controller.command(t, y, diag)
                bonds.advance_engagement(t, dt, diag, actuation, o)
                committed_values, committed_time = y.copy(), t
                rk4_contact_step(mechanics.evaluate, t, y, dt)
                t += dt
                body, tips = fleet.unpack(y)
                if not np.all(np.isfinite(y)) or np.any(body[:, 13] < -1e-12):
                    y, t = committed_values, committed_time
                    _, diag = mechanics.evaluate(t, y)
                    return finish(False, 'finite_contact_nonfinite_state_or_fuel_exhausted', diag)
                norms = np.linalg.norm(body[:, 6:10], axis=1)
                guards.maxima['quaternion_norm_error'] = max(guards.maxima['quaternion_norm_error'], float(np.max(abs(norms-1))))
                body[:, 6:10] /= norms[:, None]
                _, diag = mechanics.evaluate(t, y)
                guards.maxima['internal_force_residual_n'] = max(guards.maxima['internal_force_residual_n'], diag['internal_force'])
                guards.maxima['internal_torque_residual_nm'] = max(guards.maxima['internal_torque_residual_nm'], diag['internal_torque'])
                reason = guards.check(t, y, diag, dt)
                if reason:
                    return finish(False, reason, diag)
                lock_energy, quiet = mechanics.settling_state(y, diag)
                dwell = dwell+dt if quiet else 0.
                recorder.record(t, y, diag)
                if dwell >= o.settling_dwell_s:
                    return finish(True, 'finite_contact_settled_latch_completed', diag)
            return finish(False, 'finite_contact_duration_limit_before_settled_latch', diag)
        except (ValueError, RuntimeError, ArithmeticError) as error:
            _, diag = mechanics.evaluate(t, y)
            return finish(False, f'finite_contact_rejected: {type(error).__name__}: {error}', diag)
