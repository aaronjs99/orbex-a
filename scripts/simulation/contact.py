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
from fractions import Fraction
from pathlib import Path
from typing import Callable, Optional
import numpy as np
from orbexa.control.reachability import IntervalBox, _ProofUnresolved

from orbexa.simulation.contact_contracts import (
    ContactBodyState, ContactMember, FiniteContactOptions, FiniteContactResult,
)
from orbexa.simulation.contact_control import ContactController, _limit
from orbexa.simulation.contact_guards import ContactGuards
from orbexa.simulation.contact_mechanics import ContactMechanics
from orbexa.simulation.contact_enclosure import ContactStepEncloser
from orbexa.simulation.contact_records import ContactRecorder
from orbexa.simulation.contact_state import (
    ContactActuation, ContactBondState, ContactFleet, rk4_contact_step,
)


_CONTACT_SOURCE_FILES = (
    'contact.py', 'contact_contracts.py', 'contact_control.py', 'contact_guards.py',
    'contact_mechanics.py', 'contact_records.py', 'contact_state.py',
    'contact_enclosure.py', 'rigid_interval.py',
    'contact_boundary.py', 'contact_observation.py', 'finite_lock.py', 'finite_lock_chart.py',
    'contact_rotation.py',
    'contact_strain.py',
    'contact_strain_observation.py',
    'contact_wrench.py',
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
            feedback: Optional[Callable] = None, environment: Optional[Callable] = None,
            initial_enclosure=None, certificate_sink: Optional[Callable] = None) -> FiniteContactResult:
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
        proof_time, proved_dwell = Fraction(0), Fraction(0)
        recorder = ContactRecorder(fleet=fleet, options=o, geometry=geometry, bonds=bonds,
            actuation=actuation, extrema=guards.maxima, minimum_clearances=guards.minima,
            original=original, feedback=feedback, environment=environment)
        encloser = ContactStepEncloser()
        rest_enclosures = [None for _ in fleet.members]
        engagement_times = [None for _ in fleet.members]
        rotation_enclosures = None
        gap_enclosures = [None for _ in fleet.members]
        strain_enclosures = [None for _ in fleet.members]
        wrench_enclosures = [None for _ in fleet.members]
        proven_preload = [Fraction(0) for _ in fleet.members]
        recorder.continuous_enclosure = dict(method='arb-compliant-contact-held-flow-v1',
            initial_scope='caller_supplied_state_set' if initial_enclosure is not None
                else 'declared_numerical_start_with_unit_quaternion_interpretation',
            accepted_steps=0, accepted_duration_s=0., proof_chain_sha256=None,
            last_refusal=None, step_records=[], bond_events=[])

        def finish(success, message, diag):
            return recorder.finish(success, message, t=t, y=y, diag=diag, dwell=dwell,
                lock_energy=lock_energy, saturation_count=controller.saturation_count)

        recorder.record(0., y, original)
        try:
            if initial_enclosure is not None and callable(getattr(initial_enclosure, 'build', None)):
                initial_enclosure, boundary = initial_enclosure.build(fleet)
                recorder.continuous_enclosure['boundary_transfer'] = boundary
            enclosure = encloser.initial_box(fleet, initial_enclosure)
            recorder.continuous_enclosure['initial_set'] = enclosure.as_dict()
            while t < o.max_duration_s-1e-12:
                dt = min(o.max_step_s, o.max_duration_s-t)
                _, diag = mechanics.evaluate(t, y)
                reason = guards.check(t, y, diag, dt)
                if reason:
                    return finish(False, reason, diag)
                observation = controller.command(t, y, diag)
                conditioning = None
                strain_conditioning = None
                if callable(getattr(feedback, 'observation_support', None)):
                    from orbexa.simulation.contact_observation import ContactSetConditioner
                    observation_support = feedback.observation_support()
                    enclosure, conditioning = ContactSetConditioner().condition(fleet=fleet, options=o, bonds=bonds,
                        initial=enclosure, observation=observation, support=observation_support, representative=y,
                        exact_elapsed_time_s=proof_time)
                    from orbexa.simulation.contact_rotation import intersect
                    measured_rotations = tuple(IntervalBox(box['exact_lower'], box['exact_upper'])
                        for box in conditioning['rotation_supports'])
                    rotation_enclosures = (measured_rotations if rotation_enclosures is None else
                        tuple(intersect(before, measured) for before, measured in zip(rotation_enclosures, measured_rotations)))
                    measured_gaps = {item['member_id']: IntervalBox(item['supported_gap_m']['exact_lower'],
                        item['supported_gap_m']['exact_upper']) for item in conditioning['gap_constraints']}
                    for k, member in enumerate(fleet.members):
                        if member.id in measured_gaps:
                            gap_enclosures[k] = (measured_gaps[member.id] if gap_enclosures[k] is None else
                                intersect(gap_enclosures[k], measured_gaps[member.id]))
                event_support = encloser.event_support(fleet=fleet, options=o, bonds=bonds,
                    actuation=actuation, time_s=proof_time, initial=enclosure, rest_enclosures=rest_enclosures,
                    engagement_times=engagement_times, rotation_initial=rotation_enclosures, gap_initial=gap_enclosures,
                    strain_initial=strain_enclosures)
                previous_engagement = bonds.engagement.copy()
                # Preload time comes only from already completed, whole-set
                # proved intervals. A boundary must not credit a future step.
                bonds.preload[:] = [o.preload_dwell_s if support['preload_supported'] and value >= Fraction(o.preload_dwell_s) else 0.
                    for value, support in zip(proven_preload, event_support)]
                bonds.advance_engagement(t, 0., diag, actuation, o)
                for k, (before, after) in enumerate(zip(previous_engagement, bonds.engagement)):
                    if not np.isfinite(before) and np.isfinite(after):
                        engagement_times[k] = proof_time
                        rest_enclosures[k] = {key: event_support[k][key] for key in ('position', 'quaternion')}
                        # This digital event forms a stress-free bond at the
                        # same actual uncertain pose. Existing bonds retain
                        # their propagated strain; they are never reset here.
                        strain_enclosures[k] = IntervalBox.point((0, 0, 0, 0, 0, 0, 1))
                        recorder.continuous_enclosure['bond_events'].append(dict(member_id=fleet.members[k].id,
                            time_s=t, proved_preload_dwell_s=float(proven_preload[k]),
                            exact_event_time_s=str(proof_time), exact_proved_preload_dwell_s=str(proven_preload[k]),
                            rest={key: box.as_dict() for key, box in rest_enclosures[k].items()}))
                if conditioning is not None and np.any(np.isfinite(bonds.engagement)):
                    from orbexa.simulation.contact_strain_observation import ContactStrainConditioner
                    strain_enclosures, strain_conditioning = ContactStrainConditioner().condition(
                        fleet=fleet, options=o, bonds=bonds, actuation=actuation, time_s=proof_time,
                        initial=enclosure, rotations=rotation_enclosures, gaps=gap_enclosures,
                        strains=strain_enclosures, rest_enclosures=rest_enclosures, engagement_times=engagement_times,
                        observation=observation, support=observation_support, representative_diag=diag,
                        observation_identity=conditioning['identity'])
                    recorder.continuous_enclosure['last_strain_conditioning'] = strain_conditioning
                    by_member = {item['member_id']: item for item in strain_conditioning['members']}
                    for k, member in enumerate(fleet.members):
                        item = by_member.get(member.id)
                        if item is not None and item.get('contracted'):
                            raw = item['supported_unscaled_wrench']
                            measured = IntervalBox(raw['exact_lower'], raw['exact_upper'])
                            wrench_enclosures[k] = (measured if wrench_enclosures[k] is None else
                                intersect(wrench_enclosures[k], measured))
                certificate = encloser.refined_bound(fleet=fleet, options=o, bonds=bonds, actuation=actuation,
                    geometry=geometry, time_s=proof_time, duration_s=dt, initial=enclosure,
                    environment=environment, rest_enclosures=rest_enclosures, engagement_times=engagement_times,
                    rotation_initial=rotation_enclosures, gap_initial=gap_enclosures, strain_initial=strain_enclosures,
                    wrench_initial=wrench_enclosures)
                if not certificate.passed:
                    recorder.continuous_enclosure['last_refusal'] = certificate.as_dict()
                    return finish(False, 'finite_contact_whole_step_not_proven: '+certificate.reason, diag)
                if certificate_sink is not None:
                    certificate_sink(certificate.as_dict())
                committed_values, committed_time = y.copy(), t
                # A held controller command does not prescribe the numerical
                # integration step. The rotational bond can be much stiffer
                # than the 200 Hz command loop. Keep the same physical hold,
                # use at least the proved flow subdivision, and resolve its
                # declared damping/stiffness scales in the representative.
                masses = diag['masses']
                inverse_target = fleet.chaser_count/float(np.linalg.eigvalsh(diag['inertias'][0]).min())
                relaxation = 0.
                for k, engaged in enumerate(bonds.engagement):
                    activation = 0. if not np.isfinite(engaged) else float(np.clip((t+dt-engaged)/o.bond_engagement_s, 0., 1.))
                    inverse_inertia = inverse_target+1/float(np.linalg.eigvalsh(diag['inertias'][k+1]).min())
                    inverse_mass = fleet.chaser_count/masses[0]+1/masses[k+1]
                    angular = activation*(o.bond_rotational_damping_nm_s_rad*inverse_inertia+
                        np.sqrt(o.bond_rotational_stiffness_nm_rad*inverse_inertia))
                    linear = ((o.compression_damping_n_s_m+activation*o.bond_damping_n_s_m)*inverse_mass+
                        np.sqrt((o.compression_stiffness_n_m+activation*o.bond_stiffness_n_m)*inverse_mass))
                    relaxation = max(relaxation, angular, linear)
                numerical_count = max(certificate.context['proved_substeps'], int(np.ceil(dt*relaxation)), 1)
                if numerical_count > 4096:
                    return finish(False, 'finite_contact_numerical_substep_budget_exceeded', diag)
                numerical_dt = dt/numerical_count
                for index in range(numerical_count):
                    rk4_contact_step(mechanics.evaluate, t+index*numerical_dt, y, numerical_dt)
                exact_start_time, proof_time = proof_time, proof_time+Fraction(dt)
                t = float(proof_time)
                body, tips = fleet.unpack(y)
                if not np.all(np.isfinite(y)) or np.any(body[:, 13] < -1e-12):
                    y, t = committed_values, committed_time
                    _, diag = mechanics.evaluate(t, y)
                    return finish(False, 'finite_contact_nonfinite_state_or_fuel_exhausted', diag)
                norms = np.linalg.norm(body[:, 6:10], axis=1)
                guards.maxima['quaternion_norm_error'] = max(guards.maxima['quaternion_norm_error'], float(np.max(abs(norms-1))))
                body[:, 6:10] /= norms[:, None]
                representative = IntervalBox.point(y[:fleet.ledger_start])
                # Exact constant acceleration can already put the rounded RK4
                # point just outside an extremely narrow real-valued endpoint.
                # Carry a superset containing both; never reset to that point.
                # This records an actual error enclosure, without assuming an
                # RK4 order estimate or inventing a numerical-error tolerance.
                defect = IntervalBox(tuple(lo-point for lo, point in zip(certificate.endpoint.lower, representative.lower)),
                    tuple(hi-point for hi, point in zip(certificate.endpoint.upper, representative.upper)))
                enclosure = certificate.endpoint.hull(representative)
                import flint
                from orbexa.control.reachability import _ARB_LOCK, _ArbArithmetic
                from orbexa.simulation.rigid_interval import normalized_quaternion, quaternion_rotation
                with _ARB_LOCK, flint.ctx.workprec(encloser.precision_bits):
                    ar = _ArbArithmetic(flint)
                    unit_values = list(ar.vector(representative))
                    rotations = []
                    for k in range(fleet.body_count):
                        begin = 6+14*k+6
                        unit = normalized_quaternion(tuple(unit_values[begin:begin+4]), ar)
                        unit_values[begin:begin+4] = unit
                        rotations.append(ar.box(tuple(value for row in quaternion_rotation(unit) for value in row)))
                    enclosure = enclosure.hull(ar.box(unit_values))
                rotation_enclosures = tuple(end.hull(numerical) for end, numerical
                    in zip(certificate.rotation_endpoint, rotations))
                proof = certificate.as_dict()
                proof_hash = hashlib.sha256(json.dumps(proof, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                previous = recorder.continuous_enclosure['proof_chain_sha256'] or ''
                if conditioning is not None:
                    previous = hashlib.sha256((previous+conditioning['identity']).encode()).hexdigest()
                recorder.continuous_enclosure['proof_chain_sha256'] = hashlib.sha256((previous+proof_hash).encode()).hexdigest()
                recorder.continuous_enclosure['accepted_steps'] += 1
                recorder.continuous_enclosure['accepted_duration_s'] = t
                recorder.continuous_enclosure['exact_accepted_duration_s'] = str(proof_time)
                recorder.continuous_enclosure['final_set'] = enclosure.as_dict()
                recorder.continuous_enclosure['step_records'].append(dict(time_s=committed_time, duration_s=dt,
                    exact_start_time_s=str(exact_start_time), exact_end_time_s=str(proof_time),
                    controls_body=actuation.controls.tolist(), extension_acceleration_body_m_s2=actuation.extension_acceleration.tolist(),
                    numerical_endpoint=y[:fleet.ledger_start].tolist(),
                    numerical_integration=dict(method='held-command-subdivided-rk4', substeps=numerical_count,
                        substep_s=numerical_dt, declared_relaxation_scale_s_inverse=relaxation,
                        scope='Representative integration choice; its error is enclosed by the carried exact-flow endpoint and representative hull'),
                    numerical_endpoint_defect=defect.as_dict(), proof_sha256=proof_hash, proof=proof,
                    checks=certificate.checks))
                if conditioning is not None:
                    recorder.continuous_enclosure['step_records'][-1]['observation_conditioning'] = conditioning
                if strain_conditioning is not None:
                    recorder.continuous_enclosure['step_records'][-1]['bond_strain_conditioning'] = strain_conditioning
                proven_preload = [value+Fraction(dt) if supported else Fraction(0) for value, supported
                    in zip(proven_preload, certificate.checks['preload_supported'])]
                _, diag = mechanics.evaluate(t, y)
                gap_enclosures = [end.hull(IntervalBox.point((row['gap'],)))
                    for end, row in zip(certificate.gap_endpoint, diag['rows'])]
                from scipy.spatial.transform import Rotation
                for k, endpoint in enumerate(certificate.strain_endpoint):
                    if np.isfinite(bonds.engagement[k]):
                        row = diag['rows'][k]
                        relative = Rotation.from_matrix(row['relative_rotation']@bonds.rotation_rest[k].T).as_quat()
                        if relative[3] < 0:
                            relative = -relative
                        strain_enclosures[k] = endpoint.hull(IntervalBox.point(tuple(row['e']-bonds.bond_rest[k])+tuple(relative)))
                        numerical_wrench = tuple(o.bond_stiffness_n_m*(row['e']-bonds.bond_rest[k])+
                            o.bond_damping_n_s_m*row['u'])+tuple(o.bond_rotational_stiffness_nm_rad*row['angle']+
                            o.bond_rotational_damping_nm_s_rad*row['spin'])
                        wrench_enclosures[k] = certificate.wrench_endpoint[k].hull(IntervalBox.point(numerical_wrench))
                recorder.continuous_enclosure['final_rotation_sets'] = [box.as_dict() for box in rotation_enclosures]
                recorder.continuous_enclosure['final_gap_sets'] = [box.as_dict() for box in gap_enclosures]
                recorder.continuous_enclosure['final_strain_sets'] = [None if box is None else box.as_dict() for box in strain_enclosures]
                recorder.continuous_enclosure['final_wrench_sets'] = [None if box is None else box.as_dict() for box in wrench_enclosures]
                recorder.continuous_enclosure['final_rest_sets'] = [None if rest is None else
                    {key: box.as_dict() for key, box in rest.items()} for rest in rest_enclosures]
                recorder.continuous_enclosure['final_contact_potential_set_j'] = certificate.endpoint_potential.hull(
                    IntervalBox.point((diag['potential'],))).as_dict()
                guards.maxima['internal_force_residual_n'] = max(guards.maxima['internal_force_residual_n'], diag['internal_force'])
                guards.maxima['internal_torque_residual_nm'] = max(guards.maxima['internal_torque_residual_nm'], diag['internal_torque'])
                reason = guards.check(t, y, diag, dt)
                if reason:
                    return finish(False, reason, diag)
                lock_energy, quiet = mechanics.settling_state(y, diag)
                eligible = (quiet and certificate.checks['quiet_supported'] if o.handoff_rule == 'settled_passive'
                    else all(row['activation'] >= 1 for row in diag['rows'])
                        and certificate.checks['finite_lock_ready'])
                proved_dwell = proved_dwell+Fraction(dt) if eligible else Fraction(0)
                dwell = float(proved_dwell)
                recorder.continuous_enclosure['exact_settling_dwell_s'] = str(proved_dwell)
                recorder.record(t, y, diag)
                if proved_dwell >= Fraction(o.settling_dwell_s):
                    return finish(True, 'finite_contact_settled_latch_completed' if o.handoff_rule == 'settled_passive'
                                  else 'finite_contact_load_bounded_lock_ready', diag)
            return finish(False, 'finite_contact_duration_limit_before_settled_latch', diag)
        except (ValueError, RuntimeError, ArithmeticError, ImportError, OSError, _ProofUnresolved) as error:
            _, diag = mechanics.evaluate(t, y)
            return finish(False, f'finite_contact_rejected: {type(error).__name__}: {error}', diag)
