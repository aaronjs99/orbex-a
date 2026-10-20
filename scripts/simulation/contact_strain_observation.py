"""Intersect carried bond strain with bounded same-event pad wrench data.

For positive engagement a, the declared law is
F = F_unbonded - a*(k*d + c*u), M = -a*(kr*log(Q) + cr*spin).
Inverting these equations retains displacement/rotation information which
independent pose boxes lose. This never restarts a previously engaged bond.
The measured couple is the pad's free couple, not its bus lever-arm moment.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from orbexa.control.reachability import IntervalBox, _ARB_LOCK, _ArbArithmetic, _ProofUnresolved
from orbexa.core.attitude import finite_vector
from orbexa.core.reference_frame import fingerprint
from orbexa.simulation.contact_observation import ContactObservationBounds
from orbexa.simulation.contact_rotation import auxiliary_initial, intersect
from orbexa.simulation.contact_strain import initial_strain_boxes
from orbexa.simulation.rigid_interval import matvec, transpose, scale, subtract


_SOURCE = {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
    for name in ('contact_strain_observation.py', 'contact_enclosure.py', 'contact_strain.py',
                 'contact_rotation.py', 'rigid_interval.py')}


class ContactStrainConditioner:
    """A same-event interval contractor under the explicit compliant-pad law."""

    def condition(self, *, fleet, options, bonds, actuation, time_s, initial,
                  rotations, gaps, strains, rest_enclosures, engagement_times,
                  observation, support, representative_diag, observation_identity):
        if not isinstance(support, ContactObservationBounds):
            raise ValueError('Bond strain conditioning requires typed bounded wrench supports')
        fleet.check_observation(observation, fleet.started+float(time_s))
        if len(strains) != fleet.chaser_count:
            raise ValueError('Carried strain supports must name the exact fleet')
        result, records = list(strains), []
        import flint
        from orbexa.simulation.contact_enclosure import _ContactField
        with _ARB_LOCK, flint.ctx.workprec(128):
            ar = _ArbArithmetic(flint)
            model = _ContactField(fleet, options, bonds, actuation, ar,
                rest_enclosures=rest_enclosures, engagement_times=engagement_times)
            combined, _, _ = auxiliary_initial(fleet, initial, ar, rotations=rotations, gaps=gaps)
            combined, graph = initial_strain_boxes(model, time_s, combined, ar, strains)
            _, diag = model.evaluate(ar.scalar(time_s), ar.vector(combined))
            target_inverse = transpose(diag['rotations'][0])

            def measured(vector, radius, name):
                vector = finite_vector(vector, name, 3)
                return ar.vector(IntervalBox.from_center_radius(tuple(map(Fraction, vector)),
                    (Fraction(radius),)*3))

            for k, (member, row, prior) in enumerate(zip(fleet.members, diag['rows'], graph)):
                channel = observation.get('contact_wrenches_eci', {}).get(member.id)
                if not np.isfinite(bonds.engagement[k]) or channel is None:
                    continue
                activation = ar.box((row['activation'],))
                if activation.lower[0] <= 0:
                    records.append(dict(member_id=member.id, contracted=False,
                        reason='engagement_activation_not_strictly_positive', activation=activation.as_dict()))
                    continue
                force_eci = measured(channel['force_n'], support.wrench_force_error_n, 'Measured pad force')
                couple_eci = measured(channel['couple_nm'], support.wrench_couple_error_nm, 'Measured pad free couple')
                force, couple = matvec(target_inverse, force_eci), matvec(target_inverse, couple_eci)
                bond_wrench = scale(1/row['activation'], subtract(row['unbonded_force_patch'], force))
                rotation_wrench = scale(-1/row['activation'], couple)
                displacement = scale(1/ar.scalar(options.bond_stiffness_n_m), subtract(
                    bond_wrench,
                    scale(ar.scalar(options.bond_damping_n_s_m), row['u'])))
                angle = scale(1/ar.scalar(options.bond_rotational_stiffness_nm_rad), subtract(
                    rotation_wrench,
                    scale(ar.scalar(options.bond_rotational_damping_nm_s_rad), row['spin'])))
                scalar = prior.slice(6, 7)
                if scalar.lower[0] <= 0 or scalar.lower[0] > 1:
                    raise _ProofUnresolved('Measured bond strain meets the principal-log cut or excludes a unit quaternion')
                # For every unit Q on the positive principal graph,
                # log(Q)=beta*Q_vector, beta in [2*w_lower,2/w_lower].
                multiplier = ar.interval((2*scalar.lower[0], 2/scalar.lower[0]))
                vector = scale(1/multiplier, angle)
                cover = ar.box(displacement+vector+(ar.interval((scalar.lower[0], scalar.upper[0])),))
                posterior = intersect(prior, cover)
                actual = representative_diag['rows'][k]
                relative = Rotation.from_matrix(actual['relative_rotation']@bonds.rotation_rest[k].T).as_quat()
                if relative[3] < 0:
                    relative = -relative
                numerical = IntervalBox.point(tuple(actual['e']-bonds.bond_rest[k])+tuple(relative))
                if not posterior.contains(numerical):
                    raise ValueError(member.id+': wrench support contradicts the executed bond-strain representative')
                wrench_support = ar.box(bond_wrench+rotation_wrench)
                numerical_wrench = IntervalBox.point(tuple(options.bond_stiffness_n_m*(actual['e']-bonds.bond_rest[k])+
                    options.bond_damping_n_s_m*actual['u'])+tuple(options.bond_rotational_stiffness_nm_rad*actual['angle']+
                    options.bond_rotational_damping_nm_s_rad*actual['spin']))
                if not wrench_support.contains(numerical_wrench):
                    raise ValueError(member.id+': wrench support contradicts the executed constitutive-law representative')
                result[k] = posterior
                records.append(dict(member_id=member.id, contracted=True, activation=activation.as_dict(),
                    initial=prior.as_dict(), supported_inverse=cover.as_dict(), conditioned=posterior.as_dict(),
                    supported_unscaled_wrench=wrench_support.as_dict(),
                    measured_force_eci_n=ar.box(force_eci).as_dict(), measured_free_couple_eci_nm=ar.box(couple_eci).as_dict(),
                    local_force_n=ar.box(force).as_dict(), local_free_couple_nm=ar.box(couple).as_dict(),
                    unbonded_local_force_n=ar.box(row['unbonded_force_patch']).as_dict(),
                    relative_velocity_m_s=ar.box(row['u']).as_dict(), relative_spin_rad_s=ar.box(row['spin']).as_dict(),
                    logarithm_multiplier=ar.box((multiplier,)).as_dict()))
        record = dict(method='bounded-same-event-wrench-bond-strain-intersection-v1', source_files=dict(_SOURCE),
            physical_initial_fingerprint=fingerprint(initial.as_dict()),
            exact_elapsed_time_s=str(time_s), exact_time_s=str(Fraction(fleet.started)+Fraction(time_s)),
            observation_identity=observation_identity, support=support.as_dict(), members=records,
            law=dict(bond_stiffness_n_m=options.bond_stiffness_n_m, bond_damping_n_s_m=options.bond_damping_n_s_m,
                rotational_stiffness_nm_rad=options.bond_rotational_stiffness_nm_rad,
                rotational_damping_nm_s_rad=options.bond_rotational_damping_nm_s_rad),
            scope='Intersection of the carried same-event physical strain graph with bounded pad force/free-couple products under the declared exact constitutive law; the supported unscaled B/M wrench uses these same measurements, with no reset, independent-noise averaging, or material/sensor qualification')
        record['identity'] = fingerprint(record)
        return result, record
