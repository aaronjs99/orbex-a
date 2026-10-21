"""Source-bound transfer of a terminal navigation/tube set into contact.

The common uncertain target ephemeris occupies the translating reference;
relative coordinates are never widened by subtracting two independent copies
of that common ephemeris. No uncertainty is replaced by executed point truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import copy

import numpy as np
from scipy.spatial.transform import Rotation

from orbexa.control.certification.reachability import IntervalBox, _ARB_LOCK
from orbexa.control.certification.nonlinear import _Arithmetic
from orbexa.core.dynamics.reference_frame import exact, fingerprint
from orbexa.simulation.numerics.rigid_interval import add, cross, matvec, normalized_quaternion, quaternion_rotation, scale, transpose, norm_upper


def _saved_box(value, dimension):
    result = IntervalBox(value.get('exact_lower', value.get('lower')),
                         value.get('exact_upper', value.get('upper')))
    if result.dimension != dimension:
        raise ValueError('Contact boundary source set has the wrong dimension')
    return result


def _intersection(left, right):
    lower = tuple(max(a, b) for a, b in zip(left.lower, right.lower))
    upper = tuple(min(a, b) for a, b in zip(left.upper, right.upper))
    if any(a > b for a, b in zip(lower, upper)):
        raise ValueError('Terminal navigation and admitted tube sets are disjoint')
    return IntervalBox(lower, upper)


def _quaternion_box(matrix, operator_error, representative, ar):
    """A hemisphere quaternion cover of a supported SO(3) neighborhood.

    For eps=||R-R0||_2<2, the aligned hemisphere satisfies
    ||q-q0||_2 <= eps/sqrt(2+sqrt(4-eps^2)).
    The small numerical projection of the reported matrix to SO(3) is added
    using an exact rational entrywise one-norm bound.
    """
    numeric = np.asarray([[float(exact(x)) for x in row] for row in matrix])
    center = Rotation.from_matrix(numeric).as_quat()
    projected = Rotation.from_quat(center).as_matrix()
    rounding = sum(abs(exact(x)-Fraction(float(y)))
                   for row, prow in zip(matrix, projected) for x, y in zip(row, prow))
    unit_rotation = quaternion_rotation(normalized_quaternion(tuple(map(ar.scalar, center)), ar))
    rounding += norm_upper(tuple(value-ar.scalar(point) for row, points in zip(unit_rotation, projected)
                                 for value, point in zip(row, points)), ar)
    error = exact(operator_error)+rounding
    radius = Fraction(2) if error >= 2 else ar.box((ar.scalar(error)/
        (2+(4-ar.scalar(error)**2).sqrt()).sqrt(),)).upper[0]
    if float(np.dot(center, representative)) < 0:
        center = -center
    box = IntervalBox.from_center_radius(tuple(map(Fraction, center)), (radius,)*4)
    # Rotation.from_matrix returns rounded coordinates. The real unit
    # representative and its rounded value are both retained.
    unit = ar.box(normalized_quaternion(ar.vector(IntervalBox.point(center)), ar))
    return IntervalBox(tuple(lo-radius for lo in unit.lower),
                       tuple(hi+radius for hi in unit.upper)).hull(box)


@dataclass(frozen=True)
class TerminalContactBoundary:
    """Immutable saved same-event inputs, interpreted only when the fleet exists."""
    navigation: dict
    nominal_centers: dict
    phi: dict
    omega: dict
    lambda_gain: tuple
    mounting_rotations: dict
    clock: object
    orientation_gate_rad: float = .01
    spin_gate_rad_s: float = .001
    precision_bits: int = 128

    def __post_init__(self):
        for name in ('navigation', 'nominal_centers', 'phi', 'omega', 'mounting_rotations'):
            object.__setattr__(self, name, copy.deepcopy(getattr(self, name)))
        for name in ('nominal_centers', 'phi', 'omega'):
            object.__setattr__(self, name, {key: list(map(float, value)) for key, value in getattr(self, name).items()})
        ids = set(self.nominal_centers)
        if (not ids or any(set(getattr(self, name)) != ids for name in
                ('phi', 'omega', 'mounting_rotations')) or
                self.navigation.get('status') != 'VALID' or
                self.navigation.get('clock_fingerprint') != self.clock.fingerprint):
            raise ValueError('Contact boundary requires complete same-event valid navigation and tube inputs')
        if any(not np.isfinite(value) or value < 0 for value in
               (self.orientation_gate_rad, self.spin_gate_rad_s)):
            raise ValueError('Contact attitude admission supports must be finite and nonnegative')

    @classmethod
    def from_approach(cls, approach, clock, *, orientation_gate_rad=.01, spin_gate_rad_s=.001):
        from orbexa.core.geometry.mounting import mounting_frame
        metadata = approach.metadata
        navigation = metadata['navigation_history'][-1]
        endpoint = metadata['continuous_execution_certificate']['exact_end_anomaly']
        if exact(navigation['exact_q']) != exact(endpoint):
            raise ValueError('Contact boundary navigation is not at the committed terminal endpoint')
        ids = tuple(approach.actual_trajectories)
        assignments = {row['chaser_id']: row for row in metadata['chaser_assignments']}
        if set(assignments) != set(ids):
            raise ValueError('Contact boundary assignments differ from the recorded fleet')
        return cls(navigation=navigation,
            nominal_centers={key: approach.tube_diagnostics[key]['nominal_center'][-1] for key in ids},
            phi={key: approach.tube_diagnostics[key]['phi'][-1] for key in ids},
            omega={key: approach.tube_diagnostics[key]['omega'][-1] for key in ids},
            lambda_gain=tuple(metadata['lambda_gain']),
            mounting_rotations={key: mounting_frame(assignments[key]['docking_normal_body']).tolist() for key in ids},
            clock=clock, orientation_gate_rad=orientation_gate_rad, spin_gate_rad_s=spin_gate_rad_s)

    def build(self, fleet):
        """Return the complete packed contact start superset and its derivation."""
        import flint
        ids = tuple(member.id for member in fleet.members)
        nav = self.navigation
        if set(ids) != set(self.nominal_centers):
            raise ValueError('Contact boundary transfer fleet changed')
        relative = nav['relative_state_boxes']
        if set(relative) != set(ids):
            raise ValueError('Contact boundary navigation omits a member')
        target = _saved_box(nav['target_ephemeris_box'], 6)
        rate_box = _saved_box(nav['target_angular_velocity_box'], 3)
        q = exact(nav['exact_q'])
        source = dict(method='terminal-navigation-tube-to-contact-v1', exact_q=str(q),
            navigation_fingerprint=fingerprint(nav), clock_fingerprint=self.clock.fingerprint,
            target_ephemeris=target.as_dict(), target_body_rate=rate_box.as_dict(),
            attitude_gate_rad=self.orientation_gate_rad, spin_gate_rad_s=self.spin_gate_rad_s,
            tube_inputs=dict(nominal_centers=self.nominal_centers, phi=self.phi, omega=self.omega,
                             lambda_gain=self.lambda_gain), relative_sources={})
        with _ARB_LOCK, flint.ctx.workprec(self.precision_bits):
            ar = _Arithmetic(flint)
            frame_rotation, rate, _ = self.clock.evaluate(ar.scalar(q), ar)
            numerical, _ = fleet.unpack(fleet.initial_values)
            observed_reference = np.asarray([[float(exact(x)) for x in row]
                                              for row in nav['target_orientation_matrix']])
            observed = np.asarray(self.clock.point(q).rotation)@observed_reference
            supported = tuple(tuple(sum(frame_rotation[i][k]*ar.scalar(exact(nav['target_orientation_matrix'][k][j]))
                for k in range(3)) for j in range(3)) for i in range(3))
            defect = norm_upper(tuple(value-ar.scalar(float(center)) for row, points in zip(supported, observed)
                                     for value, center in zip(row, points)), ar)
            lifted_error = exact(nav['target_orientation_operator_error'])+defect
            target_q = _quaternion_box(observed, lifted_error, numerical[0, 6:10], ar)
            source['pose_frame_transfer'] = dict(source_frame='body-to-reference-orbit', target_frame='body-to-ECI',
                lifted_raw_rotation=observed.tolist(), clock_product_error_upper=str(defect),
                operator_error_upper=str(lifted_error))
            target_rotation = quaternion_rotation(ar.vector(target_q))
            packed = list(ar.vector(target))
            packed.extend((ar.scalar(0),)*6)
            packed.extend(ar.vector(target_q))
            packed.extend(ar.vector(rate_box))
            packed.append(ar.scalar(0))
            for number, key in enumerate(ids):
                center = tuple(map(exact, self.nominal_centers[key]))
                omega = tuple(map(exact, self.omega[key]))
                phi = tuple(map(exact, self.phi[key]))
                gains = tuple(map(exact, self.lambda_gain))
                radius = omega+tuple(p+abs(g)*o for p, g, o in zip(phi, gains, omega))
                tube = IntervalBox.from_center_radius(center, radius)
                state = _intersection(_saved_box(relative[key], 6), tube)
                value = ar.vector(state)
                displacement = matvec(frame_rotation, value[:3])
                rotating = cross((ar.scalar(0), ar.scalar(0), ar.scalar(1)), value[:3])
                velocity = matvec(frame_rotation, scale(rate, add(value[3:], rotating)))
                mount = np.asarray(self.mounting_rotations[key], dtype=float)
                center_rotation = observed@mount
                mounted = tuple(tuple(sum(supported[i][k]*ar.scalar(float(mount[k, j])) for k in range(3))
                    for j in range(3)) for i in range(3))
                mount_defect = norm_upper(tuple(value-ar.scalar(float(center))
                    for row, points in zip(mounted, center_rotation) for value, center in zip(row, points)), ar)
                bus_q = _quaternion_box(center_rotation,
                    exact(nav['target_orientation_operator_error'])+mount_defect+exact(self.orientation_gate_rad),
                    numerical[number+1, 6:10], ar)
                bus_rotation = quaternion_rotation(ar.vector(bus_q))
                inertial_rate = matvec(target_rotation, ar.vector(rate_box))
                bus_rate = matvec(transpose(bus_rotation), inertial_rate)
                uncertainty = ar.interval((-exact(self.spin_gate_rad_s), exact(self.spin_gate_rad_s)))
                packed.extend(displacement+velocity+ar.vector(bus_q)+tuple(x+uncertainty for x in bus_rate))
                packed.append(ar.scalar(numerical[number+1, 13]))
                source['relative_sources'][key] = dict(navigation=_saved_box(relative[key], 6).as_dict(),
                    admitted_tube=tube.as_dict(), intersection=state.as_dict())
            packed.extend((ar.scalar(0),)*(6*len(ids)))
            box = ar.box(packed)
        representative = IntervalBox.point(fleet.initial_values[:fleet.ledger_start])
        source['source_set'] = box.as_dict()
        source['numerical_representative_was_contained'] = box.contains(representative)
        # Including a rounded numerical coordinate in an already supported
        # set preserves every physical uncertainty and also tracks its error.
        box = box.hull(representative)
        source['initial_set'] = box.as_dict()
        source['identity'] = fingerprint(source)
        return box, source
