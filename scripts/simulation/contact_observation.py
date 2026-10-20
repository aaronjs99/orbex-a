"""Same-event bounded contact observations intersect the carried state set.

Commands alone do not retain the information in a camera, rate gyro, range,
wrench or internal encoder. This contractor never substitutes an observed
center for a propagated physical set, and unsupported callbacks do not narrow.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction

import numpy as np
from scipy.spatial.transform import Rotation

from orbexa.control.reachability import IntervalBox, _ARB_LOCK, _ArbArithmetic, _ProofUnresolved
from orbexa.core.reference_frame import fingerprint
from orbexa.simulation.contact_boundary import _intersection, _quaternion_box
from orbexa.simulation.rigid_interval import add, subtract, scale, cross, matvec, transpose, quaternion_rotation
from orbexa.simulation.contact_rotation import initial_rotation_boxes


@dataclass(frozen=True)
class ContactObservationBounds:
    common_position_error_m: float = 1.
    common_velocity_error_m_s: float = .003
    relative_position_error_m: float = .003
    relative_velocity_error_m_s: float = .0001
    target_pose_error_rad: float = .001000001
    target_rate_error_rad_s: float = .000100001
    chaser_pose_error_rad: float = .000300001
    chaser_rate_error_rad_s: float = .000010001
    encoder_position_error_m: float = .000002
    encoder_velocity_error_m_s: float = .0000002
    local_range_error_m: float = .002
    wrench_force_error_n: float = .006
    wrench_couple_error_nm: float = .0002

    def __post_init__(self):
        if any(isinstance(value, bool) or not np.isfinite(value) or value < 0 for value in vars(self).values()):
            raise ValueError('Contact observation supports must be finite nonnegative bounds')

    def as_dict(self):
        return dict(method='deterministic-current-contact-observation-support-v1', **asdict(self))


class ContactSetConditioner:
    def condition(self, *, fleet, options, bonds, initial, observation, support, representative,
                  exact_elapsed_time_s=None):
        if not isinstance(support, ContactObservationBounds):
            raise ValueError('Contact conditioning requires explicit typed observation supports')
        expected_time = (observation['target'].time_s if exact_elapsed_time_s is None
                         else fleet.started+float(exact_elapsed_time_s))
        fleet.check_observation(observation, expected_time)
        lower, upper = list(initial.lower), list(initial.upper)
        def narrow(start, box):
            current = IntervalBox(lower[start:start+box.dimension], upper[start:start+box.dimension])
            result = _intersection(current, box)
            lower[start:start+box.dimension], upper[start:start+box.dimension] = result.lower, result.upper
        def centered(value, radius):
            return IntervalBox.from_center_radius(tuple(map(Fraction, value)), (Fraction(radius),)*len(value))
        target = observation['target']
        import flint
        with _ARB_LOCK, flint.ctx.workprec(128):
            ar = _ArbArithmetic(flint)
            bodies, _ = fleet.unpack(np.asarray(representative))
            for k, state in enumerate((target,)+tuple(observation['chasers'][member.id] for member in fleet.members)):
                start = 6+14*k
                pose = support.target_pose_error_rad if k == 0 else support.chaser_pose_error_rad
                rate = support.target_rate_error_rad_s if k == 0 else support.chaser_rate_error_rad_s
                quaternion = _quaternion_box(Rotation.from_quat(state.quaternion_body_to_eci).as_matrix(),
                    pose, bodies[k, 6:10], ar)
                narrow(start+6, quaternion)
                narrow(start+10, centered(state.angular_velocity_body_rad_s, rate))
            encoders = observation.get('positioner_encoders')
            if encoders is not None:
                if set(encoders) != {member.id for member in fleet.members}:
                    raise ValueError('Contact encoder observations must name the exact fleet')
                for k, member in enumerate(fleet.members):
                    channel = encoders[member.id]
                    narrow(fleet.tip_slice.start+6*k, centered(channel['extension_body_m'], support.encoder_position_error_m))
                    narrow(fleet.tip_slice.start+6*k+3, centered(channel['extension_velocity_body_m_s'], support.encoder_velocity_error_m_s))
            # Preserve the common GNSS bias by using differences between its
            # simultaneous target and chaser products.
            for _ in range(2):
                for k, member in enumerate(fleet.members):
                    state = observation['chasers'][member.id]
                    for offset, value, radius in ((0, state.position_eci_m-target.position_eci_m, support.relative_position_error_m),
                                                  (3, state.velocity_eci_m_s-target.velocity_eci_m_s, support.relative_velocity_error_m_s)):
                        delta = ar.vector(centered(value, radius))
                        target_local = ar.vector(IntervalBox(lower[6+offset:9+offset], upper[6+offset:9+offset]))
                        start = 6+14*(k+1)+offset
                        narrow(start, ar.box(add(target_local, delta)))
                        chaser_local = ar.vector(IntervalBox(lower[start:start+3], upper[start:start+3]))
                        narrow(6+offset, ar.box(subtract(chaser_local, delta)))
                for offset, value, radius in ((0, target.position_eci_m, support.common_position_error_m),
                                              (3, target.velocity_eci_m_s, support.common_velocity_error_m_s)):
                    absolute = ar.vector(centered(value, radius))
                    local = ar.vector(IntervalBox(lower[6+offset:9+offset], upper[6+offset:9+offset]))
                    narrow(offset, ar.box(subtract(absolute, local)))
            gap_records = []
            for k, member in enumerate(fleet.members):
                values = ar.vector(IntervalBox(lower, upper))
                begin = 6+14*(k+1)
                target_body, body = values[6:20], values[begin:begin+14]
                rt, rj = quaternion_rotation(target_body[6:10]), quaternion_rotation(body[6:10])
                normal = tuple(map(ar.scalar, member.normal_body))
                direction = matvec(rt, normal)
                tip = values[fleet.tip_slice.start+6*k:fleet.tip_slice.start+6*(k+1)]
                lever = matvec(rj, add(tuple(map(ar.scalar, fleet.mounts[k])), tip[:3]))
                offset = subtract(body[:3], target_body[:3])
                constant = sum(a*b for a, b in zip(direction, lever))-sum(a*ar.scalar(b) for a, b in zip(normal, member.surface_anchor_body_m))
                gap = None
                if member.id in observation.get('contact_gaps_m', {}):
                    gap = centered((float(observation['contact_gaps_m'][member.id]),), support.local_range_error_m)
                if (not np.isfinite(bonds.engagement[k]) and
                        sum(Fraction(float(x))**2 for x in member.normal_body) == 1 and
                        member.id in observation.get('contact_wrenches_eci', {})):
                    wrench = observation['contact_wrenches_eci'][member.id]
                    force = ar.vector(centered(wrench['force_n'], support.wrench_force_error_n))
                    force_normal = ar.box((sum(a*b for a, b in zip(force, direction)),))
                    spin_t = matvec(rt, target_body[10:13]); spin_j = matvec(rj, body[10:13])
                    tip_velocity = add(add(body[3:6], cross(spin_j, lever)), matvec(rj, tip[3:]))
                    relative = subtract(subtract(tip_velocity, target_body[3:6]), cross(spin_t, add(offset, lever)))
                    speed = ar.box((sum(a*b for a, b in zip(direction, relative)),))
                    closing_upper = max(Fraction(0), -speed.lower[0])
                    if force_normal.lower[0] > 0:
                        comp_lower = max(Fraction(0), (force_normal.lower[0]-Fraction(options.compression_damping_n_s_m)*closing_upper)/
                                         Fraction(options.compression_stiffness_n_m))
                        comp_upper = max(Fraction(0), force_normal.upper[0]/Fraction(options.compression_stiffness_n_m))
                        mechanical = IntervalBox((-comp_upper,), (-comp_lower,))
                        gap = mechanical if gap is None else _intersection(gap, mechanical)
                if gap is None:
                    continue
                # Interval Gauss--Seidel on n_world.(p_bus-p_target)+b=g.
                # Coefficients that include zero cannot safely be divided.
                for axis in range(3):
                    coefficient = ar.box((direction[axis],))
                    if coefficient.lower[0] <= 0 <= coefficient.upper[0]:
                        continue
                    others = constant+sum(direction[j]*(ar.interval((lower[begin+j], upper[begin+j]))-
                        ar.interval((lower[6+j], upper[6+j]))) for j in range(3) if j != axis)
                    candidate = (ar.interval((gap.lower[0], gap.upper[0]))-others)/direction[axis]
                    target_coordinate = ar.interval((lower[6+axis], upper[6+axis]))
                    narrow(begin+axis, ar.box((candidate+target_coordinate,)))
                gap_records.append(dict(member_id=member.id, supported_gap_m=gap.as_dict()))
            rotations = initial_rotation_boxes(fleet, IntervalBox(lower, upper), ar,
                observation=observation, support=support)
        result = IntervalBox(lower, upper)
        if not result.contains(IntervalBox.point(representative[:fleet.ledger_start])):
            raise ValueError('Contact sensor support contradicts the executed numerical representative')
        record = dict(method='same-event-bounded-contact-set-intersection-v1',
            time_s=target.time_s, initial_fingerprint=fingerprint(initial.as_dict()),
            exact_elapsed_time_s=None if exact_elapsed_time_s is None else str(exact_elapsed_time_s),
            exact_time_s=None if exact_elapsed_time_s is None else str(Fraction(fleet.started)+Fraction(exact_elapsed_time_s)),
            conditioned_fingerprint=fingerprint(result.as_dict()), support=support.as_dict(), gap_constraints=gap_records,
            rotation_supports=[box.as_dict() for box in rotations],
            observation=dict(target=target.to_dict(), chasers={key: value.to_dict() for key, value in observation['chasers'].items()},
                contact_gaps_m=observation.get('contact_gaps_m'),
                contact_wrenches_eci={key: {name: np.asarray(vector).tolist() for name, vector in item.items()}
                    for key, item in observation.get('contact_wrenches_eci', {}).items()},
                positioner_encoders=None if observation.get('positioner_encoders') is None else
                    {key: {name: np.asarray(vector).tolist() for name, vector in item.items()}
                     for key, item in observation['positioner_encoders'].items()}))
        record['identity'] = fingerprint(record)
        return result, record
