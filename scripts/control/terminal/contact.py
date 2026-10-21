"""Sensing-only sufficient bridge to the physical capture speed gate.

This pointwise algebra preserves capture's separate nominal and tube norms.
It does not replace navigation support, continuous geometry, attitude, contact,
or lifecycle admission.  No physical truth or solver status is an input.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import math
from pathlib import Path

from orbexa.control.certification.reachability import IntervalBox
from orbexa.core.dynamics.reference_frame import exact, fingerprint

METHOD = 'sensed-split-capture-speed-v1'
FRAME = 'reference-orbit-frame-v1'
BODY_RATE_FRAME = 'target-body-frame-v1'
_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_SQRT_SCALE = 1 << 128


def _scalar(value, name, *, nonnegative=False, positive=False):
    try:
        result = exact(value)
        finite = math.isfinite(float(result))
    except (ArithmeticError, TypeError, ValueError) as error:
        raise ValueError(name + ' must be a finite exact declaration') from error
    if not finite or (nonnegative and result < 0) or (positive and result <= 0):
        raise ValueError(name + ' has an invalid finite bound')
    return result


def _vector(values, length, name, *, nonnegative=False):
    try:
        result = tuple(_scalar(value, name, nonnegative=nonnegative) for value in values)
    except TypeError as error:
        raise ValueError(name + ' requires a finite vector') from error
    if len(result) != length:
        raise ValueError(name + ' requires ' + str(length) + ' components')
    return result


def _identity(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('An explicit ' + name + ' is required')
    return value


def _box(value, dimension, name):
    if not isinstance(value, IntervalBox) or value.dimension != dimension:
        raise ValueError(name + ' requires a typed IntervalBox of the declared dimension')
    _vector(value.lower + value.upper, 2 * dimension, name)
    return value


def _norm_upper(values):
    """Integer arithmetic only: sqrt(sum(x*x)) rounded toward +infinity."""
    square = sum(value * value for value in values)
    numerator, denominator = square.numerator, square.denominator
    nroot, droot = math.isqrt(numerator), math.isqrt(denominator)
    if nroot * nroot == numerator and droot * droot == denominator:
        return Fraction(nroot, droot)
    scaled = numerator * _SQRT_SCALE * _SQRT_SCALE
    root = math.isqrt(scaled // denominator)
    if root * root * denominator < scaled:
        root += 1
    return Fraction(root, _SQRT_SCALE)


def _outward(value, *, upper):
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError('The contact bound has no finite binary64 display') from error
    if not math.isfinite(result):
        raise ValueError('The contact bound has no finite binary64 display')
    if ((exact(result) < value) if upper else (exact(result) > value)):
        result = math.nextafter(result, math.inf if upper else -math.inf)
    if not math.isfinite(result):
        raise ValueError('The contact bound has no finite outward binary64 display')
    return result


@dataclass(frozen=True)
class ContactAdmissionSource:
    """Identities for the same-event, authenticated navigation/goal inputs.

    The caller authenticates the referenced supports and the rigid fixed-body
    point goal.  This helper checks the declared coordinate convention and
    preserves the identities; it does not manufacture a navigation proof.
    """
    agent_id: str
    exact_q: object
    clock_fingerprint: str
    model_fingerprint: str
    navigation_fingerprint: str
    goal_bound_id: str
    frame: str = FRAME
    body_rate_frame: str = BODY_RATE_FRAME

    def __post_init__(self):
        for name in ('agent_id', 'clock_fingerprint', 'model_fingerprint',
                     'navigation_fingerprint', 'goal_bound_id'):
            _identity(getattr(self, name), name)
        object.__setattr__(self, 'exact_q', _scalar(self.exact_q, 'Exact event q'))
        if self.frame != FRAME or self.body_rate_frame != BODY_RATE_FRAME:
            raise ValueError('Contact admission requires the declared reference and body-rate frames')

    def as_dict(self):
        return dict(agent_id=self.agent_id, exact_q=str(self.exact_q),
            clock_fingerprint=self.clock_fingerprint, model_fingerprint=self.model_fingerprint,
            navigation_fingerprint=self.navigation_fingerprint, goal_bound_id=self.goal_bound_id,
            frame=self.frame, body_rate_frame=self.body_rate_frame)


@dataclass(frozen=True)
class ContactReferenceKinematics:
    """Source-bound numerical rigid-goal reference, possibly rounded.

    ``rotation_matrix`` maps target-body vectors to reference coordinates.
    The physical proper rotation must lie within the declared operator error
    of this matrix; this matrix itself need not be exactly orthogonal.  The
    source may be a same-event goal bound or a separately authenticated future
    forecast.  It does not assert that a future navigation estimate exists.
    """
    rotation_matrix: tuple
    body_rate_rad_s: tuple
    rotation_operator_error_upper: object
    goal_bound_id: str
    clock_fingerprint: str
    exact_q: object
    frame: str = FRAME
    body_rate_frame: str = BODY_RATE_FRAME

    def __post_init__(self):
        try:
            rotation = tuple(_vector(row, 3, 'Reference rotation') for row in self.rotation_matrix)
        except TypeError as error:
            raise ValueError('Reference rotation requires a finite three by three matrix') from error
        if len(rotation) != 3:
            raise ValueError('Reference rotation requires three rows')
        object.__setattr__(self, 'rotation_matrix', rotation)
        object.__setattr__(self, 'body_rate_rad_s', _vector(self.body_rate_rad_s, 3, 'Reference body rate'))
        object.__setattr__(self, 'rotation_operator_error_upper', _scalar(
            self.rotation_operator_error_upper, 'Reference rotation operator error', nonnegative=True))
        object.__setattr__(self, 'exact_q', _scalar(self.exact_q, 'Reference exact q'))
        _identity(self.goal_bound_id, 'reference goal-bound identity')
        _identity(self.clock_fingerprint, 'reference clock identity')
        if self.frame != FRAME or self.body_rate_frame != BODY_RATE_FRAME:
            raise ValueError('Reference kinematics requires the declared coordinate frames')

    def as_dict(self):
        return dict(rotation_matrix=tuple(tuple(map(str, row)) for row in self.rotation_matrix),
            body_rate_rad_s=tuple(map(str, self.body_rate_rad_s)),
            rotation_operator_error_upper=str(self.rotation_operator_error_upper),
            goal_bound_id=self.goal_bound_id, clock_fingerprint=self.clock_fingerprint,
            exact_q=str(self.exact_q), frame=self.frame, body_rate_frame=self.body_rate_frame)


@dataclass(frozen=True)
class ContactGoalSpeedBound:
    """Pure goal-only constants for terminal or independently typed forecast use."""
    observed_goal: tuple
    goal_position_error_norm_upper: Fraction
    goal_velocity_error_norm_upper: Fraction
    clock_rate_s_inv: IntervalBox
    target_body_rate_rad_s: IntervalBox
    reference_kinematics: ContactReferenceKinematics | None
    clock_rate_upper: Fraction
    body_rate_norm_upper: Fraction
    angular_coefficient_upper: Fraction
    split_goal_speed_upper_m_s: Fraction
    reference_body_rate_norm_upper: Fraction | None
    body_rate_difference_norm_upper: Fraction | None
    angular_mismatch_norm_upper: Fraction | None
    goal_position_norm_upper: Fraction | None
    kinematic_residual_endpoints: tuple | None
    kinematic_residual_norm_upper_m_s: Fraction | None
    kinematic_goal_speed_upper_m_s: Fraction | None
    goal_speed_upper_m_s: Fraction

    def as_dict(self):
        scalar_names = ('clock_rate_upper', 'body_rate_norm_upper', 'angular_coefficient_upper',
            'split_goal_speed_upper_m_s', 'reference_body_rate_norm_upper',
            'body_rate_difference_norm_upper', 'angular_mismatch_norm_upper',
            'goal_position_norm_upper', 'kinematic_residual_norm_upper_m_s',
            'kinematic_goal_speed_upper_m_s', 'goal_speed_upper_m_s')
        boxes = {name: dict(exact_lower=tuple(map(str, getattr(self, name).lower)),
                           exact_upper=tuple(map(str, getattr(self, name).upper))) for name in
                 ('clock_rate_s_inv', 'target_body_rate_rad_s')}
        result = dict(method='rigid-goal-speed-constant-v1', source_sha256=_SOURCE_SHA256,
            inputs=dict(observed_goal=tuple(map(str, self.observed_goal)),
                goal_position_error_norm_upper=str(self.goal_position_error_norm_upper),
                goal_velocity_error_norm_upper=str(self.goal_velocity_error_norm_upper), **boxes,
                reference_kinematics=None if self.reference_kinematics is None else self.reference_kinematics.as_dict()),
            exact_upper_bounds={name: None if getattr(self, name) is None else str(getattr(self, name))
                for name in scalar_names},
            kinematic_residual_endpoints=None if self.kinematic_residual_endpoints is None else
                tuple(tuple(map(str, row)) for row in self.kinematic_residual_endpoints),
            formula='K=min(nu_up*Ev+(nu_up+Wup)*Ep, '
                '(gamma*norm(w_hat)+norm(body_rate_box-w_hat)_up)*norm(g_p)'
                '+max_endpoint_norm(nu*(g_v+Z*g_p)-(Rhat*w_hat) cross g_p))',
            units=dict(goal_speed='m/s', position='m', anomaly_velocity='m/rad',
                       angular_rate='rad/s', rotation_error='operator norm'),
            assumption_scope='Physical target rotation is proper and lies in the declared operator support; '
                'body rate lies in its box; caller authenticates goal/clock/forecast linkage')
        result['identity'] = fingerprint(result)
        return result

    @classmethod
    def from_dict(cls, record):
        """Recompute every constant and require the complete current evidence record."""
        try:
            raw = record['inputs']
            parsed = dict(raw)
            for name in ('clock_rate_s_inv', 'target_body_rate_rad_s'):
                parsed[name] = IntervalBox(tuple(raw[name]['exact_lower']), tuple(raw[name]['exact_upper']))
            if raw['reference_kinematics'] is not None:
                parsed['reference_kinematics'] = ContactReferenceKinematics(**raw['reference_kinematics'])
            rebuilt = contact_goal_speed_bound(**parsed)
            if fingerprint(record) != fingerprint(rebuilt.as_dict()):
                raise ValueError('Goal speed evidence changed or is from a different source')
            return rebuilt
        except (KeyError, TypeError) as error:
            raise ValueError('Malformed source-bound goal speed evidence') from error


def contact_goal_speed_bound(observed_goal, goal_position_error_norm_upper,
                             goal_velocity_error_norm_upper, clock_rate_s_inv,
                             target_body_rate_rad_s, *, reference_kinematics=None):
    """Bound the nominal goal's rigid-relative speed without future sensing.

    The norm of an affine vector in scalar nu is convex, so its maximum on
    the exact clock interval occurs at an endpoint.  No independence between
    Cartesian quaternion components or exact orthogonality of Rhat is assumed.
    """
    goal = _vector(observed_goal, 6, 'Observed numerical goal')
    ep = _scalar(goal_position_error_norm_upper, 'Goal position error', nonnegative=True)
    ev = _scalar(goal_velocity_error_norm_upper, 'Goal velocity error', nonnegative=True)
    clock = _box(clock_rate_s_inv, 1, 'Clock rate')
    rates = _box(target_body_rate_rad_s, 3, 'Body rate')
    if clock.lower[0] <= 0:
        raise ValueError('The reference clock rate must be proved positive')
    nu = clock.upper[0]
    w = _norm_upper(tuple(max(abs(lo), abs(hi)) for lo, hi in zip(rates.lower, rates.upper)))
    angular = nu + w
    split = nu * ev + angular * ep
    if reference_kinematics is None:
        _outward(split, upper=True)
        return ContactGoalSpeedBound(goal, ep, ev, clock, rates, None, nu, w, angular,
            split, None, None, None, None, None, None, None, split)
    if not isinstance(reference_kinematics, ContactReferenceKinematics):
        raise ValueError('Correlated goal support requires typed source-bound reference kinematics')
    reference = reference_kinematics
    reference_rate = _norm_upper(reference.body_rate_rad_s)
    rate_difference = _norm_upper(tuple(max(abs(lo - center), abs(hi - center)) for lo, hi, center in
        zip(rates.lower, rates.upper, reference.body_rate_rad_s)))
    mismatch = reference.rotation_operator_error_upper * reference_rate + rate_difference
    gp = goal[:3]
    gp_norm = _norm_upper(gp)
    mapped_rate = tuple(sum(r[i] * value for i, value in enumerate(reference.body_rate_rad_s))
                        for r in reference.rotation_matrix)
    mapped_cross_goal = (mapped_rate[1] * gp[2] - mapped_rate[2] * gp[1],
        mapped_rate[2] * gp[0] - mapped_rate[0] * gp[2],
        mapped_rate[0] * gp[1] - mapped_rate[1] * gp[0])
    goal_kinematic_velocity = (goal[3] - gp[1], goal[4] + gp[0], goal[5])
    endpoints = tuple(tuple(rate * a - b for a, b in zip(goal_kinematic_velocity, mapped_cross_goal))
                      for rate in (clock.lower[0], clock.upper[0]))
    defect = max(_norm_upper(endpoint) for endpoint in endpoints)
    correlated = mismatch * gp_norm + defect
    bound = min(split, correlated)
    for value in (split, correlated, bound):
        _outward(value, upper=True)
    return ContactGoalSpeedBound(goal, ep, ev, clock, rates, reference, nu, w, angular,
        split, reference_rate, rate_difference, mismatch, gp_norm, endpoints, defect, correlated, bound)


@dataclass(frozen=True)
class ContactAdmissionInputs:
    """Separate nominal, tube, observation, clock, and hardware declarations.

    State vectors are (position m, dposition/dq m/rad) in FRAME.  The signed
    body-rate box is in BODY_RATE_FRAME; its norm is invariant under the proper
    target and reference rotations.  All scalar error supports are upper
    bounds.  Hardware values must come from the unchanged capture/vehicle
    declaration, which the integrating caller must bind to its source record.
    """
    nominal_center: tuple
    observed_goal: tuple
    phi: tuple
    omega: tuple
    lambda_gain: tuple
    goal_position_error_norm_upper: object
    goal_velocity_error_norm_upper: object
    clock_rate_s_inv: IntervalBox
    target_body_rate_rad_s: IntervalBox
    max_spin_error_rad_s: object
    gripper_offset_m: object
    max_relative_speed_m_s: object

    def __post_init__(self):
        for name, length in (('nominal_center', 6), ('observed_goal', 6),
                             ('phi', 3), ('omega', 3), ('lambda_gain', 3)):
            object.__setattr__(self, name, _vector(getattr(self, name), length, name,
                nonnegative=name in ('phi', 'omega', 'lambda_gain')))
        for name in ('goal_position_error_norm_upper', 'goal_velocity_error_norm_upper',
                     'max_spin_error_rad_s', 'gripper_offset_m', 'max_relative_speed_m_s'):
            object.__setattr__(self, name, _scalar(getattr(self, name), name,
                nonnegative=True, positive=name == 'max_relative_speed_m_s'))
        for name, dimension in (('clock_rate_s_inv', 1), ('target_body_rate_rad_s', 3)):
            _box(getattr(self, name), dimension, name)
        if self.clock_rate_s_inv.lower[0] <= 0:
            raise ValueError('The reference clock rate must be proved positive')

    def as_dict(self):
        vectors = {name: tuple(map(str, getattr(self, name))) for name in
                   ('nominal_center', 'observed_goal', 'phi', 'omega', 'lambda_gain')}
        scalars = {name: str(getattr(self, name)) for name in
                   ('goal_position_error_norm_upper', 'goal_velocity_error_norm_upper',
                    'max_spin_error_rad_s', 'gripper_offset_m', 'max_relative_speed_m_s')}
        boxes = {name: dict(exact_lower=tuple(map(str, getattr(self, name).lower)),
                           exact_upper=tuple(map(str, getattr(self, name).upper))) for name in
                 ('clock_rate_s_inv', 'target_body_rate_rad_s')}
        return vectors | scalars | boxes


@dataclass(frozen=True)
class ContactSpeedAdmission:
    source: ContactAdmissionSource
    inputs: ContactAdmissionInputs
    nominal_position_error_abs: tuple
    nominal_velocity_error_abs: tuple
    tube_velocity_radius_abs: tuple
    body_rate_abs: tuple
    nominal_position_norm_upper: Fraction
    nominal_velocity_norm_upper: Fraction
    tube_position_norm_upper: Fraction
    tube_velocity_norm_upper: Fraction
    body_rate_norm_upper: Fraction
    position_split_norm_upper: Fraction
    velocity_split_norm_upper: Fraction
    angular_coefficient_upper: Fraction
    linear_speed_term_upper_m_s: Fraction
    rotation_speed_term_upper_m_s: Fraction
    tip_speed_term_upper_m_s: Fraction
    speed_upper_m_s: Fraction
    goal_speed_bounds: ContactGoalSpeedBound | None = None

    @property
    def admissible(self):
        return self.speed_upper_m_s <= self.inputs.max_relative_speed_m_s

    @property
    def margin_m_s(self):
        return self.inputs.max_relative_speed_m_s - self.speed_upper_m_s

    def as_dict(self):
        vector_names = ('nominal_position_error_abs', 'nominal_velocity_error_abs',
                        'tube_velocity_radius_abs', 'body_rate_abs')
        scalar_names = ('nominal_position_norm_upper', 'nominal_velocity_norm_upper',
            'tube_position_norm_upper', 'tube_velocity_norm_upper', 'body_rate_norm_upper',
            'position_split_norm_upper', 'velocity_split_norm_upper', 'angular_coefficient_upper',
            'linear_speed_term_upper_m_s', 'rotation_speed_term_upper_m_s',
            'tip_speed_term_upper_m_s', 'speed_upper_m_s')
        result = dict(method=METHOD, source_sha256=_SOURCE_SHA256,
            status='SUFFICIENT' if self.admissible else 'REFUSED', admissible=self.admissible,
            source=self.source.as_dict(), inputs=self.inputs.as_dict(),
            formula='nu_up*(||c_v-g_v||+||Phi+abs(Lambda)*Omega||+E_v)'
                '+(nu_up+||body_rate_box||_up)*(||c_p-g_p||+||Omega||+E_p)'
                '+max_spin_error*gripper_offset',
            goal_kinematic_identity='nu*(g_true_v+z_cross(g_true_p))'
                '-omega_reference_cross(g_true_p)=0',
            units=dict(position='m', anomaly_velocity='m/rad', clock_rate='rad/s',
                       body_rate='rad/s', lambda_gain='1/rad', speed='m/s'),
            assumption_scope='Authenticated same-event rigid fixed-body-point goal, true proper rotations, '
                'and declared navigation/tube/hardware supports; pointwise speed only',
            components={name: tuple(map(str, getattr(self, name))) for name in vector_names},
            exact_upper_bounds={name: str(getattr(self, name)) for name in scalar_names},
            exact_margin_m_s=str(self.margin_m_s),
            speed_upper_m_s=_outward(self.speed_upper_m_s, upper=True),
            margin_lower_m_s=_outward(self.margin_m_s, upper=False))
        if self.goal_speed_bounds is not None:
            result['formula'] = ('nu_up*(||c_v-g_v||+||Phi+abs(Lambda)*Omega||)'
                '+(nu_up+||body_rate_box||_up)*(||c_p-g_p||+||Omega||)'
                '+goal_speed_constant+max_spin_error*gripper_offset')
            result['goal_speed_bounds'] = self.goal_speed_bounds.as_dict()
        result['identity'] = fingerprint(result)
        return result

    @property
    def identity(self):
        return self.as_dict()['identity']


def contact_speed_admission(inputs: ContactAdmissionInputs,
                            source: ContactAdmissionSource, *,
                            reference_kinematics: ContactReferenceKinematics | None = None) -> ContactSpeedAdmission:
    """Return an exact outward sufficient bound for capture's split speed gate.

    For the true rigid goal g, subtract its kinematic identity from the nominal
    center c, then use triangle inequalities separately for the nominal error,
    observed-goal error, and capture's tube radius.  Recombining componentwise
    nominal and radius errors inside one norm would not bound the split gate.
    """
    if not isinstance(inputs, ContactAdmissionInputs) or not isinstance(source, ContactAdmissionSource):
        raise ValueError('Typed contact inputs and same-event source identities are required')
    if reference_kinematics is not None:
        if not isinstance(reference_kinematics, ContactReferenceKinematics):
            raise ValueError('Correlated contact admission requires typed reference kinematics')
        for name, source_name in (('goal_bound_id', 'goal_bound_id'), ('clock_fingerprint', 'clock_fingerprint'),
                                  ('exact_q', 'exact_q'), ('frame', 'frame'), ('body_rate_frame', 'body_rate_frame')):
            if getattr(reference_kinematics, name) != getattr(source, source_name):
                raise ValueError('Reference kinematics and contact admission source differ: ' + name)
    position = tuple(abs(c - g) for c, g in zip(inputs.nominal_center[:3], inputs.observed_goal[:3]))
    velocity = tuple(abs(c - g) for c, g in zip(inputs.nominal_center[3:], inputs.observed_goal[3:]))
    radius = tuple(f + abs(gain) * o for f, gain, o in zip(inputs.phi, inputs.lambda_gain, inputs.omega))
    body_rate = tuple(max(abs(lo), abs(hi)) for lo, hi in
                      zip(inputs.target_body_rate_rad_s.lower, inputs.target_body_rate_rad_s.upper))
    np, nv, rp, rv, w = map(_norm_upper, (position, velocity, inputs.omega, radius, body_rate))
    p = np + rp + inputs.goal_position_error_norm_upper
    v = nv + rv + inputs.goal_velocity_error_norm_upper
    nu = inputs.clock_rate_s_inv.upper[0]
    angular = nu + w
    linear_term = nu * v
    rotation_term = angular * p
    tip_term = inputs.max_spin_error_rad_s * inputs.gripper_offset_m
    bound = linear_term + rotation_term + tip_term
    goal_bounds = None
    if reference_kinematics is not None:
        goal_bounds = contact_goal_speed_bound(inputs.observed_goal,
            inputs.goal_position_error_norm_upper, inputs.goal_velocity_error_norm_upper,
            inputs.clock_rate_s_inv, inputs.target_body_rate_rad_s,
            reference_kinematics=reference_kinematics)
        linear_term = nu * (nv + rv)
        rotation_term = angular * (np + rp)
        bound = linear_term + rotation_term + goal_bounds.goal_speed_upper_m_s + tip_term
    # Refuse a nonfinite presentation before a caller could archive/act on it.
    _outward(bound, upper=True)
    _outward(inputs.max_relative_speed_m_s - bound, upper=False)
    return ContactSpeedAdmission(source, inputs, position, velocity, radius, body_rate,
        np, nv, rp, rv, w, p, v, angular, linear_term, rotation_term, tip_term, bound, goal_bounds)


@dataclass(frozen=True)
class ContactReachPolicy:
    """Unchanged physical arm/mount limits for one assigned target patch.

    Standoff is the COM goal's distance along the patch's outward normal.
    Gripper offset is supplied separately by the same vehicle declaration used
    for speed admission. The expected proper mount points its gripper axis
    along that normal; the allowed orientation error is a rotation angle.
    """
    standoff_m: object
    arm_reach_m: object
    max_orientation_error_rad: object

    def __post_init__(self):
        for name in ('standoff_m', 'arm_reach_m', 'max_orientation_error_rad'):
            object.__setattr__(self, name, _scalar(getattr(self, name), name,
                nonnegative=True, positive=name == 'arm_reach_m'))

    def position_allowance_m(self, gripper_offset_m):
        """Exact combined split-position allowance; a negative value refuses."""
        offset = _scalar(gripper_offset_m, 'Gripper offset', nonnegative=True)
        reach = self.arm_reach_m - abs(self.standoff_m - offset) - offset * self.max_orientation_error_rad
        gap = self.standoff_m - offset
        return min(reach, gap)

    def as_dict(self):
        result = dict(method='assigned-physical-contact-reach-policy-v1', source_sha256=_SOURCE_SHA256,
            standoff_m=str(self.standoff_m), arm_reach_m=str(self.arm_reach_m),
            max_orientation_error_rad=str(self.max_orientation_error_rad),
            units=dict(standoff='m', arm_reach='m', orientation_error='rad'),
            mounting_convention='Expected proper bus gripper axis is the assigned outward patch normal')
        result['identity'] = fingerprint(result)
        return result

    @property
    def identity(self):
        return self.as_dict()['identity']

    @classmethod
    def from_dict(cls, record):
        try:
            rebuilt = cls(**{name: record[name] for name in
                ('standoff_m', 'arm_reach_m', 'max_orientation_error_rad')})
            if fingerprint(record) != fingerprint(rebuilt.as_dict()):
                raise ValueError('Contact reach policy changed or is from a different source')
            return rebuilt
        except (KeyError, TypeError) as error:
            raise ValueError('Malformed source-bound contact reach policy') from error


@dataclass(frozen=True)
class ContactReachAdmission:
    source: ContactAdmissionSource
    inputs: ContactAdmissionInputs
    policy: ContactReachPolicy
    nominal_position_error_abs: tuple
    nominal_position_norm_upper: Fraction
    tube_position_norm_upper: Fraction
    position_split_norm_upper: Fraction
    nominal_tip_gap_m: Fraction
    orientation_tip_displacement_upper_m: Fraction
    position_allowance_m: Fraction
    reach_upper_m: Fraction
    gap_lower_m: Fraction

    @property
    def reach_admissible(self):
        return self.reach_upper_m <= self.policy.arm_reach_m

    @property
    def nonpenetration_admissible(self):
        return self.gap_lower_m >= 0

    @property
    def admissible(self):
        return self.reach_admissible and self.nonpenetration_admissible

    @property
    def reach_margin_m(self):
        return self.policy.arm_reach_m - self.reach_upper_m

    def as_dict(self):
        scalar_names = ('nominal_position_norm_upper', 'tube_position_norm_upper',
            'position_split_norm_upper', 'nominal_tip_gap_m', 'orientation_tip_displacement_upper_m',
            'position_allowance_m', 'reach_upper_m', 'gap_lower_m')
        result = dict(method='sensed-split-capture-reach-v1', source_sha256=_SOURCE_SHA256,
            status='SUFFICIENT' if self.admissible else 'REFUSED', admissible=self.admissible,
            reach_admissible=self.reach_admissible, nonpenetration_admissible=self.nonpenetration_admissible,
            source=self.source.as_dict(), inputs=self.inputs.as_dict(), policy=self.policy.as_dict(),
            formula='B=||c_p-g_p||+||Omega||+Ep; '
                'reach=abs(standoff-offset)+offset*orientation_limit+B; gap=standoff-offset-B',
            units=dict(position='m', standoff='m', offset='m', orientation_limit='rad', reach='m', gap='m'),
            assumption_scope='Authenticated same-event rigid fixed-body-point COM goal and position support; '
                'proper actual/expected mount rotations with the declared angular error; '
                'caller authenticates assigned standoff and unchanged vehicle/arm limits. '
                'Reach and nonpenetration only; other patch/body/peer geometry gates remain required.',
            components=dict(nominal_position_error_abs=tuple(map(str, self.nominal_position_error_abs))),
            exact_bounds={name: str(getattr(self, name)) for name in scalar_names},
            exact_reach_margin_m=str(self.reach_margin_m),
            reach_upper_m=_outward(self.reach_upper_m, upper=True),
            reach_margin_lower_m=_outward(self.reach_margin_m, upper=False),
            gap_lower_m=_outward(self.gap_lower_m, upper=False))
        result['identity'] = fingerprint(result)
        return result

    @property
    def identity(self):
        return self.as_dict()['identity']


def contact_reach_admission(inputs: ContactAdmissionInputs, source: ContactAdmissionSource,
                            policy: ContactReachPolicy) -> ContactReachAdmission:
    """Sufficient split reach and nonpenetration bounds without truth feedback.

    For proper mount error angle theta, gripper displacement from its ideal
    direction is at most 2*l*sin(theta/2) <= l*theta. Its inward projection can
    never exceed l, so the normal gap needs no orientation-error debit. The
    saved nominal position and capture's tube radius remain separate norms.
    """
    if not isinstance(inputs, ContactAdmissionInputs) or not isinstance(source, ContactAdmissionSource):
        raise ValueError('Typed contact inputs and same-event source identities are required')
    if not isinstance(policy, ContactReachPolicy):
        raise ValueError('A typed source-bound contact reach policy is required')
    position = tuple(abs(c - g) for c, g in zip(inputs.nominal_center[:3], inputs.observed_goal[:3]))
    nominal, tube = _norm_upper(position), _norm_upper(inputs.omega)
    split = nominal + tube + inputs.goal_position_error_norm_upper
    offset = inputs.gripper_offset_m
    tip_gap = policy.standoff_m - offset
    orientation = offset * policy.max_orientation_error_rad
    reach = abs(tip_gap) + orientation + split
    gap = tip_gap - split
    allowance = policy.position_allowance_m(offset)
    _outward(reach, upper=True)
    for value in (policy.arm_reach_m - reach, gap, allowance):
        _outward(value, upper=False)
    return ContactReachAdmission(source, inputs, policy, position, nominal, tube, split,
        tip_gap, orientation, allowance, reach, gap)
