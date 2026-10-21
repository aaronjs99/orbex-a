"""Finite-budget continuous enclosures for the declared compliant-pad model.

Picard/first-exit inclusions carry the endpoint set to the next held interval.
The RK4 state is only a numerical representative; it never replaces that set.
The unilateral dashpot's contact discontinuity is covered by a branch hull.
"""
from __future__ import annotations
from orbexa.layout import PACKAGE_ROOT

from dataclasses import asdict, dataclass, replace
from fractions import Fraction
import hashlib
import copy
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from orbexa.control.certification.reachability import IntervalBox, _ARB_LOCK, _ArbArithmetic, _ProofUnresolved
from orbexa.core.dynamics.reference_frame import fingerprint
from orbexa.simulation.contact.dynamics.rotation import auxiliary_initial, rotation_derivative
from orbexa.simulation.contact.dynamics.strain import initial_strain_boxes, potential_box
from orbexa.simulation.contact.dynamics.wrench import initial_wrench_boxes, wrench_derivatives
from orbexa.simulation.numerics.rigid_interval import add, cross, inverse3, matvec, norm, norm_upper, normalized_quaternion, quaternion_conjugate, quaternion_product, quaternion_rotation, rotation_vector, scale, square, subtract, transpose


_SOURCE = {name: hashlib.sha256((PACKAGE_ROOT/name).read_bytes()).hexdigest()
    for name in (
                    'simulation/contact/enclosure.py',
                    'simulation/numerics/rigid_interval.py',
                    'simulation/contact/dynamics/rotation.py',
                    'simulation/contact/dynamics/strain.py',
                    'simulation/contact/dynamics/wrench.py',
                    'layout.py',
                )}


def _max_zero(value, ar):
    box = ar.box((value,))
    return ar.interval((max(0, box.lower[0]), max(0, box.upper[0])))


def _dot(a, b):
    return sum(x*y for x, y in zip(a, b))


@dataclass(frozen=True)
class ContactStepCertificate:
    status: str
    reason: str
    context: dict
    initial: IntervalBox
    endpoint: IntervalBox | None = None
    swept: IntervalBox | None = None
    checks: dict | None = None
    refinement_attempts: tuple = ()
    rotation_endpoint: tuple = ()
    rotation_swept: tuple = ()
    gap_endpoint: tuple = ()
    gap_swept: tuple = ()
    strain_endpoint: tuple = ()
    strain_swept: tuple = ()
    endpoint_potential: IntervalBox | None = None
    wrench_endpoint: tuple = ()
    wrench_swept: tuple = ()

    @property
    def passed(self):
        return self.status == 'PROVEN'

    def as_dict(self):
        return dict(method='arb-compliant-contact-held-flow-v1', status=self.status,
            passed=self.passed, reason=self.reason, source_files=dict(_SOURCE),
            context=copy.deepcopy(self.context), initial=self.initial.as_dict(),
            endpoint=None if self.endpoint is None else self.endpoint.as_dict(),
            swept=None if self.swept is None else self.swept.as_dict(), checks=copy.deepcopy(self.checks or {}),
            refinement_attempts=copy.deepcopy(list(self.refinement_attempts)),
            rotation_endpoint=[box.as_dict() for box in self.rotation_endpoint],
            rotation_swept=[box.as_dict() for box in self.rotation_swept],
            gap_endpoint=[box.as_dict() for box in self.gap_endpoint],
            gap_swept=[box.as_dict() for box in self.gap_swept],
            strain_endpoint=[box.as_dict() for box in self.strain_endpoint],
            strain_swept=[box.as_dict() for box in self.strain_swept],
            endpoint_potential=None if self.endpoint_potential is None else self.endpoint_potential.as_dict(),
            wrench_endpoint=[box.as_dict() for box in self.wrench_endpoint],
            wrench_swept=[box.as_dict() for box in self.wrench_swept],
            scope='Declared unit-quaternion rigid ODE, unilateral compliant pads and fixed held commands/bond parameters; conditional initial set, exact specified geometry and the maintained typed J2/drag environment when supplied')


class _ContactField:
    def __init__(self, fleet, options, bonds, actuation, ar, rest_enclosures=None, environment_model=None,
                 engagement_times=None):
        self.fleet, self.o, self.bonds, self.actuation, self.ar = fleet, options, bonds, actuation, ar
        controls = np.asarray(actuation.controls)
        extension = np.asarray(actuation.extension_acceleration)
        torque_limit = Fraction(2)*Fraction(options.attitude_moment_arm_m)*Fraction(options.attitude_nozzle_force_n)
        if (controls.shape != (fleet.body_count, 6) or extension.shape != (fleet.chaser_count, 3)
                or np.any(~np.isfinite(controls)) or np.any(~np.isfinite(extension))
                or np.any(controls[0] != 0)):
            raise _ProofUnresolved('held contact actuator declaration is invalid')
        if any(abs(Fraction(float(x))) > Fraction(options.translation_force_n) for row in controls[1:] for x in row[:3]):
            raise _ProofUnresolved('held contact translation command exceeds declared nozzle authority')
        if any(abs(Fraction(float(x))) > torque_limit for row in controls[1:] for x in row[3:]):
            raise _ProofUnresolved('held contact torque command exceeds declared nozzle authority')
        if any(sum(Fraction(float(x))**2 for x in row) > Fraction(options.extension_acceleration_m_s2)**2 for row in extension):
            raise _ProofUnresolved('held contact positioner acceleration exceeds declared authority')
        self.specific = tuple(tuple(tuple(ar.scalar(x) for x in row) for row in matrix) for matrix in fleet.specific)
        self.inverse = tuple(inverse3(matrix) for matrix in self.specific)
        self.mounts = tuple(tuple(ar.scalar(x) for x in row) for row in fleet.mounts)
        self.anchors = tuple(tuple(ar.scalar(x) for x in row) for row in fleet.anchors)
        self.normals = tuple(tuple(ar.scalar(x) for x in row) for row in fleet.normals)
        self.controls = tuple(tuple(ar.scalar(x) for x in row) for row in actuation.controls)
        self.extension = tuple(tuple(ar.scalar(x) for x in row) for row in actuation.extension_acceleration)
        self.environment = environment_model
        self.environment_parameters = None if environment_model is None else environment_model.force_parameters(ar)
        if environment_model is not None:
            declared = dict(environment_model.dry_mass)
            expected = dict(target=Fraction(float(fleet.dry[0])),
                **{member.id: Fraction(float(fleet.dry[k+1])) for k, member in enumerate(fleet.members)})
            if declared != expected:
                raise _ProofUnresolved('contact environment mass/fleet differs from the admitted bodies')
        self.rest_quaternions, self.rest_positions = [], []
        self.engagement_times = (tuple(None if not np.isfinite(value) else Fraction(float(value))
            for value in bonds.engagement) if engagement_times is None else tuple(engagement_times))
        if len(self.engagement_times)!=fleet.chaser_count or any((value is None)!=not_engaged
                for value, not_engaged in zip(self.engagement_times, ~np.isfinite(bonds.engagement))):
            raise _ProofUnresolved('exact contact engagement clocks do not match the bond state')
        for k, engaged in enumerate(bonds.engagement):
            rest = None if rest_enclosures is None else rest_enclosures[k]
            nominal_quaternion = None if not np.isfinite(engaged) else normalized_quaternion(
                tuple(ar.scalar(x) for x in Rotation.from_matrix(bonds.rotation_rest[k]).as_quat()), ar)
            if rest is not None and nominal_quaternion is not None:
                center = tuple((lo+hi)/2 for lo, hi in zip(rest['quaternion'].lower, rest['quaternion'].upper))
                if sum(float(value)*float(sum(ar.box((coordinate,)).lower+ar.box((coordinate,)).upper)/2)
                       for value, coordinate in zip(center, nominal_quaternion)) < 0:
                    nominal_quaternion = tuple(-value for value in nominal_quaternion)
            self.rest_quaternions.append(nominal_quaternion if rest is None or nominal_quaternion is None else
                ar.vector(rest['quaternion'].hull(ar.box(nominal_quaternion))))
            rest_position = IntervalBox.point(bonds.bond_rest[k])
            if rest is not None:
                rest_position = rest_position.hull(rest['position'])
            self.rest_positions.append(ar.vector(rest_position))

    def unpack(self, values):
        fleet = self.fleet
        bodies = tuple(tuple(values[6+14*k:6+14*(k+1)]) for k in range(fleet.body_count))
        start = fleet.tip_slice.start
        tips = tuple(tuple(values[start+6*k:start+6*(k+1)]) for k in range(fleet.chaser_count))
        return bodies, tips

    def gravity(self, position):
        length = norm(position, self.ar)
        if not length > 0:
            raise _ProofUnresolved('contact gravity set includes the origin')
        return scale(-self.ar.scalar(self.fleet.mu)/(length*length*length), position)

    def evaluate(self, time, values):
        fleet, o, ar = self.fleet, self.o, self.ar
        body, tips = self.unpack(values)
        rotation_start = fleet.ledger_start
        gap_start = rotation_start+9*fleet.body_count
        strain_start = gap_start+fleet.chaser_count
        wrench_start = strain_start+7*fleet.chaser_count
        augmented = len(values) in (strain_start, wrench_start, wrench_start+6*fleet.chaser_count)
        strained = len(values) in (wrench_start, wrench_start+6*fleet.chaser_count)
        wrenched = len(values) == wrench_start+6*fleet.chaser_count
        rotations = (tuple(tuple(tuple(values[rotation_start+9*k+3*i:rotation_start+9*k+3*i+3])
            for i in range(3)) for k in range(fleet.body_count)) if augmented else
            tuple(quaternion_rotation(row[6:10]) for row in body))
        spins = tuple(matvec(r, row[10:13]) for r, row in zip(rotations, body))
        masses = tuple(ar.scalar(dry)+row[13] for dry, row in zip(fleet.dry, body))
        if any(not mass > 0 for mass in masses):
            raise _ProofUnresolved('contact mass is not proved positive')
        forces = [matvec(r, command[:3]) for r, command in zip(rotations, self.controls)]
        moments = [tuple(command[3:]) for command in self.controls]
        reference_gravity = self.gravity(values[:3])
        for k, row in enumerate(body):
            absolute = add(values[:3], row[:3])
            forces[k] = add(forces[k], scale(masses[k], subtract(self.gravity(absolute), reference_gravity)))
            if self.environment is not None:
                name = 'target' if k == 0 else fleet.members[k-1].id
                extra = self.environment.evaluate(name, absolute, add(values[3:6], row[3:6]),
                    row[13], ar, self.environment_parameters)
                forces[k] = add(forces[k], scale(masses[k], extra))
            if fleet.gravity_gradient:
                length = norm(absolute, ar)
                radial = matvec(transpose(rotations[k]), scale(1/length, absolute))
                gg = scale(3*ar.scalar(fleet.mu)/(length*length*length),
                    cross(radial, scale(masses[k], matvec(self.specific[k], radial))))
                moments[k] = add(moments[k], gg)
        rows = []
        for k, member in enumerate(fleet.members):
            j = k+1
            rt, rj = rotations[0], rotations[j]
            tip_offset = matvec(rj, add(self.mounts[k], tips[k][:3]))
            tip = add(body[j][:3], tip_offset)
            # Retain the exact SO(3) identities instead of multiplying
            # independent interval copies of R^T R through fixed anchors
            # or the target's own angular velocity.
            com_target = matvec(transpose(rt), subtract(body[j][:3], body[0][:3]))
            relative_rotation = tuple(tuple(sum(rt[k][i]*rj[k][j] for k in range(3))
                for j in range(3)) for i in range(3))
            lever_target = matvec(relative_rotation, add(self.mounts[k], tips[k][:3]))
            e = subtract(add(com_target, lever_target), self.anchors[k])
            tip_velocity = add(add(body[j][3:6], cross(spins[j], tip_offset)), matvec(rj, tips[k][3:]))
            spin = subtract(matvec(relative_rotation, body[j][10:13]), body[0][10:13])
            u = add(subtract(matvec(transpose(rt), subtract(body[j][3:6], body[0][3:6])),
                cross(body[0][10:13], com_target)),
                add(cross(spin, lever_target), matvec(relative_rotation, tips[k][3:])))
            normal = self.normals[k]
            gap, speed = (values[gap_start+k] if augmented else _dot(e, normal)), _dot(u, normal)
            if augmented:
                normal_squared = _dot(normal, normal)
                projector = tuple(tuple(ar.scalar(int(i == j))-normal[i]*normal[j]/normal_squared
                    for j in range(3)) for i in range(3))
                e = add(matvec(projector, e), scale(gap/normal_squared, normal))
            compression = _max_zero(-gap, ar)
            closing = _max_zero(-speed, ar)
            compression_box = ar.box((compression,))
            dashpot = ar.scalar(o.compression_damping_n_s_m)*closing
            if compression_box.upper[0] == 0:
                dashpot = ar.scalar(0)
            elif compression_box.lower[0] <= 0:
                dashpot = ar.interval((0, ar.box((dashpot,)).upper[0]))
            normal_force = ar.scalar(o.compression_stiffness_n_m)*compression+dashpot
            # Evaluate the fixed tangent projector as a linear map. Reusing
            # independent interval copies in u-n*(n.u) invents normal slip
            # even for an exactly axis-aligned patch.
            projector = tuple(tuple(ar.scalar(int(i == j))-normal[i]*normal[j] for j in range(3))
                              for i in range(3))
            slip = matvec(projector, u)
            slip_length = (sum(square(x, ar) for x in slip)+ar.scalar(o.friction_regularization_m_s)**2).sqrt()
            friction_raw = scale(-ar.scalar(o.friction_coefficient)*normal_force/slip_length, slip)
            # The regularized law always has |F_friction| <= mu*F_normal.
            # Intersect this analytic cone bound with its direct interval
            # evaluation so an uncertain near-zero slip cannot create
            # fictitious arbitrarily large friction during Picard iteration.
            friction_limit = Fraction(o.friction_coefficient)*max(Fraction(0), ar.box((normal_force,)).upper[0])
            friction = tuple(ar.interval((max(-friction_limit, lo), min(friction_limit, hi)))
                for lo, hi in zip(ar.box(friction_raw).lower, ar.box(friction_raw).upper))
            force_patch = add(scale(normal_force, normal), friction)
            unbonded_force_patch = force_patch
            couple_patch = tuple(ar.scalar(0) for _ in range(3))
            activation = ar.scalar(0)
            bond_force = couple_unscaled = tuple(ar.scalar(0) for _ in range(3))
            deformation = angle = tuple(ar.scalar(0) for _ in range(3))
            if self.rest_quaternions[k] is not None:
                elapsed = ar.box((time-ar.scalar(self.engagement_times[k]),))
                activation = ar.interval(tuple(max(Fraction(0), min(Fraction(1), x/Fraction(o.bond_engagement_s)))
                    for x in (elapsed.lower[0], elapsed.upper[0])))
                if strained:
                    begin = strain_start+7*k
                    deformation = tuple(values[begin:begin+3])
                    relative = tuple(values[begin+3:begin+7])
                else:
                    deformation = subtract(e, self.rest_positions[k])
                    relative = quaternion_product(quaternion_conjugate(body[0][6:10]), body[j][6:10])
                    relative = quaternion_product(relative, quaternion_conjugate(self.rest_quaternions[k]))
                angle = rotation_vector(relative, ar)
                bond_force = scale(-1, add(scale(ar.scalar(o.bond_stiffness_n_m), deformation),
                    scale(ar.scalar(o.bond_damping_n_s_m), u)))
                couple_unscaled = scale(-1, add(scale(ar.scalar(o.bond_rotational_stiffness_nm_rad), angle),
                    scale(ar.scalar(o.bond_rotational_damping_nm_s_rad), spin)))
                if wrenched:
                    begin = wrench_start+6*k
                    bond_force = scale(-1, tuple(values[begin:begin+3]))
                    couple_unscaled = scale(-1, tuple(values[begin+3:begin+6]))
                force_patch = add(force_patch, scale(activation, bond_force))
                couple_patch = scale(activation, couple_unscaled)
            force, couple = matvec(rt, force_patch), matvec(rt, couple_patch)
            forces[j], forces[0] = add(forces[j], force), subtract(forces[0], force)
            moments[j] = add(moments[j], matvec(transpose(rj), add(cross(tip_offset, force), couple)))
            moments[0] = subtract(moments[0], matvec(transpose(rt), add(cross(subtract(tip, body[0][:3]), force), couple)))
            power = -_dot(force, matvec(rj, tips[k][3:]))
            rows.append(dict(e=e, u=u, gap=gap, compression=compression, normal_force=normal_force,
                force=force, force_patch=force_patch, couple=couple, spin=spin,
                unbonded_force_patch=unbonded_force_patch,
                activation=activation, bond_force_unscaled=bond_force, couple_unscaled=couple_unscaled,
                deformation=deformation, strain_angle=angle,
                power=power, tip=tip, tip_velocity=tip_velocity))
        derivative = [ar.scalar(0) for _ in values]
        target_acceleration = scale(1/masses[0], forces[0])
        derivative[:6] = (*values[3:6], *add(reference_gravity, target_acceleration))
        from orbexa.simulation.attached.assembly.model import STANDARD_GRAVITY
        for k, (row, r, mass, force, torque) in enumerate(zip(body, rotations, masses, forces, moments)):
            wb, q = row[10:13], row[6:10]
            quaternion_dot = (*scale(ar.scalar(Fraction(1, 2)), add(scale(q[3], wb), cross(q[:3], wb))),
                -ar.scalar(Fraction(1, 2))*_dot(q[:3], wb))
            alpha = matvec(self.inverse[k], subtract(scale(1/mass, torque), cross(wb, matvec(self.specific[k], wb))))
            flow = sum(abs(x) for x in self.actuation.controls[k, :3])+sum(abs(x) for x in self.actuation.controls[k, 3:])/o.attitude_moment_arm_m
            flow = 0. if k == 0 else flow/(o.specific_impulse_s*STANDARD_GRAVITY)
            relative_acceleration = ((ar.scalar(0),)*3 if k == 0 else subtract(scale(1/mass, force), target_acceleration))
            relative_velocity = (ar.scalar(0),)*3 if k == 0 else row[3:6]
            derivative[6+14*k:6+14*(k+1)] = (*relative_velocity, *relative_acceleration, *quaternion_dot, *alpha, -ar.scalar(flow))
        for k, tip in enumerate(tips):
            start = fleet.tip_slice.start+6*k
            derivative[start:start+6] = (*tip[3:], *self.extension[k])
        if augmented:
            for k, (rotation, row) in enumerate(zip(rotations, body)):
                start = rotation_start+9*k
                derivative[start:start+9] = rotation_derivative(rotation, row[10:13])
            for k, row in enumerate(rows):
                derivative[gap_start+k] = _dot(row['u'], self.normals[k])
                if strained and self.rest_quaternions[k] is not None:
                    begin = strain_start+7*k
                    derivative[begin:begin+7] = (*row['u'], *scale(ar.scalar(Fraction(1, 2)),
                        quaternion_product((*row['spin'], ar.scalar(0)), values[begin+3:begin+7])))
        diag = dict(body=body, tips=tips, rotations=rotations, spins=spins, masses=masses, rows=rows)
        if wrenched:
            derivative[wrench_start:] = wrench_derivatives(self, diag, derivative)
        return tuple(derivative), diag


@dataclass(frozen=True)
class ContactStepEncloser:
    precision_bits: int = 128
    max_picard_iterations: int = 24
    max_substeps: int = 64
    max_step_s: float = .001

    def __post_init__(self):
        for name, maximum in (('precision_bits', 4096), ('max_picard_iterations', 256), ('max_substeps', 4096)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value > maximum:
                raise ValueError('Declare finite positive contact enclosure budgets')
        if self.precision_bits < 64 or isinstance(self.max_step_s, bool) or not np.isfinite(self.max_step_s) or self.max_step_s <= 0:
            raise ValueError('Contact enclosure needs 64 or more bits and a positive finite step')

    def initial_box(self, fleet, initial_enclosure=None):
        box = IntervalBox.point(fleet.initial_values[:fleet.ledger_start]) if initial_enclosure is None else initial_enclosure
        if not isinstance(box, IntervalBox) or box.dimension != fleet.ledger_start:
            raise ValueError('Contact initial enclosure must match the packed physical coordinates without ledgers')
        if not box.contains(IntervalBox.point(fleet.initial_values[:fleet.ledger_start])):
            raise ValueError('Contact initial set does not contain the supplied numerical representative')
        if box.slice(6, 12) != IntervalBox.point((0,)*6):
            raise ValueError('Target-following contact coordinates require identically zero target-local translation; retain absolute uncertainty in the leading ephemeris')
        import flint
        with _ARB_LOCK, flint.ctx.workprec(self.precision_bits):
            ar = _ArbArithmetic(flint)
            values = list(ar.vector(box))
            for k in range(fleet.body_count):
                start = 6+14*k+6
                values[start:start+4] = normalized_quaternion(tuple(values[start:start+4]), ar)
            # Retain the supplied numerical representative while enclosing
            # its exact unit-quaternion interpretation.
            return box.hull(ar.box(values))

    def refined_bound(self, **request):
        """Refine an unresolved same-command proof before allowing motion.

        Physical time, command, initial set, model and all limits remain
        unchanged. The finite substep cap bounds every retry as well as the
        final proof. Numerical failure is never accepted as a safe hold.
        """
        width = Fraction(request['duration_s'])
        engine, attempts = self, []
        while True:
            certificate = engine.bound(**request)
            attempts.append(dict(max_step_s=engine.max_step_s, status=certificate.status,
                                 reason=certificate.reason))
            if certificate.passed or certificate.status != 'UNRESOLVED':
                break
            next_step = engine.max_step_s/2
            if max(1, -(-width//Fraction(next_step))) > self.max_substeps:
                break
            engine = replace(engine, max_step_s=next_step)
        return replace(certificate, refinement_attempts=tuple(attempts))

    def event_support(self, *, fleet, options, bonds, actuation, time_s, initial, rest_enclosures=None,
                      engagement_times=None, rotation_initial=None, gap_initial=None, strain_initial=None):
        """Current whole-set preload and actual rest sets for a digital bond event."""
        import flint
        with _ARB_LOCK, flint.ctx.workprec(self.precision_bits):
            ar = _ArbArithmetic(flint)
            model = _ContactField(fleet, options, bonds, actuation, ar, rest_enclosures,
                engagement_times=engagement_times)
            combined, _, _ = auxiliary_initial(fleet, initial, ar, rotations=rotation_initial, gaps=gap_initial)
            combined, _ = initial_strain_boxes(model, time_s, combined, ar, strain_initial)
            _, diag = model.evaluate(ar.scalar(time_s), ar.vector(combined))
            result = []
            for k, row in enumerate(diag['rows']):
                comp, force, gap = (ar.box((row[key],)) for key in ('compression', 'normal_force', 'gap'))
                body = diag['body']
                q = quaternion_product(quaternion_conjugate(body[0][6:10]), body[k+1][6:10])
                result.append(dict(position=ar.box(row['e']), quaternion=ar.box(q),
                    preload_supported=comp.lower[0] > 0 and force.lower[0] >= Fraction(.8*options.preload_n)
                        and gap.upper[0] <= Fraction(options.range_error_bound_m)))
            return result

    def bound(self, *, fleet, options, bonds, actuation, geometry, time_s, duration_s, initial, environment=None,
              rest_enclosures=None, engagement_times=None, rotation_initial=None, gap_initial=None, strain_initial=None,
              wrench_initial=None):
        time, h = Fraction(time_s), Fraction(duration_s)
        if h <= 0 or not isinstance(initial, IntervalBox) or initial.dimension != fleet.ledger_start:
            raise ValueError('Contact hold needs a positive duration and the exact packed start set')
        if initial.slice(6, 12) != IntervalBox.point((0,)*6):
            raise ValueError('Contact held-flow source is off the declared actual-target-COM chart')
        context = dict(time_s=str(time), duration_s=str(h), options=asdict(options),
            translation_chart='actual-target-COM-following nonrotating ECI axes; common ephemeris in six leading coordinates',
            controls_body=actuation.controls.tolist(), extension_acceleration_body_m_s2=actuation.extension_acceleration.tolist(),
            engagement_s=bonds.engagement.tolist(), bond_rest_body_m=bonds.bond_rest.tolist(), rotation_rest=bonds.rotation_rest.tolist(),
            exact_engagement_times_s=None if engagement_times is None else [None if value is None else str(value) for value in engagement_times],
            geometry_fingerprint=getattr(geometry, 'fingerprint', None),
            precision_bits=self.precision_bits, max_substeps=self.max_substeps, max_step_s=self.max_step_s,
            dry_mass_kg=fleet.dry.tolist(), specific_inertia_m2=fleet.specific.tolist(), mounts_body_m=fleet.mounts.tolist(),
            anchors_body_m=fleet.anchors.tolist(), normals_body=fleet.normals.tolist(), mu_m3_s2=fleet.mu,
            gravity_gradient=fleet.gravity_gradient,
            members=[dict(id=member.id, patch_id=member.patch_id, arm_reach_m=member.arm_reach_m,
                bus_radius_m=member.properties.body_radius_m) for member in fleet.members],
            rest_enclosures=None if rest_enclosures is None else [None if rest is None else
                {key: box.as_dict() for key, box in rest.items()} for rest in rest_enclosures])
        # Infinite means not yet engaged and has a stable JSON-safe representation.
        context['engagement_s'] = [None if not np.isfinite(value) else float(value) for value in bonds.engagement]
        environment_model = None
        if environment is not None:
            from orbexa.simulation.completion.environment.enclosure import NonlinearEnvironmentEnclosure
            try:
                environment_model = NonlinearEnvironmentEnclosure.from_environment(environment)
            except (_ProofUnresolved, ValueError) as error:
                return ContactStepCertificate('UNAVAILABLE', str(error), context, initial)
            context['environment'] = environment_model.as_dict()
        context['request_fingerprint'] = fingerprint(context)
        if not hasattr(geometry, 'components'):
            return ContactStepCertificate('UNAVAILABLE', 'composite_occupied_component_geometry_required', context, initial)
        try:
            import flint
            with _ARB_LOCK, flint.ctx.workprec(self.precision_bits):
                ar = _ArbArithmetic(flint)
                model = _ContactField(fleet, options, bonds, actuation, ar, rest_enclosures, environment_model, engagement_times)
                combined, rotations, gaps = auxiliary_initial(fleet, initial, ar, rotations=rotation_initial, gaps=gap_initial)
                combined, strains = initial_strain_boxes(model, time, combined, ar, strain_initial)
                combined, wrenches = initial_wrench_boxes(model, time, combined, ar, wrench_initial)
                context['rotation_initial'] = [box.as_dict() for box in rotations]
                context['gap_initial'] = [box.as_dict() for box in gaps]
                context['strain_initial'] = [box.as_dict() for box in strains]
                context['wrench_initial'] = [box.as_dict() for box in wrenches]
                context['auxiliary_scope'] = 'Redundant coordinates on the same physical initial graph: Rdot=R[omega], gdot=n.u, displacement-strain dot=u, relative-strain quaternion dot=(spin,0)*Q/2, B=k*displacement+c*u and M=kr*log(Q)+cr*spin with their exact differentiated graph; zero displacement/identity Q only at actual digital engagement, whose damping wrench is generally nonzero. Later holds intersect carried endpoints and the complete uncertain current/rest graph, never reset strain or wrench.'
                context['request_fingerprint'] = fingerprint({key: value for key, value in context.items()
                    if key != 'request_fingerprint'})
                count = max(1, -(-h//Fraction(self.max_step_s)))
                if count > self.max_substeps:
                    raise _ProofUnresolved('contact substep budget exceeded')
                dt, endpoint, swept, substeps, iterations = h/count, combined, combined, [], 0
                for substep in range(count):
                    start = ar.vector(endpoint)
                    trial = IntervalBox.from_center_radius(tuple((lo+hi)/2 for lo, hi in zip(endpoint.lower, endpoint.upper)),
                        tuple((hi-lo)/2+Fraction(1, 10**22) for lo, hi in zip(endpoint.lower, endpoint.upper)))
                    for attempt in range(self.max_picard_iterations):
                        iterations += 1
                        field, diag = model.evaluate(ar.interval((time+substep*dt, time+(substep+1)*dt)), ar.vector(trial))
                        image = ar.box(add(start, scale(ar.interval((0, dt)), field)))
                        if all(a < c and d < b for a, b, c, d in zip(trial.lower, trial.upper, image.lower, image.upper)):
                            endpoint = ar.box(add(start, scale(ar.scalar(dt), field)))
                            # Both sets contain the same exact endpoint: the
                            # point-time enclosure and the strict Picard trial.
                            # Their exact intersection removes independent
                            # outward-rounding halos without discarding flow.
                            from orbexa.simulation.contact.dynamics.rotation import intersect
                            endpoint = intersect(endpoint, trial)
                            if any(endpoint.lower[6+14*k+13] < 0 for k in range(fleet.body_count)):
                                raise _ProofUnresolved('whole-contact tank nonnegativity not proven')
                            swept = swept.hull(trial)
                            substeps.append(diag)
                            break
                        # Grow only the uncovered image excess. Multiplying
                        # the whole inherited uncertainty width by a factor
                        # invents large trial excursions even for tiny motion.
                        # Every prior point remains, and the same strict
                        # Picard inclusion and swept guards still decide proof.
                        trial = IntervalBox(tuple(lo-Fraction(3, 2)*max(Fraction(0), lo-c)-Fraction(1, 10**22)
                            for lo, c in zip(trial.lower, image.lower)),
                            tuple(hi+Fraction(3, 2)*max(Fraction(0), d-hi)+Fraction(1, 10**22)
                            for hi, d in zip(trial.upper, image.upper)))
                    else:
                        raise _ProofUnresolved('contact Picard inclusion exceeded its finite budget')
                # One guard evaluation on the complete swept hull is a
                # sufficient whole-hold proof and often avoids repeating
                # the same expensive occupied geometry at every tiny
                # integrator interval. If that hull loses useful temporal
                # dependency, retain the original substep guard proof.
                try:
                    _, whole_diag = model.evaluate(ar.interval((time, time+h)), ar.vector(swept))
                    checks = [self._checks(model, whole_diag, geometry, dt)]
                    context['guard_partition'] = 'complete-held-swept-hull'
                except _ProofUnresolved as whole_refusal:
                    context['whole_swept_guard_refusal'] = str(whole_refusal)
                    checks = [self._checks(model, local, geometry, dt) for local in substeps]
                    context['guard_partition'] = 'internal-picard-substeps'
                context.update(proved_substeps=count, picard_iterations=iterations, guard_partitions=len(checks))
                _, final_diag = model.evaluate(ar.scalar(time+h), ar.vector(endpoint))
                strain_start = fleet.ledger_start+9*fleet.body_count+fleet.chaser_count
                wrench_start = strain_start+7*fleet.chaser_count
                return ContactStepCertificate('PROVEN', 'whole_contact_hold_flow_and_guards_proven', context,
                    initial, endpoint.slice(0, fleet.ledger_start), swept.slice(0, fleet.ledger_start), dict(substeps=checks,
                        preload_supported=[all(item['preload_supported'][k] for item in checks) for k in range(fleet.chaser_count)],
                        quiet_supported=all(item['quiet_supported'] for item in checks),
                        finite_lock_ready=all(item['finite_lock_ready'] for item in checks)),
                    rotation_endpoint=tuple(endpoint.slice(fleet.ledger_start+9*k, fleet.ledger_start+9*(k+1)) for k in range(fleet.body_count)),
                    rotation_swept=tuple(swept.slice(fleet.ledger_start+9*k, fleet.ledger_start+9*(k+1)) for k in range(fleet.body_count)),
                    gap_endpoint=tuple(endpoint.slice(fleet.ledger_start+9*fleet.body_count+k, fleet.ledger_start+9*fleet.body_count+k+1) for k in range(fleet.chaser_count)),
                    gap_swept=tuple(swept.slice(fleet.ledger_start+9*fleet.body_count+k, fleet.ledger_start+9*fleet.body_count+k+1) for k in range(fleet.chaser_count)),
                    strain_endpoint=tuple(endpoint.slice(strain_start+7*k, strain_start+7*(k+1)) for k in range(fleet.chaser_count)),
                    strain_swept=tuple(swept.slice(strain_start+7*k, strain_start+7*(k+1)) for k in range(fleet.chaser_count)),
                    endpoint_potential=potential_box(model, final_diag),
                    wrench_endpoint=tuple(endpoint.slice(wrench_start+6*k, wrench_start+6*(k+1)) for k in range(fleet.chaser_count)),
                    wrench_swept=tuple(swept.slice(wrench_start+6*k, wrench_start+6*(k+1)) for k in range(fleet.chaser_count)))
        except ImportError:
            return ContactStepCertificate('UNAVAILABLE', 'arb_backend_unavailable', context, initial)
        except (_ProofUnresolved, ArithmeticError, ValueError) as error:
            return ContactStepCertificate('UNRESOLVED', str(error), context, initial)

    def _checks(self, model, diag, geometry, dt):
        fleet, o, ar = model.fleet, model.o, model.ar
        body, tips, rotations, rows = diag['body'], diag['tips'], diag['rotations'], diag['rows']
        rt = rotations[0]
        normup = lambda x: norm_upper(x, ar)
        lower = lambda x: ar.box((x,)).lower[0]
        upper = lambda x: ar.box((x,)).upper[0]
        maxima, margins, preload, quiet, rods, pads = {}, [], [], True, [], []

        def separated(point, component, radius):
            box = ar.box(point)
            middle = np.array([float((lo+hi)/2) for lo, hi in zip(box.lower, box.upper)])
            # Numeric choices only select separating directions. Every
            # support inequality below uses exact declared occupied vertices.
            directions = [middle-component.closest_point(middle)]
            directions += list(np.asarray(component.rotation_component_to_body).T)
            best = None
            for candidate in directions:
                if not np.all(np.isfinite(candidate)) or not np.any(candidate):
                    continue
                for sign in (1, -1):
                    direction = tuple(ar.scalar(float(sign*x)) for x in candidate)
                    length = normup(direction)
                    if length == 0:
                        continue
                    support = max(upper(_dot(direction, tuple(ar.scalar(x) for x in vertex)))
                        for vertex in component.vertices_body_m)
                    margin = lower(_dot(direction, point))-support-Fraction(radius)*length
                    normalized = margin/length
                    best = normalized if best is None else max(best, normalized)
                    if normalized > 0:
                        # A single exact separating support inequality is
                        # sufficient. Further numeric direction choices can
                        # improve a displayed margin but add no safety fact.
                        margins.append(normalized)
                        return
            if best is None or best <= 0:
                raise _ProofUnresolved('whole-contact occupied-component separation not proven: '+component.id)
            margins.append(best)

        for k, (member, row) in enumerate(zip(fleet.members, rows)):
            j = k+1
            patch = geometry.attachment_patch(member.patch_id)
            if normup(tips[k][:3]) > Fraction(member.arm_reach_m):
                raise _ProofUnresolved(member.id+': whole-step positioner reach not proven')
            if normup(tips[k][3:]) > Fraction(o.extension_speed_m_s)+Fraction(o.extension_acceleration_m_s2)*dt:
                raise _ProofUnresolved(member.id+': whole-step positioner speed not proven')
            quantities = dict(compression_m=upper(row['compression']), force_n=normup(row['force']),
                moment_nm=normup(row['couple']), positioner_power_w=max(abs(lower(row['power'])), abs(upper(row['power']))))
            maxima[member.id] = {name: str(value) for name, value in quantities.items()}
            if (quantities['compression_m'] > Fraction(o.max_compression_m)
                    or quantities['force_n'] > Fraction(o.positioner_force_limit_n)
                    or quantities['positioner_power_w'] > Fraction(o.positioner_power_limit_w)):
                raise _ProofUnresolved(member.id+': whole-step pad compression/force/power not proven')
            normal = model.normals[k]
            if upper(row['activation']) > 0:
                unit_normal = scale(1/norm(normal, ar), normal)
                # Both adhesive strength and wrench scale by the same
                # nonnegative activation. Test the unscaled bond law without
                # crediting compression friction; this remains sufficient at
                # every activation, including an interval beginning at zero.
                selected_force = row['bond_force_unscaled']
                fn = _dot(selected_force, unit_normal)
                shear = subtract(selected_force, scale(fn, unit_normal))
                if (max(0, -lower(fn)) > Fraction(o.tensile_limit_n)
                        or normup(shear) > Fraction(o.shear_limit_n)
                        or normup(row['couple_unscaled']) > Fraction(o.moment_limit_nm)):
                    raise _ProofUnresolved(member.id+': whole-step adhesive capacity not proven')
            com = matvec(transpose(rt), subtract(body[j][:3], body[0][:3]))
            tip = matvec(transpose(rt), subtract(row['tip'], body[0][:3]))
            rod_midpoint = scale(ar.scalar(Fraction(1, 2)), add(com, tip))
            rod_radius = normup(subtract(tip, com))/2+Fraction(o.positioner_radius_m)
            rods.append((rod_midpoint, rod_radius))
            pads.append(tip)
            for component in geometry.components:
                separated(com, component, Fraction(member.properties.body_radius_m)+Fraction(o.minimum_bus_clearance_m))
                if component.id != patch.component_id:
                    separated(rod_midpoint, component, rod_radius)
                    separated(tip, component, o.pad_radius_m)
            # Only the thin compliant tip may enter its own selected face,
            # by the declared compression travel and within the full footprint.
            point_delta = subtract(tip, tuple(ar.scalar(x) for x in patch.point_body_m))
            patch_normal = tuple(ar.scalar(x) for x in patch.normal_body)
            com_delta = subtract(com, tuple(ar.scalar(x) for x in patch.point_body_m))
            if lower(_dot(com_delta, patch_normal)) <= (
                    Fraction(member.properties.body_radius_m)+Fraction(o.minimum_bus_clearance_m))*normup(patch_normal):
                raise _ProofUnresolved(member.id+': selected-face outward bus/positioner ordering not proven')
            for tangent, half in ((patch.tangent_u_body, patch.half_extents_m[0]),
                                  (patch.tangent_v_body, patch.half_extents_m[1])):
                projected = _dot(point_delta, tuple(ar.scalar(x) for x in tangent))
                margin = Fraction(half)-Fraction(patch.margin_m)-Fraction(o.pad_radius_m)-max(abs(lower(projected)), abs(upper(projected)))
                if margin < 0:
                    raise _ProofUnresolved(member.id+': whole-step pad footprint not proven')
                margins.append(margin)
            same_face = (np.array_equal(member.normal_body, patch.normal_body)
                         and np.array_equal(member.surface_anchor_body_m, patch.point_body_m))
            face_gap = row['gap'] if same_face else _dot(point_delta, patch_normal)
            if lower(face_gap) < -Fraction(o.max_compression_m)*normup(patch_normal):
                raise _ProofUnresolved(member.id+': selected-face compliant travel not proven')
            preload.append(lower(row['compression']) > 0 and lower(row['normal_force']) >= Fraction(.8*o.preload_n)
                and upper(row['gap']) <= Fraction(o.range_error_bound_m))
            quiet = quiet and lower(row['activation']) >= 1 and normup(row['u']) <= Fraction(o.settling_speed_m_s)
            quiet = quiet and normup(row['spin']) <= Fraction(o.settling_spin_rad_s) and normup(tips[k][3:]) <= Fraction(o.settling_speed_m_s)/4
            for other in range(k+1, fleet.chaser_count):
                distance = lower(norm(subtract(body[j][:3], body[other+1][:3]), ar))
                margin = distance-Fraction(member.properties.body_radius_m)-Fraction(fleet.members[other].properties.body_radius_m)-Fraction(o.minimum_bus_clearance_m)
                if margin <= 0:
                    raise _ProofUnresolved(member.id+': whole-step peer bus separation not proven')
                margins.append(margin)
            for other, peer in enumerate(fleet.members):
                if other == k:
                    continue
                peer_com = matvec(transpose(rt), subtract(body[other+1][:3], body[0][:3]))
                distance = lower(norm(subtract(peer_com, rod_midpoint), ar))
                margin = distance-rod_radius-Fraction(peer.properties.body_radius_m)-Fraction(o.pad_radius_m)
                if margin <= 0:
                    raise _ProofUnresolved(member.id+': whole-step positioner/peer separation not proven')
                margins.append(margin)
        for k in range(fleet.chaser_count):
            for other in range(k+1, fleet.chaser_count):
                midpoint, radius = rods[k]
                peer_midpoint, peer_radius = rods[other]
                pair_margins = (
                    lower(norm(subtract(midpoint, peer_midpoint), ar))-radius-peer_radius,
                    lower(norm(subtract(pads[k], peer_midpoint), ar))-Fraction(o.pad_radius_m)-peer_radius,
                    lower(norm(subtract(pads[other], midpoint), ar))-Fraction(o.pad_radius_m)-radius,
                    lower(norm(subtract(pads[k], pads[other]), ar))-2*Fraction(o.pad_radius_m),
                )
                if min(pair_margins) <= 0:
                    raise _ProofUnresolved(fleet.members[k].id+': whole-step peer positioner/pad separation not proven')
                margins.extend(pair_margins)
        # The optimal rigidization energy is no larger than the energy
        # relative to the target's instantaneous rigid motion.
        energy = Fraction(0)
        for k in range(1, fleet.body_count):
            relative_velocity = subtract(subtract(body[k][3:6], body[0][3:6]),
                cross(diag['spins'][0], subtract(body[k][:3], body[0][:3])))
            relative_spin = subtract(diag['spins'][k], diag['spins'][0])
            mass = upper(diag['masses'][k])
            inertia_norm = normup(tuple(x for row in model.specific[k] for x in row))*mass
            energy += (mass*normup(relative_velocity)**2+inertia_norm*normup(relative_spin)**2)/2
        quiet = quiet and energy <= Fraction(o.max_lock_energy_j)
        # A small energy alone does not establish each residual's tolerance.
        for k in range(fleet.body_count):
            minimum_mass = lower(diag['masses'][k])
            inverse_norm = normup(tuple(x for row in model.inverse[k] for x in row))/minimum_mass
            quiet = quiet and 2*energy <= minimum_mass*Fraction(o.settling_speed_m_s)**2
            quiet = quiet and 2*energy*inverse_norm <= Fraction(o.settling_spin_rad_s)**2
        finite_lock_ready = all(lower(row['activation']) >= 1 for row in rows)
        return dict(member_upper=maxima, minimum_geometry_margin_m=str(min(margins)),
            preload_supported=preload, quiet_supported=bool(quiet), finite_lock_ready=bool(finite_lock_ready),
            active_lock_admission='Eligibility only; complete immutable finite-lock load/travel/power/work preflight must pass before any lock motion. The instantaneous projection-energy cap applies only to settled_passive.',
            rigidization_energy_upper_j=str(energy))
