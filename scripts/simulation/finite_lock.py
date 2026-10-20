"""Finite, load-bounded closure of the settled fleet's residual relative twist.

The declared active positioners brake relative material coordinates over a
positive duration. Newton--Euler reactions and their work are integrated;
there is no velocity impulse at the final rigid coordinate change. This is
a representative mechanical servo model, not a qualified adhesive device.

At the start, an ideal strain clamp traps the compliant pad's existing internal
elastic energy. The ensuing active articulated-joint law replaces the passive
spring law; it does not pretend that a spring's energy stays constant while
its original deformation changes. The trapped energy is retained as a separate
constant inventory through the rigid stages. Clamp strength, servo tracking,
flexure and material realization remain mechanism assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
import copy
from fractions import Fraction
from types import SimpleNamespace
import hashlib
from pathlib import Path

import numpy as np

from orbexa.control.reachability import IntervalBox, _ARB_LOCK, _ArbArithmetic, _ProofUnresolved
from orbexa.core.reference_frame import fingerprint
from orbexa.simulation.contact_enclosure import ContactStepEncloser
from orbexa.simulation.rigid_interval import (
    add, subtract, scale, cross, matvec, transpose, inverse3, square, norm_upper,
    normalized_quaternion, quaternion_rotation, quaternion_product, quaternion_conjugate,
)


def _matrix(flat):
    return tuple(tuple(flat[3*i+j] for j in range(3)) for i in range(3))


def _matmul(left, right):
    return tuple(tuple(sum(a*b for a, b in zip(row, column)) for column in transpose(right)) for row in left)


def _outer(left, right):
    return tuple(tuple(a*b for b in right) for a in left)


def _skew(value):
    x, y, z = value
    return ((0, -z, y), (z, 0, -x), (-y, x, 0))


def _dot(left, right):
    return sum(a*b for a, b in zip(left, right))


def _box(value):
    return IntervalBox(value.get('exact_lower', value['lower']), value.get('exact_upper', value['upper']))


def _source_manifest():
    root = Path(__file__).parents[1]
    names = ('simulation/finite_lock.py', 'simulation/contact_enclosure.py',
        'simulation/contact_strain.py', 'simulation/contact_rotation.py',
        'simulation/contact_state.py', 'simulation/contact_contracts.py',
        'simulation/capture.py',
        'simulation/rigid_interval.py', 'simulation/nonlinear_environment_enclosure.py',
        'simulation/nonlinear_completion_environment.py', 'control/reachability.py',
        'core/reference_frame.py', 'core/nonlinear_relative.py', 'core/spacecraft_geometry.py')
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}


def _pad_inventory(contact, count):
    energy = contact.audit.get('final_contact_potential_enclosure_j')
    if not isinstance(energy, dict):
        raise ValueError('Finite lock requires the carried whole-set stored pad-energy inventory')
    energy = _box(energy)
    if energy.dimension != 1 or energy.lower[0] < 0:
        raise ValueError('Stored pad-energy inventory must be a nonnegative scalar set')
    continuous = contact.audit['continuous_enclosure']
    strains, rests = continuous.get('final_strain_sets'), continuous.get('final_rest_sets')
    if not isinstance(strains, (list, tuple)) or not isinstance(rests, (list, tuple)) or len(strains) != count or len(rests) != count:
        raise ValueError('Finite lock requires every pad\'s complete carried strain and rest support')
    for strain, rest in zip(strains, rests):
        if not isinstance(strain, dict) or _box(strain).dimension != 7:
            raise ValueError('A trapped pad strain needs three displacement and four quaternion coordinates')
        if not isinstance(rest, dict) or set(rest) != {'position', 'quaternion'}:
            raise ValueError('A trapped pad rest needs its complete position and quaternion supports')
        if _box(rest['position']).dimension != 3 or _box(rest['quaternion']).dimension != 4:
            raise ValueError('A trapped pad rest has invalid coordinate dimensions')
    return dict(energy_j=energy.as_dict(), strain_sets=copy.deepcopy(strains),
        uncertain_rest_sets=copy.deepcopy(rests),
        constitutive_rule='Existing compliant internal strains are trapped unchanged as independent energy/strain inventory during the ideal active joint phase and subsequent rigid motion')


def _planned_outputs(audit):
    names = ('final_set', 'swept', 'rigid_state_set', 'rigid_parameter_sets',
        'integrated_absolute_interface_work_upper_j', 'internal_mechanical_work_j',
        'external_com_frame_work_j', 'internal_mechanical_work_enclosure_j',
        'external_com_frame_work_enclosure_j', 'final_com_frame_kinetic_energy_j',
        'internal_absorbed_mechanical_work_j', 'energy_balance_residual_j',
        'accepted_generalized_state_set', 'accepted_numerical_generalized_state')
    return {name: audit.pop(name) for name in names if name in audit}


class _LockField:
    def __init__(self, fleet, options, initial, environment, ar):
        self.fleet, self.o, self.ar = fleet, options, ar
        values = ar.vector(initial)
        n = fleet.body_count
        body = [values[6+14*k:6+14*(k+1)] for k in range(n)]
        tips = [values[6+14*n+6*k:6+14*n+6*(k+1)] for k in range(n-1)]
        self.fuel = tuple(row[13] for row in body)
        self.masses = tuple(ar.scalar(dry)+fuel for dry, fuel in zip(fleet.dry, self.fuel))
        self.mass = sum(self.masses)
        self.specific = tuple(tuple(tuple(ar.scalar(x) for x in row) for row in matrix) for matrix in fleet.specific)
        q0 = normalized_quaternion(body[0][6:10], ar)
        r0 = quaternion_rotation(q0)
        self.positions = tuple(matvec(transpose(r0), subtract(row[:3], body[0][:3])) for row in body)
        local_velocity = tuple(matvec(transpose(r0), row[3:6]) for row in body)
        center = scale(1/self.mass, tuple(sum(m*p[j] for m, p in zip(self.masses, self.positions)) for j in range(3)))
        velocity = scale(1/self.mass, tuple(sum(m*v[j] for m, v in zip(self.masses, local_velocity)) for j in range(3)))
        offsets = tuple(subtract(position, center) for position in self.positions)
        quaternions = tuple(quaternion_product(quaternion_conjugate(q0), normalized_quaternion(row[6:10], ar)) for row in body)
        rotations = tuple(quaternion_rotation(q) for q in quaternions)
        inertias = tuple(_matmul(_matmul(rotation, tuple(tuple(m*x for x in row) for row in specific)), transpose(rotation))
                         for rotation, specific, m in zip(rotations, self.specific, self.masses))
        inertia = self.inertia(offsets, inertias)
        spins = tuple(matvec(rotation, row[10:13]) for rotation, row in zip(rotations, body))
        angular = tuple(sum(matvec(local, spin)[j]+m*cross(offset, subtract(v, velocity))[j]
            for local, spin, m, offset, v in zip(inertias, spins, self.masses, offsets, local_velocity)) for j in range(3))
        common = matvec(inverse3(inertia), angular)
        self.relative_velocity = tuple(subtract(subtract(v, velocity), cross(common, offset))
            for v, offset in zip(local_velocity, offsets))
        self.relative_spin = tuple(subtract(spin, common) for spin in spins)
        self.tip_anchors = tuple(matvec(transpose(rotations[0]), subtract(add(self.positions[k+1],
            matvec(rotations[k+1], add(tuple(map(ar.scalar, fleet.mounts[k])), tips[k][:3]))), self.positions[0]))
            for k in range(n-1))
        self.tip_anchor_velocity = []
        for k in range(n-1):
            j = k+1
            lever = matvec(rotations[j], add(tuple(map(ar.scalar, fleet.mounts[k])), tips[k][:3]))
            relative = add(subtract(self.positions[j], self.positions[0]), lever)
            velocity = subtract(add(add(subtract(local_velocity[j], local_velocity[0]), cross(spins[j], lever)),
                matvec(rotations[j], tips[k][3:])), cross(spins[0], relative))
            self.tip_anchor_velocity.append(matvec(transpose(rotations[0]), velocity))
        absolute_center = add(values[:3], scale(1/self.mass,
            tuple(sum(m*row[j] for m, row in zip(self.masses, body)) for j in range(3))))
        absolute_velocity = add(values[3:6], scale(1/self.mass,
            tuple(sum(m*row[j+3] for m, row in zip(self.masses, body)) for j in range(3))))
        self.initial = absolute_center+absolute_velocity+q0+common+tuple(x for q in quaternions for x in q)+(ar.scalar(0), ar.scalar(0))
        self.environment = None
        if environment is not None:
            from orbexa.simulation.nonlinear_environment_enclosure import NonlinearEnvironmentEnclosure
            self.environment = NonlinearEnvironmentEnclosure.from_environment(environment)
        self.environment_parameters = None if self.environment is None else self.environment.force_parameters(ar)

    def inertia(self, offsets, inertias):
        ar = self.ar
        return tuple(tuple(sum(local[i][j]+m*(sum(square(r[k], ar) for k in range(3) if k != i)
            if i == j else -r[i]*r[j]) for local, m, r in zip(inertias, self.masses, offsets))
            for j in range(3)) for i in range(3))

    def shape(self, time, values):
        ar, n = self.ar, self.fleet.body_count
        duration = ar.scalar(self.o.lock_duration_s)
        f, g, gd = time-time*time/(2*duration), 1-time/duration, -1/duration
        positions = tuple(add(x, scale(f, u)) for x, u in zip(self.positions, self.relative_velocity))
        center = scale(1/self.mass, tuple(sum(m*p[j] for m, p in zip(self.masses, positions)) for j in range(3)))
        average = scale(1/self.mass, tuple(sum(m*u[j] for m, u in zip(self.masses, self.relative_velocity)) for j in range(3)))
        offsets = tuple(subtract(p, center) for p in positions)
        velocities = tuple(scale(g, subtract(u, average)) for u in self.relative_velocity)
        accelerations = tuple(scale(gd, subtract(u, average)) for u in self.relative_velocity)
        quaternions = tuple(values[13+4*k:17+4*k] for k in range(n))
        rotations = tuple(quaternion_rotation(q) for q in quaternions)
        inertias = tuple(_matmul(_matmul(rotation, tuple(tuple(m*x for x in row) for row in specific)), transpose(rotation))
                         for rotation, specific, m in zip(rotations, self.specific, self.masses))
        return f, g, gd, offsets, velocities, accelerations, quaternions, rotations, inertias

    def evaluate(self, time, values):
        ar, fleet, options = self.ar, self.fleet, self.o
        p, v, q, w = values[:3], values[3:6], values[6:10], values[10:13]
        _, g, gd, offsets, rdots, rddots, quaternions, rotations, inertias = self.shape(time, values)
        rotation = quaternion_rotation(q)
        inertias_dot = []
        for local, sigma in zip(inertias, self.relative_spin):
            skew = _skew(scale(g, sigma))
            left, right = _matmul(skew, local), _matmul(local, skew)
            inertias_dot.append(tuple(tuple(a-b for a, b in zip(row, column)) for row, column in zip(left, right)))
        inertia = self.inertia(offsets, inertias)
        inertia_dot = tuple(tuple(sum(local[i][j]+m*((2*_dot(r, rd) if i == j else 0)-rd[i]*r[j]-r[i]*rd[j])
            for local, m, r, rd in zip(inertias_dot, self.masses, offsets, rdots)) for j in range(3)) for i in range(3))
        internal_h = tuple(sum(m*cross(r, rd)[j]+matvec(local, scale(g, sigma))[j]
            for m, r, rd, local, sigma in zip(self.masses, offsets, rdots, inertias, self.relative_spin)) for j in range(3))
        internal_hdot = tuple(sum(m*cross(r, rdd)[j]+matvec(local_dot, scale(g, sigma))[j]+matvec(local, scale(gd, sigma))[j]
            for m, r, rdd, local_dot, local, sigma in zip(self.masses, offsets, rddots, inertias_dot, inertias, self.relative_spin)) for j in range(3))
        external_forces, external_couples = [], []
        zero = tuple(ar.scalar(0) for _ in range(3))
        for k, (mass, offset, rdot, local) in enumerate(zip(self.masses, offsets, rdots, inertias)):
            position = add(p, matvec(rotation, offset))
            velocity = add(v, matvec(rotation, add(cross(w, offset), rdot)))
            radius2 = sum(square(x, ar) for x in position)
            if not radius2 > 0:
                raise _ProofUnresolved('finite lock component set reaches the gravity singularity')
            radius = radius2.sqrt()
            acceleration = scale(-ar.scalar(fleet.mu)/radius**3, position)
            if self.environment is not None:
                name = 'target' if k == 0 else fleet.members[k-1].id
                acceleration = add(acceleration, self.environment.evaluate(name, position, velocity,
                    self.fuel[k], ar, self.environment_parameters))
            external_forces.append(scale(mass, matvec(transpose(rotation), acceleration)))
            radial = matvec(transpose(rotation), scale(1/radius, position))
            external_couples.append(scale(3*ar.scalar(fleet.mu)/radius**3, cross(radial, matvec(local, radial)))
                                    if fleet.gravity_gradient else zero)
        total_force = tuple(sum(force[j] for force in external_forces) for j in range(3))
        torque = tuple(sum(couple[j]+cross(offset, force)[j] for couple, offset, force in
            zip(external_couples, offsets, external_forces)) for j in range(3))
        angular = matvec(inverse3(inertia), subtract(subtract(subtract(torque, matvec(inertia_dot, w)),
            internal_hdot), cross(w, add(matvec(inertia, w), internal_h))))
        acceleration = scale(1/self.mass, total_force)
        forces, couples, power = [], [], ar.scalar(0)
        for mass, offset, rdot, rddot, sigma, local, environment_force, environment_couple in zip(
                self.masses, offsets, rdots, rddots, self.relative_spin, inertias, external_forces, external_couples):
            reaction = subtract(scale(mass, add(acceleration, add(cross(angular, offset), add(cross(w, cross(w, offset)),
                add(scale(ar.scalar(2), cross(w, rdot)), rddot))))), environment_force)
            spin = add(w, scale(g, sigma))
            spin_dot = add(angular, add(cross(w, scale(g, sigma)), scale(gd, sigma)))
            moment = subtract(add(matvec(local, spin_dot), cross(spin, matvec(local, spin))), environment_couple)
            forces.append(reaction); couples.append(moment)
            power += _dot(reaction, rdot)+_dot(moment, scale(g, sigma))
        qdot = scale(ar.scalar(Fraction(1, 2)), quaternion_product(q, (*w, ar.scalar(0))))
        relative_qdot = tuple(x for sigma, relq in zip(self.relative_spin, quaternions)
            for x in scale(ar.scalar(Fraction(1, 2)), quaternion_product((*scale(g, sigma), ar.scalar(0)), relq)))
        external_power = sum(_dot(subtract(force, scale(mass, acceleration)), add(cross(w, offset), rdot))+
            _dot(couple, add(w, scale(g, sigma))) for force, mass, offset, rdot, couple, sigma in
            zip(external_forces, self.masses, offsets, rdots, external_couples, self.relative_spin))
        derivative = v+matvec(rotation, acceleration)+qdot+angular+relative_qdot+(power, external_power)
        diag = dict(offsets=offsets, rdots=rdots, rddots=rddots, quaternions=quaternions,
            rotations=rotations, forces=forces, couples=couples, angular_acceleration=angular,
            g=g, gd=gd, inertias=inertias, power=power)
        return derivative, diag

    def kinetic(self, time, values):
        _, g, _, offsets, rdots, _, _, _, inertias = self.shape(time, values)
        w = values[10:13]
        return sum(m*_dot(add(cross(w, r), rd), add(cross(w, r), rd))+
            _dot(add(w, scale(g, sigma)), matvec(inertia, add(w, scale(g, sigma))))
            for m, r, rd, sigma, inertia in zip(self.masses, offsets, rdots, self.relative_spin, inertias))/2

    def assembly_payload(self, time, values):
        """Use the final actual target material axes; g(T)=0 gives one twist."""
        ar = self.ar
        f, g, _, offsets, _, _, quaternions, rotations, _ = self.shape(time, values)
        target_inverse = transpose(rotations[0])
        local_rotations = tuple(_matmul(target_inverse, rotation) for rotation in rotations)
        positions = tuple(matvec(target_inverse, subtract(offset, offsets[0])) for offset in offsets)
        specifics = tuple(_matmul(_matmul(rotation, specific), transpose(rotation))
                          for rotation, specific in zip(local_rotations, self.specific))
        state = values[:6]+quaternion_product(values[6:10], quaternions[0])+matvec(target_inverse, values[10:13])+self.fuel
        parameters = dict(position_body_m=tuple(ar.box(position) for position in positions),
            inertia_per_kg_m2=tuple(ar.box(tuple(x for row in matrix for x in row)) for matrix in specifics),
            rotation_element_to_assembly=tuple(ar.box(tuple(x for row in matrix for x in row)) for matrix in local_rotations))
        anchors = [ar.box(tuple(ar.scalar(0) for _ in range(3)))]
        for member, initial, velocity in zip(self.fleet.members, self.tip_anchors, self.tip_anchor_velocity):
            tip = add(initial, scale(f, velocity))
            normal = tuple(map(ar.scalar, member.normal_body))
            gap = _dot(subtract(tip, tuple(map(ar.scalar, member.surface_anchor_body_m))), normal)/_dot(normal, normal)
            anchors.append(ar.box(subtract(tip, scale(gap, normal))))
        parameters['attachment_anchor_body_m'] = tuple(anchors)
        return ar.box(state), parameters

    def guards(self, time, values, diag, geometry, dt):
        """Reuse the occupied-component, peer, pad, reach and capacity gates."""
        ar, fleet, options = self.ar, self.fleet, self.o
        offsets, rotations, rdots = diag['offsets'], diag['rotations'], diag['rdots']
        duration = ar.scalar(options.lock_duration_s)
        f = time-time*time/(2*duration)
        world = quaternion_rotation(values[6:10])
        body, tips, rows = [], [], []
        for k, (r, rd, relq, relative_rotation, sigma) in enumerate(zip(offsets, rdots, diag['quaternions'], rotations, self.relative_spin)):
            absolute_q = quaternion_product(values[6:10], relq)
            spin = add(values[10:13], scale(diag['g'], sigma))
            body.append(matvec(world, r)+matvec(world, add(cross(values[10:13], r), rd))+absolute_q+
                        matvec(transpose(relative_rotation), spin)+(self.fuel[k],))
        for k, member in enumerate(fleet.members):
            j = k+1
            anchor_body = add(self.tip_anchors[k], scale(f, self.tip_anchor_velocity[k]))
            anchor = matvec(rotations[0], anchor_body)
            anchor_velocity = matvec(rotations[0], scale(diag['g'], self.tip_anchor_velocity[k]))
            relative = subtract(add(offsets[0], anchor), offsets[j])
            relative_dot = add(add(subtract(rdots[0], rdots[j]), cross(scale(diag['g'], self.relative_spin[0]), anchor)), anchor_velocity)
            extension = subtract(matvec(transpose(rotations[j]), relative), tuple(map(ar.scalar, fleet.mounts[k])))
            extension_velocity = matvec(transpose(rotations[j]), subtract(relative_dot,
                cross(scale(diag['g'], self.relative_spin[j]), relative)))
            sigma0, sigmaj = self.relative_spin[0], self.relative_spin[j]
            relative_ddot = add(subtract(diag['rddots'][0], diag['rddots'][j]), add(cross(scale(diag['gd'], sigma0), anchor),
                cross(scale(diag['g'], sigma0), cross(scale(diag['g'], sigma0), anchor))))
            relative_ddot = add(relative_ddot, add(scale(ar.scalar(2), cross(scale(diag['g'], sigma0), anchor_velocity)),
                matvec(rotations[0], scale(diag['gd'], self.tip_anchor_velocity[k]))))
            extension_acceleration = matvec(transpose(rotations[j]), add(subtract(relative_ddot,
                cross(scale(diag['gd'], sigmaj), relative)), add(scale(ar.scalar(-2), cross(scale(diag['g'], sigmaj), relative_dot)),
                cross(scale(diag['g'], sigmaj), cross(scale(diag['g'], sigmaj), relative)))))
            if norm_upper(extension_acceleration, ar) > Fraction(options.extension_acceleration_m_s2):
                raise _ProofUnresolved(member.id+': finite lock positioner acceleration not proven')
            force = diag['forces'][j]
            normal = tuple(map(ar.scalar, member.normal_body))
            gap = _dot(subtract(anchor_body, tuple(map(ar.scalar, member.surface_anchor_body_m))), normal)/_dot(normal, normal)
            surface = matvec(rotations[0], subtract(anchor_body, scale(gap, normal)))
            arm = subtract(add(offsets[0], surface), offsets[j])
            couple = subtract(diag['couples'][j], cross(arm, force))
            tip_arm = subtract(add(offsets[0], anchor), offsets[j])
            tip_couple = subtract(diag['couples'][j], cross(tip_arm, force))
            force_target = matvec(transpose(rotations[0]), force)
            gap = _dot(subtract(anchor_body, tuple(map(ar.scalar, member.surface_anchor_body_m))), normal)
            bounds = ar.box((gap,))
            compression = ar.interval((max(Fraction(0), -bounds.upper[0]), max(Fraction(0), -bounds.lower[0])))
            power_upper = norm_upper(force, ar)*(norm_upper(extension_velocity, ar)+norm_upper(anchor_velocity, ar))+norm_upper(tip_couple, ar)*norm_upper(
                scale(diag['g'], subtract(self.relative_spin[j], self.relative_spin[0])), ar)
            rows.append(dict(compression=compression, force=matvec(world, force), couple=matvec(world, couple),
                power=ar.interval((-power_upper, power_upper)), activation=ar.scalar(1),
                bond_force_unscaled=force_target, force_patch=force_target, normal_force=ar.scalar(0),
                couple_unscaled=matvec(world, couple), tip=add(body[0][:3], matvec(world, anchor)),
                gap=gap, u=matvec(world, extension_velocity),
                spin=scale(diag['g'], subtract(self.relative_spin[j], self.relative_spin[0]))))
            tips.append(extension+extension_velocity)
        context = SimpleNamespace(fleet=fleet, o=options, ar=ar,
            normals=tuple(tuple(map(ar.scalar, normal)) for normal in fleet.normals),
            specific=self.specific, inverse=tuple(inverse3(value) for value in self.specific))
        checks = ContactStepEncloser()._checks(context,
            dict(body=body, tips=tips, rows=rows, rotations=tuple(_matmul(world, local) for local in rotations),
                 masses=self.masses, spins=tuple(matvec(world, add(values[10:13], scale(diag['g'], sigma)))
                    for sigma in self.relative_spin)), geometry, dt)
        return checks


@dataclass(frozen=True)
class FiniteLockResult:
    success: bool
    message: str
    elapsed_s: float
    state: IntervalBox | None
    numerical: tuple | None
    parameters: object
    audit: dict
    numerical_parameters: object = None


class FiniteLockTransition:
    """A finite active closure, with every held flow and load proved first."""
    def run(self, *, fleet, options, geometry, contact, environment=None):
        def declaration():
            return dict(source_files=_source_manifest(),
                initial=contact.audit['continuous_enclosure']['final_set'],
                numerical=contact.audit['continuous_enclosure']['step_records'][-1]['numerical_endpoint'],
                trapped_pad_inventory=_pad_inventory(contact, fleet.chaser_count), started=fleet.started,
                options=asdict(options), geometry_fingerprint=geometry.fingerprint,
                dry=fleet.dry.tolist(), specific=fleet.specific.tolist(), mounts=fleet.mounts.tolist(),
                anchors=fleet.anchors.tolist(), normals=fleet.normals.tolist(), mu=fleet.mu,
                members=[dict(id=member.id, properties=asdict(member.properties),
                    arm_reach_m=member.arm_reach_m, patch_id=member.patch_id,
                    surface_anchor_body_m=member.surface_anchor_body_m.tolist(),
                    normal_body=member.normal_body.tolist(), tangent_u_body=member.tangent_u_body.tolist(),
                    tangent_v_body=member.tangent_v_body.tolist()) for member in fleet.members],
                gravity_gradient=fleet.gravity_gradient,
                environment=None if environment is None else environment.as_dict())
        declared = copy.deepcopy(declaration())
        frozen = fingerprint(declared)
        plan = self._preflight(fleet=fleet, options=options, geometry=geometry, contact=contact, environment=environment)
        if not plan.success:
            audit = copy.deepcopy(plan.audit)
            planned = {key: audit.get(key) for key in ('accepted_steps', 'accepted_elapsed_s',
                'exact_accepted_elapsed_s', 'step_records', 'proof_chain_sha256', 'refusal')}
            audit.update(preflight=dict(status='REFUSED', declaration_identity=frozen,
                declaration=declared,
                planned_prefix=planned, planned_outputs=_planned_outputs(audit), physical_motion_committed=False),
                accepted_steps=0, accepted_elapsed_s=0., exact_accepted_elapsed_s='0',
                physical_end_time_s=fleet.started, step_records=[], proof_chain_sha256=None,
                integrated_absolute_interface_work_upper_j='0', constitutive_transition_applied=False)
            return replace(plan, elapsed_s=0., audit=audit)
        preflight = dict(status='PROVEN', declaration_identity=frozen, full_duration_s=plan.audit['duration_s'],
            declaration=declared,
            planned_steps=plan.audit['accepted_steps'], proof_chain_sha256=plan.audit['proof_chain_sha256'],
            absolute_interface_work_upper_j=plan.audit['integrated_absolute_interface_work_upper_j'],
            finite_power_work_budget_j=plan.audit['finite_power_work_budget_j'],
            trapped_pad_inventory_identity=plan.audit['trapped_pad_inventory_identity'],
            admission_scope='Complete finite profile and every supplied-state load/reach/power/geometry/resource hold proved before any lock motion')
        preflight['identity'] = fingerprint(preflight)
        audit = copy.deepcopy(plan.audit)
        audit['preflight'] = preflight
        records = audit['step_records']
        executed = []
        chain, elapsed = None, Fraction(0)
        try:
            if fingerprint(declaration()) != frozen:
                raise ValueError('Finite lock declaration changed after complete preflight')
            import flint
            with _ARB_LOCK, flint.ctx.workprec(128):
                ar = _ArbArithmetic(flint)
                initial = contact.audit['continuous_enclosure']['step_records'][-1]['numerical_endpoint']
                nominal = _LockField(fleet, options, IntervalBox.point(initial), environment, ar)
                start = ar.box(nominal.initial)
                values = np.array([float((lo+hi)/2) for lo, hi in zip(start.lower, start.upper)])
                for record in records:
                    if fingerprint(declaration()) != frozen:
                        raise ValueError('Finite lock declaration changed during admitted execution')
                    q0, q1 = Fraction(record['exact_start_time_s']), Fraction(record['exact_end_time_s'])
                    candidate = self._numerical_step(nominal, ar, values, q0, q1-q0, fleet.body_count)
                    if not np.array_equal(candidate, np.asarray(record['numerical_endpoint'])):
                        raise ValueError('Finite lock execution differs from its complete admitted trajectory')
                    values = candidate
                    executed.append(record)
                    chain = hashlib.sha256(((chain or '')+record['proof_sha256']).encode('ascii')).hexdigest()
                    elapsed = q1
        except (ImportError, _ProofUnresolved, ArithmeticError, ValueError) as error:
            audit['preflight_evidence'] = dict(planned_outputs=_planned_outputs(audit), planned_step_records=records)
            absolute_work = sum((Fraction(record['exact_end_time_s'])-Fraction(record['exact_start_time_s']))*
                sum(Fraction(value['positioner_power_w']) for value in record['checks']['member_upper'].values())
                for record in executed)
            audit.update(accepted_steps=len(executed), accepted_elapsed_s=float(elapsed),
                exact_accepted_elapsed_s=str(elapsed), physical_end_time_s=fleet.started+float(elapsed),
                step_records=executed, proof_chain_sha256=chain, refusal=str(error),
                integrated_absolute_interface_work_upper_j=str(absolute_work),
                constitutive_transition_applied=bool(executed), final_rigid_coordinate_change_applied=False,
                execution_replay=dict(status='REFUSED', preflight_identity=preflight['identity']))
            if executed:
                last = executed[-1]
                audit['accepted_numerical_generalized_state'] = last['numerical_endpoint']
                final = _box(last['proof']['endpoint']).hull(IntervalBox.point(last['numerical_endpoint']))
                audit['accepted_generalized_state_set'] = final.as_dict()
                audit['internal_mechanical_work_j'] = last['numerical_endpoint'][-2]
                audit['external_com_frame_work_j'] = last['numerical_endpoint'][-1]
                audit['internal_mechanical_work_enclosure_j'] = final.slice(final.dimension-2, final.dimension-1).as_dict()
                audit['external_com_frame_work_enclosure_j'] = final.slice(final.dimension-1, final.dimension).as_dict()
            else:
                audit.pop('accepted_numerical_generalized_state', None)
                audit.pop('accepted_generalized_state_set', None)
            return FiniteLockResult(False, 'finite_lock_execution_refused: '+str(error), float(elapsed),
                None, None, None, audit)
        audit['execution_replay'] = dict(status='COMPLETED', preflight_identity=preflight['identity'],
            independently_reintegrated_steps=len(executed), exact_elapsed_s=str(elapsed),
            all_numerical_endpoints_identical=True)
        audit.update(constitutive_transition_applied=True, final_rigid_coordinate_change_applied=True)
        return replace(plan, audit=audit)

    @staticmethod
    def _numerical_step(model, ar, values, time, duration, body_count):
        def rhs(at, state):
            derivative, _ = model.evaluate(ar.scalar(at), ar.vector(IntervalBox.point(state)))
            box = ar.box(derivative)
            return np.array([float((lo+hi)/2) for lo, hi in zip(box.lower, box.upper)])
        t, step = float(time), float(duration)
        k1 = rhs(t, values); k2 = rhs(t+step/2, values+step*k1/2)
        k3 = rhs(t+step/2, values+step*k2/2); k4 = rhs(t+step, values+step*k3)
        result = values+step*(k1+2*k2+2*k3+k4)/6
        for begin in (6,)+tuple(13+4*k for k in range(body_count)):
            result[begin:begin+4] /= np.linalg.norm(result[begin:begin+4])
        return result

    def _preflight(self, *, fleet, options, geometry, contact, environment=None):
        initial = _box(contact.audit['continuous_enclosure']['final_set'])
        numerical = contact.audit['continuous_enclosure']['step_records'][-1]['numerical_endpoint']
        if initial.dimension != fleet.ledger_start or len(numerical) != fleet.ledger_start:
            raise ValueError('Finite lock requires the complete carried contact endpoint')
        inventory = _pad_inventory(contact, fleet.chaser_count)
        inventory_energy = _box(inventory['energy_j'])
        duration = Fraction(options.lock_duration_s)
        count = max(1, int(np.ceil(float(duration)/options.max_step_s)))
        dt = duration/count
        audit = dict(method='finite-prescribed-relative-motion-lock-v1', duration_s=str(duration),
            initial_contact_set=initial.as_dict(), accepted_steps=0, proof_chain_sha256=None,
            physical_start_time_s=fleet.started, physical_end_time_s=fleet.started,
            accepted_elapsed_s=0., exact_accepted_elapsed_s='0',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            source_files=_source_manifest(),
            retained_locked_pad_energy_j=contact.audit['final_contact_potential_j'],
            retained_locked_pad_energy_enclosure_j=inventory_energy.as_dict(), trapped_pad_inventory=inventory,
            trapped_pad_inventory_identity=fingerprint(inventory),
            dissipated_pad_energy_at_instantaneous_event_j=0., step_records=[],
            constitutive_transition='Passive compliant pads to ideal strain-trapping clamps and active articulated joints; existing stored elastic energy is a retained independent constant inventory, not the continuing passive spring potential',
            gravity_gradient=fleet.gravity_gradient,
            environment_contract=environment.as_dict() if callable(getattr(environment, 'as_dict', None)) else None,
            actuation='Active massless positioners follow finite braking profiles f(t)=t-t^2/(2T); bounded Newton-Euler pad reactions; exact internal servo is a declared mechanism assumption',
            scope='Declared finite rigid/positioner mechanism and supplied complete endpoint set; no material, servo hardware or flight qualification')
        try:
            import flint
            with _ARB_LOCK, flint.ctx.workprec(128):
                ar = _ArbArithmetic(flint)
                model = _LockField(fleet, options, initial, environment, ar)
                nominal_model = _LockField(fleet, options, IntervalBox.point(numerical), environment, ar)
                current = ar.box(model.initial)
                y = np.array([float((lo+hi)/2) for lo, hi in zip(ar.box(nominal_model.initial).lower,
                                                                          ar.box(nominal_model.initial).upper)])
                current = current.hull(IntervalBox.point(y))
                before = nominal_model.kinetic(ar.scalar(0), ar.vector(IntervalBox.point(y)))
                before_box = ar.box((before,))
                audit['initial_com_frame_kinetic_energy_enclosure_j'] = ar.box((model.kinetic(
                    ar.scalar(0), ar.vector(current)),)).as_dict()
                absolute_work_upper = Fraction(0)
                swept, time = current, Fraction(0)
                for index in range(count):
                    start = ar.vector(current)
                    trial = IntervalBox(tuple(x-Fraction(1, 10**20) for x in current.lower),
                                        tuple(x+Fraction(1, 10**20) for x in current.upper))
                    for attempt in range(24):
                        field, diag = model.evaluate(ar.interval((time, time+dt)), ar.vector(trial))
                        image = ar.box(add(start, scale(ar.interval((0, dt)), field)))
                        selected = tuple(not (a < c and d < b) for a, b, c, d in
                                         zip(trial.lower, trial.upper, image.lower, image.upper))
                        if not any(selected):
                            checks = model.guards(ar.interval((time, time+dt)), ar.vector(trial), diag, geometry, dt)
                            endpoint = ar.box(add(start, scale(ar.scalar(dt), field)))
                            break
                        joined = trial.hull(image)
                        trial = IntervalBox(tuple((a+b)/2-Fraction(3, 4)*(b-a)-Fraction(1, 10**20) if change else old
                            for a, b, old, change in zip(joined.lower, joined.upper, trial.lower, selected)),
                            tuple((a+b)/2+Fraction(3, 4)*(b-a)+Fraction(1, 10**20) if change else old
                            for a, b, old, change in zip(joined.lower, joined.upper, trial.upper, selected)))
                    else:
                        raise _ProofUnresolved('finite lock Picard inclusion exceeded its finite budget')
                    y = self._numerical_step(nominal_model, ar, y, time, dt, fleet.body_count)
                    proof = dict(exact_start_time_s=str(time), exact_end_time_s=str(time+dt),
                        initial=current.as_dict(), endpoint=endpoint.as_dict(), swept=trial.as_dict(), checks=checks)
                    digest = fingerprint(proof)
                    audit['proof_chain_sha256'] = hashlib.sha256(((audit['proof_chain_sha256'] or '')+digest).encode('ascii')).hexdigest()
                    audit['step_records'].append(dict(exact_start_time_s=str(time), exact_end_time_s=str(time+dt),
                        proof_sha256=digest, proof=proof, numerical_endpoint=y.tolist(), checks=checks))
                    current = endpoint.hull(IntervalBox.point(y))
                    swept = swept.hull(trial)
                    time += dt
                    audit['accepted_steps'] += 1
                    absolute_work_upper += dt*sum(Fraction(value['positioner_power_w'])
                        for value in checks['member_upper'].values())
                    audit.update(accepted_elapsed_s=float(time), exact_accepted_elapsed_s=str(time),
                        physical_end_time_s=fleet.started+float(time),
                        integrated_absolute_interface_work_upper_j=str(absolute_work_upper),
                        accepted_generalized_state_set=current.as_dict(), accepted_numerical_generalized_state=y.tolist())
                audit.update(final_set=current.as_dict(), swept=swept.as_dict(),
                    internal_mechanical_work_j=float(y[-2]), external_com_frame_work_j=float(y[-1]),
                    internal_mechanical_work_enclosure_j=current.slice(current.dimension-2, current.dimension-1).as_dict(),
                    external_com_frame_work_enclosure_j=current.slice(current.dimension-1, current.dimension).as_dict(),
                    finite_power_work_budget_j=str(Fraction(options.positioner_power_limit_w)*duration*fleet.chaser_count),
                    work_scope='The ideal active joint can exchange mechanical work in either direction within every proved power bound; finite throughput is bounded by the integral of absolute interface power, not an assumed instantaneous energy deletion or a qualified battery/heat-sink capacity',
                    impulse_at_final_coordinate_change_kg_m_s=0.,
                    angular_impulse_at_final_coordinate_change_kg_m2_s=0.)
                after = nominal_model.kinetic(ar.scalar(duration), ar.vector(IntervalBox.point(y)))
                after_box = ar.box((after,))
                before_value = float(sum(before_box.lower+before_box.upper)/2)
                after_value = float(sum(after_box.lower+after_box.upper)/2)
                audit.update(initial_com_frame_kinetic_energy_j=before_value,
                    final_com_frame_kinetic_energy_j=after_value,
                    internal_absorbed_mechanical_work_j=-float(y[-2]),
                    energy_balance_residual_j=after_value-before_value-float(y[-2])-float(y[-1]))
                state, parameters = model.assembly_payload(ar.scalar(duration), ar.vector(current))
                point_state, point_parameters = nominal_model.assembly_payload(ar.scalar(duration), ar.vector(IntervalBox.point(y)))
                numeric = tuple(float((lo+hi)/2) for lo, hi in zip(point_state.lower, point_state.upper))
                parameters = {key: tuple(box.hull(numerical) for box, numerical in zip(value, point_parameters[key]))
                              for key, value in parameters.items()}
                state = state.hull(IntervalBox.point(numeric))
                audit['rigid_state_set'] = state.as_dict()
                audit['rigid_parameter_sets'] = {key: [box.as_dict() for box in value] for key, value in parameters.items()}
                return FiniteLockResult(True, 'finite_load_bounded_lock_completed', float(duration), state,
                    numeric, parameters, audit, point_parameters)
        except (ImportError, _ProofUnresolved, ArithmeticError, ValueError) as error:
            audit['refusal'] = str(error)
            return FiniteLockResult(False, 'finite_lock_not_proven: '+str(error),
                float(audit['accepted_steps']*dt), None, None, None, audit)
