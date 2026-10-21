"""Carried whole-state inclusion for the maintained postcapture rigid model.

Every held propagation starts from the previous endpoint set. Fuel-coordinate
jumps follow the same split law as ``AttachedAssembly.advance_with_audit``.
An unresolved inclusion refuses propagation instead of restarting at a point.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import copy
import hashlib
from pathlib import Path

import numpy as np

from orbexa.control.certification.reachability import IntervalBox, _ARB_LOCK, _ArbArithmetic, _ProofUnresolved
from orbexa.core.dynamics.reference_frame import fingerprint
from orbexa.simulation.numerics.rigid_interval import add, subtract, scale, cross, matvec, transpose, inverse3, square, normalized_quaternion, quaternion_rotation, quaternion_product


def packed_state(state):
    return tuple(state.position_eci_m)+tuple(state.velocity_eci_m_s)+tuple(state.quaternion_body_to_eci)+tuple(
        state.angular_velocity_body_rad_s)+tuple(state.fuel_kg)


@dataclass(frozen=True)
class AssemblyStateEnclosure:
    box: IntervalBox
    exact_time_s: Fraction
    parameter_fingerprint: str
    source: dict
    completed_holds: int = 0
    proof_chain_sha256: str | None = None
    last_swept: IntervalBox | None = None

    def as_dict(self):
        return dict(method='carried-rigid-assembly-state-set-v1', box=self.box.as_dict(),
            exact_time_s=str(self.exact_time_s), parameter_fingerprint=self.parameter_fingerprint,
            source=copy.deepcopy(self.source), completed_holds=self.completed_holds,
            proof_chain_sha256=self.proof_chain_sha256,
            last_swept=None if self.last_swept is None else self.last_swept.as_dict())

    def successor(self, certificate, numerical_state):
        if not certificate.passed or certificate.initial != self.box:
            raise ValueError('A carried assembly successor requires its same-start proved flow')
        representative = IntervalBox.point(packed_state(numerical_state))
        box = certificate.endpoint.hull(representative)
        proof_hash = fingerprint(certificate.as_dict())
        chain = hashlib.sha256(((self.proof_chain_sha256 or '')+proof_hash).encode('ascii')).hexdigest()
        return AssemblyStateEnclosure(box, self.exact_time_s+certificate.duration_s,
            self.parameter_fingerprint, self.source, self.completed_holds+1, chain, certificate.swept)


@dataclass(frozen=True)
class AssemblyFlowCertificate:
    status: str
    reason: str
    context: dict
    initial: IntervalBox
    duration_s: Fraction
    endpoint: IntervalBox | None = None
    swept: IntervalBox | None = None

    @property
    def passed(self):
        return self.status == 'PROVEN'

    def as_dict(self):
        return dict(method='arb-full-rigid-held-flow-and-split-fuel-v1', status=self.status,
            passed=self.passed, reason=self.reason, context=copy.deepcopy(self.context),
            initial=self.initial.as_dict(), duration_s=str(self.duration_s),
            endpoint=None if self.endpoint is None else self.endpoint.as_dict(),
            swept=None if self.swept is None else self.swept.as_dict(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            scope='Exact declared rigid assembly with held physical nozzle commands and split co-rotating fuel; supplied initial and parameter sets, no flexible-body or flight claim')


def assembly_parameter_fingerprint(assembly):
    return fingerprint(dict(elements=[element.to_dict() for element in assembly.elements],
        parameter_enclosure=None if assembly.parameter_enclosure is None else {
            key: [box.as_dict() for box in value] for key, value in assembly.parameter_enclosure.items()},
        gravity_gradient=assembly.gravity_gradient, mu_m3_s2=assembly.mu_m3_s2,
        environment=None if assembly.environment is None else assembly.environment.as_dict()))


class _AssemblyField:
    def __init__(self, assembly, propulsion, commands, ar):
        self.assembly, self.ar = assembly, ar
        self.environment = None
        if assembly.environment is not None:
            from orbexa.simulation.completion.environment.enclosure import NonlinearEnvironmentEnclosure
            self.environment = NonlinearEnvironmentEnclosure.from_environment(assembly.environment)
        self.environment_parameters = None if self.environment is None else self.environment.force_parameters(ar)
        parameters = assembly.parameter_enclosure
        self.positions, self.specific, self.local_rotations = [], [], []
        for index, element in enumerate(assembly.elements):
            if parameters is None:
                position = ar.vector(IntervalBox.point(element.position_body_m))
                specific = ar.vector(IntervalBox.point(element.inertia_per_kg_m2.flat))
                rotation = ar.vector(IntervalBox.point(element.rotation_element_to_assembly.flat))
            else:
                position = ar.vector(parameters['position_body_m'][index])
                specific = ar.vector(parameters['inertia_per_kg_m2'][index])
                rotation = ar.vector(parameters['rotation_element_to_assembly'][index])
            self.positions.append(position)
            self.specific.append(tuple(tuple(specific[3*i+j] for j in range(3)) for i in range(3)))
            self.local_rotations.append(tuple(tuple(rotation[3*i+j] for j in range(3)) for i in range(3)))
        zero = tuple(ar.scalar(0) for _ in range(3))
        self.forces = [zero for _ in assembly.elements]
        self.moments_at_origin = [zero for _ in assembly.elements]
        self.flow = [Fraction(0) for _ in assembly.elements]
        from orbexa.simulation.attached.assembly.model import STANDARD_GRAVITY
        for command, nozzle in zip(commands, propulsion.thrusters):
            k = nozzle.owner_index
            element = assembly.elements[k]
            # Nozzle installation coordinates are expressed in the actual
            # constituent bus, so uncertain capture geometry rotates them.
            direction = element.rotation_element_to_assembly.T@nozzle.direction_body
            lever = element.rotation_element_to_assembly.T@(nozzle.position_body_m-element.position_body_m)
            force = scale(ar.scalar(command), matvec(self.local_rotations[k], tuple(map(ar.scalar, direction))))
            origin = add(self.positions[k], matvec(self.local_rotations[k], tuple(map(ar.scalar, lever))))
            self.forces[k] = add(self.forces[k], force)
            self.moments_at_origin[k] = add(self.moments_at_origin[k], cross(origin, force))
            self.flow[k] += abs(Fraction(float(command)))/(Fraction(nozzle.specific_impulse_s)*Fraction(STANDARD_GRAVITY))

    def mass_properties(self, fuel):
        ar = self.ar
        masses = tuple(ar.scalar(element.dry_mass_kg)+value for element, value in zip(self.assembly.elements, fuel))
        total = sum(masses)
        if not total > 0:
            raise _ProofUnresolved('assembly mass set is not positive')
        center = tuple(sum(m*p[j] for m, p in zip(masses, self.positions))/total for j in range(3))
        offsets = tuple(subtract(p, center) for p in self.positions)
        inertia = tuple(tuple(sum(m*(specific[i][j]+(
            sum(square(offset[k], ar) for k in range(3) if k != i) if i == j else -offset[i]*offset[j]))
            for m, specific, offset in zip(masses, self.specific, offsets)) for j in range(3)) for i in range(3))
        return masses, total, center, offsets, inertia

    def deplete(self, values, amount):
        """Exact finite COM-coordinate change without interval cancellation."""
        ar = self.ar
        old = values[13:]
        _, total, center, _, _ = self.mass_properties(old)
        changes = tuple(-ar.scalar(amount*rate) for rate in self.flow)
        new = tuple(a+b for a, b in zip(old, changes))
        if any(ar.box((value,)).lower[0] < 0 for value in new):
            raise _ProofUnresolved('carried assembly tank set cannot support the held command')
        new_total = total+sum(changes)
        shift = tuple(sum(dm*(p[j]-center[j]) for dm, p in zip(changes, self.positions))/new_total for j in range(3))
        rotation = quaternion_rotation(values[6:10])
        return add(values[:3], matvec(rotation, shift))+add(values[3:6], matvec(rotation, cross(values[10:13], shift)))+values[6:13]+new

    def evaluate(self, values):
        ar, assembly = self.ar, self.assembly
        p, v, q, w, fuel = values[:3], values[3:6], values[6:10], values[10:13], values[13:]
        masses, total, center, offsets, inertia = self.mass_properties(fuel)
        rotation = quaternion_rotation(q)
        radius2 = sum(square(x, ar) for x in p)
        if not radius2 > 0:
            raise _ProofUnresolved('assembly orbital set includes the gravity singularity')
        radius = radius2.sqrt()
        zero = tuple(ar.scalar(0) for _ in range(3))
        force = tuple(sum(local[j] for local in self.forces) for j in range(3))
        moment_origin = tuple(sum(local[j] for local in self.moments_at_origin) for j in range(3))
        torque = subtract(moment_origin, cross(center, force))
        common_extra = zero
        if self.environment is not None:
            # A virtual reference on the same typed model/parameters. The
            # current fixed-mass COM identity sum(m_i*r_i)=0 removes its
            # torque exactly; it is not a relocated or sensed spacecraft.
            common_extra = self.environment.evaluate(assembly.elements[0].name,
                p, v, fuel[0], ar, self.environment_parameters)
            for k, (element, mass, offset) in enumerate(zip(assembly.elements, masses, offsets)):
                position = add(p, matvec(rotation, offset))
                velocity = add(v, matvec(rotation, cross(w, offset)))
                extra = self.environment.evaluate(element.name, position, velocity, fuel[k], ar, self.environment_parameters)
                local = scale(mass, matvec(transpose(rotation), subtract(extra, common_extra)))
                force = add(force, local)
                torque = add(torque, cross(offset, local))
        if assembly.gravity_gradient:
            radial = matvec(transpose(rotation), scale(1/radius, p))
            torque = add(torque, scale(3*ar.scalar(assembly.mu_m3_s2)/radius**3,
                                      cross(radial, matvec(inertia, radial))))
        linear = add(add(scale(-ar.scalar(assembly.mu_m3_s2)/radius**3, p), common_extra), scale(1/total, matvec(rotation, force)))
        qdot = scale(ar.scalar(Fraction(1, 2)), quaternion_product(q, (*w, ar.scalar(0))))
        angular = matvec(inverse3(inertia), subtract(torque, cross(w, matvec(inertia, w))))
        return v+linear+qdot+angular+tuple(ar.scalar(0) for _ in fuel)


@dataclass(frozen=True)
class AssemblyFlowEncloser:
    precision_bits: int = 128
    max_step_s: float = .05
    max_substeps: int = 4096
    max_picard_iterations: int = 24

    def bound(self, *, assembly, state, propulsion, commands_n, duration_s):
        h = Fraction(float(duration_s))
        if h <= 0 or state.enclosure is None:
            raise ValueError('Carried assembly flow requires a positive hold and an explicit start set')
        initial = state.enclosure.box
        parameter_id = assembly_parameter_fingerprint(assembly)
        if state.enclosure.parameter_fingerprint != parameter_id:
            raise ValueError('Carried assembly geometry/environment declaration changed')
        if not initial.contains(IntervalBox.point(packed_state(state))):
            raise ValueError('Carried assembly start set omits its numerical representative')
        commands = np.asarray(commands_n, dtype=float)
        if commands.shape != (len(propulsion.thrusters),) or np.any(~np.isfinite(commands)):
            raise ValueError('Carried assembly propagation needs complete physical nozzle commands')
        if any(abs(command) > nozzle.max_force_n for command, nozzle in zip(commands, propulsion.thrusters)):
            raise ValueError('Carried assembly command exceeds the declared hardware')
        context = dict(
            physical_graph='All supplied-state physical quaternions are unit; qdot preserves that graph. For every uncertain fixed-mass member family, center=sum(m_i*p_i)/sum(m_i) and r_i=p_i-center imply sum(m_i*r_i)=0. Unchanged half-depletion maps reform that same identity at their actual new masses.',
            common_environment_reference='Same typed environment and parameters evaluated at the virtual assembly COM with the first member drag law/fuel; every actual body retains its own position, velocity, fuel and drag law. The reference and every body must pass their declared force domains.',
            exact_start_time_s=str(state.enclosure.exact_time_s),
            parameter_fingerprint=parameter_id, commands_n=commands.tolist(), propulsion=propulsion.to_dict(),
            max_step_s=self.max_step_s, max_substeps=self.max_substeps,
            max_picard_iterations=self.max_picard_iterations, precision_bits=self.precision_bits)
        try:
            import flint
            with _ARB_LOCK, flint.ctx.workprec(self.precision_bits):
                ar = _ArbArithmetic(flint)
                model = _AssemblyField(assembly, propulsion, commands, ar)
                # Match the actual Strang depletion boundaries. Further
                # Picard subdivisions do not introduce extra fuel jumps.
                count = max(1, int(np.ceil(float(h)/assembly.max_integration_step_s)))
                dt = h/count
                subcount = max(1, int(-(-dt//Fraction(self.max_step_s))))
                if count*subcount > self.max_substeps:
                    raise _ProofUnresolved('full assembly flow exceeds its finite substep budget')
                endpoint, swept, iterations = initial, initial, 0
                for _ in range(count):
                    endpoint = ar.box(model.deplete(ar.vector(endpoint), dt/2))
                    swept = swept.hull(endpoint)
                    step = dt/subcount
                    for _ in range(subcount):
                        start = ar.vector(endpoint)
                        trial = IntervalBox(tuple(x-Fraction(1, 10**18) for x in endpoint.lower),
                                            tuple(x+Fraction(1, 10**18) for x in endpoint.upper))
                        for _ in range(self.max_picard_iterations):
                            iterations += 1
                            field = model.evaluate(ar.vector(trial))
                            image = ar.box(add(start, scale(ar.interval((0, step)), field)))
                            selected = tuple(not (a < c and d < b) for a, b, c, d in
                                             zip(trial.lower, trial.upper, image.lower, image.upper))
                            if not any(selected):
                                endpoint = ar.box(add(start, scale(ar.scalar(step), field)))
                                # Each independently outward endpoint cover
                                # intersects the already proved whole-flow set.
                                from orbexa.simulation.contact.dynamics.rotation import intersect
                                endpoint = intersect(endpoint,trial)
                                swept = swept.hull(trial)
                                break
                            # Keep the inherited uncertainty and enlarge only
                            # the uncovered Picard image excess. The same
                            # strict inclusion condition decides the proof.
                            trial = IntervalBox(tuple(lo-Fraction(3, 2)*max(Fraction(0),lo-c)-Fraction(1, 10**18)
                                for lo,c in zip(trial.lower,image.lower)),
                                tuple(hi+Fraction(3, 2)*max(Fraction(0),d-hi)+Fraction(1, 10**18)
                                for hi,d in zip(trial.upper,image.upper)))
                        else:
                            raise _ProofUnresolved('full assembly Picard inclusion exceeded its finite budget')
                    endpoint = ar.box(model.deplete(ar.vector(endpoint), dt/2))
                    swept = swept.hull(endpoint)
                context.update(proved_substeps=count*subcount, picard_iterations=iterations)
                return AssemblyFlowCertificate('PROVEN', 'complete_held_flow_from_carried_set', context, initial, h, endpoint, swept)
        except ImportError:
            return AssemblyFlowCertificate('UNAVAILABLE', 'arb_backend_unavailable', context, initial, h)
        except (_ProofUnresolved, ArithmeticError, ValueError) as error:
            return AssemblyFlowCertificate('UNRESOLVED', str(error), context, initial, h)
