"""Whole-held-step Newton--Euler bounds for a declared rigid attached assembly.

The enclosure concerns the exact underlying rigid-body ODE and its specified
fuel-depletion/COM-coordinate jumps. It does not validate an RK4 trajectory or
turn an approximate previous endpoint into a certified initial set. Extra
environment callbacks require their own interval model and are refused here.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import copy
from pathlib import Path

import numpy as np

from orbexa.control.reachability import (
    IntervalBox, _ARB_LOCK, _ArbArithmetic, _ProofUnresolved,
)
from orbexa.core.reference_frame import fingerprint
from orbexa.simulation.rigid_interval import (
    cross as _cross, add as _add, subtract as _sub, scale as _scale,
    matvec as _matvec, inverse3 as _inverse, transpose,
    normalized_quaternion, quaternion_rotation,
)


_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_SOURCE_FILES = {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
    for name in ('attachment_enclosure.py', 'rigid_interval.py')}


@dataclass(frozen=True)
class AttachmentLoadCertificate:
    """A source-bound sufficient load condition; refusal is not an overload claim."""
    status: str
    reason: str
    context: dict
    angular_velocity_swept: IntervalBox | None = None
    members: dict | None = None
    maximum_capacity_ratio_upper: str | None = None
    iterations: int = 0

    @property
    def passed(self):
        return self.status == 'PROVEN'

    def as_dict(self):
        return dict(method='arb-whole-held-attachment-loads-v1', status=self.status,
            passed=self.passed, reason=self.reason, backend_source_sha256=_SOURCE_SHA256,
            source_files=dict(_SOURCE_FILES),
            context=copy.deepcopy(self.context), angular_velocity_swept=None if self.angular_velocity_swept is None
                else self.angular_velocity_swept.as_dict(), members=copy.deepcopy(self.members or {}),
            maximum_capacity_ratio_upper=self.maximum_capacity_ratio_upper,
            picard_iterations=self.iterations,
            scope='Exact declared rigid held-wrench ODE and split fuel depletion, conditional on the declared start set; no RK4 error, compliant-latch, flexible-body or flight certificate')


@dataclass(frozen=True)
class AttachmentLoadEncloser:
    """Finite-budget Arb inclusion and uniform joint loads before a held command."""
    precision_bits: int = 128
    max_picard_iterations: int = 24
    max_substeps: int = 256
    max_step_s: float = .05

    def __post_init__(self):
        if (isinstance(self.precision_bits, bool) or not isinstance(self.precision_bits, int)
                or not 64 <= self.precision_bits <= 4096):
            raise ValueError('Attachment bounds require 64--4096 Arb precision bits')
        if (isinstance(self.max_picard_iterations, bool) or not isinstance(self.max_picard_iterations, int)
                or not 1 <= self.max_picard_iterations <= 256):
            raise ValueError('Declare a finite attachment Picard iteration budget')
        if (isinstance(self.max_substeps, bool) or not isinstance(self.max_substeps, int)
                or not 1 <= self.max_substeps <= 4096 or not np.isfinite(self.max_step_s) or self.max_step_s <= 0):
            raise ValueError('Declare finite positive attachment substep limits')

    def bound(self, *, assembly, state, propulsion, commands_n, duration_s,
              initial_spin: IntervalBox | None = None,
              initial_position: IntervalBox | None = None,
              initial_velocity: IntervalBox | None = None, capacity_fraction=1.):
        from orbexa.simulation.assembly import STANDARD_GRAVITY, positive
        duration = Fraction(positive(duration_s, 'Attachment hold duration'))
        fraction = Fraction(positive(capacity_fraction, 'Whole-step attachment capacity fraction'))
        if fraction > 1:
            raise ValueError('Whole-step attachment admission cannot increase physical capacities')
        commands = np.asarray(commands_n, dtype=float)
        if commands.shape != (len(propulsion.thrusters),) or np.any(~np.isfinite(commands)):
            raise ValueError('Whole-step loads require the actual finite nozzle commands')
        if any(abs(command) > nozzle.max_force_n for command, nozzle in zip(commands, propulsion.thrusters)):
            raise ValueError('Whole-step loads cannot admit commands outside nozzle authority')
        assembly.mass_properties(state.fuel_kg)
        boxes = tuple(box or IntervalBox.point(value) for box, value in (
            (initial_spin, state.angular_velocity_body_rad_s),
            (initial_position, state.position_eci_m), (initial_velocity, state.velocity_eci_m_s)))
        if any(not isinstance(box, IntervalBox) or box.dimension != 3 for box in boxes):
            raise ValueError('Attachment initial spin, position and velocity sets need three components')
        if any(not box.contains(IntervalBox.point(value)) for box, value in zip(boxes,
                (state.angular_velocity_body_rad_s, state.position_eci_m, state.velocity_eci_m_s))):
            raise ValueError('The declared attachment start sets must contain the supplied state')
        context = dict(state=state.to_dict(), initial_spin=boxes[0].as_dict(),
            initial_position=boxes[1].as_dict(), initial_velocity=boxes[2].as_dict(),
            commands_n=commands.tolist(), duration_s=str(duration), assembly=assembly.to_dict(),
            propulsion=propulsion.to_dict(), precision_bits=self.precision_bits,
            capacity_fraction=str(fraction), max_step_s=self.max_step_s,
            max_substeps=self.max_substeps, max_picard_iterations=self.max_picard_iterations)
        context['request_fingerprint'] = fingerprint(context)
        if not assembly.attachment_limits:
            return AttachmentLoadCertificate('PROVEN', 'no_declared_attachment_limits', context,
                boxes[0], {}, '0', 0)
        environment_model = None
        if assembly.environment is not None:
            from orbexa.simulation.nonlinear_environment_enclosure import NonlinearEnvironmentEnclosure
            try:
                environment_model = NonlinearEnvironmentEnclosure.from_environment(assembly.environment)
            except (_ProofUnresolved, ValueError) as error:
                return AttachmentLoadCertificate('UNAVAILABLE', str(error), context)
            context['interval_environment'] = environment_model.as_dict()
        try:
            import flint
            with _ARB_LOCK, flint.ctx.workprec(self.precision_bits):
                return self._bound(assembly, state, propulsion, commands, duration, boxes,
                    context, _ArbArithmetic(flint), STANDARD_GRAVITY, environment_model)
        except ImportError:
            return AttachmentLoadCertificate('UNAVAILABLE', 'arb_backend_unavailable', context)
        except (_ProofUnresolved, ArithmeticError, ValueError) as error:
            return AttachmentLoadCertificate('UNRESOLVED', str(error), context)

    def _bound(self, assembly, state, propulsion, commands, h, boxes, context, ar, g0, environment_model):
        z, one = ar.scalar(0), ar.scalar(1)

        def norm_upper(vector):
            box = ar.box(vector)
            squared = sum(max(abs(lo), abs(hi))**2 for lo, hi in zip(box.lower, box.upper))
            return ar.box((ar.scalar(squared).sqrt(),)).upper[0]

        def square(value):
            lo, hi = ar.box((value,)).lower[0], ar.box((value,)).upper[0]
            return ar.interval((0 if lo <= 0 <= hi else min(lo*lo, hi*hi), max(lo*lo, hi*hi)))

        def symmetric(radius):
            return tuple(ar.interval((-radius, radius)) for _ in range(3))

        n = len(assembly.elements)
        flow = [Fraction(0) for _ in range(n)]
        force = [tuple(z for _ in range(3)) for _ in range(n)]
        own_torque = [tuple(z for _ in range(3)) for _ in range(n)]
        positions = [tuple(ar.scalar(x) for x in element.position_body_m) for element in assembly.elements]
        for command, nozzle in zip(commands, propulsion.thrusters):
            index = nozzle.owner_index
            f = tuple(ar.scalar(command)*ar.scalar(x) for x in nozzle.direction_body)
            lever = _sub(tuple(ar.scalar(x) for x in nozzle.position_body_m), positions[index])
            force[index] = _add(force[index], f)
            own_torque[index] = _add(own_torque[index], _cross(lever, f))
            flow[index] += abs(Fraction(float(command)))/(Fraction(nozzle.specific_impulse_s)*Fraction(g0))
        fuel = []
        for initial, rate in zip(state.fuel_kg, flow):
            remaining = Fraction(float(initial))-h*rate
            if remaining < 0:
                raise _ProofUnresolved('held command cannot prove nonnegative tank mass for the full hold')
            fuel.append((remaining, Fraction(float(initial))))
        masses = tuple(ar.interval((Fraction(element.dry_mass_kg)+lo,
            Fraction(element.dry_mass_kg)+hi)) for element, (lo, hi) in zip(assembly.elements, fuel))
        mass = sum(masses)
        center = tuple(sum(m*p[j] for m, p in zip(masses, positions))/mass for j in range(3))
        offsets = tuple(_sub(p, center) for p in positions)
        inertias = tuple(tuple(tuple(m*ar.scalar(value) for value in row)
            for row in element.inertia_per_kg_m2) for m, element in zip(masses, assembly.elements))
        inertia = tuple(tuple(sum(local[i][j]+m*(
            sum(square(x[k]) for k in range(3) if k != i) if i == j else -x[i]*x[j])
            for m, x, local in zip(masses, offsets, inertias)) for j in range(3)) for i in range(3))
        inverse = _inverse(inertia)
        total_force = tuple(sum(f[j] for f in force) for j in range(3))
        torque = tuple(sum(t[j]+_cross(x, f)[j] for t, x, f in zip(own_torque, offsets, force)) for j in range(3))
        _, p0, v0 = map(ar.vector, boxes)
        r0sq = sum(square(x) for x in p0)
        radial_lower = ar.box((r0sq.sqrt(),)).lower[0]
        if radial_lower <= 0:
            raise _ProofUnresolved('initial assembly radius is not proved positive')
        radial_reserve = radial_lower/2
        offset_upper = max(norm_upper(x) for x in offsets)
        component_radial = radial_reserve-offset_upper
        if component_radial <= 0:
            raise _ProofUnresolved('component radial domain includes the gravity singularity')
        mu = Fraction(assembly.mu_m3_s2)
        gg = (3*mu/component_radial**3*norm_upper(tuple(x for row in inertia for x in row))
              if assembly.gravity_gradient else Fraction(0))
        gg_vector = symmetric(gg)

        diameter = max(norm_upper(_sub(a, b)) for a in positions for b in positions)
        minimum_mass = ar.box((mass,)).lower[0]
        shift = diameter*h*sum(flow)/minimum_mass
        extra_upper, absolute_acceleration = Fraction(0), None
        environment_parameters = None
        historical_spin_upper = norm_upper(ar.vector(boxes[0]))
        if environment_model is not None:
            expected = {element.name: Fraction(element.dry_mass_kg) for element in assembly.elements}
            if dict(environment_model.dry_mass) != expected:
                raise _ProofUnresolved('attachment environment mass/fleet differs from the admitted assembly')
            velocity_limit = max(Fraction(20000), 2*norm_upper(v0)+1000)
            minimum_by_name = {element.name: ar.box((m,)).lower[0] for element, m in zip(assembly.elements, masses)}
            extra_upper = max(environment_model.norm_upper_over_domain(velocity_limit, minimum_by_name, ar).values())
            absolute_acceleration = mu/radial_reserve**2+norm_upper(total_force)/minimum_mass+extra_upper
            orientation = quaternion_rotation(normalized_quaternion(tuple(ar.scalar(x)
                for x in state.quaternion_body_to_eci), ar))
            position_swept = _add(p0, symmetric(h*velocity_limit+shift))
            environment_parameters = environment_model.force_parameters(ar)
            context.update(environment_velocity_bootstrap_m_s=str(velocity_limit),
                environment_extra_acceleration_norm_upper_m_s2=str(extra_upper))

        def environmental_wrenches(w):
            if environment_model is None:
                return tuple(tuple(z for _ in range(3)) for _ in masses), tuple(z for _ in range(3)), tuple(z for _ in range(3))
            # The present angular trial alone cannot bound accumulated
            # attitude change or a prior COM-coordinate velocity jump when a
            # later substep is braking. Retain every already proved spin set.
            angular_upper = max(historical_spin_upper, norm_upper(w))
            if norm_upper(v0)+h*absolute_acceleration+angular_upper*(shift+offset_upper) >= velocity_limit:
                raise _ProofUnresolved('attachment environment velocity bootstrap is not contained')
            angle = min(Fraction(2), h*angular_upper)
            rotation = tuple(tuple(value+ar.interval((-angle, angle)) for value in row) for row in orientation)
            velocity_swept = _add(v0, symmetric(h*absolute_acceleration+angular_upper*shift))
            extra_forces = []
            for k, (element, offset, wet) in enumerate(zip(assembly.elements, offsets, masses)):
                position = _add(position_swept, _matvec(rotation, offset))
                velocity = _add(velocity_swept, _matvec(rotation, _cross(w, offset)))
                extra = environment_model.evaluate(element.name, position, velocity,
                    ar.interval(fuel[k]), ar, environment_parameters)
                extra_forces.append(_scale(wet, _matvec(transpose(rotation), extra)))
            net_force = tuple(sum(f[j] for f in extra_forces) for j in range(3))
            net_torque = tuple(sum(_cross(x, f)[j] for x, f in zip(offsets, extra_forces)) for j in range(3))
            return tuple(extra_forces), net_force, net_torque

        def acceleration(w):
            _, _, environment_torque = environmental_wrenches(w)
            return _matvec(inverse, _sub(_add(_add(torque, gg_vector), environment_torque), _cross(w, _matvec(inertia, w))))

        count = max(1, -(-h//Fraction(self.max_step_s)))
        if count > self.max_substeps:
            raise _ProofUnresolved('whole-hold angular substep count exceeds its finite budget')
        dt, initial, swept, iterations = h/count, boxes[0], boxes[0], 0
        for _ in range(count):
            w0 = ar.vector(initial)
            trial = IntervalBox.from_center_radius(tuple((lo+hi)/2 for lo, hi in zip(initial.lower, initial.upper)),
                tuple((hi-lo)/2+Fraction(1, 10**12) for lo, hi in zip(initial.lower, initial.upper)))
            for attempt in range(self.max_picard_iterations):
                iterations += 1
                field = acceleration(ar.vector(trial))
                image = ar.box(_add(w0, _scale(ar.interval((0, dt)), field)))
                # Strict inclusion supplies a first-exit/Picard enclosure for
                # all permitted fuel split times; spin is unchanged at jumps.
                if all(a < c and d < b for a, b, c, d in zip(trial.lower, trial.upper, image.lower, image.upper)):
                    initial = ar.box(_add(w0, _scale(ar.scalar(dt), field)))
                    swept = swept.hull(trial)
                    historical_spin_upper = max(historical_spin_upper, norm_upper(ar.vector(trial)))
                    break
                joined = trial.hull(image)
                selected = [not (a < c and d < b) for a, b, c, d in
                    zip(trial.lower, trial.upper, image.lower, image.upper)]
                trial = IntervalBox(tuple((lo+hi)/2-Fraction(3, 4)*(hi-lo)-Fraction(1, 10**12) if change else old
                    for lo, hi, old, change in zip(joined.lower, joined.upper, trial.lower, selected)),
                    tuple((lo+hi)/2+Fraction(3, 4)*(hi-lo)+Fraction(1, 10**12) if change else old
                    for lo, hi, old, change in zip(joined.lower, joined.upper, trial.upper, selected)))
            else:
                context['last_angular_trial'] = trial.as_dict()
                context['last_angular_picard_image'] = image.as_dict()
                raise _ProofUnresolved('whole-hold angular Picard inclusion exceeded its finite budget')
        w = ar.vector(swept)
        w_upper = norm_upper(w)
        # COM-coordinate changes total at most D*consumed/minimum mass.
        acceleration_upper = mu/radial_reserve**2+norm_upper(total_force)/minimum_mass+extra_upper
        radial_travel = h*(norm_upper(v0)+h*acceleration_upper+w_upper*shift)+shift
        if radial_travel >= radial_lower-radial_reserve:
            raise _ProofUnresolved('whole-hold orbital radial bootstrap is not contained')
        alpha = acceleration(w)
        extra_forces, extra_force, _ = environmental_wrenches(w)
        combined_force = _add(total_force, extra_force)
        members, maximum = {}, Fraction(0)
        index_by_name = {element.name: index for index, element in enumerate(assembly.elements)}
        for limit in assembly.attachment_limits:
            k = index_by_name[limit['name']]
            element, offset, wet = assembly.elements[k], offsets[k], masses[k]
            tidal = 2*mu/component_radial**3*norm_upper(offset)
            linear = _add(_add(_scale(one/mass, combined_force), _cross(alpha, offset)),
                _add(_cross(w, _cross(w, offset)), symmetric(tidal)))
            reaction = _sub(_scale(wet, linear), _add(force[k], extra_forces[k]))
            local_gg = (3*mu/component_radial**3*norm_upper(tuple(x for row in inertias[k] for x in row))
                if assembly.gravity_gradient else Fraction(0))
            moment_com = _sub(_add(_matvec(inertias[k], alpha), _cross(w, _matvec(inertias[k], w))),
                _add(own_torque[k], symmetric(local_gg)))
            arm = _sub(tuple(ar.scalar(x) for x in limit['surface_anchor_body_m']), positions[k])
            moment = _sub(moment_com, _cross(arm, reaction))
            normal = tuple(ar.scalar(x) for x in limit['normal_body'])
            fn = sum(x*y for x, y in zip(reaction, normal))
            normal_box = ar.box((fn,))
            shear = _sub(reaction, _scale(fn, normal))
            ratios = dict(tensile=max(Fraction(0), -normal_box.lower[0])/Fraction(limit['tensile_limit_n']),
                compression=max(Fraction(0), normal_box.upper[0])/Fraction(limit['compression_limit_n']),
                shear=norm_upper(shear)/Fraction(limit['shear_limit_n']),
                moment=norm_upper(moment)/Fraction(limit['moment_limit_nm']))
            maximum = max(maximum, *ratios.values())
            members[element.name] = dict(force_body_n=ar.box(reaction).as_dict(),
                moment_at_pad_body_nm=ar.box(moment).as_dict(),
                capacity_ratio_upper={name: str(value) for name, value in ratios.items()})
        context.update(fuel_swept_kg=[tuple(map(str, pair)) for pair in fuel],
            radial_lower_m=str(radial_reserve), orbital_travel_upper_m=str(radial_travel),
            center_shift_total_upper_m=str(shift), proved_substeps=count)
        fraction = Fraction(context['capacity_fraction'])
        return AttachmentLoadCertificate('PROVEN' if maximum <= fraction else 'UNRESOLVED',
            'whole_hold_loads_within_declared_capacities' if maximum <= fraction else 'whole_hold_load_upper_exceeds_capacity',
            context, swept, members, str(maximum), iterations)
