"""Bounded chaser-carried navigation for the captured rigid assembly.

The target supplies no cooperative receiver. Gyros and star trackers on the
chasers observe one common rigid twist through uncertain captured mountings;
their deterministic error balls are intersected, without averaging a claimed
noise variance. GNSS products retain their full common-bias support.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
import copy
import hashlib

import numpy as np
from scipy.spatial.transform import Rotation

from orbexa.control.reachability import IntervalBox, _ARB_LOCK, _ArbArithmetic
from orbexa.core.reference_frame import fingerprint
from orbexa.simulation.assembly_enclosure import _AssemblyField
from orbexa.simulation.contact_boundary import _intersection, _quaternion_box
from orbexa.simulation.rigid_interval import matvec, quaternion_rotation, subtract, cross, norm_upper


@dataclass(frozen=True)
class AssemblySensorBounds:
    gyro_error_norm_rad_s: float = .00001
    star_tracker_error_rad: float = .0003
    gnss_position_error_m: float = 1.
    gnss_velocity_error_m_s: float = .003

    def __post_init__(self):
        if any(isinstance(value, bool) or not np.isfinite(value) or value <= 0 for value in vars(self).values()):
            raise ValueError('Declare positive finite attached-sensor supports')

    def as_dict(self):
        return dict(method='chaser-carried-rigid-assembly-sensor-support-v1', **vars(self))


class BoundedAssemblyObserver:
    """Synthetic packet producer and a separate bounded-set consumer."""
    def __init__(self, support=None):
        self.support = support or AssemblySensorBounds()
        if not isinstance(self.support, AssemblySensorBounds):
            raise ValueError('Assembly observer requires typed deterministic sensor supports')
        self.calls = 0

    def acquire(self, assembly, state):
        physical = assembly.component_states(state)
        names = tuple(element.name for element in assembly.elements if element.name != 'target')
        if not names:
            raise ValueError('The uncooperative target cannot supply the assembly sensor products')
        measurements = {}
        elements = {element.name: element for element in assembly.elements}
        for k, name in enumerate(names):
            body = physical[name]
            # A declared deterministic calibration realization. Other biases
            # within the same support can be supplied by an injected producer.
            z = 1.-2*(k+.5)/len(names)
            angle = np.pi*(3.-np.sqrt(5.))*k
            direction = np.array((np.sqrt(1-z*z)*np.cos(angle), np.sqrt(1-z*z)*np.sin(angle), z))
            gyro = elements[name].rotation_element_to_assembly.T@state.angular_velocity_body_rad_s+.95*self.support.gyro_error_norm_rad_s*direction
            star = Rotation.from_quat(body['quaternion_body_to_eci'])*Rotation.from_rotvec(
                .8*self.support.star_tracker_error_rad*direction)
            measurements[name] = dict(angular_velocity_body_rad_s=gyro.tolist(),
                quaternion_body_to_eci=star.as_quat().tolist(),
                position_eci_m=(np.asarray(body['position_eci_m'])+np.array((.35, -.4, .2))).tolist(),
                velocity_eci_m_s=(np.asarray(body['velocity_eci_m_s'])+np.array((.001, -.001, 0.))).tolist())
        return dict(method='synthetic-chaser-carried-assembly-packets-v1', time_s=state.time_s,
            measurements=measurements, support=self.support.as_dict(),
            common_gnss_bias_semantics='shared deterministic bias; no independence or variance-reduction assumption')

    def update(self, assembly, state, *, packet=None):
        """Intersect observations with the prior; retain the physical plant point."""
        if state.enclosure is None:
            raise ValueError('Bounded attached observation requires a carried physical state set')
        packet = self.acquire(assembly, state) if packet is None else copy.deepcopy(packet)
        if (packet.get('time_s') != state.time_s or packet.get('support') != self.support.as_dict() or
                set(packet.get('measurements', {})) != {element.name for element in assembly.elements if element.name != 'target'}):
            raise ValueError('Attached observations must name the exact current noncooperative-target fleet and support')
        prior = state.enclosure.box
        lower, upper = list(prior.lower), list(prior.upper)
        def narrow(start, box):
            result = _intersection(IntervalBox(lower[start:start+box.dimension], upper[start:start+box.dimension]), box)
            lower[start:start+box.dimension], upper[start:start+box.dimension] = result.lower, result.upper
        balls, attitude_packets = [], []
        import flint
        with _ARB_LOCK, flint.ctx.workprec(128):
            ar = _ArbArithmetic(flint)
            for k, element in enumerate(assembly.elements):
                if element.name == 'target':
                    continue
                data = packet['measurements'][element.name]
                nominal = element.rotation_element_to_assembly
                mount_box = (IntervalBox.point(nominal.flat) if assembly.parameter_enclosure is None else
                             assembly.parameter_enclosure['rotation_element_to_assembly'][k])
                difference = IntervalBox(tuple(lo-Fraction(float(x)) for lo, x in zip(mount_box.lower, nominal.flat)),
                    tuple(hi-Fraction(float(x)) for hi, x in zip(mount_box.upper, nominal.flat)))
                mount_error = norm_upper(ar.vector(difference), ar)
                measured = tuple(map(float, data['angular_velocity_body_rad_s']))
                exact_center = ar.box(matvec(tuple(tuple(ar.scalar(x) for x in row) for row in nominal),
                                            tuple(map(ar.scalar, measured))))
                center = tuple(float((lo+hi)/2) for lo, hi in zip(exact_center.lower, exact_center.upper))
                rounding = norm_upper(ar.vector(IntervalBox(tuple(lo-Fraction(x) for lo, x in zip(exact_center.lower, center)),
                    tuple(hi-Fraction(x) for hi, x in zip(exact_center.upper, center)))), ar)
                radius = Fraction(self.support.gyro_error_norm_rad_s)+mount_error*norm_upper(tuple(map(ar.scalar, measured)), ar)+rounding
                balls.append((tuple(map(Fraction, center)), radius, element.name))
                observed = Rotation.from_quat(data['quaternion_body_to_eci']).as_matrix()@nominal.T
                quaternion = _quaternion_box(observed, Fraction(self.support.star_tracker_error_rad)+mount_error,
                                             state.quaternion_body_to_eci, ar)
                narrow(6, quaternion)
                attitude_packets.append(dict(member_id=element.name, mounting_operator_error_upper=str(mount_error)))
            # A norm-ball constraint gives a stronger bound than a cube,
            # while retaining every deterministic sensor error realization.
            for _ in range(8):
                for center, radius, _ in balls:
                    for axis in range(3):
                        remaining = radius*radius
                        for j in range(3):
                            if j == axis:
                                continue
                            lo, hi = lower[10+j]-center[j], upper[10+j]-center[j]
                            remaining -= 0 if lo <= 0 <= hi else min(lo*lo, hi*hi)
                        if remaining < 0:
                            raise ValueError('Attached gyro ball contradicts the retained rigid-rate set')
                        reach = ar.box((ar.scalar(remaining).sqrt(),)).upper[0]
                        narrow(10+axis, IntervalBox((center[axis]-reach,), (center[axis]+reach,)))
            # Receiver positions/velocities refer to actual chasers. The
            # uncertain mounting, COM and attitude are retained when inferring
            # the virtual assembly COM product.
            dummy = type('NoNozzles', (), {'thrusters': ()})()
            shape = _AssemblyField(assembly, dummy, (), ar)
            fuel = ar.vector(IntervalBox(lower[13:], upper[13:]))
            _, _, _, offsets, _ = shape.mass_properties(fuel)
            rotation = quaternion_rotation(ar.vector(IntervalBox(lower[6:10], upper[6:10])))
            spin = ar.vector(IntervalBox(lower[10:13], upper[10:13]))
            for k, element in enumerate(assembly.elements):
                if element.name == 'target':
                    continue
                data = packet['measurements'][element.name]
                absolute = IntervalBox.from_center_radius(data['position_eci_m'], (self.support.gnss_position_error_m,)*3)
                velocity = IntervalBox.from_center_radius(data['velocity_eci_m_s'], (self.support.gnss_velocity_error_m_s,)*3)
                narrow(0, ar.box(subtract(ar.vector(absolute), matvec(rotation, offsets[k]))))
                narrow(3, ar.box(subtract(ar.vector(velocity), matvec(rotation, cross(spin, offsets[k])))))
        posterior = IntervalBox(lower, upper)
        from orbexa.simulation.assembly_enclosure import packed_state
        if not posterior.contains(IntervalBox.point(packed_state(state))):
            raise ValueError('Attached measurement support excludes the executed numerical representative')
        declaration = replace(state.enclosure, box=posterior)
        physical_state = replace(state, enclosure=declaration)
        midpoint = lambda start, end: np.array([float((lo+hi)/2) for lo, hi in zip(posterior.lower[start:end], posterior.upper[start:end])])
        from orbexa.simulation.assembly import AssemblyState
        feedback = AssemblyState(state.time_s, midpoint(0, 3), midpoint(3, 6), midpoint(6, 10), midpoint(10, 13), state.fuel_kg)
        record = dict(method='bounded-common-rigid-rate-and-fleet-gnss-intersection-v1',
            exact_time_s=str(state.enclosure.exact_time_s), packet=packet,
            previous_proof_chain_sha256=state.enclosure.proof_chain_sha256,
            prior_fingerprint=fingerprint(prior.as_dict()), posterior=posterior.as_dict(),
            gyro_balls=[dict(member_id=name, center=tuple(map(str, center)), radius=str(radius)) for center, radius, name in balls],
            attitude_mount_supports=attitude_packets,
            controller_uses='posterior center; physical plant state is retained separately')
        record['identity'] = fingerprint(record)
        chain = hashlib.sha256(((state.enclosure.proof_chain_sha256 or '')+record['identity']).encode('ascii')).hexdigest()
        physical_state = replace(physical_state, enclosure=replace(declaration, proof_chain_sha256=chain))
        self.calls += 1
        return physical_state, feedback, record

    def as_dict(self):
        return dict(method='bounded-chaser-carried-rigid-assembly-observer-v1', calls=self.calls,
            support=self.support.as_dict(), synthetic_bias_realization='0.95 support-norm gyro bias and0.8support-angle star bias on fixed Fibonacci directions; shared GNSS bias',
            scope='Synthetic processed chaser sensor products and uncertain rigid-mount fusion; no target receiver, hardware calibration or universal identification claim')
