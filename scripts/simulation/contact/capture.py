"""Qualified approach admission followed by finite compliant physical capture.

Approach translation alone does not validate chaser attitude or finite bus
geometry. Both missing pieces require explicit caller assumptions. The
reference-point offset relocates a *declared reference*, never an existing
center of mass; using it with a COM trajectory would be invalid.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Any, Mapping, Optional, TYPE_CHECKING
import numpy as np
from scipy.spatial.transform import Rotation

from orbexa.core.dynamics.attitude import finite_vector
from orbexa.core.geometry.mounting import mounting_frame
from orbexa.simulation.attached.assembly.model import AttachedAssembly, AssemblyState, MassElement, finite_scalar, physical_inertia, positive
from orbexa.simulation.contact.simulator import ContactBodyState, ContactMember, FiniteContactOptions, FiniteContactSimulator
from orbexa.core.dynamics.anomaly import orbital_radius_terms

if TYPE_CHECKING:
    from orbexa.simulation.approach.terminal.handoff import PhysicalTerminalHandoff


@dataclass(frozen=True)
class TargetPhysicalProperties:
    """Representative rigid target values; no observed spacecraft identity."""
    mass_kg: float = 1000.0
    inertia_kg_m2: np.ndarray = field(default_factory=lambda: np.diag([190., 190., 320.]))

    def __post_init__(self):
        object.__setattr__(self, "mass_kg", positive(self.mass_kg, "Target mass"))
        matrix = physical_inertia(self.inertia_kg_m2)
        matrix.setflags(write=False)
        object.__setattr__(self, "inertia_kg_m2", matrix)


@dataclass(frozen=True)
class ChaserPhysicalProperties:
    dry_mass_kg: float = 8.0
    fuel_kg: float = 4.0
    body_radius_m: float = .15
    gripper_offset_m: float = .16

    def __post_init__(self):
        object.__setattr__(self, "dry_mass_kg", positive(self.dry_mass_kg, "Chaser dry mass"))
        object.__setattr__(self, "fuel_kg", positive(self.fuel_kg, "Chaser fuel", allow_zero=True))
        object.__setattr__(self, "body_radius_m", positive(self.body_radius_m, "Chaser spherical body radius"))
        object.__setattr__(self, "gripper_offset_m", positive(self.gripper_offset_m, "Gripper-to-COM distance", allow_zero=True))
        if self.gripper_offset_m < self.body_radius_m:
            raise ValueError("Gripper tip must reach outside the declared spherical bus envelope")

    @property
    def mass_kg(self):
        return self.dry_mass_kg + self.fuel_kg

    @property
    def inertia_per_kg_m2(self):
        return np.eye(3) * (.4 * self.body_radius_m**2)


@dataclass(frozen=True)
class CaptureOptions:
    arm_reach_m: float = .010
    max_relative_speed_m_s: float = .001
    max_orientation_error_rad: float = .01
    max_spin_error_rad_s: float = .0001
    assume_attitude_matched: bool = False
    assume_approach_point_is_capture_reference: bool = False
    minimum_bus_clearance_m: float = .001
    earth_radius_m: float = 6378137.0
    gravity_gradient: Optional[bool] = None
    max_integration_step_s: float = 1.0
    idealized_instantaneous: bool = False

    def __post_init__(self):
        for name in ("arm_reach_m", "max_relative_speed_m_s", "max_orientation_error_rad",
                     "max_spin_error_rad_s", "earth_radius_m", "max_integration_step_s"):
            object.__setattr__(self, name, positive(getattr(self, name), name))
        object.__setattr__(self, "minimum_bus_clearance_m", positive(
            self.minimum_bus_clearance_m, "Minimum bus clearance", allow_zero=True))
        if not isinstance(self.idealized_instantaneous, bool):
            raise ValueError('idealized_instantaneous must be an explicit boolean')


@dataclass(frozen=True)
class CaptureTruth:
    """Explicit physical boundary for callers supplying independently known truth.

    The fixed ECI basis may be chosen coincident with the initial LVLH basis;
    no terrestrial longitude or absolute epoch is inferred by the adapter.
    """
    anomaly: float
    time_s: float
    anomaly_rate_rad_s: float
    position_eci_m: np.ndarray
    velocity_eci_m_s: np.ndarray
    quaternion_body_to_eci: np.ndarray
    angular_velocity_body_rad_s: np.ndarray
    rotation_lvlh_to_eci: np.ndarray
    mu_m3_s2: float

    def __post_init__(self):
        for name in ("anomaly", "time_s"):
            object.__setattr__(self, name, finite_scalar(getattr(self, name), f"Capture truth {name}"))
        object.__setattr__(self, "anomaly_rate_rad_s", positive(self.anomaly_rate_rad_s, "Capture anomaly rate"))
        object.__setattr__(self, "mu_m3_s2", positive(self.mu_m3_s2, "Capture gravitational parameter"))
        for name, size in (("position_eci_m", 3), ("velocity_eci_m_s", 3),
                           ("quaternion_body_to_eci", 4), ("angular_velocity_body_rad_s", 3)):
            value = finite_vector(getattr(self, name), name, size)
            if name == "quaternion_body_to_eci":
                if np.linalg.norm(value) == 0:
                    raise ValueError("Capture truth quaternion cannot be zero")
                value /= np.linalg.norm(value)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        rotation = np.asarray(self.rotation_lvlh_to_eci, dtype=float)
        if (rotation.shape != (3, 3) or np.any(~np.isfinite(rotation))
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0)
                or not np.isclose(np.linalg.det(rotation), 1, atol=1e-10, rtol=0)):
            raise ValueError("Capture LVLH-to-ECI rotation must be proper orthogonal")
        rotation = rotation.copy()
        rotation.setflags(write=False)
        object.__setattr__(self, "rotation_lvlh_to_eci", rotation)


@dataclass(frozen=True)
class CaptureResult:
    success: bool
    message: str
    gates: dict
    audit: dict
    assumptions: tuple[str, ...]
    assembly: Optional[AttachedAssembly] = None
    state: Optional[AssemblyState] = None

    def to_dict(self):
        return dict(success=self.success, message=self.message, gates=self.gates, audit=self.audit,
                    assumptions=list(self.assumptions), assembly=None if self.assembly is None else self.assembly.to_dict(),
                    final_state=None if self.state is None else self.assembly.state_record(self.state))


class CaptureModel:
    def __init__(self, *, target: Optional[TargetPhysicalProperties] = None,
                 chasers: Optional[Mapping[str, ChaserPhysicalProperties]] = None,
                 default_chaser: Optional[ChaserPhysicalProperties] = None,
                 options: Optional[CaptureOptions] = None,
                 contact_options: Optional[FiniteContactOptions] = None,
                 contact_environment=None):
        self.target = target or TargetPhysicalProperties()
        self.chasers = dict(chasers or {})
        self.default_chaser = default_chaser or ChaserPhysicalProperties()
        self.options = options or CaptureOptions()
        self.contact_options = contact_options or FiniteContactOptions()
        self.contact_environment = contact_environment

    def recorded_truth(self, approach) -> CaptureTruth:
        metadata, target = approach.metadata, approach.target
        if target.get("attitude_model") not in ("torque_free_rigid_body", "gravity_gradient_rigid_body"):
            raise ValueError("Physical capture requires recorded rigid-body target kinematics or explicit CaptureTruth")
        if not np.allclose(np.asarray(target.get("inertia_kg_m2")), self.target.inertia_kg_m2, rtol=1e-10, atol=1e-10):
            raise ValueError("Capture target inertia conflicts with the physical approach target")
        kinematics = metadata.get("target_kinematics", {})
        anomalies = np.asarray(kinematics.get("anomaly", []), dtype=float)
        if (anomalies.shape != (len(approach.anom_history),) or not np.allclose(
                anomalies, approach.anom_history, rtol=0, atol=1e-12)):
            raise ValueError("Physical target kinematics must match every recorded approach sample")
        q = float(anomalies[-1])
        params = target.get("clock_parameters", {})
        required = ("eccentricity", "alpha", "mu", "semi_major_axis")
        if not all(name in params for name in required):
            raise ValueError("Physical capture requires explicit analytical-orbit truth parameters")
        if any(not np.isclose(float(params[name]), float(approach.truth[name]), rtol=1e-10, atol=1e-12)
               for name in required):
            raise ValueError("Recorded attitude clock conflicts with approach orbital truth")
        radius, prime, h = orbital_radius_terms(q, **{key: params[key] for key in required})
        rate = h * np.exp(-params["alpha"] * q) / radius**2
        if not np.isclose(rate, kinematics["anomaly_rate"][-1], rtol=1e-9, atol=1e-14):
            raise ValueError("Recorded physical anomaly rate conflicts with orbit truth")
        initial_q = float(target["anomaly_domain"][0])
        lvlh_to_eci = Rotation.from_rotvec([0, 0, q - initial_q]).as_matrix()
        quaternion = np.asarray(kinematics["inertial_quaternions"][-1], dtype=float)
        rotation = Rotation.from_quat(quaternion).as_matrix()
        if not np.allclose(rotation, lvlh_to_eci @ np.asarray(kinematics["rotations"][-1]), atol=1e-9, rtol=0):
            raise ValueError("Recorded physical target rotation frames are inconsistent")
        # Mission LVLH axes are x = minus transverse, y = outward radial.
        # Differentiating C(q) @ (0, r(q), 0) gives C(q) @ (-r, r', 0).
        # Keep this fixed initial-LVLH inertial basis identical to the target's
        # orbital position and gravity-gradient torque throughout the latch.
        return CaptureTruth(q, kinematics["physical_time"][-1], rate,
                            lvlh_to_eci @ np.array([0, radius, 0]),
                            lvlh_to_eci @ np.array([-radius * rate, prime * rate, 0]),
                            quaternion, kinematics["body_angular_velocity"][-1], lvlh_to_eci, params["mu"])

    def capture(self, approach, *, truth: Optional[CaptureTruth] = None,
                chaser_attitudes: Optional[Mapping[str, tuple[Any, Any]]] = None,
                terminal_handoff: Optional[PhysicalTerminalHandoff] = None,
                contact_feedback=None, contact_environment=None, contact_initial_enclosure=None) -> CaptureResult:
        """Latch the complete recorded fleet; incomplete gates return a result.

        Explicit chaser attitudes are (body-to-ECI quaternion, inertial body
        angular velocity in chaser components). Otherwise the caller must
        opt into perfect bus attitude/rate matching as an idealization.
        """
        assumptions = ["declared target mass/inertia and recorded physical initial states",
                       "rigid spherical chaser mass properties; no arm mass, flexure or fuel slosh"]
        if self.options.idealized_instantaneous:
            assumptions.extend(["REPRODUCIBILITY ONLY: perfectly inelastic instantaneous latch; unresolved contact impulse and latch strength",
                "REPRODUCIBILITY ONLY: a carried tip instantaneously closes the remaining gap; no finite deployment dynamics"])
        else:
            assumptions.extend(["finite coupled target and fleet dynamics with representative compliant adhesive pads",
                "finite-bandwidth massless carried positioners with explicit reaction work; target needs no docking fixture",
                "adhesive properties and hull compatibility are declared study assumptions, not hardware qualification"])
        gates, audit = {}, {}

        def refused(message):
            return CaptureResult(False, message, gates, audit, tuple(assumptions))

        meta = approach.metadata
        dwell = meta.get("terminal_dwell_updates", 0)
        qualification = bool(approach.success and meta.get("mission_complete") is True and dwell >= 1
            and meta.get("terminal_dwell_observed", 0) >= dwell and meta.get("sampled_tube_containment") is True
            and meta.get("active_target_margins_safe") is True and meta.get("pairwise_spacing_safe") is True
            and approach.sample_phase_history and approach.sample_phase_history[-1] == "docking")
        gates["qualified_approach"] = qualification
        if not qualification:
            return refused("approach_not_qualified_for_capture")
        if chaser_attitudes is None and not self.options.assume_attitude_matched:
            return refused("chaser_attitude_unrecorded: explicit matched-attitude assumption or physical attitude data required")
        if self.options.assume_approach_point_is_capture_reference:
            assumptions.append("EXPLICIT ALTERNATE REFERENCE: recorded translation is a capture-arm reference; bus COM is offset outward by configured gripper distance")
        else:
            assumptions.append("recorded approach position is the literal chaser COM; gripper offset is inward along the assigned surface normal")
        if chaser_attitudes is None:
            assumptions.append("IDEALIZATION: chaser bus attitude and inertial spin exactly match each assigned mounting frame; no attitude alignment controller was validated")
        try:
            if terminal_handoff is not None:
                from orbexa.simulation.approach.terminal.handoff import PhysicalTerminalHandoff
                if not isinstance(terminal_handoff, PhysicalTerminalHandoff):
                    return refused("capture_requires_verified_physical_terminal_handoff")
                if self.options.assume_approach_point_is_capture_reference:
                    return refused("terminal_handoff_is_a_literal_center_of_mass_record")
                terminal_handoff.verify_approach(approach)
                if truth is not None:
                    return refused("capture_truth_override_conflicts_with_verified_terminal_handoff")
                truth = terminal_handoff.capture_truth
                expected_anomaly = float(terminal_handoff.exact_end_anomaly)
                audit["terminal_handoff"] = terminal_handoff.as_dict()
                assumptions.append("Capture uses the literal final certified terminal state and its reference-centered enclosure; the original MPC record is preserved")
            else:
                truth = truth or self.recorded_truth(approach)
                expected_anomaly = approach.anom_history[-1]
            if not np.isclose(truth.anomaly, expected_anomaly, atol=1e-12, rtol=0):
                return refused("capture_truth_anomaly_does_not_match_approach")
            target_rotation = Rotation.from_quat(truth.quaternion_body_to_eci).as_matrix()
            composite = None
            if approach.target.get('shape') == 'composite_spacecraft':
                from orbexa.core.geometry.composite import SpacecraftGeometry
                composite = SpacecraftGeometry.from_dict(approach.target['geometry'])
                if composite.fingerprint != approach.target.get('geometry_fingerprint'):
                    return refused('target_geometry_fingerprint_mismatch')
                if not np.isclose(self.target.mass_kg,approach.target['mass_kg'],rtol=0,atol=1e-9):
                    return refused('capture_mass_conflicts_with_physical_target')
                if not np.allclose(self.target.inertia_kg_m2,approach.target['inertia_kg_m2'],rtol=1e-10,atol=1e-10):
                    return refused('capture_inertia_conflicts_with_physical_target')
                assumptions.append('Composite contact uses the nominal declared structural patch. Geometry uncertainty reserves free-flight clearance; it does not locate the actual hull, certify patch strength or guarantee contact on an unknown surface.')
            elif approach.target.get('shape') == 'cylinder':
                radius, half_length = float(approach.target['radius']), float(approach.target['half_length'])
                diagonal = np.diag(self.target.inertia_kg_m2)
                if (diagonal[2] > self.target.mass_kg * radius**2 + 1e-10
                        or max(diagonal[:2]) > self.target.mass_kg * (radius**2 + half_length**2) + 1e-10
                        or np.trace(self.target.inertia_kg_m2) > 2 * self.target.mass_kg * (radius**2 + half_length**2) + 1e-10):
                    return refused('target_mass_inertia_inconsistent_with_declared_cylinder_envelope')
            else:
                return refused('unsupported_target_geometry_at_capture')
            lvlh_rotation = truth.rotation_lvlh_to_eci
            body_to_lvlh = lvlh_rotation.T @ target_rotation
            target_spin_eci = target_rotation @ truth.angular_velocity_body_rad_s
            spin_lvlh = lvlh_rotation.T @ target_spin_eci
            assignments = {item["chaser_id"]: item for item in meta.get("chaser_assignments", [])}
            ids = list(approach.actual_trajectories)
            if set(assignments) != set(ids):
                return refused("capture_assignments_do_not_match_recorded_fleet")
            elements = [MassElement("target", self.target.mass_kg, 0,
                                     self.target.inertia_kg_m2 / self.target.mass_kg, np.zeros(3))]
            relative_velocities, intrinsic_momenta = [np.zeros(3)], [target_rotation @ self.target.inertia_kg_m2 @ truth.angular_velocity_body_rad_s]
            intrinsic_energies = [float(truth.angular_velocity_body_rad_s @ self.target.inertia_kg_m2 @ truth.angular_velocity_body_rad_s / 2)]
            bus_envelopes, surface_contacts, contact_members = {}, {}, []
            current_gain = finite_vector(meta["lambda_gain"] if terminal_handoff is None else
                                        terminal_handoff.lambda_gain, "Recorded tube gain", 3)
            for key in ids:
                properties = self.chasers.get(key, self.default_chaser)
                if terminal_handoff is not None:
                    certified_radius = Fraction(audit["terminal_handoff"]["controller_bounds"][key]["body_radius"])
                    enclosing_radius = Fraction(properties.body_radius_m) + (Fraction(properties.gripper_offset_m)
                        if self.options.assume_approach_point_is_capture_reference else 0.)
                    if enclosing_radius > certified_radius:
                        return refused(f"{key}: physical_bus_exceeds_certified_terminal_envelope")
                assignment = assignments[key]
                point = finite_vector(assignment["docking_point_body"], "Assigned docking point", 3)
                normal = finite_vector(assignment["docking_normal_body"], "Assigned outward normal", 3)
                if np.linalg.norm(normal) == 0:
                    return refused(f"{key}: invalid_surface_normal")
                normal /= np.linalg.norm(normal)
                standoff = positive(assignment["docking_standoff"], "Recorded standoff", allow_zero=True)
                surface = point - standoff * normal
                patch = None
                if composite is not None:
                    patch = composite.attachment_patch(assignment['attachment_patch_id'])
                    if (assignment.get('docking_surface') != 'attachment_patch'
                            or assignment.get('attachment_component_id') != patch.component_id
                            or not patch.contains(surface)
                            or not np.allclose(normal,patch.normal_body,atol=1e-10,rtol=0)):
                        return refused(f'{key}: anchor_is_not_on_declared_structural_patch')
                elif (assignment.get("docking_surface") != "cylinder_side"
                        or abs(np.linalg.norm(surface[:2]) - float(approach.target["radius"])) > 1e-8
                        or abs(surface[2]) > float(approach.target["half_length"]) + 1e-10
                        or not np.allclose(normal, np.r_[surface[:2] / np.linalg.norm(surface[:2]), 0.], atol=1e-9, rtol=0)):
                    return refused(f"{key}: anchor_is_not_on_declared_cylinder_surface")
                if terminal_handoff is None:
                    state = finite_vector(approach.actual_trajectories[key][-1], "Approach state", 6)
                    diagnostic = approach.tube_diagnostics[key]
                    nominal = finite_vector(diagnostic["nominal_center"][-1], "Current nominal center", 6)
                    phi = finite_vector(diagnostic["phi"][-1], "Current sliding tube", 3)
                    omega = finite_vector(diagnostic["omega"][-1], "Current position tube", 3)
                else:
                    state = finite_vector(terminal_handoff.states[key], "Final terminal actual state", 6)
                    nominal = finite_vector(terminal_handoff.nominal_centers[key], "Terminal reference state", 6)
                    phi = finite_vector(terminal_handoff.phi[key], "Final terminal sliding enclosure", 3)
                    omega = finite_vector(terminal_handoff.omega[key], "Final terminal position enclosure", 3)
                if np.any(phi < 0) or np.any(omega < 0):
                    return refused(f"{key}: invalid_current_tube")
                position_radius = float(np.linalg.norm(omega))
                velocity_radius = float(np.linalg.norm(phi + np.abs(current_gain) * omega))
                nominal_body = body_to_lvlh.T @ nominal[:3]
                actual_body = body_to_lvlh.T @ state[:3]
                relative_velocity = truth.anomaly_rate_rad_s * (state[3:] + np.cross([0, 0, 1.], state[:3])) - np.cross(spin_lvlh, state[:3])
                nominal_velocity = truth.anomaly_rate_rad_s * (nominal[3:] + np.cross([0, 0, 1.], nominal[:3])) - np.cross(spin_lvlh, nominal[:3])
                speed_bound = float(np.linalg.norm(nominal_velocity) + truth.anomaly_rate_rad_s * velocity_radius
                    + np.linalg.norm(truth.anomaly_rate_rad_s * np.array([0, 0, 1.]) - spin_lvlh) * position_radius)
                mount = mounting_frame(normal)
                expected = target_rotation @ mount
                if chaser_attitudes is None:
                    chaser_rotation, spin_eci = expected, target_spin_eci
                else:
                    attitude, spin = chaser_attitudes[key]
                    chaser_rotation = Rotation.from_quat(finite_vector(attitude, "Chaser attitude", 4)).as_matrix()
                    spin_eci = chaser_rotation @ finite_vector(spin, "Chaser body angular velocity", 3)
                orientation_error = float(Rotation.from_matrix(expected.T @ chaser_rotation).magnitude())
                spin_error = float(np.linalg.norm(spin_eci - target_spin_eci))
                gripper_to_com = target_rotation.T @ chaser_rotation @ np.array([properties.gripper_offset_m, 0., 0.])
                if self.options.assume_approach_point_is_capture_reference:
                    tip_offset, com_offset = np.zeros(3), gripper_to_com
                else:
                    tip_offset, com_offset = -gripper_to_com, np.zeros(3)
                actual_latch_length = float(np.linalg.norm(actual_body + tip_offset - surface))
                tube_latch_bound = float(np.linalg.norm(nominal_body + tip_offset - surface) + position_radius)
                # With supplied attitudes, tip velocity includes their measured
                # spin difference; the tube bound also reserves its worst gate value.
                relative_velocity += lvlh_rotation.T @ np.cross(spin_eci - target_spin_eci, target_rotation @ tip_offset)
                speed_bound += self.options.max_spin_error_rad_s * np.linalg.norm(tip_offset)
                actual_com_body, nominal_com_body = actual_body + com_offset, nominal_body + com_offset
                actual_bus_clearance = float(np.dot(actual_com_body - surface, normal) - properties.body_radius_m)
                # The position box is centered at the nominal state, even when
                # the contained actual endpoint lies on an outward-facing edge.
                bus_clearance = float(np.dot(nominal_com_body - surface, normal)
                                      - properties.body_radius_m - position_radius)
                patch_margin = None
                if composite is not None:
                    actual_bus_clearance = min(actual_bus_clearance,
                        composite.signed_distance(actual_com_body)-properties.body_radius_m)
                    bus_clearance = min(bus_clearance,composite.signed_distance(nominal_com_body)
                        -properties.body_radius_m-position_radius)
                    # A successful COM gate also reserves patch edge margin
                    # for every possible articulating-tip endpoint.
                    delta = nominal_body+tip_offset-np.asarray(patch.point_body_m)
                    patch_margin = min(patch.half_extents_m[0]-patch.margin_m
                        -abs(delta@patch.tangent_u_body)-position_radius,
                        patch.half_extents_m[1]-patch.margin_m
                        -abs(delta@patch.tangent_v_body)-position_radius)
                    # The straight segment must remain separated from every
                    # other occupied appendage. Distance to a convex solid
                    # along this short segment is bounded by its midpoint
                    # distance minus half length (1-Lipschitz).
                    tip_start = nominal_body+tip_offset
                    nominal_segment_length = float(np.linalg.norm(tip_start-surface))
                    for component in composite.components:
                        if component.id == patch.component_id:
                            continue
                        midpoint = (tip_start+surface)/2
                        if component.signed_distance(midpoint)-nominal_segment_length/2-position_radius <= 0:
                            return refused(f'{key}: contact_segment_not_clear_of_{component.id}')
                tube_membership = float(min(np.min(omega - np.abs(state[:3] - nominal[:3])),
                    np.min(phi - np.abs(state[3:] - nominal[3:] + current_gain * (state[:3] - nominal[:3])))))
                tip_gap_lower = float(np.dot(nominal_body + tip_offset - surface, normal) - position_radius)
                gate = dict(actual_latch_length_m=actual_latch_length, whole_tube_latch_length_bound_m=tube_latch_bound,
                    arm_reach_m=self.options.arm_reach_m, actual_relative_speed_m_s=float(np.linalg.norm(relative_velocity)),
                    whole_tube_relative_speed_bound_m_s=speed_bound, max_relative_speed_m_s=self.options.max_relative_speed_m_s,
                    orientation_error_rad=orientation_error, spin_error_rad_s=spin_error,
                    current_tube_membership_margin=tube_membership,
                    whole_tube_tip_surface_gap_lower_bound_m=tip_gap_lower,
                    actual_bus_surface_clearance_lower_bound_m=actual_bus_clearance,
                    whole_tube_bus_surface_clearance_lower_bound_m=bus_clearance,
                    conservative_bus_surface_clearance_m=bus_clearance, capture_reference_body_m=actual_body.tolist(),
                    actual_center_of_mass_body_m=actual_com_body.tolist(),
                    nominal_center_of_mass_body_m=nominal_com_body.tolist(),
                    gripper_tip_body_m=(actual_body + tip_offset).tolist(),
                    chaser_rotation_in_assembly=(target_rotation.T @ chaser_rotation).tolist(),
                    chaser_body_radius_m=properties.body_radius_m,
                    surface_anchor_body_m=surface.tolist(), approach_position_semantics=(
                        "capture_reference_explicit_opt_in" if self.options.assume_approach_point_is_capture_reference else "chaser_center_of_mass"))
                gate["passed"] = bool(actual_latch_length <= self.options.arm_reach_m and tube_latch_bound <= self.options.arm_reach_m
                    and np.linalg.norm(relative_velocity) <= self.options.max_relative_speed_m_s
                    and speed_bound <= self.options.max_relative_speed_m_s
                    and orientation_error <= self.options.max_orientation_error_rad
                    and spin_error <= self.options.max_spin_error_rad_s
                    and tube_membership >= -1e-10 and tip_gap_lower >= 0
                    and actual_bus_clearance >= self.options.minimum_bus_clearance_m
                    and bus_clearance >= self.options.minimum_bus_clearance_m
                    and (patch_margin is None or patch_margin >= 0))
                if composite is not None:
                    gate.update(attachment_patch_id=patch.id,geometry_fingerprint=composite.fingerprint,
                        whole_tube_patch_edge_margin_m=float(patch_margin),
                        whole_tube_all_component_clearance_lower_bound_m=bus_clearance)
                gates[key] = gate
                if not gate["passed"]:
                    return refused(f"{key}: capture_envelope_or_pose_gate_failed")
                velocity_relative = (lvlh_rotation @ (truth.anomaly_rate_rad_s * (state[3:] + np.cross([0, 0, 1.], state[:3])))
                                     + np.cross(spin_eci, target_rotation @ com_offset))
                if not self.options.idealized_instantaneous:
                    tangent_u, tangent_v = (np.asarray(patch.tangent_u_body), np.asarray(patch.tangent_v_body)) if patch is not None else (mount[:, 1], mount[:, 2])
                    # mounting_frame's columns are (normal, tangent, binormal),
                    # so the tangents followed by normal remain right handed.
                    contact_members.append(ContactMember(key, properties, ContactBodyState(
                        truth.time_s, truth.position_eci_m+target_rotation@actual_com_body,
                        truth.velocity_eci_m_s+velocity_relative, Rotation.from_matrix(chaser_rotation).as_quat(),
                        chaser_rotation.T@spin_eci, properties.fuel_kg), surface, normal, tangent_u, tangent_v,
                        self.options.arm_reach_m, None if patch is None else patch.id))
                tip_before = actual_body + tip_offset
                correction = surface - tip_before
                surface_contacts[key] = dict(
                    event="surface_contact_and_rigid_attachment", time_s=truth.time_s,
                    anomaly=truth.anomaly, target_docking_fixture_required=False,
                    surface_point_body_m=surface.tolist(), tip_before_contact_body_m=tip_before.tolist(),
                    tip_at_contact_body_m=surface.tolist(), tip_correction_body_m=correction.tolist(),
                    correction_length_m=actual_latch_length, whole_tube_reach_bound_m=tube_latch_bound,
                    available_tip_reach_m=self.options.arm_reach_m,
                    initial_outward_gap_m=float(np.dot(tip_before - surface, normal)),
                    whole_tube_initial_outward_gap_lower_bound_m=tip_gap_lower,
                    contact_residual_m=float(np.linalg.norm(tip_before + correction - surface)),
                    path_model="Straight articulating-tip segment to the selected hull point; supporting-plane gap remains nonnegative",
                    time_model="Contact and inelastic attachment are coincident idealized events; no finite arm deployment dynamics")
                offset_eci = target_rotation @ com_offset
                com_body = actual_com_body
                bus_envelopes[key] = (actual_com_body, nominal_com_body, properties.body_radius_m, position_radius)
                # The tip and COM are two points on the same pre-latch rigid bus.
                velocity_relative = (lvlh_rotation @ (truth.anomaly_rate_rad_s * (state[3:] + np.cross([0, 0, 1.], state[:3])))
                                     + np.cross(spin_eci, offset_eci))
                inertia_eci = chaser_rotation @ (properties.mass_kg * properties.inertia_per_kg_m2) @ chaser_rotation.T
                inertia_body_per_kg = target_rotation.T @ inertia_eci @ target_rotation / properties.mass_kg
                elements.append(MassElement(key, properties.dry_mass_kg, properties.fuel_kg, inertia_body_per_kg, com_body,
                                             target_rotation.T @ chaser_rotation))
                relative_velocities.append(velocity_relative)
                intrinsic_momenta.append(inertia_eci @ spin_eci)
                intrinsic_energies.append(float(spin_eci @ inertia_eci @ spin_eci / 2))
            actual_pair_margins = {f"{left}|{right}": float(np.linalg.norm(bus_envelopes[left][0] - bus_envelopes[right][0])
                                   - bus_envelopes[left][2] - bus_envelopes[right][2])
                                   for index, left in enumerate(ids) for right in ids[index+1:]}
            pair_margins = {f"{left}|{right}": float(np.linalg.norm(bus_envelopes[left][1] - bus_envelopes[right][1])
                            - sum(bus_envelopes[left][2:]) - sum(bus_envelopes[right][2:]))
                            for index, left in enumerate(ids) for right in ids[index+1:]}
            gates["actual_bus_pair_clearance_m"] = actual_pair_margins
            gates["whole_tube_bus_pair_clearance_lower_bound_m"] = pair_margins
            gates["finite_bus_pair_clearance_m"] = pair_margins
            if pair_margins and min(min(pair_margins.values()), min(actual_pair_margins.values())) < 0:
                return refused("finite_chaser_body_envelopes_overlap_at_capture")
            gravity_gradient = (bool(approach.target.get("gravity_gradient", False)) if self.options.gravity_gradient is None
                                else self.options.gravity_gradient)
            if not self.options.idealized_instantaneous:
                environment = self.contact_environment if contact_environment is None else contact_environment
                contact = FiniteContactSimulator(replace(self.contact_options,
                    minimum_bus_clearance_m=max(self.options.minimum_bus_clearance_m,
                                               self.contact_options.minimum_bus_clearance_m))).run(
                    target_properties=self.target, target_state=ContactBodyState(truth.time_s,
                        truth.position_eci_m, truth.velocity_eci_m_s, truth.quaternion_body_to_eci,
                        truth.angular_velocity_body_rad_s), members=contact_members,
                    geometry=composite if composite is not None else dict(radius=radius, half_length=half_length),
                    mu_m3_s2=truth.mu_m3_s2, gravity_gradient=gravity_gradient, feedback=contact_feedback,
                    environment=environment, initial_enclosure=contact_initial_enclosure)
                audit['finite_contact'] = contact.to_dict()
                audit['surface_contacts'] = dict(contact.surface_contacts)
                audit['finite_contact_handoff_eligible'] = bool(contact.success)
                audit['engaged_pad_count'] = sum(value.get('engagement_start_time_s') is not None
                    for value in contact.surface_contacts.values())
                audit['attached_chaser_count'] = 0
                audit['rigid_attachment_completed'] = False
                if not contact.success:
                    return refused(contact.message)
                return self._finite_handoff(contact, contact_members, truth, gravity_gradient, gates, audit, assumptions,
                                            environment, composite)
            assembly = AttachedAssembly(tuple(elements), truth.mu_m3_s2, self.options.earth_radius_m, gravity_gradient,
                                        self.options.max_integration_step_s)
            fuel = assembly.initial_fuel_kg
            mass, center, inertia = assembly.mass_properties(fuel)
            masses = np.array([element.initial_mass_kg for element in elements])
            velocities = np.asarray(relative_velocities)
            com_velocity_relative = masses @ velocities / mass
            offsets_eci = np.array([target_rotation @ (element.position_body_m - center) for element in elements])
            relative_com_velocities = velocities - com_velocity_relative
            momentum = np.sum(masses[:, None] * relative_com_velocities, axis=0)
            angular = np.sum(intrinsic_momenta, axis=0) + np.sum(np.cross(offsets_eci, masses[:, None] * relative_com_velocities), axis=0)
            final_spin = np.linalg.solve(inertia, target_rotation.T @ angular)
            before_energy = float(np.sum(intrinsic_energies) + np.sum(masses * np.sum(relative_com_velocities**2, axis=1)) / 2)
            after_energy = float(final_spin @ inertia @ final_spin / 2)
            state = AssemblyState(truth.time_s, truth.position_eci_m + target_rotation @ center,
                                  truth.velocity_eci_m_s + com_velocity_relative, truth.quaternion_body_to_eci, final_spin, fuel)
            audit.update(linear_momentum_residual_kg_m_s=float(np.linalg.norm(momentum)),
                         angular_momentum_residual_kg_m2_s=float(np.linalg.norm(assembly.angular_momentum_eci(state) - angular)),
                         pre_latch_angular_momentum_eci_kg_m2_s=angular.tolist(),
                         pre_latch_com_frame_kinetic_energy_j=before_energy,
                         post_latch_rotational_energy_j=after_energy,
                         dissipated_kinetic_energy_j=before_energy - after_energy,
                         target_to_assembly_com_shift_body_m=center.tolist(),
                         capture_anomaly=truth.anomaly, capture_time_s=truth.time_s,
                         approach_qualification_provenance=meta.get("qualification_provenance", "recorded ADTMPC mission result"),
                         target_position_eci_m=truth.position_eci_m.tolist(), target_velocity_eci_m_s=truth.velocity_eci_m_s.tolist())
            if before_energy - after_energy < -1e-10:
                return refused("capture_energy_audit_failed")
            audit.update(surface_contacts=surface_contacts, attached_chaser_count=len(surface_contacts),
                         target_interface="Selected bare-hull surface points; no target docking hardware")
            return CaptureResult(True, "rigid_latch_completed", gates, audit, tuple(assumptions), assembly, state)
        except (KeyError, TypeError, ValueError, IndexError) as error:
            return refused(f"capture_boundary_rejected: {error}")

    def _finite_handoff(self, contact, members, truth, gravity_gradient, gates, audit, assumptions,
                        environment=None, geometry=None):
        """Brake relative motion over finite time, then change coordinates only."""
        from orbexa.control.certification.reachability import IntervalBox
        from orbexa.simulation.contact.dynamics.state import ContactFleet
        from orbexa.simulation.contact.lock.transition import FiniteLockTransition
        from orbexa.simulation.attached.assembly.flow import AssemblyStateEnclosure, assembly_parameter_fingerprint, packed_state
        current_members = tuple(replace(member, initial_state=contact.final_chasers[member.id]) for member in members)
        fleet = ContactFleet.create(target_properties=self.target, target_state=contact.final_target,
            members=current_members, mu_m3_s2=truth.mu_m3_s2, gravity_gradient=gravity_gradient)
        lock = FiniteLockTransition().run(fleet=fleet, options=self.contact_options,
            geometry=geometry, contact=contact, environment=environment)
        audit['finite_lock'] = lock.audit
        if not lock.success:
            return CaptureResult(False, lock.message, gates, audit, tuple(assumptions))
        numeric = np.asarray(lock.numerical)
        elements, limits = [], []
        for k, name in enumerate(('target',)+tuple(member.id for member in members)):
            position_box = lock.numerical_parameters['position_body_m'][k]
            matrix_box = lock.numerical_parameters['rotation_element_to_assembly'][k]
            position = np.array([float((lo+hi)/2) for lo, hi in zip(position_box.lower, position_box.upper)])
            matrix = np.array([float((lo+hi)/2) for lo, hi in zip(matrix_box.lower, matrix_box.upper)]).reshape(3, 3)
            local_rotation = Rotation.from_matrix(matrix).as_matrix()
            if k == 0:
                dry, specific, position, local_rotation = self.target.mass_kg, self.target.inertia_kg_m2/self.target.mass_kg, np.zeros(3), np.eye(3)
            else:
                member = members[k-1]
                dry = member.properties.dry_mass_kg
                specific = local_rotation@member.properties.inertia_per_kg_m2@local_rotation.T
                anchor_box = lock.numerical_parameters['attachment_anchor_body_m'][k]
                anchor = [float((lo+hi)/2) for lo, hi in zip(anchor_box.lower, anchor_box.upper)]
                limits.append(dict(name=name, normal_body=member.normal_body.tolist(),
                    surface_anchor_body_m=anchor,
                    tensile_limit_n=self.contact_options.tensile_limit_n, shear_limit_n=self.contact_options.shear_limit_n,
                    moment_limit_nm=self.contact_options.moment_limit_nm,
                    compression_limit_n=self.contact_options.positioner_force_limit_n))
            elements.append(MassElement(name, dry, numeric[13+k], specific, position, local_rotation))
        anchors = {value['name']: value['surface_anchor_body_m'] for value in limits}
        parameters = {key: tuple(box.hull(IntervalBox.point(np.asarray(getattr(element, key) if key != 'attachment_anchor_body_m'
            else anchors.get(element.name, (0., 0., 0.))).reshape(-1)))
            for box, element in zip(value, elements)) for key, value in lock.parameters.items()}
        assembly = AttachedAssembly(tuple(elements), truth.mu_m3_s2, self.options.earth_radius_m,
            gravity_gradient, self.options.max_integration_step_s, attachment_limits=tuple(limits),
            environment=environment, parameter_enclosure=parameters)
        state = AssemblyState(contact.final_target.time_s+lock.elapsed_s, numeric[:3], numeric[3:6],
            numeric[6:10], numeric[10:13], numeric[13:])
        continuous = contact.audit['continuous_enclosure']
        exact_time = Fraction(truth.time_s)+Fraction(continuous['exact_accepted_duration_s'])+Fraction(self.contact_options.lock_duration_s)
        carried = AssemblyStateEnclosure(lock.state.hull(IntervalBox.point(packed_state(state))), exact_time,
            assembly_parameter_fingerprint(assembly), dict(kind='carried_contact_through_finite_load_bounded_lock',
                contact_proof_chain_sha256=continuous['proof_chain_sha256'],
                finite_lock_proof_chain_sha256=lock.audit['proof_chain_sha256'],
                terminal_boundary_transfer=continuous.get('boundary_transfer', {}).get('identity'),
                retained_locked_pad_energy_j=contact.audit['final_contact_potential_j']))
        carried = replace(carried, source=dict(carried.source,
            retained_locked_pad_energy_enclosure_j=lock.audit['retained_locked_pad_energy_enclosure_j'],
            trapped_pad_inventory_identity=lock.audit['trapped_pad_inventory_identity']))
        state = replace(state, enclosure=carried)
        reconstructed = assembly.component_states(state)
        target = reconstructed['target']
        for value in audit.get('surface_contacts', {}).values():
            value['latch_time_s'] = state.time_s
        audit.update(capture_time_s=state.time_s, capture_admission_anomaly=truth.anomaly, capture_anomaly=None,
            attached_chaser_count=len(members), rigid_attachment_completed=True,
            target_position_eci_m=target['position_eci_m'], target_velocity_eci_m_s=target['velocity_eci_m_s'],
            target_to_assembly_com_shift_body_m=assembly.mass_properties(state.fuel_kg)[1].tolist(),
            stored_pad_energy_dissipated_at_lock_j=0.,
            stored_pad_energy_retained_in_locked_structure_j=contact.audit['final_contact_potential_j'],
            stored_pad_energy_retained_enclosure_j=lock.audit['retained_locked_pad_energy_enclosure_j'],
            trapped_pad_inventory=lock.audit['trapped_pad_inventory'],
            active_lock_absorbed_mechanical_work_j=lock.audit['internal_absorbed_mechanical_work_j'],
            active_lock_work_scope='Signed mechanical work absorbed by the declared bidirectional active joint; no irreversible-dissipation or thermal-storage qualification',
            com_frame_kinetic_energy_decrease_j=(lock.audit['initial_com_frame_kinetic_energy_j']-
                lock.audit['final_com_frame_kinetic_energy_j']),
            pre_latch_com_frame_kinetic_energy_j=lock.audit['initial_com_frame_kinetic_energy_j'],
            post_latch_rotational_energy_j=lock.audit['final_com_frame_kinetic_energy_j'],
            lock_projection=dict(method='finite_internal_braking_then_coordinate_change',
                residual_linear_impulse_kg_m_s=0., residual_angular_impulse_kg_m2_s=0.),
            carried_assembly_boundary=carried.as_dict(),
            target_interface='Finite compliant carried adhesive pads and finite load-bounded active positioner lock on bare-hull patches')
        assumptions.append('Finite active positioners impose the declared internal braking profile under whole-step force, moment, power, travel and geometry gates; locked pad strain energy is retained')
        return CaptureResult(True, 'finite_contact_and_load_bounded_lock_completed', gates, audit, tuple(assumptions), assembly, state)
