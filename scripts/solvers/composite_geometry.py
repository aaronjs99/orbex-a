"""Nonlinear support-plane exclusion of every declared spacecraft component.

For each OBB and prediction node a free unit normal separates the complete
position box plus spherical bus from that component. The normals are decision
variables, so the union exclusion remains nonconvex and uses the IPOPT path.
No enclosing cylinder/sphere, binary variable or complementarity max is used.
"""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Integral
import numpy as np

from orbexa.core.spacecraft_geometry import OBBComponent

_OBB_NOMINAL_VERTICES = OBBComponent.nominal_vertices_body_m
_OBB_VERTICES = OBBComponent._vertices


class GekkoCompositeGeometry:
    """Attach exact OBB/tube support inequalities to a dynamic-tube backend.

    Component uncertainty is an isotropic displacement ball and is added once.
    Optional target rotation/translation reserves are explicit declarations.
    Numerical node verification is separate from a continuous-flow admission.
    """

    def __init__(self, target, rotations, center):
        from orbexa.core.spacecraft_geometry import SpacecraftGeometry
        payload = target.get("geometry")
        if payload is None:
            raise ValueError("Composite target planning requires its complete explicit spacecraft geometry")
        self.geometry = payload if isinstance(payload, SpacecraftGeometry) else SpacecraftGeometry.from_dict(payload)
        self.rotations = np.asarray(rotations, dtype=float)
        self.center = np.asarray(center, dtype=float)
        self.reserve = self._nonnegative(target.get("safety_reserve", 0.), "Composite finite-body/clearance reserve")
        self.rotation_error = self._nonnegative(target.get("orientation_matrix_error", 0.), "Composite rotation error")
        self.center_error = self._nonnegative(target.get("center_radius_m", 0.), "Composite center error")
        self.corridor_patch = None
        if target.get("robust_attachment_corridor", False):
            patch_id, component_id = target.get("attachment_patch_id"), target.get("attachment_component_id")
            if not isinstance(patch_id, str) or not isinstance(component_id, str):
                raise ValueError("An attachment corridor requires explicit patch and component IDs")
            self.corridor_patch = self.geometry.attachment_patch(patch_id)
            if self.corridor_patch.component_id != component_id:
                raise ValueError("The attachment corridor patch belongs to a different component")
        self.normals = []
        self._component_reserves = {}

    @staticmethod
    def _nonnegative(value, name):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(name + " must be a finite nonnegative scalar")
        value = float(value)
        if not np.isfinite(value) or value < 0:
            raise ValueError(name + " must be a finite nonnegative scalar")
        return value

    @staticmethod
    def _initial_normal(local, half_extents):
        displacement = local - np.clip(local, -half_extents, half_extents)
        norm = np.linalg.norm(displacement)
        if norm > 1e-12:
            return displacement / norm
        # An infeasible warm seed still needs a defined outward escape face.
        axis = int(np.argmin(half_extents - np.abs(local)))
        normal = np.zeros(3)
        normal[axis] = -1. if local[axis] < 0 else 1.
        return normal

    def _component_reserve(self, component):
        # The declared component matrix is used as an affine map. Include its
        # small permitted departure from orthogonality in the rotation reserve.
        key = self._component_reserve_key(component)
        if key is not None and key in self._component_reserves:
            return self._component_reserves[key]
        extent = max(np.linalg.norm(point) for point in component.nominal_vertices_body_m) + component.uncertainty_m
        result = self.reserve + self.center_error + self.rotation_error * extent
        if key is not None and key == self._component_reserve_key(component):
            if len(self._component_reserves) >= 64:
                self._component_reserves.pop(next(iter(self._component_reserves)))
            self._component_reserves[key] = result
        return result

    def _component_reserve_key(self, component):
        """Reuse only the exact built-in calculation under its current inputs.

        Whole geometry identities and every admission/source check remain live.
        Custom vertex behavior takes the original calculation on every call.
        """
        if (type(component) is not OBBComponent or '_vertices' in component.__dict__
                or OBBComponent.nominal_vertices_body_m is not _OBB_NOMINAL_VERTICES
                or OBBComponent._vertices is not _OBB_VERTICES):
            return None
        rotation = component.rotation_component_to_body
        if type(rotation) is not tuple or len(rotation) != 3:
            return None
        rows = (component.center_m, component.half_extents_m) + rotation
        if any(type(row) is not tuple or len(row) != 3 for row in rows):
            return None
        values = tuple(value for row in rows for value in row) + (
            component.uncertainty_m, self.reserve, self.center_error, self.rotation_error)
        if any(type(value) is not float or not math.isfinite(value) for value in values):
            return None
        # hex retains signed zero as well as every finite binary64 declaration.
        return tuple(value.hex() for value in values)

    def add_node(self, backend, node, position_minus_target_center, omega, reference_position):
        """Add each union exclusion; position is already target-center relative."""
        rotation = self.rotations[node]
        body_guess = rotation.T @ (np.asarray(reference_position, dtype=float) - self.center)
        for component in self.geometry.components:
            backend.constraint_context = f"composite_target[{node}:{component.id}]"
            component_rotation = np.asarray(component.rotation_component_to_body, dtype=float)
            local_guess = component_rotation.T @ (body_guess - component.center_m)
            seed = rotation @ component_rotation @ self._initial_normal(local_guess, np.asarray(component.half_extents_m, dtype=float))
            seed /= np.linalg.norm(seed)
            normal = [backend.model.Var(value=float(value), lb=-1., ub=1.) for value in seed]
            backend._eq(sum(value * value for value in normal), 1.)
            combined_rotation = rotation @ component_rotation
            local_normal = backend._dot(combined_rotation.T, normal)
            local_seed = combined_rotation.T @ seed
            box_support = sum(float(component.half_extents_m[i]) * backend._abs(local_normal[i], local_seed[i])
                              for i in range(3))
            tube_support = sum(backend._abs(normal[i], seed[i]) * omega[i] for i in range(3))
            uncertainty_support = float(component.uncertainty_m) * backend._norm(backend._dot(rotation.T, normal))
            component_center = rotation @ np.asarray(component.center_m)
            backend._ge(sum(normal[i] * (position_minus_target_center[i] - float(component_center[i])) for i in range(3)),
                        box_support + tube_support + uncertainty_support + self._component_reserve(component))
            if self.corridor_patch is not None and self.corridor_patch.component_id == component.id:
                # A declared approach face adds a conservative corridor; all
                # exact union exclusions above remain active independently.
                face = rotation @ np.asarray(self.corridor_patch.normal_body)
                face /= np.linalg.norm(face)
                face_support = np.abs(combined_rotation.T @ face) @ np.asarray(component.half_extents_m)
                face_uncertainty = component.uncertainty_m * np.linalg.norm(rotation.T @ face)
                backend._ge(sum(float(face[i]) * (position_minus_target_center[i] - float(component_center[i])) for i in range(3)),
                    float(face_support + face_uncertainty + self._component_reserve(component))
                    + sum(abs(float(face[i])) * omega[i] for i in range(3)))
            self.normals.append((node, component, normal))

    def numerical_margins(self, states, omega):
        """Recheck normalized support planes using plain numerical geometry."""
        margins = []
        for node, component, variables in self.normals:
            normal = np.asarray([float(value.value[0]) for value in variables])
            norm = np.linalg.norm(normal)
            if not np.isfinite(norm) or norm == 0:
                raise ValueError("Composite separating normal is not finite and nonzero")
            normal /= norm
            rotation = self.rotations[node]
            component_rotation = np.asarray(component.rotation_component_to_body, dtype=float)
            combined_rotation = rotation @ component_rotation
            offset = states[:3, node] - self.center - rotation @ np.asarray(component.center_m)
            margin = (normal @ offset - np.abs(combined_rotation.T @ normal) @ component.half_extents_m
                      - np.abs(normal) @ omega[node]
                      - component.uncertainty_m * np.linalg.norm(rotation.T @ normal)
                      - self._component_reserve(component))
            if not np.isfinite(margin):
                raise ValueError("Composite support-plane margin is not finite")
            margins.append(dict(node=node, component_id=component.id, separation_margin_m=float(margin)))
        return margins

    def summary(self, margins):
        return dict(model="per-component-unit-support-plane-v1", geometry_id=self.geometry.geometry_id,
                    mode="free", stricter_reference_corridor=False,
                    geometry_fingerprint=self.geometry.fingerprint, component_count=len(self.geometry.components),
                    checked_component_nodes=len(margins),
                    attachment_corridor_patch_id=None if self.corridor_patch is None else self.corridor_patch.id,
                    minimum_support_margin_m=min(row["separation_margin_m"] for row in margins),
                    scope="Nonlinear numerical node constraints for the complete declared OBB union; continuous admission is separate")


class GekkoReferenceCorridorGeometry(GekkoCompositeGeometry):
    """Prescribe one native initial-iterate witness per component/node, keeping all supports.

    Fixing these normals restricts the free-normal feasible set: a safe detour
    on another face may be excluded. Support coefficients and the six original
    absolute-value objective terms become numerical constants. The nonlinear
    dynamics, tube, input and terminal problem and original normalized geometry
    postcheck remain active. This supplies no continuous-flow certificate.
    """

    def __init__(self, target, rotations, center):
        super().__init__(target, rotations, center)
        if self.geometry.frame != "target_body_COM":
            raise ValueError("The reference corridor requires target_body_COM geometry")
        raw = np.asarray(rotations)
        if (raw.dtype.kind not in "iuf" or self.rotations.ndim != 3 or not len(self.rotations)
                or self.rotations.shape[1:] != (3, 3) or not np.all(np.isfinite(self.rotations))):
            raise ValueError("Reference-corridor rotations require finite numeric (T,3,3) data")
        if (not np.allclose(self.rotations.transpose(0, 2, 1) @ self.rotations, np.eye(3), atol=1e-7, rtol=0)
                or not np.allclose(np.linalg.det(self.rotations), 1., atol=1e-7, rtol=0)):
            raise ValueError("Reference-corridor target matrices must be proper rotations")
        self.center = self._vector(center, "Reference-corridor target center")
        self.witnesses = []
        self.normal_parameters = []
        self.constant_geometry_objective = 0.

    @staticmethod
    def _vector(value, name, *, nonnegative=False):
        raw = np.asarray(value)
        if raw.shape != (3,) or raw.dtype.kind not in "iuf":
            raise ValueError(name + " requires three real numeric scalars")
        result = raw.astype(float)
        if not np.all(np.isfinite(result)) or (nonnegative and np.any(result < 0)):
            raise ValueError(name + " requires finite scalars with valid signs")
        return result

    @classmethod
    def _unit(cls, value):
        value = cls._vector(value, "Reference-corridor normal")
        norm = math.hypot(*value)
        if not math.isfinite(norm) or abs(norm - 1.) > 1e-12:
            raise ValueError("A reference-corridor normal must be finite, nonzero and unit length")
        return value

    @staticmethod
    def _numeric_dot(matrix, vector):
        # Match the scalar operation order of the original support expression.
        return np.array([sum(float(a) * float(b) for a, b in zip(row, vector) if a != 0) for row in matrix])

    @classmethod
    def _reference_margin(cls, normal, offset, half, local, omega, uncertainty, target_local, reserve):
        normal = cls._unit(normal)
        offset = cls._vector(offset, "Reference-corridor component displacement")
        half = cls._vector(half, "Reference-corridor component half extents", nonnegative=True)
        local = cls._vector(local, "Reference-corridor local normal")
        omega = cls._vector(omega, "Reference tube seed", nonnegative=True)
        target_local = cls._vector(target_local, "Reference-corridor body normal")
        value = (float(normal @ offset) - float(half @ np.abs(local)) - float(np.abs(normal) @ omega)
                 - float(uncertainty) * math.sqrt(float(target_local @ target_local) + 1e-20) - float(reserve))
        if not math.isfinite(value) or value < 0:
            raise ValueError("The initial-iterate witness does not separate its finite seed tube from the component; "
                             "this does not establish dynamic infeasibility")
        return value

    def add_node(self, backend, node, position_minus_target_center, omega, reference_position):
        if isinstance(node, (bool, np.bool_)) or not isinstance(node, Integral) or not 0 <= node < len(self.rotations):
            raise ValueError("A reference-corridor node must name a prediction node")
        reference = self._vector(reference_position, "Reference-corridor initial iterate position")
        seed_omega = self._vector([float(np.ravel(getattr(v.value, "value", v.value))[0]) for v in omega],
                                  "Reference tube seed", nonnegative=True)
        rotation = self.rotations[node]
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                body_reference = rotation.T @ (reference - self.center)
                for component in self.geometry.components:
                    self._add_component(backend, node, position_minus_target_center, omega, reference,
                                        seed_omega, rotation, body_reference, component)
        except (FloatingPointError, OverflowError) as error:
            raise ValueError("Reference-corridor witness arithmetic is not finite") from error

    def _add_component(self, backend, node, position, omega, reference, seed_omega, rotation, body_reference, component):
        component_rotation = np.asarray(component.rotation_component_to_body, dtype=float)
        local_guess = component_rotation.T @ (body_reference - component.center_m)
        candidate = rotation @ component_rotation @ self._initial_normal(local_guess, np.asarray(component.half_extents_m))
        candidate /= math.hypot(*candidate)
        normal = self._unit(candidate)
        combined = rotation @ component_rotation
        local = self._numeric_dot(combined.T, normal)
        target_local = self._numeric_dot(rotation.T, normal)
        world_center = rotation @ np.asarray(component.center_m)
        reserve = self._component_reserve(component)
        reference_margin = self._reference_margin(normal, reference - self.center - world_center,
            component.half_extents_m, local, seed_omega, component.uncertainty_m, target_local, reserve)
        box = float(np.asarray(component.half_extents_m) @ np.abs(local))
        uncertainty = float(component.uncertainty_m) * math.sqrt(float(target_local @ target_local) + 1e-20)
        parameters = [backend.model.Param(value=float(value)) for value in normal]
        self.normal_parameters.extend(parameters)
        backend.constraint_context = f"reference_component_support[{node}:{component.id}]"
        backend._ge(sum(parameters[i] * (position[i] - float(world_center[i])) for i in range(3)),
                    box + uncertainty + reserve + sum(abs(float(normal[i])) * omega[i] for i in range(3)))
        # Restrict the original epigraph objective too, including its constants.
        for coefficient in list(np.abs(local)) + list(np.abs(normal)):
            term = 1e-6 * float(coefficient)
            backend.model.Minimize(term)
            self.constant_geometry_objective += term
        if self.corridor_patch is not None and self.corridor_patch.component_id == component.id:
            face = rotation @ np.asarray(self.corridor_patch.normal_body)
            face /= np.linalg.norm(face)
            face_support = np.abs(combined.T @ face) @ np.asarray(component.half_extents_m)
            face_uncertainty = component.uncertainty_m * np.linalg.norm(rotation.T @ face)
            backend._ge(sum(float(face[i]) * (position[i] - float(world_center[i])) for i in range(3)),
                        float(face_support + face_uncertainty + reserve)
                        + sum(abs(float(face[i])) * omega[i] for i in range(3)))
        self.normals.append((node, component, parameters))
        self.witnesses.append(dict(node=int(node), component_id=component.id, initial_iterate_position_m=reference.tolist(),
            seed_tube_halfwidths_m=seed_omega.tolist(), normal_world=normal.tolist(),
            normal_squared_norm=float(normal @ normal), local_projection=local.tolist(),
            tube_support_coefficients=np.abs(normal).tolist(), box_support=box, uncertainty_support=uncertainty,
            physical_and_pose_reserve=reserve, initial_iterate_seed_tube_margin_m=reference_margin))

    def summary(self, margins):
        context = dict(geometry_fingerprint=self.geometry.fingerprint, target_center_m=self.center.tolist(),
                       target_rotations=self.rotations.tolist(), witnesses=self.witnesses)
        fingerprint = hashlib.sha256(json.dumps(context, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        return dict(super().summary(margins), mode="reference_corridor", stricter_reference_corridor=True,
                    witness_policy="initial-iterate-component-support-v1", witness_count=len(self.witnesses),
                    witness_fingerprint=fingerprint, witness_source="per-node current-state nominal RK4 control rollout (node zero prescribed) and initial predicted tube",
                    initial_iterate_dynamic_feasibility_claim=False,
                    constant_geometry_objective=self.constant_geometry_objective,
                    minimum_initial_iterate_seed_tube_margin_m=min(row["initial_iterate_seed_tube_margin_m"] for row in self.witnesses),
                    restriction="Prescribed initial-iterate support normals; safe detours on other faces may be excluded",
                    continuous_proof_claim=False)
