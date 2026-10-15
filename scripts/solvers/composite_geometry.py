"""Nonlinear support-plane exclusion of every declared spacecraft component.

For each OBB and prediction node a free unit normal separates the complete
position box plus spherical bus from that component. The normals are decision
variables, so the union exclusion remains nonconvex and uses the IPOPT path.
No enclosing cylinder/sphere, binary variable or complementarity max is used.
"""
from __future__ import annotations

import numpy as np


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
        extent = max(np.linalg.norm(point) for point in component.nominal_vertices_body_m) + component.uncertainty_m
        return self.reserve + self.center_error + self.rotation_error * extent

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
                    geometry_fingerprint=self.geometry.fingerprint, component_count=len(self.geometry.components),
                    checked_component_nodes=len(margins),
                    attachment_corridor_patch_id=None if self.corridor_patch is None else self.corridor_patch.id,
                    minimum_support_margin_m=min(row["separation_margin_m"] for row in margins),
                    scope="Nonlinear numerical node constraints for the complete declared OBB union; continuous admission is separate")
