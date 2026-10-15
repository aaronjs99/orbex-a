"""Declared composite spacecraft solids and unprepared surface candidates.

Numerical distances select candidate routes; they are not collision proofs.
Exact expanded vertices bind interval proofs to the same occupied components.
There is no cylindrical or spherical replacement for a composite geometry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import itertools
import json
import math
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence
import numpy as np


def _real(value, name, *, nonnegative=False, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real numeric scalar")
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite binary64 value") from error
    if not math.isfinite(value) or (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError(f"{name} has an invalid finite value or sign")
    return value


def _vector(value, size, name):
    try:
        values = tuple(value)
    except TypeError as error:
        raise ValueError(f"{name} must have {size} components") from error
    if len(values) != size:
        raise ValueError(f"{name} must have {size} components")
    return tuple(_real(v, name) for v in values)


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _freeze(value):
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Geometry metadata keys must be strings")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Real):
        return _real(value, "Geometry metadata number")
    raise ValueError("Geometry metadata must contain finite JSON values")


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class OBBComponent:
    """Nominal c+R[-h,h], dilated by an isotropic uncertainty radius.

    `vertices_body_m` contains EXACT rational vertices of an enclosing box,
    already including uncertainty once. Numerical distances use the nominal
    proper rotation and are candidate calculations, not validated arithmetic.
    """

    id: str
    center_m: Sequence[float]
    half_extents_m: Sequence[float]
    rotation_component_to_body: Sequence[Sequence[float]] = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    uncertainty_m: float = 0.
    role: str = "structure"
    provenance: Mapping = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "id", _identifier(self.id, "Component id"))
        object.__setattr__(self, "role", _identifier(self.role, "Component role"))
        center = _vector(self.center_m, 3, "Component center")
        half = _vector(self.half_extents_m, 3, "Component half extents")
        if min(half) <= 0:
            raise ValueError("Every occupied component must have positive three-dimensional extents")
        rotation = tuple(_vector(row, 3, "Component rotation") for row in self.rotation_component_to_body)
        if len(rotation) != 3:
            raise ValueError("Component rotation must be 3 by 3")
        exact = tuple(tuple(Fraction(v) for v in row) for row in rotation)
        gram_error = tuple(tuple(sum(exact[k][i]*exact[k][j] for k in range(3)) - int(i == j)
                                  for j in range(3)) for i in range(3))
        deviation = max(sum(abs(v) for v in row) for row in gram_error)
        determinant = (exact[0][0]*(exact[1][1]*exact[2][2]-exact[1][2]*exact[2][1])
            - exact[0][1]*(exact[1][0]*exact[2][2]-exact[1][2]*exact[2][0])
            + exact[0][2]*(exact[1][0]*exact[2][1]-exact[1][1]*exact[2][0]))
        if deviation > Fraction(1, 100000000) or determinant <= 0:
            raise ValueError("Component rotation must describe a proper orthogonal frame within 1e-8")
        uncertainty = _real(self.uncertainty_m, "Component uncertainty", nonnegative=True)
        for name, value in (("center_m", center), ("half_extents_m", half),
                            ("rotation_component_to_body", rotation), ("uncertainty_m", uncertainty),
                            ("provenance", _freeze(self.provenance)), ("_gram_deviation", deviation)):
            object.__setattr__(self, name, value)
        # Refuse unrenderable finite declarations rather than export infinities.
        for vertex in self.vertices_body_m:
            try:
                radius = math.hypot(*(float(v) for v in vertex))
            except OverflowError as error:
                raise ValueError("Occupied component vertices must remain finite") from error
            if not math.isfinite(radius):
                raise ValueError("Occupied component vertices must remain finite")

    def _vertices(self, include_uncertainty):
        center = tuple(map(Fraction, self.center_m))
        rotation = tuple(tuple(map(Fraction, row)) for row in self.rotation_component_to_body)
        # ||R^-1||2 <= 1/sqrt(1-d) <= 1/(1-d). Therefore local axis
        # padding u/(1-d) contains the declared Euclidean uncertainty ball.
        padding = Fraction(self.uncertainty_m)/(1-self._gram_deviation) if include_uncertainty else Fraction(0)
        half = tuple(Fraction(v)+padding for v in self.half_extents_m)
        return tuple(tuple(center[i]+sum(rotation[i][j]*half[j]*sign[j] for j in range(3))
                           for i in range(3)) for sign in itertools.product((-1, 1), repeat=3))

    @property
    def vertices_body_m(self):
        """Eight conservative occupied vertices, as tuples of exact Fractions."""
        return self._vertices(True)

    @property
    def nominal_vertices_body_m(self):
        return tuple(tuple(float(v) for v in row) for row in self._vertices(False))

    def signed_distance(self, point_body_m, *, include_uncertainty=True):
        point = np.array(_vector(point_body_m, 3, "Body point"))
        with np.errstate(over='ignore', invalid='ignore'):
            local = np.asarray(self.rotation_component_to_body).T @ (point-np.asarray(self.center_m))
            delta = np.abs(local)-np.asarray(self.half_extents_m)
            result = float(np.linalg.norm(np.maximum(delta, 0))+min(float(np.max(delta)), 0))
            result -= self.uncertainty_m if include_uncertainty else 0.
        if not math.isfinite(result):
            raise ValueError("Body-point distance exceeds finite numerical geometry arithmetic")
        return result

    def closest_point(self, point_body_m, *, include_uncertainty=True):
        point = np.array(_vector(point_body_m, 3, "Body point"))
        rotation, center, half = np.asarray(self.rotation_component_to_body), np.asarray(self.center_m), np.asarray(self.half_extents_m)
        with np.errstate(over='ignore', invalid='ignore'):
            local = rotation.T @ (point-center)
            closest = center+rotation @ np.clip(local, -half, half)
            distance = float(np.linalg.norm(point-closest))
        if not math.isfinite(distance) or not np.all(np.isfinite(closest)):
            raise ValueError("Closest-point query exceeds finite numerical geometry arithmetic")
        radius = self.uncertainty_m if include_uncertainty else 0.
        if distance <= radius:
            return point
        return closest+(radius/distance)*(point-closest) if distance else closest

    def as_dict(self):
        return dict(id=self.id, center_m=list(self.center_m), half_extents_m=list(self.half_extents_m),
            rotation_component_to_body=[list(row) for row in self.rotation_component_to_body],
            uncertainty_m=self.uncertainty_m, role=self.role, provenance=_plain(self.provenance),
            nominal_vertices_body_m=[list(row) for row in self.nominal_vertices_body_m],
            exact_vertices_body_m=[[str(v) for v in row] for row in self.vertices_body_m])

    @classmethod
    def from_dict(cls, value):
        self = cls(**{key: value[key] for key in ("id", "center_m", "half_extents_m", "rotation_component_to_body", "uncertainty_m", "role", "provenance") if key in value})
        if "exact_vertices_body_m" in value:
            claimed = tuple(tuple(Fraction(v) for v in row) for row in value["exact_vertices_body_m"])
            if claimed != self.vertices_body_m:
                raise ValueError("Serialized occupied vertices contradict the physical component declaration")
        if "nominal_vertices_body_m" in value and value["nominal_vertices_body_m"] != [list(row) for row in self.nominal_vertices_body_m]:
            raise ValueError("Serialized display vertices contradict the physical component declaration")
        return self


@dataclass(frozen=True)
class AttachmentPatch:
    """Unprepared nominal structural-face candidate; no strength qualification."""

    id: str
    component_id: str
    point_body_m: Sequence[float]
    normal_body: Sequence[float]
    tangent_u_body: Sequence[float]
    tangent_v_body: Sequence[float]
    half_extents_m: Sequence[float]
    margin_m: float = 0.
    provenance: Mapping = field(default_factory=dict)

    def __post_init__(self):
        for name in ("id", "component_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        for name in ("point_body_m", "normal_body", "tangent_u_body", "tangent_v_body"):
            object.__setattr__(self, name, _vector(getattr(self, name), 3, name))
        frame = np.column_stack((self.tangent_u_body, self.tangent_v_body, self.normal_body))
        if not np.allclose(frame.T@frame, np.eye(3), rtol=0, atol=1e-10) or np.linalg.det(frame) <= 0:
            raise ValueError("Patch tangents and outward normal must form a right-handed orthonormal frame")
        half = _vector(self.half_extents_m, 2, "Patch half extents")
        margin = _real(self.margin_m, "Patch edge margin", nonnegative=True)
        if min(half) <= margin:
            raise ValueError("Patch edge margin must leave a positive accessible rectangle")
        object.__setattr__(self, "half_extents_m", half)
        object.__setattr__(self, "margin_m", margin)
        object.__setattr__(self, "provenance", _freeze(self.provenance))

    def point_at(self, u_m=0., v_m=0.):
        u, v = _real(u_m, "Patch coordinate u"), _real(v_m, "Patch coordinate v")
        if abs(u) > self.half_extents_m[0]-self.margin_m or abs(v) > self.half_extents_m[1]-self.margin_m:
            raise ValueError("Requested point is outside the declared patch edge margin")
        return np.asarray(self.point_body_m)+u*np.asarray(self.tangent_u_body)+v*np.asarray(self.tangent_v_body)

    def contains(self, point_body_m, *, tolerance_m=1e-9):
        delta = np.array(_vector(point_body_m, 3, "Patch query"))-np.asarray(self.point_body_m)
        tolerance = _real(tolerance_m, "Patch query tolerance", nonnegative=True)
        return bool(abs(delta@self.normal_body) <= tolerance
            and abs(delta@self.tangent_u_body) <= self.half_extents_m[0]-self.margin_m+tolerance
            and abs(delta@self.tangent_v_body) <= self.half_extents_m[1]-self.margin_m+tolerance)

    def as_dict(self):
        return dict(id=self.id, component_id=self.component_id, point_body_m=list(self.point_body_m),
            normal_body=list(self.normal_body), tangent_u_body=list(self.tangent_u_body), tangent_v_body=list(self.tangent_v_body),
            half_extents_m=list(self.half_extents_m), margin_m=self.margin_m, provenance=_plain(self.provenance))

    @classmethod
    def from_dict(cls, value):
        return cls(**{key: value[key] for key in ("id", "component_id", "point_body_m", "normal_body", "tangent_u_body", "tangent_v_body", "half_extents_m", "margin_m", "provenance") if key in value})


@dataclass(frozen=True)
class SpacecraftGeometry:
    """An occupied UNION of explicit components in a declared target COM frame."""

    geometry_id: str
    components: Sequence[OBBComponent]
    attachment_patches: Sequence[AttachmentPatch] = ()
    frame: str = "target_body_COM"
    metadata: Mapping = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "geometry_id", _identifier(self.geometry_id, "Geometry id"))
        object.__setattr__(self, "frame", _identifier(self.frame, "Geometry frame"))
        components, patches = tuple(self.components), tuple(self.attachment_patches)
        if not components or any(not isinstance(c, OBBComponent) for c in components) or len({c.id for c in components}) != len(components):
            raise ValueError("A composite requires nonempty uniquely identified OBBComponent solids")
        if any(not isinstance(p, AttachmentPatch) for p in patches) or len({p.id for p in patches}) != len(patches):
            raise ValueError("Attachment patches must be uniquely identified AttachmentPatch objects")
        by_id = {c.id: c for c in components}
        for patch in patches:
            if patch.component_id not in by_id:
                raise ValueError("An attachment patch must name an occupied component")
            component = by_id[patch.component_id]
            rotation = np.asarray(component.rotation_component_to_body)
            normal = rotation.T@patch.normal_body
            axis = int(np.argmax(np.abs(normal)))
            local = rotation.T@(np.asarray(patch.point_body_m)-component.center_m)
            if abs(abs(normal[axis])-1)>1e-8 or np.max(np.abs(np.delete(normal,axis)))>1e-8:
                raise ValueError("An OBB attachment normal must be an outward component-face normal")
            if abs(local[axis]-math.copysign(component.half_extents_m[axis],normal[axis]))>1e-8:
                raise ValueError("An attachment patch must lie on its named nominal component face")
            for u,v in itertools.product((-1,1), repeat=2):
                corner = np.asarray(patch.point_body_m)+u*patch.half_extents_m[0]*np.asarray(patch.tangent_u_body)+v*patch.half_extents_m[1]*np.asarray(patch.tangent_v_body)
                if np.any(np.abs(rotation.T@(corner-component.center_m))>np.asarray(component.half_extents_m)+1e-8):
                    raise ValueError("The complete attachment patch must fit its named nominal face")
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "attachment_patches", patches)
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    def component(self, component_id):
        try:
            return next(c for c in self.components if c.id == component_id)
        except StopIteration as error:
            raise KeyError(f"Unknown component {component_id}") from error

    def attachment_patch(self, patch_id):
        try:
            return next(p for p in self.attachment_patches if p.id == patch_id)
        except StopIteration as error:
            raise KeyError(f"Unknown attachment patch {patch_id}") from error

    def component_clearances(self, point_body_m, *, include_uncertainty=True):
        return {c.id: c.signed_distance(point_body_m, include_uncertainty=include_uncertainty) for c in self.components}

    def signed_distance(self, point_body_m, *, include_uncertainty=True):
        """Minimum component distance in metres; negative denotes occupied union.

        The exterior distance is Euclidean. Inside overlapping components it
        is a signed occupancy indicator, not the exact distance to leave all
        solids. This numerical query does not establish a swept-set proof.
        """
        return min(self.component_clearances(point_body_m, include_uncertainty=include_uncertainty).values())

    def _payload(self):
        return dict(schema="orbexa.spacecraft-geometry.v1", geometry_id=self.geometry_id, frame=self.frame,
            components=[c.as_dict() for c in self.components], attachment_patches=[p.as_dict() for p in self.attachment_patches],
            metadata=_plain(self.metadata))

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self._payload(),sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()

    def as_dict(self):
        return dict(self._payload(), fingerprint=self.fingerprint)

    @classmethod
    def from_dict(cls, value):
        if "geometry" in value and "components" not in value:
            value = value["geometry"]
        if value.get("schema", "orbexa.spacecraft-geometry.v1") != "orbexa.spacecraft-geometry.v1":
            raise ValueError("Unsupported spacecraft geometry schema")
        self = cls(geometry_id=value["geometry_id"], components=tuple(OBBComponent.from_dict(c) for c in value["components"]),
            attachment_patches=tuple(AttachmentPatch.from_dict(p) for p in value.get("attachment_patches",())),
            frame=value.get("frame","target_body_COM"), metadata=value.get("metadata",{}))
        if "fingerprint" in value and value["fingerprint"] != self.fingerprint:
            raise ValueError("Spacecraft geometry fingerprint differs from its exact declarations")
        return self

    @classmethod
    def from_json(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))
