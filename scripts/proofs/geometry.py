"""geometry.py: Continuous exclusion and reference clearance for a declared convex union.

Every occupied spacecraft component is checked separately.  Expanded exact
vertices already contain the geometry model's dimension/placement uncertainty;
that uncertainty is not added twice.  Numerical closest points select candidate
planes only.  Admission uses exact supports and outward Arb norm bounds.

Copyright: Aaron John Sabu
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from numbers import Integral
from typing import Any

from orbexa.common.intervals import (
    _ARB_LOCK,
    IntervalBox,
    _ArbArithmetic,
    _exact,
    _ProofUnresolved,
)
from orbexa.proofs.reachability import (
    FRAME,
    CollisionCertificate,
    EnclosureStatus,
    FlowCertificate,
    GuardStatus,
    SweptCollisionGuard,
)

COMPOSITE_SWEEP_METHOD = "arb-composite-swept-separation-v1"
COMPOSITE_DISTANCE_METHOD = "arb-composite-reference-separation-v1"


def _point(values):
    """Return three exact Fractions for a 3-vector.

    Raises ValueError unless there are exactly three components; ``_exact``
    converts each value.
    """
    result = tuple(map(_exact, values))
    if len(result) != 3:
        raise ValueError("A composite geometry vector needs three finite components")
    return result


def _dot(left, right):
    """Return the exact dot product of two vectors as a Fraction."""
    return sum((a * b for a, b in zip(left, right)), Fraction(0))


def _multiply(matrix, vector):
    """Return the exact product of a row-major matrix and a vector."""
    return tuple(_dot(row, vector) for row in matrix)


def _geometry_context(geometry):
    """Validate a spacecraft geometry and return ``(parts, context, fingerprint)``.

    Raises TypeError unless ``geometry`` is a ``SpacecraftGeometry``. Raises
    ValueError unless its frame is ``target_body_COM``, its components are
    nonempty with unique ids and each has exact vertices, and ``as_dict()`` is
    finite JSON-serializable. ``context`` is that dictionary.
    """
    # Import lazily: result loading and the cylinder baseline need not import
    # the spacecraft asset/provenance layer.
    from orbexa.models.geometry.composite import SpacecraftGeometry

    if not isinstance(geometry, SpacecraftGeometry):
        raise TypeError(
            "Supply an explicit supported SpacecraftGeometry; no geometry fallback is available"
        )
    if geometry.frame != "target_body_COM":
        raise ValueError(
            "Composite motion and occupied vertices must share the declared target-body COM frame"
        )
    parts = tuple(geometry.components)
    if not parts or len({part.id for part in parts}) != len(parts):
        raise ValueError(
            "The spacecraft needs nonempty uniquely identified occupied components"
        )
    for part in parts:
        vertices = tuple(_point(vertex) for vertex in part.vertices_body_m)
        if not vertices:
            raise ValueError(
                "Every occupied component needs conservative exact convex vertices"
            )
    context = geometry.as_dict()
    # Reject nonfinite or nonserializable provenance before accepting a shape.
    json.dumps(context, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return parts, context, str(geometry.fingerprint)


def _candidate_normals(point, component, matrix):
    """Directions can be approximate; every returned direction is re-proved."""
    import numpy as np

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            point = np.asarray(tuple(map(float, point)))
            transform = np.asarray(tuple(tuple(map(float, row)) for row in matrix))
            center = np.asarray(component.center_m, dtype=float)
            axes = np.asarray(component.rotation_component_to_body, dtype=float)
            extents = np.asarray(component.half_extents_m, dtype=float)
            if not all(
                np.all(np.isfinite(value))
                for value in (point, transform, center, axes, extents)
            ):
                raise ValueError(
                    "Candidate coordinates exceed the finite numerical search domain"
                )
            local = axes.T @ (transform.T @ point - center)
            nearest = transform @ (center + axes @ np.clip(local, -extents, extents))
            directions = [point - nearest, point - transform @ center]
            directions += [
                sign * (transform @ axes[:, index])
                for index in range(3)
                for sign in (1, -1)
            ]
            directions += [
                sign * np.eye(3)[index] for index in range(3) for sign in (1, -1)
            ]
    except (OverflowError, FloatingPointError, ValueError) as error:
        raise _ProofUnresolved(
            f"Numerical component-plane candidate search refused: {type(error).__name__}: {error}"
        ) from error
    result, seen = [], set()
    for direction in directions:
        if np.all(np.isfinite(direction)) and np.any(direction):
            normal = _point(direction)
            if normal not in seen:
                result.append(normal)
                seen.add(normal)
    return tuple(result)


def _matrix(value, error):
    """Return an exact 3x3 matrix after consistency checks against ``error``.

    Raises ValueError if it is not 3x3, if ``M^T M`` differs from identity by more
    than ``2*error + error^2`` in any entry, or if ``error < 1`` and the
    determinant is not positive. These are necessary conditions, not a proof that a
    rotation lies within ``error``.
    """
    matrix = tuple(_point(row) for row in value)
    if len(matrix) != 3:
        raise ValueError("Composite target pose requires a three by three matrix")
    # Necessary consistency conditions for a proper R with ||M-R|| <= error.
    limit = 2 * error + error * error
    if any(
        abs(sum(matrix[k][i] * matrix[k][j] for k in range(3)) - int(i == j)) > limit
        for i in range(3)
        for j in range(3)
    ):
        raise ValueError(
            "Composite target matrix contradicts its declared rotation error"
        )
    determinant = (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )
    if error < 1 and determinant <= 0:
        raise ValueError("Composite target matrix cannot be close to a proper rotation")
    return matrix


@dataclass(frozen=True)
class RotatingCompositeBounds:
    """One rigid convex union under an explicit whole-hold pose/rate contract.

    ``orientation_matrix_error`` bounds the operator error relative to the true
    body-to-flow-frame rotation at the exact hold start, including query-clock
    displacement.  The rate is the relative angular speed per anomaly.  The
    same physical geometry and rigid rotation apply to every component.
    """

    geometry: Any
    orientation_matrix_at_start: Sequence[Sequence[Any]]
    orientation_matrix_error: Any
    angular_rate_per_anomaly: Any
    center_at_start: IntervalBox = field(
        default_factory=lambda: IntervalBox.point((0, 0, 0))
    )
    center_speed_per_anomaly: Any = 0
    frame: str = FRAME
    scope: str = (
        "Declared rigid composite geometry and pose/rate bounds over the complete hold"
    )

    def __post_init__(self):
        """Validate and normalize the fields in place.

        Checks the geometry, converts the three nonnegative scalar fields to exact
        values, converts the start matrix with ``_matrix``, and requires a
        three-dimensional ``IntervalBox`` center and nonempty ``frame`` and ``scope``.
        Raises ValueError (or TypeError for the geometry) on any violation.
        """
        _geometry_context(self.geometry)
        for name in (
            "orientation_matrix_error",
            "angular_rate_per_anomaly",
            "center_speed_per_anomaly",
        ):
            value = _exact(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "orientation_matrix_at_start",
            _matrix(self.orientation_matrix_at_start, self.orientation_matrix_error),
        )
        if (
            not isinstance(self.center_at_start, IntervalBox)
            or self.center_at_start.dimension != 3
        ):
            raise ValueError(
                "Composite target center needs a three-dimensional exact box"
            )
        if (
            not isinstance(self.frame, str)
            or not self.frame
            or not isinstance(self.scope, str)
            or not self.scope.strip()
        ):
            raise ValueError(
                "Identify the composite target frame and whole-hold bound source"
            )

    def as_dict(self):
        """Return the bounds as a dictionary with the geometry context and fingerprint,
        the matrix and scalars as strings, and the center box.
        """
        _, geometry, identity = _geometry_context(self.geometry)
        result = {
            "shape": "composite_spacecraft",
            "geometry": geometry,
            "geometry_fingerprint": identity,
            "orientation_matrix_at_start": tuple(
                tuple(map(str, row)) for row in self.orientation_matrix_at_start
            ),
            "orientation_matrix_error": str(self.orientation_matrix_error),
            "angular_rate_per_anomaly": str(self.angular_rate_per_anomaly),
            "center_at_start": self.center_at_start.as_dict(),
            "center_speed_per_anomaly": str(self.center_speed_per_anomaly),
            "frame": self.frame,
            "scope": self.scope,
        }
        return result


class CompositeSweptCollisionGuard:
    """Prove finite-bus separation from each part and every other chaser.

    A failed candidate search is UNRESOLVED, never a collision claim.  A union
    cannot pass by being outside only its bus, one chosen appendage, a center
    sphere, or a substitute cylinder.
    """

    def __init__(self, *, precision_bits=128):
        """Validate ``precision_bits`` and create the pair guard at that precision.

        ``precision_bits`` must be an integer (not bool) of at least 64. Raises
        ValueError otherwise.
        """
        if (
            isinstance(precision_bits, bool)
            or not isinstance(precision_bits, Integral)
            or precision_bits < 64
        ):
            raise ValueError("Use an integer precision_bits >= 64")
        self.precision_bits = int(precision_bits)
        self.pair_guard = SweptCollisionGuard(precision_bits=self.precision_bits)

    def check(
        self,
        flows,
        body_radii,
        *,
        target,
        clearance=0,
        minimum_pair_center_distance=0,
        normals=(),
    ):
        """Return a ``CollisionCertificate`` for swept chaser sets against the target.

        Inputs: ``flows`` are proven six-dimensional ``FlowCertificate`` objects with
        equal start anomaly, hold width and the target frame; ``body_radii`` gives one
        nonnegative radius per flow; ``clearance`` and ``minimum_pair_center_distance``
        are nonnegative; ``normals`` are extra nonzero candidate separating
        directions in the target frame.
        Outputs: PROVEN_SAFE only if every flow segment is strictly separated from
        every component by an Arb-checked plane and, for several flows, the pair guard
        admits them. Otherwise UNRESOLVED or UNAVAILABLE (python-flint missing), with
        the evidence gathered so far. Raises TypeError if ``target`` is not
        ``RotatingCompositeBounds`` and ValueError for invalid radii, clearances or
        normals.
        """
        if not isinstance(target, RotatingCompositeBounds):
            raise TypeError(
                "Composite exclusion requires RotatingCompositeBounds; unsupported geometry refuses"
            )
        flows, radii = tuple(flows), tuple(map(_exact, body_radii))
        clearance, pair_distance = (
            _exact(clearance),
            _exact(minimum_pair_center_distance),
        )
        if (
            not flows
            or len(flows) != len(radii)
            or min(radii) < 0
            or clearance < 0
            or pair_distance < 0
        ):
            raise ValueError(
                "Supply a nonnegative physical body radius for every flow and nonnegative clearances"
            )
        fixed_normals = tuple(_point(value) for value in normals)
        if any(not any(value) for value in fixed_normals):
            raise ValueError("Candidate separating normals must be nonzero")
        common = {
            "method": COMPOSITE_SWEEP_METHOD,
            "target_bounds": target.as_dict(),
            "body_radii": tuple(map(str, radii)),
            "clearance": str(clearance),
            "minimum_pair_center_distance": str(pair_distance),
        }
        if any(
            not isinstance(flow, FlowCertificate)
            or not isinstance(flow.status, EnclosureStatus)
            or not flow.proven
            for flow in flows
        ):
            return CollisionCertificate(
                GuardStatus.UNRESOLVED,
                "Every continuous flow must first be PROVEN",
                **common,
            )
        try:
            for flow in flows:
                SweptCollisionGuard._check_flow_structure(flow)
            common["flow_fingerprints"] = tuple(flow.fingerprint for flow in flows)
        except (
            ValueError,
            TypeError,
            OverflowError,
            AttributeError,
            RecursionError,
        ) as error:
            return CollisionCertificate(
                GuardStatus.UNRESOLVED,
                f"Invalid continuous flow coverage or identity: {type(error).__name__}: {error}",
                **common,
            )
        first = flows[0]
        if any(
            flow.initial_state.dimension != 6
            or flow.frame != target.frame
            or flow.start_anomaly != first.start_anomaly
            or flow.hold_width != first.hold_width
            for flow in flows
        ):
            return CollisionCertificate(
                GuardStatus.UNRESOLVED,
                "Incompatible composite flow frames or exact clocks",
                **common,
            )
        parts, _, fingerprint = _geometry_context(target.geometry)
        try:
            import flint
        except ImportError as error:
            return CollisionCertificate(
                GuardStatus.UNAVAILABLE, f"python-flint is required: {error}", **common
            )
        evidence = []
        with _ARB_LOCK, flint.ctx.workprec(self.precision_bits):
            ar = _ArbArithmetic(flint)
            try:
                for member, (flow, radius) in enumerate(zip(flows, radii)):
                    for segment in SweptCollisionGuard._segments(flow):
                        for part in parts:
                            proof = self._component_segment(
                                segment,
                                radius + clearance,
                                target,
                                part,
                                fixed_normals,
                                ar,
                            )
                            if proof is None:
                                return CollisionCertificate(
                                    GuardStatus.UNRESOLVED,
                                    f"No composite separation proved for chaser {member}, component {part.id}, offset {segment.offset}",
                                    tuple(evidence),
                                    **common,
                                )
                            evidence.append(
                                dict(
                                    kind="target_component",
                                    chaser=member,
                                    component_id=part.id,
                                    geometry_fingerprint=fingerprint,
                                    **proof,
                                )
                            )
            except _ProofUnresolved as error:
                return CollisionCertificate(
                    GuardStatus.UNRESOLVED, str(error), tuple(evidence), **common
                )
        if len(flows) > 1:
            pairs = self.pair_guard.check(
                flows,
                radii,
                clearance=clearance,
                minimum_pair_center_distance=pair_distance,
            )
            evidence.extend(pairs.evidence)
            if not pairs.admissible:
                return CollisionCertificate(
                    pairs.status, pairs.reason, tuple(evidence), **common
                )
        if fingerprint != _geometry_context(target.geometry)[2]:
            return CollisionCertificate(
                GuardStatus.UNRESOLVED,
                "Composite geometry changed during admission",
                tuple(evidence),
                **common,
            )
        return CollisionCertificate(
            GuardStatus.PROVEN_SAFE,
            "Every swept finite-bus set is strictly separated from every declared spacecraft component and every peer",
            tuple(evidence),
            **common,
        )

    @staticmethod
    def _component_segment(segment, body_radius, target, component, normals, ar):
        """Search for a plane separating one swept segment from one component.

        Candidates are ``normals`` plus numerical directions from the segment's
        relative midpoint. A plane is accepted when the lower bound of the normal
        projection over the swept position and target center, minus the exact support
        of the component under the start matrix, exceeds the norm times the body
        radius plus a reserve for rotation, matrix error and center motion.
        Returns an evidence dictionary for the first accepted plane, else None.
        """
        vertices = tuple(_point(vertex) for vertex in component.vertices_body_m)
        matrix, center = target.orientation_matrix_at_start, target.center_at_start
        position = segment.swept_state.slice(0, 3)
        relative_mid = tuple(
            (lo + hi - clo - chi) / 2
            for lo, hi, clo, chi in zip(
                position.lower, position.upper, center.lower, center.upper
            )
        )
        candidates = normals + _candidate_normals(relative_mid, component, matrix)
        radius_squared = max(_dot(vertex, vertex) for vertex in vertices)
        component_radius = ar.scalar(radius_squared).sqrt()
        rotation = min(Fraction(2), target.angular_rate_per_anomaly * segment.offset[1])
        reserve = component_radius * ar.scalar(
            rotation + target.orientation_matrix_error
        ) + ar.scalar(target.center_speed_per_anomaly * segment.offset[1])
        for normal in candidates:
            norm = ar.scalar(_dot(normal, normal)).sqrt()
            # Exact support of the complete declared convex component at M.
            support = max(
                _dot(normal, _multiply(matrix, vertex)) for vertex in vertices
            )
            separating_lower = sum(
                (
                    min(n * lo, n * hi) - max(n * clo, n * chi)
                    for n, lo, hi, clo, chi in zip(
                        normal,
                        position.lower,
                        position.upper,
                        center.lower,
                        center.upper,
                    )
                ),
                Fraction(0),
            )
            margin = ar.scalar(separating_lower - support) - norm * (
                ar.scalar(body_radius) + reserve
            )
            if margin.is_finite() and margin > 0:
                return {
                    "offset": tuple(map(str, segment.offset)),
                    "normal": tuple(map(str, normal)),
                    "margin": ar.box((margin,)).as_dict(),
                    "exact_component_support": str(support),
                    "component_radius_squared_upper": str(radius_squared),
                    "rotation_norm_bound": str(rotation),
                    "rotation_and_translation_reserve": ar.box((reserve,)).as_dict(),
                }
        return None


@dataclass(frozen=True)
class CompositePointSeparation:
    """Result of a point-to-union clearance check in the target body frame.

    ``distance_lower`` is a lower bound of the distance from ``point_body`` to the
    union, meaningful when ``admissible``; otherwise it is zero. ``evidence`` holds
    one record per component.
    """

    status: str
    reason: str
    distance_lower: Fraction
    point_body: tuple[Fraction, ...]
    geometry_fingerprint: str
    evidence: tuple[dict, ...] = ()
    method: str = COMPOSITE_DISTANCE_METHOD

    @property
    def admissible(self):
        """Return True when the status is ``PROVEN_OUTSIDE_EVERY_COMPONENT``."""
        return self.status == "PROVEN_OUTSIDE_EVERY_COMPONENT"

    def as_dict(self):
        """Return the result as a dictionary with numeric fields as strings."""
        return {
            "status": self.status,
            "reason": self.reason,
            "distance_lower": str(self.distance_lower),
            "point_body": tuple(map(str, self.point_body)),
            "geometry_fingerprint": self.geometry_fingerprint,
            "evidence": self.evidence,
            "method": self.method,
        }


class CompositeTerminalGeometry:
    """Rotation-invariant lower clearance to the complete occupied union.

    A true common rigid rotation preserves the body-frame distances.  The
    reference point and all parts must share it.  Whole-terminal-set erosion
    and body/clearance radii are applied by the terminal inequality layer.
    """

    def __init__(self, geometry, *, precision_bits=160):
        """Validate ``precision_bits`` (an integer, not bool, at least 64) and the
        geometry, and store the parts, context and fingerprint.

        Raises ValueError for an invalid precision or geometry and TypeError for an
        unsupported geometry type.
        """
        if (
            isinstance(precision_bits, bool)
            or not isinstance(precision_bits, Integral)
            or precision_bits < 64
        ):
            raise ValueError("Use an integer precision_bits >= 64")
        self.parts, self.context, self.fingerprint = _geometry_context(geometry)
        self.geometry, self.precision_bits = geometry, int(precision_bits)

    def point_separation(self, point_body):
        """Return a ``CompositePointSeparation`` for a point in the target body frame.

        For each component a candidate plane is accepted when the exact projection gap
        is positive and the Arb distance lower bound is positive; the best plane per
        component is kept and the minimum is reported. UNAVAILABLE without
        python-flint; UNRESOLVED if a component has no such plane, candidate search is
        refused, or the geometry fingerprint changes during the proof.
        """
        point = _point(point_body)
        try:
            import flint
        except ImportError as error:
            return CompositePointSeparation(
                "UNAVAILABLE", str(error), Fraction(0), point, self.fingerprint
            )
        identity = (
            (Fraction(1), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(0), Fraction(1)),
        )
        evidence, distances = [], []
        with _ARB_LOCK, flint.ctx.workprec(self.precision_bits):
            ar = _ArbArithmetic(flint)
            for part in self.parts:
                vertices = tuple(_point(vertex) for vertex in part.vertices_body_m)
                best = None
                try:
                    candidates = _candidate_normals(point, part, identity)
                except _ProofUnresolved as error:
                    return CompositePointSeparation(
                        "UNRESOLVED",
                        str(error),
                        Fraction(0),
                        point,
                        self.fingerprint,
                        tuple(evidence),
                    )
                for normal in candidates:
                    gap = _dot(normal, point) - max(
                        _dot(normal, vertex) for vertex in vertices
                    )
                    if gap <= 0:
                        continue
                    distance = ar.box(
                        (ar.scalar(gap) / ar.scalar(_dot(normal, normal)).sqrt(),)
                    )
                    if distance.lower[0] > 0 and (
                        best is None or distance.lower[0] > best[0]
                    ):
                        best = (
                            distance.lower[0],
                            {
                                "component_id": part.id,
                                "normal": tuple(map(str, normal)),
                                "exact_support": str(
                                    max(_dot(normal, vertex) for vertex in vertices)
                                ),
                                "exact_projection_gap": str(gap),
                                "distance": distance.as_dict(),
                            },
                        )
                if best is None:
                    return CompositePointSeparation(
                        "UNRESOLVED",
                        f"No strict exterior plane proved for component {part.id}",
                        Fraction(0),
                        point,
                        self.fingerprint,
                        tuple(evidence),
                    )
                distances.append(best[0])
                evidence.append(best[1])
        if self.fingerprint != _geometry_context(self.geometry)[2]:
            return CompositePointSeparation(
                "UNRESOLVED",
                "Geometry changed during global clearance proof",
                Fraction(0),
                point,
                self.fingerprint,
                tuple(evidence),
            )
        return CompositePointSeparation(
            "PROVEN_OUTSIDE_EVERY_COMPONENT",
            "Exact convex supports and Arb norms prove distance from every component of the common rigid target",
            min(distances),
            point,
            self.fingerprint,
            tuple(evidence),
        )

    def distance_lower(self, point_body):
        """Return the proven lower bound of the distance from the point to the union.

        Raises ValueError with the certificate reason unless the separation is proven.
        """
        certificate = self.point_separation(point_body)
        if not certificate.admissible:
            raise ValueError(certificate.reason)
        return certificate.distance_lower
