"""proximity.py: Target-size-aware proposal regimes; collision proofs remain authoritative.

Copyright: Aaron John Sabu
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import inf, isfinite, isqrt, nextafter
from numbers import Real

from orbexa.models.geometry.composite import SpacecraftGeometry

# Largest componentwise position box among the existing refuge proposals.
# The geometry selector must account for this complete three-dimensional box.
REFUGE_POSITION_COMPONENT_RADIUS_M = ".15"


def _nonnegative(value, name, *, positive=False):
    """Return value as an exact Fraction.

    Raises ValueError for a bool, a non-numeric or non-finite value, a negative value,
    or zero when positive is True; name appears in the message.
    """
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(name + " must be a finite numeric declaration")
    result = Fraction(value)
    if result < 0 or positive and result == 0:
        raise ValueError(name + " has an invalid sign")
    return result


def _sqrt_upper(value):
    """Exact outward dyadic square root, without a rounded norm predicate."""
    scale = 1 << 64
    scaled = (
        value.numerator * scale * scale + value.denominator - 1
    ) // value.denominator
    root = isqrt(scaled)
    if root * root < scaled:
        root += 1
    return Fraction(root, scale)


@dataclass(frozen=True)
class TargetProximityRegion:
    """Resolve a requested COM radius above the complete occupied envelope.

    All-close selects the near tracking proposals; any-close selects observed
    pose geometry for the complete fleet. Neither selection admits motion.
    """

    requested_radius_m: Fraction
    target_vertex_radius_m: Fraction
    body_radius_m: Fraction
    target_clearance_m: Fraction
    com_standoff_m: Fraction
    position_support_m: Fraction
    geometry_id: str

    @classmethod
    def resolve(
        cls,
        geometry,
        *,
        requested_radius_m=30.0,
        body_radius_m,
        target_clearance_m,
        com_standoff_m,
        position_component_radius_m,
    ):
        """Build a region from the complete geometry and declared radii (all in m).

        Inputs: geometry SpacecraftGeometry; requested_radius_m target COM radius;
            body_radius_m chaser body radius; target_clearance_m; com_standoff_m;
            position_component_radius_m the three refuge position box radii.
        Outputs: a region whose target_vertex_radius_m is the outward square root of the
            largest squared body-frame vertex norm over all components and whose
            position_support_m is the outward square root of the summed squared position
            radii. Raises TypeError if geometry is not SpacecraftGeometry and ValueError
            for an invalid declaration or a position radius list not of length three.
        """
        if not isinstance(geometry, SpacecraftGeometry):
            raise TypeError("A proximity region needs complete SpacecraftGeometry")
        requested = _nonnegative(
            requested_radius_m, "Near target COM radius", positive=True
        )
        body = _nonnegative(body_radius_m, "Finite chaser body radius", positive=True)
        clearance = _nonnegative(target_clearance_m, "Target clearance")
        standoff = _nonnegative(
            com_standoff_m, "Attachment COM standoff", positive=True
        )
        position = tuple(
            _nonnegative(v, "Refuge position component radius")
            for v in position_component_radius_m
        )
        if len(position) != 3:
            raise ValueError(
                "A complete three-dimensional refuge position box is required"
            )
        occupied = max(
            sum(v * v for v in vertex)
            for component in geometry.components
            for vertex in component.vertices_body_m
        )
        return cls(
            requested,
            _sqrt_upper(occupied),
            body,
            clearance,
            standoff,
            _sqrt_upper(sum(v * v for v in position)),
            geometry.fingerprint,
        )

    @property
    def geometry_lower_bound_m(self):
        """Return the lower bound on the COM radius that clears the geometry (m).

        It is the vertex radius, plus the larger of body radius plus clearance and the
        COM standoff, plus the position support.
        """
        return (
            self.target_vertex_radius_m
            + max(self.body_radius_m + self.target_clearance_m, self.com_standoff_m)
            + self.position_support_m
        )

    @property
    def effective_radius_m(self):
        """Return the larger of the requested radius and geometry lower bound (m)."""
        return max(self.requested_radius_m, self.geometry_lower_bound_m)

    @property
    def comparison_radius_m(self):
        """Return effective_radius_m as a float, one ulp higher if below it."""
        value = float(self.effective_radius_m)
        return (
            nextafter(value, inf)
            if Fraction(value) < self.effective_radius_m
            else value
        )

    @property
    def legacy_default(self):
        """Return True when the requested and effective radii both equal 30 m."""
        return self.requested_radius_m == 30 and self.effective_radius_m == 30

    def as_dict(self):
        """Return the method, identities, radii and scope text as a dict.

        Radii are strings except comparison_radius_m, which is a float (m).
        """
        return {
            "method": "whole-occupied-envelope-com-proximity-v1",
            "geometry_id": self.geometry_id,
            "requested_radius_m": str(self.requested_radius_m),
            "target_vertex_radius_m": str(self.target_vertex_radius_m),
            "chaser_body_radius_m": str(self.body_radius_m),
            "target_clearance_m": str(self.target_clearance_m),
            "attachment_com_standoff_m": str(self.com_standoff_m),
            "refuge_position_support_m": str(self.position_support_m),
            "geometry_lower_bound_m": str(self.geometry_lower_bound_m),
            "effective_radius_m": str(self.effective_radius_m),
            "comparison_radius_m": self.comparison_radius_m,
            "scope": "COM-based proposal selection: all close for near tracking, any close for observed-pose geometry; every original dynamics, sensing, geometry, force and fuel proof remains required",
        }
