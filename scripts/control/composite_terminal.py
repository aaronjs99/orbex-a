"""Physical terminal bounds for an explicitly declared spacecraft part union.

Orbital, rigid-body spin and held-feedback inequalities retain the existing
analytical derivation.  Geometry is separately proved against every occupied
component; the cylinder service is never called for a composite target.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .composite_geometry import CompositeTerminalGeometry, _geometry_context
from .physical_terminal import PhysicalTerminalBoundsBuilder


COMPOSITE_TERMINAL_METHOD = "arb-composite-orbit-rigid-body-terminal-v1"


@dataclass(frozen=True)
class CompositePhysicalTerminalBoundsBuilder(PhysicalTerminalBoundsBuilder):
    """One reference/body port and one explicit complete spacecraft geometry.

    Inherited cylinder radius/height fields are inactive compatibility metadata;
    no composite exclusion, distance or certificate depends on those values.
    The actual rigid target and peers must share ``target_geometry`` and the
    same true rigid rotation throughout the declared finite window.
    """

    target_geometry: Any = None

    def __post_init__(self):
        super().__post_init__()
        _geometry_context(self.target_geometry)

    def _geometry_distance(self, ar, point):
        service = CompositeTerminalGeometry(self.target_geometry, precision_bits=self.precision_bits)
        certificate = service.point_separation(point)
        if not certificate.admissible:
            raise ValueError("Composite reference geometry refused: " + certificate.reason)
        distance = certificate.distance_lower - self.errors.target_solid_position_error_radius
        if distance <= 0:
            raise ValueError("Composite reference has no strictly positive proved distance to the complete occupied union")
        return distance, dict(target_geometry=service.context,
            target_geometry_fingerprint=service.fingerprint, target_geometry_distance=certificate.as_dict(),
            cylinder_geometry_used=False)

    def _geometry_proof_reference(self):
        return COMPOSITE_TERMINAL_METHOD

    def _spin_bound(self, ar, entry_norm, duration, quadratic, constant, jmin):
        # E=.5 omega^T J omega and |E'|<=|omega|*tau_max. For
        # V=sqrt(2E/Jmin), |omega|<=V and |V'|<=tau_max/Jmin.
        # The comparison at V=0 follows by positive regularization and its
        # zero-limit, not division by a zero energy. Both time directions are
        # covered, including the expanded reference-query interval.
        omega = ar.vector(self.entry_spin)
        energy_twice = sum((omega[i] * ar.scalar(self.inertia[i][j]) * omega[j]
                            for i in range(3) for j in range(3)), ar.arb(0))
        quadratic_upper = ar.box((energy_twice,)).upper[0]
        if quadratic_upper < 0:
            raise ValueError("The validated entry-energy upper bound cannot be negative for physical inertia")
        energy_speed = ar.box((ar.scalar(quadratic_upper / jmin).sqrt(),)).upper[0]
        energy_spin = energy_speed + constant * duration
        spin, discriminant, margin, selected = energy_spin, None, None, "energy_comparison"
        branch = dict(status="UNRESOLVED")
        try:
            first_exit, first_discriminant, first_margin, _ = super()._spin_bound(
                ar, entry_norm, duration, quadratic, constant, jmin)
            branch = dict(status="PROVEN", spin_norm_upper=first_exit,
                          discriminant=first_discriminant, self_inclusion_margin=first_margin)
            if first_exit < spin:
                spin, discriminant, margin, selected = first_exit, first_discriminant, first_margin, "scalar_first_exit"
        except ValueError as error:
            branch["reason"] = str(error)
        return spin, discriminant, margin, dict(spin_bound_method="minimum_of_energy_and_validated_scalar_first_exit",
            entry_twice_energy_upper=quadratic_upper, entry_energy_speed_upper=energy_speed,
            energy_speed_rate_upper=constant, energy_comparison_duration_upper=duration,
            energy_spin_norm_upper=energy_spin, selected_spin_bound=selected,
            scalar_first_exit_comparison=branch,
            zero_energy_scope="Positive regularization and its zero-limit; no division by zero energy",
            first_exit_branch_required=selected == "scalar_first_exit")

    def build(self):
        return replace(super().build(), method=COMPOSITE_TERMINAL_METHOD)
