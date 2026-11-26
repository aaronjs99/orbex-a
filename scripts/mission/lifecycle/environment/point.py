"""point.py: Continuation adapter for the declared Cartesian J2/vector-drag environment.

Copyright: Aaron John Sabu
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from orbexa.mission.attached.assembly.model import positive
from orbexa.mission.contact.simulator import ContactBodyState
from orbexa.mission.lifecycle.nonlinear.contracts import _require
from orbexa.models.dynamics.reference_frame import FloatArithmetic
from orbexa.models.dynamics.relative import (
    NonlinearModelBounds,
    acceleration,
    parameters,
)


class NonlinearEnvironment:
    """The original point J2/vector-drag law, as extra acceleration beyond central.

    Default sigma=CdA/m remains the constant declared by the approach model.
    Optional explicit CdA values instead use each current dry+fuel mass. The
    radius domain extends to the declared delivery interface's numerical event
    bracket; the force law and inclined physical ECI axes are unchanged.
    """

    def __init__(
        self,
        model,
        *,
        target_mass_kg,
        chasers,
        drag_area_m2_by_name=None,
        minimum_altitude_m=80000.0,
    ):
        """Validate the model and fleet and store the continuation environment.

        Inputs:
            model: NonlinearModelBounds of the approach.
            target_mass_kg: target mass in kg.
            chasers: mapping of chaser name to a properties object with dry_mass_kg.
            drag_area_m2_by_name: optional explicit CdA in m^2 for every body,
                including "target".
            minimum_altitude_m: altitude in m above the body radius that lowers the
                lower end of the radius domain when it is smaller.

        Outputs:
            Sets model, parameters, dry masses, drag areas and a call counter.
            Raises ValueError when the model is untyped, the fleet is empty or
            includes "target", or the drag-area names differ from the bodies.
        """
        _require(
            isinstance(model, NonlinearModelBounds),
            "environment requires a typed point model",
        )
        self.original_model = model
        minimum = float(model.body_radius_m) + positive(
            minimum_altitude_m, "Continuation minimum altitude"
        )
        self.model = replace(
            model,
            radius_domain_m=(
                min(minimum, float(model.radius_domain_m[0])),
                model.radius_domain_m[1],
            ),
        )
        self.arithmetic, self.par = (
            FloatArithmetic(),
            parameters(model, FloatArithmetic()),
        )
        self.dry_mass = dict(
            target=positive(target_mass_kg, "Target mass"),
            **{key: value.dry_mass_kg for key, value in chasers.items()},
        )
        _require(
            "target" not in chasers and chasers,
            "environment needs the actual named fleet",
        )
        self.drag_area = (
            None if drag_area_m2_by_name is None else dict(drag_area_m2_by_name)
        )
        if self.drag_area is not None:
            _require(
                set(self.drag_area) == set(self.dry_mass),
                "explicit CdA values must name every body",
            )
            self.drag_area = {
                key: positive(value, key + " CdA", allow_zero=True)
                for key, value in self.drag_area.items()
            }
        self.calls = 0

    def __call__(self, time_s, bodies):
        """Return the extra acceleration and torque of every body at ``time_s``.

        Inputs:
            time_s: physical time in s.
            bodies: mapping of body name to ContactBodyState at that time.

        Outputs:
            Mapping of body name to acceleration_eci_m_s2 (full J2 and drag law
            minus the central term, in m/s^2) and a zero torque_body_nm. Counts the
            call. Raises ValueError for a non-finite time, changed body names, or
            states that are not current ContactBodyState values.
        """
        _require(
            np.isfinite(time_s) and set(bodies) == set(self.dry_mass),
            "environment body IDs or physical clock changed",
        )
        result = {}
        for key, state in bodies.items():
            _require(
                isinstance(state, ContactBodyState)
                and abs(state.time_s - time_s) < 1e-8,
                "environment needs current physical body states",
            )
            vector = tuple(np.r_[state.position_eci_m, state.velocity_eci_m_s])
            sigma = self.par["sigma_target" if key == "target" else "sigma_chaser"]
            if self.drag_area is not None:
                sigma = self.drag_area[key] / (self.dry_mass[key] + state.fuel_kg)
            full = np.asarray(
                acceleration(vector, self.model, self.par, self.arithmetic, sigma)
            )
            radius = np.linalg.norm(state.position_eci_m)
            central = -self.par["mu"] * state.position_eci_m / radius**3
            result[key] = {
                "acceleration_eci_m_s2": full - central,
                "torque_body_nm": np.zeros(3),
            }
        self.calls += 1
        return result

    def as_dict(self):
        """Return the environment record with its method, frame and scope."""
        return {
            "method": "same-declared-point-j2-vector-drag-minus-central-v1",
            "source_model": self.original_model.as_dict(),
            "source_model_fingerprint": self.original_model.fingerprint,
            "continuation_radius_domain_m": tuple(map(str, self.model.radius_domain_m)),
            "dry_mass_kg": self.dry_mass,
            "drag_area_m2_by_name": self.drag_area,
            "drag_semantics": (
                "declared constant CdA/m from approach"
                if self.drag_area is None
                else "explicit CdA divided by actual current dry plus fuel mass"
            ),
            "frame": "unchanged physical ECI basis; original inclined J2 and atmosphere rotation axes",
            "intrinsic_extra_torque": "zero; configured central gravity-gradient torque remains separate",
            "scope": "Same numerical force law through delivery; radius-domain extension is not atmospheric demise or flight qualification",
        }
