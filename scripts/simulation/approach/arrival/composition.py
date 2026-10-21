"""Prospective spin-specific arrival composition from declared physical bounds."""
from __future__ import annotations

import gzip
import json
import math
from fractions import Fraction
from orbexa.simulation.recording.artifacts import _write_json


def _arrival_policy(profile, prior, admission, directory):
    """Derive final-port query errors for this job's exact spin and domain.

    The global controller may refuse; only its globally derived reference jets
    are used here as a planning/entry screen, never as terminal certification.
    """
    from orbexa.control.terminal.composite import CompositePhysicalTerminalBoundsBuilder
    from orbexa.control.terminal.physical import PhysicalTerminalErrorBounds, _encode
    from orbexa.control.certification.reachability import IntervalBox, OrbitalModelBounds
    from orbexa.simulation.approach.arrival.policy import ArrivalPolicyConfig
    target, cfg = profile.target, profile.config
    chasers = profile.scenario.fleet.chasers("multi")
    ports = tuple(tuple(c.docking_point_body) for c in chasers)
    index = max(range(len(ports)), key=lambda i: sum(Fraction(float(v))**2 for v in ports[i]))
    common = dict(mu=target.mu, semi_major_axis=target.semi_major_axis)
    orbit = OrbitalModelBounds(eccentricity=(target.eccentricity,)*2, alpha=(target.alpha,)*2, beta=(0, 0), **common)
    plant = OrbitalModelBounds(**prior.feasible_sets, **common)
    nominal = OrbitalModelBounds(**{key: (value, value) for key, value in prior.estimates.items()}, **common)
    errors = PhysicalTerminalErrorBounds(sensing_error_abs=admission.measurement_radius,
        reference_state_error_abs=(1e-6,)*3+(1e-4,)*3, anomaly_evaluation_error_abs=1e-10,
        declaration="Caller-declared uniform direct reference evaluation errors1e-6m/1e-4m per rad, sensor bounds from admission and initial-spin containment1e-8rad/s. Arb derives only finite-domain query-displacement inflation. No flight metrology or DOP853 tolerance certificate.")
    builder = CompositePhysicalTerminalBoundsBuilder(start_anomaly=target.anomaly_domain[0],
        end_anomaly=target.anomaly_domain[1], target_orbit=orbit, plant_parameters=plant, nominal_parameters=nominal,
        inertia=target.inertia, entry_spin=IntervalBox.from_center_radius(target.initial_body_angular_velocity, (1e-8,)*3),
        body_ports=ports, port_index=index, errors=errors, reference_id="live-global-final-port-errors",
        target_geometry=target.geometry, body_radius=profile.options.body_radius_m,
        maximum_hold=cfg.anom_step/cfg.tube.feedback_substeps,
        input_limits=IntervalBox(tuple(b['lower'] for b in cfg.mpc.input_bounds), tuple(b['upper'] for b in cfg.mpc.input_bounds)),
        bandwidth_candidates=(80,), precision_bits=192)
    result = builder.build()
    if not result.derived or "effective_reference_state_error_abs" not in result.derived:
        raise ValueError("Spin-specific global reference-error derivation refused: " + result.reason)
    rounded = []
    for value in result.derived["effective_reference_state_error_abs"]:
        number = float(value)
        if Fraction(number) < value:
            number = math.nextafter(number, math.inf)
        if not math.isfinite(number) or Fraction(number) < value:
            raise ValueError("Cannot represent the outward reference-error bound")
        rounded.append(number)
    payload = dict(mode="deadline_terminal_suffix", planning_fraction=.8, required_terminal_region=dict(
        omega=[.00115]*3, phi=[.02]*3, lambda_gain=[20]*3,
        reference_state_error_abs=rounded, maximum_query_displacement=1e-10,
        anomaly_domain=list(target.anomaly_domain), frame="physical-lvlh-v1",
        error_source=errors.declaration + " Full-domain rigid-body energy/orbital Arb bounds at the longest exact-norm final COM port dominate all eight port query jets; the global physical-controller certificate is not claimed."))
    _write_json(directory / "arrival-policy.json", payload)
    with gzip.open(directory / "reference-error-derivation.json.gz", "wt", encoding="utf-8") as stream:
        json.dump(_encode(dict(builder=vars(builder), result=result.as_dict(), selected_port=index,
            all_ports=ports, scope="Global final-port planning error bound only; controller refusal retained")), stream, allow_nan=False)
    return ArrivalPolicyConfig.from_dict(payload)
