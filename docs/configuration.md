# Configure a mission

Envisat is the default study, not the controller's target type. A mission uses
three explicit files: an orbit/controller YAML, a spacecraft JSON, and a study
JSON. Give each run a new output directory outside the checkout.

```bash
python run.py --config my-orbit.yaml --spacecraft my-spacecraft.json --study my-study.json --case central --adaptation adaptive --bandwidth dynamic --output-dir ../orbexa-output/my-target
```

## Which file controls what

| Input | Declare |
|---|---|
| Orbit/controller YAML | Fleet count, reference orbit, MPC input limits, chaser separation, target clearance and solver settings. Use `target.shape: composite_spacecraft`. |
| Spacecraft JSON | Oriented box enclosures for every occupied component, their uncertainties, attachment patches, target mass/inertia and chaser hardware. Include solar arrays and appendages. |
| Study JSON | Deployment, tube gains/bandwidth, horizons, parameter priors, sensing, actuator uncertainty, observation windows, contact/disposal settings, update cap and independently selected physical cases. |

The study's named fields override corresponding controller defaults. Resolved
values, source bytes and input hashes are recorded before motion. Changing the
model requires new evidence; a previous target's result does not qualify it.

## Spacecraft and fleet

The existing `config/spacecraft/envisat.json` is a geometry-only declaration
that retains the original study vehicle defaults. For another vehicle use
the explicit wrapper below. `geometry` is a complete `SpacecraftGeometry`
record with the same structure as that file; it is an object, not a file path.

```json
{
  "schema": "orbexa.spacecraft-mission.v1",
  "description": "Identify the target, geometry sources and sizing assumptions",
  "geometry": {},
  "vehicle_design": {
    "chaser": {
      "dry_mass_kg": 30,
      "fuel_kg": 100,
      "body_radius_m": 0.55,
      "gripper_offset_m": 0.8
    },
    "com_standoff_m": 0.85,
    "arm_reach_m": 0.10,
    "translation_force_n": 100,
    "attitude_nozzle_force_n": 3,
    "attitude_moment_arm_m": 0.4,
    "specific_impulse_s": 300,
    "detumbling_gain_nm_s": 2000,
    "pointing_attitude_gain_nm": 500,
    "pointing_rate_gain_nm_s": 8000
  }
}
```

Replace the empty `geometry` with its complete components and patches; an empty
object is deliberately invalid. The numbers above explain the existing vehicle
fields, not appropriate sizing for another mission. Target mass and inertia
belong in `geometry.metadata.mass_properties`, as `nominal_mass_kg` and the
positive-definite 3 × 3 `inertia_kg_m2` matrix.

Set `num_chasers` in the YAML. The target must supply at least that many valid
patches. There is no fixed eight-agent override. The planner checks every
declared component and peer. The study currently uses one hardware design for
all chasers; heterogeneous vehicles require a per-agent design interface.

## Deployment, spin and uncertainty

The short form keeps a common payload cluster: `options.release_distance_m`
and `options.release_pitch_m`. For an off-axis carrier, selected patches or
different route timing, supply `options.deployment` using the named fields of
`CompositeDeploymentOptions`. Its `carrier_center_lvlh_m` is required. When
this object is present it controls deployment; the short-form distance and
pitch are unused. Reference speed, acceleration and tube settings schedule a
route whose feasibility still requires the controller and admission checks.

`options.nominal_body_spin_rad_s` accepts three body-axis components. Each
physical case can supply `body_spin_rad_s`. If omitted, the original
`nominal_spin_deg_s` / `target_spin_deg_s` z-axis settings apply. Declare an
initial spin box with `initial_target_spin_prior.center_rad_s` and
`radius_rad_s`; it must cover the selected truth within
`options.target_rate_bound_rad_s`.

Other exposed study fields include:

- `rendezvous_horizon_steps`, `docking_horizon_steps` and
  `terminal_activation_distance_m`.
- `controller_parameter_prior` with `eccentricity`, `alpha` and `beta` intervals;
  `target_sigma_prior_m2_kg` and `chaser_sigma_prior_m2_kg` for physical drag.
- `initial_target_orientation_error`, `target_rate_bound_rad_s` and
  `target_acceleration_bound_rad_s2`.
- The complete `sensor` record and command-dependent actuator error bounds.
- Linked identification's finite history, nuisance-cell and flow budgets.
- `near_feedback_subdivisions`: a positive integer dividing the base feedback
  hold near the target. The actual executor obeys the shorter certified hold;
  this is a different control rate, not additional interpolation of a movie.
- Optional `far_contingency`, with `coast_holds`, `transition_holds` and
  `acceleration_limit_m_per_rad2`, adds an acceleration-prefix proposal before
  the existing braking candidates. It uses the projected derivative enclosure,
  never a future observation. Every original entry, nonlinear, geometry,
  force and fuel gate still applies. Leaving it absent preserves the original
  search; a longer unresolved forecast can retry the shorter original search.

## Contact, detumbling and disposal

Use `options.completion` in the study JSON to change stage policies without
changing spacecraft hardware. Missing fields retain the typed stage defaults;
unknown fields and invalid physical values refuse setup.

```json
{
  "contact": {
    "handoff_rule": "load_bounded_finite_lock",
    "max_duration_s": 60,
    "max_step_s": 0.005,
    "lock_duration_s": 0.25
  },
  "detumbling": {
    "rate_tolerance_rad_s": 0.00001,
    "dwell_s": 30,
    "max_duration_s": 1200,
    "control_step_s": 1
  },
  "deorbit": {
    "target_perigee_altitude_m": 80000,
    "entry_interface_altitude_m": 120000,
    "control_step_s": 2,
    "coast_sample_step_s": 30,
    "max_burn_s": 7200,
    "max_coast_orbits": 2
  }
}
```

The names above are the public option names of the corresponding stage
dataclasses. Hardware force, moment arm and specific impulse remain in
`vehicle_design`; the contact policy cannot override them. Numerical contact
substeps and interval-proof partitions resolve a held command without changing
its sensor or controller frequency. A longer allowed stage duration is a
budget, never evidence that its acceptance gates passed.

Disposal ends at a descending entry-interface handoff. It does not establish
atmospheric breakup, demise or a safe ground footprint. A different target
also needs valid patches, mass/inertia and load capacities; selecting another
mesh cannot qualify the new mission.

The maintained environment is Earth central gravity, inclined J2 and a declared
co-rotating exponential atmosphere. Configurable orbit and target properties
do not establish arbitrary celestial-body, flexible-appendage or unmodeled
contact dynamics. A solid-component envelope is not a flexible-body model.

## Current evidence

Two distinct synthetic spacecraft declarations, with two and three chasers,
have initialized and installed complete initial nonlinear finite backups with
their own target mass/inertia and propulsion. These are interface and initial
proof results. Their full approach, attachment, detumbling and disposal
missions remain unverified. See [mission readiness](mission-readiness.md).
