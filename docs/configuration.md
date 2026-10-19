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
| Study JSON | Deployment, tube gains/bandwidth, horizons, parameter priors, sensing, actuator uncertainty, observation windows, update cap and independently selected physical cases. |

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
