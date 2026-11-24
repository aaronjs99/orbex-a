# Scenario inputs

The current sensed nonlinear mission uses three explicit inputs:

- `envisat.yaml`: orbital constants, MPC cost/constraints, native GEKKO/IPOPT and execution grid.
- `spacecraft/envisat.json`: eleven occupied components, eight attachment patches, vehicle mass and propulsion.
- `nonlinear-study.json`: the complete prospective profile and independently selected truth/noise cases, crossed with adaptive/frozen belief and dynamic/fixed bandwidth.

`NonlinearStudyOptions.controller_config` explicitly resolves the current tube,
sensing and terminal settings over the base YAML. Its full resolved mapping is
saved before motion. The profile uses one 300 m payload cluster, 1.8 m spacing,
a fixed reference clock and a common 1000/rad initial/fixed bandwidth. The
terminal COM position bound is 40 mm; the separate physical relative-speed gate
is 10 mm/s. A 100 mm mechanism closes the remaining contact gap.

The initial target spin prior is nonzero and independent of the selected truth.
Physical sigma bounds, processed sensor errors/latency/dropout, actuator gain/tilt/
bias and every case's noise seed are declarations. They are not measured Envisat
telemetry or hardware accuracy. See [the nonlinear contract](../docs/nonlinear-mission.md).

Each chaser is 130 kg wet (30 kg dry, 100 kg fuel), with a 0.55 m body radius,
0.85 m COM standoff, 0.8 m tip offset, 100 N translation authority and 300 s
specific impulse. The target mass remains 7827.867 kg. Route construction,
finite-body separation, force/fuel, arm reach, contact load and disposal gates
may refuse a run. A resource cap is not completion.

`near_target_radius_m` optionally declares a positive minimum COM distance for
near proposals (30 m by default). The planner raises it when necessary to cover
every expanded target component, the larger of attachment COM standoff or finite
bus radius plus clearance, and the largest refuge position box. All members must be inside that radius for
near tracking; one member inside requests observed-pose geometry for the whole
fleet. The default Envisat radius remains 30 m. This selects proposals; changing
target geometry, mass, inertia or spin still requires every original dynamics,
sensing, force, fuel, contact and whole-set collision check. It does not establish
arrival for every target.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `envisat.yaml` | Orbital constants, MPC cost and constraints, native GEKKO/IPOPT settings and the execution grid. | `models/configuration` loads it | `run.py`, `serve.py`, `orbexa-run-nonlinear`, `orbexa-serve-studio` |
| `nonlinear-study.json` | The prospective study profile: explicit options and the independently selected truth and noise cases. | `studies/request.py` validates it | `run.py`, `orbexa-run-nonlinear` |
| `spacecraft/` | The complete vehicle geometry JSON: components, attachment patches, mass and propulsion. | `models/geometry` | `run.py`, `serve.py`, `mission/lifecycle/configuration` |

