# Compact post-release free-flight deployment

`CompactArcDeployment` adds an optional finite compact start to the maintained
nonlinear ADTMPC mission. All chasers begin after separation and acquisition of
the modeled exact state feedback. Their initial positions lie on a short arc
near one common target-centered direction. They spread to a full ring, contract
to a rotating staging shell, and approach their explicitly assigned side points.
The ordinary `MissionScenario` and default YAML retain their previous start and
rendezvous-to-docking policy.

This initializer models a free-flight handoff, rather than carrier ejection,
recoil, release ports or navigation acquisition. `PhysicalApproachScenario`
adds finite bus volumes, physical rigid-target motion and optional continuous
held-motion admission; the post-approach lifecycle adds actual nozzle, latch
and assembly models. Their assumptions remain explicit. Exact identical free-flyer positions are rejected;
minimum separation and every existing constraint remain active.

## Composition and use

The following example uses an editable installation of this checkout, runs from
its root, and saves results outside the repository. The explicit controller prior
is copied and validated separately from plant truth. Supplying a prior never
recenters or enlarges it from the plant. Membership consistency still depends on
the truth/model and measurement assumptions; an inconsistent prior may fail.

```python
from pathlib import Path
from orbexa.core.config import SimulationConfig
from orbexa.estimation.belief import ParameterBelief
from orbexa.simulation import (
    ADTMPCMissionRunner, CompactArcDeployment, CompactDeploymentOptions,
    MissionResultStore,
)

config = SimulationConfig.load("config/default.yaml")  # num_chasers = 8
prior = ParameterBelief(
    feasible_sets={
        "eccentricity": (0.14, 0.22),
        "alpha": (1.5e-7, 2.5e-7),
        "beta": (3.5e-7, 5.5e-7),
    },
    estimates={"eccentricity": 0.17, "alpha": 1.8e-7, "beta": 4.2e-7},
)
deployment = CompactArcDeployment(config, options=CompactDeploymentOptions(
    separation_buffer=0.10,
    fanout_duration=3.0,       # radians of unwrapped true anomaly
    contraction_duration=1.5,
    staging_radius=3.0,       # meters from target center
))
scenario = deployment.create_scenario(initial_belief=prior)
result = ADTMPCMissionRunner(scenario=scenario).run(mission="multi", steps=500)
MissionResultStore.save(result, Path.home() / "orbexa-output" / "compact-eight")
print(result.success, result.message)
```

The update count is a run budget, not a completion promise. Use a valid
configuration with another `num_chasers` for a smaller fleet. `mission="single"`
requires a deployment constructed with one chaser; `mission="multi"` uses the
whole configured fleet. Plant truth, supported target geometry, safety policy,
point estimator and controller factory remain injectable through the deployment
constructor or its `create_scenario` method. A custom target must implement the
supported centered-cylinder collision model and exact rigid-point kinematics.

`CompactDeploymentOptions` is immutable and validates its durations, radii,
buffer and acceptance tolerances. `launch_radius=None` uses the norm of the
configured initial position, about 6.1814 m with the default scenario.
`launch_azimuth=None` projects its direction into the target's initial equatorial
plane. A direction on the target axis needs an explicit azimuth. A caller may
inject `CompactDeploymentFleet` or `CompactDeploymentPolicy` directly into
`MissionScenario`; both share one deployment object so count, ordering and timing
stay consistent. The fleet's public construction method also rejects invalid,
coincident or insufficiently separated supplied release states.

The common launch region is finite. With eight default chasers and a 0.10 m
buffer, the neighboring pitch is 0.45 m, the arc spans about 0.5097 rad, and the
extreme chord is about 3.1167 m. Initial anomaly velocities match the selected
target rotation, which may be prescribed or physical. They are derivatives in m/rad and are not physical release
velocities in m/s. No initial navigation-error tube is added by this initializer.

## Reference geometry and derivatives

Let `q` denote unwrapped true anomaly in radians and let a prime denote its
derivative. Let `N` be the fleet size, `i` its index from zero through `N-1`,
`c_i = i-(N-1)/2` its centered order, and `theta_c` the common body-frame angular
origin. Let `R_L` and `R_S` denote the positive launch and staging radii in meters,
`T_f` and `T_r` the positive fan-out and contraction durations in radians, and
`Q(q)` the common proper rotation from target body coordinates to LVLH. Define
`d_min` as the unchanged minimum center spacing and `b` as the prospective extra
initial buffer, both in meters. Define the neighboring initial and final angular
spacings, `Delta_0` and `Delta_f`, by

```text
Delta_0 = 2 asin((d_min+b)/(2 R_L)),    Delta_f = 2 pi/N.
```

The constructor requires a positive initial pitch and `Delta_0 <= Delta_f` for
more than one chaser. It rejects a launch arc, staging ring or assigned docking
ring that cannot satisfy its geometric minimum spacing. It never expands the
launch radius or reduces `d_min` to repair a requested packing.

Let `tau` be normalized progress clamped to `[0,1]`, and let `h` be its quintic
blend. Within the open interval, its derivatives with respect to `tau` are

```text
h(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5
h_tau  = 30 tau^2 (1-tau)^2
h_tautau = 60 tau (1-tau)(1-2 tau).
```

Outside that interval the blend is constant and both derivatives are zero.
For a duration `T`, anomaly derivatives introduce factors `1/T` and `1/T^2`.
Let `Delta(q)` be angular spacing, `r(q)` radial distance, and `theta_i(q)` the
body-frame angle of chaser `i`. The complete reference clock is

```text
Delta(q) = Delta_0 + (Delta_f-Delta_0) h(q/T_f)
r(q) = R_L + (R_S-R_L) h((q-T_f)/T_r)
theta_i(q) = theta_c + c_i Delta(q).
```

Each blend uses the clamping defined above. Thus fan-out occurs at constant
launch radius, radial contraction begins with a complete ring, and both motions
stop relative to the rotating target with zero first and second endpoint
derivatives. The target may continue rotating.

Let `e_i` be the planar radial unit vector at `theta_i` and `t_i` its tangent.
Let `b_i`, `b_i'`, `b_i''` be the body-coordinate position and its anomaly
derivatives, and let `p_i`, `p_i'`, `p_i''` be the corresponding LVLH quantities.
The implementation evaluates the exact chain rule, including common rotation:

```text
e_i = [cos(theta_i), sin(theta_i), 0]
t_i = [-sin(theta_i), cos(theta_i), 0]
b_i = r e_i
b_i' = r' e_i + r theta_i' t_i
b_i'' = (r''-r theta_i'^2) e_i + (2 r' theta_i'+r theta_i'') t_i
p_i = Q b_i
p_i' = Q' b_i + Q b_i'
p_i'' = Q'' b_i + 2 Q' b_i' + Q b_i''.
```

`Q`, `Q'` and `Q''` are obtained by evaluating the target's exact rigid-point
state and acceleration APIs on the three unit basis vectors. This reuses the
same rotation convention and clock as target exclusion and docking references.
It introduces no numerical differentiation of the reference.

Let `A(q)` be the nominal six-state dynamics matrix, `B` the input matrix and
`d(q)` the additive nominal disturbance. Let `x_ref = [p_i,p_i']` and
`x_ref' = [p_i',p_i'']` be the reference state and derivative. The existing
tracking-reference sampler forms midpoint holding inputs using

```text
u_ref(q) = B^dagger (x_ref'(q) - A(q) x_ref(q) - d(q)).
```

These inputs are in m/rad². The unchanged nonlinear planner retains input bounds,
ancillary reserves, uncertainty propagation, pair constraints and target
exclusion. Duration selection and reference geometry do not certify those
constraints or actual trajectory feasibility.

## Reference spacing argument

During fan-out, `Delta` grows monotonically from `Delta_0` to `2 pi/N`. The first
`N-1` cyclic gaps equal `Delta`; the remaining gap equals
`2 pi-(N-1)Delta` and is at least `2 pi/N`. Every cyclic gap is therefore at least
`Delta_0`. The shorter angular separation of any two distinct points is at least
`Delta_0` and at most `pi`. Its chord is at least
`2 R_L sin(Delta_0/2) = d_min+b`. A common proper rotation preserves that distance
and target-center range.

During radial contraction the ring is uniform, with neighboring chord
`2 r(q) sin(pi/N) >= 2 R_S sin(pi/N)`. Across the whole reference, pair distance
is consequently at least the smaller of the initial pitch and this staging
chord. This argument concerns reference point centers. Actual nominal paths,
their two tubes, finite physical bodies, continuous-time separation and recursive
feasibility are not established by it. The actual planner and sampled executor
retain their own checks. A one-chaser deployment has no pairwise spacing claim.

## Phase ownership and docking readiness

`MissionReferencePolicy.initial_phase()` selects the first phase;
`select_phase(phase, state, chasers)` sees the current committed actual states,
nominal centers, component tubes and shared anomaly. The base implementation
preserves the existing range/velocity rendezvous transition. The compact policy
starts in `deployment`, changes to `staging` after fan-out, and selects `docking`
only after the reference clock has finished and all readiness margins are
nonnegative. Deployment and staging use `operation="rendezvous"`, retaining
the target bounding-sphere exclusion and all fleet constraints. Docking uses the
existing rotating-cylinder side corridors. Initial samples record the selected
phase rather than assuming rendezvous.

Let `z`, `x`, `Omega` and `Phi` denote a chaser's current nominal center, actual
state, component position envelope and component sliding envelope. Let `Lambda`
be the configured diagonal sliding gain, and `n` its unit assigned side normal
rotated to LVLH at the current anomaly. Define `rho = norm(Omega)` as the solver's
Euclidean tube inflation. The gate requires the actual state and complete
nominal component box to track the moving shell within the configured position
and anomaly-velocity tolerances. For a nominal error `e` and component box `a`,
its exact farthest Euclidean error is `norm(abs(e)+a)`. The velocity box is
`Phi+Lambda Omega`.

The current side support must satisfy
`n @ z_position >= radius*(1+tolerance)+rho+target_barrier_clearance`, matching
the solver's stronger Euclidean tube inflation and enclosing the nominal box.
The gate also checks actual and nominal-tube target exclusion, actual fleet
spacing, and nominal fleet spacing inflated by both tubes. Incomplete fleets,
nonfinite states or negative envelopes are rejected. `staging_margins` exposes
each gate margin for outside diagnostics.

A successful gate permits a docking solve. A later failed solve remains a
recorded failure; this policy does not add a physical abort or staging backup.
Whole-tube terminal acceptance and consecutive endpoint dwell remain active.
The shared terminal bound now uses the exact component-box norm described above
for both position and velocity. Reaching a 5 mm point-center standoff is still
a free-flight result, with the model and evidence limits in
[the ADTMPC derivation](adtmpc-design.md).

Deployment configuration, resolved geometry, assignment strategy, timing and
scope are supplied through `MissionScenario.scenario_metadata()` for result
recording. The actual assigned starts are retained in the existing per-chaser
assignment metadata. An explicit controller prior is copied again at each run
so adaptation does not mutate the scenario's starting belief.
