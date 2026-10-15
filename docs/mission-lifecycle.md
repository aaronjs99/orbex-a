# Physical mission lifecycle

## Certified terminal handoff

An optional `PhysicalTerminalHandoff` connects a successful physical terminal
execution to its exact saved approach. The adapter binds the original file and
normalized record, unchanged initial fleet states, exact shared anomaly event,
target/port/parameter context, and the complete command/flow/swept-geometry proof
chain. Refused or unrelated executions cannot supply a capture endpoint.

`CompleteMissionRunner.complete(approach, terminal_handoff=handoff)` preserves the
original MPC result. It uses an explicitly combined actual trajectory and target
kinematics only for attitude replay and conservative translation fuel accounting.
No MPC solve or membership update is invented during terminal feedback. Attachment
uses the literal final terminal states and reference-centered tube, with the
reference evaluation error added outward to both position and sliding envelopes.
The physical stages and saved lifecycle retain the compact handoff provenance.
This finite certified window remains conditional on its declared analytical
plant and target bounds; it does not certify a different orbital plant.

The paper's adaptive dynamic-tube MPC remains the approach controller. The
post-approach model adds qualified attachment, external-thruster detumbling,
and a finite-burn maneuver followed by a descending entry-interface handoff.
It uses explicit representative spacecraft parameters, rather than a measured
spacecraft, flight-qualified gripper, or validated disposal system.

## Interfaces and state conventions

`CaptureModel`, `DetumblingSimulator`, and `DeorbitSimulator` are independently
injectable stages. `MissionLifecycle` composes them, stops on failure, and keeps
the stage results. `MissionLifecycleStore` writes `lifecycle_data.json` with
schema `orbexa.physical-mission-lifecycle.v1`. No module selects an input path,
output path, observed spacecraft identity, or mission dataset.

`CompleteMissionRunner` adds upstream free-bus attitude control and translation
propellant accounting. It accepts an injected physical approach runner or a
saved successful approach. `CompleteMissionStore` reserves a new output directory
and records completed stages atomically. Expected stage exceptions and ordinary
rejected outcomes retain the last completed physical state and its history.

Positions and translational velocities after attachment use meters and meters
per second in a fixed inertial basis. That basis is chosen coincident with the
target's initial LVLH basis: it is a valid planar ECI coordinate choice, without
an inferred terrestrial longitude, orbit inclination, or absolute epoch. The
scalar-last quaternion maps the assembly body basis to that inertial basis.
Angular velocity is the inertial body angular velocity, expressed in body
components, in rad/s. Euler/anomaly derivatives are never treated as physical
spin.

Let `q` denote the recorded unwrapped anomaly, `ν` its physical rate in rad/s,
`C` the LVLH-to-ECI rotation, `x` the recorded relative position, and `x'` its
recorded anomaly derivative. Let `r_t` and `v_t` denote the target's physical
position and velocity. Let `k` be the positive orbit-normal unit vector. The
adapter uses the existing analytical target orbit and its actual truth clock:

$$
r_c=r_t+Cx,\qquad v_c=v_t+C\nu(x'+k\times x).
$$

The frame-rate term is required even for a stationary LVLH point. The recorded
elliptical model uses `x` opposite the transverse direction and `y` radially
outward. Let `r(q)` be orbital radius and `r'(q)` its anomaly derivative. The
target adapter therefore uses

$$
r_t=C(0,r,0)^T,\qquad v_t=C\nu(-r,r',0)^T.
$$

This radial-axis convention also defines the direction used for the physical
target's gravity-gradient torque. The recorded
physical target quaternion, body rate, anomaly rate, clock parameters, and
rotation frames must agree with the approach samples. Both torque-free and
gravity-gradient rigid target records are supported. Prescribed Euler motion
requires independently supplied `CaptureTruth`; it cannot silently become a
physical torque-free target. Assembly gravity-gradient torque inherits the
physical approach setting unless explicitly configured otherwise.

## Representative geometry and mass

The default target mass is 1000 kg, with body inertia
`diag(190, 190, 320)` kg m². Those values are consistent with a uniform solid
cylinder of radius 0.8 m and full length 0.6 m, but that agreement does not
identify an observed target. Mass and inertia are configurable and checked for
positive definiteness, the principal-moment triangle rule, and necessary bounds
imposed by the declared cylinder envelope.

Each default chaser has 8 kg dry mass and 4 kg fuel. In a direct `CaptureModel`
call this fuel is the **remaining amount at attachment**. The complete-mission
runner instead interprets it as initially installed fuel, subtracts a conservative
translation allowance and modeled free-bus attitude consumption, then passes
the remaining tanks to capture. No approach propellant is silently restored.
Its declared 0.150 m
spherical bus has the solid-sphere specific inertia `0.4 R² I`; wet inertia is
therefore `diag(0.108, 0.108, 0.108)` kg m². Fuel distribution is approximated as
proportional to the component's dry mass distribution. The 0.160 m inward
gripper offset is measured from its actual COM along the assigned outward
surface normal. The 0.010 m latch stroke bridges the remaining finite gap.

The approach position is a literal chaser COM by default. A COM docking
standoff of 0.165 m places the gripper approximately 5 mm from the surface and
leaves at least 15 mm nominal radial bus clearance. The old 5 mm **COM**
standoff cannot fit this bus and is rejected. A separately declared
capture-reference trajectory can be accepted only with
`assume_approach_point_is_capture_reference=True`; that option is an explicit
change of input semantics, never a reinterpretation of a recorded COM run.

## Attachment gates and conservation

The target is **unprepared**. Assigned surface points are planning coordinates,
not docking ports or hardware mounted on the target. The default attachment
interface is an idealized chaser-carried articulating tip. The COM follows its
finite-body standoff reference; the tip closes the remaining qualified reach
and then makes a rigid inelastic attachment. No part of this objective requires
the actual COM to reach its uncertainty-tube boundary.

Capture requires the approach's consecutive terminal dwell, successful mission
completion, recorded tube containment, physical safety, and docking phase. It
checks the **current** nominal center and componentwise position/sliding tube,
not a maximum radius from a future prediction horizon. The actual state must
belong to that tube. The entire current position tube must fit the known
surface-anchor latch reach, stay outside the surface, and leave finite bus and
pairwise bus clearance. The physical closing-speed bound includes the anomaly
clock, rotating LVLH basis, tube velocity bound, and target spin.

The default relative latch-speed limit is 0.001 m/s. The orientation and spin
gates default to 0.01 rad and 0.0001 rad/s. The existing approach controls
translation; the separate [chaser attitude controller](chaser-attitude.md) records
actual bounded nozzle couples, quaternion/spin evolution and fuel. The complete
runner requires that controller's successful final attitudes and rates. A direct
capture call without physical chaser attitudes refuses by default. Callers may
also explicitly set `assume_attitude_matched=True` for a labeled idealization.
The latter assumes exact alignment to each mounting frame and exact matching
of the physical target spin. It remains visible in the saved assumptions and
is not an attitude-control validation.

Let `a` be a selected bare-hull point, `n` its outward supporting normal and
`t₀` the actual tip before contact, all in target body coordinates. The remaining
tip correction is `a − t₀`, already bounded by the reach gate. The idealized
contact path uses a progress variable `ξ ∈ [0,1]`:

$$
t(\xi)=t_0+\xi(a-t_0),\qquad
n^T[t(\xi)-a]=(1-\xi)n^T(t_0-a)\ge0.
$$

The same supporting-plane argument holds for every tip in the admitted position
box. Thus the tip closes to the surface without entering the solid, while the
whole COM tube retains bus clearance. Saved `surface_contacts` record each
assigned point, pre-contact tip, correction vector, reach bound, final zero-gap
contact point and event clock. Contact and rigid attachment are coincident
idealized events; no finite deployment time, adhesive law or gripper hardware
performance is inferred. This is an explicit mission-planning abstraction.

Let `m_i`, `p_i`, `v_i`, `J_i`, and `ω_i` denote the component wet mass, centroid,
velocity, centroidal inertia in common body axes, and inertial spin. Let `M`
be total mass, `c` the combined COM, `V` its momentum velocity, and `d_i=p_i-c`
the component lever arm. All positions in the following expressions share
one inertial or body basis as appropriate. The composite inertia and its
pre-latch angular momentum about the combined COM are:

$$
M=\sum_i m_i,\quad c=M^{-1}\sum_i m_ip_i,\quad
V=M^{-1}\sum_i m_iv_i,
$$
$$
J=\sum_i\left[J_i+m_i(\|d_i\|^2I-d_id_i^T)\right],\qquad
H=\sum_i\left[J_i\omega_i+d_i\times m_i(v_i-V)\right].
$$

All intrinsic spin momenta, including the chasers', and all lever arms are
retained. The post-latch spin is `J⁻¹H`, after the explicit frame conversion.
The actual COM positions remain in place at the instant of latching; only
their velocities change through the ideal inelastic impulse. The reported
kinetic loss is pre-latch kinetic energy in the combined COM frame minus the
resulting rigid rotational energy. Linear and angular momentum residuals,
COM shift, and dissipated energy are recorded. Latch loads, impulse duration,
flexibility, arm inertia, and mechanical strength remain unresolved.

Whole-tube bus and pair gates are centered on the **recorded nominal COM**,
using the current component radii; actual-vehicle gates are checked separately.
Centering those boxes on an outward-displaced actual state could incorrectly
admit an inward unsafe corner. An explicitly declared tip-reference input also
uses the physical chaser rotation to recover the COM offset.

## External torque, force allocation, and fuel

`ThrusterSystem` represents signed, bounded channels as opposed nozzle pairs.
Each fueled chaser carries three translation channels and three two-nozzle
torque couples. Their directions follow that chaser's actual mounting frame.
The default translation-channel limit is 10 N, the torque-channel limit is
0.2 N, and the couple lever arm is 0.1 m. An explicit representative profile
may reduce those limits, for example to 5 N. These are **per-channel** limits, not a
guaranteed total bus force or a sourced hardware design. Default specific
impulse is 300 s. Continuous throttling, available pressure, clear exhaust,
and sufficient electrical/thermal support are assumptions.

Let `f_j` be a signed nozzle force, `a_j` its fixed unit direction, `b_j` its
position relative to the current COM, `Isp_j` its specific impulse, and `g₀`
the standard acceleration 9.80665 m/s². The allocator solves bounded weighted
least squares for the requested force and torque, then propagates the
**applied** wrench and absolute nozzle mass flow:

$$
F=\sum_j a_jf_j,\qquad \tau=\sum_j b_j\times a_jf_j,\qquad
\dot m_{\mathrm{fuel},j}=-\frac{|f_j|}{Isp_jg_0}.
$$

Allocation residuals, saturation, rank, tank depletion, and failures are
visible. Commands are uniformly reduced before a hold can consume unavailable
fuel. A custom actuator layout can be injected; a missing torque axis or
insufficient wrench rank fails explicitly. Reaction wheels are not treated as
an external angular-momentum sink.

Let `Q=(u,s)` denote the vector and scalar quaternion parts, `ω` the body rate,
and `τ_g` the optional gravity-gradient torque. The integrated rigid equations
during each frozen-mass substep are:

$$
\dot Q=\tfrac12(s\omega+u\times\omega,-u\cdot\omega),\qquad
J\dot\omega=\tau+\tau_g-\omega\times J\omega.
$$

Fuel depletion uses symmetric splits around each RK4 substep, recomputing mass,
COM, parallel-axis inertia, and force lever arms. Removed fuel carries the
local co-rotation angular momentum. The structure stays fixed while its COM
coordinate changes. This finite-step variable-mass approximation omits tank
flow, slosh and plume coupling; it is not a full open-system multibody model.
The angular audit separately records external torque impulse and the
co-rotating mass-removal contribution, rather than silently resetting spin or
claiming remaining-body angular momentum is conserved while fuel leaves.

`RateDampingPolicy` requests dissipative external torque `-Kω`. Its optional
`compensate_gravity_gradient` flag subtracts the known modeled gravity-gradient
torque. Composite-spacecraft completion enables this flag for both detumbling
and `ThrustPointingPolicy`. `AttachedAssembly.gravity_gradient_torque_body`
and the integrator share the same full-inertia expression, evaluated with the
current fuel-dependent mass properties. The legacy policy default remains
explicitly uncompensated; saved policy parameters identify each case.

Let `e_r^B` denote the unit radial direction expressed in the assembly body,
`r` its orbital radius, `μ` the central gravitational parameter, `J` the
current inertia, `K>0` the rate damping gain and `T` rotational kinetic energy.
The compensated request and the ideal continuously evaluated, frozen-mass
energy identity are

$$
\tau_g=\frac{3\mu}{r^3}\,e_r^B\times(Je_r^B),\qquad
\tau_c=-K\omega-\tau_g,\qquad
\dot T=-K\|\omega\|^2.
$$

Pure rate damping leaves `τ_g` as a persistent forcing term; near zero spin,
the scale `||τ_g||/K` can exceed the rate gate. Increasing the duration alone
does not remove this mechanism. Compensation assumes the declared orbit,
attitude and current mass properties are known. Allocation limits and the
actual held wrench remain authoritative; no continuous-feedback identity is
substituted for their numerical records. Within a hold the gravity-gradient
torque continues to evolve in the dynamics, while the commanded compensation
is held from the control endpoint. Fuel splitting and its co-rotation flux
remain unchanged and are audited separately.

The default completion gate remains a rate norm below `1e-5 rad/s` for 30 s,
within 1200 s. Dwell is measured at consecutive integration endpoints. It is
not a continuous-time enclosure. Both orbital translation and quaternion
attitude continue through this phase.

## Finite-burn deorbit and terminal event

The deorbit stage computes osculating orbital elements from the actual
assembly COM state. Let `r` and `v` denote that Cartesian state, `μ` Earth's
gravitational parameter, `ε` specific energy, `h` specific angular momentum,
`e` the eccentricity magnitude, `p` the semilatus rectum, and `r_p` perigee:

$$
\epsilon=\tfrac12v^Tv-\mu/\|r\|,\quad h=r\times v,\quad
e=\left\|\frac{v\times h}{\mu}-\frac{r}{\|r\|}\right\|,\quad
p=\|h\|^2/\mu,\quad r_p=p/(1+e).
$$

An apoapsis vis-viva calculation supplies an impulsive reference and an ideal
fuel lower bound for the planned strategy. It is not the simulated maneuver.
The stage coasts toward the next apoapsis, starts bounded quaternion pointing
control in advance, and maintains a physical bus `+x` axis near the retrograde
tangential direction. Thrust is enabled only inside pointing/rate tolerances.
The Cartesian state, variable mass, attitude, force allocation, fuel, and actual
perigee are integrated through the finite burn. No impulsive orbital velocity
reset is used.

Composite completion injects `AxialBurnPlanner`: the force direction is selected
from the installed nozzle directions and body axes using the actual assembly
mass center. A large projection of nozzle force onto an axis does not establish
that the same force and required torque can be delivered together. In particular,
chasers attached to one face can create a large force lever when thrusting
parallel to that face.

Let `W` denote the six-row installed force/torque matrix about the current COM,
`u` signed nozzle commands, `u_max` their installed limits, `d` a candidate unit
body direction and `s` the scalar axial force. The capacity problem is

$$
\max s\quad\text{subject to}\quad
Wu=(sd,0),\qquad |u|\le u_{\max},\qquad s\ge0.
$$

The planned request retains the existing 0.65 fraction of this coupled capacity.
A second LP minimizes total modeled fuel flow at that actual request, with all
six wrench equalities and the same channel bounds. Every candidate also passes
through the unchanged operational allocator at the planned force and control
hold; selection uses its realized flow per newton. The saved plan retains
capacity/minimum-flow witnesses, operational allocation, owner fuel rates and
residuals. These are numerical planning calculations, not interval proofs or
guarantees of future fuel balance. Pointing maps the selected **fixed body axis**
to the moving retrograde direction, and the same axis defines the requested
burn force. Bounded attitude motion, actual allocation, changing tanks/inertia,
pointing/rate gates and orbital events remain authoritative at every hold.

The nominal burn cutoff is the root of the actual osculating-perigee event at
80 km altitude. If fuel is exhausted after an entry-intersecting orbit has
already been reached, the model can coast to the interface and separately
report `target_perigee_achieved=False`. Otherwise it reports fuel failure.
Successful completion requires a descending crossing of the configured 120 km
entry interface, located by an event root. A powered-phase crossing is also
located as an event. Merely lowering altitude or perigee is not completion.

An initially entry-intersecting bound orbit can coast to that descending
interface without requiring unused burn authority or fuel. The entry endpoint
and the requested perigee target remain separate outcomes. If entry and burn
cutoff occur during the same held command, both roots are bracketed against the
original hold and the earliest physical event wins; entry wins a numerical tie.
The assembly is never propagated below the declared handoff before reporting it.

The 120 km interface is an explicit rounded model boundary. NASA's published
Starliner example uses 400,000 ft, or 121.92 km; an interface is a convention,
not a universal atmospheric edge. This model stops there. Atmospheric drag,
heating, breakup, demise, impact, geographic targeting, and public-risk
assessment are not modeled. The baseline orbit uses two-body gravity; J2 and
atmospheric perturbations are not included.

For the representative 1000 kg target orbit with eccentricity 0.18 and 300 km
initial perigee, the impulsive reference to an 80 km perigee is about 57.74 m/s.
A single 12 kg chaser carrying 4 kg fuel at 300 s specific impulse cannot
provide the approximately 19.67 kg ideal fuel need of that combined mass. The
single-chaser lifecycle therefore preserves a fuel-limit outcome. A fleet
result must be evaluated with the actual allocator and finite burn; an ideal
rocket-equation budget alone is insufficient.

## Use and records

```python
from functools import partial
from orbexa.simulation.capture import CaptureModel, CaptureOptions
from orbexa.simulation.lifecycle import MissionLifecycle, MissionLifecycleStore
from orbexa.simulation.propulsion import ThrusterSystem

engine = MissionLifecycle(
    capture=CaptureModel(options=CaptureOptions(assume_attitude_matched=True)),
    propulsion_factory=partial(ThrusterSystem.for_assembly,
                               translation_force_n=10., specific_impulse_s=300.),
)
result = engine.run(qualified_physical_approach)
MissionLifecycleStore.save(result, output_directory)
```

The explicit attitude assumption in this example is retained in the result.
`python -m orbexa.cli.complete_mission --help` exposes the configurable boundary.
Its default mode performs actual bounded chaser attitude replay and requires
`--input-component-bound` for conservative approach propellant accounting.
`--assume-attitude-matched` explicitly selects the idealized direct-capture
example above. Both modes preserve the assumption in their output.

`assembly_history` contains ordered SI state records with phase, time, COM
position/velocity, quaternion, body rate, component fuel, current mass, COM
offset, inertia, angular momentum, and rotational energy. Deorbit records add
osculating elements and altitude. Stage control records contain requested and
applied wrenches, each nozzle command, tank flow, and angular audits. The
assembly description supplies component centroids and mounting rotations.
`AttachedAssembly.component_states(state)` supplies physical component poses
for renderers using the current COM offset, without duplicating a dynamics
model in the UI. Configuration and declared assumptions are serialized.

Synthetic endpoint fixtures can exercise latch conservation, allocation,
detumbling, finite burns, failures and terminal events. Their qualification
provenance must say that they are synthetic. They do not establish a successful
ADTMPC approach. Repository instructions keep such engineering probes and
generated outputs outside the maintained source tree.

## Primary references

- [NASA/JPL, Fundamentals of Orbital Mechanics, chapter 7](https://spsweb.fltops.jpl.nasa.gov/portaldataops/mpg/MPG_Docs/MPG%20Book/Release/Chapter7-OrbitalMechanics.pdf): Cartesian orbital invariants and vis-viva relations.
- [NASA Glenn, Ideal Rocket Equation](https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/ideal-rocket-equation/): ideal mass-ratio and velocity relation used for a budget reference.
- [NASA, Small Spacecraft In-Space Propulsion](https://www.nasa.gov/smallsat-institute/sst-soa/in-space_propulsion/): thrust, specific impulse and propulsion tradeoffs; this does not validate the representative bus/nozzle layout.
- [NASA NTRS, Space Attitude Determination and Control](https://ntrs.nasa.gov/api/citations/20240009554/downloads/Space%20Attitude%20Development%20Control.pdf): rigid-body rotational dynamics and attitude-control context.
- [NASA, Starliner Begins Re-entry](https://www.nasa.gov/blogs/commercialcrew/2024/09/06/nasas-coverage-continues-starliner-begins-re-entry/): the example 400,000 ft entry-interface convention.
- [NASA Orbital Debris Program Office, Debris Reentry](https://orbitaldebris.jsc.nasa.gov/reentry/): atmospheric breakup and survivability are distinct analyses below an entry-interface handoff.
