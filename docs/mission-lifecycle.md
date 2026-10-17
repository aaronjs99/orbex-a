# Physical mission lifecycle

## Current nonlinear handoff

The ordinary [sensed nonlinear mission](nonlinear-mission.md) continues through
`complete_nonlinear_mission`. It requires a successful saved approach, timely
whole-set terminal entry, consecutive accepted terminal dwell and the complete
source-bound admitted-hold journal. An incomplete prefix or failed terminal gate
cannot authorize attachment. The adapter supplies the literal executed endpoint,
original reference clock, remaining tanks and declared design to
[finite compliant capture](finite-contact.md).

The target and all eight buses move through finite deployment, compliant contact,
adhesion and settling before rigid assembly. Actual poses continue into external
thruster detumbling, finite burn and coast. A failed stage stops continuation;
successful delivery requires the actual descending entry-interface event.
These are representative numerical spacecraft and pad models, not measured
Envisat material properties, a flight-qualified gripper or a disposal approval.

The nonlinear adapter preserves the same central/J2/co-rotating vector-drag law
through contact and assembly, including the original inclined physical axes.
It supplies acceleration beyond central gravity through the environment callback;
configured gravity-gradient torque remains separate. A direct lifecycle without
that callback uses its declared central-gravity baseline.

## Optional analytical terminal handoff

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

This optional analytical reproduction path does not supply nonlinear mission
evidence. Its attachment stage still uses finite contact by default. Only
`CaptureOptions.idealized_instantaneous=True` selects instantaneous reproduction.

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

Positions and translational velocities during contact and after attachment use
meters and meters per second in the retained fixed physical ECI basis. The
nonlinear model declares its J2 and atmosphere rotation axes in that basis;
they need not be global `z`. Approach relative states use the commanded
reference-orbit frame. The scalar-last quaternion maps each body basis, or the
post-lock assembly basis, to physical ECI.
Angular velocity is the inertial body angular velocity, expressed in body
components, in rad/s. Euler/anomaly derivatives are never treated as physical
spin.

Let `q` denote the recorded unwrapped anomaly, `ν` its physical rate in rad/s,
`C` the LVLH-to-ECI rotation, `x` the recorded relative position, and `x'` its
recorded anomaly derivative. Let `r_t` and `v_t` denote the target's physical
position and velocity. Let `k` be the positive orbit-normal unit vector. The
nonlinear adapter uses the actual saved target Cartesian state and original
commanded reference clock:

$$
r_c=r_t+Cx,\qquad v_c=v_t+C\nu(x'+k\times x).
$$

The frame-rate term is required even for a stationary reference-frame point.
No analytical target ephemeris is substituted for the actual nonlinear state.
For analytical reproduction only, the fixed ECI basis is chosen coincident with
the initial LVLH basis. Its elliptical model uses `x` opposite the transverse
direction and `y` radially outward. With orbital radius `r(q)` and its anomaly
derivative `r'(q)`, that analytical adapter uses

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

The current Envisat design retains the 7827.867 kg target mass and full inertia
tensor in [the source-qualified geometry declaration](envisat.md). Bus, solar
wing and appendages remain separate occupied components throughout approach
and capture.

Each chaser starts with 30 kg dry mass and 100 kg propellant. Its 0.55 m
spherical envelope uses the specific inertia `0.4 R² I`, giving an initial
wet inertia of `diag(15.73, 15.73, 15.73)` kg m². Fuel distribution is
approximated as proportional to the dry mass distribution. The complete runner
subtracts conservative translational consumption and modeled attitude-nozzle
consumption before capture; capture receives the remaining tanks.

The tracked point is the actual chaser COM. The 0.85 m COM standoff and 0.8 m
inward gripper offset leave a nominal 50 mm tip gap. The declared 100 mm closure
reach must cover the admitted position and orientation errors as well as that
gap. No free-flight COM is placed inside the hull.

### Held-command propellant bound

The nonlinear `NonlinearTranslationAccountant` binds the approach to every
completed hold in its source-bound proof archive, verifies exact clock coverage
and actual endpoint alignment, and rebuilds conservative fuel from applied-input
boxes. It does not infer inputs from outer means or use the analytical CH journal.
See [the continuation contract](finite-contact.md#nonlinear-mission-continuation).

For analytical reproduction, `HeldInputPropellantAccountant` binds the raw approach, its complete ordered
guard journals and the immutable producer source. It reconstructs the original
interval-flow and swept-geometry witnesses before using their held actuator
boxes. Missing or refused holds, changed clocks, mismatched states or incomplete
source lineage refuse the smaller allowance. A restarted mission requires its
complete original journal followed by every new admitted hold.

For hold $j$, let $U_{ij}$ bound the absolute anomaly-acceleration command in
component $i$, let $\bar\nu_j$ bound the anomaly rate in physical seconds, and
let $\bar\Delta t_j$ bound its physical duration. The initial wet mass $m_0$
bounds later mass. The sum of absolute body-nozzle forces $F_{1j}$ is bounded
for any proper attitude by

$$F_{1j}\le m_0\bar\nu_j^2\sqrt{3\sum_i U_{ij}^2},\qquad
\Delta m\le\frac{\sum_j F_{1j}\bar\Delta t_j}{g_0 I_{sp}}.$$

Here $g_0$ is standard gravity and $I_{sp}$ is specific impulse. Square roots
and sums are rounded outward; applied-input means are insufficient for this
bound. The installed-force, fuel and 2% constant-inertia gates remain binding.
The global input-cap allowance remains available when no authenticated journal
accountant is supplied. These are conservative fuel debits under the recorded
actuation assumptions, rather than measurements of consumption.

## Attachment gates and conservation

The target is **unprepared**. Assigned surface points are planning coordinates,
not docking ports or hardware mounted on the target. The default interface uses
representative chaser-carried compliant adhesive pads, with finite positioner
reach, speed, acceleration, force and power. All nine bodies respond to contact
wrenches. Unilateral compression, closing damping and friction precede adhesion;
adhesion begins only after actual touch, preload and dwell. Lock requires all
pads to satisfy the physical settling dwell. The full parameter and observation
contracts are in [finite compliant capture](finite-contact.md).

Capture requires the approach's consecutive terminal dwell, successful mission
completion, recorded tube containment, physical safety, and docking phase. It
checks the **current** nominal center and componentwise position/sliding tube,
not a maximum radius from a future prediction horizon. The actual state must
belong to that tube. The entire current position tube must fit the known
surface-anchor latch reach, stay outside the surface, and leave finite bus and
pairwise bus clearance. The physical closing-speed bound includes the anomaly
clock, rotating LVLH basis, tube velocity bound, and target spin.

Direct `CaptureOptions` admission defaults are 0.001 m/s relative speed,
0.01 rad orientation error and 0.0001 rad/s spin error. The nonlinear adapter
explicitly uses 0.01 m/s, 0.01 rad and 0.001 rad/s, with the declared 40 mm
whole-position terminal profile. These are precontact admission limits; finite
contact retains its stricter 0.0001 m/s and 0.00001 rad/s physical settling
gates and residual-lock-energy limit. The existing approach controls
translation; the separate [chaser attitude controller](chaser-attitude.md) records
actual bounded nozzle couples, quaternion/spin evolution and fuel. The complete
runner requires that controller's successful final attitudes and rates. A direct
capture call without physical chaser attitudes refuses by default. Callers may
also explicitly set `assume_attitude_matched=True` for a labeled idealization.
The latter assumes exact alignment to each mounting frame and exact matching
of the physical target spin. It remains visible in the saved assumptions and
is not an attitude-control validation. The nonlinear adapter uses causal past
navigation observations for the separate attitude replay and declared biased
contact observations. Actual physical states determine its final gates.

Finite contact records external impulse/work, propellant flux, positioner work,
bond-engagement work, spring energy and dissipation. Linear/angular momentum and
energy/work balance gates must pass. The actual component positions and
orientations define the assembly at lock. Only a bounded residual twist is
projected to rigid motion, with its impulses, couples and kinetic loss logged;
stored pad energy removed at lock is separate. No body is placed at a nominal
patch. The same bond capacities are checked against actual nozzle and
environmental loads during detumble, burn and recorded coast states. Contact
guards and downstream loads are numerical checks, not continuous structural
proofs.

### Explicit instantaneous reproduction

Only `CaptureOptions.idealized_instantaneous=True` selects the following ideal
articulating-tip path and coincident inelastic latch. It has no finite deployment
time, compliant force history, adhesive law or strength qualification. The
analytic geometry and conservation formulas below are reproduction checks;
they do not replace the ordinary finite-contact history or gates.

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
The actual COM positions remain in place at this idealized latch; only their
velocities change through the ideal inelastic impulse. The reported
kinetic loss is pre-latch kinetic energy in the combined COM frame minus the
resulting rigid rotational energy. Linear and angular momentum residuals,
COM shift, and dissipated energy are recorded. Latch loads, impulse duration,
flexibility, arm inertia, and mechanical strength remain unresolved in this
instantaneous reproduction. The finite model separately bounds loads, settling
and residual rigidization within its stated rigid/massless-arm scope.

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
fuel. For finite-contact assemblies, a pre-motion load check also dilates the
bounded command vector against the full environmental/inertial pad loads,
with a declared 0.98 capacity utilization target. The unchanged force/moment
capacities remain hard limits. An infeasible zero-command load, or a load that
cannot meet that reserve, refuses the step. Applied endpoint and history load
audits remain binding; this point check is not a continuous load enclosure.
A custom actuator layout can be injected; a missing torque axis or
insufficient wrench rank fails explicitly. Reaction wheels are not treated as
an external angular-momentum sink.

Let `Q=(u,s)` denote the vector and scalar quaternion parts, `ω` the body rate,
and `τ_g` the optional gravity-gradient torque. With `τ_env` denoting the
aggregate extra environmental torque about the current assembly COM, including
intrinsic component torques and the moments of their additional forces, the integrated rigid equations
during each frozen-mass substep are:

$$
\dot Q=\tfrac12(s\omega+u\times\omega,-u\cdot\omega),\qquad
J\dot\omega=\tau+\tau_g+\tau_{\mathrm{env}}-\omega\times J\omega.
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
\dot T=-K\|\omega\|^2+\omega^T\tau_{\mathrm{env}}.
$$

With zero aggregate extra environmental torque this reduces to
`T_dot = -K ||omega||²`. Zero intrinsic component torques alone are insufficient:
nonuniform environmental forces can still produce a moment about the assembly COM.
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

`FiniteBurnThrottlePlanner` then compares 0.65, 0.8, 0.9 and 0.95 of that
coupled capacity. A Cartesian two-body preview starts at the predicted
osculating apoapsis, applies retrograde **transverse** thrust, decreases total
mass and uses the operational allocator's static owner-specific fuel rates.
Each preview stops at the first perigee goal, descending entry, burn-duration
cap or owner fuel reserve. A declared 10% reserve is retained in every owner
tank. Selection minimizes predicted fuel among candidates reaching the perigee
goal, and saves every candidate including refused previews. Missing feasible
predictions authorize no burn. Nozzle ratings, installed tanks, actual burn
cap, pointing gates and final orbital events are unchanged.

This forecast includes finite-burn timing that an impulsive rocket-equation
allowance omits. Its static allocation and omitted attitude/inertia evolution
remain approximations; the 10% reserve is a planning margin, not a bound on
their errors. The predicted apoapsis is never assigned to the actual spacecraft.
The actual orbital state, bounded attitude maneuver, changing mass/inertia,
allocator and tanks are propagated independently and must satisfy the original
perigee and descending-interface gates.

The nominal burn cutoff is the root of the actual osculating-perigee event at
80 km altitude. If fuel is exhausted after an entry-intersecting orbit has
already been reached, the model can coast to the interface and separately
report `target_perigee_achieved=False`. Otherwise it reports fuel failure.
Successful completion requires a descending crossing of the configured 120 km
entry interface, located by an event root. A powered-phase crossing is also
located as an event. Merely lowering altitude or perigee is not completion.
The 80 km value guides burn cutoff; it is not the hard delivery event. With
perturbations, osculating perigee can change during the subsequent coast, so
the final `target_perigee_achieved` flag can be false after an 80 km cutoff.
The result retains that flag and the explicit entry/perigee outcome message.

An initially entry-intersecting bound orbit can coast to that descending
interface without requiring unused burn authority or fuel. The entry endpoint
and the requested perigee target remain separate outcomes. If entry and burn
cutoff occur during the same held command, both roots are bracketed against the
original hold and the earliest physical event wins; entry wins a numerical tie.
The assembly is never propagated below the declared handoff before reporting it.

The 120 km interface is an explicit rounded model boundary. NASA's published
Starliner example uses 400,000 ft, or 121.92 km; an interface is a convention,
not a universal atmospheric edge. This model stops there. Heating, breakup,
demise, impact, geographic targeting and public-risk assessment are not modeled.
Current nonlinear continuation retains its declared central/J2/vector-drag law
through this event, with no force-parameter switch at capture. Its radial
exponential atmosphere is a study force law, not an atmospheric reentry model.
Direct calls without an extra-environment callback retain central gravity.
The two-body osculating-element, vis-viva and burn-preview formulas are planning
references; actual configured propagation and the terminal event decide completion.

For the representative 1000 kg target orbit with eccentricity 0.18 and 300 km
initial perigee, the impulsive reference to an 80 km perigee is about 57.74 m/s.
A single 12 kg chaser carrying 4 kg fuel at 300 s specific impulse cannot
provide the approximately 19.67 kg ideal fuel need of that combined mass. The
single-chaser lifecycle therefore preserves a fuel-limit outcome. A fleet
result must be evaluated with the actual allocator and finite burn; an ideal
rocket-equation budget alone is insufficient.

## Use and records

The current nonlinear path retains the actual runner or supplies the explicit
typed model, clock, design and complete source-bound hold archive:

```python
from orbexa.simulation.nonlinear_completion import complete_nonlinear_mission

result = complete_nonlinear_mission(approach, nonlinear_runner=runner,
                                    hold_records=closed_hold_archive)
```

An incomplete approach returns an incomplete result. Saved-input continuation
must pass the arrival and source-lineage checks in
[the nonlinear continuation contract](finite-contact.md#nonlinear-mission-continuation).

The following is explicitly idealized analytical reproduction:

```python
from functools import partial
from orbexa.simulation.capture import CaptureModel, CaptureOptions
from orbexa.simulation.lifecycle import MissionLifecycle, MissionLifecycleStore
from orbexa.simulation.propulsion import ThrusterSystem

engine = MissionLifecycle(
    capture=CaptureModel(options=CaptureOptions(assume_attitude_matched=True,
                                               idealized_instantaneous=True)),
    propulsion_factory=partial(ThrusterSystem.for_assembly,
                               translation_force_n=10., specific_impulse_s=300.),
)
result = engine.run(qualified_physical_approach)
MissionLifecycleStore.save(result, output_directory)
```

The explicit attitude and instantaneous-contact assumptions in this example
are retained in the result.
`python -m orbexa.cli.complete_mission --help` exposes the configurable boundary.
Its default mode performs actual bounded chaser attitude replay and requires
`--input-component-bound` for conservative approach propellant accounting.
`--assume-attitude-matched` selects the attitude idealization; finite contact
remains the default unless instantaneous contact is also explicitly selected.
The separate `orbexa-run-nonlinear` entry point uses the current nonlinear
approach and continuation path. All modes preserve their assumptions in output.

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
