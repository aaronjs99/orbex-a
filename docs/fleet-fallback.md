# Finite nonlinear fleet feedback fallback

`control/refuge.py` proves a sampled feedback tracking tube over a declared
finite domain. `control/fleet_fallback.py` admits complete fleet routes ending
in a holding region and admits performance actions as one transaction.
`simulation/nonlinear_refuge.py` constructs exact C2 polynomial routes, bounds
the actual nonlinear drift with Arb, transforms complete component geometry,
and propagates a bounded sensor blackout without a solver or flow service.

These modules provide a conditional finite safety contract. They do not make
the old ordinary approach recursively feasible, replace a physical deadline
scheduler, prove capture, or authorize stopping physical time after a failure.

## State, clock and reference

The supported actual transformed plant is

\[
 p'=v,\qquad v'=F(q,p,v,\theta)+u+\epsilon_u.
\]

Here q is the immutable commanded reference clock, p is metres, v is metres per
radian, and u is metres per radian squared. A different kinematic position row
or input matrix requires a sound transformation or a different theorem.

The reference has proved jets r, r', r'' and an acceleration Lipschitz bound J.
`ReferenceBounds` encloses the true jets on the whole finite window. Declared
reference evaluation errors cover the samples used by the controller.
`C2RefugeReference` uses exact rational quintic Hermite coefficients, exact C2
joins, and Bernstein convex-hull bounds for the complete jets. Its final piece
is an exact constant common-frame point. This does not rigidly rotate a distant
300 m point with the target.

Define e=p-r, w=v-r', s=w+Lambda e and

\[
 T(q)=\{|e|\le\omega,\ |s|\le\phi\},\qquad
 V_e=\phi+\Lambda\omega.
\]

At the actual sample, with bounded measured p_m,v_m, evaluate and hold

\[
 u_k=r''_k-\widehat F(q_k,p_m,v_m)
       -\Lambda(v_m-r'_k)
       -\beta\,\operatorname{sat}((v_m-r'_k+\Lambda(p_m-r_k))/\phi).
\]

Products above are componentwise. Every actual actuator value and binary64
command conversion lies in the returned applied-input interval. Rounding and
physical actuator errors have separate explicit budgets. Construction rederives
the certificate; a stored success flag never grants execution permission.

## Exact sufficient inequalities

Bounds include nonlinear mismatch W=|F-Fhat|, nominal derivative bounds
L_p,L_v,L_q, nominal numerical error epsilon_F, sensing errors epsilon_p,
epsilon_v, reference evaluation errors delta_p,delta_v,delta_a, and total
actuator plus conversion error eta. All bounds cover the full state tube,
sensor-expanded domain, hardware input box, and finite clock scope.

Let A_r bound |r''|, V_r bound |r'|, tau be the largest hold, and D_v be a
candidate bound on |v'|. The code computes

\[
 E=W+\epsilon_F+\eta+\tau L_q
   +L_p(\epsilon_p+\tau(V_r+V_e))
   +L_v(\epsilon_v+\tau D_v),
\]
\[
 D_w=D_v+A_r,\quad D_s=D_w+\Lambda V_e,\quad
 \epsilon_s=\epsilon_v+\delta_v+\Lambda(\epsilon_p+\delta_p).
\]

It verifies, using exact rational comparisons,

\[
 \Lambda\omega-\phi\ge0,
\]
\[
 D_v-[A_r+\delta_a+\Lambda(V_e+\epsilon_v+\delta_v)+\beta+E]\ge0,
\]
\[
 \beta-[E+\delta_a+\tau J
       +\Lambda(\epsilon_v+\delta_v+\tau D_w)
       +(\beta/\phi)(\epsilon_s+\tau D_s)]\ge0.
\]

The complete evaluated command range, including all error budgets, must lie
inside the hardware input interval. The full true and sensor-expanded tubes
must lie inside the supplied nonlinear model domains.

Before a hypothetical first exit, the acceleration self-bound controls sample
displacement. Saturation is 1/phi-Lipschitz, giving the sliding-boundary inward
inequality. Since e'=s-Lambda e, the position-boundary inequality is inward too.
This first-exit argument proves invariance of T(q) through the finite scope.
Positive margins are recorded; an unresolved or negative margin refuses entry.

`NonlinearZeroDriftBounds` provides a conservative executable construction:
Fhat=0, so all nominal Jacobians and numerical errors are exactly zero, while W
encloses the entire actual nonlinear drift. It evaluates the shared central/J2,
vector drag, rotating-frame and fixed-clock algebra with Arb on validated
target ephemeris segments. It never uses sampled maxima or numerical target
resets. More informative nominal drift bounds can use the same theorem.

## Whole fleet geometry

`FleetGeometryOracle` checks every named component and every peer on the same
exact hold clock. Missing components, changed body radii, partial continuous
segments, changed source identities and negative clearances refuse admission.
The chaser radius encloses its complete current physical configuration.

`OccupiedGeometryBounds` supplies component AABBs and optional
`OccupiedProjection(component_id,direction,projected_interval)` records.
Directions are fixed nonzero rational vectors; exact unit normals are
unnecessary. For a proved positive projected gap g along n,

\[
 \operatorname{distance}^2\ge g^2/(n^T n).
\]

The checker takes the strongest of this bound and exact AABB distance squared,
and requires strict separation exceeding the physical radius plus clearance.
This retains free space next to a tilted component face that an AABB fills.
`CompositeOccupiedBoundsProvider` projects every exact expanded vertex, so
declared component uncertainty is already included once.

The pose source is injectable. `AnyProperTargetPose` analytically encloses
arbitrary proper target rotations for distant refuges. For near-patch use,
`AttitudePoseBoundsProvider` maps each queried validated swept-quaternion box
through Arb quaternion normalization and C_ref(q)^T. The reference clock is an
explicit constructor argument. It verifies complete attitude segment coverage
and same target/model/clock identities. If an ambient quaternion interval
contains zero, the conservative all-proper-rotation enclosure is retained.
Future numerical attitude previews never supply occupied admission bounds.

Repeated interval queries cache only immutable continuous proof payloads. Every
lookup freshly hashes mutable proof contexts (including nested model/request
metadata); changing those contexts invalidates target/attitude identities and
the local box cache. Certificate identities on fully immutable fleet proof
records are computed once. Cache entries and retained query windows are bounded.

`construct_patch_refuge_candidates` prepares a finite set of complete joint
routes before installation. It accepts exact initial reference jets, complete
initial enclosures, supplied outward vectors and two common fixed reference
frame tangent vectors. Each candidate applies one common tangent offset to
every agent and constructs exact C2 retreat references followed by holding.
It returns `PatchRefugeSearchResult` with a complete plan or explicit per-candidate
refusals, plus `.reference_map` and `.context_map`. Success requires initial-box
entry as well as every invariant, component, peer, force and fuel check.
Neither fixed outward vectors nor tangent offsets bypass swept geometry.
The finite candidate budget bounds preparation work; search never happens at
the failure boundary.

## Complete fallback and performance transaction

Construct `RefugeController(original_builder)` for every fleet member/window,
then `FleetFeedbackSegment`, `FleetFallbackPlan(segments, geometry_oracle)`.
The plan verifies common clocks, complete geometry, reference continuity,
invariant-set inclusion across gain/tube changes, terminal constant holding,
and the fuel required through the complete remaining finite route.

`plan.suffix(q)` restricts an already proved route without calling providers.
Its reference, tube, model and geometry remain unchanged, their original global
proofs cover a superset, and the source plan identity is recorded. It does not
grant state entry into a different reference.

Each complete held command must fit one installed feedback segment. A scheduler
clips the next hold at `plan.segment_at(q).end`, or chooses reference proof
windows aligned to its hold grid. Clipping preserves the original maximum-hold
inequalities and still advances physical time to the exact requested endpoint.

`FleetAdmissionRuntime` requires whole-box initial entry before motion. A ready
`PerformanceProposal` binds its exact start/hold, current observation/context
identities, applied-input boxes, actual attempt, and a complete successor plan.
Every actual nonlinear flow must match the entire request and physical source
identities. One shared geometry transaction proves all complete swept sets;
every endpoint must belong to its successor feedback region. Only then may
the whole fleet choose PERFORMANCE and atomically install the successor.

On solver failure, unavailable performance flow service, stale proposal,
missing proof or rejected successor, the runtime evaluates the previously
installed feedback policy and returns FALLBACK_TRACK or FALLBACK_HOLD. The
actual failed attempt remains in `HeldFleetDecision`. `commit` verifies the
executed decision/clock and observation consistency before updating fuel and
the active plan. Independent per-chaser acceptance is not supported.

## Prospective performance successors

`ProspectiveFleetEndpoint.from_flows` in `nonlinear_safety` carries all eight
complete nonlinear held-flow certificates, the target hold and physical-time
endpoint, exact shared clocks, applied-input requests and original
observation/controller identities. It requires the full global 300 m per rad
squared input uncertainty. Command-dependent navigation subsets cannot create
a successor entry. Fresh fingerprints reject a changed request or proof
context before and after the potentially long preparation.

`NonlinearSafetyPlanner.prepare_successor(projection, sensing_prior=estimate,
certificate_sink=sink)` returns a proved plan and compact witness linkage. It
uses exact flow endpoint sets and their midpoints to choose reference jets.
It does not create a future navigation estimate, assume a future sensor
arrival, mutate the active runtime, or install the candidate. The entire
endpoint set must fit the new entry, and the existing fleet transaction still
checks the original performance sweep, hardware, complete successor and fuel.

Far candidates use the all-proper-attitude component enclosure. Near candidates
continue the current observed pose/rate inclusion from its real observation
clock across the proposed hold and the complete retreat/holding interval.
Geometry queries the validated quaternion windows at the future times. The
witness retains both the target tail initialized at the projected endpoint and
the complete target forecast used by that attitude inclusion.

The full `fallback_successor` witness, including its projection, references,
controllers and geometry proof, must be accepted by the supplied sink before
preparation returns. The caller owns durable acknowledgment or inclusion of
the complete witness in its staged-before-motion envelope. Only a subsequent
accepted performance transaction and `commit` install the successor. An
unavailable or refused builder leaves the stored feedback policy available.
Trying the old admitted suffix first avoids unnecessary preparation work;
building a new reference always requires a fresh complete proof.

The selected far study candidate uses lambda=200, omega=0.15 m exactly,
phi=30 m per rad, beta=22000, Dv=120000 m per rad squared and a declared
30 mm/6.5 m per rad sensing budget. Its common duration is rounded upward to
64 feedback holds from max(0.05 q, 3 max(norm(v))/60000). The exact quintic
starts with declared acceleration -3v/T and stops at p+Tv/4+2 outward, followed
by a finite holding interval. Every actual case still needs all nonlinear,
input, geometry, entry and resource inequalities. The larger sensing budget is
a prospective declaration, rather than a value fitted from future truth.

## Same-time peer geometry

Fallback polynomial references retain their common exact clock. For each pair,
the geometry proof subtracts the position polynomials before applying an exact
Bernstein convex hull on every shared piece/window. With p_i=r_i+e_i and
|e_i|<=omega_i, the resulting relative-position enclosure is

\[
 [r_i(q)-r_j(q)]+[-(\omega_i+\omega_j),\omega_i+\omega_j].
\]

Its distance from zero is a valid whole-window peer lower bound. The checker
takes the larger of this bound and the existing independent swept-box bound,
then applies the same finite-body radii, clearance and minimum center distance.
Every position error remains independent; there is no formation constraint or
shared-disturbance assumption. Component geometry and performance-flow geometry
retain their existing checks.

The full certificate stores all source polynomial coefficients, reference and
controller identities, exact windows, frame/clock identities, and each actual
omega. The coefficient digest must match the original analytic reference-bound
proof. Partial, stale or uncovered sources refuse. Installed plans and their
suffixes also guard the current provider, its methods and coefficient digest.
This avoids counting shared motion at different times as relative closing
motion while preserving both complete error tubes.

`NonlinearSafetyPlanner.retain_runtime` can retain an already committed plan
across an outer planning update. It checks the exact current clock, no pending
action, complete future scope, remaining fuel, original nonlinear proof sources,
and unchanged fleet/model/frame/clock/sensor/actuator/hardware declarations.
Current observations must still satisfy their original sensing budget and
consistency checks. Whole navigation-rectangle entry is not required again:
the runtime already owns the relational physical invariant. A separate
`fallback_reuse` archive acknowledgement references the existing plan; it does
not claim a new installation or a new full proof. Failure of this preflight
requires fresh preparation or the existing explicit assumption-failure path.

The far constructor searches three finite candidates in a fixed order: the
original lambda=200, omega=0.15, phi=30 domain with a 60000 reference-duration
target; omega=0.12, phi=24 with the same route; then that smaller domain with a
63000 duration target. Beta=22000, Dv=120000, global actuator error 300, sensing
floors 0.03 m/6.5 m per rad, input limits, installed hardware and all geometry
checks remain unchanged. Actual sensing budgets also cover the supplied current
initial-set widths. The duration target determines a candidate C2 polynomial;
exact polynomial bounds and all nonlinear/input/force/fuel inequalities still
decide admission. It is not a new physical thrust rating.

Each candidate requires its own complete fleet certificate and whole initial-set
entry before selection. Compact records preserve every attempted profile,
reference identity and refusal. Full candidate witnesses also retain failed
reference sources and any completed refused fleet certificate. If every choice
refuses, no new installation or successor is acknowledged. Near-patch candidates
retain their existing separate search.

The bounded 32-substep study covered 300 m and 150 m releases, common inward
velocities 1500/2500/3125 m per rad, and initial velocity uncertainty 5/6.5 m per
rad: twelve cases per alternative. The smaller 60000 route passed ten cases;
both 150 m/3125 cases refused target geometry. The 63000 route passed all twelve
and an authentic saved navigation case. Every passing case also completed six
moving feedback blackout holds with a fixed nonzero bounded actuator realization
and independent DOP853 containment checks. These are finite qualification cases;
every actual current or prospective endpoint still needs its complete proof.
Unchanged 8- and 16-substep alternatives failed their broader eligibility cases,
so this evidence does not authorize changing the control sampling schedule.

## Physical resources and sensor blackout

`StudyPropulsionBounds` defaults to the existing prospective 130 kg, 100 N,
300 s Isp design and an injectable 10 kg fallback budget. An attitude force
allocation is explicit. For arbitrary proper body rotation C,
|C u|_1 <= sqrt(3)|u|_2 <= 3 max|u_i|. Actual nu_ref squared converts anomaly
command to physical acceleration; fuel per q includes dt/dq=1/nu_ref.
Force authority and finite fuel are bounded with outward clock-rate intervals.
These are declared study hardware bounds, not thruster acceptance evidence.

Arbitrary performance inputs need their own `CommandResourceBound`, supplied
by `performance_resource_bound(id,input_box,q,h)`. Without it, the proposal
must fit the installed feedback command envelope. This prevents charging a
high-thrust performance action at the smaller fallback fuel rate.

`BlackoutObservationPropagator` advances a valid enclosure using known applied
commands and the globally proved actual-drift envelope. It needs no fresh
flow proof or optimizer. Its midpoint is a virtual measurement while both the
installed sensing budget and explicit finite blackout-q horizon hold.
Recovered observations must intersect the propagated enclosure. Performance
navigation uses its actual flow certificate, since performance can leave the
old tracking tube.

Exhausting sensing, fuel, model or clock assumptions is recorded as an
assumption failure. Finite-window expiry is explicitly reported; it does not
authorize an infinite safe physical pause. Extend a complete successor proof
before expiry. The calling runner owns scheduling, physical propagation,
drivers, source provenance, final capture, and the response to assumption failure.

## Evidence boundary

Bounded mission operations belong outside the repository. The initial study
qualification uses all Envisat components, eight 130 kg finite bodies at 300 m,
real nonlinear interval drift, exact C2 reference bounds, 30 mm/6 m per rad
sensing budgets and 300 m per rad squared actuator error. It distinguishes
the continuous conditional certificate from sampled numerical propagation and
from full mission or physical hardware completion. Near-patch refusal remains
a refusal until a complete same-context route, holding interval and geometry
certificate are actually admitted.

A separate bounded eight-patch study prepared a 0.015 q retreat by 2 m outward
plus a common 0.5 m positive patch-tangent offset, followed by 0.003 q holding.
All eight COMs began 0.85 m above distinct upper-bus patch points with nominal
velocities following 2.4 deg/s body spin. The validated observed pose used
Rxyz(0,0.2,-0.1), 0.001 operator error and 0.0001 rad/s body-rate uncertainty.
The joint proof included every one of the eleven declared solid components,
0.55 m finite vehicle radii and peer separation. With 5 mm/1.75 m per rad
declared sensing, 300 m per rad squared actuator error and maximum hold 1/32000 q,
the higher-gain candidate used lambda=300, omega=0.05 m, phi=12 m per rad,
beta=35000 and Dv=120000 m per rad squared. Its least sliding margin was 6845.9,
and maximum finite route/hold fuel bound was 0.24857 kg per vehicle.
Three injected-failure blackout holds used 4 mm/1.5 m per rad initial enclosures
to retain spare sensing budget. They produced moving numerical states and
nonzero feedback commands without a flow service; independent DOP853 endpoints
remained in the propagated boxes. This finite case does not establish admission
at arbitrary future target attitudes, navigation widths, or a complete mission.

The same joint entry also qualified beta=10000 and Dv=80000 with hold-aligned
0.0001875 q proof windows. Its least sliding margin was 1494.51, least squared
geometry margin was 0.19777 m squared, and maximum complete fuel bound was
0.18638 kg per vehicle. All three moving blackout commits again contained
the independent numerical endpoints. Gains remain declared candidate inputs;
every new observation, reference and finite horizon needs its own admission.
