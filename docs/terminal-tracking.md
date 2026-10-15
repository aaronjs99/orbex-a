# Terminal tracking and admissible continuation

## 1. Verdict

**Pass with stated conditions.** The terminal module checks sufficient
invariance, actuator, geometry and accuracy inequalities using exact rational
arithmetic. The continuation module admits an initial set only after proving a
complete continuous-flow backup chain into that terminal set. A failed or
unacceptable optimizer proposal selects a previously certified held input.

The terminal certificate is conditional on the supplied global bounds and their
declared validity domain. Its algebra checker cannot establish that arbitrary
user-supplied sensor, reference, uncertainty or geometry bounds are true. A
`BoundScope` requires an identified analytical or validated interval global
bound source and explicit assumptions. Sampled maxima are refused. A proof
source string records provenance; it is not a verification of that source.

The public modules are `orbexa.control.terminal_tracking` and
`orbexa.control.continuation`. Continuous flow and occupied-set separation use
[the validated reachability layer](continuous-safety.md). The approach NLP stays
ADTMPC. These modules do not prove that its nonconvex optimization problem is
recursively feasible from every initial condition.

## 2. Conventions and Assumptions

The independent variable is unwrapped anomaly $q$ in radians. The six-state
plant is $x=[p^T,v^T]^T$ in the common physical LVLH frame, with $v=p'$.
The acceleration model is $v'=A_a(q,\theta)x+d_a(q,\theta)+u+w_d$.
The additive disturbance satisfies the declared componentwise bound
$|w_d|\le\eta$. The full parameter and model family must stay inside the
global bounds for the complete certificate scope.

The authoritative reference is a twice differentiable continuous function
$x_g=[p_g^T,p_g'^T]^T$, with known $a_g=p_g''$. Its feedforward input is
$u_g=a_g-\widehat A_a x_g-\widehat d_a$. The default start policy evaluates
these values at the actual sample anomaly. The separately identified midpoint
policy previews only feedforward at the actual hold midpoint, as defined below.
These references are not an RK4 shooting node or a persistent
numerical nominal center. Any reference evaluation, clock or sensing error must
be bounded in the supplied assumptions; giving numerical samples a reference
type does not establish this contract.

The state measurement at sample $q_j$ is $x_j^m=x(q_j)+n_j$ with
$|n_j|\le N=[N_p^T,N_v^T]^T$. The applied command may differ from its exact
requested value by $\epsilon_u$. Both error bounds are explicit. The input is
held for a positive anomaly interval no longer than $\tau$. Gains and the
nominal-model context are fixed within that hold. The global bound source must
cover every model context used by the controller. Missed deadlines or clock
jitter require separate proven timing bounds; no paused-plant assumption is
part of the controller proof.

The certificate domain is either an inclusive finite anomaly interval or all
future anomalies from a declared start. A finite-domain certificate permits
only holds wholly contained in that interval. At expiry the system needs a
separately admitted successor mission phase or renewed bounds.

## 3. Symbol, Shape, Unit, and Frame Table

| Symbol | Meaning and role | Shape | Unit and frame |
|---|---|---|---|
| $e=x-x_g$ | True reference tracking error | 6 | m and m/rad; physical LVLH |
| $s=e_v+\Lambda e_p$ | Sliding tracking error | 3 | m/rad; physical LVLH |
| $\Omega_*,\Phi_*$ | Fixed position and sliding half-widths | 3 | m and m/rad |
| $\Lambda,\kappa$ | Positive diagonal sliding gain and bandwidth | 3 | 1/rad |
| $\bar A,\delta_A,A_{\rm var}$ | Absolute nominal matrix, uncertainty and within-hold change bounds | $3\times6$ | Consistent acceleration/state units |
| $Z_g,\delta_d,\eta$ | Reference-state magnitude, affine mismatch and disturbance bounds | 6, 3, 3 | State and anomaly acceleration |
| $L_u$ | Feedforward Lipschitz bound per anomaly | 3 | m/rad³ |
| $N,\epsilon_u$ | State-measurement and applied-command error bounds | 6, 3 | State and anomaly acceleration |
| $R,D_v,D_e,D_s$ | Input reserve and derivative bounds | 3, 3, 6, 3 | Input or relevant derivative units |
| $d_g,r_b,\sigma$ | Reference-to-solid distance lower bound, body radius and clearance | Scalars | m; common frame |
| $X_j,U_j$ | Admitted backup state box and held-input box | 6, 3 | State and anomaly acceleration |

## 4. Line-by-Line Audit

| Location | Claimed step | Status | Issue or required condition | Correction |
|---|---|---|---|---|
| T1 | $V=\Phi_*+\Lambda\Omega_*$ encloses error velocity | Valid | $e_v=s-\Lambda e_p$, gains nonnegative | Componentwise triangle bound |
| T2 | Measured feedback fits the reserved input | Valid under condition | Measurement and actuator errors are bounded | Reserve measured-error widths and actuator error |
| T3 | Held feedforward equals continuous feedforward | Incorrect without a bound | A moving reference can require changing force | Add $\tau L_u$ to forcing |
| T4 | Held damping can use the continuous damping term | Valid under condition | Held-state and sensing changes must be bounded | Add derivative and sensing terms in T5–T6 |
| T7 | Fixed comparison widths imply terminal invariance | Valid under condition | Initial containment and global model/reference bounds | Check inward inequalities exactly |
| T8 | A centered error box is continuously collision-free | Valid under condition | Global reference-to-solid and peer gap bounds | Require strict whole-set geometry gaps |
| C1 | One safe performance hold ensures continued feasibility | Incorrect | Its endpoint may have no safe successor | Admit only endpoints in the next certified backup entry set |
| C2 | Solver failure permits a zero/braking input | Incorrect in general | Such an input can violate force or collision constraints | Apply the already certified scheduled backup |
| C3 | A certified chain implies supervised admissibility | Valid under condition | Exact execution clocks, trustworthy model/geometry proofs and terminal scope | Induction over the admitted chain |

## 5. First Consequential Error

Endpoint accuracy and an attractive Riccati tail do not establish terminal
invariance. Likewise, a safe next interval does not establish that a following
safe input exists. The terminal controller and successor admission rule below
provide the additional contracts required for these two claims.

The global reference, model, sensing and geometry bounds are still proof
obligations for each mission configuration. A narrowly bounded circular example
does not discharge them for an arbitrary tumbling spacecraft on an uncertain
eccentric orbit.

## 6. Corrected Derivation

### Fixed reference-relative set and actual held law

The terminal set $\mathcal T_*$ uses strictly positive fixed half-widths.
Define the velocity bound $V$, full state-error bound $E$, sensing sliding-error
bound $N_s$, and the componentwise saturation function $\operatorname{sat}$:

$$
\mathcal T_* = \{e:|e_p|\le\Omega_*,\ |e_v+\Lambda e_p|\le\Phi_*\},
\quad V=\Phi_*+\Lambda\Omega_*,\quad
E=\begin{bmatrix}\Omega_*\\V\end{bmatrix},\quad
N_s=N_v+\Lambda N_p.
\tag{T1}
$$

At sample $q_j$, let $e_j^m=x_j^m-x_g(q_j)$ and
$s_j^m=e_{vj}^m+\Lambda e_{pj}^m$. The requested input, held until the next
sample, is

$$
u_j=u_g(q_j)-\widehat A_a(q_j)e_j^m-\Lambda e_{vj}^m
 -[\delta_A|e_j^m|+\kappa\Phi_*]\odot
 \operatorname{sat}(s_j^m\oslash\Phi_*).
\tag{T2}
$$

The input reserve $R$ bounds the entire ancillary command. Its measured-state
terms include the sensing error, even when measurement error pushes the
measured sliding variable outside its layer. The saturation remains bounded by
one. Applied-command error is then reserved separately:

$$
R=(\bar A+\delta_A)(E+N)+\Lambda(V+N_v)+\kappa\Phi_*,
\qquad
u_{g,\min}-R-\epsilon_u\ge u_{\min},\quad
u_{g,\max}+R+\epsilon_u\le u_{\max}.
\tag{T3}
$$

The reference feedforward variation satisfies
$|u_g(q)-u_g(q_j)|\le L_u(q-q_j)$ throughout every hold. Define a base forcing
$w_b$ that covers model/reference mismatch, process disturbance, command error,
held feedforward change, and sensing-induced feedback change:

$$
w_b=\delta_A Z_g+\delta_d+\eta+\epsilon_u+\tau L_u
 +(\bar A+\delta_A)N+\Lambda N_v+\kappa N_s.
\tag{T4}
$$

Until a putative first exit, derivative bounds follow from the full reserved
command. Counting sensing forcing again in this derivative bound is
conservative and avoids cancellation assumptions:

$$
D_v=(\bar A+\delta_A)E+w_b+R,\qquad
D_e=\begin{bmatrix}V\\D_v\end{bmatrix},\qquad
D_s=D_v+\Lambda V.
\tag{T5}
$$

Adding and subtracting current continuous linear cancellation and damping in
the true sliding dynamics yields the hold correction $\rho$ and total forcing
$w_*$. The uncertain-error drift and the held robust term are bounded
separately. The sensing contribution $\delta_A N$ is already in equation T4.
Saturation is 1-Lipschitz, so the measured damping differs from the true
within-layer damping by at most $\kappa N_s$:

$$
\rho=2\delta_AE+A_{\rm var}E
 +\tau[\bar A D_e+\Lambda D_v+\kappa D_s],
\qquad w_*=w_b+\rho.
\tag{T6}
$$

Thus $D^+|s|\le-\kappa|s|+w_*$ while the error remains in the candidate set.
The fixed comparison set is invariant if its componentwise inward conditions
hold. The position comparison follows from $e_p'=s-\Lambda e_p$:

$$
w_*\le\kappa\Phi_*,\qquad
\Phi_*\le\Lambda\Omega_*.
\tag{T7}
$$

The comparison and first-exit argument applies on each hold and concatenates
across all admitted samples inside the bound scope. This is robust invariance
of a nonzero set; nonzero disturbance bounds do not imply exact asymptotic
tracking to zero. No RK4 error estimate is substituted into this argument.
The terminal law is evaluated relative to the exact continuous reference.

### Explicit midpoint feedforward policy

Let $s$ be the exact actual hold start, $h$ its actual width, and
$m=s+h/2$ its midpoint. The policy `midpoint_feedforward_v1` replaces only
$u_g(q_j)$ in T2 by $u_g(m)$. Measurement, reference-relative error, nominal
matrix and all feedback terms stay at $s$. The reference preview is causal
under the declared known-reference contract. It does not advance a measurement
or the physical flow start.

The true global componentwise Lipschitz bound is still $L_u$. Since every
$q\in[s,s+h]$ obeys $|q-m|\le h/2$, the new residual satisfies

$$
|u_g(q)-u_g(m)|\le L_u h/2\le L_u\tau/2.
\tag{T9}
$$

Only the direct $\tau L_u$ term in T4 becomes $\tau L_u/2$. T3, the full
feedback hold correction in T6, model uncertainty, sensing, actuator and
geometry bounds retain their original forms. T5 and T6 are recomputed from
the changed base forcing. A shorter final hold uses its own exact midpoint;
the midpoint is never computed from the maximum hold or whole proof window.

`certify_terminal_tracking(bounds, policy=TerminalFeedforwardPolicy.MIDPOINT_V1)`
records `exact-held-midpoint-reference-terminal-v1`. It preserves the true
`bounds.feedforward_lipschitz` and reports the residual factor separately.
The default call retains the original start-law method and serialized
dictionaries. An unknown or inconsistent policy/method refuses.

`TerminalHoldReferences` carries both canonical samples and their numerical
query clocks. `command(..., hold_references=pair)` requires the pair for
midpoint execution, verifies its exact clock relation and policy, and retains
both samples in the new command record. The exact-reference controller allows
no undeclared query displacement. The
[physical adapter](physical-terminal.md) separately accounts for both numerical
evaluation errors. A certificate for one held law cannot silently execute
the other.

### Geometry and complete-set accuracy

The whole-set position radius squared is $r_T^2=\sum_i\Omega_{*i}^2$; the
anomaly-velocity radius squared is $r_V^2=\sum_i V_i^2$. Let $d_g$ be a global
lower bound on distance from the reference center to the occupied target solid.
The terminal body has enclosing radius $r_b$ and required clearance $\sigma$.
The sufficient strict target condition and requested accuracy bounds are

$$
d_g-r_b-\sigma>0,\quad
(d_g-r_b-\sigma)^2>r_T^2,\qquad
r_T^2\le\varepsilon_p^2,\quad r_V^2\le\varepsilon_v^2.
\tag{T8}
$$

For a peer, replace $d_g$ with the proven global reference-center distance,
subtract both physical body radii and clearance, and require the remaining gap
to exceed $r_T+r_{T,\rm peer}$. The checker uses an equivalent pair of exact
squared inequalities, avoiding rounded square roots. Global distances can be
proved analytically for rigidly rotating assigned body points, provided the
reference/target pose errors and all peer widths are included.

### Finite certified backup and successor admission

Let $X_0,\ldots,X_{N-1}$ be entry boxes at exact scheduled anomalies and $U_j$
the admissible held-input boxes. A validated continuous enclosure returns
the full flow $\mathcal F_j(X_j,U_j)$, its swept set and its endpoint set
$X_j^+$. The collision oracle must establish swept separation from every
occupied object and peer, using a compatible frame and common clock. Admission
requires

$$
X_j^+\subseteq X_{j+1}\quad (j<N-1),\qquad
X_{N-1}^+-x_g(q_N)\subseteq\mathcal T_*.
\tag{C1}
$$

At stage $j$, a performance input is accepted only when its independently
validated swept flow is safe and its complete endpoint set satisfies the same
successor inclusion. Otherwise the stored $U_j$ is selected. Since the current
true-state box is contained in $X_j$, the stored backup proof applies. Induction
gives admissible finite continuation to $\mathcal T_*$. The terminal result
then applies throughout its declared remaining domain. If any flow, geometry,
input-limit, successor, terminal or timing requirement cannot be proved, initial
admission is refused. This refusal is not a proof of physical infeasibility.

This architecture follows the established separation of a proposed performance
input from a backup-based safety policy. It does not claim novelty for that
general idea. [Wabersich and Zeilinger (2021)](https://doi.org/10.1016/j.automatica.2021.109597).
Boundary-layer tube control likewise has established antecedents;
the implementation-specific held law and checks above must be assessed under
their stated assumptions. [Lopez, Slotine and How (2019)](https://arxiv.org/abs/1907.06553).

### Runtime and serialization contract

`certify_terminal_tracking(bounds)` returns exact quantities and refusal
reasons. `TerminalTrackingController` recomputes that certificate at
construction. Its `command` method requires an authoritative
`TerminalReferenceSample`, an independently valid current-state enclosure, a
measurement consistent with the declared sensing budget, and a complete hold
inside scope. `float_command()` refuses a conversion outside the declared
actuator-error budget.

`ContinuationBuilder.certify(...)` validates the complete backup before
execution. Constructing its supervisor recomputes all flow, geometry and
terminal checks. `decide(...)` returns either an admitted performance command
or the already proved backup. `advance(...)` uses its proven successor and
may intersect an independently valid observation box. An inconsistent
observation is refused. `terminal_handoff()` succeeds only after the complete
successor enters the terminal set.
The unchanged v1 continuation format explicitly supports only the start
terminal policy. It refuses a midpoint certificate instead of silently
recertifying it as the start law; midpoint physical execution uses its
separate policy-aware runner and handoff.

The execution layer must preload the fallback command and invoke the supervisor
on the actual schedule independently of an optimizer that may be late or fail.
The classes neither pause a plant nor provide a real-time operating system,
deadline scheduler, actuator driver or sensor. A planner result arriving after
the proved hold boundary cannot be applied as if time had stopped.

`as_dict()` records exact rational bounds alongside readable exports. Terminal
bound dictionaries can be restored with `TerminalTrackingBounds.from_dict`,
then recertified. A serialized `admissible` flag never authorizes execution.
Continuation dictionaries are evidence records; rebuild them from their inputs
and trusted proof providers before execution. The flow provider, terminal
global-bound source and geometry oracle must all describe the same actual
plant, reference, occupied bodies, uncertainty and timing context.

## 7. Domain-Specific Checks

The start terminal controller uses the same held ancillary law as the approach model,
with explicit extra forcing for changing reference feedforward and sensing.
It retains the full nonlinear analytical orbital model in the reachability
provider when `OrbitalHeldPlant` is selected. `AffineHeldPlant` is a separate,
explicit constant-affine plant adapter. It never inserts a continuous $A+BK$
matrix to certify a controller whose input is actually held.

Finite body radii and strictly positive clearance are part of both terminal
geometry and swept geometry. A near-surface gripper reference and a chaser
center are different physical points. Capture needs its own admitted hybrid
transition; a symmetric free-flight set centered on a solid contact surface
cannot satisfy strict exclusion.

All exact algebra uses Python `Fraction`, including input conversions interpreted
as their exact binary rationals. Nonfinite data, malformed shapes, zero layers,
missing global-bound provenance, invalid domains and inadequate margins fail
closed. The validated flow library remains an optional dependency; absence or
enclosure exhaustion cannot produce admission.

## 8. Sanity and Numerical Checks

An external saved exercise uses the exact circular, zero-drag limiting model,
a static along-track reference 2 m from the target center, and a four-pulse
transfer from 20 mm initial position error. It models a 1 m radius by 1 m long
solid cylinder, a 50 mm chaser body radius and 10 mm required clearance.

The terminal set has $\Omega_*=0.5$ mm per component,
$\Phi_*=0.005$ m/rad, $\Lambda=20$/rad, $\kappa=80$/rad, and maximum hold
$0.0005$ rad. Its whole-set position radius is approximately 0.866 mm, below
the 2 mm requested tolerance. Measurement bounds are 0.1 micrometre position
and 1 micrometre/rad anomaly velocity per component; command error is
$10^{-12}$ m/rad². These are illustrative declared bounds, not measured sensor
or actuator capabilities. The smallest exact sliding inward margin is greater
than 0.3487 m/rad²; all input reserves are below 0.732 m/rad².

All four continuous orbital flows and their complete swept target separation
were proved by Arb interval Picard enclosures. An injected solver failure on
pulse two selected the stored backup. A deliberately unsafe performance
proposal on pulse three was refused and also selected the backup. The complete
final endpoint enclosure lies in the terminal set; its along-track position
half-width is below 20 micrometres. The companion numerical execution spans
approximately 185.53 s and maintains at least 0.95 m body-to-solid clearance.
This is a focused supervisor qualification, not the full eccentric tumbling
multi-agent mission.

A numerical corner audit evaluates 64 true-state corners, 64 sensing corners
per state, and 17 instants in one hold. Maximum position and sliding occupancies
are at their initial boundary within $7\times10^{-13}$ numerical roundoff.
The corner audit supports implementation correspondence; the exact comparison
inequalities and stated global bounds carry the invariance argument.

The exercise also refuses excessive hold duration, feedforward variation,
sensing error, inadequate input reserve, inadequate geometry gap, an oversized
terminal tolerance set, sampled-only bound sources, missing bound provenance,
a wrong reference identity and a backup that does not reach the terminal set.
Raw exact certificates, source hashes, execution samples, a plot and a GIF are
kept outside the repository under the mission work products.

## 9. Unresolved Ambiguities

The maintained eccentric rigid-target profile now has a dedicated
[finite-domain bound builder and execution adapter](physical-terminal.md).
Its actual entry membership remains a separate gate; an initialized terminal
exercise does not imply that an approach has passed it.

The implementation does not supply global terminal reference and geometry
bounds for every user-defined tumbling target or uncertain orbit. Each mission
must produce and identify those bounds, admit a suitable backup, and connect
the supervisor to the actual scheduled execution. Parameter adaptation may
tighten a proven bound; an unexplained estimator output cannot replace a
valid retained parameter set.

Recursive admissibility of an admitted backup chain does not establish
recursive feasibility or global optimality of the GEKKO/IPOPT NLP, nor does it
guarantee that a useful backup exists for an arbitrary compact deployment.
Physical collision safety remains conditional on the declared plant, occupied
geometry, clock, sensing and actuator bounds. Attachment, contact compliance,
detumbling thrust allocation and orbital deorbit propagation are separate
mission phases with their own model and transition assumptions.
