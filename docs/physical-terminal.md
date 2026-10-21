# Finite terminal bounds for the physical uncontrolled target

## 1. Verdict

**Pass with stated conditions.** `PhysicalTerminalBoundsBuilder` derives global
bounds for a finite window of the maintained eccentric-orbit and rigid-body
model. It then checks the held terminal controller, complete terminal-set
accuracy, actuator reserve, target separation and peer separation. The
derivation uses interval arithmetic and exact rational inequalities. It does
not use extrema fitted to a recorded target history.

This closes the missing mission-specific bound source for the
[terminal tracking controller](terminal-tracking.md). It does not establish
that an arbitrary approach endpoint belongs to the terminal set.
`PhysicalTerminalFleet.admit` checks the supplied endpoint enclosures and
refuses endpoints outside that set. It never resets a spacecraft state.

The result remains conditional on the initial spin enclosure and the declared
instrumentation, numerical-reference, occupied-geometry and process-error
bounds. A numerical attitude integration is not converted into a validated
attitude trajectory by attaching a certificate label to its samples.

`ComponentwiseCompositePhysicalTerminalBoundsBuilder` is a separate service
for the explicit midpoint-feedforward law and complete composite geometry.
Its full-inertia componentwise first-exit and port-derivative bounds are
described below. Algebra/source qualification of this service does not
establish a controlled Envisat arrival or completed mission.

## 2. Conventions and Assumptions

The independent variable is unwrapped anomaly $q$ in radians. Physical time is
$t$ in seconds; a prime means differentiation in $q$ and a dot means
differentiation in $t$. The true reference state is
$x_g=[p_g^T,p_g'^T]^T$ in physical LVLH coordinates. Its acceleration is
$a_g=p_g''$. The body port $b$ is fixed in the target body, and its exact
physical reference is $p_g(q)=R(q)b$, where $R$ maps target body coordinates
into LVLH. All admitted body ports share this actual rigid rotation.

The orbital radius and clock use the same analytical inverse-radius quadratic
drag model as [the approach dynamics](theory.md). The uncertainty box covers
the retained eccentricity and drag constants. The nominal controller context
is a fixed point; the target clock is separately specified. Their gravitational
parameter and fixed semi-major axis must agree. These are bounds for that
analytical small-separation model, not a proof for an unspecified atmosphere
or full nonlinear two-spacecraft dynamics.

The target has fixed, exactly symmetric, physically realizable inertia $J$.
The entry box encloses its true body angular velocity at the start of the
window. Environmental torque is either zero or central spherical-field gravity
gradient torque. Magnetic, aerodynamic, solar-radiation, contact and flexible
body torques require additional bounds and are outside this target model.

`PhysicalTerminalErrorBounds` separates physical sensing error, reference
state and acceleration evaluation error, nominal matrix and affine evaluation
error, evaluation-clock error, actuator error, process acceleration and a
target-solid position error radius. Its nonempty declaration identifies these
assumptions. None is inferred from plotting resolution or solver tolerance.
Evaluation-clock error concerns a reference query; the actual command schedule
must still satisfy the certified maximum hold.

## 3. Symbol, Shape, Unit, and Frame Table

| Symbol | Meaning and role | Shape | Unit and frame |
|---|---|---|---|
| $J,j_-,j_+,Delta$ | Body inertia, spectral lower/upper bounds and spread | $3\times3$, scalars | kg m²; body |
| $r,\nu$ | Orbital radius and $dq/dt$ | Scalars | m, rad/s |
| $W_0,W$ | Entry and global physical spin-norm bounds | Scalars | rad/s |
| $D_1,D_2$ | Bounds on $\|\dot\omega\|$ and $\|\ddot\omega\|$ | Scalars | rad/s², rad/s³ |
| $K_0,K_1,K_2$ | Bounds on $\omega/\nu$ and its first two anomaly derivatives | Scalars | Consistent anomaly units |
| $P_0,P_1,P_2,P_3$ | Reference position, velocity, acceleration and jerk norm bounds | Scalars | m, m/rad, m/rad², m/rad³ |
| $\bar A,\delta_A,A_{\rm var}$ | Nominal magnitude, plant mismatch and held nominal change bounds | $3\times6$ | Acceleration/state units |
| $E_g,E_a,E_A,E_d$ | Reference and nominal evaluation error bounds | 6, 3, $3\times6$, 3 | Corresponding state/model units |
| $N,\epsilon_u$ | Physical sensor and applied-command error bounds | 6, 3 | State, m/rad² |
| $\Omega_*,\Phi_*$ | Position and sliding half-widths | 3 | m, m/rad; physical LVLH |

## 4. Line-by-Line Audit

| Location | Claimed step | Status | Issue or required condition | Correction |
|---|---|---|---|---|
| P1 | Orbital samples bound the whole window | Incorrect | Samples can miss extrema | Arb interval jets evaluate the complete anomaly and parameter box |
| P2 | SPD inertia is physically realizable | Incorrect | Principal-moment triangles are also necessary | Check SPD and positive semidefiniteness of $\operatorname{tr}(J)I-2J$ exactly |
| P3 | Entry spin is a global spin bound | Incorrect | Gravity gradient and rigid-body coupling change spin | Use the selected scalar first-exit, energy comparison or strict componentwise first-exit proof |
| P4 | Physical angular acceleration directly gives $R''$ | Incorrect | The clock and rotating LVLH frame contribute derivatives | Include $\nu'$, $\nu''$ and all rotating-frame product terms |
| P5 | Numerical reference values meet an exact-reference API | Incorrect without conditions | Evaluation error also affects admission and commanded input | Use robust reference-box membership and the derived evaluation forcing budget |
| P6 | Rotation can reduce assigned port separation | Valid under condition | The references share the same actual proper rotation | Preserve exact body-point distances; include all terminal body/tube radii |
| P7 | Passing a terminal certificate admits an approach endpoint | Incorrect | Initial containment is a separate hypothesis | Check the actual supplied endpoint enclosure without projection or reset |
| P8 | A finite-window result applies indefinitely | Incorrect | Bounds and holds have a finite declared domain | Renew the bounds or admit the next mission phase before expiry |

## 5. First Consequential Error

An arbitrary `proof_reference` string cannot establish global bounds for an
eccentric gravity-gradient target. The generic terminal algebra checker
correctly treats such bounds as assumptions. This module supplies their
derivation for the specified physical target family: interval orbital
derivatives, an exact spin comparison, derivative norm bounds and rigid
geometry. Initial spin and evaluation-error validity remain declared inputs.

The related numerical-to-theory error is to compare an observed state only
with an approximate reference and call that exact terminal membership.
`robust_contains` instead subtracts every reference in the declared evaluation
box. A slightly uncertain numerical reference therefore shrinks admissible
entry, even though the true invariant set keeps its original widths.

## 6. Corrected Derivation

### Global orbital coefficients and clock

Let $Q$ be the complete finite anomaly interval, enlarged by the reference
query-clock error. Let $\Theta$ be the complete plant parameter box. The
nominal point context is $\widehat\theta$. The acceleration matrix is $A_a$
and the affine acceleration is $d_a$. Arb evaluates the maintained analytical
expressions with a formal interval-base-point jet $q=Q+z$. Its coefficient
of $z^k$ encloses $f^{(k)}(q)/k!$ for every $q\in Q$ and retained parameter.
The coefficients are derivative enclosures; no truncated polynomial is used
as an approximation of a trajectory. The derived global quantities are

$$
\bar A\ge|A_a(Q,\widehat\theta)|,\quad
\delta_A\ge|A_a(Q,\Theta)-A_a(Q,\widehat\theta)|,\quad
A_{\rm var}=\tau\sup_Q|A_a'(q,\widehat\theta)|,
\quad
\delta_d\ge|d_a(Q,\Theta)-d_a(Q,\widehat\theta)|.
\tag{P1}
$$

Dependency loss can enlarge these intervals; it cannot invalidate enclosure.
The same calculation bounds the positive orbital radius and rate, and the
first two anomaly derivatives of the rate. If $\nu_-$ is the proved positive
rate lower bound and $\delta_q$ is the query-clock bound, the elapsed-time
upper bound from entry to any query in the window is

$$
T=\frac{q_b-q_a+\delta_q}{\nu_-}.
\tag{P2}
$$

`bound_orbital_clock(q_a,q_b,parameters)` exposes the radius/rate boxes and
$(q_b-q_a)/\nu_-$ independently of terminal tracking. This supports a global
translation force/fuel budget without sampling the orbit. Its endpoints and
duration are exact rationals. `IntervalBox.float_bounds()` exports outward
floating-point bounds.

Arb's real balls and power series are documented by
[FLINT](https://flintlib.org/doc/arb.html) and
[python-flint](https://python-flint.readthedocs.io/en/latest/arb_series.html).
The exercised backend is python-flint 0.8.0; the derivation does not rely on
a binary64 transcendental value for a proof decision.

### Physical inertia and spin

The spectral bounds satisfy $0<j_-\le\lambda_{\min}(J)$ and
$\lambda_{\max}(J)\le j_+$. Their difference is $\Delta=j_+-j_-$.
Exact principal-minor checks prove physical realizability through
$\operatorname{tr}(J)I-2J\succeq0$. Diagonal inertias have exact spectral
bounds. General symmetric inertias use exact rational Sylvester tests to
refine Gershgorin brackets; no floating-point eigenvalue is treated as proof.

Let $C=3\mu/r_-^3$ for gravity-gradient mode, or $C=0$ for torque-free mode.
For any unit radial body vector $n$, centering $J$ by a scalar multiple of the
identity gives $\|n\times Jn\|\le\Delta/2$. The standard Euler equation and
gravity-gradient torque are

$$
J\dot\omega=-\omega\times J\omega+\tau_B,
\qquad \tau_B=\frac{3\mu}{r^3}\,n\times Jn.
\tag{P3}
$$

These are established rigid-body mechanics; see the
[MIT rigid-body dynamics notes](https://ocw.mit.edu/courses/8-09-classical-mechanics-iii-fall-2014/6fe39e8d5ce4ce746ca256dfea665eda_MIT8_09F14_Chapter_2.pdf)
and the [ESA GODOT gravity-gradient torque interface](https://godot.io.esa.int/docs/api_reference/generated/generated/generated/godot.core.astro.gravityGradientTorque.html).

The entry spin box gives the norm bound $W_0$. Define the nonnegative
comparison coefficients $a=\Delta/j_-$ and $b=C\Delta/(2j_-)$. A candidate
global spin bound $W$ is accepted only if the following exact rational
inequality is strict:

$$
W>W_0+(aW^2+b)T.
\tag{P4}
$$

Until a putative first exit from $\|\omega\|<W$, Euler's equation gives
$\|\dot\omega\|\le aW^2+b$. Integration to that exit contradicts P4.
The smaller quadratic root provides a candidate; its rounded value is never
accepted without the exact P4 check. A nonpositive discriminant or inadequate
margin refuses the window. Shorter windows or tighter entry bounds may help;
the code does not enlarge accuracy tolerances to force acceptance.

Let $H$ bound $|r'/r|$, and let $\nu_+$ bound the rate. The torque-factor
derivative has bound $C_t=3CH\nu_+$. The radial-vector body derivative has
norm at most $W+\nu_+$. The physical spin derivative bounds are therefore

$$
D_1=\frac{\Delta}{2j_-}(W^2+C),\qquad
T_1=\frac{\Delta}{2}C_t+C\Delta(W+\nu_+),\qquad
D_2=\frac{T_1+\Delta W D_1}{j_-}.
\tag{P5}
$$

The anisotropic inertia part bounds the derivative of
$\omega\times J\omega$ by $\Delta W D_1$. These bounds cover all target
attitudes and every spin admitted by the entry box.

### Reference derivatives and feedforward

Let $N_1$ and $N_2$ bound $|\nu'|$ and $|\nu''|$. Define $K=\omega/\nu$.
The following bounds account for both the physical-time dynamics and their
conversion into anomaly derivatives:

$$
K_0=\frac W{\nu_-},\quad
K_1=\frac{D_1+WN_1}{\nu_-^2},\quad
K_2=\frac{D_2+3D_1N_1+2WN_1^2}{\nu_-^3}
       +\frac{WN_2}{\nu_-^2}.
\tag{P6}
$$

The LVLH frame rotates by positive anomaly about the orbit normal. With its
skew matrix $Z$ and $[K]_\times$ for the body anomaly rate,
$R'=-ZR+R[K]_\times$. Repeated product differentiation and the spectral norm
give global bounds $B_1,B_2,B_3$ for the first three rotation derivatives:

$$
B_1=1+K_0,\qquad B_2=(1+K_0)^2+K_1,\qquad
B_3=(1+K_0)^3+3(1+K_0)K_1+K_2.
\tag{P7}
$$

If $P_0$ encloses $\|b\|$, then $P_j=P_0B_j$ bounds
$\|p_g^{(j)}\|$ for $j=1,2,3$. The componentwise reference state bound is
$Z_g=[P_0,P_0,P_0,P_1,P_1,P_1]^T$, and its derivative has bound
$Z_g'=[P_1,P_1,P_1,P_2,P_2,P_2]^T$. For
$u_g=a_g-\widehat A_a x_g-\widehat d_a$, a global feedforward Lipschitz
bound follows directly:

$$
L_u=P_3\mathbf1+
\sup_Q|\widehat A_a'|Z_g+
\bar A Z_g'+\sup_Q|\widehat d_a'|.
\tag{P8}
$$

The feedforward range is obtained by interval evaluation using these reference
state and acceleration boxes. It need not be tight to be valid. The terminal
algebra then includes $\tau L_u$ in its forcing and checks each candidate
bandwidth without changing the requested tube widths or tolerances.

### Numerical evaluation and robust admission

Let $E_g$ bound the reference state evaluation error after adding the query
clock contribution $\delta_q Z_g'$. Analogous derivative bounds enlarge
$E_a,E_A,E_d$. A true sensor error bounded by $N$ then enters measured tracking
error as $n-E_g$, so the effective sensing budget is $N+E_g$.

Let $E$ be the true terminal state-error magnitude bound. Comparing the
command evaluated with approximate reference/model values to the exact
reference held law leaves the following sufficient extra input-error budget:

$$
\epsilon_{\rm eval}=E_a+E_d+E_A(Z_g+E+N)+\bar A E_g.
\tag{P9}
$$

The physical actuator budget and P9 are added in the terminal forcing and
reserve calculation. Actual binary64 command conversion is checked against
the physical actuator budget. The wrapper does not spend the larger
reference-evaluation allowance to excuse arbitrary command-rounding error.

For admission, let $\widetilde x_g$ be the evaluated reference and let $X$ be
an independently valid current-state box. `robust_contains` checks the
position and sliding inequalities for every $x\in X$ and every
$x_g\in[\widetilde x_g-E_g,\widetilde x_g+E_g]$. The physical sensing
budget is checked separately against the actual measured state. Evaluated
reference/model values that contradict the derived global magnitude bounds
are refused. The declared pointwise evaluation-error bounds still require a
trustworthy source; a broad global magnitude check cannot prove them.

`PhysicalTerminalController(builder)` reruns the global derivation and exact
terminal algebra. It does not authorize execution from a saved result flag.
`PhysicalTerminalFleet` additionally checks common target, port, peer geometry
and clock contexts and requires one unique member for every shared port.

### Anisotropic endpoint admission

A Euclidean 2 mm endpoint condition does not imply membership in the default
1.15 mm component cube. Terminal controller selection may therefore evaluate
a declared finite family of anisotropic widths without changing any endpoint
state. Useful prospective shapes include every axis permutation of
$(1.6,0.8,0.8)$ mm and $(1.8,0.6,0.6)$ mm, together with the qualified
isotropic shape. Their Euclidean radii remain below 2 mm.

For each candidate, retain positive $\Lambda$ and choose
$\Phi_*=0.8\Lambda\Omega_*$ componentwise. With $\Lambda=20$/rad, both
example shapes also meet the 0.08 m/rad complete-set velocity tolerance.
The builder must still prove every sliding inward, actuator, geometry and
accuracy inequality and robust entry containment. A declared bandwidth list
such as $(80,120,160,240,300)$/rad may be evaluated; acceptance never changes
the original accuracy tolerance. A small finite family need not cover every
point of the 2 mm sphere, and some accurate endpoints lack a proved invariant
set under the chosen uncertainty and actuation bounds.

Different agents may select different component widths. In that case set
`peer_position_radius_squared_upper` to a shared conservative upper bound,
for example the exact square of 2 mm. The fleet facade verifies that each
member's assumed peer radius covers every other member's selected radius.
Using an agent's smaller own radius as an unexamined peer radius is refused.
This permits anisotropic choices while retaining the rigid common-rotation
pair-distance proof.

### Executing the actual endpoint

`orbexa.simulation.approach.terminal.execution.PhysicalTerminalRunner` owns
the offline runtime loop. Its constructor receives `builders`, the actual
physical `target`, a point `truth` for numerical propagation, an explicit
`observation_source`, and a declared `orientation_matrix_error`. Controller
and Arb flow calculations retain the builders' parameter prior; propagation
truth never replaces that prior.

The matrix-error allowance must cover pose estimation and numerical matrix
evaluation, plus $B_1$ times the complete sample-query displacement. This
includes conversion of the exact rational sample anomaly to binary64. A
matrix error bound at the rounded query alone is insufficient for a geometry
certificate at the exact scheduled sample.

The `run` call requires the real `initial_anomaly`, `initial_states` and
`initial_observation`. A `TerminalObservation` contains the measured values
and independently valid state enclosures. It is supplied separately at entry,
so a saved approach endpoint cannot be replaced by an observation generated
at the docking reference. The observation callback supplies later observations
at their actual scheduled anomaly. It may intersect independent observation
and preceding flow boxes; this refines knowledge and never changes a state.

The minimal integration has this shape, with every input obtained from the
actual approach and the explicit bound configuration:

```python
from orbexa.simulation import PhysicalTerminalRunner, TerminalObservation

runner = PhysicalTerminalRunner(
    builders=admitted_builders,
    target=physical_target,
    truth=simulation_truth,
    observation_source=bounded_observation_source,
    orientation_matrix_error="1e-8",
    record_sink=write_certificate_record,
)
result = runner.run(
    initial_anomaly=approach_endpoint_anomaly,
    initial_states=approach_endpoint_states,
    initial_observation=TerminalObservation(
        measurements=endpoint_measurements,
        state_enclosures=endpoint_measurement_enclosures,
    ),
)
```

Admission failure returns a refused result with the supplied endpoint intact
and zero executed holds. During execution, each proposed held command is
checked against the fixed terminal bounds, then an Arb flow and a whole-fleet
swept collision proof are constructed before numerical propagation. The
journal records exact rational start/hold clocks, observations, applied-input
boxes, complete flow certificates and geometry. Flow fingerprints bind the
geometry to the actual proved flows; an explicit command/flow/geometry linkage
has its own SHA-256 digest. Numerical endpoints must lie in their flow boxes.
An inconsistent observation, command conversion, clock evaluation or process
source is refused. The result retains the last valid observation's anomaly
separately if execution ends on an observation failure.
An ordinary exception raised by an external observation or process source
also returns a refused result with its completed prefix retained; it cannot
turn a successful earlier prefix into an unrecorded apparent restart.

The numerical truth must match the actual physical target clock and remain
inside the retained prior. The declared entry spin must enclose the actual
numerical target entry. These are useful correspondence checks; they do not
prove that a hardware sensor satisfies the assumed error budget. A successful
result means the requested finite terminal window completed. It is not a
claim of real-time scheduling or indefinite tracking after that window.

### Global occupied geometry

Write $b_\perp$ for the selected body's first two port coordinates. The target
cylinder radius is $r_c$ and its relative geometry inflation is $\epsilon_c$.
If $\epsilon_p$ bounds additional solid-position error, the global
reference-to-solid distance has the conservative lower bound

$$
d_g\ge\|b_\perp\|-r_c(1+\epsilon_c)-\epsilon_p.
\tag{P10}
$$

The lower square-root endpoint is obtained by Arb. Axial cylinder extent
cannot reduce this radial lower bound. For peers, proper rotation preserves
$\|R(b_i-b_j)\|=\|b_i-b_j\|$. The checker uses downward Arb distance
enclosures, complete tube radii, physical enclosing body radii and required
clearance. Different target poses are not assigned to different agents.

### Componentwise full-inertia port jets

The reusable `bound_componentwise_terminal_jets` service retains the full
fixed inertia $J$. Let $M=J^{-1}$, let $F$ be the physical-time spin
derivative, and let $\tau_g$ be central gravity-gradient torque. Exact
rational expansion combines coefficients of
$F=M[-\omega\times(J\omega)+\tau_g]$ before interval evaluation.

The entry box has center $c$ and half-widths $r_0$. A proposed whole-query
spin box $W=c+[-b,b]$ is accepted only if the absolute time bound $T$ and
the complete torque enclosure give the strict componentwise inequalities

$$
r_{0i}+T\sup_{\omega\in W,\ \tau_g\in\mathcal U_g}|F_i|<b_i.
\tag{P11}
$$

This is a first-exit comparison in both time directions, including backward
evaluation queries. It does not assert contraction. Its elapsed time is
measured from the actual entry to the furthest query. Centered-inertia
bounds enclose gravity-gradient torque and its derivative, including body
and radial-direction motion. Then
$\dot F=D_\omega F_{\rm gyro}F+M\dot\tau_g$.

For the fixed body port $b_p$, define the inertial physical-time velocity,
acceleration and jerk in body coordinates before taking any norm:

$$
\begin{aligned}
v_b&=\omega\times b_p,\\
a_b&=F\times b_p+\omega\times(\omega\times b_p),\\
j_b&=\dot F\times b_p+F\times(\omega\times b_p)
 +2\omega\times(F\times b_p)
 +\omega\times(\omega\times(\omega\times b_p)).
\end{aligned}
\tag{P12}
$$

Exact expanded polynomials in spin, torque and torque derivative preserve
the cancellations associated with each actual port and full off-diagonal
inertia. Arb evaluates them over the entire first-exit enclosure. Physical
rotation preserves their Euclidean norms; the positive orbital clock and
all anomaly/LVLH product terms then give global reference state,
acceleration and jerk bounds. No target-history maximum is used.

### Two-clock numerical evaluation and execution

The new physical method and scope proof identity are both
`arb-componentwise-composite-midpoint-terminal-v1`; the separate geometry
provenance remains the complete-component geometry proof. Its algebra method
is `exact-held-midpoint-reference-terminal-v1`. The builder, controller,
execution record and handoff parser must agree on this policy. Relabeling
an old certificate or an outside prospective candidate is refused.

The midpoint and start matrices are evaluated separately. Let $G$ bound the
true six-state reference, $E$ the tracking error, $N$ physical sensing error,
$E_g$ reference evaluation, $E_A$ matrix evaluation, $E_a$ acceleration
evaluation and $E_d$ affine evaluation. All are componentwise uniform bounds
covering both queries. The numerical input reserve is

$$
E_u=E_a+E_d+E_A(G+E_g)
       +E_A(E+N+E_g)+|\widehat A|E_g.
\tag{P13}
$$

The two matrix-error terms cannot use the same-clock cancellation in P9.
Effective feedback sensing is $N+E_g$; physical actuator error remains
separate. The true global feedforward Lipschitz constant is preserved, and
only its held variation uses the half-width residual proved in
[T9](terminal-tracking.md#explicit-midpoint-feedforward-policy).

`PhysicalTerminalRunner` obtains the actual width before calling
`hold_references_at(s, h)` to evaluate the two exact clocks $s$ and $s+h/2$.
Each corresponding binary64 query displacement is independently checked
against the original anomaly-error declaration. The complete hold endpoint
is checked separately. Every new command and linkage records both reference
samples, exact/query clocks, error contracts, policy, target/source identity,
physical input interval, continuous-flow fingerprints and swept geometry.
The source observation and continuous flow still start at $s$.

Midpoint execution requires an explicit `source_context`. For an actual
approach, prepare it from the literal source file and a separately frozen
terminal runtime:

```python
from orbexa.simulation import terminal_source_lineage

source_context = {
    "origin": "actual_approach_endpoint",
    "approach_result_sha256": original_approach_file_sha256,
    "source_snapshot_sha256": frozen_terminal_source_sha256,
    "error_source_sha256": preserved_error_declaration_file_sha256,
    "source_approach_lineage": terminal_source_lineage(approach.to_dict()),
}
```

Pass this context to `PhysicalTerminalRunner`. Its new execution method is
`physical-terminal-midpoint-execution-v1`, with schema
`orbexa.physical_terminal_execution.midpoint.v1`. The context binds the
original arrival policy, required region, runtime state and transition
lineage without appending fictitious MPC or adaptation records.
An initialized terminal fixture must instead declare
`origin="initialized_terminal_fixture"`; it cannot become an actual-approach
handoff.

`PhysicalTerminalHandoff.from_files` selects the supported policy and
geometry together. It rebuilds the complete global certificate, both samples
and commands, every applied-input flow, all component/peer swept separation,
observation refinements and linkage. It compares the full saved proof
records. Only a complete successful reproof can provide the literal final
state/clock to the unchanged capture interface. Missing policy, clock,
sample or source fields refuse. Old start-law records retain their
original serialization and reproof path. Partial terminal resume remains
unsupported; an approach checkpoint is not terminal execution evidence.

The saved Envisat prospective family uses the original maximum individual
hold $1152921504606847/36893488147419103232$ rad, approximately
$3.125\times10^{-5}$ rad. Each proof window is exactly 32 such holds,
approximately $0.001$ rad or $0.96$ s. The 128 eight-port/event certificates
are finite conditional mathematics. Their preserved limits, full prior,
error declarations and true Lipschitz constants match the maintained
derivation; the real endpoint must still pass fresh all-eight admission.
The unchanged recorded $q=0.16$ endpoint remains outside this entry region.

## 7. Domain-Specific Checks

For the representative body ports, the radial center-of-mass standoff is
0.165 m, the chaser body radius is 0.15 m, the cylinder radius is 0.8 m, and
the relative target geometry inflation is 0.001. A 1 mm required clearance
still leaves more than 13 mm before the terminal position-radius allowance.
The complete position radius for 1.15 mm component widths is approximately
1.99186 mm, below the 2 mm tolerance. With $\Phi_*=0.020$ m/rad and
$\Lambda=20$/rad, the complete anomaly-velocity radius is approximately
0.07448 m/rad, below 0.08 m/rad. The componentwise position inward margin is
0.003 m/rad.

The isotropic inertia limit has zero gravity-gradient and Euler coupling
torque. Zero inertial spin then retains the expected LVLH counterrotation.
Physical inertia triangles, clock positivity, spin self-inclusion, all terminal
inward/input inequalities and strict geometry separation are checked before
admission. Missing Arb support, failed bounds or invalid inputs cannot produce
an execution certificate.

The general concept of a terminal set or backup-based MPC safety policy has
established antecedents; this implementation does not claim novelty for it.
See [Wabersich and Zeilinger (2021)](https://doi.org/10.1016/j.automatica.2021.109597).

## 8. Sanity and Numerical Checks

The saved exercise uses the current physical orbit: semi-major axis
8,144,068.6585 m, eccentricity 0.18, target drag $2\times10^{-7}$ and chaser
drag $4.5\times10^{-7}$. The independent retained plant prior is
$e\in[0.14,0.19]$, $\alpha\in[1.5,2.5]\times10^{-7}$ and
$\beta\in[3.5,5.5]\times10^{-7}$. The nominal point is
$(0.17,1.8\times10^{-7},4.2\times10^{-7})$. The gravity-gradient target
has principal inertia $(190,190,320)$ kg m² and initial body spin
$(0.0002,0.0003,0.00015)$ rad/s at anomaly zero.

At entry anomaly 4, the entry spin is the numerical model value with a
declared $10^{-8}$ rad/s component enclosure. The derived global norm bound
over $[4,4.1]$ is below $0.00094023$ rad/s. The $\kappa=80$/rad candidate
is refused; $\kappa=120$/rad passes. The requested position and sliding
widths are unchanged. The minimum sliding inward margin remains above
0.0574 m/rad² with 1 micrometre position sensing, $10^{-4}$ m/rad velocity
sensing and $10^{-5}$ m/rad² process forcing. These are declared synthetic
budgets, not measured capabilities. An independent $[7,7.1]$ window also
passes at $\kappa=120$/rad with its own entry spin enclosure.

The actual eight-port held-input replay spans anomaly $[4,4.1]$, about
140.095 physical seconds, in 204 held intervals. All 1,632 uncertain orbital
continuous-flow enclosures are proved by Arb. All 204 whole-fleet swept
collision checks pass, including finite 0.15 m bodies. Every numerical
endpoint lies inside its corresponding proved flow endpoint. Independent
observations intersect the preceding flow boxes at all 1,624 later agent
updates. The numerical maximum position/sliding occupancies decrease from
0.348/0.200 to below 0.086/0.046. The largest absolute command is below
3.112 m/rad² against the 100 m/rad² component limit.

The replay begins inside the terminal set by construction. It qualifies the
terminal stage and its proof/implementation correspondence; it is not an
approach handoff. Source hashes, raw exact certificates, complete numerical
states, plots and animation are retained outside the repository under the
physical-terminal work products.

The production `PhysicalTerminalRunner` was separately exercised on the same
eight-port finite window with source hashes unchanged. Its 204 held stages,
1,632 flows, geometry fingerprints and all command/flow/geometry SHA links
passed complete serialized readback. Its initial input array and first
recorded state were identical. Feeding the actual ongoing compact-deployment
prefix to this adapter instead returned `REFUSED_INITIAL`, executed zero
holds and preserved every supplied state. That refusal is the expected
behavior before the actual approach enters an admitted terminal set.

## 9. Unresolved Ambiguities

The error budgets and entry spin enclosure are declared assumptions. This
module derives their consequences; it does not certify an attitude sensor,
validate the DOP853 attitude interpolation error or supply a flight clock.
The global derivative bounds cover uncertainty in entry spin, while the
reference-evaluation assumptions must separately cover the actual target
preview used by the controller.

An actual approach endpoint must pass robust admission with its own state
enclosure and exact assigned body ports. The finite-window certificate must
be renewed or handed to an admitted capture phase before it expires. A
refused bound or endpoint is an unresolved sufficient proof, not a proof of
physical impossibility. General ADTMPC recursive feasibility, real-time
hardware execution and contact/capture are separate obligations.
