# Continuous held-input reachability and swept safety

## Verdict

**Pass with stated conditions for the continuous affine model and a supplied
held candidate.** `control/reachability.py` provides a validated flow bridge,
including numerical rounding and the complete intersample motion. A successful
flow certificate is independent of the executor's RK4 approximation. A
successful collision certificate separates that whole flow from declared
occupied geometry. The guard establishes admissibility of a particular hold;
existence of an admissible backup and recursive feasibility require additional
conditions described below.

The optional dependency is `python-flint` (the bounded examples used version
0.8.0 on CPython 3.11, Windows x64). Imports remain lazy. Missing arithmetic
returns `UNAVAILABLE`; exhausted resources, unresolved domains and failed
separation return `UNRESOLVED`. Neither permits execution. Unexpected program
exceptions propagate. No failure is converted to a positive certificate.

## Conventions and assumptions

The frame is `physical-lvlh-v1`: the x axis is opposite transverse motion, y is
radially outward and z follows orbital angular momentum. The independent
variable is unwrapped true anomaly. State velocities are derivatives with
respect to anomaly, not seconds. Physical geometry uses metres.

The certificate is conditional on all of the following premises:

1. The actual free-flight state satisfies the maintained analytical affine
   orbital model throughout the declared hold. All omitted acceleration is
   contained in the supplied acceleration-disturbance box. This is a model
   premise; it does not validate an atmosphere, J2 model, contact law or hardware.
2. The initial state, actual held acceleration, start clock and actual hold
   duration belong to the explicitly supplied boxes. Nonconstant actuator error
   is also allowed if its acceleration remains within those input bounds.
3. The true eccentricity, both drag constants and phase lie in the complete
   parameter box. The semi-major axis and gravitational parameter have the
   declared fixed values. The retained orbital denominator is positive.
4. All occupied chaser shapes fit inside their declared physical balls. The
   cylinder bounds enclose the target, its initial pose, and its complete
   relative rotation and translation over the hold. Those are geometric
   assumptions, not consequences of observing a floating target pose.
5. Arb, Python rational arithmetic and the execution platform behave according
   to their specifications. This is a numerical certificate with an explicit
   implementation trust boundary, not a machine-checked proof of that software.

For a trajectory spanning several holds, initialize each successor from its
predecessor's **certified endpoint box**, or intersect that box with a new
measurement box known to contain the true state. Resetting to an RK4 point at
every hold establishes isolated statements about different initial states; it
does not establish one continuous trajectory. A numerical parameter contractor
for an RK4 map cannot establish the continuous-model parameter premise without
a separately valid model-mismatch bound.

## Symbol, shape, unit, and frame table

| Symbol | Meaning and role | Shape and units |
|---|---|---|
| q | Actual unwrapped anomaly clock | Scalar, rad |
| s | Elapsed anomaly since the hold starts | Scalar, rad |
| Q0 | Box for the actual start clock | Interval, rad |
| H | Box for actual nonnegative hold duration | Interval, rad |
| x, p, v | State, position, and anomaly derivative of position | 6, 3, 3; m and m/rad |
| X0 | Initial measurement or predecessor endpoint box | Box in state coordinates |
| u, eta | Held acceleration and additional bounded acceleration | 3; m/rad² |
| theta | Eccentricity, target drag, chaser drag and phase | Four parameters; phase in rad |
| a, mu, h | Semi-major axis, gravitational parameter, angular-momentum constant | m, m³/s², m²/s |
| T, Y, F | One clock interval, candidate swept state box, derivative enclosure | Interval, state box, state-derivative box |
| delta | Width of one integration substep | Scalar, rad |
| K | Body cylinder occupied set | Set in body coordinates, m |
| r, ell, Rc | Radius, half-length, circumsphere radius of the target | Scalars, m |
| R0, M, eM | True initial rotation, supplied approximate matrix, its norm error | 3×3, 3×3, dimensionless |
| n | Fixed nonzero separating normal in the common frame | 3; arbitrary scale |
| W, V | Bounds on relative angular speed and center speed per anomaly | 1/rad, m/rad |
| b, c | Chaser physical radius and additional required clearance | Scalars, m |

## Line-by-line audit

| Location | Claimed step | Status | Issue or required condition | Correction |
|---|---|---|---|---|
| `_exact`, `_ArbArithmetic.scalar` | Interpret supplied floating values | Valid | Float means its exact binary rational value, not an intended decimal | Strings or `Fraction` express exact decimal/rational bounds |
| `_ArbArithmetic.interval` | Enclose all parameter/clock values | Valid under condition | Arb's ball operations obey containment semantics | Midpoint/radius rounding is included by Arb |
| `enclose_orbital` | Enclose every orbital coefficient | Valid under condition | Positive denominator and h² over the complete domain | Evaluate sin, cos, exp and nonlinear parameter products as balls |
| `_picard` | Prove continuous swept containment | Valid under condition | Strict componentwise self-inclusion | Check `contains_interior`, not agreement between sample values |
| `_integrate` | Compose continuous endpoint boxes | Valid under condition | Complete finite partition, no unproved gap | Exact rational durations and finite adaptive subdivision |
| `_target_segment` | Separate the rotating finite target | Valid under condition | Declared pose/rate/geometry bounds | Include matrix error, complete rotation reserve and translation reserve |
| `_box_distance_squared` | Separate every chaser pair | Valid under condition | Common frame and clock, finite body bounds | Exact rational minimum box distance over overlapping swept segments |
| Candidate/backup integration | Recursive feasibility | Not established by this guard alone | Need a backup at every certified successor | Supply an invariant admissible backup set or a valid terminal argument |

## First consequential error addressed

A sampled RK4 trajectory, even one with small steps and safe endpoints, has an
unproved integration remainder and can miss an intersample collision. Bounds
on its difference from another sampled trajectory do not repair that gap.
This module uses a continuous integral inclusion instead. It changes neither
the optimizer nor the executor's numerical integrator; their outputs may
propose a candidate, while this independent guard accepts only a proved hold.

## Corrected derivation

### Full-domain orbital coefficients

Let eccentricity be e, target drag be alpha, chaser drag be beta, and orbital
phase be q0. Let E be the exponential factor, D the orbital denominator, r_o
the orbital radius, k its logarithmic derivative, C its curvature coefficient,
gamma the drag difference and d the relative damping coefficient. Their
definitions, including the eccentricity dependence of h, are

```math
h^2 = mu a (1-e^2),\qquad
E=\exp(2\alpha q),\qquad D=E+e\cos(q-q_0),
```

```math
r_o=\frac{h^2(1+4\alpha^2)}{\mu D},\qquad
k=\frac{e\sin(q-q_0)-2\alpha E}{D},\qquad
C=\frac{(1+4\alpha^2)E}{D},\qquad
\gamma=\beta-\alpha,\qquad d=2k-\gamma.
```

The acceleration block is A_a and its affine forcing is f_a. Both are
evaluated over the full anomaly and parameter boxes, including every interior
parameter value:

```math
A_a=\begin{bmatrix}
1-C&-d&0&d&2&0\\
d&1+2C&0&-2&d&0\\
0&0&-C&0&0&d
\end{bmatrix},\qquad
f_a=\begin{bmatrix}\gamma r_o\\-\gamma r_o k\\0\end{bmatrix}+u+\eta.
```

The top block is p'=v. These are the coefficients of
`core/dynamics.py:orbital_ellp_drag` with an authoritative semi-major axis.
No parameter-corner extremum assumption is used. Dependency overestimation
can widen the enclosure or prevent a proof; it cannot justify discarding an
interior parameter value.

### Continuous Picard enclosure

Within one proposed substep, T encloses every actual anomaly and X encloses
its initial state. Let A(T,Theta) and f(T,Theta) enclose the matrix and forcing.
For any state box Y, define F(Y) as the componentwise affine derivative
enclosure. Define P(Y) as the integral Picard box:

```math
F(Y)=A(T,\Theta)Y+f(T,\Theta),\qquad
P(Y)=X+[0,\delta]F(Y).
```

The implementation accepts a substep only when every component of P(Y) is
strictly inside the corresponding component of Y. Suppose a solution first
leaves Y at elapsed anomaly s. Before that exit its derivative belongs to the
convex box F(Y); integration gives its state in X+sF(Y), which lies in P(Y),
strictly inside Y. This contradicts the first exit. The affine coefficients
are bounded and locally Lipschitz in state on the positive-denominator domain,
so finite-time continuation is valid; bounded measurable additive forcing
gives the corresponding absolutely continuous solution. Thus the complete
substep stays in Y.

Let X_next denote the endpoint enclosure. Since the integral average of a
box-valued derivative remains in that convex box,

```math
x(s)\in Y\quad(0\le s\le\delta),\qquad
X_{next}=X+\delta F(Y).
```

This includes all local integration error without estimating an RK4 or Taylor
remainder. It is conservative first-order interval integration. Failed
self-inclusion triggers exact halving; a successful step may grow again, up
to `max_step`. The remaining exact duration is evenly partitioned to avoid a
spurious sub-machine-size final tail. `min_step`, `max_substeps` and a finite
Picard iteration limit bound the work. Exhaustion returns no usable endpoint.
Strict padding and inflation only propose a candidate Y; they are never
substitutes for the inclusion predicate.

When the actual stop time is uncertain, the terminal box is the union of
validated partial-step integral boxes over all elapsed times in H. The swept
box covers the complete latest possible hold. A zero-duration hold returns
the supplied initial box.

Let nu denote the positive anomaly rate and v_t the LVLH coordinate derivative
with respect to seconds. The optional physical-velocity and elapsed-time
enclosures use

```math
\nu=\frac{h\exp(-\alpha q)}{r_o^2},\qquad
v_t=\nu v,\qquad dt/dq=1/\nu.
```

The reported physical velocity is a coordinate derivative in LVLH. It is not
an inertial relative velocity without the frame-rotation term.

### Swept target separation

The cylinder is the body set K with radius r and half-length ell; Rc is its
circumsphere radius. Let a0 be its unit axial direction at the initial pose.
For a fixed common-frame normal n, its support at that pose is

```math
R_c=\sqrt{r^2+\ell^2},\qquad
h_{R_0K}(n)=r\sqrt{\|n\|^2-(n^Ta_0)^2}+\ell|n^Ta_0|.
```

Euler-angle boxes are converted with Arb trigonometric functions, so the
true axis is included without assuming a rounded matrix is exactly orthogonal.
Alternatively, let v_M be M transpose times n for a supplied matrix M. The
support of its linearly transformed cylinder is exact for arbitrary M:

```math
v_M=M^Tn,\qquad h_{MK}(n)=r\sqrt{v_{M,x}^2+v_{M,y}^2}+\ell|v_{M,z}|.
```

The caller declares a bound eM on the operator norm of M minus the true proper
initial rotation. Thus the initial support error is at most eM Rc ||n||.
The implementation rejects matrices that violate necessary Gram-matrix or
determinant compatibility conditions; passing those checks does not measure
or independently establish the declared pose error. This interface is usable
near an Euler singularity.

Let s_max be the upper elapsed anomaly of a swept segment. Since the norm of
a relative rotation derivative is bounded by relative angular speed, the
rotation and center-translation reserves are

```math
\|R(q)-R_0\|_2\le \min(2,W s_{max}),\qquad
\Delta_c\le V s_{max}.
```

Let P be that segment's swept position box and C0 its target-center box at
hold start. Let h0 be either initial support bound above and let b be the
chaser's physical ball radius. A separating plane proves clearance c whenever
the Arb lower bound of the following expression is strictly positive:

```math
\inf_{p\in P,c_0\in C_0}n^T(p-c_0)-h_0(n)
-\|n\|\{b+c+R_c[e_M+\min(2,W s_{max})]+V s_{max}\}>0.
```

For Euler pose boxes eM is zero. The matrix error is included only for the
matrix interface. Supplied normals and heuristically proposed directions may
be arbitrary nonzero rational vectors: their quality affects conservatism,
while the independently evaluated support inequality supplies the proof.
Future floating poses are not treated as certified poses. A rotating target's
circumsphere alone would also be valid, but is unnecessarily conservative for
side docking close to a long cylinder.

### Swept pair separation

For every pair of overlapping time segments, let P_i and P_j be the swept
position boxes. Let g_k be their nonnegative interval gap along component k
and let D_req be the larger of the required center separation and the sum of
the two physical radii plus clearance. Exact rational arithmetic checks

```math
g_k=\max(0,P_{i,k}^- -P_{j,k}^+,P_{j,k}^- -P_{i,k}^+),\qquad
\sum_k g_k^2>D_{req}^2.
```

Both flows must use the same clock bounds and coordinate frame. Different
adaptive partitions are compared over all overlapping segments. Time
correlation is discarded conservatively. Coincident boxes and touching bodies
cannot pass. The actual start and stop events must be those of one joint hold;
equal numerical clock intervals alone do not establish synchronization of
independently started maneuvers.

### Candidate and backup composition

An evaluated candidate can be certified as follows: enclose its complete hold,
check the swept geometry, check all applicable state and actuator bounds, and
feed its certified endpoint into the successor certificate. A finite backup
sequence can be composed by the same endpoint-containment argument. This
establishes a finite safe sequence, conditional on its model and geometry
bounds.

Recursive feasibility additionally requires an admissible backup from every
possible successor state and clock, including the latest permitted execution
delay. For an infinite-horizon claim, a terminal set must itself be invariant
under an admissible backup policy, with all input, state and geometry bounds
proved. An optimizer returning a previous plan, zero command, sampled hover,
or a finite list of certified holds does not supply that invariant terminal
argument. A guard that rejects all candidates must report a control-authority
or unresolved condition; it cannot relabel an unchecked fallback as safe.

## Domain-specific checks

`angular_rate_per_anomaly` is a bound on the norm of relative angular velocity
between target and LVLH, valid over the entire hold. For prescribed XYZ Euler
rates, the sum of their absolute upper bounds is a valid norm bound. For a
physical inertial spin bound omega_max and a strictly positive anomaly-rate
lower bound nu_min, a valid conservative bound is
`omega_max/nu_min + 1`, provided the orbital frame rotates at anomaly rate.
A spin value at a numerical node alone is not a whole-hold bound when torque
or unconstrained rigid-body motion can increase it. Clock error, pose error,
actuator bounds and disturbance bounds must be supplied in the declared units.

This free-flight certificate applies before contact. Contact, attachment,
detumbling and deorbit need their own mode dynamics and transition contracts.
Physical deployment correctness and allowable thrust cannot be inferred from
this affine flow computation.

## Sanity and numerical checks

The following bounded development probes were run outside the repository,
under the workspace `work/completion-2026-10-14/reachability` directory. They
are evidence about these examples, not an exhaustive proof of the software.

| Example | Observed result |
|---|---|
| Zero one-dimensional field with a non-dyadic initial interval | Exact initial interval retained; six validated substeps across a binary .01 hold |
| One-dimensional identity field | Endpoint contains independently evaluated Arb exp(.01) |
| No-drag circular six-state model, .00049 rad | Endpoint contains the analytic mapped HCW solution evaluated independently at 192 bits |
| Interior eccentricity/drag values | Three independent high-accuracy numerical trajectories stayed inside the full-box flow; those numerical comparisons alone are not proof |
| 5 mm finite-body standoff, relative rotation bound 1/rad | Strict separation proved with a complete rotation reserve |
| Opposite safe endpoints with an intervening cylinder crossing | Guard returns UNRESOLVED and refuses the candidate |
| Coincident chasers / separated chasers | Coincident pair refused; separated pair proved |
| Domain failure and explicit budget exhaustion | Both return UNRESOLVED without a usable endpoint |
| Matrix pose near a pitch singularity / exact reflection | Matrix pose certified in a far-field example; incompatible reflection rejected |

One further bounded size probe used a radius .8 m, half-length .3 m cylinder,
a .150 m chaser ball, .165 m COM standoff beyond the target surface, .00049 rad
hold, a relative spin bound 1/rad and matrix error 1e-8. The initial position
noise was ±1e-6 m, anomaly-velocity noise ±1e-4 m/rad; eccentricity was
[.17,.19], target drag [1e-7,3e-7], chaser drag [3e-7,5e-7], and added
acceleration ±1e-5 m/rad². The swept position widths were approximately
[3.6672,2.6553,2.6553] micrometres and the certified separation lower bound was
.0145795 m. In 100 repeats on the development machine, flow plus target guard
took about .396 ms at the median and .562 ms at the observed maximum. These
are bounded timing observations; long uncertain sequences, wider boxes,
unfavorable geometry and subdivision can cost much more. No real-time deadline
or fallback existence follows from this timing sample.

## Unresolved ambiguities

Bounds on omitted physical acceleration and target pose evolution need an
independent source. Parameter boxes fitted to numerical trajectories need a
continuous-model consistency argument before serving as physical guarantees.
The reachability and geometry guard cannot prove those input premises.

Arb's contract is documented in the primary
[FLINT real-ball reference](https://flintlib.org/doc/arb.html): operations enclose
all point choices, and strict containment is a set predicate. The
[python-flint real-number interface](https://python-flint.readthedocs.io/en/stable/arb.html)
documents directed endpoint extraction and exact mantissa/exponent export.
The implementation extracts those exact rational bounds; printed decimals are
for inspection. FLINT also documents the usual software-correctness limits in
its [technical conventions](https://flintlib.org/doc/issues.html).
