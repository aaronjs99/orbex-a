# Equation and implementation audit

## 1. Verdict

**Pass with stated conditions for the repaired held-feedback comparison.**
Independent mathematics and implementation reviews found the original
continuous-to-held-feedback transfer invalid and supplied a concrete
counterexample. The repaired derivation and implementation include every held
cancellation, sliding, damping and robust-feedback term. The real-arithmetic
inclusion argument remains conditional. Subsequent work added a separate
validated continuous-flow/swept-geometry layer and conditional stored
continuation and physical terminal constructions. These additions do not
establish recursive feasibility of every ordinary performance NLP.

Four independently scoped reviews covered mathematics, systems, evaluation and
research value. A separate reviewer checked the resulting hold-bound source.
Reviewers worked from raw artifacts and source, not from a vote or a desired
conclusion. Subsequent solver qualification is recorded outside the repository;
this document does not turn its sample counts into universal reliability.

## 2. Conventions and Assumptions

- Position is meters in the opposite-transverse/radial/normal LVLH frame.
- Velocity is position derivative per unwrapped anomaly, in m/rad.
- Control is anomaly acceleration in m/rad²; physical input is multiplied by
  the square of the anomaly rate.
- Vector inequalities and gain products are componentwise; matrix absolute
  values are entrywise, not induced norms.
- The truth parameter lies in the retained box. The nominal point lies in it
  too. The supported interval model uses fixed semi-major-axis context.
- For the hold-inclusion proof, the exact continuous ODE, coefficient/nominal
  enclosures, initial error containment and a valid disturbance bound are assumed.
- Exact state/target observations, ideal full-axis actuation and one shared
  anomaly clock are simulation assumptions. Solver computation does not advance
  the simulated plant clock.
- A numerical residual tolerance is a consistency rule, not an interval proof.

## 3. Symbol, Shape, Unit, and Frame Table

These symbols are defined before the audit uses their relationships. Full
derivations and named intermediate quantities appear in [theory](theory.md).

| Symbol | Meaning and role | Shape / units / frame |
|---|---|---|
| $x,z,e$ | Actual state, nominal state and their difference | Six; m and m/rad; LVLH |
| $s$ | Sliding error, velocity error plus gain-weighted position error | Three; m/rad |
| $\Phi,\Omega$ | Sliding and position error half-widths | Three; m/rad and m |
| $\Lambda,\kappa$ | Sliding gain and ancillary bandwidth | Diagonal three-vectors; 1/rad |
| $h,\tau$ | Outer interval and maximum feedback hold | Scalars; rad |
| $\delta_A,\delta_d$ | Acceleration coefficient/affine uncertainty bounds | 3×6 and 3; mixed-column acceleration map and m/rad² |
| $\bar A,A_{\rm var}$ | Nominal coefficient magnitude and within-hold change bounds | 3×6; same column units as acceleration map |
| $w_0,\rho,w$ | Base forcing, hold correction and total forcing | Three; m/rad² |
| $R_u$ | Complete ancillary input reserve | Three; m/rad² |
| $\mathcal R,n_B,n_W$ | Body-to-LVLH rotation, body/world separating normals | 3×3, three, three; dimensionless |

## 4. Line-by-Line Audit

The locations name current modules and functions, so file movement does not turn
a line-number snapshot into an apparently current proof. The original review
and raw probes remain separate immutable work records.

| Location | Claimed step | Status | Issue or required condition | Correction |
|---|---|---|---|---|
| `core/dynamics.py: orbital_ellp_drag` | Physical-coordinate relative dynamics | Approximation | Carter–Humi neglects relative angular-rate and radial-speed effects; it is not a full atmosphere model. | Preserve named model and quantify omitted physics separately. The original coefficient signs match the primary equations. |
| `control/uncertainty.py: uncertainty_envelope` | Box-wide coefficient error envelope | Valid under condition | Nominal point inside convex parameter box; domain denominators positive; interval derivatives enclose the full Jacobian. Host transcendental rounding is not machine-certified. | Retain full-box/domain checks and conditional arithmetic scope. |
| `control/dynamic_tube_model.py: boundary_layer_feedback` | Continuous cancellation and boundary damping | Valid under condition | Exact continuously evaluated law and valid uncertainty bound. Zero-layer components require contained sliding error zero. | Preserve law; bound the executed hold separately. |
| `control/sample_hold.py: sample_hold_expressions` | Complete held-input correction | Valid under condition | Stage peaks bound the error until first exit; hold width does not exceed its declared maximum. | Added bounded-term correction, including both uncertain-error terms and all linear cancellation changes. |
| `solvers/gekko_dynamic_tube.py: stage tube equations` | Joint forcing, peak and reserve optimization | Valid under condition | Numerically feasible coupled equalities and inequalities. | Use the same solved stage vectors in every expression, retain initialization diagnostics and reject inconsistent outputs. |
| `simulation/coupled_execution.py: _stage_preflight` | Executed bound matches planned bound | Valid under condition | Shape, version, gain, hold, initial-state, peak, reserve and equality consistency checks pass. | Added preflight before actuation; no silent extrapolation of a legacy profile. |
| `simulation/coupled_execution.py: advance` | Runtime nominal and tube remain persistent | Valid under condition | Own runtime center/envelope retained; outer/dense RK4 defect reported rather than hidden by recentering. | Removed unsupported max-width center switch; recorded componentwise defect with distinct units. |
| `estimation/interval_membership.py` | Budget exhaustion preserves consistent parameters | Valid under condition | Outward held-input RK4 evaluation, declared observation boxes, all queued/unresolved boxes retained. | Preserve queue/hull logic and distinguish retained from uniquely identified. |
| `estimation/mission_adaptation.py` | Point fitting cannot overwrite membership box | Valid | Point output is checked against copied authoritative box and exact parameter keys. | Reject malformed/escaping point estimates without replacing the box. |
| `control/constraints.py: CylinderConstraint.margins` | Physical clearance units | Valid | Radial and axial margins now both lengths; their maximum is a conservative exclusion margin near a rim, not exact outside-rim Euclidean distance. | Replaced squared radial margin; independent geometric SDF remains separately reported. |
| `solvers/gekko_dynamic_tube.py: cylinder support` | Position box lies outside target at a node | Valid under condition | Proper rotation and unit separating normal; inflation/reserve consistent. | Use exact rotated-box support instead of a ball radius. |
| `control/terminal_cost.py: local_terminal_cost` | Frozen discrete Riccati tail matches stage weights | Valid | Frozen local model; no invariant-set theorem. | Scale both DARE weights by the outer interval. State running cost ends before the terminal node; endpoint tube regularization remains explicit. |
| `simulation/mission_policy.py: terminal_bounds` | Entire tube satisfies endpoint accuracy | Valid under condition | Error containment established under the declared model. | Exact box norm replaces triangle bound for both position and velocity. |
| `solvers/gekko_dynamic_tube.py: terminal inequality` | Small physical tolerance is enforced meaningfully | Valid under condition | Dimensionless post-solve residual consistent with optimizer settings. | Normalize squared norm by squared tolerance; do not compare a square-meter residual to a generic length tolerance. |
| `simulation/deployment*.py` | Compact launch reference avoids geometric conflict | Valid under condition | Finite initial pitch, cyclic order, shared proper rotation and safe radius. | Arc-gap proof plus current-state/tube corridor gate. This proves reference geometry, not actual trajectory feasibility. |
| Dense/node obstacle and dwell checks | Continuous physical safety and future feasibility | Insufficient alone | Samples do not cover the complete hold or supply a safe successor. | Use the separate admitted certificate layers; do not infer them from sample density. |
| `control/reachability.py` and `simulation/certified_execution.py` | Continuous orbital flow and swept finite-body exclusion | Valid under condition | Arb Picard enclosure, original prior, input/measurement/process/pose bounds and complete hold coverage. | Fail closed on unresolved or malformed certificates; bind flow/geometry/request identities and exact clock. |
| `control/continuation.py` | Solver-failure admissible continuation | Valid under condition | A complete backup chain is admitted before performance control; each successor lies in the next certified entry set. | Retain the stored command on failure/rejection. An ordinary approach stop or checkpoint restart is not this theorem. |
| `control/physical_terminal.py` and `simulation/physical_terminal_execution.py` | Finite-domain reference-centered invariance | Valid under condition | Derived eccentric-orbit/rigid-target bounds, actuator/sensing assumptions and real entry-box membership. | Refuse nonmembers without state projection; certify held flow and whole-fleet swept geometry. |
| `simulation/capture.py` and physical lifecycle | Attachment and subsequent disposal dynamics | Numerical model | Finite bus/reach/attitude/speed gates, known mass/inertia/nozzles and ideal inelastic latch. | Preserve actual physical states and failed stages; audit momentum, force/fuel and descending entry events. |

## 5. First Consequential Error

The original comparison assumed cancellation updated continuously while the
simulator held the computed feedback. Smaller holds reduce the error but do not
make that substitution exact. A no-uncertainty example violated the original
next sliding envelope even at the default hold width.

The subsequent implementation review exposed two numerical boundaries: a finite
seed iteration could stop before a feasible NLP was attempted, and a squared
terminal residual could accept a physical error larger than its millimeter
tolerance. Seed failure now changes initialization only; normalized constraints
and stricter consistency checks preserve a meaningful acceptance scale.

## 6. Corrected Derivation

Let $E$ enclose the complete state error over a stage. Let $D_e,D_v,D_s$ bound
the derivatives of complete error, velocity error and sliding error. Let $q_j$
be the hold's initial anomaly and $r_j$ its bounded uncertain-error feedback
term. Inside the declared tube, the current held law can be rewritten exactly as

$$
u_j=-\widehat A_a(q_j)e_j-\Lambda e_{vj}-\kappa s_j-r_j,
\qquad |r_j|\le\delta_AE.
$$

This identity remains valid at zero layer width because the contained sliding
component is then zero. Adding/subtracting current linear cancellation yields
the residual bound

$$
\rho=2\delta_AE+A_{\rm var}E+
 \tau(\bar A D_e+\Lambda D_v+\kappa D_s).
$$

The first-exit comparison uses $w=w_0+\rho$ and the jointly optimized stage
peaks/reserve. It does not count $\rho$ a second time in physical derivative
bounds: it is an algebraic residual assembled from those same held dynamics.
See equations (10)–(13) of [theory](theory.md) for the complete construction.

## 7. Domain-Specific Checks

- **Frames:** rotated-box support is $|\mathcal R n_B|^T\Omega$, since the error
  box is aligned with LVLH axes, not target body axes.
- **Dimensions:** base/hold forcing and reserve are m/rad². Cylinder margins
  are m. Terminal residuals are dimensionless after normalization.
- **Comparison:** constant-source sliding and peak-driven position comparisons
  are componentwise monotone, so their endpoint peak constraints suffice.
- **Terminal geometry:** worst Euclidean error of a translated box is
  `norm(abs(nominal error)+half-width)`, attained at an aligned corner.
- **Optimization:** initialization is not a feasibility oracle. A local feasible
  optimum is neither a global optimum nor a recursively feasible policy.
- **Observability:** contraction of eccentricity and accurate drag difference
  do not establish separate identification of both drag constants.

## 8. Sanity and Numerical Checks

Independent bounded examples, executed outside the checkout, include:

| Probe | Observation and interpretation |
|---|---|
| Original held-feedback witness | Old envelope missed exact next sliding error by $8.8961\times10^{-7}$ m/rad; RK4/exact difference about $5.5\times10^{-16}$. |
| Repaired circular comparison | All 64 initial six-dimensional sliding-box corners remained enclosed through 32 exact held steps; minimum position margin about $1.88\times10^{-8}$ m. Bounded example only. |
| Zero layer and malformed profiles | Finite bound at zero width; unsupported hold/profile/version refused before state commit. |
| Box support and terminal norm | Matched exhaustive corner calculations under a nontrivial rotation. |
| DARE homogeneity | Scaling both weights scales the tail; floating discrepancy about $4\times10^{-14}$ in the bounded example. |
| Slow seed witness | A positive fixed-point map with spectral radius 0.9886 has a finite solution despite 200-iteration nonconvergence; a forced finite initializer still allowed a fully constrained actual nonlinear solve. |
| Compact deployment reference | Finite-separated 8-agent geometry retained 0.45 m minimum reference spacing; independent derivative checks covered transition knots and interiors. |

Actual full mission results have their own saved source/configuration, outcomes
and raw histories. Failed and interrupted qualification attempts are retained;
they are not relabeled as completed demonstrations.

## 9. Remaining Scope and Independent Review

The physical-terminal bound construction received a separate ten-part review:
orbital coefficients and derivatives, rigid-spin first exit, reference derivatives,
held forcing, input reserve, tolerance, geometry, runtime enclosure and entry
membership. A source-qualified initialized eight-port window exercised 204 held
intervals and 1,632 continuously enclosed orbital flows. Initialization inside
that set does not establish admission of a different approach endpoint.

The ordinary guarded approach still needs an already admissible backup before
claiming continued physical safety after an NLP failure. Numerical checkpoint
restart preserves evidence and state; it does not supply that control proof.
Sensed target bounds and disturbance reserves remain caller declarations, not
measured properties. Solver-delay plant evolution, valve/pressure dynamics,
uncertain release, latch loads and atmospheric demise require additional models
or experiments. Arbitrary reserves and denser plots cannot replace them.
