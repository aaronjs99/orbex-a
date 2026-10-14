# Research red-team review

## 1. Decision and Executive Verdict

**Proceed conditionally.** The supported decision is to develop and evaluate
ADTMPC for tube-contained spacecraft standoff rendezvous, including a congested
post-release deployment phase. The falsifiable contribution is that conservative
nonlinear orbital set adaptation and optimized feedback bandwidth improve
completion/accuracy/control tradeoffs under the same physical limits and sensing
assumptions as credible competing methods. A working demonstration does not yet
establish that comparative claim. Confidence in the narrower implementation
claim is higher than confidence in adaptive superiority or flight readiness.

## 2. Claim-Evidence Map

| Claim | Supplied evidence | Evidentiary dependence | Status | Decisive gap |
|---|---|---|---|---|
| ADTMPC machinery is implemented | Nominal/tube/bandwidth NLP, separate interval membership and point estimation | Model and software artifacts | Artifact-backed fact | Mathematical and numerical assumptions remain explicit. |
| Entire terminal tube fits accuracy | Exact box criterion, endpoint dwell and separately derived finite terminal certificate | Containment, known reference and explicit terminal entry | Conditional result | Ordinary endpoint dwell alone does not establish entry or continued safety. |
| Adaptation causes improvement | Matched adaptive/frozen and repaired adaptive/fixed cases, with independent initial prior | Deterministic same-model, exact-feedback experiments | Narrow empirical result | Smaller tubes in the tested cases; broad superiority, drag-truth sweeps and noisy sensing remain unestablished. |
| Both drag coefficients are identified | Earlier saved point fits and nested sets | Same-model exact observations | Not established | Rank deficiency/confounding; assess identifiable combinations. |
| Common-carrier deployment is handled | New finite arc, cyclic assignment, staging references and readiness checks | Post-release geometry and ideal actuation | Implementation plus bounded reference proof | Release uncertainty, carrier exclusion and commissioning delay are absent. |
| Fleet is real-time | Earlier sequential solve-time records | Computer load and paused plant clock | Not established | Include plant motion during measured planning latency. |
| Viewer improves auditability | Exact saved-data projection, own-chaser tube, synchronized charts and provenance | Rendered records | Artifact-backed fact | Adds presentation value rather than control novelty. |

## 3. Independent Red-Team Findings

Four independent reviewers were used, with separate raw-source tasks. A fifth
review checked implementation of the mathematical remedy.

- **Mathematical validity:** continuous boundary-layer algebra is sound under
  its assumptions, but holding feedback needed an explicit correction. The
  resulting first-exit bound now accounts for held cancellation/damping.
  Exact terminal box support and consistent cost/residual scales reduce
  avoidable conservatism and numerical ambiguity.
- **Evaluation validity:** historical comparisons prove regression equality,
  not causality. Observations and target preview are ideal. The earlier eight
  chaser record had solver-only deadline misses in 42 of 94 updates; sequential
  solve time reached 1.85 times its physical interval. Those timing observations
  apply to that historical run, not every machine or new controller revision.
- **Systems validity:** exact co-location violates positive separation.
  Compact release needs staging and corridor readiness. A numerical stop after
  a violation or failed solve is not an onboard safe abort trajectory.
- **Research value:** dynamic tubes, online uncertainty identification,
  tumbling-target rendezvous and multi-servicer MPC each have close prior work.
  The combination needs an isolated benefit and a precise scope.

## 4. Consensus, Disagreement, and Shared Blind Spots

The initial reviews identified **free-flight translational standoff** as the
supported scope at that stage. Subsequent work added conditionally certified
flow/terminal services and distinct physical lifecycle stages. Those additions
have their own entry conditions and numerical evidence; the original reviews do
not certify them retrospectively. Common model assumptions shared by controller,
plant and replay limit independence: a separately written replay catches many
implementation errors while still sharing the analytical physical model.

The orbital review initially questioned common-drag terms. Primary-source
equations showed that the coefficients match Carter–Humi's stated angular-rate
approximation. This concern is retained as fidelity bias, rather than promoted
to a false coding-error diagnosis. No vote was used to resolve it.

## 5. Ranked Threats

| Issue | Severity | Evidence | Confidence | Consequence | Remedy |
|---|---|---|---|---|---|
| Unisolated adaptive contribution | Major weakness | No matched competing controller in earlier demonstrations | High | Success may arise from nominal point updates or favorable geometry | Paired ablations with identical starting belief and physical limits. |
| Executed hold vs original tube | Major weakness, repaired analytically | Exact admissible-state counterexample | High | Continuous law alone could not predict executed tube | Derived hold forcing, consistent solver/runtime and fresh qualification. |
| Privileged synthetic information | Major weakness for deployment claims | Exact states/preview and config-centered historical priors | High | Robust physical inference unsupported | Independently specified priors and bounded adverse sensing errors. |
| Onboard delay and ordinary-approach fallback | Major weakness for deployment claims | Historical fleet solve-time misses; plant clock paused during solve; separate supervisor requires an admitted backup | High | Offline success cannot establish onboard viability | Delay-aware propagation and admission of a backup for the actual approach state. |
| Parameter confounding | Major weakness for full identification | Historical fits had rank one or two; no rank-three windows | High | Separate drag estimates can remain inaccurate | Report eccentricity and drag-difference sensitivity/accuracy. |
| Statistical generality | Ordinary limitation | Deterministic seed-zero cases sharing one truth | High | Chasers/substeps are correlated, not independent trials | Prospectively selected truth/geometry/noise cases and reported failures. |
| Capture/release fidelity | Ordinary limitation in numerical lifecycle scope | Rigid inelastic latch and bounded nozzle models are implemented; carrier release and contact loads are not | High | Numerical stages do not establish mechanical feasibility | Keep physical assumptions explicit and measure the missing hardware properties. |

## 6. Decisive Tests or Analyses

Use one independently specified physical uncertainty box and the same initial
point within every paired comparison. Select truths prospectively within that
box. Start with compact three-agent geometry, then the full fleet. Preserve
identical dynamics, input limits, target motion, observations, horizons, warm
starts, approach policy and terminal acceptance. Include every failed/incomplete
run in the comparison.

| Variant | Point estimates | Membership box | Bandwidth |
|---|---|---|---|
| Full ADTMPC | Updated | Contracted conservatively | Optimized |
| Frozen-belief DTMPC | Frozen | Frozen | Optimized |
| Point-only | Updated | Initial box retained | Optimized |
| Fixed-bandwidth adaptive tube | Updated | Contracted conservatively | Fairly tuned and fixed |

An identity point estimator alone is not a frozen-belief baseline: membership
can still contract and project the point. Record the frozen policy explicitly.
If a contraction-only variant projects an excluded fixed point, disclose that
projection as part of the method.

Measure completion under unchanged sampled safety/input conditions, physical
completion time, whole-tube terminal bound, minimum physical separation and
clearance, applied effort with declared units, retained true-parameter membership,
set width, held-out transition prediction, and solve-time tails/deadline misses.
Anomaly-integrated squared input is not fuel consumption. Preselect any required
improvement threshold before seeing outcomes.

For three agents, compare all six planning orders: large outcome differences
make scheduling an algorithmic component. For delay stress, keep the plant
moving under an explicit waiting control while replaying measured planning
latency. For sensing stress, include adverse signs at the stated error bound.
Successful random draws cannot prove a universal enclosure guarantee.

Historical reanalysis found strong eccentricity contraction but retained drag
ambiguity. In the default case, drag interval widths did not contract; the final
individual drag point errors were about −36% and −16%, while their difference was
within about 0.6%. This supports evaluating identifiable combinations, not
claiming separate accurate recovery from a visually good trajectory.

## 7. Salvage or Pivot Options

Retain ADTMPC as requested. The most defensible prospective contribution is
conservative nonlinear orbital set adaptation coupled with dynamic feedback
allocation and whole-tube accuracy during congested deployment and rotating
approach. A narrower methods/system paper remains valuable if adaptation's
comparative improvement is modest. A negative result about identifiability or
deadline limitations can also be useful if evaluated rigorously.

The focused literature search used SciSpace for discovery and primary
manuscripts or publisher abstracts for evidence. It was not exhaustive.

| Primary work | Why it matters here | Access scope |
|---|---|---|
| [Lopez, Slotine and How, 2019](https://arxiv.org/abs/1907.06553) | Joint dynamic tube/bandwidth/nominal optimization is established. | Primary manuscript |
| [Morozov et al., 2021](https://arxiv.org/abs/2012.12403) | ADTMPC and SMID performance studies precede this project; linear parameterization assumptions require care. | Primary manuscript |
| [Oestreich, Linares and Gondhalekar, 2023](https://doi.org/10.2514/1.G006438) | Close spacecraft tube-MPC comparator with uncertainty identification and an uncertain tumbling target. | Publisher abstract; full method not inspected |
| [Specht, Bishnoi and Lampariello, 2023](https://doi.org/10.2514/1.G007280) | Existing estimation/planning/tube-tracking pipeline for tumbling rendezvous. | Publisher abstract; full method not inspected |
| [Quartullo et al., journal extension](https://www3.diism.unisi.it/~giannibi/research/papers/jgcd26.pdf) | Moving-terminal conservatism, feasibility and finite completion are already addressed under normalized HCW/additive-disturbance assumptions. | Primary author manuscript; extends the 2024 preprint |
| [Jönsson et al., 2026 preprint](https://arxiv.org/abs/2609.26593) | Multi-servicer 6-DOF MPC is close prior work; its stated uncertainty limitations matter for comparison. | Primary preprint, not established peer-reviewed result |
| [Koenig and D'Amico, 2018](https://doi.org/10.1016/j.actaastro.2018.01.037) | Common-carrier deployment needs uncertain-release and passive commissioning safety comparisons. | Primary author precursor and publisher record |

## 8. Recommendation and Immediate Next Action

**Proceed conditionally.** The corrected held-input implementation, compact
deployment, validated-flow safety layer, terminal/continuation interfaces and
physical lifecycle now have separate source-qualified numerical evidence. The
following completed comparison addresses the matched adaptive/frozen obligation;
it does not establish a novel method merely by implementing established ADTMPC.

### Completed frozen-source comparison, 14 October 2026

Six cases used two finite 0.15 m buses, compact deployment, a gravity-gradient
target, literal eccentricities 0.155 and 0.185, and one independently fixed prior:
eccentricity `[0.14,0.19]`, alpha `[1.5e-7,2.5e-7]`, beta `[3.5e-7,5.5e-7]`.
The initial point was `(0.17,1.8e-7,4.2e-7)`. Drag truths stayed at the declared
interval midpoints, so this is an eccentricity contrast, not a drag-truth sweep.
The frozen variant froze both the point and membership box. All cases used the
same target spin, finite COM geometry and prospective 550-update limit.

| Truth eccentricity | Adaptive updates / final whole-tube position | Frozen updates / final whole-tube position | Adaptive mean tube-radius change | Paired physical squared-acceleration effort change |
|---|---|---|---|---|
| 0.155 | 373 / 1.758509 mm | 373 / 1.780966 mm | −19.25% | +0.110% |
| 0.185 | 372 / 1.802466 mm | 373 / 1.779247 mm | −18.57% | +0.0633% |

These deterministic cases support reduced conservatism with almost unchanged
effort and completion time. At eccentricity 0.185, the one-update difference is
12.72 simulated seconds. Physical squared-acceleration effort uses a stated
outer-midpoint clock approximation; it is not measured fuel consumption.

The remaining approximately 1.4 mm actual goal error is mainly nominal-center
tracking tolerance. Worst final actual-to-center position errors were 0.360 µm
and 0.163 µm for the adaptive cases, versus 36.560 µm and 22.261 µm when frozen.
Those errors concern exact simulated feedback and target preview. Some component
tubes are submicron; they must not be presented as containing an additional 1 µm
sensing error. Sensing-robust statements require the distinct validated state
enclosure and terminal construction.

Both original fixed-bandwidth κ=80 cases stopped on numerical solver failure,
after 96 and 5 committed updates. Their failed physical problems were retained
and each passes a separate corrected-backend replay. The repair removes redundant
fixed-variable constraints and disables native bound relaxation; see
[solver acceptance](nonlinear-solver.md). These failures therefore do not support
dynamic-bandwidth superiority. A separate prospective source cohort completed the
adaptive/fixed comparison after the repair; it never replaces the original ledger.

No attempted update exceeded its physical hold using the recorded sequential
backend time in this cohort. That excludes some model-build/plant work and was
measured during concurrent offline jobs; it is not a real-time eight-agent result.
These matched runs deliberately omitted continuous-flow admission in all variants.
Their dense clearances and tube diagnostics are numerical evidence, separate from
the continuously guarded physical fleet and terminal-window qualifications.

The cohort runtime SHA-256 is
`4ff020ca31ae3d9cacbc297eb0197e3ed5aaa5f30cf43c8a6526f788c22081de`.
The external experiment directories retain the six raw results and 3,194
solver-attempt archives. The portable presentation includes the raw mission
results, paired metrics, failure diagnostics, plots and source/config manifests;
the much larger optimizer archive collection remains separately retained.
Keep failed and interrupted attempts visible when reporting extensions.

### Completed repaired-backend adaptive/fixed comparison

A separate predeclared pair reused eccentricity 0.155, the same independent
prior, release geometry and physical target, changing only the reviewed backend.
Adaptive and fixed-bandwidth cases both completed, in 373 and 374 updates. Their
worst final whole-tube position bounds were 1.758499 and 1.807491 mm; minimum
dense finite-body clearances were 15.224428 and 15.163048 mm.

Over the common completed prefix, the arithmetic mean recorded position-tube
radius was 0.647093 mm with adaptive bandwidth and 1.150082 mm with fixed
bandwidth: a 43.735% reduction in this deterministic case. Physical
squared-acceleration effort was effectively equal (adaptive/fixed ratio
0.999987575), with the difference smaller than the clock-approximation envelope.
Adaptive completion was one update, or 13.524 simulated seconds, earlier.
Both retained the true parameters in every recorded membership box. The
eccentricity width contracted to 0.87890625% of its prior width; the two drag
widths did not contract. Neither run had a nonlinear failure or observed backend
deadline miss. These remain offline exact-feedback numerical outcomes, not
population statistics or a universal advantage of adaptive bandwidth.

The repaired cohort runtime SHA-256 is
`82627f7622eb1418e3a6b81edac1f284abffa66861c631601d52ec2f617bb722`.

## 9. Assumptions and Unanswered Questions

The chosen physical release speed, commissioning delay, carrier exclusion,
thruster authority, state/target sensing accuracy, rigid-body inertia/initial
spin and actuator dynamics are not mission measurements supplied by the user.
The requested modeled mission includes attachment, detumbling and deorbit;
representative physical choices must remain labeled
as modeled scenarios. These choices can change feasibility and should be fixed
before a deployment or experimental claim.
