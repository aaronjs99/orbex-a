# Research review and implementation decisions

Review date: 19 October 2026. Baseline: `c30e833`. This is a focused primary-source review, not an exhaustive novelty search. SciSpace was used for discovery; the links below identify the primary material inspected. Preprints are distinguished from published articles, and abstract-only access is stated explicitly.

## 1. Decision and executive verdict

**Proceed conditionally.** The worthwhile research objective is a reproducible eight-chaser mission under a declared nonlinear dynamics, sensing, actuation, contact and resource contract. A combination of adaptive tubes, MPC and multiple chasers is insufficient as a novelty claim. Mission progress and computation must be measured alongside safety.

The closed 300 m central-case run `full-mission-c30e833-20261018T183651` failed before attachment. It reached 959.876949 s of simulated physical time and exhausted its installed finite fallback. Its last replacement was refused because the reserved fuel did not cover the proposed complete fallback. There is no attachment, detumbling or disposal result from this arm.

The archive contains 31,941 completed holds; the saved approach contains 31,936. Five acknowledged holds were lost from the compact recorder when the enclosing update raised an exception. The strict verifier correctly refused the mismatch. Neither a zero process exit code nor a successful numerical optimizer result establishes mission completion.

## 2. Claim–evidence map

| Claim | Evidence inspected | Present conclusion |
| --- | --- | --- |
| The optimizer is numerically usable | 7,984 successful native solves in 7,985 attempts in the closed arm | Solver return is not command admission or arrival. |
| Most execution follows the optimizer | 10,509 performance holds out of 31,936 compactly saved holds | False for this arm: about 32.9% were performance holds. The five-hold recording defect prevents treating these as complete-archive counts. |
| The finite free-flight argument is internally coherent | Held/sensed comparison bounds, regional Arb residual enclosure, full nonlinear flow, whole-set peer/component and successor gates | Conditional on the stated model, supports, exact clock and hold schedule. No consequential local algebra error was identified in this review. |
| Physical parameters adapt usefully online | All four initial drag candidates survive; physical hull equals the prior | Not demonstrated. Initial GNSS orbit conditioning explains the observed eccentricity contraction. |
| The implementation meets its online schedule | All 999 attempted outer updates miss their physical-period deadline; median full attempt 26.01 s versus approximately 0.962 s available | Not demonstrated. The sequential offline simulator does not advance physical time while computing. |
| Capture and post-capture loads have continuous robust certificates | Finite compliant contact with numerical RK4 and sampled/current-state load guards | Not established; these are separate hybrid and structural proof obligations. |
| Full deployment-to-disposal mission completes | Closed source/configuration-bound arm | False for this arm. Downstream stages were never reached. |

## 3. Independent review findings and literature

Four independent reviews covered mathematics, evaluation, systems realism and research value. They used the same baseline and therefore are complementary inspections, not statistically independent experiments.

| Primary work and access | What changes the ORBEX-A decision | Assumptions that must not be silently transferred |
| --- | --- | --- |
| [Lopez, Slotine and How, *Dynamic Tube MPC*, 2019 preprint](https://arxiv.org/abs/1907.06553), paper inspected | Optimizing tube geometry and boundary-layer feedback bandwidth is established prior work. Retain the joint nominal/tube formulation and measure its benefit. | The uncertainty and sliding-feedback hypotheses still need to hold for the delivered sampled commands. |
| [Morozov et al., *Performance Analysis of Adaptive Dynamic Tube MPC*, 2020 preprint / 2021 SciTech](https://arxiv.org/abs/2012.12403), primary abstract inspected | ADTMPC/DTMPC/TMPC comparisons and experimental adaptive-tube benefits predate ORBEX-A. Isolate a mission-specific improvement under shared conditions. | Pendulum experiments do not establish spacecraft identification, contact or online fleet performance. |
| [Köhler et al., nonlinear robust adaptive MPC](https://arxiv.org/abs/1911.02899), primary paper inspected | Estimator updates, uncertainty descriptions and tube construction need compatible monotonicity conditions. | A smaller parameter box alone does not prove recursive feasibility. |
| [Rebollo et al., *MPC for Tracking Applied to Rendezvous with Tumbling Targets*, 2024](https://arxiv.org/abs/2403.10986), full text inspected | Envisat tracking with artificial reference trajectories and terminal ingredients is a close baseline. Include a strong tracker under the same admission rules. | HCW translational dynamics, realizable chaser attitude and the specified target propagation differ from uncertain nonlinear free flight and contact. |
| [Quartullo et al., variable-horizon robust MPC for moving targets, 2024 preprint / 2025 article](https://arxiv.org/abs/2410.08807), full text inspected | Terminal sequences and horizon adaptation directly address final-distance conservatism and finite-time interception. | The proof is for the stated discrete linear system, bounded disturbances and known moving target. It is not a nonlinear exact-contact theorem. |
| [Quartullo et al., *Robust Variable Horizon MPC for Rendezvous with Rotating Target*, JGCD, online December 2025](https://www3.diism.unisi.it/~giannibi/research/papers/jgcd26.pdf), primary author manuscript inspected | The journal extension includes Envisat and Monte Carlo benchmarks and adaptive terminal constraints for finite completion to a final set. It is a particularly close final-distance comparator. | The normalized HCW/additive-disturbance assumptions and finite acceptable final set must remain explicit. |
| [Wabersich and Zeilinger, predictive safety filtering](https://arxiv.org/abs/1812.05506), primary formulation inspected | Keep a performance proposal separate from the policy that enforces safety. A complete valid continuation must remain available after every accepted proposal. | A finite backup window does not supply an invariant terminal refuge or infinite viability. |
| [Didier et al., predictive stability filters, 2024](https://arxiv.org/abs/2401.11183), primary abstract/formulation inspected | Safety and convergence need separate conditions; inspect decrease of an appropriate extended-state value function. | A pointwise safety filter or successful NLP does not prove progress. |
| [Leeman, Köhler and Zeilinger, robust nonlinear MPC via disturbance feedback, 2025 preprint](https://arxiv.org/abs/2509.18760), primary abstract inspected | Joint nominal, feedback and error-bound design via successive convexification is a useful alternative proposal architecture. | Its recursive-feasibility and stability results require its own model, feedback and terminal assumptions. No superiority over the present implementation was measured. |
| [Buerger and Cannon, ellipsoidal-tube adaptive NMPC, 2026 preprint](https://arxiv.org/abs/2603.05029), primary abstract inspected | Bounded linearization errors and feasibility-preserving search offer a principled route to cheaper nonlinear proposals. | The specified parameter-affine basis model and robust terminal ingredients require a new derivation for ORBEX-A. |
| [Frey, Nurkanović and Diehl, AS-RTI, 2024](https://arxiv.org/abs/2403.07101), primary abstract inspected | Separate a computational preparation phase from a short feedback phase; benchmark the complete accepted-command latency. | Local convergence/error results do not establish global collision or contact safety. GEKKO remains the reference backend here. |
| [Behrendt et al., time-constrained spacecraft MPC, 2025](https://arxiv.org/abs/2501.13236), primary abstract inspected | Six-degree-of-freedom rendezvous evaluated with space-grade processing is a more relevant timing reference than a desktop solver return alone. | Different plant and constraints prohibit a direct raw-runtime comparison. |
| [Xu et al., distributed cascaded multi-spacecraft capture, *Advances in Space Research*, July 2026](https://www.sciencedirect.com/science/article/abs/pii/S027311772600517X), publisher abstract/introduction/contributions inspected; full paper unavailable | Close prior work already combines multi-spacecraft tumbling-target capture, CLF/CBF layers, pose control, collision avoidance and deadlock handling. | Its reported accuracy is not a comparison against ORBEX's uncertain whole-set contact gates. Do not adopt its performance or global-optimality claims without the complete formulation and matched experiments. |
| [Jönsson et al., decentralized multiple-servicer rendezvous and docking, September 2026 preprint](https://arxiv.org/abs/2609.26593), primary abstract inspected | A recent decentralized 6-DOF MPC proof of concept includes soft docking and inter-servicer/obstacle constraints. | The stated tractable LTI formulation and three-servicer MATLAB study with fixed obstacle arrangements do not cover this uncertain nonlinear eight-agent lifecycle. |

The mathematical review distinguishes an unsupported guarantee from a demonstrated numerical accident. No collision or attachment overload is inferred merely because a whole-set contact certificate has not yet been constructed.

### The tube edge is a contact-set issue

Let \(\bar d\) be nominal signed normal distance from a hull patch, and let \(r_n>0\) bound the normal error of every admissible state. Before contact, sufficient whole-tube nonpenetration is

\[
  \bar d-r_n\geq0.
\]

Placing the nominal point at \(\bar d=r_n\) lets the nearest admissible state touch, while another admissible state can remain at distance \(2r_n\). Exact simultaneous point contact for every state is impossible under that nonzero two-sided uncertainty. Changing the animation or moving the nominal target does not resolve this geometry.

The mission therefore needs a finite hybrid capture set: patch footprint, approach cone, nonpenetration, closing-speed and compliant-pad travel bounds, followed by a modeled contact/latch transition. The current contact-admission gates are useful sufficient entry conditions. The subsequent contact integration and post-lock held-load bounds still need their own whole-step uncertainty and integration-defect treatment.

Dynamic goals and dynamic obstacles can use a common target pose/geometry forecast, but impose different constraints: tracking asks for entry into a moving acceptable set; avoidance asks for exclusion from moving occupied sets throughout the hold. Their uncertainty and terminal conditions are not interchangeable.

### Coordinated braking and the failed successor

For initial fleet velocities \(v_i\), define the common velocity \(v_c=N^{-1}\sum_i v_i\) and relative velocity \(w_i=v_i-v_c\). The existing coordinated reference is

\[
 p_i(t)=p_i(0)+v_c b_{T_c}(t)+w_i b_{T_r}(t)+d s_{T_c}(t),
 \qquad b_T'(t)=(1-t/T)^3,
\]

until the corresponding stopping time; \(b_T(T)=T/4\), and \(s_T\) is a quintic unit step with zero endpoint velocity and acceleration. Shared terms cancel in every same-time peer difference. This helps preserve a compact deployed formation, but its common stopping displacement is \(v_cT_c/4\): a slow common stop can approach the solar wing too closely.

A component replay from the first fallback's adjacent acknowledged endpoint reproduced the failure: the original coordinated route could not prove solar-array separation; the independent alternatives could not prove one peer separation. Shortening the shared duration to the next rational-grid candidate produced a complete 89-segment nonlinear finite-refuge certificate with 10,324 geometry records. The minimum peer squared-distance surplus was approximately 0.43246 m²; maximum reserved fuel was approximately 2.00908 kg per chaser. More aggressive candidates were refused by the unchanged hardware input check. This is a discriminating component result, not a replay of the unavailable refused performance endpoint and not a completed mission.

The candidate generator searches finite common-duration scales \(1,9/10,4/5,2/3\), rounds to the existing exact hold grid, and preserves the faster relative stop. Every proposal goes through the original model, sensing, input, force, fuel, geometry and full-entry proofs. No physical limit is increased to obtain acceptance.

## 4. Consensus, disagreement and shared blind spots

All reviews agree that finite safety, mission completion, useful identification and real-time execution are distinct claims. They also agree that adding another solver or attractive visualization cannot supply a missing continuation or contact proof.

The finite held/sensed free-flight argument survived the inspected local mathematical checks. This is compatible with a failed mission: valid finite retreat can consume time and reserve while never reaching attachment. The reviewers did not perform independent high-fidelity hardware or flight validation. Both the numerical plant and its interval guard currently share the declared central-gravity/J2/drag model.

## 5. Ranked threats

| Issue | Severity | Evidence | Confidence | Consequence | Remedy |
| --- | --- | --- | --- | --- | --- |
| Finite refuge cannot be renewed | Critical | Closed run, explicit fuel-reserve renewal refusal followed by expiry | High for this arm | Mission stops before contact | Reduce justified resource overbounds, expand admissible continuation candidates, then demonstrate timely renewal and arrival. |
| Deadline misses are hidden from physical-time evolution | Critical for online claims | Sequential runner and 999/999 measured misses | High | The offline hold theorem's schedule is not an online implementation result | Independent actuation/fallback scheduler, aged-plan admission and physical-time overrun replay. |
| Five completed holds absent from saved approach | High | Archive/compact counts and fault reproduction | High | Incorrect endpoint/source/effort summaries; strict accounting refusal | Record every acknowledged hold immediately, including failed-update tails. |
| Contact and held loads lack continuous set bounds | High | Point accelerations, RK4 and current-state load scaling | High | Robust hybrid safety is unproved | Whole-step contact/load enclosures and integration-defect bounds. |
| Translation/attitude/thrust delivery are not a fully coupled onboard model | High | Translation execution and subsequent attitude replay | High | Delivered body-frame force could differ from the admitted ideal force | Coupled attitude/nozzle allocation and explicitly bounded delivery error. |
| Frozen/adaptive comparison confounds initialization | High for research attribution | Frozen mode suppresses initial orbit inference and online membership | High | Apparent adaptive benefit may be initialization only | Separate initialization policy and use the same accepted initial observations. |
| Drag hypotheses remain indistinguishable | Medium/high | Four candidates survive; last single-hold fleet transitions used intermittently | High for this arm | No demonstrated useful online physical learning | Compute distinguishability before retaining longer multi-epoch evidence or refining partitions. |
| Sensor visibility/post-latch estimation are incomplete | High for realism | Range/dropout model and direct assembly-state feedback | High | Results omit occlusion and uncertain post-capture feedback | FOV/occlusion gating and an assembly observation interface with blackout handling. |

## 6. Decisive experiments

1. **Acknowledgment accounting:** inject failure after five genuinely archived nonlinear holds, at an outer-update boundary, and before archive acknowledgment. Require exact agreement of acknowledged count, endpoint, command sources and held effort. Keep FAILED status and reject unconfirmed motion as a successful prefix.
2. **Continuation:** preserve the failing archived state, compare the original and expanded braking searches with identical supports, and repeat a real nonlinear mission. Record every successor refusal and reserve; a component certificate is insufficient for arrival.
3. **Causal adaptation:** condition frozen and adaptive arms on identical accepted initial packets. Compare frozen subsequent belief, nominal point updates under the original robust set, and full set adaptation. Charge identification computation. Report hypotheses eliminated and the resulting change in feasibility or fuel, not eccentricity width alone.
4. **Bandwidth:** compare dynamic bandwidth with fixed bandwidths selected on separate development cases. A dynamic upper budget of 1,200 versus one fixed choice of 1,000 is not an isolated adaptation comparison.
5. **Online execution:** advance the physical plant during measured planner/proof/archive latency, with an independently scheduled stored fallback. Reject stale proposals and report every missed deadline and finite-renewal expiry.
6. **Robustness:** independently vary spin, physical drag truth, sensor noise/dropouts/visibility and actuator uncertainty on held-out cases. Equal RNG seeds alone do not ensure equal disturbances at physical time when acquisition branches consume the stream differently.
7. **Hybrid mission:** require source-bound approach, full-set capture entry, actual compliant contact/latch, momentum-consistent assembly, detumbling and disposal outcomes. Step refinement and energy balances are numerical diagnostics, not substitutes for whole-set contact safety.

## 7. Salvage and contribution boundary

The strongest prospective contribution is an auditable connection between sensed nonlinear execution, finite backup renewal, tube-aware contact admission and resource-aware fleet planning, with measured completion and computation under matched conditions. This remains a hypothesis to evaluate, not a result established by the bibliography.

Fast RTI, successive-convexification and distributed CLF/CBF methods are appropriate proposal comparators. A new proposal backend must pass the same independent physical admission. Retain GEKKO/IPOPT as the numerical reference and report when solver success is rejected by the guard.

## 8. Recommendation and implemented scope

Prioritize the closed run's concrete failure causes before expanding controller families. The current changes address finite braking candidates, acknowledged-tail recording, command-specific fallback resource debit, and independent initial-orbit-conditioning policy. Qualification evidence and the exact revision must accompany each claimed fix. None of these changes alone supplies an invariant terminal set, an online scheduler, a robust contact theorem or a completed disposal mission.

The initialization option is explicit: `initial_orbit_conditioning="enabled"` allows the same accepted initial GNSS conditioning in adaptive and frozen modes; `"disabled"` suppresses it; default `"legacy"` preserves previous saved configurations. Online membership remains independently controlled by the existing adaptive/frozen mode. Comparisons must record both settings.

### Qualification of the current fixes

These outside-repository qualifications preserve the original failures and
exact source hashes. They exercise components and real first-update execution;
they do not establish a complete mission or flight timing.

| Change | Discriminating result | Preserved boundary |
| --- | --- | --- |
| Immediate acknowledged-hold recording | A real eight-agent failure after five holds changes from archive/compact counts 5/0 to 5/5. Normal 32-hold and interrupted 32-hold cases count each hold once. Refusal to acknowledge hold five records 4/4 and explicitly retains the fifth physical interval as unconfirmed. Held effort agrees with independent archived-input integration. | Every failed arm remains FAILED and uncertified; strict archive and downstream fuel-accountant rejection remain in force. Sampled diagnostics keep their separate complete-outer endpoint. |
| Independent initial orbit policy | Six qualification groups pass; fresh-process checks preserve old legacy behavior. Identical accepted initial packets give enabled adaptive/frozen arms the same eccentricity interval. | Online exclusion is tested with an explicitly amplified synthetic-density fixture. This is not evidence of useful drag identification in the mission regime. |
| Command-specific fallback resource debit | 28 behavioral and five cross-source checks pass, including 32 actual nonlinear eight-agent fallback holds, invalid/stale-bound refusal, unchanged callback-absent decisions and exact ledger conservation. Initial-hold upper debits decrease by 47.7–54.7%. | Actual commands and applied uncertainty supports are unchanged. These are tighter physical upper bounds, not measured fuel savings. Full future fallback reservation remains required. |
| Finite common-stop retiming | The original adjacent-state component replay refuses; the expanded search proves the 89-segment refuge described above. | The unavailable refused performance projection is not reconstructed. Whole-mission progress and renewal still require actual execution. |

The final retiming implementation reproduces the normal-domain witness
byte-for-byte (SHA-256
`f6b658d87fe6902845d0b124242a1859aea17f3a669571f366386aef2525247c`).
A separate exact truncated-clock replay leaves 0.18 rad of reference-domain
coverage: the original route plus hold needs approximately 0.196 rad, while the
9/10 candidate needs approximately 0.178 rad. The shorter candidate receives a
complete proof; the out-of-domain route is refused individually. This closes a
review-found candidate-filtering bug without extending the clock domain.

For a held fallback command, let \(c_{\rm uniform}\) be the installed certificate's
fuel upper debit, and let \(c_{\rm command}\) be a typed bound for the same exact
clock, hold width, applied-input box and hardware domain. The new debit is

\[
 c=\min(c_{\rm uniform},c_{\rm command}),\qquad
 c+R_{\rm suffix}(q+h)\leq F_{\rm remaining}.
\]

The minimum is valid because both bounds enclose the same physical cost. A
missing, refused or mismatched optional command bound leaves the original
uniform debit in force. The complete future reserve \(R_{\rm suffix}\) is not
discounted. Exact resource provenance is retained when the smaller debit is used.

### Earlier analytical comparison retained with its original scope

The 14 October analytical comparison used two 0.15 m buses, exact feedback and
target preview, and no continuous-flow admission. It is not the current sensed
eight-agent Envisat mission. Six saved cases retained a common independent prior
`e=[0.14,0.19]`, `alpha=[1.5e-7,2.5e-7]`, `beta=[3.5e-7,5.5e-7]`, initial point
`(0.17,1.8e-7,4.2e-7)`, midpoint drag truths and a 550-update limit.

| Truth e | Adaptive / frozen updates | Adaptive / frozen final whole-tube position | Adaptive mean tube-radius change | Physical squared-acceleration effort change |
| --- | --- | --- | --- | --- |
| 0.155 | 373 / 373 | 1.758509 / 1.780966 mm | −19.25% | +0.110% |
| 0.185 | 372 / 373 | 1.802466 / 1.779247 mm | −18.57% | +0.0633% |

These deterministic outcomes support reduced conservatism in that analytical
setting. Eccentricity contracted; neither drag interval contracted. Point fits
were confounded: the previously inspected default case's individual drag errors
were approximately −36% and −16%, while their difference was within 0.6%.
Neither chasers nor dense substeps are independent experimental trials.

The original fixed-bandwidth κ=80 cases stopped after 96 and 5 committed updates.
Both failing problems passed a separate corrected-backend replay; see
[solver acceptance](nonlinear-solver.md). A separate repaired source cohort at
`e=0.155` completed adaptive/fixed arms in 373/374 updates, with final whole-tube
position 1.758499/1.807491 mm and minimum dense clearance 15.224428/15.163048 mm.
Mean tube radius over the common prefix was 0.647093/1.150082 mm, a 43.735%
reduction; the physical squared-acceleration effort ratio was 0.999987575.
The one-update completion difference was 13.524 simulated seconds. Eccentricity
width reached 0.87890625% of its prior width; drag widths did not contract.

Effort used a stated outer-midpoint clock approximation and is not fuel.
Historical timing excluded some model-build/plant work and did not establish
real-time eight-agent execution. The original source-cohort hashes remain
`4ff020ca31ae3d9cacbc297eb0197e3ed5aaa5f30cf43c8a6526f788c22081de`
and `82627f7622eb1418e3a6b81edac1f284abffa66861c631601d52ec2f617bb722`.
External evidence retains six results and 3,194 native attempt archives;
failed runs are retained. These historical results are preserved from the
previous review, not rerun during this review.

## 9. Assumptions and unanswered questions

- What finite intervention deadline or invariant terminal refuge remains feasible after a complete planner/proof outage?
- What body-frame force/torque allocation, minimum impulse, mass depletion and attitude uncertainty can be enclosed at the actual feedback rate?
- Which sensor geometry and uncertainty support make drag hypotheses distinguishable during the available mission, without unsafe excitation?
- What finite contact and post-lock load model is both physically defensible and computationally certifiable?
- Does full-set adaptation improve held-out mission completion or fuel after accounting for computation, once initialization is shared?
- Can a complete admitted-command cycle meet its physical deadline while the fallback continues independently?

These questions are release criteria for the corresponding claims. They are not evidence that any unobserved flight failure occurred.
