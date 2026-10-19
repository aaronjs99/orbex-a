# Mission readiness

ORBEX-A has not yet demonstrated one complete sensed nonlinear mission from
the 300 m release through attachment, detumbling and disposal. Older movies are
separate historical evidence. Start with the
[readiness page](https://orbex-a-mission-studio.aaronjs99.chatgpt.site/studio/research-review.html)
for plots and a short status view.

## What changed

| Problem | Implementation | Evidence and limit |
|---|---|---|
| Planning blocks feedback | A separate process prepares future proof products; one executor owns the finite backup, observations, clock and fuel. | Numerical fallback continues while a worker is busy. Current physical-period checks still refuse; a general-purpose process split is not a flight deadline guarantee. |
| Each hold gets an unrelated actuator error | A bounded linked window keeps one affine actuator hypothesis per chaser and physical candidate. | Discriminating examples exclude inconsistent hypotheses and retain truth. The actual mission window retains all four drag candidates and is too slow for synchronous online use. |
| Sampling misses intermediate contact/load peaks | Continuous interval contact flows and held-assembly load bounds gate numerical propagation. | Interior-overload examples are refused. These remain conditional on their initial sets, declared rigid/contact model and supported environment. |
| Rotation bounds obstruct otherwise safe approach regions | All-attitude component supports retain rotation orthogonality. | The unchanged Envisat geometry admits a separation example previously refused by independent matrix-entry bounds. Full mission convergence does not follow. |
| Sensor overlap cannot be audited | Acquisition and delivery events stream full measured packets before use. | Matched 22-update prefixes cover the same 6,001 physical-slot/agent/channel keys and execute control after adaptation. Packet identities need not match when trajectories differ. |
| Envisat and eight chasers are embedded in setup | Explicit composite-target, vehicle and fleet declarations; configurable horizons, sensing/prior bounds and three-axis spin. | Two different synthetic targets with fleets of two and three install initial nonlinear finite backups. This does not qualify their complete missions. |

Generated qualification artifacts stay outside this checkout. Their source,
inputs, closed-archive hashes, failure outcomes and scope accompany the results.
Older cohorts are not combined with the v6 linked-estimator study.

The closed v6 central comparison has four 22-update arms, each with 704
acknowledged fleet holds and 21.1496 simulated seconds. All 88 outer updates
miss their physical timing budget; all four physical drag candidates remain.
The independent archive audit authenticates source/native records, measured
packets, certificate links and exact fuel/effort accounting. It does not rerun
the interval differential proofs. This is development evidence, not a held-out
superiority result or a complete mission.

[Mission configuration](configuration.md) explains the three input files and
which declarations the maintained physical model supports.

## Code map

- Execution: `simulation/online_worker.py`, `simulation/online_execution.py`,
  `control/prepared_admission.py`. The default nonlinear mission runner is still
  an explicitly offline study.
- Identification: `estimation/linked_membership.py`; observations enter through
  `simulation/nonlinear_execution.py`. The estimator receives declared commands
  and uncertainty, never the realized matrix, bias or physical drag truth.
- Contact and loads: `simulation/contact_enclosure.py`,
  `simulation/attachment_enclosure.py`, `simulation/rigid_interval.py` and
  `simulation/nonlinear_environment_enclosure.py`.
- Geometry: `control/rotation_support.py`, used by the finite refuge provider.
- Measurement evidence: `simulation/nonlinear_sensing.py`.

## The mathematical distinctions that matter

For a fixed actuator matrix \(M_i\), bias \(b_i\), commanded input \(u_k\)
and bounded evaluation error \(r_k\), a chaser receives
\(u_k^{\mathrm{applied}}=M_i u_k+b_i+r_k\). A linked branch must explain every
retained observation with the same \((M_i,b_i)\). State and target/time endpoint
sets are carried between contiguous holds. A physical candidate is removed
only after every covering actuator branch is proved inconsistent; unresolved
or unevaluated branches remain. This establishes conditional exclusion
soundness, not excitation or useful identification.

For a fixed projection direction \(n\), expanded component vertex \(v\)
and any proper rotation \(R\),

\[
 |n^T Rv|\le\|n\|_2\|v\|_2.
\]

The maximum squared vertex norm is exact and its square root rounds outward.
Convexity extends the bound to the solid component. This avoids treating nine
rotation entries as independent physical degrees of freedom. It reduces
unnecessary conservatism while retaining every declared appendage and its
uncertainty.

Whole-step contact/load bounds concern an exact declared ODE and a supplied
initial set. They do not turn a numerical point from an earlier stage into an
authenticated physical uncertainty set. The contact-to-rigid-assembly handover
and an uninterrupted complete mission still need their own source-bound result.

## What still fails

Useful drag identification, timely useful replanning and full mission progress
remain empirical obligations. The default short observation window is weakly
informative; increasing it also increases proof cost. A stored policy suffix
keeps its original finite expiry. It cannot replace an extended safe policy or
an arrival argument. Physical clock synchronization and bounded dispatch jitter
also need an actuator/sensor adapter contract.

See [nonlinear mission theory](nonlinear-mission.md),
[sensed feedback](sensed-tube.md), [finite fallback](fleet-fallback.md),
[finite contact](finite-contact.md) and the
[primary-source review](research-review.md) for the detailed assumptions.
