# ORBEX-A completion and release plan

> For agentic workers: execute with subagent-driven-development and review each
> workstream independently. Repository rules prohibit test and validation
> infrastructure. Run qualifications as real examples. Their inputs go in
> `./data/` and their evidence in `./results/`; both are gitignored (see W1b).

## Decisions recorded (24 November 2026)

- **Timing hardware:** no fixed reference machine. Each run records its host
  fingerprint: CPU model, core count, OS, Python, IPOPT and GEKKO versions.
  L7 is measured and reported per host.
- **Paper:** the conference paper is already published: *Capturing Tumbling
  Objects in Orbit with Adaptive Tube Model Predictive Control*, IEEE Aerospace
  Conference 2025, doi:10.1109/AERO63441.2025.11068695, linked from
  aaronjs.me/research/orbex-a/. The completion results become a follow-up
  journal paper. W9 keeps the conference figures reproducible through the
  analytical reproduction APIs and adds the new results.
- **Data and results:** stored in the checkout under `./data/` and
  `./results/`, with one modular layout (W1b). This replaces
  `../orbexa-output/`.

**Goal:** Close every open claim in [mission readiness](../../mission-readiness.md).
Then publish the paper results package and a tagged final release.

**Done means:** Every item in the obligation ledger ends in one of two states.
**Demonstrated** means a source-bound closed record satisfies the gate.
**Bounded** means a rigorous negative or limit result exists: a proof, a
certified bound, or a measured characterisation that the paper states as a
finding. "Under investigation", "remains unverified" and "still needs" may not
appear in the released docs.

**Architecture constraints (from `AGENTS.md`):** Keep the GEKKO/IPOPT nonlinear
path as the reference. Any added proposal backend must pass the same independent
admission. Keep the formulation and solver boundaries. Keep the `orbexa` → `scripts/`
mapping and the user's paper layout. Commit no generated evidence and no PDFs.

## Termination rule

This plan exists so the project actually ends. These four rules make that happen.

1. **Ledger freeze.** Phase 0 records the complete obligation ledger. A new
   obligation found later is either fixed within the workstream that found it,
   or recorded there as a declared modelling assumption. The ledger does not grow.
2. **Attempt budget.** Each gate gets a fixed number of design iterations,
   listed per workstream. When the budget is spent, the workstream writes up its
   strongest certified bound. The item closes as Bounded with the failure cause
   recorded.
3. **No retroactive upgrades.** Every result names its source hash. A change to
   the source means rerunning the affected cases.
4. **One release candidate.** Studies run once, on a frozen release-candidate
   source (Phase 5). A defect found after the freeze reopens only the cases it
   affects.

## Obligation ledger

| ID | Obligation | Current state | Closing gate | Workstream |
|---|---|---|---|---|
| L1 | Finite refuge renewal across the approach | Closed run stops on a fuel-reserve renewal refusal and then expiry | Central case: all 8 chasers reach the whole-set terminal standoff set with no expiry | W2 |
| L2 | Production approach → contact handoff | Component checks pass. The handoff is under investigation | Contact starts from the authenticated journal endpoint of L1's run, not a fixture | W3 |
| L3 | Continuous contact/load bounds and rigid-lock impulse | Whole-step local gates. Unresolved uncertainty transfers | All 8 attachments complete under whole-step contact/load enclosures with certified lock | W3 |
| L4 | Post-capture sensing | Direct assembly-state feedback | Assembly observation interface with blackout handling drives detumbling | W4 |
| L5 | Coupled attitude/thrust delivery | Translation first, then attitude replay | Delivered body-frame force error is enclosed and charged in admission | W4 |
| L6 | Detumble, disposal burn, 120 km crossing in one record | Burn, coast and entry are only qualified separately | One closed record from release to the descending whole-set entry bracket | W4 |
| L7 | Online deadline (30 ms command cycle) | Every update misses | Every hold either meets the deadline on the declared hardware, or the plant advances during latency, a stale proposal is refused and the independent fallback keeps the hold safe. Miss rate reported | W5 |
| L8 | Sensor visibility | Range/dropout only | FOV and target-occlusion gating in `SensorSuite`, with the filter consuming the gaps | W6 |
| L9 | Drag identification | All four candidates survive. No contraction | Certified contraction on mission, or a certified non-identifiability bound for the declared sensors and duration (Bounded) | W7 |
| L10 | Adaptation attribution | Development cases only | Held-out adaptive/frozen × dynamic/tuned-fixed factorial with shared initial conditioning | W8 |
| L11 | Generalisation | Two synthetic spacecraft install initial backups only | Both synthetic spacecraft complete missions, or are Bounded with the cause recorded | W8 |
| L12 | Robustness | No held-out variation | Held-out sweep over spin, drag truth, noise/dropout/visibility and actuator error | W8 |

The flexible appendages, hardware qualification, demise and regulatory
approval listed in `README.md` stay declared out of scope. They are not ledger items.

## Dependency order

```
W0 ─► W1b ─► W1 ─► W2 ─► W3 ─► W4 ─┐
                   W5 ─────────────┤
                   W6 ─► W7 ───────┼─► W8 (frozen RC) ─► W9 paper ─► W10 release
```

W2→W3→W4 is the critical path. W5, W6 and W7 run in parallel with it. All three
must merge before the release-candidate freeze.

## W0 Ledger and scope freeze

- [ ] Make `docs/mission-readiness.md` the single ledger above, with a status column and an evidence-link column.
- [ ] Remove "current changes" narration from `README.md` and `research-review.md`. Point both to the ledger.
- [x] Record the user's decisions in this plan (see Decisions recorded).

## W1b Data and results layout

Files: `scripts/layout.py`, `scripts/sessions.py`, `scripts/simulation/study/{api,request,records}.py`, `scripts/cli/*.py`, `serve.py`, `scripts/simulation/service/`, `scripts/visualization/data/`, `.gitignore`, `README.md`, `INSTALL.md`, `docs/configuration.md`, `docs/deployment.md`.

- [ ] Add a dependency-free `ProjectLayout` to `scripts/layout.py`, with `data_root` and `results_root`. The defaults are `<checkout>/data` and `<checkout>/results`; an installed wheel falls back to the working directory. Overrides come from `ORBEXA_DATA` / `ORBEXA_RESULTS`, then the CLI flags.
- [ ] `./data/` holds inputs only: external datasets, downloaded geometry, frozen priors and reusable proof caches. `./data/paper/` stays tracked, as now.
- [ ] `./results/` holds generated evidence only, at `results/<study>/<case>/<arm>/<run-id>/`. `<run-id>` is the UTC timestamp plus a short source hash. Keep the existing manifest, journal and approach/completion contents unchanged.
- [ ] Keep `results/index.jsonl` as an append-only catalog: run id, path, source and input hashes, host fingerprint, status and stage reached. Write it atomically when a run starts and when it finishes.
- [ ] Make `--output-dir` optional on every CLI and default it to the layout. Keep the rule that a fresh directory is required, which unique run ids now guarantee.
- [ ] Studio, `MissionPlots`, the comparison audit and W8 aggregation discover runs through the index rather than hand-supplied paths.
- [ ] Add `orbexa-results` with `list`, `show` and `clean` subcommands. `clean` removes only aborted runs, and only with explicit `--yes`.
- [ ] Replace every `../orbexa-output` reference in code and docs. Change README's "Generated evidence stays outside the checkout" to describe the gitignored folders.

## W1 Environment and baseline reproduction

Files: `install_dependencies.sh`, `INSTALL.md`.

- [ ] Install `.[certification,studio,video]` into a clean environment. Record the exact versions in the source manifest.
- [ ] Rerun the closed v6 central case, adaptive/dynamic arm, on the current head. Confirm that it reproduces the archived refusal sequence.
- [ ] Write one failure-cause table per arm: the first refused renewal, the binding reserve term and the remaining fuel. W2 works from this table.

## W2 Approach completion (L1). Budget: 4 iterations

Files: `control/fallback/{refuge,plan,fuel,runtime}.py`, `simulation/nonlinear/safety/{candidates,planner,resources}.py`, `simulation/nonlinear/runner.py`.

- [ ] Rank the reserve overbounds from W1 by contribution. Tighten the largest valid ones: command-specific suffix reserve, attitude-allocation slack and terminal-hold reserve.
- [ ] Expand the renewal candidates: earlier renewal before the expiry margin, mixed staging/terminal successors and shorter-hold successors. Keep the first-refusal-continues selection.
- [ ] Add a renewal-horizon monitor. When the remaining finite window falls below the proved renewal search time, preempt performance in favour of renewal.
- [ ] Gate: the central-case adaptive/dynamic run reaches the terminal set for all 8 chasers. Record the zero-expiry, fuel and update count.

## W3 Contact, lock and handoff (L2, L3). Budget: 4 iterations

Files: `simulation/completion/nonlinear/{boundary,coordinator}.py`, `simulation/contact/{simulator,capture,enclosure}.py`, `simulation/contact/lock/transition.py`, `control/terminal/contact.py`.

- [ ] Wire `orbexa-run-nonlinear` so that it continues straight into completion from the authenticated approach journal. Drop the manual `--terminal-execution` path for the maintained mission.
- [ ] Carry the remaining tube, fuel and clock uncertainty through the contact chart. No separate re-sampling.
- [ ] Close the rigid-lock impulse certificate: bound the momentum jump and the pad load over the lock transition interval.
- [ ] Sequence the eight attachments, with each completed lock updating the assembly mass and inertia for the next.
- [ ] Gate: eight certified attachments start from L1's run endpoint.

## W4 Attached flight to entry (L4–L6). Budget: 3 iterations

Files: `simulation/attached/{assembly/model,propulsion,detumbling}.py`, `simulation/attached/disposal/{simulator,axis,preview}.py`, `simulation/completion/{lifecycle,nonlinear/observers}.py`, `simulation/approach/physical/attitude.py`.

- [ ] Assembly observation interface: chaser-carried sensors produce bounded assembly pose/rate, with blackout propagation reusing `execution/blackout.py`.
- [ ] Coupled attitude/nozzle allocation for approach and attached flight. Charge an enclosed delivered-force error as actuator uncertainty in admission.
- [ ] Run detumbling → disposal burn → coast → the descending whole-set 120 km bracket in one `MissionLifecycle` record.
- [ ] Gate: one closed record from 300 m release to entry for the central case.

## W5 Online execution (L7). Budget: 4 iterations

Files: `simulation/nonlinear/execution/{coordinator,plant}.py`, `simulation/recording/journal.py`, `solvers/gekko/{tube,solver}.py`, optional `solvers/casadi/`.

- [ ] Physical-time mode: the plant advances during the measured planner, proof and archive latency. A proposal is admitted only when it is aged to the current state, and a stale one is refused.
- [ ] Profile the complete command cycle. Attack the largest costs first: per-agent proof parallelism, warm starts, cached geometry and incremental admission.
- [ ] Optional real-time proposal comparator (CasADi/RTI). It passes the same admission and GEKKO stays the reference.
- [ ] Gate: report the deadline miss rate on the declared hardware, with fallback safety held on every missed hold. A full meet closes L7 as Demonstrated. Otherwise L7 closes as Bounded: measured latency plus proved fallback continuity.

## W6 Sensor visibility (L8). Budget: 2 iterations

Files: `estimation/navigation/{sensors,records,filter}.py`, `core/geometry/composite.py`.

- [ ] Add FOV cones and component occlusion against the expanded target solids. Occluded samples become declared dropouts.
- [ ] Gate: the central case runs end to end with visibility on. Record the outage durations.

## W7 Identification (L9). Budget: 3 iterations

Files: `estimation/membership/nonlinear.py`, `estimation/adaptation/{smid,nonlinear,belief}.py`, the integral membership estimator.

- [ ] Compute certified distinguishability between drag candidates for the declared sensors, clock and mission duration (W2–W4 arcs). This covers the integral constraints over the long attached-flight and coast arcs, where drag dominates.
- [ ] If the bound shows separability, implement multi-epoch linked evidence. Optionally add safe excitation inside the admitted tube.
- [ ] Gate: either a certified support contraction on the mission record (Demonstrated), or the certified non-identifiability bound (Bounded). Do not narrow the declared noise or density.

## W8 Release-candidate studies (L10–L12)

- [ ] Freeze the release-candidate source and tag `v3.0.0-rc1`.
- [ ] Choose a tuned fixed bandwidth on development cases only. Declare the held-out cases before running them.
- [ ] Run the full matched factorial: adaptive/frozen × dynamic/tuned-fixed, for the central case and each held-out case. Include every failure.
- [ ] Run complete missions for both synthetic spacecraft.
- [ ] Run the robustness sweep: spin, drag truth, noise, dropout, visibility and actuator error.
- [ ] Audit: matching source and input hashes, the complete grid, and the archived journals.

## W9 Paper results package

- [ ] Map each claim in the manuscript to its ledger evidence (claim → record hash → figure/table).
- [ ] Generate every figure and table from saved evidence through `MissionPlots`, `NonlinearMissionPlots` and `LifecycleRenderer`. Keep the scripts in the package. Keep the outputs out of the commit.
- [ ] Place the assets in `paper/IEEE Aerospace Conference 2025/assets/` per the user's layout, or a new venue folder.
- [ ] Write the Bounded results (L7 and L9 if applicable) as stated findings with their bounds.

## W10 Release

- [ ] Cut the docs: merge historical narration into one `docs/history.md` and keep theory docs that match the final code. Fix the stale references found by a full-link check.
- [ ] Remove dead routes and prototypes no longer reached by the maintained mission. Keep the analytical reproduction APIs.
- [ ] Build the wheel and check its imports from outside the checkout. Bump to `3.0.0`. Add `CITATION.cff`.
- [ ] Publish the final Studio recording of the closed mission and export the selected evidence from `./results/` as a release bundle (optionally deposit it on Zenodo).
- [ ] Tag `v3.0.0`. After that, the repository accepts maintenance fixes only.

## Risks

- **L7 may be physically out of reach** for eight-agent interval proofs within 30 ms on commodity hardware. The Bounded exit stops this from blocking the release.
- **L9 is likely non-identifiable** at approach scale. Including the long post-capture arcs is the realistic route to contraction.
- **Compute.** Full factorial and robustness runs need hours per arm. They need a dedicated machine, not an ephemeral cloud session.
- **W2 failure cascades** to W3 and W4. If W2 exhausts its budget, the closed mission becomes Bounded. W3 and W4 still close their components from the furthest certified prefix.
