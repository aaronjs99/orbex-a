# Swarm repository consolidation

Useful control and learning concepts from the private
`GRANDE-Robots/swarm-clf-cbf-playground` now have explicit ORBEX-A homes.
The reference source is master `07073759b14e2afef96f744362b5a78577020d5f`;
the unmerged RL branch is `1324b890cda49d664b797413cdb1bb95a6a85fad`.
The [mathematical guide](control-foundations.md) derives the methods and their
assumptions. [MIT attribution](../THIRD_PARTY_NOTICES.md) is retained.

## Integration and online behavior

The native nonlinear GEKKO formulation and physical motion admission retain
control authority. New trajectory providers enter through
`MPCController(proposal_provider=...)` or `ProposalSeededSolver`. Current caller
warm starts win by default. Invalid or late completed hints are discarded;
the decorator records their preparation time and native optimization still
runs. The optional time budget is an acceptance check, not preemption.

PD/feedforward, potential preferences, condensed affine MPC, joint consensus,
and deterministic PPO can supply initial guesses. Local CLF/HOCBF projection,
braking envelopes, spectral connectivity and GP statistics are reusable
planning/diagnostic APIs. They do not become physical certificates by being
imported. GP fitting and policy training belong outside actuation. No learned
policy is claimed trained or qualified for this nonlinear mission. The default
mission does not silently enable these optional tools or add new Studio modes.

The optional `learning` extra installs PyTorch. Core imports do not require
PyTorch, CVXOPT, OSQP or scikit-learn. The existing `all` extra retains its
previous scope; learning is explicitly requested separately.

## Qualification and limits

Outside-repository qualification passed 13 independent component groups:
moving/joint high-order rows and initial conditions; hard projection and
failure; GP prior/posterior projection; consensus; graph finite differences;
RK4 seeds; PPO updates and complete continuation; finite training/logging;
actual CLF derivatives and braking; condensed/joint MPC; features and reward;
native solver boundary/budget; and stale-policy/parent-checkpoint handling.
The graph derivative finite-difference error was below 4e-11 in its checked
simple-eigenvalue example. These checks are not an all-state theorem.

Four small native GEKKO dynamic-tube cases checked zero actuator reserve,
nonzero reserve with/without a proposal, and an infeasible reserve. Feasible
cases passed native residual readback and total input tightening; an error
allowance of 0.2 reduced peak nominal acceleration from 0.999716 to 0.799716
under a ±1 command limit. A 1.1 allowance produced no accepted solve. This
also qualifies the dedicated actuator-input support correction, independently
from the earlier pure-algebra checks. It does not establish the cause of the
old mission's repeated fallback.

Warm measurements on this host, 50 samples per operation, were approximately:

| Operation | Median | 95th percentile | Largest observed |
| --- | ---: | ---: | ---: |
| Four-node RK4 tracking hint | 0.46 ms | 0.74 ms | 0.78 ms |
| GP prediction, 64 samples | 0.037 ms | 0.048 ms | 0.070 ms |
| Three-input hard-row QP | 0.28 ms | 0.52 ms | 1.35 ms |
| CPU PPO inference, 64 hidden units | 0.035 ms | 0.048 ms | 0.222 ms |

These measurements exclude startup and are not hard deadlines. The small
native dynamic-tube solves took roughly 0.9 s;
the one supplied hint did not demonstrate a material speed improvement.
The full eight-agent nonlinear admission/continuation workload is much larger.
A full source/input-bound mission, matched comparison grid and measured
end-to-end deadline behavior are still required. No complete 300 m mission
or new published mission recording is asserted by this consolidation.

## Preservation and retirement

All four advertised branch heads, five tags and remote-tracking references
were preserved in a Git bundle. An independent bare restoration matched all
13 refs and all 331 reachable objects; `git fsck` passed. Master and the RL
branch snapshots, exact path hashes, the complete branch diff, repository
metadata and restoration instructions accompany it. The captured repository
had no issues/PRs or releases, and wiki/discussions were disabled.

The recovery package is `data/swarm-playground-recovery-20261018.zip` in the
canonical project's existing data folder. It is archival data, never an active
module, selectable simulation mode or runtime dependency. A copy is retained
in the outside review workspace. Removing the original GitHub repository is
separate from proving ORBEX's full physical mission. GitHub account settings, actions secrets, external project boards and unadvertised/unreachable Git
objects are not claimed preserved. The original remote has not been deleted.

## Every master source file

“Incorporated” means its useful concept is implemented with the corrections
above. “Covered” identifies an existing maintained subsystem. “Archived”
identifies exact recoverable material deliberately excluded from the live
orbital runtime. A composite disposition records both useful and obsolete
parts. SHA prefixes below identify the original bytes; the recovery manifest
retains full SHA256 values.

| Original path | SHA256 prefix | Disposition / destination | Justification |
| --- | --- | --- | --- |
| `.gitignore` | `d0f9876cf7b9` | Covered: canonical .gitignore | Existing generated-data exclusions remain authoritative. |
| `LICENSE` | `716effeaa82e` | Incorporated: THIRD_PARTY_NOTICES.md | Complete MIT copyright and permission text retained. |
| `README.md` | `19025f7f35ad` | Incorporated/archived: docs/control-foundations.md | Useful theory explained; unsupported safety/SAC claims replaced by actual implementation scope. |
| `config/controller/cbf.yaml` | `a9f6f2792298` | Incorporated/archived: barrier.py; connectivity.py; trajectory_hints.py; consensus.py | Rates, ranges, gain and iteration settings are explicit typed interfaces; soft safety and zero-on-failure defaults rejected. |
| `config/controller/connectivity.yaml` | `e275b3cb214d` | Incorporated/archived: barrier.py; connectivity.py; trajectory_hints.py; consensus.py | Rates, ranges, gain and iteration settings are explicit typed interfaces; soft safety and zero-on-failure defaults rejected. |
| `config/controller/mpc.yaml` | `708ef8cf241c` | Incorporated/archived: MPCProblem; ConvexTrajectoryHint | Horizon, weights and input limits come from the current problem; hardcoded toy obstacle sizes and stride/blend/disable-near-goal behavior are archived. |
| `config/controller/nominal.yaml` | `ac0a44e45ef5` | Incorporated/archived: barrier.py; connectivity.py; trajectory_hints.py; consensus.py | Rates, ranges, gain and iteration settings are explicit typed interfaces; soft safety and zero-on-failure defaults rejected. |
| `config/controller/solver.yaml` | `1e0c4f669c45` | Incorporated/archived: barrier.py; connectivity.py; trajectory_hints.py; consensus.py | Rates, ranges, gain and iteration settings are explicit typed interfaces; soft safety and zero-on-failure defaults rejected. |
| `config/experiment/3d_default.yaml` | `5527d78abb04` | Archived: all-refs recovery bundle | Bounded Cartesian wall/lane scenarios are a different benchmark, not orbital mission inputs; recoverable exactly without live legacy modes. |
| `config/experiment/default.yaml` | `9e2daf958dfd` | Archived: all-refs recovery bundle | Bounded Cartesian wall/lane scenarios are a different benchmark, not orbital mission inputs; recoverable exactly without live legacy modes. |
| `config/sim/default.yaml` | `74b69eb1e388` | Archived: all-refs recovery bundle | Bounded Cartesian wall/lane scenarios are a different benchmark, not orbital mission inputs; recoverable exactly without live legacy modes. |
| `config/viz/live.yaml` | `5416331ea5c5` | Covered: Mission Studio and mission recording | The maintained live controls, plots and video already use one source-bound mission clock. |
| `config/viz/record.yaml` | `772c20a12528` | Covered: Mission Studio and mission recording | The maintained live controls, plots and video already use one source-bound mission clock. |
| `config/world/2d_lanes.yaml` | `0d057acc8b63` | Archived: all-refs recovery bundle | Bounded Cartesian wall/lane scenarios are a different benchmark, not orbital mission inputs; recoverable exactly without live legacy modes. |
| `config/world/3d_box.yaml` | `eba3b9055abb` | Archived: all-refs recovery bundle | Bounded Cartesian wall/lane scenarios are a different benchmark, not orbital mission inputs; recoverable exactly without live legacy modes. |
| `run.py` | `b392f47bbb4f` | Covered: run.py; pyproject.toml; scripts/cli/ | Existing relocatable package and mission launchers avoid a second runnable toy mode. |
| `scripts/controllers/__init__.py` | `3dfc3c04f6cf` | Covered: orbexa package interfaces | Module scaffolding is superseded by the existing maintained package; exact original remains in the bundle. |
| `scripts/controllers/cbf/__init__.py` | `44884b46a187` | Covered: orbexa package interfaces | Module scaffolding is superseded by the existing maintained package; exact original remains in the bundle. |
| `scripts/controllers/cbf/ecbf.py` | `1f13fac9b2d2` | Incorporated: control/barrier.py | Second-order distance and translating-halfspace rows, moving/controlled neighbors, robust acceleration support and initial-set checks; braking diagnostics include latency. |
| `scripts/controllers/cbf/gp_learner.py` | `670b71b4f90f` | Incorporated: estimation/residual_gp.py | Bounded-memory independent-output posterior, nonzero prior uncertainty and dimensionally projected statistical radii; never shrinks authenticated tube bounds. |
| `scripts/controllers/constraints/__init__.py` | `f1b99b1a94df` | Covered: orbexa package interfaces | Module scaffolding is superseded by the existing maintained package; exact original remains in the bundle. |
| `scripts/controllers/constraints/builders.py` | `20339e5d54cc` | Incorporated: control/barrier.py | Second-order distance and translating-halfspace rows, moving/controlled neighbors, robust acceleration support and initial-set checks; braking diagnostics include latency. |
| `scripts/controllers/constraints/manager.py` | `bf505fc98e3f` | Covered: MPCProblem; hard QP row interface; physical admission | Keep explicit numerical constraints separate from continuous physical witnesses; no parallel generic soft/hard registry. |
| `scripts/controllers/dynamics.py` | `ed20fc2088f4` | Covered: MPCController; nonlinear_mission.py; proposal_seed.py | Maintain the native ADTMPC and independent admission orchestration; do not add gravity after the constrained command. |
| `scripts/controllers/mpc_layer.py` | `76461bce21ab` | Incorporated: planning/trajectory_hints.py | Condensed affine MPC retains configurable costs and hard rows with dynamics-consistent RK4, explicit numerical failure and native nonlinear follow-through. |
| `scripts/controllers/network.py` | `77be9831a9f2` | Incorporated: planning/connectivity.py | Smooth weighted Laplacian, spectral derivative, link mask and repeated-eigenvalue handling replace an incorrectly labeled binary-graph attraction gradient. |
| `scripts/controllers/nominal/__init__.py` | `e3b0c44298fc` | Covered: orbexa package interfaces | Module scaffolding is superseded by the existing maintained package; exact original remains in the bundle. |
| `scripts/controllers/nominal/clf_progress.py` | `df560d80155d` | Incorporated: QuadraticTrackingCLF; TrackingHint | Moving-reference PD/feedforward plus an actual CARE-based quadratic CLF row; the old CLF-named strategy only returned PD. |
| `scripts/controllers/nominal/pd_tracking.py` | `987b8e1a18dd` | Incorporated: QuadraticTrackingCLF; TrackingHint | Moving-reference PD/feedforward plus an actual CARE-based quadratic CLF row; the old CLF-named strategy only returned PD. |
| `scripts/controllers/nominal/potential_fields.py` | `daec84bd89bc` | Incorporated: planning/trajectory_hints.py:PotentialRepulsion | Bounded reciprocal-gap preference with coincident-center rejection; local-minimum and heuristic scope explicit. |
| `scripts/controllers/policies/connectivity.py` | `d69402fdb973` | Incorporated: planning/connectivity.py | Smooth weighted Laplacian, spectral derivative, link mask and repeated-eigenvalue handling replace an incorrectly labeled binary-graph attraction gradient. |
| `scripts/controllers/policies/nominal.py` | `18338bbdad40` | Incorporated: QuadraticTrackingCLF; TrackingHint | Moving-reference PD/feedforward plus an actual CARE-based quadratic CLF row; the old CLF-named strategy only returned PD. |
| `scripts/controllers/safety_filter.py` | `0d804d6cdd4d` | Incorporated: control/command_projection.py; solvers/proposal_seed.py | Hard input/barrier rows, progress-only slack and explicit failure; numerical proposals have no direct actuation authority. |
| `scripts/controllers/solvers/__init__.py` | `e3b0c44298fc` | Covered: orbexa package interfaces | Module scaffolding is superseded by the existing maintained package; exact original remains in the bundle. |
| `scripts/controllers/solvers/admm_engine.py` | `cde41a677775` | Incorporated: planning/consensus.py; ConsensusInitialGuess | Frozen synchronous shared-variable snapshots, explicit joint agent blocks and primal/dual convergence replace averaging private actions. |
| `scripts/controllers/solvers/base_qp.py` | `70030d2f0f32` | Incorporated: control/command_projection.py; solvers/proposal_seed.py | Hard input/barrier rows, progress-only slack and explicit failure; numerical proposals have no direct actuation authority. |
| `scripts/controllers/solvers/interface.py` | `a06d7c39f5a9` | Incorporated: control/command_projection.py; solvers/proposal_seed.py | Hard input/barrier rows, progress-only slack and explicit failure; numerical proposals have no direct actuation authority. |
| `scripts/controllers/solvers/wrapper.py` | `cb93673027b7` | Incorporated: control/command_projection.py; solvers/proposal_seed.py | Hard input/barrier rows, progress-only slack and explicit failure; numerical proposals have no direct actuation authority. |
| `scripts/controllers/state_manager.py` | `948b52efbd1c` | Incorporated: planning/consensus.py; ConsensusInitialGuess | Frozen synchronous shared-variable snapshots, explicit joint agent blocks and primal/dual convergence replace averaging private actions. |
| `scripts/rl/README.md` | `b4a6c57bcd0e` | Incorporated/archived: docs/control-foundations.md sections 11 and 15 | Actual PPO, normalization, reward and environment contracts documented; original instructions recoverable. |
| `scripts/rl/__init__.py` | `e3b0c44298fc` | Covered: orbexa package interfaces | Module scaffolding is superseded by the existing maintained package; exact original remains in the bundle. |
| `scripts/rl/agent.py` | `093006ea8e3c` | Incorporated: planning/ppo_proposal.py | Optional PPO with matching tanh likelihood, GAE, terminal/truncation semantics and complete checkpoints; not a trained mission policy. |
| `scripts/rl/environment.py` | `197eb4a0f7c5` | Incorporated/covered: policy_features.py; training.py; maintained simulator | Nearest-threat masked observations and reward terms ported; orbital state, resets, physical contact and success gates remain environment-owned. |
| `scripts/rl/evaluate.py` | `c56cd3b28772` | Incorporated/covered: PPOProposalPolicy.predict/load_checkpoint; PPOInitialGuess | Deterministic, fingerprinted proposal inference; no toy evaluation trajectory is promoted to flight success. |
| `scripts/rl/rl_utils.py` | `1ec8b98e0d9e` | Incorporated: planning/training.py | Finite seeded rollout contracts, append-only step/episode logs, plots, checkpoint continuation and learning metrics. |
| `scripts/rl/train.py` | `39497dc13a7c` | Incorporated: planning/training.py | Finite seeded rollout contracts, append-only step/episode logs, plots, checkpoint continuation and learning metrics. |
| `scripts/utils/__init__.py` | `b5540d6d3452` | Covered: orbexa package interfaces | Module scaffolding is superseded by the existing maintained package; exact original remains in the bundle. |
| `scripts/utils/animator.py` | `09b7b5eca172` | Covered: visualization/mission_studio.py; mission_plots.py; nonlinear_plots.py | Existing interactive 3D replay, recorded clocks and plotting are maintained instead of duplicate Cartesian GUI modes. |
| `scripts/utils/config.py` | `aef16cbbeb02` | Covered/archived: core/config_loader.py; config_types.py | Explicit injectable mission configuration replaces recursive toy YAML/dotted overrides; original layering syntax retained in archive. |
| `scripts/utils/geometry.py` | `a3222215dcb1` | Covered: core/spacecraft_geometry.py; core/target_geometry.py | Composite spacecraft and appendage geometry covers the mission; spherical feature summaries stay nonauthoritative. |
| `scripts/utils/logger.py` | `a73c872bcd75` | Covered/corrected: simulation/nonlinear_recording.py; nonlinear_archive.py | Source-bound motion and safety witnesses replace toy pair-distance logs that used one radius instead of the pair sum. |
| `scripts/utils/sim_engine.py` | `6775da6482c3` | Covered/archived: simulation/nonlinear_mission.py; nonlinear_execution.py | Maintain actual causal orbital integration/admission; RK4 proposals share the optimizer model rather than mismatched toy integration. |
| `scripts/utils/stepper.py` | `b5f36f89b55a` | Covered/archived: simulation/nonlinear_mission.py; nonlinear_execution.py | Maintain actual causal orbital integration/admission; RK4 proposals share the optimizer model rather than mismatched toy integration. |
| `scripts/utils/visualization.py` | `b5808980c520` | Covered: visualization/mission_studio.py; mission_plots.py; nonlinear_plots.py | Existing interactive 3D replay, recorded clocks and plotting are maintained instead of duplicate Cartesian GUI modes. |
| `scripts/world/__init__.py` | `e3b0c44298fc` | Covered: orbexa package interfaces | Module scaffolding is superseded by the existing maintained package; exact original remains in the bundle. |
| `scripts/world/environment.py` | `d1fa11632e3d` | Covered/archived: core/nonlinear_relative.py; nonlinear_target.py; simulation/contact.py | Orbital truth, finite contact and assembly dynamics replace wall clamps, speed clipping, post-step collision repair and toy gravity. |
| `scripts/world/physics.py` | `089c186b971a` | Covered/archived: core/nonlinear_relative.py; nonlinear_target.py; simulation/contact.py | Orbital truth, finite contact and assembly dynamics replace wall clamps, speed clipping, post-step collision repair and toy gravity. |
| `setup.py` | `4d4ef568f155` | Covered: run.py; pyproject.toml; scripts/cli/ | Existing relocatable package and mission launchers avoid a second runnable toy mode. |

## The unmerged RL branch

All unchanged package moves (`R100`) retain the dispositions above. The branch
also changed root README/entry points/build metadata, shortened the toy MPC
horizon from 10 to 5, and raised its soft safety penalty from 1e5 to 2.5e5.
Package portability is already supplied by ORBEX; finite budgets remain
configurable, and raising a soft collision penalty is not accepted as safety.
Those exact historical files are recoverable from the RL snapshot and bundle.

The six changed RL implementation/documentation files contribute linear
learning-rate decay, entropy/exploration controls, continuation, expanded
metrics, checkpoint selection and distance-progress shaping. They map to
`ppo_proposal.py`, `training.py` and `policy_features.py`. ORBEX corrects
clipped-action likelihoods, truncated-episode bootstrapping, discounted
potential shaping and incomplete optimizer/RNG checkpoints. It preserves
earlier logs by forking a run rather than destructively truncating them.
The source's `src/` move is preserved as history rather than moving ORBEX's
user-chosen layout. No surviving branch work depends on retaining the remote.
