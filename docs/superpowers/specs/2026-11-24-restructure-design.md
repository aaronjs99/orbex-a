# ORBEX-A restructure design

**Decisions (owner, 24 November 2026)**

- `scripts/` stays the `orbexa` package, and `run.py` and `serve.py` stay outside it. This matches the owner's ROS 2 package layout, and the repository should become ROS 2 compatible.
- Retire the analytical pipeline and everything that only it uses. That covers `ADTMPCMissionRunner`, `orbexa-run-mission`, `orbexa-complete-mission`, `orbexa-generate-demo`, sessions, the historical Studio resume route and their docs. Nothing outdated stays in the tree; git history keeps it.
- Figures are results. They are generated into `results/` from recorded data, in folders named by topic. The 40 tracked PNGs under `paper/` are retired; new figures replace them once data and generators exist.
- Do a full layered redesign.

## Target layers

Imports only go downward. Each layer is one top-level package of `orbexa`.

| Layer | Package | Responsibility |
|---|---|---|
| 0 | `common` | Exact interval boxes and Arb arithmetic (`intervals`), exact-record encoding, serialization, paths (`ProjectLayout`), IO, discretization |
| 1 | `models` | Clocks and frames, translational/attitude dynamics, geometry and constraints, targets and their snapshots, configuration |
| 2 | `proofs` | Picard flow enclosures, orbital hold proofs, attitude flow, rotation bounds, rigid interval numerics |
| 3 | `control` | Tubes, MPC formulation, finite fallback, terminal and contact control, actuation |
| 4 | `estimation` | Sensors, bounded navigation, membership, adaptation |
| 5 | `solvers` | GEKKO/IPOPT reference backend, optional backends, the `MPCController` dispatcher and proposal seeds |
| 6 | `evidence` | Journals, proof archives, result records, stage envelopes, native solver evidence, source freezing, data projections |
| 7 | `mission` | Approach, execution, safety, contact, attached flight, lifecycle, records collected during a run, coordination, online execution |
| 8 | `studies` | Study arms, telemetry and metrics |
| 9 | `figures` | Plots, renderers and Studio export into `results/<topic>/` |
| 10 | `service` | Loopback live jobs and HTTP |
| 11 | `cli` | Thin command lines |

Imports only go downward. The one exception is a typing-only import: `proofs.rotation_bounds` names `control.terminal.physical.OrbitalClockBounds` under `TYPE_CHECKING`.

Package `__init__` files hold no eager re-exports. Callers import a class from the module that defines it, and there are no compatibility facades.

## Retirement set

The retirement set is computed, not listed by hand. It is every module outside the import closure of two starting sets:

- the maintained entry points:
  - `cli.run_nonlinear` and `cli.serve_studio`;
  - the study API and execution;
  - the nonlinear live job and worker process;
  - the registered solver backends;
- the keep list below.

The closure counts every `import`, every string module name passed to `import_module`, and every lazy export map.

First the analytical entry points go: `cli.run_mission`, `cli.complete_mission`, `cli.generate_demo`, `sessions`, `visualization.workflow`, the historical Studio resume route and the analytical-only inputs `--prior`, `--admission` and `--deployment`. The closure is then recomputed until it stops changing.

Two kinds of entanglement are cut by hand in later steps:

1. **Shared helpers inside analytical modules.** For example, `plain` lives in `approach.records.checkpoint`. Each helper moves to its owning layer before the analytical module is deleted.
2. **Analytical branches inside maintained classes.** For example, `MPCController` imports SMID `run_adaptation`. These branches are removed, and the superseded SMID batch, fitting and enclosure modules then fall out of the closure.

Unwired but kept, because a plan workstream wires them in. Their whole dependency closure is kept too, for example `control.fallback.replacement`, which `simulation.online` imports.

| Module | Workstream it feeds |
|---|---|
| `simulation.online` | W5, independent online execution |
| `estimation.membership.integral` | W7, identification |
| `planning.coordination.allocation` | Multi-chaser assignment in `mission.coordination` |
| `visualization.plots.nonlinear`, `visualization.data.comparison` | `figures` |

## Repository shape

- **Root:** `run.py`, `serve.py`, `pyproject.toml`, `README.md`, `INSTALL.md`, `AGENTS.md`, `LICENSE`, `THIRD_PARTY_NOTICES.md`, `config/`, `docs/` and `scripts/`.
  - `setup.py` stays, because ROS 2 `ament_python` builds call it.
  - `package.xml` and `resource/orbexa` are added for ROS 2.
  - `requirements.txt` and `install_dependencies.sh` collapse into `pyproject.toml` extras.
- **Generated output:** gitignored `data/` (records) and `results/` (figures, videos, Studio bundles).
- **`paper/`:** removed.
- **`docs/`:** one guide per layer plus theory. Historical narration moves to `docs/history.md`.

## Execution and verification

Each step is one pull request:

1. Remove eager package re-exports, so callers import from defining modules.
2. Retire the analytical pipeline, the unmaintained modules and `paper/`.
3. Move modules into the layers in bottom-up order, one PR per layer, using `git mv` plus a scripted import rewrite.
4. Build the figures pipeline into `results/`.
5. Root and ROS 2 files.
6. Docs consolidation and folder READMEs.

**Equivalence check for every step**

- The checks:
  - every module imports;
  - `compileall` passes;
  - the CLI `--help` screens run;
  - a capped real run (`run.py --case central --adaptation adaptive --bandwidth dynamic --steps 3`) matches the recorded control after masking hashes, timings and paths, for both the certificate journal and the run records.
- Moves change module paths inside recorded identities only through source hashes, and the masking already removes those.
- A step that changes any masked value is a behavior change and does not merge.

Copyright: Aaron John Sabu
