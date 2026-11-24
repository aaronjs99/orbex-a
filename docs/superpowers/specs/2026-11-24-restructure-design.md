# ORBEX-A restructure design

**Decisions (owner, 24 November 2026)**

- `scripts/` stays the `orbexa` package, and `run.py` and `serve.py` stay outside it. This matches the owner's ROS 2 package layout, and the repository should become ROS 2 compatible.
- Retire the analytical pipeline and everything that only it uses. That covers `ADTMPCMissionRunner`, `orbexa-run-mission`, `orbexa-complete-mission`, `orbexa-generate-demo`, sessions, the historical Studio resume route and their docs. Nothing outdated stays in the tree; git history keeps it.
- Figures are results. They are generated into `results/` from recorded data, in folders named by topic. The 40 tracked PNGs under `paper/` are retired; new figures replace them once data and generators exist.
- Do a full layered redesign.

## Target layers

Imports only go downward. Each layer is one top-level package of `orbexa`.

| Layer | Package | Responsibility | Comes from |
|---|---|---|---|
| 0 | `common` | Exact intervals and boxes, paths (`ProjectLayout`), JSON/IO, hashing, discretization | `utils`, `layout.py`, `contracts.py`, `IntervalBox` from `control.certification.reachability` |
| 1 | `models` | Clocks and frames, translational/attitude dynamics, geometry, vehicles, targets, configuration | `core` |
| 2 | `proofs` | Picard flow enclosures, orbital hold proofs, attitude flow, rigid interval numerics | `control.certification`, `simulation.numerics` |
| 3 | `estimation` | Sensors, bounded navigation, membership, adaptation | `estimation` (maintained parts) |
| 4 | `control` | Tubes, MPC formulation, finite fallback, terminal and contact control, actuation | `control` minus `certification` |
| 4 | `solvers` | GEKKO/IPOPT reference backend and optional backends; depends only on formulations | `solvers` |
| 5 | `mission` | Stages and their orchestration: `approach`, `contact`, `attached`, `lifecycle`, `coordination` | `simulation.nonlinear`, `simulation.contact`, `simulation.attached`, `simulation.completion`, the maintained parts of `simulation.approach`, `planning` |
| 6 | `evidence` | Journals, archives, manifests, source freezing and fuel accounting | `simulation.recording`, `simulation.study.source`, records and archive modules |
| 7 | `studies` | Study arms, metrics, comparisons | `simulation.study` |
| 8 | `figures` | Plots, renderers and Studio export from saved data into `results/<topic>/` | `visualization` (maintained parts) |
| 9 | `service` | Loopback live jobs and HTTP | `simulation.service` |
| 10 | `cli` | Thin command lines only | `cli` |

Package `__init__` files hold no eager re-exports. Callers import a class from the module that defines it, and there are no compatibility facades.

## Retirement set

Retired, because only the analytical pipeline reaches them or nothing reaches them at all:

- **Analytical pipeline:** the analytical `simulation.approach` runner, checkpoint and scenario modules, `cli.run_mission`, `cli.complete_mission`, `cli.generate_demo`, `sessions`, `visualization.workflow`, the analytical plots/renderers/publishing, and `service.jobs.historical`/`resume`.
- **Superseded estimation and planning:** SMID batch and fitting (`estimation.adaptation.batch/smid/records`, `estimation.fitting`, `estimation.enclosures`, `solvers.gekko.estimation`), old objectives (`planning.objectives`) and learned proposals (`planning.proposals.ppo/training/features`).
- **Unused control and helpers:** `control.certification.barriers`, `control.mpc.linearization/projection`, `control.fallback.continuation/replacement`, `core.dynamics.inertial`, `utils.geometry.regions/sampling` and `utils.runtime`.
- **Unmaintained modules:** a module is unmaintained if, after the eager re-exports are removed, it is still unreachable from the maintained entry points and is not on the keep list below.

Unwired but kept, because a plan workstream wires them in:

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
