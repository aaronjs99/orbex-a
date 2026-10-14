# ORBEX-A modular completion implementation plan

> For agentic workers: execute with subagent-driven-development, with independent review and runtime evidence after integration. Repository rules prohibit test infrastructure; use actual examples outside the checkout.

**Goal:** Complete the supported physical ADTMPC software boundaries while respecting the user's relocated source and paper tree.

**Architecture:** Preserve the public `orbexa` API through a package mapping to
`scripts/`. Move responsibilities into focused data, policy, execution and
rendering modules with explicit dependencies; keep compatibility facades small.

**Tech stack:** Python 3.10+, NumPy, SciPy, GEKKO/IPOPT, Matplotlib, Plotly, YAML.

**Spec:** `docs/superpowers/specs/2026-10-14-modularity-design.md`.

## Global constraints

- Work on the existing `codex/orbexa-reliability` feature branch.
- Do not recreate `src/orbexa`; preserve `scripts/` and the renamed paper folder.
- Preserve all user content; stage only reviewed moves and authorized edits.
- Preserve controller equations, assembly order, default policy and result schemas.
- No repository test/validation infrastructure, generated plots or local PDFs in commits.

## Task 1: Relocation and structured configuration

Files: `run.py`, `pyproject.toml`, `scripts/layout.py`, `scripts/core/config.py`,
`config/default.yaml`, `AGENTS.md`, `README.md`, `INSTALL.md`.

- [x] Map logical `orbexa` packages to physical `scripts/`; bootstrap checkout entrypoint.
- [x] Centralize checkout/config/output paths and protect the actual source folder.
- [x] Build/install the editable package and exercise both help entrypoints from another directory.
- [x] Add mission policy, point-estimation and solver settings with unchanged defaults.
- [x] Check invalid numeric/configuration examples outside the checkout.

## Task 2: Mission runtime services

Files: `scripts/simulation/adtmpc_mission.py`, `coupled_mission.py`, new focused
mission model/result/target/scenario/safety/planning/execution/recording modules,
`scripts/simulation/__init__.py`.

- [x] Extract plain models and standalone result I/O; preserve old public imports.
- [x] Compose fleet, reference and safety services through an explicit scenario.
- [x] Replace runtime calls to runner private helpers with public service methods.
- [x] Separate coupled planning, feedback execution and adaptation/recording without changing equations or update order.
- [x] Exercise fresh default, fast-target and fleet missions and independently replay the measured outcomes.

## Task 3: Artifact pipeline

Files: `scripts/visualization/demo.py`, new config/manifest/renderer/plot/animation/
catalog/application modules and `scripts/visualization/__init__.py`.

- [x] Extract focused classes and retain the existing demo facade exports.
- [x] Import saved result records without constructing the mission engine.
- [x] Preserve current-model filtering, marker migration, atomic sessions and cleanup guards.
- [x] Render recorded data with missing optional media, partial families and deliberate incomplete missions.

## Task 4: Completion and public surface

Files: reviewed estimation/planning/core modules, package exports and developer docs.

- [x] Resolve or retire actual broken prototypes after checking consumers.
- [x] Remove nonexistent global-parameter/renderer imports and expose deliberate APIs.
- [x] Build a wheel and check its package contents/imports from outside the checkout.
- [x] Document module responsibilities, extension points and remaining physics boundaries.
- [x] Review the integrated diff independently; fix all material findings.
- [x] Refresh current plots/GIFs, videos, offline bundle and provenance.
- [x] Publish the reviewed source and completion record through PR #8.

## Recorded outcome

Fresh single, eight-chaser and four-times-target-motion missions completed in
92, 94 and 91 updates. All numeric state, nominal-path, control, feasible-set and
point-estimate histories exactly matched their qualified pre-refactor baselines.
Every applicable independent dense replay check passed. Largest whole-tube
terminal position bound: 1.546 mm against 2 mm; minimum fleet spacing: 0.616 m
against 0.35 m. These are sampled free-flight simulation results.

Independent review also repaired escaped injected point estimates, Chaser sample
period mismatches and ending-node torque replay alignment. Saved-data rendering,
optional encoding failures, full media decoding and isolated wheel imports passed.
The current offline gallery contains 73 PNGs, seven GIFs, three MP4s and twelve
interactive views, including standalone attitude and architecture evidence.
