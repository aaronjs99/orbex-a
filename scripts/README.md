# scripts (the `orbexa` package)

orbexa package root. Adaptive dynamic tube MPC for multi-chaser capture and disposal of a tumbling spacecraft.

Imports only go downward through these layers: `common`, `models`, `proofs`, `control`, `estimation`, `solvers`, `evidence`, `mission`, `studies`, `figures`, `service`, `cli`. The design is in [the restructure spec](../docs/superpowers/specs/2026-11-24-restructure-design.md).

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `cli/` | Command-line entrypoints for ORBEX-A. | `common`, `figures/data`, `figures/plots`, `figures/publishing`, `figures/renderers`, `service`, +1 more | none |
| `common/` | Shared paths, exact intervals, serialization, IO and geometry helpers. | `numpy`, `pyyaml` | `cli`, `control`, `control/fallback`, `control/mpc`, `control/terminal`, `control/tubes`, +36 more |
| `control/` | Tubes, MPC formulation, finite fallback, terminal control and proof services. | `common`, `models/dynamics`, `numpy`, `proofs`, `python-flint`, `scipy` | `estimation/membership`, `estimation/navigation`, `evidence/records`, `mission`, `mission/approach`, `mission/approach/arrival`, +11 more |
| `estimation/` | Sensing, bounded navigation, membership and parameter adaptation. | `common`, `control`, `control/fallback`, `models/dynamics`, `numpy`, `proofs`, +2 more | `mission`, `mission/approach`, `mission/execution`, `mission/online`, `mission/records`, `mission/safety`, +1 more |
| `evidence/` | Durable journals, native solver evidence, records and accounting. | `common`, `common/geometry`, `common/io`, `control/fallback`, `gekko`, `models/dynamics`, +5 more | `figures/plots`, `figures/publishing`, `figures/renderers`, `mission/approach/terminal`, `mission/attached/disposal`, `mission/execution`, +6 more |
| `figures/` | Figures, renderers and Studio export from recorded mission data. | `common`, `common/io`, `evidence/projections`, `evidence/records`, `imageio-ffmpeg`, `matplotlib`, +4 more | `cli`, `service/jobs` |
| `mission/` | Mission stages and their orchestration: approach, contact, attached flight and lifecycle. | `common`, `common/geometry`, `common/io`, `control`, `control/fallback`, `control/mpc`, +19 more | `service/jobs`, `studies` |
| `models/` | Physical models: clocks and frames, dynamics, geometry, targets and configuration. | `common`, `common/geometry`, `common/io`, `matplotlib`, `numpy`, `python-flint`, +1 more | `control`, `control/fallback`, `control/terminal`, `control/tubes`, `estimation/adaptation`, `estimation/membership`, +30 more |
| `proofs/` | Validated enclosures: Picard flows, orbital and attitude hold proofs, geometry bounds. | `common`, `models/dynamics`, `models/geometry`, `numpy`, `python-flint`, `scipy` | `control`, `control/fallback`, `control/terminal`, `estimation/membership`, `estimation/navigation`, `mission`, +12 more |
| `service/` | Loopback live simulation service and HTTP adapter. | `common`, `evidence`, `evidence/projections`, `evidence/records`, `figures/renderers`, `mission`, +5 more | `cli` |
| `solvers/` | ORBEX-A Solvers Module. | `casadi`, `common`, `common/geometry`, `common/io`, `control/mpc`, `control/terminal`, +6 more | `evidence`, `mission/approach` |
| `studies/` | Study arms: requests, execution, telemetry and metrics. | `common`, `estimation/navigation`, `evidence`, `evidence/records`, `mission`, `mission/approach/physical`, +5 more | `cli` |
| `__init__.py` | orbexa package root. Adaptive dynamic tube MPC for multi-chaser capture and disposal of a tumbling spacecraft. | none | none |
