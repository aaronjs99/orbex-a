# ORBEX-A

## Overview

Adaptive dynamic tube MPC for a configurable fleet deployed in a common cluster
from a tumbling spacecraft. The default study releases eight chasers at 300 m.
The mission sequence is approach, finite attachment, assembly detumbling and
orbital lowering to a descending 120 km interface. The default target uses
Envisat bus, solar-wing and appendage geometry. Other solid-component targets,
patch layouts and vehicle designs enter through
[explicit mission configuration](docs/configuration.md).

**Start with [mission readiness](docs/mission-readiness.md).** It lists the
current results, unresolved problems and the modules responsible for each one.
The complete sensed nonlinear mission and real-time operation remain unverified.

[Nonlinear mission theory](docs/nonlinear-mission.md) connects the equations and
assumptions. The commanded reference orbit frame has a fixed clock; it is not
the perturbed target LVLH. Carter–Humi dynamics remain the optimizer's nominal
surrogate. Independently bounded nonlinear residuals and physical flows bridge
that surrogate to the declared truth model. The COM uncertainty tube stays
outside the hull, and a finite carried mechanism closes the remaining gap after
the whole terminal tube meets the position and physical-speed gates.

## Responsibilities

| Component | Implementation | Package |
| --- | --- | --- |
| Physical translation | Cartesian central gravity, inclined J2 and co-rotating vector quadratic drag | `models` |
| Target rotation | Quaternion/Euler rigid-body dynamics with gravity-gradient torque | `models` |
| Validated enclosures | Arb Picard flows, held-input proofs, rotation and attitude bounds | `proofs` |
| ADTMPC | GEKKO/IPOPT nominal motion, dynamic tube and feedback bandwidth; sensing-aware held-input reserve | `control`, `solvers` |
| Navigation | Bounded range/bearing, pose/rate and own-GNSS products with bias, acquisition time, latency and dropout | `estimation` |
| Adaptation | Initial osculating eccentricity inference and nonlinear set-membership parameter exclusion | `estimation` |
| Motion admission | Independent interval flows, every spacecraft component and peer, finite backup and fuel reservation | `control`, `mission` |
| Attachment | Finite positioners, compliant pad contact, friction, finite bonds and settling/load gates | `mission` |
| Post-contact flight | Actual assembled mass/inertia, external thrusters, propellant and environmental dynamics | `mission` |
| Records | Durable journals, proof archives and result records | `evidence` |
| Studies | Matched adaptive/frozen by dynamic/fixed arms | `studies` |
| Figures and Studio | Plots, replay and browser Studio from recorded data | `figures`, `service` |

Source lives in `scripts/`, installed as the `orbexa` package. Imports only go
downward through `common`, `models`, `proofs`, `control`, `estimation`,
`solvers`, `evidence`, `mission`, `studies`, `figures`, `service` and `cli`.
See the [architecture map](docs/architecture.md).

## Requirements

- Python 3.10 or later.
- NumPy, SciPy, GEKKO (with IPOPT), Matplotlib and PyYAML.
- `certification` extra: python-flint (Arb) and zstandard. Without them every certified step refuses.
- `studio` extra: the theory renderer. `video` extra: a local video encoder. `casadi` extra: an optional proposal backend.
- ROS 2 is optional. `package.xml` makes the package an `ament_python` package.

## Installation

```bash
python -m pip install -e ".[certification,studio,video]"
```

[INSTALL.md](INSTALL.md) covers the extras, the colcon build and the packages
rosdep cannot provide.

## Usage

```bash
python run.py --case central --adaptation adaptive --bandwidth dynamic
```

The launcher supplies `config/envisat.yaml`, `config/spacecraft/envisat.json`
and `config/nonlinear-study.json`; override them explicitly when needed. The
installed command is `orbexa-run-nonlinear`, and `python -S run.py --help` needs
no scientific packages. Each invocation runs one declared case and arm. Run all
four adaptive/frozen belief by dynamic/fixed bandwidth arms for each case to
form a matched study.

Each run writes its records to a new
`data/nonlinear/<case>/<adaptation>-<bandwidth>/<UTC run id>/` directory. Rendered
output goes to `results/`. Both folders are gitignored. `ORBEXA_DATA` and
`ORBEXA_RESULTS` move the roots, and `--output-dir` chooses one run's directory.
`--steps` caps the number of executed updates. A capped or refused run returns a
nonzero status and keeps its recorded prefix; a cap never means completion.

Before motion, each process freezes its Python source, explicit inputs and
resolved study declaration. It streams complete admission witnesses before each
applied hold, then records completion or assumption violation. A run directory
holds `source-manifest.json`, `study-manifest.json`, `approach/`, `completion/`,
`nonlinear-certificates.jsonl.zst` and `study-metrics.json`.

```bash
orbexa-figures mission data/nonlinear/central/adaptive-dynamic/<run id>
python serve.py
```

`orbexa-figures` renders a run's plots, and a matched comparison of arms with its
`comparison` subcommand, into `results/`. `python serve.py` starts the local
Studio at `http://127.0.0.1:8881/studio/`, where release distance, spacing and
target-spin knobs start fresh native solver runs through a loopback API. The
browser is a viewer; it does not run GEKKO.

## Validation

This repository keeps no test or validation infrastructure. Evidence is the
recorded output of real runs, each bound to its source and input hashes:

- every run's manifests, certificate journal and metrics under `data/`;
- the matched study comparison, which refuses unmatched sources, declarations or
  grids, and reports failed and capped arms;
- [mission readiness](docs/mission-readiness.md), which separates what is
  verified, what is only a component check, and what is open.

No complete sensed mission from the 300 m release through disposal has been
recorded, and no run establishes a real-time deadline or useful drag learning.
Hardware qualification, flexible appendages, atmospheric demise and regulatory
approval are outside this simulation. The
[research review](docs/research-review.md) compares the primary literature and
states the remaining obligations.

## File Structure

| File | Relevance | Dependencies | Used by |
| --- | --- | --- | --- |
| `scripts/` | The `orbexa` package: layered models, proofs, control, mission stages, evidence, studies, figures, service and command lines. | `numpy`, `scipy`, `gekko`, `matplotlib`, `pyyaml`, `python-flint`, `zstandard` | `run.py`, `serve.py`, installed commands |
| `config/` | Scenario inputs: orbit and controller YAML, vehicle geometry and the study declaration. | none | `run.py`, `serve.py` |
| `docs/` | Guides for the models, proofs and stages, and the design record. | none | readers |
| `run.py` | Checkout launcher for one study arm. | `scripts/cli/run_nonlinear.py`, `config/` | users |
| `serve.py` | Checkout launcher for the local Studio service. | `scripts/cli/serve_studio.py`, `config/` | users |
| `setup.py` | Package metadata, the `scripts/` to `orbexa` mapping and the ament data files. | `setuptools`, `package.xml`, `resource/` | `pip`, `colcon` |
| `setup.cfg` | Console-script install directory for ROS 2. | none | `colcon` |
| `pyproject.toml` | Build system and `ruff` settings. | `setuptools` | `pip`, `ruff` |
| `package.xml`, `resource/` | ROS 2 `ament_python` package manifest and resource marker. | none | `colcon` |
| `INSTALL.md` | Installation, extras, ROS 2 build and figure commands. | none | readers |
| `AGENTS.md` | Standing rules for code, documentation and workflow. | none | maintainers |
| `LICENSE`, `THIRD_PARTY_NOTICES.md` | Project license and the notice for adapted third-party ideas. | none | readers |

## License

GPL-2.0-only. See `LICENSE`.

### Citation

```bibtex
@inproceedings{johnsabu2025orbexa,
  title={Capturing Tumbling Objects in Orbit with Adaptive Tube Model Predictive Control},
  author={Aaron John Sabu and Brett T. Lopez},
  booktitle={IEEE Aerospace Conference},
  year={2025}
}
```
