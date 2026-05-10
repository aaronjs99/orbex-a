# ORBEX-A

ORBEX-A implements the paper system from “Capturing Tumbling Objects in Orbit with Adaptive Tube Model Predictive Control”: elliptical relative dynamics with quadratic drag, nonlinear GEKKO/IPOPT MPC constraints, dynamic tubes, SMID feasible-set adaptation, tumbling-cylinder docking, and single/multi-chaser missions.

The paper workflow is the primary surface. Legacy `mpc`, `tube`, and `adtmpc` mode helpers are retained only as compatibility wrappers for older experiments.

## Install

```bash
pip install -e .
```

GEKKO is the authoritative nonlinear solver path. SciPy/SLSQP is available only for labeled linearized comparison runs.

## Generate Paper-System Artifacts

```bash
orbexa-generate-demo \
  --output results/paper_system \
  --data-output data/paper_system \
  --steps 450 \
  --mission all \
  --primary-solver gekko \
  --secondary-solver scipy \
  --run-linearized \
  --linearized-steps 20 \
  --clean-generated
```

Here `--steps` is the maximum number of receding-horizon MPC updates, not a
scripted trajectory length. Each update solves a fixed-horizon OCP, applies the
first configured control command(s), replans from the propagated plant state,
and stops only when the rendezvous/docking goal is reached. Fresh nonlinear
primary runs raise an error if the update limit is exhausted first.
The optional SciPy/SLSQP linearized comparison is capped separately by
`--linearized-steps` because it is not the authoritative nonlinear mission.

This writes:

- `results/paper_system/single/nonlinear/`
- `results/paper_system/multi/nonlinear/`
- optional matching `linearized/` folders
- raw mission JSON under `data/paper_system/`

Each results folder contains `manifest.json`, `index.html`, `trajectory.html`, `trajectory.mp4`, and diagnostics for actual/nominal trajectories, tube geometry, controls, rendezvous and docking margins, SMID FSS widths, parameter estimates vs truth, target attitude/angular velocity, and multi-chaser spacing.

Regenerate plots from saved data without rerunning solvers:

```bash
orbexa-generate-demo \
  --output results/paper_system \
  --data-output data/paper_system \
  --mission all \
  --run-linearized \
  --from-data
```

## Verify

```bash
python -m compileall -q src tests
pytest -q
```

Generated plots, videos, JSON outputs, caches, and local paper PDFs are ignored by git.

## Repository Layout

Source code and CLIs live under `src/orbexa`. Generated data belongs under
`data/`; rendered artifacts belong under `results/`. The old top-level
`scripts/` wrapper is no longer needed because package console entry points are
installed by `pip install -e .`.

## Citation

```bibtex
@inproceedings{johnsabu2025orbexa,
  title={Capturing Tumbling Objects in Orbit with Adaptive Tube Model Predictive Control},
  author={Aaron John Sabu and Brett T. Lopez},
  booktitle={IEEE Aerospace Conference},
  year={2025}
}
```

## License

GPLv2. See `LICENSE`.
