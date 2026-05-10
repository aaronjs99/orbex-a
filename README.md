# ORBEX-A

![ORBEX-A multi-agent ADTMPC demo](assets/demo.gif)

ORBEX-A implements adaptive dynamic tube MPC (ADTMPC) rendezvous and docking: elliptical relative dynamics with quadratic drag, nonlinear GEKKO/IPOPT target constraints, dynamic tubes, SMID feasible-set adaptation, tumbling-cylinder docking, and single/multi-chaser missions.

The ADTMPC mission workflow is the primary surface. Legacy `mpc`, `tube`, and `adtmpc` mode helpers are retained only as compatibility wrappers for older experiments.

## Install

```bash
pip install -e .
```

GEKKO is the authoritative nonlinear solver path. SciPy/SLSQP is available only for labeled linearized comparison runs.

## Generate ADTMPC Mission Artifacts

Simplest full run from the repository root:

```bash
python run.py --mission all --run-linearized --clean-generated
```

Equivalent console entry point:

```bash
orbexa-generate-demo \
  --output results \
  --data-output data \
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

- `results/<session_id>/single/nonlinear/`
- `results/<session_id>/multi/nonlinear/`
- optional matching `linearized/` folders
- raw mission JSON under `data/<session_id>/`
- editable run-note templates at `results/<session_id>/README.md` and `data/<session_id>/README.md`
- symlinks `results/latest` and `data/latest` pointing at the newest generated session

Each results folder contains `manifest.json`, `index.html`, `trajectory.html`, `tube_trajectory.html`, `diagnostics.html`, `trajectory.mp4`, `trajectory.gif`, and diagnostics for actual/nominal trajectories, tube geometry, controls, physical and tube-tightened margins, SMID FSS widths, parameter estimates vs truth, target roll/pitch/yaw, and multi-chaser spacing. Multi-agent runs also expose `results/<session_id>/demo.gif` as a symlink to `multi/nonlinear/trajectory.gif`.

Generated session artifacts stay out of git. The tracked repository preview is `assets/demo.gif`; refresh it intentionally from `results/latest/demo.gif` after a run when you want to update the README animation.

The physical margin plots answer collision safety. The tube-tightened margin plots answer whether the nominal robust tube stayed inside the tightened constraint; they can go below zero even when the physical active safety margin is positive.

`run.py` defaults to the ADTMPC mission workflow. Useful options:

- `--mission single|multi|all`
- `--steps N`
- `--output results`
- `--data-output data`
- `--session-id NAME`
- `--primary-solver gekko|scipy|casadi`
- `--run-linearized`
- `--linearized-steps N`
- `--clean-generated`
- `--from-data`
- `--workflow legacy --mode mpc|tube|adtmpc|all` for older simulation modes

Regenerate plots from saved data without rerunning solvers:

```bash
orbexa-generate-demo \
  --output results \
  --data-output data \
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
