# ORBEX-A

ORBEX-A implements adaptive dynamic tube MPC (ADTMPC) rendezvous and docking: elliptical relative dynamics with quadratic drag, nonlinear GEKKO/IPOPT target constraints, dynamic tubes, SMID feasible-set adaptation, tumbling-cylinder docking, and single/multi-chaser missions.

The ADTMPC mission workflow is the sole public execution surface.

`python run.py --help` works directly from a source checkout, including after
moving the folder, without installing the package or its numerical dependencies.
Mission execution still requires the runtime dependencies below. The checkout
launcher and installed console command use the same options.

## Install

```bash
pip install -e .
```

GEKKO is the authoritative nonlinear solver path. SciPy/SLSQP is available only for labeled linearized comparison runs.

## Generate ADTMPC Mission Artifacts

Default full nonlinear run from the repository root:

```bash
python run.py
```

Add `--run-linearized` only when you want the optional SciPy/SLSQP comparison artifacts.
The default YAML scenario uses 8 chasers for the multi-agent mission. Select a
different scenario with `--config path/to/scenario.yaml`; its `num_chasers` sets
the fleet size, and its orbit parameters set the simulated plant truth while
the controller starts from a separate uncertain belief. The checkout default
is resolved relative to the package location, so launching from another working
directory works. Wheel installations must supply `--config` if the checkout
scenario is unavailable.

Equivalent console entry point:

```bash
orbexa-generate-demo \
  --output results \
  --data-output data \
  --steps 450 \
  --mission all \
  --primary-solver gekko
```

Here `--steps` is the maximum number of receding-horizon MPC updates, not a
scripted trajectory length. Each update solves a fixed-horizon OCP, applies the
first configured control command(s), replans from the propagated plant state,
and stops only when the rendezvous/docking goal is reached. Fresh nonlinear
primary runs raise an error if the update limit is exhausted first.
If enabled, the optional SciPy/SLSQP linearized comparison is capped separately by
`--linearized-steps` because it is not the authoritative nonlinear mission.

This writes:

- `results/<session_id>/single/nonlinear/`
- `results/<session_id>/multi/nonlinear/`
- optional matching `linearized/` folders when `--run-linearized` is set
- raw mission JSON under `data/<session_id>/`
- editable run-note templates at `results/<session_id>/README.md` and `data/<session_id>/README.md`
- portable `results/latest.json` and `data/latest.json` references to the newest generated session; directory symlinks are also created when supported

Each results folder contains `manifest.json`, `index.html`, `trajectory.html`, `tube_trajectory.html`, `diagnostics.html`, `trajectory.gif`, optional `trajectory.mp4`, and diagnostics for actual/nominal trajectories, tube geometry, controls, physical and tube-tightened margins, SMID FSS widths, parameter estimates vs truth, target roll/pitch/yaw, and multi-chaser spacing. Multi-agent runs also expose `results/<session_id>/demo.gif`, using a symlink when available and a copy otherwise.

Playable MP4 output requires FFmpeg. Without it, the report links to the GIF and `manifest.json` records why MP4 was skipped; no empty video is presented as a generated artifact.

Generated session artifacts stay out of git. Add a curated preview intentionally when a representative mission artifact is available.

The physical margin plots answer collision safety. The tube-tightened margin plots answer whether the nominal robust tube stayed inside the tightened constraint; they can go below zero even when the physical active safety margin is positive. Parameter-estimate diagnostics show the point belief, truth, and the active feasible-set band. Weak or collinear data keep unidentifiable marginal intervals broad. Point estimates can still learn measurable parameter combinations, while preserving the prior along unsupported directions. Each emitted point estimate is checked against an RK4 replay of the measured state/control window; its prediction and fit errors are recorded separately from the candidate set's verification witness.

Solver results contain `T` state samples, including the initial state, and
`T - 1` control intervals. Control column `k` drives state `k` to state `k + 1`.
Backend adapters normalize their grid conventions before returning results.
The mission applies the optimized interval command and tube ancillary correction,
followed by the existing input limits and docking target-safety filter.
The default mission adds a local Riccati terminal penalty to account for regulation
beyond its short prediction horizon. This uses the current belief and the configured
state/input weights; set `mpc.terminal_cost_enabled: false` to disable it. The
penalty adds to the existing endpoint state cost, giving an endpoint weight of
`Q + P`, where `P` is the local Riccati matrix. It does not replace the stage cost.
The local linear approximation is a performance aid, not a proof of nonlinear feasibility
or physical safety. Mission completion and physical-margin checks still determine
success. The docking safety filter measures velocity relative to the rotating
target surface.

`run.py` defaults to the ADTMPC mission workflow. Useful options:

- `--mission single|multi|all`
- `--steps N`
- `--output results`
- `--data-output data`
- `--session-id NAME` (fresh runs require an unused, portable folder name)
- `--config PATH`
- `--primary-solver gekko|scipy|casadi`
- `--run-linearized`
- `--linearized-steps N`
- `--clean-generated` (explicitly removes only the selected generated roots, including their saved session notes; source and scenario inputs are protected)
- `--from-data`

Regenerate plots from saved data without rerunning solvers:

```bash
orbexa-generate-demo \
  --output results \
  --data-output data \
  --mission all \
  --from-data
```

Add `--run-linearized` to saved-data rendering only when that session also contains the optional linearized comparison data. Use `--session-id NAME` and `--mission single|multi` to select an older or partial saved session. Rendering uses the saved data and does not load `--config` or construct a mission runner. Every requested family is loaded before existing artifacts or notes are changed.

Fresh runs never overwrite an existing session. Session names are single folder
names; `latest`, `latest.json`, path traversal, and Windows device names are
reserved. Existing ordinary `latest` folders and editable run notes are retained.
Raw mission data, manifests, and latest references are written atomically.

Incomplete or solver-failed primary missions return a nonzero status and retain
their raw `mission_data.json` for inspection. Select that session with
`--from-data` to render the failure. Use `--no-require-primary-success` deliberately
for bounded diagnostic runs; their reports still show `Success: False` when the
mission did not complete.

Generated plots, videos, JSON outputs, caches, and local paper PDFs are ignored by git.

## Repository Layout

Source code and CLIs live under `src/orbexa`. Generated data belongs under
`data/`; rendered artifacts belong under `results/`. Package console entry
points are installed by `pip install -e .`.

Useful manuscript source, editable diagrams, and historical experiment references
from the older `mas_space_sim` repository are preserved under
[`paper/legacy`](paper/legacy/README.md). That directory documents provenance,
local manuscript edits, the retained thesis stash, and full recovery information.

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
