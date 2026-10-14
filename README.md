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

ADTMPC missions use the coupled nonlinear GEKKO/IPOPT controller. Ordinary MPC
and other solver adapters remain available through the Python API.

## Generate ADTMPC Mission Artifacts

Default full nonlinear run from the repository root:

```bash
python run.py
```

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

Here `--steps` is the maximum number of receding-horizon MPC updates. The
nonlinear default jointly optimizes nominal motion, dynamic tube states, and
feedback bandwidth with GEKKO/IPOPT. It carries the nominal center and tube
between updates and applies ancillary feedback within each control interval.
The default scenario uses 12 prediction nodes, one applied nominal interval,
and 32 ancillary feedback substeps. Phase horizons come from YAML; API callers
can override `horizon_steps`. The coupled policy replans every outer interval.
Fresh nonlinear primary runs return failure if the goal is not reached.

This writes:

- `results/<session_id>/index.html`: the session gallery, with mission selection, animations, interactive replays, plot filters, and full-resolution images
- `results/<session_id>/single/nonlinear/`
- `results/<session_id>/multi/nonlinear/`
- raw mission JSON under `data/<session_id>/`
- editable run-note templates at `results/<session_id>/README.md` and `data/<session_id>/README.md`
- portable `results/latest.json` and `data/latest.json` references to the newest generated session; directory symlinks are also created when supported

Each mission folder contains its own gallery, `manifest.json`, `replay.html`,
`trajectory.html`, `tube_trajectory.html`, `diagnostics.html`, `trajectory.gif`,
optional `trajectory.mp4`, and the existing control, safety, tube, target, and
parameter diagnostics. Matched tube views show position/sliding-error containment,
terminal accuracy including the entire tube, bandwidth allocation, and ancillary
input-reserve use. Other views show docking errors, the measurable drag
difference and retained marginal ambiguity, emitted-point SMID errors, a body-frame
docking close-up PNG/GIF, and a minimum pairwise-spacing heatmap for fleets.
`insights_manifest.json` records source sample indices, playback rate, fallback
assumptions, and reasons for unavailable views. Replays use recorded states;
playback is sample-based and does not represent physical time. Unsupported views
remain absent for incomplete or older saved data. Multi-agent runs also expose
`results/<session_id>/demo.gif`, using a symlink when available and a copy otherwise.

Open the session's `index.html` in a browser. All interactive views include their
plotting runtime and work offline. Partial saved-data rendering retains the
other rendered mission families through `gallery.json` or their older manifests.

Playable MP4 output requires FFmpeg. Without it, the report links to the GIF and `manifest.json` records why MP4 was skipped; no empty video is presented as a generated artifact.

Generated session artifacts stay out of git. Add a curated preview intentionally when a representative mission artifact is available.

Physical constraint margins concern the actual vehicle. Tube-expanded margins
concern each persistent nominal center and its own matched-node tube. The older
cylinder constraint value combines radial squared and axial linear margins;
its sign indicates exclusion and its magnitude is not a clearance in meters.
Parameter plots distinguish point estimates from the retained uncertainty box.
Local optimization may update a point estimate; only the outward-rounded interval
RK4 membership contractor removes parameter values. Unprocessed boxes are retained
when its budget expires. Exact held ancillary substep inputs are used for adaptation.

Solver results contain `T` state samples, including the initial state, and
`T - 1` control intervals. Control column `k` drives state `k` to state `k + 1`.
Backend adapters normalize their grid conventions before returning results.
The coupled mission reserves every ancillary-input term inside the optimization.
Saturation, reserve exceedance, measured containment failure, and inconsistent
membership stop the mission and preserve the available evidence. It does not
change a nominal command through an unaccounted docking projection. The assigned
side corridor is deliberately conservative and retains nonlinear cylinder
exclusion. Fleet planning constrains contemporaneous nominal paths and both tubes.

Moving-reference preview is enabled by default. State previews have shape
`(n, T)` and feedforward previews `(m, T - 1)`. Near docking, the solver enforces
a terminal set around the future reference. Completion requires nominal position
error plus the whole position-tube radius to fit 2 mm, and nominal velocity error
plus its full envelope to fit 0.08 m/rad, at five consecutive outer endpoints.
The requested goal remains the configured 5 mm standoff. A local Riccati terminal
penalty aids performance; it does not prove nonlinear terminal invariance.

The physical-coordinate model uses meters, derivatives per true anomaly, and
anomaly-acceleration inputs. Physical acceleration increments are `qdot² * u`;
there is no mass/thruster-force actuator model. The default
orbit now has a 300 km perigee at eccentricity 0.18. Its analytical quadratic-drag
approximation and prescribed Euler target motion are explicitly limited models.
See [the derivation and focused literature review](docs/adtmpc-design.md) for frame
conventions, moving-goal/obstacle distinctions, and evidence boundaries.

Nonlinear geometry is constrained at prediction nodes and checked more densely
during execution. Sampled simulation results do not establish continuous-time
safety, recursive feasibility, physical capture, or a fully realistic atmosphere.
Saved-data mission rendering requires the current physical model and coupled
controller metadata. Retired runs remain in external recovery archives.

`run.py` defaults to the ADTMPC mission workflow. Useful options:

- `--mission single|multi|all`
- `--steps N`
- `--output results`
- `--data-output data`
- `--session-id NAME` (fresh runs require an unused, portable folder name)
- `--config PATH`
- `--primary-solver gekko`
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

Use `--session-id NAME` and `--mission single|multi` to select a saved physical
ADTMPC session. Rendering uses its data and does not load `--config` or construct
a mission runner. Every requested family is checked before existing artifacts
or notes are changed. Partial rendering retains other current model families.

Fresh runs never overwrite an existing session. Session names are single folder
names; `latest`, `latest.json`, path traversal, and Windows device names are
reserved. Existing ordinary `latest` folders and editable run notes are retained.
Raw mission data, manifests, and latest references are written atomically.

Incomplete or solver-failed primary missions save their raw data and visual
gallery before returning a nonzero status. Empty and one-state saved results
show their available evidence and omit unsupported motion views. Use
`--no-require-primary-success` deliberately to allow a zero exit status for
bounded diagnostic runs; their galleries still identify unmet mission criteria.

Generated plots, videos, JSON outputs, caches, and local paper PDFs are ignored by git.

## Repository Layout

Source code and CLIs live under `src/orbexa`. Generated data belongs under
`data/`; rendered artifacts belong under `results/`. Package console entry
points are installed by `pip install -e .`.

The active checkout contains the maintained physical-coordinate implementation,
scenario settings, and current design notes. Imported historical manuscripts,
diagrams and prototypes are preserved in verified external recovery archives
and Git history, outside the active project tree.

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
