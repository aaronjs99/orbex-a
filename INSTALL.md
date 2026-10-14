# ORBEX-A Installation

## Basic Install

Use Python 3.10 or newer. The physical `scripts/` directory supplies the
`orbexa` package through the installation mapping; keep the user's directory layout.

```bash
python -m pip install -e .
```

The default install includes NumPy, SciPy, GEKKO, Matplotlib, Plotly, and PyYAML.
The source checkout's `python run.py --help` needs only Python and remains usable
after the checkout is moved. Re-run the editable installation from the new
location to update an existing installed console command.

## Optional Solvers

```bash
python -m pip install -e ".[casadi]"
```

ADTMPC missions use the coupled nonlinear GEKKO/IPOPT path. The ordinary MPC
Python API also supports the SciPy and optional CasADi adapters.

## Physical mission and local presentation

```bash
python -m pip install -e ".[certification,studio,video]"
orbexa-run-mission --config config/default.yaml --prior config/physical-prior.json --admission config/physical-admission.json --output-dir ../orbexa-output/physical-eight --steps 550
```

`certification` supplies python-flint/Arb for continuous held-motion enclosures;
an unavailable backend refuses certification. `studio` supplies the local theory
renderer; browser graphics and equation fonts are packaged offline. `video`
supplies a local encoder. Representative sensing and vehicle assumptions are
documented in [config/README.md](config/README.md).

The physical runner reserves a new output directory, uses finite buses and a
gravity-gradient target, saves the approach and continuous certificate journal,
then records chaser attitude control, attachment, detumbling and orbital delivery.
It preserves failed and completed prefixes. The update cap does not imply
completion. Commands operate on a numerical representative spacecraft model.

Use `--resume PATH_TO_UNFINISHED_APPROACH --steps ADDITIONAL_UPDATES` with the
same configuration/prior/admission and a new output directory to numerically
continue a saved run. The target epoch/domain and physical safety assumptions
must remain unchanged. See [checkpoint compatibility](docs/mission-checkpoint.md)
for legacy source context and the distinction from safe backup control.

To finish an already successful physical COM approach:

```bash
orbexa-complete-mission PATH/mission_data.json --output-dir NEW_DIRECTORY --input-component-bound 100
```

The bound is a declaration about all actual held approach inputs, not merely
saved means. It supports conservative translation fuel accounting; actual
bounded chaser attitude replay is the default. Physical capture refuses missing
attitude, unsafe bus geometry or insufficient gripper reach. See
[the lifecycle](docs/mission-lifecycle.md) for independently injectable stage APIs.

## Run ADTMPC Missions

```bash
python run.py
```

Choose a scenario explicitly with `python run.py --config path/to/scenario.yaml`.
The default scenario follows the checkout location, rather than the terminal's
working directory. Output roots default to the checkout's `results/` and `data/`;
use `--output` and `--data-output` to place generated files elsewhere.
The same options, including `--fps`, logging, and diagnostic-run control, apply
to `orbexa-generate-demo`.

`--steps` is a maximum MPC update count, not a scripted trajectory length. The
controller replans over a fixed horizon and applies only the first control
command(s) before solving again. The nonlinear primary run is considered valid
only when `mission_complete` and `success` are both true.
ADTMPC mission generation uses the coupled nonlinear GEKKO/IPOPT controller.

Use `orbexa-generate-demo --from-data` (or `python run.py --from-data`) to rebuild figures and HTML from existing `mission_data.json` files without rerunning MPC or SMID. This replaces the removed `orbexa-plot-results` command.

Open `results/<session_id>/index.html` for the offline mission gallery, animated
recorded replay, docking close-ups, and filtered evidence plots. Partial renders
retain other rendered families. Failed primary runs also create this gallery
before returning a nonzero status. Saved data and rendered output may be on
different Windows drives; the gallery then identifies the raw-data location.

Moving-goal prediction is enabled by default through
`mpc.reference_preview_enabled: true`. Docking plans include the moving goal,
nonlinear cylinder exclusion and the assigned conservative side corridor.
Execution stops on a measured safety or tube-containment violation.

For local MP4 generation, install `ffmpeg` on the system path or use
`python -m pip install -e ".[video]"` for a packaged local encoder.
If encoding is unavailable or fails, the report shows the reason
and provides a GIF instead of an empty MP4. Windows and cloud-mounted folders
do not need symlink privileges: `latest.json` selects the latest saved session,
and the session preview falls back to a copied GIF.

Use a new `--session-id` for every fresh run. Existing sessions are protected.
Saved rendering preflights the requested data and retains raw JSON and notes.
`--clean-generated` removes the selected generated roots only; it rejects source,
scenario inputs, overlapping roots, and invalid directories before deletion.
