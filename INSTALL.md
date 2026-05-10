# ORBEX-A Installation

## Basic Install

```bash
python -m pip install -e .
```

The default install includes NumPy, SciPy, GEKKO, Matplotlib, Plotly, and PyYAML.

## Optional Solvers

```bash
python -m pip install -e ".[casadi]"
```

GEKKO/IPOPT remains the primary nonlinear path for paper missions. SciPy/SLSQP is only a secondary linearized comparison path.

## Run the Paper System

```bash
orbexa-generate-demo --output results/paper_system --data-output data/paper_system --steps 450 --mission all --primary-solver gekko --run-linearized --linearized-steps 20
```

`--steps` is a maximum MPC update count, not a scripted trajectory length. The
controller replans over a fixed horizon and applies only the first control
command(s) before solving again. The nonlinear primary run is considered valid
only when `mission_complete` and `success` are both true.
Use `--linearized-steps` to cap optional SciPy/SLSQP comparison artifacts.

Use `--from-data` to rebuild figures and HTML from existing `mission_data.json` files without rerunning MPC or SMID.

## Verify the Environment

```bash
python -m compileall -q src tests
pytest -q
```

If MP4 generation is skipped, install `ffmpeg`; the renderer will still write all other artifacts.
