# ORBEX-A installation

Use Python 3.10 or newer. `scripts/` is the `orbexa` package; `run.py` and
`serve.py` launch it from the checkout.

```bash
python -m pip install -e ".[certification,studio,video]"
```

- **Base install:** NumPy, SciPy, GEKKO, Matplotlib, Plotly and PyYAML.
- **`certification`:** python-flint/Arb for the held-motion enclosures and zstandard for proof journals. Without them, certification refuses.
- **`studio`:** the local theory renderer.
- **`video`:** a packaged local video encoder.
- **`casadi`:** an optional proposal backend. GEKKO/IPOPT stays the reference.

## Run a mission arm

```bash
python run.py --case central --adaptation adaptive --bandwidth dynamic
```

The launcher supplies `config/envisat.yaml`, `config/spacecraft/envisat.json`
and `config/nonlinear-study.json`. The installed command is `orbexa-run-nonlinear`.

Each run writes its records to a new
`data/nonlinear/<case>/<adaptation>-<bandwidth>/<UTC run id>/` directory. The
`ORBEXA_DATA` and `ORBEXA_RESULTS` environment variables move the record and
result roots, and `--output-dir` chooses one run's directory.

`--steps` caps the number of executed updates. A capped or refused run returns a
nonzero status and keeps its recorded prefix; a cap never means completion.

## Live Studio

```bash
python serve.py
```

Open the printed Studio URL. Jobs go to `data/live-jobs/` and viewers to
`results/studio/`. If the default ports are taken, run
`python serve.py --ui-port 8893 --api-port 8894`.

## Figures

```bash
orbexa-figures mission data/nonlinear/central/adaptive-dynamic/<run id>
```

This writes plots for one recorded run to
`results/missions/central/adaptive-dynamic/<run id>/plots/`. When the run
recorded a completed lifecycle, lifecycle plots and replay data go to
`lifecycle/` beside them. `orbexa-figures comparison NAME ARM...` renders a
matched comparison of study arms into `results/comparisons/NAME/`. It refuses
arms whose sources, declarations or grid do not match. Figures are always
regenerated from recorded data and never committed.
