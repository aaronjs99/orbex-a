# ORBEX-A installation

Use Python 3.10 or newer. `scripts/` is the `orbexa` package; `run.py` and
`serve.py` launch it from the checkout.

```bash
python -m pip install -e ".[certification,studio,video]"
```

- **Base install:** NumPy, SciPy, GEKKO, Matplotlib and PyYAML.
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

## ROS 2

`package.xml` declares `ament_python`, so the package builds with colcon:

```bash
python3 -m pip install gekko "python-flint>=0.8,<0.10" "zstandard==0.25.0"
colcon build --packages-select orbexa --symlink-install
source install/setup.bash
ros2 run orbexa orbexa-run-nonlinear --help
```

rosdep resolves NumPy, SciPy, Matplotlib and PyYAML. It has no key for GEKKO,
python-flint or zstandard, and colcon installs without dependencies, so the
`pip` line above is required before building. Without GEKKO the solver cannot
load. Without python-flint and zstandard every certified step refuses. Add
`mistune` for the theory renderer and `imageio-ffmpeg` for video.
All package metadata is in `setup.py`, because colcon's build directory does
not link `pyproject.toml`.
