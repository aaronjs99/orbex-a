# `control/tubes`

control.tubes package.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `model.py` | Solver-independent boundary-layer feedback and dynamic tube arithmetic. | `numpy`, `sample_hold` | `mission/approach`, `mission/execution`, `mission/records`, `solvers/gekko` |
| `sample_hold.py` | First-exit comparison bounds for the implemented held ancillary controller. | `numpy` | `mission/approach`, `mission/records`, `model`, `sensed`, `solvers/gekko` |
| `sensed.py` | Sensed sliding-tube algebra, plan checks and certified endpoint transfer. | `common`, `control/fallback`, `models/dynamics`, `numpy`, `python-flint`, `sample_hold` | `mission/execution` |
| `specification.py` | Typed coupled-tube data at the solver boundary. | `control/terminal`, `models/dynamics`, `numpy` | `mission`, `mission/approach`, `solvers/gekko` |
| `uncertainty.py` | Componentwise uncertainty bounds for the physical anomaly-domain LVLH model. | `numpy` | `mission/approach` |
