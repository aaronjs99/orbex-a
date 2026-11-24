# `models/dynamics`

Translational, attitude and rigid-body dynamics with their clocks and frames.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `anomaly.py` | Keplerian true-anomaly helpers: orbital radius terms, anomaly-to-time conversion and the anomaly rate dq/dt with its reciprocal. | `matplotlib`, `numpy` | `mission/contact`, `orbital`, `rigid_motion`, `solvers/gekko` |
| `attitude.py` | Standalone rigid-body attitude dynamics, separate from prescribed mission motion. | `numpy`, `scipy` | `mission/approach/physical`, `mission/attached`, `mission/attached/assembly`, `mission/attached/disposal`, `mission/contact`, `mission/contact/control`, +6 more |
| `orbital.py` | ORBEX-A Orbital Dynamics Models. | `anomaly`, `numpy` | `mission/approach`, `mission/execution` |
| `parameters.py` | Fixed plant parameters, independent of mission orchestration and estimators. | `models/configuration`, `numpy` | `estimation/adaptation`, `mission/approach` |
| `reference_frame.py` | Immutable commanded orbit coordinates; these are not the perturbed LVLH. | `common`, `numpy`, `python-flint` | `control`, `control/fallback`, `control/terminal`, `control/tubes`, `estimation/adaptation`, `estimation/membership`, +21 more |
| `relative.py` | Declared Cartesian forces and exact motion in commanded orbit coordinates. | `numpy`, `reference_frame` | `estimation/adaptation`, `estimation/membership`, `estimation/navigation`, `mission`, `mission/execution`, `mission/lifecycle/environment`, +4 more |
| `rigid_motion.py` | Shape-independent unactuated quaternion/Euler motion on a declared orbital clock. | `anomaly`, `attitude`, `common/geometry`, `numpy`, `scipy` | `models/targets` |
