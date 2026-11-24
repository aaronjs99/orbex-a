# `models/targets`

Target truth and observed target previews.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `cylinder.py` | Unactuated rigid-body attitude with the supported cylinder collision envelope. | `models/configuration`, `models/dynamics`, `models/geometry`, `numpy`, `prescribed` | `snapshot` |
| `docking.py` | Cylindrical side-point assignment from each chaser approach direction. | `numpy`, `prescribed`, `scipy` | `prescribed` |
| `nonlinear.py` | Separate executed nonlinear target truth and observation-driven preview. | `models/dynamics`, `models/geometry`, `numpy`, `scipy` | `mission` |
| `prescribed.py` | Prescribed cylindrical target geometry and anomaly-domain kinematics. | `common/geometry`, `docking`, `models/configuration`, `models/geometry`, `numpy` | `cylinder`, `docking`, `mission/approach`, `mission/approach/references` |
| `snapshot.py` | Physical target reconstruction from recorded snapshots and target signatures. | `cylinder`, `spacecraft` | `mission/approach/terminal` |
| `spacecraft.py` | Rigid spacecraft motion with explicit component solids and surface patches. | `models/dynamics`, `models/geometry`, `numpy` | `mission/lifecycle/configuration`, `snapshot` |
