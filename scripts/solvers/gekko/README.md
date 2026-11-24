# `solvers/gekko`

solvers.gekko package.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `geometry.py` | Nonlinear support-plane exclusion of every declared spacecraft component. | `models/geometry`, `numpy` | `tube` |
| `iterations.py` | Read authoritative per-attempt IPOPT iterations without zero defaults. | none | `tube` |
| `solver.py` | GEKKO solver backend for MPC optimization. | `common`, `common/geometry`, `gekko`, `models/dynamics`, `numpy`, `solvers`, +1 more | `solvers` |
| `tube.py` | Coupled nominal and dynamic-tube multiple shooting for GEKKO/IPOPT. | `common`, `common/geometry`, `control/terminal`, `control/tubes`, `gekko`, `geometry`, +5 more | `evidence`, `solver` |
