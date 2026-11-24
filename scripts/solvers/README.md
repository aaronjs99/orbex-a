# `solvers`

ORBEX-A Solvers Module.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `gekko/` | solvers.gekko package. | `common`, `common/geometry`, `control/terminal`, `control/tubes`, `gekko`, `models/dynamics`, +4 more | `evidence`, `solvers` |
| `base.py` | Abstract base classes and data structures for MPC solvers. | `common`, `numpy`, `scipy` | `__init__`, `casadi_solver`, `proposal_seed`, `scipy_solver`, `solvers/gekko` |
| `casadi_solver.py` | CasADi solver backend for MPC optimization. | `base`, `casadi`, `numpy` | `__init__` |
| `controller.py` | ORBEX-A MPC Controller Module. | `control/mpc`, `numpy`, `proposal_seed` | `evidence`, `mission/approach` |
| `proposal_seed.py` | Optional warm-start decorator preserving the authoritative solver boundary. | `base`, `numpy`, `proposals` | `controller` |
| `proposals.py` | Dynamics-consistent initial guesses for the existing MPC problem boundary. | `numpy` | `proposal_seed` |
| `scipy_solver.py` | SciPy solver backend for MPC optimization. | `base`, `numpy`, `scipy` | `__init__` |
