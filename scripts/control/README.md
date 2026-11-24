# `control`

Tubes, MPC formulation, finite fallback, terminal control and proof services.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `fallback/` | Finite feedback fallback: refuges, fleet plans, admission and fuel accounting. | `common`, `models/dynamics`, `proofs` | `control/tubes`, `estimation/navigation`, `evidence/records`, `mission`, `mission/execution`, `mission/lifecycle/nonlinear`, +3 more |
| `mpc/` | control.mpc package. | `common`, `numpy`, `scipy` | `mission/approach`, `mission/approach/deployment`, `mission/approach/references`, `solvers` |
| `terminal/` | control.terminal package. | `common`, `models/dynamics`, `proofs`, `python-flint` | `control/tubes`, `mission`, `mission/approach/arrival`, `mission/approach/terminal`, `mission/lifecycle`, `mission/lifecycle/nonlinear`, +1 more |
| `tubes/` | control.tubes package. | `common`, `control/fallback`, `control/terminal`, `models/dynamics`, `numpy`, `python-flint` | `mission`, `mission/approach`, `mission/execution`, `mission/records`, `solvers/gekko` |
| `actuation.py` | Declared command support and independent numerical actuator realizations. | `common`, `models/dynamics`, `numpy`, `proofs` | `estimation/membership`, `mission/execution` |
