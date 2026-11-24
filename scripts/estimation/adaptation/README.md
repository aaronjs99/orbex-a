# `estimation/adaptation`

estimation.adaptation package.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `belief.py` | Controller feasible sets and point estimates with reproducible priors. | `models/dynamics`, `numpy` | `mission`, `mission/approach`, `mission/records` |
| `nonlinear.py` | Observation-only initial-orbit contraction and nonlinear drag membership. | `common`, `estimation/membership`, `estimation/navigation`, `models/dynamics`, `python-flint` | `mission` |
