# `control/mpc`

control.mpc package.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `cost.py` | Local Riccati penalty for a finite MPC horizon. | `numpy`, `scipy` | `mission/approach` |
| `problem.py` | MPC Problem Builder. | `common`, `numpy` | `solvers` |
| `references.py` | Sample moving state goals and their nominal holding inputs for MPC. | `numpy` | `mission/approach/deployment`, `mission/approach/references` |
