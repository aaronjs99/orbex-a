# `mission/records`

Mission records collected during execution: per-chaser arrays and the nonlinear recorder.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `arrays.py` | Mission history and diagnostic recording, separate from solver execution. | `control/tubes`, `estimation/adaptation`, `mission/approach`, `numpy` | `recorder` |
| `recorder.py` | Separate physical truth, observed navigation and controller mission records. | `arrays`, `control/fallback`, `control/tubes`, `evidence/projections`, `evidence/records`, `models/dynamics`, +1 more | `mission` |
