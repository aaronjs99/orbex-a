# `mission/execution`

Held-command admission, physical commit and sensing delivery for each hold.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `bindings.py` | Independent binding of an optimized tube stage to its declared sources. | `common`, `control/fallback`, `models/dynamics`, `numpy`, `proofs` | `coordinator` |
| `blackout.py` | Finite causal observation propagation during declared sensor blackouts. | `common`, `control/fallback`, `mission/safety` | `coordinator` |
| `contracts.py` | Execution evidence and failures, independent of the execution coordinator. | `common`, `estimation/navigation`, `mission/approach` | `coordinator` |
| `coordinator.py` | Sensed dense execution with separate nonlinear truth and finite fallback. | `bindings`, `blackout`, `common`, `contracts`, `control`, `control/fallback`, +12 more | `mission` |
| `nominal.py` | Certified CH nominal propagation and physical tube endpoint transfer. | `common`, `control/tubes`, `estimation/navigation`, `models/dynamics`, `numpy`, `proofs` | `coordinator` |
| `plant.py` | Numerical truth propagation, separate from admission and navigation. | `common`, `numpy`, `scipy` | `coordinator` |
| `sensing.py` | Acquisition and delayed packet delivery for nonlinear execution. | `models/dynamics` | `coordinator` |
