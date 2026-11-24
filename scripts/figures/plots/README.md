# `figures/plots`

Static mission figures.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `canvas.py` | Headless figures without selecting a process-wide Matplotlib backend. | `matplotlib` | `figures/data`, `nonlinear` |
| `nonlinear.py` | Static diagnostics from aligned, saved nonlinear mission observations. | `canvas`, `evidence/records`, `figures`, `matplotlib`, `models/dynamics`, `numpy` | `cli`, `figures/data` |
