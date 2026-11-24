# `mission/contact/control`

Observed contact control and load guards.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `controller.py` | Measured-state stationkeeping and finite-bandwidth positioner commands. | `models/dynamics`, `numpy`, `scipy` | `mission/contact` |
| `guards.py` | Capacity and geometry guards with their observed extrema and clearances. | `numpy` | `mission/contact` |
