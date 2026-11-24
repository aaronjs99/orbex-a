# `evidence/projections`

Projections of recorded mission data into sample tables and Studio views.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `samples.py` | Shared recorded-sample alignment and saved target kinematics. | `common/geometry`, `evidence/records`, `models/geometry`, `numpy` | `studio` |
| `studio.py` | Portable, sample-aligned projection of a recorded mission for browser replay. | `numpy`, `samples` | `figures/publishing`, `mission/records`, `service/jobs` |
