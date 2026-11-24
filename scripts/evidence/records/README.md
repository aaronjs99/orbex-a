# `evidence/records`

Mission records, results and proof archives.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `archive.py` | Durable streaming nonlinear witnesses and a reusable source-bound hold view. | `control/fallback`, `models/dynamics`, `numpy`, `zstandard` | `figures/plots`, `mission/lifecycle`, `mission/lifecycle/nonlinear`, `mission/records`, `service/jobs`, `studies` |
| `results.py` | Serializable mission records and standalone result I/O. | `common`, `common/io` | `evidence/projections`, `figures/plots`, `figures/publishing`, `figures/renderers`, `mission/approach/terminal`, `mission/lifecycle`, +3 more |
