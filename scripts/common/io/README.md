# `common/io`

Explicit configuration and data file IO.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `configuration.py` | Explicit YAML loading and relocated-checkout configuration fallback. | `common`, `pyyaml` | `models/configuration`, `solvers` |
| `data.py` | Atomic JSON persistence, data conversion and result-folder naming. | `numpy` | `evidence/records`, `figures/renderers`, `mission/lifecycle` |
