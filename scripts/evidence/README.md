# `evidence`

Durable journals, native solver evidence, records and accounting.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `projections/` | Projections of recorded mission data into sample tables and Studio views. | `common/geometry`, `evidence/records`, `models/geometry`, `numpy` | `figures/publishing`, `mission/records`, `service/jobs` |
| `records/` | Mission records, results and proof archives. | `common`, `common/io`, `control/fallback`, `models/dynamics`, `numpy`, `zstandard` | `evidence/projections`, `figures/plots`, `figures/publishing`, `figures/renderers`, `mission/approach/terminal`, `mission/lifecycle`, +4 more |
| `artifacts.py` | Atomic worker records and streamed source-file identities. | none | `native`, `service`, `service/jobs` |
| `journal.py` | Durable publication and acknowledged completed-hold retention. | none | `mission/execution` |
| `native.py` | Shared native solver transcript retention for isolated study and live jobs. | `artifacts`, `common`, `gekko`, `solvers`, `solvers/gekko` | `service/jobs`, `studies` |
| `source.py` | Freeze coherent study source, explicit inputs, and runtime identities. | none | `studies` |
| `stages.py` | Optional append-only storage of complete accepted stage evidence. | `models/dynamics` | `mission/attached/disposal` |
