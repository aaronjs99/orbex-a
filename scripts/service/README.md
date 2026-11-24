# `service`

Loopback live simulation service and HTTP adapter.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `jobs/` | Nonlinear live-job execution in an isolated worker process. | `common`, `evidence`, `evidence/projections`, `evidence/records`, `figures/renderers`, `mission`, +5 more | `service` |
| `contracts.py` | Explicit live protocol, parameter limits and worker input contracts. | `common` | `cli`, `http`, `session` |
| `http.py` | Bounded loopback HTTP adapter for the injected live session owner. | `contracts`, `session` | `cli` |
| `process.py` | Spawned frozen-source bootstrap and isolated solver descendant lifetime. | `evidence`, `service/jobs` | `session` |
| `session.py` | One replaceable worker with retained jobs and monotone state revisions. | `contracts`, `evidence`, `process`, `snapshot` | `cli`, `http` |
| `snapshot.py` | Coherent Python and explicit-input snapshots before worker execution. | `common`, `evidence` | `session` |
