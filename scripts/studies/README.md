# `studies`

Study arms: requests, execution, telemetry and metrics.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `api.py` | Freeze and execute one explicit prospective nonlinear study arm. | `common`, `evidence`, `execution`, `request` | `cli` |
| `execution.py` | Execution lifecycle for one already frozen nonlinear study arm. | `common`, `estimation/navigation`, `evidence`, `evidence/records`, `metrics`, `mission`, +8 more | `api` |
| `metrics.py` | Study metrics from recorded physical prefixes, sampled histories and refusals. | `evidence/records`, `numpy` | `execution` |
| `records.py` | Atomic study records and acknowledged execution progress. | `numpy` | `execution` |
| `request.py` | Explicit, prospective inputs for one nonlinear study arm. | `common` | `api`, `execution` |
