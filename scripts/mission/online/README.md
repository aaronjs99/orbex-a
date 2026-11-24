# `mission/online`

Independent online execution: clocks, decisions, worker planning and observation.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `clock.py` | Explicit process-comparable wall-clock domains for online receipts. | none | `execution`, `worker` |
| `decision.py` | Prepared evidence for an already admitted command; no proof gate is skipped. | `common`, `control/fallback` | `execution` |
| `execution.py` | Executor-owned finite feedback while a separate planning process runs. | `clock`, `common`, `control/fallback`, `decision`, `observation` | none |
| `observation.py` | Bounded delayed sensing using the installed feedback proof only. | `common`, `control/fallback`, `estimation/navigation`, `mission/safety` | `execution` |
| `planning.py` | Detached finite-backup renewal with complete new nonlinear proofs. | `common`, `control/fallback`, `estimation/navigation`, `mission/safety`, `proofs` | none |
| `worker.py` | One bounded, spawned planning process; polling never waits for its work. | `clock` | none |
