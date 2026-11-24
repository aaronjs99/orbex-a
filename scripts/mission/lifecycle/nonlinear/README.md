# `mission/lifecycle/nonlinear`

Authenticated approach boundary and nonlinear lifecycle composition.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `boundary.py` | Authenticate arrival evidence and recover the retained Cartesian boundary. | `common`, `contracts`, `control/fallback`, `control/terminal`, `evidence/records`, `mission/contact`, +4 more | `bridge`, `coordinator`, `journal`, `mission` |
| `bridge.py` | An explicit verified old-source approach boundary for a later completion. | `boundary`, `common`, `evidence/records`, `models/dynamics` | `coordinator` |
| `contracts.py` | Shared exact-value contracts and implementation identity for completion. | `common`, `control/fallback` | `boundary`, `coordinator`, `journal`, `mission/lifecycle`, `mission/lifecycle/environment`, `observers` |
| `coordinator.py` | Physical continuation of an authenticated sensed nonlinear approach. | `boundary`, `bridge`, `common`, `contracts`, `evidence/records`, `mission/contact`, +4 more | `service/jobs`, `studies` |
| `journal.py` | Bind and authenticate the complete saved nonlinear hold journal. | `boundary`, `common`, `contracts`, `control/fallback`, `evidence/records`, `models/dynamics`, +1 more | `mission/lifecycle` |
| `observers.py` | Causal navigation prediction and declared synthetic contact observations. | `common`, `contracts`, `mission/contact`, `mission/contact/sensing`, `models/dynamics`, `numpy`, +1 more | `coordinator` |
