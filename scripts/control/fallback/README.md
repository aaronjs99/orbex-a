# `control/fallback`

Finite feedback fallback: refuges, fleet plans, admission and fuel accounting.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `common.py` | Exact fleet identity, ordered member and plant-context checks. | `refuge` | `geometry`, `plan`, `records`, `replacement`, `runtime` |
| `fuel.py` | Exact available resources for selecting a complete finite continuation. | `models/dynamics` | `mission/execution`, `mission/safety` |
| `geometry.py` | Whole-fleet occupied-component and peer separation certificates. | `common`, `polynomial`, `proofs`, `refuge` | `mission/safety`, `plan`, `records` |
| `plan.py` | Complete finite feedback plans, integrity and remaining resources. | `common`, `geometry`, `polynomial`, `refuge` | `estimation/navigation`, `mission/execution`, `mission/online`, `mission/safety`, `records`, `replacement`, +1 more |
| `polynomial.py` | Exact shared-clock polynomial references and independent peer bounds. | `common`, `refuge` | `geometry`, `mission/safety`, `plan` |
| `records.py` | Immutable performance attempts, proposals, decisions and resource evidence. | `common`, `geometry`, `plan`, `refuge` | `mission`, `mission/execution`, `mission/online`, `mission/safety`, `replacement`, `runtime` |
| `refuge.py` | Finite-domain sampled feedback refuge certificates. | `common` | `common`, `control/tubes`, `evidence/records`, `geometry`, `mission`, `mission/execution`, +9 more |
| `replacement.py` | Prepare and independently activate one future certified feedback policy. | `common`, `plan`, `records`, `refuge`, `runtime` | `mission/online` |
| `runtime.py` | Transactional held-fleet admission, fallback dispatch and commit. | `common`, `plan`, `proofs`, `records`, `refuge` | `mission/execution`, `mission/online`, `mission/safety`, `replacement` |
