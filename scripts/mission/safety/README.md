# `mission/safety`

Finite fallback refuges, candidates and backup planning for the fleet.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `braking.py` | Exact fleet reference proposals with separate common and relative braking. | `common`, `control/fallback`, `reference` | `contingency`, `planner` |
| `cache.py` | Frozen proof identities with mutable-context guards. | `control/fallback` | `drift`, `geometry` |
| `candidates.py` | Finite refuge construction, target preflight and bounded candidate search. | `common`, `control/fallback`, `drift`, `reference`, `resources` | `mission/online`, `planner` |
| `contingency.py` | Exact acceleration-prefix proposals followed by finite coordinated braking. | `braking`, `common`, `control/fallback`, `reference` | `mission`, `planner` |
| `drift.py` | Actual nonlinear drift enclosures for the exact-zero nominal refuge. | `cache`, `common`, `control/fallback`, `models/dynamics`, `proofs`, `python-flint` | `candidates`, `mission/execution`, `mission/online`, `planner` |
| `geometry.py` | Complete occupied geometry from declared target pose enclosures. | `cache`, `common`, `control/fallback`, `proofs`, `python-flint` | `mission/online`, `planner` |
| `planner.py` | Prepare complete nonlinear fleet refuge proofs from navigation snapshots. | `braking`, `candidates`, `common`, `contingency`, `control/fallback`, `drift`, +9 more | `mission`, `mission/execution` |
| `providers.py` | Serializable fixed proof sources for detached planning contexts. | none | `mission/online`, `planner` |
| `proximity.py` | Target-size-aware proposal regimes; collision proofs remain authoritative. | `models/geometry` | `planner` |
| `reference.py` | Exact C2 polynomial routes and Bernstein reference enclosures. | `common`, `control/fallback` | `braking`, `candidates`, `contingency`, `mission/online`, `planner` |
| `resources.py` | Study propulsion envelopes for finite refuge commands. | `common`, `control/fallback`, `proofs` | `candidates`, `planner` |
| `target_bounds.py` | Validated moving-goal errors about an observation-driven numerical preview. | `common`, `models/dynamics`, `numpy`, `proofs`, `python-flint`, `scipy` | `mission`, `mission/lifecycle/nonlinear` |
