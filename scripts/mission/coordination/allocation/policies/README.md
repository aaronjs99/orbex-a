# `mission/coordination/allocation/policies`

Greedy, Hungarian and auction assignment policies.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `auction.py` | Incremental auction policy with caller-owned stepping order. | `mission/coordination/allocation`, `numpy` | `mission/coordination/allocation` |
| `greedy.py` | Local goal-swap policy with its original distance heuristic. | `common/geometry`, `mission/coordination/allocation` | `mission/coordination/allocation` |
| `hungarian.py` | Local Hungarian reassignment with Euclidean travel costs. | `mission/coordination/allocation`, `numpy`, `scipy` | none |
