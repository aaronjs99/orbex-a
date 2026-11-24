# `mission/coordination/allocation`

Task allocation contracts, networks and policies.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `policies/` | Greedy, Hungarian and auction assignment policies. | `common/geometry`, `mission/coordination/allocation`, `numpy`, `scipy` | `mission/coordination/allocation` |
| `__main__.py` | Run the optional allocation demonstration explicitly. | none | none |
| `contracts.py` | Typed agent memory and policy/network contracts. | `numpy` | `demo`, `mission/coordination/allocation/policies`, `networks`, `system` |
| `demo.py` | Optional allocation figures and explicit demonstration. | `contracts`, `matplotlib`, `mission/coordination/allocation/policies`, `networks`, `numpy`, `system` | none |
| `networks.py` | Explicit static, nearest-neighbor and radius communication networks. | `contracts`, `numpy` | `demo` |
| `system.py` | Agent construction and allocation-policy scheduling. | `contracts`, `numpy` | `demo` |
