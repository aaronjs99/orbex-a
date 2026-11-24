# `mission/approach/deployment`

Fleet deployment from the payload cluster.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `component.py` | Compact release and staged, component-aware spacecraft approach references. | `component_policy`, `mission/approach`, `mission/rendezvous`, `numpy` | `mission` |
| `component_policy.py` | Readiness gates and reference previews for explicit spacecraft components. | `control/mpc`, `mission/approach`, `mission/approach/references`, `numpy` | `component` |
