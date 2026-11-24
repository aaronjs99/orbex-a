# `mission/rendezvous`

Rendezvous routes around the target.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `carrier.py` | A parallel carrier-centred transfer into rotating proximity references. | `numpy`, `routes` | `mission/approach/deployment` |
| `routes.py` | Component-aware attachment assignment and finite-bus reference routes. | `models/geometry`, `numpy`, `scipy` | `carrier`, `mission/approach/deployment` |
