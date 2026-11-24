# `models/geometry`

Composite spacecraft geometry, attachment patches and mounting.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `composite.py` | Declared composite spacecraft solids and unprepared surface candidates. | `numpy` | `mission/approach/terminal`, `mission/contact`, `mission/lifecycle/configuration`, `mission/rendezvous`, `mission/safety`, `models/targets`, +2 more |
| `constraints.py` | Paper-level rendezvous and docking constraint helpers. | `common/geometry`, `numpy` | `evidence/projections`, `mission/approach`, `models/targets` |
| `mounting.py` | Common right-handed mounting frames for arbitrary structural surface normals. | `models/dynamics`, `numpy` | `mission/approach/physical`, `mission/contact` |
| `rendezvous.py` | Explicit bus-patch matching; finite-body route policies live separately. | `numpy`, `scipy` | `models/targets` |
