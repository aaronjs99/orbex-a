# `mission/approach/terminal`

Terminal execution and the physical terminal handoff.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `execution.py` | Offline execution of a certified physical terminal stage from its real entry. | `common`, `control/terminal`, `models/geometry`, `numpy`, `proofs`, `scipy` | `handoff` |
| `handoff.py` | Verify a separate physical terminal record before using its actual endpoint. | `common`, `control/terminal`, `evidence/records`, `execution`, `mission/contact`, `models/geometry`, +3 more | `mission/contact`, `mission/lifecycle` |
