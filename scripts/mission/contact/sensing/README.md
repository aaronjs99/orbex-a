# `mission/contact/sensing`

Contact observations and strain sensing.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `observation.py` | Same-event bounded contact observations intersect the carried state set. | `common`, `mission/contact`, `mission/contact/dynamics`, `models/dynamics`, `numpy`, `proofs/numerics`, +2 more | `mission/contact`, `mission/lifecycle/nonlinear`, `strain_observation` |
| `strain_observation.py` | Intersect carried bond strain with bounded same-event pad wrench data. | `common`, `mission/contact`, `mission/contact/dynamics`, `models/dynamics`, `numpy`, `observation`, +3 more | `mission/contact` |
