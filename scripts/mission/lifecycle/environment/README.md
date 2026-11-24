# `mission/lifecycle/environment`

Physical environment models for completion.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `enclosure.py` | Typed interval counterpart of the maintained completion J2/drag adapter. | `common`, `models/dynamics`, `point`, `proofs/numerics` | `mission/attached`, `mission/attached/assembly`, `mission/contact`, `mission/contact/lock` |
| `point.py` | Continuation adapter for the declared Cartesian J2/vector-drag environment. | `mission/attached/assembly`, `mission/contact`, `mission/lifecycle/nonlinear`, `models/dynamics`, `numpy` | `enclosure`, `mission/lifecycle/nonlinear` |
