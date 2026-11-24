# `estimation/navigation`

Sensor sampling, delivered packets and bounded navigation filtering.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `boxes.py` | Exact navigation box intersection and numerical representatives. | `common` | `estimation/adaptation`, `estimation/membership`, `filter`, `mission/execution`, `mission/online`, `records` |
| `filter.py` | Bounded navigation consuming declared priors, proved flow and packets. | `boxes`, `common`, `control/fallback`, `models/dynamics`, `moments`, `proofs`, +2 more | `mission` |
| `moments.py` | Exact acceleration-history moment indexing for navigation queries. | `common`, `proofs` | `filter` |
| `noise.py` | Versioned physical-time common random field for synthetic sensor studies. | `numpy` | `records`, `sensors` |
| `records.py` | Immutable processed sensor declarations, packets and navigation products. | `boxes`, `common`, `models/dynamics`, `noise`, `proofs` | `filter`, `mission`, `mission/online`, `mission/safety`, `sensors`, `studies` |
| `sensors.py` | Truth-side bounded sensor sampling with versioned draw ordering. | `common`, `models/dynamics`, `noise`, `numpy`, `proofs`, `records` | `mission` |
| `target_pose.py` | Source-bound target ephemeris inference and delayed pose propagation. | `common`, `models/dynamics`, `numpy`, `proofs`, `python-flint`, `scipy` | `estimation/adaptation`, `filter` |
