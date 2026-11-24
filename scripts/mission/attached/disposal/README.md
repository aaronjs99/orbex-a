# `mission/attached/disposal`

Disposal burn previews and simulation to the entry interface.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `axis.py` | Numerical coupled-wrench planning for a fixed installed thrust direction. | `mission/attached/assembly`, `numpy`, `scipy` | `mission/lifecycle/configuration` |
| `entry.py` | Whole-set descending entry-interface brackets from contiguous held flows. | `models/dynamics` | `simulator` |
| `preview.py` | Finite Cartesian burn previews for numerical throttle selection. | `mission/attached/assembly`, `models/dynamics`, `numpy`, `schedule`, `scipy`, `simulator` | `mission/lifecycle/configuration` |
| `schedule.py` | A numerical sensed-state request schedule; safety admission stays separate. | `common`, `mission/attached/assembly`, `models/dynamics`, `numpy` | `preview`, `simulator` |
| `simulator.py` | Apoapsis finite-burn deorbit, terminating at a declared entry interface. | `entry`, `evidence`, `mission/attached`, `mission/attached/assembly`, `models/dynamics`, `numpy`, +2 more | `mission/lifecycle`, `mission/lifecycle/configuration`, `preview` |
