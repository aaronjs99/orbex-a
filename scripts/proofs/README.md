# `proofs`

Validated enclosures: Picard flows, orbital and attitude hold proofs, geometry bounds.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `numerics/` | Rigid-body interval arithmetic shared by contact proofs. | `common` | `mission/attached`, `mission/attached/assembly`, `mission/contact`, `mission/contact/dynamics`, `mission/contact/lock`, `mission/contact/sensing`, +1 more |
| `attitude.py` | Conditional quaternion/Euler inclusions driven by nonlinear ephemeris. | `common`, `geometry`, `models/dynamics`, `nonlinear`, `numpy`, `python-flint`, +2 more | `control`, `estimation/navigation`, `mission`, `mission/safety`, `rotation` |
| `geometry.py` | Continuous exclusion and reference clearance for a declared convex union. | `common`, `models/geometry`, `numpy`, `python-flint`, `reachability` | `attitude`, `control/terminal`, `mission/approach/terminal` |
| `nonlinear.py` | Finite-budget Arb inclusions for the explicitly declared nonlinear plant. | `common`, `models/dynamics`, `python-flint`, `reachability` | `attitude`, `estimation/membership`, `estimation/navigation`, `mission`, `mission/execution`, `mission/online`, +1 more |
| `orbital.py` | Authenticated analytical orbital holds with separate VoC endpoints. | `common`, `python-flint`, `reachability` | `mission/execution` |
| `reachability.py` | Validated continuous held-input flow and swept collision certificates. | `common`, `python-flint` | `attitude`, `control/fallback`, `control/terminal`, `estimation/membership`, `estimation/navigation`, `geometry`, +7 more |
| `rotation.py` | Exact occupied-set supports when a body's attitude is any proper rotation. | `attitude`, `common` | `mission/safety` |
