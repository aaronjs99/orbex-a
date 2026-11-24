# `common`

Shared paths, exact intervals, serialization, IO and geometry helpers.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `geometry/` | Shared geometric algebra and visibility helpers. | `numpy` | `evidence/projections`, `mission/coordination/allocation/policies`, `models/dynamics`, `models/geometry`, `models/targets`, `solvers/gekko` |
| `io/` | Explicit configuration and data file IO. | `common`, `numpy`, `pyyaml` | `evidence/records`, `figures/renderers`, `mission/lifecycle`, `models/configuration`, `solvers` |
| `contracts.py` | Backend-neutral optimization records and maintained mission identities. | `numpy` | `control/mpc`, `evidence/records`, `mission`, `mission/approach`, `solvers` |
| `encoding.py` | Exact-record encoding of rationals, records and containers for certificate dictionaries. | none | `control/terminal`, `mission/approach/terminal` |
| `intervals.py` | exact rational interval boxes and Arb ball arithmetic shared by every layer. | none | `control`, `control/fallback`, `control/terminal`, `control/tubes`, `estimation/adaptation`, `estimation/membership`, +20 more |
| `layout.py` | dependency-free paths for the checkout, package and generated outputs. | none | `cli`, `common/io`, `figures/publishing`, `mission/attached`, `mission/contact`, `mission/contact/lock`, +7 more |
| `serialization.py` | Source-qualified numerical continuation of a committed mission record. | `numpy` | `evidence`, `mission/approach/terminal`, `mission/lifecycle`, `mission/lifecycle/nonlinear` |
| `solver_options.py` | Backend option contracts that do not import an optimization runtime. | none | `models/configuration`, `solvers/gekko` |
