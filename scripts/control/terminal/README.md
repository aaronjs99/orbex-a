# `control/terminal`

control.terminal package.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `body_jets.py` | Validated componentwise Euler and fixed-body-port derivative enclosures. | `common`, `physical`, `python-flint` | `componentwise` |
| `componentwise.py` | Full-inertia componentwise terminal bounds for midpoint feedforward. | `body_jets`, `common`, `composite`, `physical`, `tracking` | `mission/approach/terminal` |
| `composite.py` | Physical terminal bounds for an explicitly declared spacecraft part union. | `physical`, `proofs` | `componentwise`, `mission/approach/terminal` |
| `contact.py` | Sensing-only sufficient bridge to the physical capture speed gate. | `common`, `models/dynamics` | `control/tubes`, `mission`, `mission/lifecycle/nonlinear`, `solvers/gekko` |
| `physical.py` | Derived finite-domain terminal bounds for an eccentric rigid-body target. | `common`, `proofs`, `python-flint`, `tracking` | `body_jets`, `componentwise`, `composite`, `mission/approach/terminal`, `mission/lifecycle` |
| `region.py` | Declared reference-centered region shared by arrival planning and entry screens. | none | `control/tubes`, `mission/approach/arrival`, `solvers/gekko` |
| `tracking.py` | Exact algebra certificates for a held-input terminal tracking controller. | `common`, `proofs` | `componentwise`, `mission/approach/terminal`, `physical` |
