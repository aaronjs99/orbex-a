# `mission/attached`

Assembled-spacecraft flight after attachment.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `assembly/` | Assembly mass, inertia and state. | `common`, `mission/attached`, `mission/contact`, `mission/contact/dynamics`, `mission/lifecycle/environment`, `models/dynamics`, +4 more | `mission/approach/physical`, `mission/attached`, `mission/attached/disposal`, `mission/contact`, `mission/contact/dynamics`, `mission/lifecycle`, +2 more |
| `disposal/` | Disposal burn previews and simulation to the entry interface. | `common`, `evidence`, `mission/attached`, `mission/attached/assembly`, `models/dynamics`, `numpy`, +1 more | `mission/lifecycle`, `mission/lifecycle/configuration` |
| `detumbling.py` | Fuel-limited detumbling by bounded external thruster torque. | `mission/attached/assembly`, `numpy`, `propulsion` | `mission/lifecycle`, `mission/lifecycle/configuration` |
| `loads.py` | Whole-held-step Newton--Euler bounds for a declared rigid attached assembly. | `common`, `mission/attached/assembly`, `mission/lifecycle/environment`, `models/dynamics`, `numpy`, `proofs/numerics`, +1 more | `mission/attached/assembly` |
| `propulsion.py` | Bounded external-thruster wrench allocation shared by mission stages. | `mission/attached/assembly`, `models/dynamics`, `numpy`, `scipy` | `detumbling`, `mission/attached/disposal`, `mission/lifecycle` |
