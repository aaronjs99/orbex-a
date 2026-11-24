# `mission/attached/assembly`

Assembly mass, inertia and state.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `flow.py` | Carried whole-state inclusion for the maintained postcapture rigid model. | `common`, `mission/contact/dynamics`, `mission/lifecycle/environment`, `model`, `models/dynamics`, `numpy`, +2 more | `mission/attached/disposal`, `mission/contact`, `model`, `observation` |
| `model.py` | Physical post-capture rigid assembly, in SI units and a fixed inertial frame. | `common`, `flow`, `mission/attached`, `mission/contact`, `models/dynamics`, `numpy`, +1 more | `flow`, `mission/approach/physical`, `mission/attached`, `mission/attached/disposal`, `mission/contact`, `mission/contact/dynamics`, +4 more |
| `observation.py` | Bounded chaser-carried navigation for the captured rigid assembly. | `common`, `flow`, `mission/contact`, `model`, `models/dynamics`, `numpy`, +3 more | `mission/attached`, `mission/attached/disposal` |
