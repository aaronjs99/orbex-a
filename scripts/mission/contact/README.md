# `mission/contact`

Finite contact, capture and lock.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `control/` | Observed contact control and load guards. | `models/dynamics`, `numpy`, `scipy` | `mission/contact` |
| `dynamics/` | Contact state, wrench and rigid-body mechanics. | `common`, `mission/attached/assembly`, `mission/contact`, `models/dynamics`, `numpy`, `proofs/numerics`, +1 more | `mission/attached/assembly`, `mission/contact`, `mission/contact/sensing` |
| `lock/` | Finite lock transition to rigid attachment. | `common`, `mission/contact`, `mission/lifecycle/environment`, `models/dynamics`, `numpy`, `proofs/numerics`, +1 more | `mission/contact` |
| `sensing/` | Contact observations and strain sensing. | `common`, `mission/contact`, `mission/contact/dynamics`, `models/dynamics`, `numpy`, `proofs/numerics`, +2 more | `mission/contact`, `mission/lifecycle/nonlinear` |
| `boundary.py` | Source-bound transfer of a terminal navigation/tube set into contact. | `common`, `models/dynamics`, `models/geometry`, `numpy`, `proofs/numerics`, `python-flint`, +1 more | `mission/attached/assembly`, `mission/contact/sensing`, `mission/lifecycle/nonlinear` |
| `capture.py` | Qualified approach admission followed by finite compliant physical capture. | `common`, `contracts`, `mission/approach/terminal`, `mission/attached/assembly`, `mission/contact/dynamics`, `mission/contact/lock`, +5 more | `mission/approach/physical`, `mission/approach/terminal`, `mission/lifecycle`, `mission/lifecycle/configuration`, `mission/lifecycle/nonlinear` |
| `contracts.py` | Public SI contracts for finite compliant contact, independent of execution. | `mission/attached/assembly`, `models/dynamics`, `numpy` | `capture`, `mission/contact/dynamics`, `mission/lifecycle/configuration`, `records`, `simulator` |
| `enclosure.py` | Finite-budget continuous enclosures for the declared compliant-pad model. | `common`, `mission/attached/assembly`, `mission/contact/dynamics`, `mission/lifecycle/environment`, `models/dynamics`, `numpy`, +3 more | `mission/contact/lock`, `mission/contact/sensing`, `simulator` |
| `records.py` | Contact history and conservation audit serialization. | `contracts`, `numpy` | `simulator` |
| `simulator.py` | Finite compliant capture of a rigid target by independently moving buses. | `common`, `contracts`, `enclosure`, `mission/contact/control`, `mission/contact/dynamics`, `mission/contact/sensing`, +5 more | `capture`, `mission/attached/assembly`, `mission/lifecycle`, `mission/lifecycle/environment`, `mission/lifecycle/nonlinear` |
