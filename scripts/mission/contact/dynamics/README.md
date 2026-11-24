# `mission/contact/dynamics`

Contact state, wrench and rigid-body mechanics.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `mechanics.py` | Coupled rigid-body derivatives, compliant pad laws and balance ledgers. | `mission/attached/assembly`, `models/dynamics`, `numpy`, `scipy` | `mission/contact` |
| `rotation.py` | Measured SO(3) supports retained alongside contact quaternion coordinates. | `common`, `proofs/numerics` | `mission/attached/assembly`, `mission/contact`, `mission/contact/sensing`, `strain`, `wrench` |
| `state.py` | Packed physical coordinates and explicit mutable contact/actuator state. | `mission/attached/assembly`, `mission/contact`, `numpy`, `scipy` | `mission/contact` |
| `strain.py` | Stress-free digital bond formation with carried uncertain rest geometry. | `common`, `proofs/numerics`, `rotation` | `mission/contact`, `mission/contact/sensing` |
| `wrench.py` | Redundant adhesive-wrench graph with measured same-event initialization. | `common`, `proofs/numerics`, `rotation` | `mission/contact` |
