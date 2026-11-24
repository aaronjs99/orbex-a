# `estimation`

Sensing, bounded navigation, membership and parameter adaptation.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `adaptation/` | estimation.adaptation package. | `common`, `estimation/membership`, `estimation/navigation`, `models/dynamics`, `numpy`, `python-flint` | `mission`, `mission/approach`, `mission/records` |
| `membership/` | estimation.membership package. | `common`, `control`, `estimation/navigation`, `models/dynamics`, `proofs` | `estimation/adaptation`, `mission/execution` |
| `navigation/` | Sensor sampling, delivered packets and bounded navigation filtering. | `common`, `control/fallback`, `models/dynamics`, `numpy`, `proofs`, `python-flint`, +1 more | `estimation/adaptation`, `estimation/membership`, `mission`, `mission/execution`, `mission/online`, `mission/safety`, +1 more |
