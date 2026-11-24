# `mission/contact/lock`

Finite lock transition to rigid attachment.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `chart.py` | Exact positioner motion in a relative-to-initial attitude chart. | `proofs/numerics` | `transition` |
| `transition.py` | Finite, load-bounded closure of the settled fleet's residual relative twist. | `chart`, `common`, `mission/contact`, `mission/lifecycle/environment`, `models/dynamics`, `numpy`, +2 more | `mission/contact` |
