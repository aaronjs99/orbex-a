# `estimation/membership`

estimation.membership package.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `integral.py` | Bounded-error identification from three positions and fixed actuator cells. | `models/dynamics` | none |
| `linked.py` | Bounded nonlinear set membership with shared affine-actuator hypotheses. | `common`, `control`, `estimation/navigation`, `models/dynamics`, `nonlinear`, `proofs` | `estimation/adaptation` |
| `nonlinear.py` | Conservative candidate-box membership for the declared nonlinear model. | `common`, `control`, `models/dynamics`, `proofs` | `estimation/adaptation`, `linked`, `mission/execution` |
