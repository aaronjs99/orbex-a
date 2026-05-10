from gekko import GEKKO
import numpy as np
from orbexa.utils.anomaly import dt_dq

m = GEKKO(remote=False)
m.time = np.linspace(0, 1, 10)
q_var = m.Var(value=0)
m.Equation(q_var.dt() == 1)

try:
    dt_dq_expr = dt_dq(
        q_var, eccentricity=0.1, mean_motion=0.001, t_periapsis=0, solver=m
    )
    print("dt_dq call successful")
    print("Type of dt_dq_expr:", type(dt_dq_expr))
    # Test if it can be used in an equation
    v = m.Var()
    m.Equation(v.dt() == dt_dq_expr)
    print("Equation addition successful")
except Exception as e:
    print("Error:", e)
    import traceback

    traceback.print_exc()
