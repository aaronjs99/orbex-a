from gekko import GEKKO
import numpy as np
from orbexa.utils.anomaly import dt_dtheta

m = GEKKO(remote=False)
m.time = np.linspace(0, 1, 10)
q_var = m.Var(value=0)
m.Equation(q_var.dt() == 1)

try:
    dtdq = dt_dtheta(q_var, eccentricity=0.1, mean_motion=0.001, t_periapsis=0, m=m)
    print("dt_dtheta call successful")
    print("Type of dtdq:", type(dtdq))
    # Test if it can be used in an equation
    v = m.Var()
    m.Equation(v.dt() == dtdq)
    print("Equation addition successful")
except Exception as e:
    print("Error:", e)
    import traceback

    traceback.print_exc()
