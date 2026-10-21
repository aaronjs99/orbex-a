# /***********************************************************
# * Copyright (c) 2026                                      *
# * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
# * University of California, Los Angeles                   *
# * Authors: Aaron John Sabu                                *
# * Contact: aaronjs@ucla.edu                               *
# ***********************************************************/

"""Explicit-input fitting and synthetic attitude observations.

No experiment, random seeding, plotting or remote solve occurs on import.
These helpers are independent of the mission's SMID adaptation policy.
"""

import numpy as np


def _time_stream(value):
    times = np.asarray(value, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0)):
        raise ValueError("t_stream must contain at least two finite, strictly increasing times")
    return times


def estimator(t_stream, y_stream, model_functions, prev_theta=None, *,
              backend="gekko", remote=False, disp=False, solver_type=3, max_iter=1000):
    """Fit a scalar stream as a linear combination of explicit basis functions.

    Each basis callable accepts ``(time, m=backend)``. GEKKO receives a symbolic
    parameter; the optional NumPy backend receives a numeric scalar with ``m=np``.
    Return ``(sum_squared_residual, coefficient_array)``. ``prev_theta`` supplies
    the GEKKO initial guess. GEKKO runs locally unless ``remote=True`` is selected.
    """
    times = _time_stream(t_stream)
    values = np.asarray(y_stream, dtype=float)
    if values.shape != times.shape or not np.all(np.isfinite(values)):
        raise ValueError("y_stream must be finite and match t_stream")
    functions = tuple(model_functions)
    if not functions or not all(callable(function) for function in functions):
        raise ValueError("model_functions must be a nonempty sequence of callables")
    initial = np.zeros(len(functions)) if prev_theta is None else np.asarray(prev_theta, dtype=float)
    if initial.shape != (len(functions),) or not np.all(np.isfinite(initial)):
        raise ValueError("prev_theta must contain one finite value per basis function")
    if not isinstance(remote, bool) or not isinstance(disp, bool):
        raise ValueError("remote and disp must be booleans")
    if backend not in {"gekko", "numpy"}:
        raise ValueError("Fitting backend must be 'gekko' or 'numpy'")
    if backend == "numpy":
        if remote:
            raise ValueError("The NumPy fitting backend is local; remote=True requires backend='gekko'")
        basis = np.array([[function(t, m=np) for function in functions] for t in times], dtype=float)
        if basis.shape != (len(times), len(functions)) or not np.all(np.isfinite(basis)):
            raise ValueError("Basis functions must return finite scalar values")
        coefficients = np.linalg.lstsq(basis, values, rcond=None)[0]
        residual = values - basis @ coefficients
        return float(residual @ residual), coefficients
    if isinstance(solver_type, bool) or solver_type not in (1, 2, 3):
        raise ValueError("GEKKO solver_type must be 1, 2 or 3")
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")

    from gekko import GEKKO

    solver = GEKKO(remote=remote)
    try:
        time_parameter = solver.Param(value=times)
        measured = solver.Param(value=values)
        coefficients = [solver.FV(value=float(value)) for value in initial]
        for coefficient in coefficients:
            coefficient.STATUS = 1
        predicted = solver.Intermediate(sum(function(time_parameter, m=solver) * coefficient
                                            for function, coefficient in zip(functions, coefficients)))
        solver.Minimize((measured - predicted) ** 2)
        solver.options.IMODE = 2
        solver.options.SOLVER = int(solver_type)
        solver.options.MAX_ITER = int(max_iter)
        solver.solve(disp=disp)
        estimate = np.array([coefficient.value[0] for coefficient in coefficients], dtype=float)
        error = float(solver.options.OBJFCNVAL)
        if not np.all(np.isfinite(estimate)) or not np.isfinite(error):
            raise RuntimeError("Fitting solver returned a nonfinite result")
        return error, estimate
    finally:
        solver.cleanup()


def gen_y_stream(y_0, phi_0, omega_0, I, t_stream, *, backend="scipy", remote=False,
                 disp=False, rtol=1e-10, atol=1e-12):
    """Rotate a fixed body-frame point through unforced rigid-body motion.

    ``phi_0`` is roll/pitch/yaw in radians; ``omega_0`` is body angular velocity
    in rad/s; times are seconds. Return shape ``(3, number_of_samples)`` including
    the initial point. This numeric generator uses local quaternion integration.
    It does not reinterpret body rates as Euler-angle derivatives.
    """
    if backend != "scipy" or remote:
        raise ValueError("Synthetic attitude observations support only the local 'scipy' backend")
    if not isinstance(remote, bool) or not isinstance(disp, bool):
        raise ValueError("remote and disp must be booleans")
    times = _time_stream(t_stream)
    from scipy.spatial.transform import Rotation
    from orbexa.core.dynamics.attitude import RigidBodyAttitude, euler_to_quaternion, finite_vector

    point = finite_vector(y_0, "Body-frame point", 3)
    quaternion = euler_to_quaternion(phi_0)
    trajectory = RigidBodyAttitude(I, rtol=rtol, atol=atol).propagate(
        quaternion, omega_0, times[1:] - times[0])
    orientations = np.vstack((quaternion, trajectory.quaternions))
    return Rotation.from_quat(orientations).apply(point).T
