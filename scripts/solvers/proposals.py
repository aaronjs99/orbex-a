"""proposals.py: Dynamics-consistent initial guesses for the existing MPC problem boundary.

These hints own no motion authority. The actual moving reference, frame, input
units and RK4 coefficients come from MPCProblem. No toy simulator is substituted.
MIT source attribution: THIRD_PARTY_NOTICES.md.

Copyright: Aaron John Sabu
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AffinePrediction:
    state_matrices: np.ndarray
    input_matrices: np.ndarray
    offsets: np.ndarray
    initial_state: np.ndarray

    @classmethod
    def from_problem(cls, problem):
        n, m, h = problem.num_states, problem.num_inputs, float(problem.anom_step)
        if not np.isfinite(h) or h <= 0:
            raise ValueError("Prediction step must be positive and finite")
        start = float(problem.extra_params.get("start_anom", 0.0))
        tp = problem.extra_params.get("time_periapsis", 0.0)
        a_function = problem.extra_params.get("dynamics_callable")
        d_function = problem.extra_params.get("disturbance_callable")

        def coefficients(q):
            a = np.asarray(
                a_function(q, tp) if callable(a_function) else problem.dynamics_matrix,
                dtype=float,
            )
            raw = problem.extra_params.get("disturbance_vector", np.zeros(n))
            if raw is None:
                raw = np.zeros(n)
            d = np.asarray(
                d_function(q, tp) if callable(d_function) else raw, dtype=float
            )
            if (
                a.shape != (n, n)
                or d.shape != (n,)
                or not np.all(np.isfinite(a))
                or not np.all(np.isfinite(d))
            ):
                raise ValueError("Prediction needs finite affine coefficients")
            return a, d

        b = np.asarray(problem.input_matrix, dtype=float)
        if b.shape != (n, m) or not np.all(np.isfinite(b)):
            raise ValueError("Invalid input matrix")
        aa = []
        bb = []
        dd = []
        for k in range(problem.num_steps - 1):
            q = start + k * h
            c0, cm, c1 = coefficients(q), coefficients(q + h / 2), coefficients(q + h)

            def step(x, u):
                if problem.dynamics_type == "discrete":
                    return c0[0] @ x + b @ u + c0[1]
                if problem.dynamics_type != "continuous":
                    raise ValueError("Unknown nominal dynamics type")

                def rhs(v, c):
                    return c[0] @ v + b @ u + c[1]

                k1 = rhs(x, c0)
                k2 = rhs(x + h * k1 / 2, cm)
                k3 = rhs(x + h * k2 / 2, cm)
                k4 = rhs(x + h * k3, c1)
                return x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

            zero = step(np.zeros(n), np.zeros(m))
            aa.append(
                np.column_stack(
                    [step(np.eye(n)[i], np.zeros(m)) - zero for i in range(n)]
                )
            )
            bb.append(
                np.column_stack(
                    [step(np.zeros(n), np.eye(m)[i]) - zero for i in range(m)]
                )
            )
            dd.append(zero)
        initial = np.asarray(problem.initial_state, dtype=float).copy()
        if initial.shape != (n,) or not np.all(np.isfinite(initial)):
            raise ValueError("Prediction initial state must be finite")
        return cls(np.stack(aa), np.stack(bb), np.stack(dd), initial)

    def rollout(self, controls):
        u = np.asarray(controls, dtype=float)
        count = len(self.state_matrices)
        m = self.input_matrices.shape[2]
        if u.shape != (m, count) or not np.all(np.isfinite(u)):
            raise ValueError("Initial controls must match the prediction intervals")
        x = np.empty((len(self.initial_state), count + 1))
        x[:, 0] = self.initial_state
        for k in range(count):
            x[:, k + 1] = (
                self.state_matrices[k] @ x[:, k]
                + self.input_matrices[k] @ u[:, k]
                + self.offsets[k]
            )
        if not np.all(np.isfinite(x)):
            raise ValueError("Nominal initial-guess rollout overflowed")
        return x


def input_limits(problem):
    bounds = problem.input_bounds
    if bounds is None:
        return np.full(problem.num_inputs, -np.inf), np.full(problem.num_inputs, np.inf)
    if len(bounds) != problem.num_inputs:
        raise ValueError("One input-bound mapping is required per command")
    lo = np.array(
        [-np.inf if b.get("lower") is None else b["lower"] for b in bounds], dtype=float
    )
    hi = np.array(
        [np.inf if b.get("upper") is None else b["upper"] for b in bounds], dtype=float
    )
    if np.any(np.isnan(lo)) or np.any(np.isnan(hi)) or np.any(lo > hi):
        raise ValueError("Initial-guess input bounds are inconsistent")
    return lo, hi
