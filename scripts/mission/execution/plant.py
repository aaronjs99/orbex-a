"""plant.py: Numerical truth propagation, separate from admission and navigation.

A proposal has no committed time/state effects. The executor commits only after
all admitted numerical trajectories have been computed. Clearance is diagnostic.

Copyright: Aaron John Sabu
"""

from __future__ import annotations

import numpy as np
from orbexa.common.intervals import IntervalBox
from scipy.integrate import solve_ivp


class NonlinearFleetPlant:
    """Numerically propagate each chaser's relative truth over one hold.

    It proposes the motion only; it commits no time or state.
    """

    def __init__(self, target, relative_dynamics, actuator):
        """Store the target, the relative dynamics and the actuator realization."""
        self.target, self.relative_dynamics, self.actuator = (
            target,
            relative_dynamics,
            actuator,
        )

    def propose_hold(self, relative_states, q, h, decision, navigation_supports):
        """Integrate every agent's relative truth over one hold without committing it.

        The target interval proposal is taken for (q, q + h). Each agent is integrated
        over anomaly q to q + h with DOP853 at rtol 1e-11, atol 1e-10 and max step
        min(h, 1e-4), using the realized actuator output as the applied input.

        Inputs:
        - relative_states: agent key to six-element relative state at q (not mutated).
        - q, h: exact start anomaly and hold width.
        - decision: admitted decision giving float commands and applied-input supports.
        - navigation_supports: per-agent navigation actuation supports.

        Outputs:
        - (proposal, results, solutions): target proposal, final relative state per
          agent and the dense solve_ivp solution per agent. Raises RuntimeError if an
          integration fails or is nonfinite; ValueError from the right-hand side if an
          actuator output is not finite, not 3 elements, or outside either support.
        """
        proposal = self.target.propose_interval(q + h)
        commands, applied = decision.float_commands(), dict(decision.applied_inputs)
        results = {}
        solutions = {}
        for key, initial in relative_states.items():
            command = np.asarray(commands[key])

            def rhs(value, relative, key=key, command=command):
                """Return the relative-state derivative at anomaly `value`.

                The realized actuator output must be a finite 3-vector inside the
                admitted
                applied-input support and the navigation actuation subset, otherwise
                ValueError
                is raised.
                """
                actual = np.asarray(
                    self.actuator.apply(key, command, value), dtype=float
                )
                if (
                    actual.shape != (3,)
                    or not np.all(np.isfinite(actual))
                    or not applied[key].contains(IntervalBox.point(actual))
                ):
                    raise ValueError(
                        "Numerical actuator violates its declared applied-input support"
                    )
                if not navigation_supports[key].applied_input.contains(
                    IntervalBox.point(actual)
                ):
                    raise ValueError(
                        "Numerical actuator violates the declared navigation actuation subset"
                    )
                return self.relative_dynamics.relative_rhs(
                    value, relative, proposal.state_at(value)[:6], actual
                )

            solution = solve_ivp(
                rhs,
                (float(q), float(q + h)),
                initial,
                method="DOP853",
                dense_output=True,
                max_step=min(float(h), 0.0001),
                rtol=1e-11,
                atol=1e-10,
            )
            if not solution.success or not np.all(np.isfinite(solution.y)):
                raise RuntimeError(
                    "Numerical relative truth refused: " + str(solution.message)
                )
            results[key], solutions[key] = solution.y[:, -1], solution
        return proposal, results, solutions


def sampled_clearance(state, q, phase, *, target, safety, composite_center):
    """Return the diagnostic clearance (m) of a relative state at anomaly `q`.

    With a composite center the state is rotated into the target body frame and the
    signed distance to the target geometry minus the chaser body radius is returned;
    otherwise the active margin of `safety` for the state and phase is returned. It
    does not substitute for continuous fleet admission.
    """
    # Same declared occupied-union distance as MissionSafety.composite_margin,
    # without copying a full serialized geometry for every offline sample.
    # This diagnostic never substitutes for continuous fleet admission.
    if composite_center is not None:
        body = target.rotation_matrix_at(q).T @ (
            np.asarray(state[:3]) - composite_center
        )
        return float(target.geometry.signed_distance(body) - safety.chaser_body_radius)
    return safety.active_margin_for_state(state, q, phase)
