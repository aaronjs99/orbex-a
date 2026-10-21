"""Exact acceleration-prefix proposals followed by finite coordinated braking.

These curves carry no admission authority. The ordinary nonlinear tracking,
whole-fleet geometry, force, fuel and uncertain-entry checks still apply.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from orbexa.control.certification.reachability import _exact
from orbexa.control.fallback.refuge import _vec
from orbexa.simulation.nonlinear.safety.braking import coordinated_braking_references
from orbexa.simulation.nonlinear.safety.refuge import C2RefugeReference, PolynomialReferencePiece


@dataclass(frozen=True)
class AccelerationContingencyOptions:
    """Finite far-flight reference proposals; admission limits are separate."""
    coast_holds: int = 16
    transition_holds: int = 64
    acceleration_limit_m_per_rad2: float = 50000.

    def __post_init__(self):
        for name in ('coast_holds', 'transition_holds'):
            value = getattr(self, name)
            minimum = 0 if name == 'coast_holds' else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(name+' must be an explicit nonnegative/positive integer')
        value = self.acceleration_limit_m_per_rad2
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError('The proposal acceleration limit must be positive and finite')


def acceleration_contingency_references(
    *, initial_states: Mapping[str, Sequence], initial_accelerations: Mapping[str, Sequence],
    start, coast_duration, transition_duration, common_duration, relative_duration,
    holding_duration, outward_displacement=(0, 0, 0), frame: str, clock_id: str,
) -> dict[str, C2RefugeReference]:
    """Propose a C2 prefix, bounded-jerk transition, braking and final hold.

    The supplied accelerations are declared proposal values, for example
    clipped midpoints of prospective interval-flow derivatives. They are never
    observations or true accelerations. A constant acceleration prefix lasts
    ``coast_duration``; constant jerk then joins the coordinated braking law.
    The end velocity is solved exactly because the braking acceleration also
    depends on that velocity. All durations use one explicit reference clock.
    """
    if (not isinstance(initial_states, Mapping) or not initial_states
            or not isinstance(initial_accelerations, Mapping)
            or set(initial_accelerations) != set(initial_states)):
        raise ValueError('The complete same named fleet needs states and proposal accelerations')
    states = {key: _vec(value, 6, 'initial fleet state') for key, value in initial_states.items()}
    accelerations = {key: _vec(initial_accelerations[key], 3, 'proposal acceleration') for key in states}
    q, coast, transition, common, relative, hold = map(_exact,
        (start, coast_duration, transition_duration, common_duration, relative_duration, holding_duration))
    if coast < 0 or transition <= 0 or not 0 < relative <= common or hold <= 0:
        raise ValueError('Declare nonnegative coast, positive transition/hold and 0 < relative <= common')
    count = len(states)
    mean_velocity = tuple(sum(value[axis+3] for value in states.values())/count for axis in range(3))
    mean_acceleration = tuple(sum(value[axis] for value in accelerations.values())/count for axis in range(3))
    common_end_velocity = tuple((v+a*(coast+transition/2))/(1+3*transition/(2*common))
                                for v, a in zip(mean_velocity, mean_acceleration))
    prefixes, endpoints = {}, {}
    for key, state in states.items():
        rows, ending = [], []
        for axis in range(3):
            p, v, a = state[axis], state[axis+3], accelerations[key][axis]
            relative_end_velocity = ((v-mean_velocity[axis])
                +(a-mean_acceleration[axis])*(coast+transition/2))/(1+3*transition/(2*relative))
            end_velocity = common_end_velocity[axis]+relative_end_velocity
            end_acceleration = -3*common_end_velocity[axis]/common-3*relative_end_velocity/relative
            p_coast, v_coast = p+v*coast+a*coast**2/2, v+a*coast
            jerk = (end_acceleration-a)/transition
            rows.append((p_coast, v_coast, a/2, jerk/6))
            end_position = p_coast+v_coast*transition+a*transition**2/2+jerk*transition**3/6
            if v_coast+a*transition+jerk*transition**2/2 != end_velocity:
                raise ArithmeticError('Exact contingency velocity matching failed')
            ending.append((end_position, end_velocity))
        pieces = []
        if coast:
            pieces.append(PolynomialReferencePiece(q, q+coast,
                tuple((state[axis], state[axis+3], accelerations[key][axis]/2) for axis in range(3))))
        pieces.append(PolynomialReferencePiece(q+coast, q+coast+transition, tuple(rows)))
        prefixes[key] = pieces
        endpoints[key] = tuple(value[0] for value in ending)+tuple(value[1] for value in ending)
    braking = coordinated_braking_references(initial_states=endpoints, start=q+coast+transition,
        common_duration=common, relative_duration=relative, holding_duration=hold,
        outward_displacement=outward_displacement, frame=frame, clock_id=clock_id)
    return {key: C2RefugeReference((*prefixes[key], *reference.pieces), frame=frame, clock_id=clock_id)
            for key, reference in braking.items()}
