"""Exact fleet reference proposals with separate common and relative braking.

These curves propose a finite refuge. They do not certify tracking, force,
fuel, obstacle clearance or entry from an uncertain state. The existing
nonlinear fleet plan constructor must prove all of those before admission.
"""
from __future__ import annotations

from fractions import Fraction
from math import comb
from typing import Mapping, Sequence

from orbexa.control.certification.reachability import _exact
from orbexa.control.fallback.refuge import _identity, _vec
from orbexa.simulation.nonlinear.safety.reference import C2RefugeReference, PolynomialReferencePiece


def _shift(coefficients, offset):
    """Change the exact power basis from t to t-offset."""
    return tuple(sum(coefficients[j] * comb(j, i) * offset**(j-i)
                     for j in range(i, len(coefficients)))
                 for i in range(len(coefficients)))


def coordinated_braking_references(
    *, initial_states: Mapping[str, Sequence], start, common_duration,
    relative_duration, holding_duration, outward_displacement=(0, 0, 0),
    frame: str, clock_id: str,
) -> dict[str, C2RefugeReference]:
    """Stop velocity differences before stopping the fleet's shared velocity.

    All derivatives use the caller's single declared clock. For initial
    position p_i, velocity v_i, fleet mean v_c and w_i=v_i-v_c, the route is

        p_i(t) = p_i + v_c*b_Tc(t) + w_i*b_Tr(t) + d*s_Tc(t).

    Here b_T'= (1-t/T)^3 until T and zero afterwards, b_T(T)=T/4,
    and s_T is the quintic zero-velocity/acceleration unit step. Both shared
    terms cancel exactly in every same-time peer difference. After Tr the
    relative configuration is constant, while shared braking continues to Tc.
    A constant holding piece follows; every join is exactly C2.

    Duration selection is a planning choice, not a force or safety claim.
    No chaser's state, uncertainty radius or physical clearance is modified.
    """
    if not isinstance(initial_states, Mapping) or not initial_states:
        raise ValueError("A nonempty named fleet of six-state initial conditions is required")
    if any(not isinstance(key, str) or not key.strip() for key in initial_states):
        raise ValueError("Fleet members need nonempty distinct string identities")
    states = {key: _vec(value, 6, "initial fleet state") for key, value in initial_states.items()}
    q, common, relative, hold = map(_exact, (start, common_duration, relative_duration, holding_duration))
    if not 0 < relative <= common or hold <= 0:
        raise ValueError("Braking requires 0 < relative duration <= common duration and a positive hold")
    frame, clock_id = _identity(frame, "reference frame"), _identity(clock_id, "clock identity")
    displacement = _vec(outward_displacement, 3, "shared outward displacement")
    mean = tuple(sum(state[axis+3] for state in states.values()) / len(states) for axis in range(3))
    common_basis = (Fraction(0), Fraction(1), -3/(2*common), 1/common**2, -1/(4*common**3), Fraction(0))
    relative_basis = (Fraction(0), Fraction(1), -3/(2*relative), 1/relative**2, -1/(4*relative**3), Fraction(0))
    step = (Fraction(0), Fraction(0), Fraction(0), 10/common**3, -15/common**4, 6/common**5)
    result = {}
    for key, state in states.items():
        early, late, terminal = [], [], []
        for position, velocity, center, outward in zip(state[:3], state[3:], mean, displacement):
            difference = velocity-center
            shared = tuple(center*b + outward*s for b, s in zip(common_basis, step))
            first = tuple(value + difference*b for value, b in zip(shared, relative_basis))
            early.append((first[0]+position, *first[1:]))
            stopped = (shared[0]+position+difference*relative/4, *shared[1:])
            late.append(_shift(stopped, relative))
            terminal.append(position+center*common/4+difference*relative/4+outward)
        pieces = [PolynomialReferencePiece(q, q+relative, early)]
        if relative < common:
            pieces.append(PolynomialReferencePiece(q+relative, q+common, late))
        pieces.append(PolynomialReferencePiece(q+common, q+common+hold, tuple((v,) for v in terminal)))
        result[key] = C2RefugeReference(pieces, frame=frame, clock_id=clock_id)
    return result
