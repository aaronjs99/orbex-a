"""Full-inertia componentwise terminal bounds for midpoint feedforward.

This is a distinct implemented held policy, not a reinterpretation of an old
start-feedforward certificate. Complete-component geometry and the original
orbital/error contracts are composed with exact expanded rigid-body port jets.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from orbexa.control.terminal.composite import COMPOSITE_TERMINAL_METHOD, CompositePhysicalTerminalBoundsBuilder
from orbexa.control.terminal.physical import _abs, _add, _dot
from orbexa.control.certification.reachability import IntervalBox
from orbexa.control.terminal.body_jets import bound_componentwise_terminal_jets
from orbexa.control.terminal.tracking import TerminalFeedforwardPolicy


COMPONENTWISE_MIDPOINT_TERMINAL_METHOD = "arb-componentwise-composite-midpoint-terminal-v1"


@dataclass(frozen=True)
class ComponentwiseCompositePhysicalTerminalBoundsBuilder(CompositePhysicalTerminalBoundsBuilder):
    """Fixed-body-port global bounds and the explicit midpoint held law.

    Every inherited limit, physical prior, error declaration and finite clock
    remains authoritative. The componentwise first-exit comparison may refuse;
    it is never replaced by a sampled spin/derivative maximum.
    """

    held_policy: str = TerminalFeedforwardPolicy.MIDPOINT_V1.value

    def __post_init__(self):
        super().__post_init__()
        policy = TerminalFeedforwardPolicy(self.held_policy)
        if policy is not TerminalFeedforwardPolicy.MIDPOINT_V1:
            raise ValueError("The componentwise midpoint builder requires its explicit midpoint held policy")
        object.__setattr__(self, "held_policy", policy.value)

    def _terminal_policy(self):
        return TerminalFeedforwardPolicy.MIDPOINT_V1

    def _terminal_proof_reference(self):
        return COMPONENTWISE_MIDPOINT_TERMINAL_METHOD

    def _reference_jet_bounds(self, flint, ar, *, point, point_radius, rotation_derivatives,
                              duration, rate, rate_prime, rate_double, torque_factor, radius_log_prime):
        jets = bound_componentwise_terminal_jets(inertia=self.inertia, entry_spin=self.entry_spin,
            body_port=point, physical_duration_upper=duration, anomaly_rate=rate,
            anomaly_rate_prime_abs=rate_prime, anomaly_rate_double_prime_abs=rate_double,
            torque_factor_upper=torque_factor, radius_log_derivative_abs=radius_log_prime,
            precision_bits=self.precision_bits)
        scalar = tuple(point_radius * value for value in rotation_derivatives)
        return jets.reference_norm_bounds, dict(componentwise_euler_port_evidence=jets.evidence,
            superseded_scalar_reference_bounds=dict(reference_velocity_norm_upper=scalar[0],
                reference_acceleration_norm_upper=scalar[1], reference_jerk_norm_upper=scalar[2]),
            held_policy=self.held_policy, geometry_proof_reference=COMPOSITE_TERMINAL_METHOD,
            two_clock_evaluation_error_terms_explicit=True)

    def _feedforward_range(self, ar, nominal, p2, reference_state, matrix_abs):
        magnitude = _add((p2,) * 3, _dot(matrix_abs, reference_state),
                         tuple(_abs(ar, value[0]) for value in nominal["affine"]))
        return IntervalBox(tuple(-value for value in magnitude), magnitude)

    def _evaluation_input_error(self, *, acceleration_error, affine_error, matrix_error,
                                reference_state, error_state, reference_error, matrix_abs):
        # Midpoint feedforward and start feedback use separate evaluated
        # matrices. Their matrix errors do not have same-clock cancellation.
        return _add(acceleration_error, affine_error,
            _dot(matrix_error, _add(reference_state, reference_error)),
            _dot(matrix_error, _add(error_state, self.errors.sensing_error_abs, reference_error)),
            _dot(matrix_abs, reference_error))

    def build(self):
        return replace(super().build(), method=COMPONENTWISE_MIDPOINT_TERMINAL_METHOD)
