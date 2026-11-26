"""actuation.py: Declared command support and independent numerical actuator realizations.

Copyright: Aaron John Sabu
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
from pathlib import Path

import numpy as np
from orbexa.common.intervals import IntervalBox
from orbexa.models.dynamics.reference_frame import exact, fingerprint
from orbexa.proofs.attitude import _sqrt_upper

_ACTUATION_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class CommandActuationSupport:
    """Hold the exact command-dependent support of one applied actuator input.

    Inputs: ``command`` is the commanded input, ``error_abs`` the per-component
    absolute error bound, ``applied_input`` the box command -/+ error, and
    ``prior_fingerprint`` the fingerprint of the prior that produced them."""

    command: tuple
    error_abs: tuple
    applied_input: IntervalBox
    prior_fingerprint: str

    def as_dict(self):
        """Return the JSON-ready record of this support with its method label."""
        return {
            "command": tuple(map(str, self.command)),
            "error_abs": tuple(map(str, self.error_abs)),
            "applied_input": self.applied_input.as_dict(),
            "prior_fingerprint": self.prior_fingerprint,
            "method": "declared-gain-tilt-bias-command-support-v1",
        }

    @cached_property
    def fingerprint(self):
        """Return the SHA-256 fingerprint of ``as_dict()`` (cached)."""
        return fingerprint(self.as_dict())


@dataclass(frozen=True)
class CommandDependentActuationPrior:
    """Prospective command-dependent support; no actual matrix sets its bounds.

    M=R D gives M-I=(R-I)D+(D-I). With ||R-I||2<=theta,
    diagonal gain error <=g and bias <=b, each component error is bounded
    by theta*(1+g)*||u||2+g*abs(u_i)+b, plus declared rounding reserves.
    """

    gain_error: object = ".001"
    tilt_rad: object = ".001"
    bias_abs: object = ".2"
    round_relative: object = "1e-10"
    round_absolute: object = "1e-9"
    command_abs_limit: object = 100000
    declaration: str = "Prospective bounded synthetic diagonal gain/axis tilt/bias study prior; no device qualification"

    def __post_init__(self):
        """Convert the bounds to exact values and validate them.

        Outputs: replaces each numeric field with its exact value. Raises ValueError
        when a bound is negative, when ``gain_error`` is at least 1, when
        ``command_abs_limit`` is not positive, or when ``declaration`` is blank."""
        for name in (
            "gain_error",
            "tilt_rad",
            "bias_abs",
            "round_relative",
            "round_absolute",
            "command_abs_limit",
        ):
            value = exact(getattr(self, name))
            if value < 0:
                raise ValueError("Actuation prior bounds must be nonnegative")
            object.__setattr__(self, name, value)
        if (
            self.gain_error >= 1
            or self.command_abs_limit <= 0
            or not isinstance(self.declaration, str)
            or not self.declaration.strip()
        ):
            raise ValueError("Invalid declared gain/tilt/bias command prior")

    def as_dict(self):
        """Return the JSON-ready record of the prior.

        The record holds the exact bounds as strings, the declaration, the SHA-256 of
        this source file, the units note and the error law."""
        return dict(
            model="declared-gain-tilt-bias-command-support-v1",
            source_sha256=_ACTUATION_SOURCE_SHA256,
            **{
                name: str(getattr(self, name))
                for name in (
                    "gain_error",
                    "tilt_rad",
                    "bias_abs",
                    "round_relative",
                    "round_absolute",
                    "command_abs_limit",
                )
            },
            declaration=self.declaration,
            units="u and bias in m/rad^2; dimensionless gain and tilt in rad",
            law="eta_i=theta*(1+g)*sqrt_upper(sum(u_j^2))+g*abs(u_i)+b+round_relative*norm_upper+round_absolute",
        )

    @cached_property
    def fingerprint(self):
        """Return the SHA-256 fingerprint of ``as_dict()`` (cached)."""
        return fingerprint(self.as_dict())

    def error_for_component_limits(self, limits):
        """Return the per-component error bounds for command magnitude limits.

        Inputs: ``limits`` are three nonnegative bounds on the absolute command
        components, in command units.
        Outputs: a tuple of three exact bounds, each
        ``(tilt*(1+g)+round_relative)*sqrt_upper(sum(limit^2)) + bias + round_absolute
        + g*limit_i``. Raises ValueError when there are not three limits or one is
        negative or above ``command_abs_limit``."""
        values = tuple(map(exact, limits))
        if len(values) != 3 or any(
            value < 0 or value > self.command_abs_limit for value in values
        ):
            raise ValueError("Command escapes the declared actuation-prior cap")
        norm = _sqrt_upper(sum(value * value for value in values))
        common = (
            (self.tilt_rad * (1 + self.gain_error) + self.round_relative) * norm
            + self.bias_abs
            + self.round_absolute
        )
        return tuple(common + self.gain_error * value for value in values)

    def support(self, command):
        """Return the command support for a commanded input.

        Inputs: ``command`` is three command components.
        Outputs: a CommandActuationSupport whose error bounds come from
        ``error_for_component_limits`` on the absolute components. Raises ValueError
        when a component magnitude exceeds ``command_abs_limit``."""
        values = tuple(map(exact, command))
        errors = self.error_for_component_limits(tuple(abs(value) for value in values))
        return CommandActuationSupport(
            values,
            errors,
            IntervalBox(
                tuple(v - e for v, e in zip(values, errors)),
                tuple(v + e for v, e in zip(values, errors)),
            ),
            self.fingerprint,
        )

    def validate_realization(self, realization):
        """Exact independent support check, including binary64 dot rounding.

        Dproxy clips diag(M) into [1-g,1+g]. This does not estimate the
        prior: each row of M-Dproxy must fit the prospectively fixed slope.
        The test covers a superset of proper R*D realizations and avoids
        treating a rounded scipy rotation as exactly orthogonal.
        """
        if type(realization) is not AffineActuatorRealization:
            raise TypeError(
                "A command-dependent prior requires the supported exact affine realization"
            )
        gamma = Fraction(16, 2**53) / (1 - Fraction(16, 2**53))
        tiny = Fraction(16, 2**1074) * (1 + 3 * self.command_abs_limit)
        diagonal = tuple(
            min(1 + self.gain_error, max(1 - self.gain_error, realization.matrix[i][i]))
            for i in range(3)
        )
        allowed_slope = self.tilt_rad * (1 + self.gain_error) + self.round_relative
        rows = []
        for i, row in enumerate(realization.matrix):
            residual = tuple(
                value - (diagonal[i] if i == j else 0) for j, value in enumerate(row)
            )
            residual_norm = _sqrt_upper(sum(value * value for value in residual))
            matrix_norm = _sqrt_upper(sum(value * value for value in row))
            required_slope = residual_norm + gamma * matrix_norm
            required_constant = (1 + gamma) * abs(realization.bias[i]) + tiny
            if (
                abs(realization.bias[i]) > self.bias_abs
                or required_slope > allowed_slope
                or required_constant > self.bias_abs + self.round_absolute
            ):
                raise ValueError(
                    "Numerical actuator realization violates the declared command-dependent prior"
                )
            rows.append(
                {
                    "diagonal_proxy": str(diagonal[i]),
                    "residual_norm_upper": str(residual_norm),
                    "binary64_rounding_slope_upper": str(gamma * matrix_norm),
                    "slope_slack": str(allowed_slope - required_slope),
                    "constant_slack": str(
                        self.bias_abs + self.round_absolute - required_constant
                    ),
                }
            )
        record = {
            "status": "EXACT_DECLARED_SUPPORT_CHECK_PASSED",
            "prior_fingerprint": self.fingerprint,
            "realization_fingerprint": realization.fingerprint,
            "rows": rows,
            "realization_matrix": tuple(
                tuple(map(str, row)) for row in realization.matrix
            ),
            "realization_bias": tuple(map(str, realization.bias)),
            "realization_declaration": realization.declaration,
            "method": "exact-diagonal-residual-row-norm-plus-gamma16-binary64-v1",
            "scope": "Independent realization containment check; controller bounds remain the declared prior",
        }
        record["identity"] = fingerprint(record)
        return record


@dataclass(frozen=True)
class AffineActuatorRealization:
    """Hold a fixed three-dimensional affine actuator map ``matrix @ command + bias``.

    Inputs: ``matrix`` is 3x3, ``bias`` has three entries and ``declaration`` is a
    nonempty description. Entries are stored as exact values."""

    matrix: tuple = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    bias: tuple = (0, 0, 0)
    declaration: str = "Explicit fixed study gain/axis/bias realization"

    def __post_init__(self):
        """Convert ``matrix`` and ``bias`` to exact values and validate their shape.

        Outputs: replaces ``matrix`` and ``bias`` with exact tuples. Raises ValueError
        when the matrix is not 3x3, the bias does not have three entries, or
        ``declaration`` is empty."""
        matrix = tuple(tuple(map(exact, row)) for row in self.matrix)
        bias = tuple(map(exact, self.bias))
        if (
            len(matrix) != 3
            or any(len(row) != 3 for row in matrix)
            or len(bias) != 3
            or not self.declaration
        ):
            raise ValueError(
                "Actuator realization needs finite three-dimensional affine declarations"
            )
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "bias", bias)

    @cached_property
    def fingerprint(self):
        """Return the SHA-256 fingerprint of matrix, bias and declaration (cached)."""
        return fingerprint(
            {
                "matrix": tuple(tuple(map(str, row)) for row in self.matrix),
                "bias": tuple(map(str, self.bias)),
                "declaration": self.declaration,
            }
        )

    def apply(self, agent, command, q):
        """Return ``matrix @ command + bias`` as a float array of three entries.

        Inputs: ``command`` is three command components. ``agent`` and ``q`` are not
        used. The entries are converted to float before the sums are formed."""
        return np.array(
            [
                sum(float(a) * float(u) for a, u in zip(row, command)) + float(b)
                for row, b in zip(self.matrix, self.bias)
            ]
        )
