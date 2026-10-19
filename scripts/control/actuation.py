"""Declared command support and independent numerical actuator realizations."""
from __future__ import annotations

import hashlib
from pathlib import Path
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
import numpy as np

from orbexa.core.reference_frame import exact, fingerprint
from orbexa.control.reachability import IntervalBox
from orbexa.control.attitude_flow import _sqrt_upper

_ACTUATION_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

@dataclass(frozen=True)
class CommandActuationSupport:
    command: tuple
    error_abs: tuple
    applied_input: IntervalBox
    prior_fingerprint: str

    def as_dict(self):
        return dict(command=tuple(map(str,self.command)),error_abs=tuple(map(str,self.error_abs)),
            applied_input=self.applied_input.as_dict(),prior_fingerprint=self.prior_fingerprint,
            method='declared-gain-tilt-bias-command-support-v1')

    @cached_property
    def fingerprint(self):return fingerprint(self.as_dict())


@dataclass(frozen=True)
class CommandDependentActuationPrior:
    """Prospective command-dependent support; no actual matrix sets its bounds.

    M=R D gives M-I=(R-I)D+(D-I). With ||R-I||2<=theta,
    diagonal gain error <=g and bias <=b, each component error is bounded
    by theta*(1+g)*||u||2+g*abs(u_i)+b, plus declared rounding reserves.
    """
    gain_error: object = '.001'
    tilt_rad: object = '.001'
    bias_abs: object = '.2'
    round_relative: object = '1e-10'
    round_absolute: object = '1e-9'
    command_abs_limit: object = 100000
    declaration: str = 'Prospective bounded synthetic diagonal gain/axis tilt/bias study prior; no device qualification'

    def __post_init__(self):
        for name in ('gain_error','tilt_rad','bias_abs','round_relative','round_absolute','command_abs_limit'):
            value=exact(getattr(self,name))
            if value<0:raise ValueError('Actuation prior bounds must be nonnegative')
            object.__setattr__(self,name,value)
        if self.gain_error>=1 or self.command_abs_limit<=0 or not isinstance(self.declaration,str) or not self.declaration.strip():
            raise ValueError('Invalid declared gain/tilt/bias command prior')

    def as_dict(self):
        return dict(model='declared-gain-tilt-bias-command-support-v1',source_sha256=_ACTUATION_SOURCE_SHA256,
            **{name:str(getattr(self,name)) for name in ('gain_error','tilt_rad','bias_abs','round_relative','round_absolute','command_abs_limit')},
            declaration=self.declaration,units='u and bias in m/rad^2; dimensionless gain and tilt in rad',
            law='eta_i=theta*(1+g)*sqrt_upper(sum(u_j^2))+g*abs(u_i)+b+round_relative*norm_upper+round_absolute')

    @cached_property
    def fingerprint(self):return fingerprint(self.as_dict())

    def error_for_component_limits(self,limits):
        values=tuple(map(exact,limits))
        if len(values)!=3 or any(value<0 or value>self.command_abs_limit for value in values):
            raise ValueError('Command escapes the declared actuation-prior cap')
        norm=_sqrt_upper(sum(value*value for value in values))
        common=(self.tilt_rad*(1+self.gain_error)+self.round_relative)*norm+self.bias_abs+self.round_absolute
        return tuple(common+self.gain_error*value for value in values)

    def support(self,command):
        values=tuple(map(exact,command))
        errors=self.error_for_component_limits(tuple(abs(value) for value in values))
        return CommandActuationSupport(values,errors,IntervalBox(tuple(v-e for v,e in zip(values,errors)),
            tuple(v+e for v,e in zip(values,errors))),self.fingerprint)

    def validate_realization(self,realization):
        """Exact independent support check, including binary64 dot rounding.

        Dproxy clips diag(M) into [1-g,1+g]. This does not estimate the
        prior: each row of M-Dproxy must fit the prospectively fixed slope.
        The test covers a superset of proper R*D realizations and avoids
        treating a rounded scipy rotation as exactly orthogonal.
        """
        if type(realization) is not AffineActuatorRealization:
            raise TypeError('A command-dependent prior requires the supported exact affine realization')
        gamma=Fraction(16,2**53)/(1-Fraction(16,2**53))
        tiny=Fraction(16,2**1074)*(1+3*self.command_abs_limit)
        diagonal=tuple(min(1+self.gain_error,max(1-self.gain_error,realization.matrix[i][i])) for i in range(3))
        allowed_slope=self.tilt_rad*(1+self.gain_error)+self.round_relative
        rows=[]
        for i,row in enumerate(realization.matrix):
            residual=tuple(value-(diagonal[i] if i==j else 0) for j,value in enumerate(row))
            residual_norm=_sqrt_upper(sum(value*value for value in residual))
            matrix_norm=_sqrt_upper(sum(value*value for value in row))
            required_slope=residual_norm+gamma*matrix_norm
            required_constant=(1+gamma)*abs(realization.bias[i])+tiny
            if (abs(realization.bias[i])>self.bias_abs or required_slope>allowed_slope
                    or required_constant>self.bias_abs+self.round_absolute):
                raise ValueError('Numerical actuator realization violates the declared command-dependent prior')
            rows.append(dict(diagonal_proxy=str(diagonal[i]),residual_norm_upper=str(residual_norm),
                binary64_rounding_slope_upper=str(gamma*matrix_norm),slope_slack=str(allowed_slope-required_slope),
                constant_slack=str(self.bias_abs+self.round_absolute-required_constant)))
        record=dict(status='EXACT_DECLARED_SUPPORT_CHECK_PASSED',prior_fingerprint=self.fingerprint,
            realization_fingerprint=realization.fingerprint,rows=rows,
            realization_matrix=tuple(tuple(map(str,row)) for row in realization.matrix),
            realization_bias=tuple(map(str,realization.bias)),realization_declaration=realization.declaration,
            method='exact-diagonal-residual-row-norm-plus-gamma16-binary64-v1',
            scope='Independent realization containment check; controller bounds remain the declared prior')
        record['identity']=fingerprint(record)
        return record


@dataclass(frozen=True)
class AffineActuatorRealization:
    matrix: tuple = ((1,0,0),(0,1,0),(0,0,1))
    bias: tuple = (0,0,0)
    declaration: str = 'Explicit fixed study gain/axis/bias realization'

    def __post_init__(self):
        matrix = tuple(tuple(map(exact,row)) for row in self.matrix)
        bias = tuple(map(exact,self.bias))
        if len(matrix)!=3 or any(len(row)!=3 for row in matrix) or len(bias)!=3 or not self.declaration:
            raise ValueError('Actuator realization needs finite three-dimensional affine declarations')
        object.__setattr__(self,'matrix',matrix);object.__setattr__(self,'bias',bias)

    @cached_property
    def fingerprint(self):
        return fingerprint(dict(matrix=tuple(tuple(map(str,row)) for row in self.matrix),
                                bias=tuple(map(str,self.bias)),declaration=self.declaration))

    def apply(self,agent,command,q):
        return np.array([sum(float(a)*float(u) for a,u in zip(row,command))+float(b)
                         for row,b in zip(self.matrix,self.bias)])
