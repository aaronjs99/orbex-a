"""Bounded-memory, independent-output RBF regression for planning proposals.

Posterior covariance is a statistical quantity, not an authenticated physical
disturbance set. This module never contracts SMID beliefs or tube reserves.
Derived from the MIT Swarm CLF-CBF Playground; see THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular


@dataclass(frozen=True)
class ResidualPrediction:
    mean: np.ndarray
    covariance: np.ndarray
    sample_count: int
    certified_support: bool = False

    def projected_radius(self, direction, *, beta=2.):
        """beta*sqrt(a.T Sigma a), in the projected quantity's units."""
        a=np.asarray(direction,dtype=float)
        if a.shape!=self.mean.shape or not np.all(np.isfinite(a)) or not np.isfinite(beta) or beta<0:
            raise ValueError('Use a finite projection vector and nonnegative statistical multiplier')
        return float(beta*np.sqrt(max(0.,a @ self.covariance @ a)))


class ResidualGaussianProcess:
    """Explicit feature scaling and per-output covariance; no hidden GP refits.

    fit() publishes a new complete posterior. add() alone leaves the preceding
    posterior intact. An untrained predictor returns its nonzero prior variance.
    Fixed hyperparameters avoid interpreting optimizer fits as confidence proofs.
    """
    def __init__(self, feature_dimension, output_dimension=3, *, capacity=256,
                 length_scale=1., amplitude=1., noise=.01):
        if any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in
               (feature_dimension,output_dimension,capacity)):
            raise ValueError('GP dimensions and capacity must be positive integers')
        self.feature_dimension,self.output_dimension=feature_dimension,output_dimension
        def parameter(value, size, allow_zero=False):
            a=np.broadcast_to(np.asarray(value,dtype=float),(size,)).copy()
            if not np.all(np.isfinite(a)) or np.any(a<0 if allow_zero else a<=0):
                raise ValueError('GP scales must be finite and positive; observation noise may be zero')
            a.setflags(write=False);return a
        self.length_scale=parameter(length_scale,feature_dimension)
        self.amplitude=parameter(amplitude,output_dimension)
        self.noise=parameter(noise,output_dimension,True)
        self.samples=deque(maxlen=capacity)
        self._posterior=None

    def add(self, features, residual):
        x,y=np.asarray(features,dtype=float),np.asarray(residual,dtype=float)
        if (x.shape!=(self.feature_dimension,) or y.shape!=(self.output_dimension,)
                or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y))):
            raise ValueError('GP sample has invalid features or residual outputs')
        self.samples.append((x.copy(),y.copy()))

    def fit(self):
        if not self.samples:
            raise ValueError('A GP fit requires at least one sample')
        snapshot=tuple(self.samples)
        x=np.stack([s[0] for s in snapshot]);y=np.stack([s[1] for s in snapshot])
        d=(x[:,None,:]-x[None,:,:])/self.length_scale
        base=np.exp(-.5*np.sum(d*d,axis=2));factors=[];coefficients=[]
        for j in range(self.output_dimension):
            kernel=self.amplitude[j]**2*base
            kernel+=np.eye(len(x))*(self.noise[j]**2+1.e-10*self.amplitude[j]**2)
            factor=cho_factor(kernel,lower=True,check_finite=True)
            factors.append(factor);coefficients.append(cho_solve(factor,y[:,j]))
        self._posterior=(x,tuple(factors),tuple(coefficients))
        return len(x)

    def predict(self, features):
        x=np.asarray(features,dtype=float)
        if x.shape!=(self.feature_dimension,) or not np.all(np.isfinite(x)):
            raise ValueError('GP query has invalid feature shape or values')
        snapshot=self._posterior
        if snapshot is None:
            return ResidualPrediction(np.zeros(self.output_dimension),np.diag(self.amplitude**2),0)
        samples,factors,coefficients=snapshot
        base=np.exp(-.5*np.sum(((samples-x)/self.length_scale)**2,axis=1))
        mean=np.empty(self.output_dimension);variance=np.empty_like(mean)
        for j in range(self.output_dimension):
            kernel=self.amplitude[j]**2*base
            mean[j]=kernel @ coefficients[j]
            projected=solve_triangular(factors[j][0],kernel,lower=True)
            variance[j]=max(0.,self.amplitude[j]**2-projected @ projected)
        return ResidualPrediction(mean,np.diag(variance),len(samples))
