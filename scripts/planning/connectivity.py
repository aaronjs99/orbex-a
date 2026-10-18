"""Smooth range-weighted graph diagnostics and local spectral derivatives.

This is a planning preference. Radio/line-of-sight availability and connectivity
invariance require additional models. An unweighted k-NN graph does not have the
distance gradient used here. MIT source attribution: THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ConnectivityResult:
    adjacency: np.ndarray
    algebraic_connectivity: float
    gradient: np.ndarray | None
    eigenvalue_gap: float | None
    status: str


@dataclass(frozen=True)
class SpectralConnectivity:
    range_m: float
    eigenvalue_tolerance: float = 1.e-9

    def __post_init__(self):
        if any(not np.isfinite(v) or v<=0 for v in (self.range_m,self.eigenvalue_tolerance)):
            raise ValueError('Communication range and eigenvalue tolerance must be positive')

    def evaluate(self, positions, *, available_links=None):
        p=np.asarray(positions,dtype=float)
        if p.ndim!=2 or p.shape[1]!=3 or len(p)<1 or not np.all(np.isfinite(p)):
            raise ValueError('Use a finite nonempty n-by-3 position array')
        n=len(p);mask=np.ones((n,n),dtype=bool) if available_links is None else np.asarray(available_links)
        if mask.shape!=(n,n) or mask.dtype!=np.bool_ or not np.array_equal(mask,mask.T):
            raise ValueError('Available links must be an explicit symmetric Boolean matrix')
        delta=p[:,None,:]-p[None,:,:];squared=np.sum(delta**2,axis=2)/self.range_m**2
        weight=np.where((squared<1)&mask,(1-squared)**2,0.);np.fill_diagonal(weight,0.)
        if n==1:
            return ConnectivityResult(weight,0.,np.zeros_like(p),None,'SINGLE_AGENT')
        laplacian=np.diag(weight.sum(axis=1))-weight
        values,vectors=np.linalg.eigh(laplacian);value=max(0.,float(values[1]))
        gap=float(min(values[1]-values[0],values[2]-values[1] if n>2 else np.inf))
        if gap<=self.eigenvalue_tolerance:
            return ConnectivityResult(weight,value,None,gap,'REPEATED_OR_DISCONNECTED_EIGENVALUE')
        fiedler=vectors[:,1];gradient=np.zeros_like(p)
        for i in range(n):
            for j in range(n):
                if i!=j and squared[i,j]<1 and mask[i,j]:
                    dw=-4*(1-squared[i,j])*delta[i,j]/self.range_m**2
                    gradient[i]+=(fiedler[i]-fiedler[j])**2*dw
        return ConnectivityResult(weight,value,gradient,gap,'SIMPLE_EIGENVALUE_LOCAL_DERIVATIVE')
