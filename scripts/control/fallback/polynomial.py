"""Exact shared-clock polynomial references and independent peer bounds."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
from math import comb
from typing import Any
from orbexa.control.certification.reachability import IntervalBox, _exact
from orbexa.control.fallback.refuge import _identity, _scope, fingerprint


POLYNOMIAL_PEER_METHOD = "exact-same-time-polynomial-reference-peer-v1"


def _polynomial_shift(coefficients, offset):
    if offset==0:return coefficients
    return tuple(sum(coefficients[i]*comb(i,j)*offset**(i-j)
                     for i in range(j,len(coefficients))) for j in range(len(coefficients)))


def _polynomial_range(coefficients, lower, upper):
    """Exact Bernstein convex hull, after retaining the common time variable."""
    # Shared fleet motion can cancel every nonconstant coefficient exactly.
    # Its interval is then a point; no basis shift or degree elevation remains.
    if coefficients and not any(coefficients[1:]):
        return coefficients[0], coefficients[0]
    degree=len(coefficients)-1
    shifted=coefficients if lower==0 else _polynomial_shift(coefficients,lower)
    power=tuple(value*(upper-lower)**j for j,value in enumerate(shifted))
    values=tuple(sum(power[j]*Fraction(comb(k,j),comb(degree,j)) for j in range(k+1))
                 for k in range(degree+1))
    return min(values),max(values)


@dataclass(frozen=True)
class PolynomialReferenceSpan:
    start: Any
    end: Any
    coefficients: Any

    def __post_init__(self):
        start,end=_scope(self.start,self.end)
        rows=tuple(tuple(map(_exact,row)) for row in self.coefficients)
        if len(rows)!=3 or any(not row or len(row)>6 for row in rows):
            raise ValueError("Three exact polynomials of degree at most five are required")
        object.__setattr__(self,"start",start);object.__setattr__(self,"end",end)
        object.__setattr__(self,"coefficients",rows)

    def jet(self,q,order):
        value=_exact(q)-self.start
        result=[]
        for row in self.coefficients:
            derivative=row
            for _ in range(order):
                derivative=tuple(i*c for i,c in enumerate(derivative) if i) or (Fraction(0),)
            shifted=_polynomial_shift(derivative,value)
            result.append(shifted[0])
        return tuple(result)


@dataclass(frozen=True)
class PolynomialReferenceSource:
    """Full immutable analytic source for one existing C2 reference provider.

    The coefficient digest must also match the controller's original analytic
    reference-bound proof. This source never asserts a relation between errors.
    """
    pieces: Any
    reference_id: str
    frame: str
    clock_id: str

    def __post_init__(self):
        pieces=tuple(self.pieces)
        if not pieces or any(type(piece) is not PolynomialReferenceSpan for piece in pieces):
            raise TypeError("Complete typed exact polynomial reference pieces are required")
        for old,new in zip(pieces,pieces[1:]):
            if old.end!=new.start or any(old.jet(old.end,k)!=new.jet(new.start,k) for k in range(3)):
                raise ValueError("Polynomial peer sources must cover one gap-free C2 reference")
        object.__setattr__(self,"pieces",pieces)
        for name in ("reference_id","frame","clock_id"):_identity(getattr(self,name),name)

    @cached_property
    def coefficient_fingerprint(self):return fingerprint(self.pieces)

    @cached_property
    def fingerprint(self):return fingerprint(self)

    @property
    def proof_reference(self):return "exact-quintic-Bernstein:"+self.coefficient_fingerprint

    def difference_bounds(self,other,start,end):
        start,end=_scope(start,end)
        if (type(other) is not PolynomialReferenceSource or self.frame!=other.frame
                or self.clock_id!=other.clock_id or any(source.pieces[0].start>start
                or source.pieces[-1].end<end for source in (self,other))):
            raise ValueError("Same-frame/clock polynomial sources must cover the entire peer window")
        cuts=sorted({start,end,*(v for source in (self,other) for piece in source.pieces
                               for v in (piece.start,piece.end) if start<v<end)})
        result=None
        for lower,upper in zip(cuts,cuts[1:]):
            left=next((piece for piece in self.pieces if piece.start<=lower<upper<=piece.end),None)
            right=next((piece for piece in other.pieces if piece.start<=lower<upper<=piece.end),None)
            if left is None or right is None:raise ValueError("Incomplete polynomial peer source window")
            ranges=[]
            for first,second in zip(left.coefficients,right.coefficients):
                if left.start==right.start:
                    # Cancel shared translation in the original exact basis
                    # before shifting once. The resulting Bernstein numbers
                    # are identical to shifting both polynomials separately.
                    a,b=first,second
                    local_lower,local_upper=lower-left.start,upper-left.start
                else:
                    a,b=_polynomial_shift(first,lower-left.start),_polynomial_shift(second,lower-right.start)
                    local_lower,local_upper=Fraction(0),upper-lower
                coefficients=tuple((a[i] if i<len(a) else 0)-(b[i] if i<len(b) else 0)
                                   for i in range(max(len(a),len(b))))
                ranges.append(_polynomial_range(coefficients,local_lower,local_upper))
            box=IntervalBox(tuple(v[0] for v in ranges),tuple(v[1] for v in ranges))
            result=box if result is None else result.hull(box)
        return result


@dataclass(frozen=True)
class PolynomialPeerBounds:
    first_id: str
    second_id: str
    start: Fraction
    end: Fraction
    first_controller_id: str
    second_controller_id: str
    first_reference_id: str
    second_reference_id: str
    first_source_id: str
    second_source_id: str
    first_omega: tuple
    second_omega: tuple
    reference_difference: IntervalBox
    relative_position: IntervalBox
    frame: str
    clock_id: str
    method: str = POLYNOMIAL_PEER_METHOD

    def __post_init__(self):
        start,end=_scope(self.start,self.end)
        object.__setattr__(self,"start",start);object.__setattr__(self,"end",end)
        for name in ("first_id","second_id","first_controller_id","second_controller_id",
                     "first_reference_id","second_reference_id","first_source_id","second_source_id","frame","clock_id"):
            _identity(getattr(self,name),name)
        first,second=tuple(map(_exact,self.first_omega)),tuple(map(_exact,self.second_omega))
        if (self.first_id==self.second_id or len(first)!=3 or len(second)!=3
                or any(value<0 for value in first+second)
                or any(not isinstance(box,IntervalBox) or box.dimension!=3
                       for box in (self.reference_difference,self.relative_position))
                or self.method!=POLYNOMIAL_PEER_METHOD):
            raise ValueError("Complete exact independent peer error/reference bounds are required")
        radius=tuple(a+b for a,b in zip(first,second))
        expected=IntervalBox(tuple(v-r for v,r in zip(self.reference_difference.lower,radius)),
                             tuple(v+r for v,r in zip(self.reference_difference.upper,radius)))
        if self.relative_position!=expected:raise ValueError("Peer error expansion differs from the independent Omega sum")
        object.__setattr__(self,"first_omega",first);object.__setattr__(self,"second_omega",second)


def _reference_methods(provider):
    return tuple(getattr(value,"__func__",value) for value in
                 (provider.sample,provider.bounds,provider.polynomial_source))
