"""Small ball-arithmetic operations shared by rigid contact/load enclosures."""
from __future__ import annotations

from orbexa.control.certification.reachability import _ProofUnresolved


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def add(a, b):
    return tuple(x+y for x, y in zip(a, b))


def subtract(a, b):
    return tuple(x-y for x, y in zip(a, b))


def scale(s, a):
    return tuple(s*x for x in a)


def matvec(a, b):
    return tuple(sum(x*y for x, y in zip(row, b)) for row in a)


def transpose(matrix):
    return tuple(zip(*matrix))


def inverse3(a):
    """Adjugate inclusion; cycling both retained indices gives cofactor signs."""
    cofactor = tuple(tuple(a[(i+1)%3][(j+1)%3]*a[(i+2)%3][(j+2)%3]
        -a[(i+1)%3][(j+2)%3]*a[(i+2)%3][(j+1)%3]
        for j in range(3)) for i in range(3))
    determinant = sum(a[0][j]*cofactor[0][j] for j in range(3))
    if not determinant > 0:
        raise _ProofUnresolved('rigid inertia determinant is not proved positive')
    return tuple(tuple(cofactor[j][i]/determinant for j in range(3)) for i in range(3))


def square(value, ar):
    box = ar.box((value,))
    lo, hi = box.lower[0], box.upper[0]
    return ar.interval((0 if lo <= 0 <= hi else min(lo*lo, hi*hi), max(lo*lo, hi*hi)))


def norm(vector, ar):
    # A ball hull of [0,u] can have a tiny negative lower endpoint because
    # its radius is rounded. Take square roots of exact nonnegative rational
    # endpoints separately rather than asking sqrt of that enclosing hull.
    box = ar.box(vector)
    lower = sum(0 if lo <= 0 <= hi else min(lo*lo, hi*hi) for lo, hi in zip(box.lower, box.upper))
    upper = sum(max(lo*lo, hi*hi) for lo, hi in zip(box.lower, box.upper))
    low_root = ar.box((ar.scalar(lower).sqrt(),)).lower[0]
    high_root = ar.box((ar.scalar(upper).sqrt(),)).upper[0]
    return ar.interval((low_root, high_root))


def norm_upper(vector, ar):
    return ar.box((norm(vector, ar),)).upper[0]


def quaternion_product(a, b):
    vector = add(add(scale(a[3], b[:3]), scale(b[3], a[:3])), cross(a[:3], b[:3]))
    return (*vector, a[3]*b[3]-sum(x*y for x, y in zip(a[:3], b[:3])))


def quaternion_conjugate(q):
    return (-q[0], -q[1], -q[2], q[3])


def normalized_quaternion(q, ar):
    length = norm(q, ar)
    if not length > 0:
        raise _ProofUnresolved('quaternion set includes zero')
    return tuple(value/length for value in q)


def quaternion_rotation(q):
    """Polynomial matrix on unit quaternions, without an interval normalization."""
    x, y, z, w = q
    return ((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
            (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
            (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)))


def rotation_vector(q, ar):
    """Principal log inclusion away from the pi cut, continuous at identity.

    For a unit q with w>0, atan(x)/x is between 1/(1+x^2)
    and 1. Consequently its rotation-vector multiplier lies in [2w,2/w].
    The bound avoids dividing by the vector norm at identity.
    """
    scalar = ar.box((q[3],))
    if scalar.upper[0] < 0:
        q = tuple(-value for value in q)
        scalar = ar.box((q[3],))
    if scalar.lower[0] <= 0:
        raise _ProofUnresolved('relative contact orientation meets the principal-log cut')
    multiplier = ar.interval((2*scalar.lower[0], 2/scalar.lower[0]))
    return scale(multiplier, q[:3])
