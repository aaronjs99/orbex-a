"""Source-bound target ephemeris inference and delayed pose propagation."""
from __future__ import annotations

import numpy as np

from orbexa.core.dynamics.reference_frame import exact, bounds
from orbexa.control.certification.reachability import IntervalBox


def infer_target_ephemeris(gnss_packet, relative_box, clock, precision_bits=128):
    """Infer target state from a chaser receiver and relative navigation only.

    The caller must respect delivery time before using this initial prior.
    No cooperative target receiver or truth ephemeris is assumed.
    """
    if (gnss_packet.measurement_kind != 'chaser_gnss' or
            gnss_packet.frame_id != 'inertial-cartesian-si-v1' or len(gnss_packet.value) != 6):
        raise ValueError('A chaser inertial GNSS packet is required')
    from orbexa.control.certification.nonlinear import _Arithmetic
    from orbexa.control.certification.reachability import _ARB_LOCK
    from orbexa.core.dynamics.relative import matvec, z
    import flint
    with _ARB_LOCK, flint.ctx.workprec(precision_bits):
        ar = _Arithmetic(flint)
        support = gnss_packet.deterministic_error_box
        absolute = ar.vector(IntervalBox(tuple(v-hi for v, hi in zip(gnss_packet.value, support.upper)),
                                        tuple(v-lo for v, lo in zip(gnss_packet.value, support.lower))))
        relative = ar.vector(relative_box)
        rotation, rate, _ = clock.evaluate(ar.interval(gnss_packet.acquired_q), ar)
        delta_r = matvec(rotation, relative[:3])
        delta_v = matvec(rotation, tuple(rate*(a+b) for a,b in zip(relative[3:], z(relative[:3]))))
        return ar.box(tuple(a-b for a,b in zip(absolute[:3], delta_r))+
                      tuple(a-b for a,b in zip(absolute[3:], delta_v)))

def propagate_pose_estimate(rotation_reference,operator_error,body_rate_box,q_start,q_end,
                            time_start,time_end,clock,angular_acceleration_bound,precision_bits=128):
    """Propagate an observed pose center, with rigorous group-error padding.

    The center uses constant midpoint body rate. The deterministic error adds
    rate uncertainty, bounded angular acceleration, timing and an Arb bound on
    all floating matrix/rotation evaluation error. This prevents latency from
    becoming an unmodeled pose lag while retaining the full error support.
    """
    from scipy.spatial.transform import Rotation
    from orbexa.control.certification.nonlinear import _Arithmetic
    from orbexa.control.certification.attitude import _sqrt_upper
    from orbexa.control.certification.reachability import _ARB_LOCK
    import flint
    q0,q1 = bounds(q_start),bounds(q_end)
    delta_lo = max(exact(0),time_end.lower[0]-time_start.upper[0])
    delta_hi = max(exact(0),time_end.upper[0]-time_start.lower[0])
    tau = (delta_lo+delta_hi)/2
    omega = tuple((lo+hi)/2 for lo,hi in zip(body_rate_box.lower,body_rate_box.upper))
    radii = tuple((hi-lo)/2 for lo,hi in zip(body_rate_box.lower,body_rate_box.upper))
    omega_error,omega_norm = _sqrt_upper(sum(v*v for v in radii)),_sqrt_upper(sum(v*v for v in omega))
    error = exact(operator_error)+omega_error*delta_hi+exact(angular_acceleration_bound)*delta_hi**2/2
    error += omega_norm*max(tau-delta_lo,delta_hi-tau)
    c0 = np.asarray(clock.point(float(sum(q0)/2)).rotation)
    c1 = np.asarray(clock.point(float(sum(q1)/2)).rotation)
    observed = tuple(tuple(map(exact,row)) for row in rotation_reference)
    center = c1.T@c0@np.asarray(observed,dtype=float)@Rotation.from_rotvec(np.array(list(map(float,omega)))*float(tau)).as_matrix()
    center = tuple(tuple(map(exact,row)) for row in center)
    with _ARB_LOCK,flint.ctx.workprec(precision_bits):
        ar = _Arithmetic(flint)
        first,last = clock.evaluate(ar.interval(q0),ar)[0],clock.evaluate(ar.interval(q1),ar)[0]
        ww = tuple(map(ar.scalar,omega))
        norm2 = sum(ar.square(v) for v in ww)
        skew = ((ar.scalar(0),-ww[2],ww[1]),(ww[2],ar.scalar(0),-ww[0]),(-ww[1],ww[0],ar.scalar(0)))
        if omega_norm==0:
            exp_rotation = tuple(tuple(ar.scalar(int(i==j)) for j in range(3)) for i in range(3))
        else:
            norm = ar.sqrt(norm2)
            theta = norm*ar.scalar(tau)
            k2 = tuple(tuple(sum(skew[i][k]*skew[k][j] for k in range(3)) for j in range(3)) for i in range(3))
            exp_rotation = tuple(tuple(ar.scalar(int(i==j))+ar.sin(theta)/norm*skew[i][j]
                +(1-ar.cos(theta))/norm2*k2[i][j] for j in range(3)) for i in range(3))
        obs = tuple(tuple(ar.scalar(v) for v in row) for row in observed)
        fixed = tuple(tuple(sum(first[i][k]*obs[k][j] for k in range(3)) for j in range(3)) for i in range(3))
        inertial = tuple(tuple(sum(fixed[i][k]*exp_rotation[k][j] for k in range(3)) for j in range(3)) for i in range(3))
        propagated = tuple(tuple(sum(last[k][i]*inertial[k][j] for k in range(3)) for j in range(3)) for i in range(3))
        residual = ar.box(tuple(propagated[i][j]-ar.scalar(center[i][j]) for i in range(3) for j in range(3)))
        error += _sqrt_upper(sum(max(abs(lo),abs(hi))**2 for lo,hi in zip(residual.lower,residual.upper)))
    return center,error
