# ADTMPC with bounded observation error

The performance controller retains the GEKKO/IPOPT multiple-shooting problem.
Its nominal dynamics are linear in the relative state, with coefficients that
depend on the scheduled reference anomaly. A separately admitted regional
nonlinear residual belongs in the acceleration forcing. The reference clock
and coordinate frame are explicit; a nonlinear Cartesian orbit does not make
its true anomaly equal the scheduled reference anomaly.

Let the true error from the nominal center be `e=(ep,ev)` and the feedback
observation be `e+n`, with component bounds `|n| <= N`. Let
`s=ev+Lambda ep`, `|ep|<=Omega`, `|s|<=Phi`,
`V=Phi+Lambda Omega` and `R=(Omega,V)`. All products and inequalities below
are componentwise except matrix-vector products. The actual implemented
ancillary law is

```
ua(e) = -A e - Lambda ev
        -(deltaA |e| + alpha Phi) sat(s/Phi).
```

At a zero component of `Phi`, the implementation uses `sign(s)` for that
component; the term `alpha Phi sat(s/Phi)` is exactly `alpha clip(s,-Phi,Phi)`
and vanishes. No proof step divides an observation error by `Phi`.

The complete sampled input, evaluated on the observation, satisfies

```
|ua(e+n)| <= (|A|+deltaA)(R+N) + Lambda(V+Nv) + alpha Phi.
```

The right side is the reserved ancillary input. Reserving only the ideal
feedback input would permit actuator saturation to invalidate the tube.

For the forcing due to observing a perturbed state, use
`||e+n|-|e||<=N`, `|sat(a)-sat(b)|<=2`, and the nonexpansiveness of clipping:

```
|ua(e+n)-ua(e)| <= (|A|+deltaA)N + Lambda Nv
                  +2 deltaA R + alpha(Nv+Lambda Np) = Fobs.
```

This bounded inequality holds even at `Phi=0`. It is conservative: the
uncertain saturation term may use its full two-sided range. `Fobs` is added to
the held-feedback comparison forcing, and the enlarged input reserve is used
when bounding the error derivatives during each hold. The existing first-exit
comparison then applies with these larger derivative and forcing bounds,
provided the complete state, coefficient, residual, observation, input and
maximum-hold declarations cover the executed interval. The resulting positive
systems are `Phi'=-alpha Phi+Ftotal` and
`Omega'=-Lambda Omega+Phi`; positive stage peaks cover their intersample values.

`measurement_error_abs` is optional. An absent or identically zero vector
retains the previous exact-state expressions, including their operation order.
Nonzero vectors identify `held-ancillary-observation-bounded-v2` in solver
records. These expressions establish a conditional tube comparison. Actual
navigation coverage, nonlinear residual bounds, swept geometry, fallback entry
and finite contact are separate admission requirements.

The one-off 2026-10-16 qualification checked exact backward equivalence in
1,200 cases and 18,000 finite feedback boundary/noise samples, including zero
sliding widths. Native local GEKKO solved both the original and sensed
formulations and returned the corresponding version and forcing fields.
Those numerical samples exercise the implementation; the inequalities above
state the sufficient bound and its assumptions.
