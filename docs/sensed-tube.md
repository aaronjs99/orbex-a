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

## Shorter execution holds

The planner reserves numerical error for transferring the nominal center
after a hold of width \(H>0\). Let \(\epsilon_p\) and \(\epsilon_v\) be its
componentwise position and velocity error budgets. They are numerical
enclosure allowances, separate from physical disturbance and sensing bounds.
An installed backup or a segment boundary can require an actual hold
\(0<h\le H\). Reusing the full error budget at that faster cadence would
understate accumulated reset error; requiring \(h=H\) would reject every
otherwise admissible shortened hold.

For a nonnegative decay rate \(a\), define its positive comparison weight

\[
w_a(t)=\begin{cases}(1-e^{-at})/a,&a>0,\\t,&a=0.\end{cases}
\]

The planned position and sliding-error reset forcing is respectively
\(\epsilon_p/w_\lambda(H)\) and
\((\epsilon_v+\lambda\epsilon_p)/w_\alpha(H)\), component by component.
Concavity and \(w_a(0)=0\) give
\(w_a(h)/w_a(H)\ge h/H\). The executor therefore requires its independently
certified endpoint defects to satisfy the stricter rational budgets

\[
\delta_p\le\frac hH\epsilon_p,\qquad
\delta_v\le\frac hH\epsilon_v.
\]

These imply both positive-comparison reset allowances at the actual hold,
including the coupled sliding jump \(\delta_v+\lambda\delta_p\). The
executor records the effective budget, the prospective budget and the exact
scale. It refuses nonpositive or overlong holds and any excessive certified
defect. At \(h=H\), the previous budget and transfer record are unchanged.
Stage peaks, observation coverage, physical flow, geometry, continuation and
resource admission are still independently required. This correction removes
a deterministic cadence refusal; it does not establish mission convergence.
