# Nonlinear solver acceptance and failure handling

The coupled ADTMPC formulation remains a nonlinear GEKKO/IPOPT problem. A
backend success flag is necessary but insufficient: every retained physical,
tube, epigraph and terminal constraint is independently evaluated on the returned
candidate. A residual above `1e-8` is rejected. The largest named residuals and
their stage/component context are saved in `solver_info`.

## Hard bounds and fixed bandwidth

IPOPT normally relaxes user variable bounds before optimization. Its final
reported constraint violation can exclude violations of the original unrelaxed
bounds. The maintained ADTMPC backend therefore sets native
`bound_relax_factor 0` on every attempt and records that option. This preserves
the intended hard bounds; it does not weaken the independent residual gate.
See the [official IPOPT option definition](https://coin-or.github.io/Ipopt/OPTIONS.html#OPT_bound_relax_factor).

In a reconstructed eight-chaser problem, the earlier backend returned success
while the normalized terminal-velocity constraint had residual approximately
`5.22233e-6`. Tightening only the outer GEKKO convergence settings did not repair
it. Disabling native bound relaxation solved the same problem with maximum
retained residual approximately `2.08e-14`. This is evidence about that saved
problem, not a guarantee of success on every future nonlinear program.

When bandwidth is fixed by equal bounds or zero permitted change, its values
are GEKKO parameters, rather than equal-bound optimization variables with
redundant zero-rate inequalities. The initial bandwidth must satisfy the declared
bounds; it is never clipped to make an inconsistent initial tube appear feasible.
Adaptive bandwidth remains an optimization variable.

A separate retained fixed-bandwidth problem previously failed warm and cold
attempts. The corrected parameter representation and hard-bound policy solved
that identical problem with residual below `1e-14`. A full mission comparison
after a backend repair must use a separate matched source cohort. Replacing a
failed baseline with one successful re-solve would confound the scientific study.

## Bounded attempts and cleanup

The backend first uses its supplied warm start, then a cold retry on failure.
If a numerical solution fails independent residual acceptance, a bounded stricter
refinement may use the candidate with tighter convergence settings and scaling
disabled. Physical constraints, tolerance, uncertainty and actuator limits remain
unchanged. All temporary local models are cleaned after success or failure.

An optimization failure is an **unknown numerical outcome**, not a proof that
the requested motion is impossible. The ordinary mission preserves the last
committed fleet state and failed solve diagnostics. It does not apply an
unverified zero input, arbitrary brake or fabricated trajectory. The separate
[continuation supervisor](terminal-tracking.md) can execute an already admitted
backup when its explicit chain and terminal conditions are available.
