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

The prescribed initial nominal state and initial `Phi`/`Omega` box are also
GEKKO parameters. Their finite values, nonnegative tube components and complete
initial state-box bounds are validated before construction. Future states,
tubes and adaptive bandwidths remain decisions with the same objective and
hard constraints. This removes redundant initial equalities and the artificial
optimization direction through the norm of a fixed zero position box. In an
exact reconstructed Envisat problem, the earlier representation exhausted
1200 iterations despite a numerically feasible initial physical iterate. The
parameter representation passed the same retained residual gate on its first
attempt, with maximum residual approximately `1.52e-12`. This qualifies that
numerical problem; it does not establish a complete Envisat mission or universal
solver convergence.

For composite spacecraft, `solver.composite_geometry_mode` defaults to `free`.
The explicit `reference_corridor` option fixes one reference support witness per
component and node. It keeps every component, the original finite-body/tube and
uncertainty supports, the normalized component-margin postcheck and the
continuous-flow guard. It removes free normal/absolute-value decision variables
and retains their restricted objective constants. It is a stricter feasible
set, so a safe detour using another component face may be rejected. Invalid
modes or nonseparating reference witnesses refuse; there is no automatic mode
fallback. See the [support equations, trapping example and same-problem timing
evidence](composite-approach.md#optional-reference-support-corridor).

Solver diagnostics record the selected mode, including failed numerical
attempts. Checkpoint restarts may explicitly change solver settings and record
their source/options as a new numerical segment. Such a restart is not a matched
comparison cohort and cannot replace the original failed result.

A separate retained fixed-bandwidth problem previously failed warm and cold
attempts. The corrected parameter representation and hard-bound policy solved
that identical problem with residual below `1e-14`. A full mission comparison
after a backend repair must use a separate matched source cohort. Replacing a
failed baseline with one successful re-solve would confound the scientific study.

## Bounded attempts and cleanup

APMonitor's `MAX_MEMORY` option is a base-ten exponent for the sparsity storage
available per equation. It is **not a number of megabytes**: the maintained
default `4` means `10**4` sparsity elements. Every public setting accepts an
integer from `2` through `10`, rejecting booleans, floats and values such as the
former `512`. See the [official APMonitor option definition](https://apmonitor.com/wiki/index.php/Main/OptionApmMaxMemory).

A retained Envisat study failed during model allocation with `MAX_MEMORY=1024`,
before any fleet update or safety admission. A memory-only replay at the former
default `512` reproduced the allocation failure before writing a result.
Replaying that exact archived
nonlinear model with only this setting changed to `4` reached native IPOPT
success with an unscaled constraint residual of `1.46e-11`. The other model
input bytes were identical. This isolates the allocation failure for that
problem; numerical solution and subsequent mission admission are separate
checks. Historical raw inputs retain the invalid setting, and follow-up runs
use a separately frozen source cohort.

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
