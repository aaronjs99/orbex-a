# Arrival with a fixed deadline

`ArrivalPolicyConfig` adds an explicit finite arrival commitment to the nonlinear ADTMPC mission. The historical `SimulationConfig` and policy-absent metadata remain unchanged. Enabling this policy defines a new source cohort; a previously saved distance-gated run cannot acquire it through ordinary numerical resume.

The purpose is to prevent a reachable docking endpoint from remaining outside a proximity-only terminal gate, or being postponed by every new prediction horizon. No state is recentered or projected. Geometry, finite spacecraft bodies, uncertainty, tube and input bounds, native residual admission, continuous held-flow checks and the outer mission tolerances remain authoritative.

## What changed

Before the first accepted docking plan, the policy proposes arrival at the current exact guard clock plus the fixed horizon's number of intervals. Every chaser must return an accepted nonlinear plan and the aggregate planner must pass its simultaneous separation checks. A failed candidate is retained as an unknown numerical result and authorizes no motion.

The accepted fleet anchors `ArrivalCommitment`: the cumulative update, absolute deadline update, exact start/deadline anomaly, exact outer step, policy fingerprint, original complete candidate fingerprint and optional policy-transition identity. A later solve cannot move that deadline. The original complete candidate and latest complete candidate retain controls, states, tubes, reference/neighbor/model context and numerical acceptance information; they are explicitly **candidates**, not proved backups.

Let $j$ denote the current cumulative update, $j_D$ the recorded deadline update and $N$ the unchanged number of prediction nodes. The terminal suffix starts at

$$k_D=\max(0,j_D-j).$$

Every node from $k_D$ through $N-1$ has hard whole-tube terminal constraints. The initial accepted plan uses $k_D=N-1$; subsequent plans use an earlier index until the deadline reaches node zero. The same absolute target trajectory and clock are used throughout.

Future suffix nodes use an explicit strict planning fraction, normally 0.8. With outer tolerances 2 mm and 0.08 m/rad, the prospective planning targets are 1.6 mm and 0.064 m/rad. Fixed node zero uses the unchanged **outer** bounds: its carried data cannot be adjusted by the optimizer, and the planning interior is the reserve for carrying and independent endpoint-enclosure differences. Every future suffix node continues to use the planning interior. Actual completion still requires the original consecutive accepted endpoints.

The terminal pose formulas use nominal position and velocity errors $e_p$ and $e_v$, position half widths $\Omega$, sliding half widths $\Phi$ and diagonal sliding gain $\Lambda$:

$$D_p=\|\,|e_p|+\Omega\,\|_2,\qquad D_v=\|\,|e_v|+\Phi+\Lambda\Omega\,\|_2.$$

These bounds concern the complete error box conditional on its valid containment. The native backend rechecks every explicit suffix node from its returned arrays, with the existing numerical residual tolerance. This is numerical planning admission, not an integration enclosure theorem.

The backend's injectable `terminal_solve_reserve` defaults to $10^{-6}$ for
explicit future suffix rows. It multiplies their native tolerances by
$1-10^{-6}$, preserving unscaled error allowances, the node-zero outer set and
the original readback gate. This stricter native set leaves room for finite
precision result export; it does not change a returned state. Zero reserve
reproduces the earlier formulation, and policy-absent graphs are unchanged.
The saved native replay passes the original gate with this reserve, but no
single reserve value guarantees serialization accuracy for every scaling.

## Required reference-centered entry region

A terminal pose ball alone need not fit the physical tracking controller's component/sliding region. Optional `TerminalRegion` therefore supplies the required position half widths $\Omega_r$, sliding half widths $\Phi_r$, gain $\Lambda_r$ and explicit complete reference-state evaluation error bounds $E_{g,p}$ and $E_{g,v}$. Its error source, state frame, finite anomaly domain and maximum query-clock displacement are mandatory declarations. No velocity-evaluation bound is inferred from a pose rotation error or integrator tolerance.

The MPC tube satisfies $|\delta p|\le\Omega$ and $|\delta v+\Lambda\delta p|\le\Phi$. Its exact component support in the reference region is

$$|e_p|+\Omega+E_{g,p}\le\Omega_r,$$

$$|e_v+\Lambda_r e_p|+\Phi+|\Lambda_r-\Lambda|\Omega+E_{g,v}+\Lambda_r E_{g,p}\le\Phi_r.$$

Equal gains cancel the extra position term in the sliding expression. Adding another $\Lambda\Omega$ there would count the same correlated position error twice. Future-node planning scales only $\Omega_r$ and $\Phi_r$, keeping every error allowance unscaled; node zero uses the outer region.

Each numerical suffix query must lie in the error declaration's domain and within its explicit displacement from the canonical guard-derived clock. Actual endpoint membership uses the full independently enclosed state box and the same declared reference errors, with exact rational comparisons. The physical terminal controller still needs its own qualified global bounds, fleet geometry, finite scope and actual `PhysicalTerminalFleet.admit` gate. A region JSON or native inequality never substitutes for that certificate.

## Explicit API and activation record

Inject the typed policy into `MissionScenario`. The guarded mission owns its exact clock; this bounded implementation requires that guard. `DynamicTubeSpec` adds optional `terminal_start_index`, `terminal_planning_fraction` and `terminal_region` fields. Absent optional values are omitted at the solver boundary, preserving the historical formulation.

Enabled records also require a separate `arrival_activation` marker. It binds the immutable enablement update and exact anomaly to the fresh mission origin or explicit transition seam, and binds the original commitment after its first acceptance. Reload checks the original/latest candidate identities and ordered event ledger against that marker. Completed docking intervals since enablement cannot be reclassified as precommit by deleting the runtime ledger or artifacts. A first refused docking candidate with no committed docking interval remains a valid unaccepted state.

## Evidence and scope

The design was motivated by the source-qualified 160-update Envisat prefix: approximately 0.625 m nominal gap, approximately 50–65 micrometre tubes, and approximately exp(-2 delta-anomaly) local docking decay. Its velocity cost already used the target-relative preview. The historical 35 mm activation gate did not represent input/time reachability.

One reconstructed first-chaser problem, retaining all seven neighbor paths, passed the existing hard terminal pose rows in 8.36 seconds with unchanged objective coefficients and physical/numerical gates. That isolated candidate reached the numerical 2 mm/0.08 boundary but lacked both carrying reserve and required physical-region membership. It is a scheduling diagnostic, not a joint eight-chaser solution or mission completion. The new policy requires separate current-source fleet/execution qualification.

Holding a fixed deadline does not prove recursive nonlinear feasibility, real-time operation or a physical safe fallback. The ordinary continuous guard admits each actual shared hold; it does not establish a feasible successor. Until a complete held-feedback or input-box tail and terminal successor are independently certified, later failure retains its stopped simulation prefix and reports refusal. It does not claim that a physical plant can pause. Richer nonlinear inertial/J2-model qualification and modeled attachment/detumbling/deorbit retain their separate gates.
