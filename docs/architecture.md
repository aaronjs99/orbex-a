# ORBEX-A architecture

The maintained sensed nonlinear pipeline separates physical truth, bounded
observations, nominal GEKKO planning, held-command admission and recorded evidence.
The public Python package is `orbexa`; its checkout source is `scripts/`.
Analytical translation reproduction remains available under its own model and
clock identities. Sharing an interface or renderer does not equate these models.

## Entry points and composition

`NonlinearMissionRunner` composes one explicit `IndependentTruthCase` with
`NonlinearStudyOptions`, a vehicle design and structured `SimulationConfig`.
`orbexa-run-nonlinear` creates a `StudyArmRequest` for `run_study_arm`, which
freezes the selected source/configuration/case and runs the requested
adaptive/frozen and dynamic/fixed arm in a new output directory. The update cap
does not turn an unfinished approach into a completed mission.

`LiveSimulationService` uses this same nonlinear runner, planner, executor and
completion composition. Its worker boundary owns job control, frozen inputs and
committed projections; the browser consumes those projections. Historical
analytical resume is a separate, explicitly selected route. Neither entry point
implements a second controller or physical plant.

```mermaid
flowchart TD
    CLI[StudyArmRequest and frozen study arm] --> R[NonlinearMissionRunner]
    LIVE[LiveSimulationService worker] --> R
    R --> TR[Separate target and relative physical truth]
    TR --> SS[SensorSuite truth-side sampler]
    SS --> NAV[BoundedNavigationFilter]
    NAV --> PRE[Observed NonlinearTargetPreview]
    NAV --> AD[NonlinearMissionAdaptation and membership]
    AD --> PL[NonlinearMissionPlanner]
    PRE --> PL
    PL --> GK[Nominal GEKKO/IPOPT problem]
    GK --> PROP[Performance proposal]
    NAV --> SAFE[NonlinearSafetyPlanner and finite backup]
    SAFE --> ADM[FleetAdmissionRuntime]
    PROP --> ADM
    FLOW[NonlinearFlowEncloser and AttitudeFlowEncloser] --> PL
    FLOW --> ADM
    ADM --> EX[NonlinearDenseExecutor]
    EX --> TR
    EX --> NAV
    EX --> ARC[Staged and applied hold archive]
    EX --> REC[NonlinearMissionRecorder]
    REC --> RAW[Saved approach result]
    RAW --> COMP[Authenticated completion boundary]
    ARC --> COMP
    COMP --> LIFE[Finite contact and physical lifecycle]
    RAW --> PROJ[StudioDataset]
    LIFE --> LPROJ[LifecycleDataset]
    PROJ --> STUDIO[One MissionStudio replay]
    LPROJ --> STUDIO
```

The arrows show responsibility and data flow. In particular, truth reaches the
controller through sensor packets and navigation; recorded truth and offline
containment diagnostics are separate outputs.

## Package boundaries

The source hierarchy follows runtime responsibility. Package APIs expose public
interfaces; internal callers import their canonical owner. Old flat file paths
do not receive forwarding modules.

| Area | Nested responsibilities |
|---|---|
| `core/` | Configuration, force/frame dynamics, geometry, spacecraft state and target motion |
| `control/` | MPC, tube algebra, whole-step certification, fallback and terminal admission |
| `estimation/` | Navigation, conservative membership, adaptation policy and numerical fitting |
| `planning/` | Rendezvous, coordination, trajectory/policy proposals and optimization objectives |
| `solvers/gekko/` | GEKKO formulation, geometry, fitting and native IPOPT iteration evidence |
| `simulation/` | Approach, contact, attached flight, online execution, studies and recording |
| `visualization/` | Saved data, plots, renderers, publishing and session workflow |
| `utils/` | Shared geometry, discretization, explicit data/configuration IO and runtime helpers |

The physical `scripts/` name and the five installed commands remain stable.
Regular Python packages are discovered recursively; static Studio assets remain
package data rather than incidental namespace packages. Exporters resolve the
asset directory from the package root, independently of their own nesting depth.

## Maintained module ownership

Paths in this table are relative to `scripts/`.

| Module | Main interfaces and responsibility |
|---|---|
| `models/configuration/types.py`, `models/configuration/loader.py`, `models/configuration/model.py` | Structured scenario validation, YAML loading and configuration composition |
| `common/contracts.py`, `common/layout.py`, `sessions.py` | Solver problem/result records, relocated-checkout paths and explicit generated-session ownership |
| `models/dynamics/reference_frame.py` | `ReferenceClockFrame`: declared reference orbit, commanded anomaly clock and frame transforms |
| `models/dynamics/relative.py` | `NonlinearModelBounds`, `ExponentialAtmosphere`, `NonlinearRelativeDynamics`: central gravity, J2, co-rotating vector drag and exact transformed relative RHS |
| `models/targets/nonlinear.py` | `NonlinearTargetTruth` owns numerical Cartesian, quaternion and body-rate truth; `NonlinearTargetPreview` is updated from current navigation observations |
| `models/dynamics/inertial.py` | Independent Cartesian SI dynamics and frame calculations used to check the physical equations separately |
| `models/geometry/composite.py`, `mission/lifecycle/configuration/design.py` | Exact expanded component solids, patch geometry and prospective mass/actuator/contact declarations |
| `proofs/nonlinear.py` | `NonlinearFlowEncloser`, `TargetEphemerisCertificate`, `RegionalResidualCertificate`: strict Picard flow bounds, physical-time support, nonlinear-minus-nominal regional bounds and Arb nominal coefficients |
| `proofs/attitude.py`, `mission/safety/target_bounds.py` | `AttitudeFlowEncloser` and source-bound moving-goal errors from observed pose/rate supports; continuous occupied-geometry bounds remain separate |
| `estimation/navigation/records.py` | `StudySensorProfile`, `SensorPacket`, `NavigationEstimate`: immutable sensor declarations, processed packets and retained navigation products |
| `estimation/navigation/sensors.py` | `SensorSuite`: truth-side sampling with versioned random draws, delivery latency and deterministic error support |
| `estimation/navigation/filter.py` | `BoundedNavigationFilter`: prior/prediction/observation intersections, delayed packets and finite-history velocity bounds |
| `estimation/navigation/moments.py` | Exact acceleration-history moment indexing with guarded fallback to the original segment scan |
| `estimation/navigation/boxes.py` | Exact box intersection, interval products and numerical midpoint representatives shared by records and the filter |
| `estimation/navigation/target_pose.py` | Chaser GNSS target-ephemeris inference and acquisition-age target-pose propagation |
| `estimation/navigation/noise.py` | `PhysicalSlotNoise`: versioned bounded common random field keyed to physical-time bins and sensor channels; acquisition scheduling remains owned by `SensorSuite` |
| `estimation/membership/nonlinear.py`, `estimation/adaptation/nonlinear.py`, `estimation/adaptation/belief.py` | Nonlinear transition/observation membership, retained parameter sets and admissible nominal updates; unresolved parameter boxes are retained |
| `mission/profile.py` | `NonlinearStudyOptions`, `IndependentTruthCase`, declared model/clock/prior and compact fleet composition |
| `mission/planning.py`, `mission/approach/planning.py` | `NonlinearPlanningContext`, `NonlinearMissionPlanner`: sensed uncertainty, target forecasts, regional residuals and coupled nominal/tube candidates |
| `control/tubes/specification.py`, `control/tubes/sample_hold.py`, `solvers/gekko/tube.py` | `DynamicTubeSpec`, sensed held-feedback bounds and `GekkoDynamicTube` nonlinear GEKKO/IPOPT formulation |
| `solvers/gekko/iterations.py` | Native IPOPT final-summary iteration counts with log hashes and exit provenance; unavailable work remains unknown |
| `mission/coordination/allocation/{contracts,networks,system}.py` | Agent state, explicit neighbor graphs and task-allocation scheduling |
| `mission/coordination/allocation/policies/`, `mission/coordination/allocation/demo.py` | Greedy, Hungarian and auction policies; optional plotting and demonstration stay outside controller imports |
| `control/fallback/fuel.py` | `FleetFuelBudget`: exact same-event, same-hardware remaining-fuel budget for finite successor selection; admission independently rechecks the debit and complete reserve |
| `proofs/reachability.py`, `proofs/orbital.py` | Interval/held-input/flow records and geometry guards; `OrbitalHoldProofService` refines the analytical nominal endpoint while retaining its swept witness |
| `control/fallback/refuge.py`, `control/fallback/plan.py` | `RefugeController`, `FleetFallbackPlan`: finite feedback authority, complete route construction and retained certificates |
| `control/fallback/polynomial.py`, `control/fallback/geometry.py` | Exact polynomial peer bounds, occupied-component projections and whole-fleet separation |
| `control/fallback/records.py`, `control/fallback/runtime.py` | Immutable proposal/decision/resource records and `FleetAdmissionRuntime` transactions, current-state checks and fuel accounting |
| `mission/safety/reference.py`, `mission/safety/drift.py` | Exact C2 retreat bounds and Arb actual-drift envelopes under a fixed nominal model |
| `mission/safety/{geometry,resources,cache}.py` | Complete occupied-pose adapters, physical propulsion limits and guarded frozen-proof identities |
| `mission/safety/candidates.py`, `mission/safety/planner.py` | Bounded refuge search, backup preparation/reuse and `ProspectiveFleetEndpoint` successors |
| `mission/execution/blackout.py` | Finite causal observation propagation while the installed backup executes |
| `control/actuation.py`, `control/tubes/sensed.py` | Declared actuator support/realization, exact sensed sliding-tube algebra, stage checks, feedback evaluation and physical tube witnesses |
| `mission/execution/coordinator.py`, `mission/execution/contracts.py`, `mission/approach/state.py` | `NonlinearDenseExecutor`: admission, physical commit and failure ordering; explicit execution outcomes and controller estimate state |
| `mission/execution/plant.py`, `mission/execution/sensing.py` | `NonlinearFleetPlant` proposes numerical motion; `MissionSensorDelivery` owns acquisition and delayed packet queues |
| `mission/execution/bindings.py`, `mission/execution/nominal.py` | `PerformanceStageBindings` checks planning sources; `NominalTubePropagator` owns certified nominal flow/cache and tube endpoint transfer |
| `evidence/journal.py` | `ExecutionJournal` publishes durable events and retains acknowledged completed holds; later recorder failures preserve that prefix |
| `mission/runner.py`, `mission/records/recorder.py` | Outer orchestration, arrival/deadline/dwell gates and `NonlinearMissionRecorder` with explicit truth/navigation/control-source histories |
| `evidence/records/archive.py` | `NonlinearProofArchive`, `NonlinearHoldArchive`: durable staged/applied/failure witnesses, hash-bound lightweight indexes and lazy complete-hold iteration |
| `studies/api.py`, `studies/request.py`, `evidence/source.py`, `cli/run_nonlinear.py` | Guarded study entry, prospective request/declaration and coherent source/input freezing |
| `studies/execution.py`, `studies/records.py`, `studies/metrics.py` | `StudyArmExecution`, `StudyTelemetry`, atomic records and outcome metrics; setup, approach, completion and failure retention remain separate stages |
| `mission/lifecycle/nonlinear/coordinator.py`, `mission/lifecycle/nonlinear/contracts.py` | Physical lifecycle composition, compatible imports and shared completion contracts/source identity |
| `mission/lifecycle/nonlinear/boundary.py`, `mission/lifecycle/nonlinear/journal.py`, `mission/lifecycle/accounting.py` | Arrival/contact-boundary authentication, complete-hold authentication and translation fuel accounting |
| `mission/lifecycle/environment/point.py`, `mission/lifecycle/nonlinear/observers.py` | Physical environment and causal attitude/contact observations |
| `mission/approach/physical/attitude.py`, `mission/contact/simulator.py`, `mission/contact/capture.py` | Controlled chaser-attitude replay, finite-contact coordination, reach/contact gates and accepted assembly construction |
| `mission/contact/contracts.py`, `mission/contact/dynamics/state.py`, `mission/contact/control/controller.py`, `mission/contact/dynamics/mechanics.py`, `mission/contact/control/guards.py`, `mission/contact/records.py` | Typed contact inputs/results, packed fleet state, observed control, physical wrench/RHS, geometric/load guards and balance/history recording |
| `mission/attached/assembly/model.py`, `mission/attached/propulsion.py` | `AttachedAssembly`, `AssemblyState`, `ThrusterSystem`: mass/inertia/fuel evolution, bounded nozzle allocation, pre-motion pad-load admission and actual load reconstruction |
| `mission/attached/disposal/axis.py`, `mission/attached/disposal/preview.py`, `mission/attached/detumbling.py`, `mission/attached/disposal/simulator.py`, `mission/lifecycle/stages.py` | Burn-axis/throttle previews, controlled rate damping, finite-burn events and `MissionLifecycle` stage ownership and load/failure gates |
| `service/{contracts,session,http}.py` | Typed live inputs, one worker/session owner and the bounded loopback HTTP adapter |
| `service/{snapshot,process}.py`, `service/jobs/` | Coherent source/input freeze, isolated spawned lifetime and explicit nonlinear/historical execution routes |
| `evidence/{artifacts,native}.py`, `mission/approach/arrival/composition.py` | Shared durable/native evidence instrumentation and prospective arrival composition, independent of HTTP |
| `evidence/projections/studio.py`, `figures/renderers/lifecycle.py`, `figures/publishing/studio.py`, `figures/studio/` | `StudioDataset`, `LifecycleDataset`, `MissionStudio`: one recorded-data browser replay and synchronized approach/lifecycle views |
| `common/geometry/`, `common/discretization.py` | Geometry algebra, sampling, occlusion and region calculations; numerical discretization imports its backend only when selected |
| `common/io/{configuration,data}.py`, `utils/runtime.py` | Explicit YAML configuration, atomic data persistence and opt-in runtime/keyboard helpers |

## Changing one subsystem

Change numerical integration in `mission/execution/plant.py`, sensor
delivery in `mission/execution/sensing.py`, and tube equations in
`control/tubes/sensed.py`. Change contact forces in
`mission/contact/dynamics/mechanics.py`, observed contact control in
`mission/contact/control/controller.py`, and acceptance limits in
`mission/contact/control/guards.py`. Change
arrival authentication, fuel arithmetic, and study reporting in their respective
completion/study modules. The coordinators connect these parts and own stage
ordering; they do not contain alternate implementations of their equations.

The public package interfaces expose the maintained mission services. Internal
imports use the nested implementation owners. The execution coordinator forwards
its established packet, journal, sink and nominal-cache attributes to their owners,
so there is one queue/cache rather than copied state. Actuation and tube records
bind the implementation file responsible for their mathematics; study manifests
bind the complete Python source inventory. Contact and completion also expose
source bundles covering every component in their respective implementations.

This decomposition preserves the numerical model and admission rules. It does
not establish full-mission success, real-time speed or an adaptive advantage.

## Clock, model and estimation boundaries

`ReferenceClockFrame` prescribes the reference rotation and positive anomaly
rate. Relative position is `C_ref(q).T @ (r_chaser - r_target)` and relative
velocity is its derivative with respect to `q`. This frame is not asserted to
be the actual nonlinear target's physical LVLH frame. Cartesian ephemeris,
physical seconds and attitude are propagated with the declared clock bridge.

The numerical and Arb nonlinear RHS share equations and immutable physical
declarations. Certificates bind the model, clock, target ephemeris, input,
regional domain and source. The independent inertial service checks those
equations; it is not a replacement source of controller measurements. See
[navigation](nonlinear-navigation.md) and the
[historical analytical fidelity audit](dynamics-fidelity.md).

`SensorSuite` produces truth-side packets. Navigation consumes declared priors,
validated held-flow predictions and delivered bounded packets. Own-chaser GNSS
plus relative geometry can constrain target ephemeris; a cooperative target
GNSS feed is not assumed. The observed target preview uses current estimated
pose/rate and prospective prediction, without future truth access. Sensor and
actuator magnitudes are explicit study parameters, not device qualifications.

The nonlinear membership contractor uses the actual nonlinear transition and
observation boxes. It removes only certified inconsistent parameter boxes and
retains unresolved ones. The analytical RK4 contractor does not establish
membership for this physical model. A failed NLP is a retained solver attempt,
not an infeasibility or unreachability label.

## Planning, admission and physical commits

The optimizer is a nominal Carter–Humi surrogate with prospectively charged
regional nonlinear residuals, measurement/actuator supports, sample-and-hold
terms and numerical endpoint-transfer allowances. `NonlinearPlanningContext`
binds one observation/model/clock/target forecast snapshot to that solve.
`GekkoDynamicTube` remains the nonconvex GEKKO/IPOPT backend; SciPy and optional
CasADi supporting solvers do not silently replace this formulation.

Before performance motion, the executor checks the source bindings, declared
regional domain, current sensing support, reserves and tube peaks.
`FleetAdmissionRuntime` independently checks the whole fleet's held physical
flows, component/peer separation, successor entry, context and finite fuel.
`NonlinearSafetyPlanner` may retain an authenticated remaining backup, prepare
a fresh finite one, or prepare a prospective successor from the complete proposed
physical-flow endpoint boxes. Preparation is not installation. A refusal preserves the
installed feedback decision; it does not silently substitute zero thrust.

Candidate selection checks the complete continuation against the current fuel
ledger. For a proposed hold, its typed physical upper debit is subtracted once
to form the prospective endpoint budget. An unaffordable stored suffix triggers
the same finite candidate search as failed endpoint containment. Far and patch
searches continue past unaffordable candidates; the final admission transaction
still checks the held debit plus the whole successor reserve. Proof services
cannot alter the installed plan, clock, ledger, pending decision, disturbance
or navigation state while preparing a successor. Near-target forecasts use
only the declared retreat candidates whose entire retreat and holding suffix
fit the remaining clock domain.

Performance and fallback use the same dense executor and physical plant.
Admitted physical-tube witnesses can carry relational inclusion across
performance holds; initialization and fallback resets use the authoritative
navigation enclosure. Bounded proof-service blackout uses the already admitted
finite fallback and its observation propagator. Its authority, geometry, clock,
sensor and fuel conditions remain finite and explicit.

`ADMITTED_BEFORE_MOTION` binds the decision and full witnesses before numerical
plant motion. `APPLIED_HOLD_COMPLETE` records the actual endpoint, input and
navigation update; downstream failure records `ASSUMPTION_VIOLATION` and the
actual committed clock/state. A failed planning attempt and the fallback motion
that follows it remain distinct evidence. Lightweight recorder indexes require
the corresponding durable full archive; the complete-hold iterator supplies
the accountant without retaining every witness in memory. See the detailed
[nonlinear mission contract](nonlinear-mission.md) and
[solver formulation](nonlinear-solver.md).

## Physical continuation

`complete_nonlinear_mission` requires a successful sensed approach and verified
arrival record, matching model/clock/vehicle, retained actual boundary and the
complete admitted hold journal. Translation fuel is charged before the next
stage. `PastNavigationAttitudeObserver` supplies recorded past observations to
the separate controlled chaser-attitude replay; physical truth remains separate.
That replay and its attitude/fuel gates precede finite contact.

`FiniteContactSimulator` models declared compliant contact and observed local
feedback. An accepted contact/capture result supplies the attached assembly
and its actual pose, rate and remaining tanks to detumbling.
`ThrusterSystem` applies hardware/fuel limits and fixed-state Newton–Euler
pad-load admission before controlled motion. Detumbling and deorbit retain
applied endpoints, audit their loads immediately, and preserve the full sampled
load-history lifecycle gate. These are numerical point/history checks, not a
continuous load enclosure or hardware bond qualification.

`MissionLifecycle` owns attachment, detumbling and deorbit results and retains
completed stages on later refusal. Burn-axis/throttle predictions are candidates;
the actual propagated pointing, fuel, rate, load and orbital-event gates decide
the result. A descending entry-interface handoff and achieving a lower target
perigee are separate recorded outcomes. Below-interface heating, breakup,
demise and impact are outside this pipeline.

A separately reconstructed post-contact replay has its own source/boundary and
stage evidence. Its success cannot fill a missing approach prefix or upgrade an
older analytical record into a sensed nonlinear full mission. See
[finite contact](finite-contact.md) and [mission lifecycle](mission-lifecycle.md).

## One Studio and evidence ownership

`StudioDataset` and `LifecycleDataset` project plain saved approach/lifecycle
records. `MissionStudio` exports their shared local browser assets and verifies
raw-result/lifecycle linkage. Static replay, plots and browser video use recorded
samples; the browser does not run a solver or plant. Live operation feeds the
same projection contract from acknowledged worker commits. Approach and
lifecycle clocks remain explicitly labeled, and absent histories are not
invented to complete a scene.

Saved results, full proof archives, source/configuration snapshots, case/arm
declarations and lifecycle records have different responsibilities. A controller
proposal is not physical truth; a numerical sample is not a validated flow;
a replay is not new execution evidence. Old records keep their original
identities and qualification scope. Current study runs, full-mission completion
and published galleries must be assessed from their own closed evidence.

Setuptools maps the checkout's `scripts/` to `orbexa`; wheels contain `orbexa/`.
Generated sessions and qualification outputs go to the gitignored `data/` and
`results/` folders, resolved by `ProjectLayout` in `common/layout.py`; source, configuration
and paper folders refuse them. Source paper figures remain publication assets. Repository policy
retains no test/validation infrastructure. Mathematical claims remain
conditional on the finite declared domains and admission contracts in
[the nonlinear formulation](nonlinear-mission.md); this architecture makes no flight, device,
universal environmental-coverage or real-time execution claim.
