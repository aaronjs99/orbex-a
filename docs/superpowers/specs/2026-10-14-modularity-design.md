# ORBEX-A modular completion design

The user-requested layout is authoritative: maintained Python code lives directly
under `scripts/`; paper assets live under `paper/IEEE Aerospace Conference 2025/`.
Setuptools maps the public `orbexa` namespace to `scripts/`. No former source
folders are recreated. A dependency-free layout module supplies checkout paths.

The completion target is a coherent, reusable implementation of the existing
physical adaptive dynamic tube MPC model. Its equations, uncertainty policy,
whole-tube terminal checks and numerical solver constraints remain intact.
Simulation evidence is not a claim of complete spacecraft physics or perfection.

## Module boundaries

- Mission data models, target geometry, saved results and I/O have independent
  modules. Reading a saved result must not construct a controller or solver.
- Fleet initialization, phase/reference policy and safety checks are distinct
  services composed into an injectable mission scenario. The public runner is
  a small entrypoint; execution must not call its private implementation helpers.
- Coupled planning, dense feedback execution, parameter adaptation and recording
  use explicit state/step interfaces. The nominal center and tube persist between
  updates; all chasers commit a shared clock step together.
- The GEKKO adapter owns symbolic backend details. Dynamics and interval
  enclosures remain backend-independent. No heuristic fallback is reinstated.
- Visualization config, manifests, plots, animations, catalogs and session
  orchestration are separate. Existing `visualization.demo` imports remain usable
  as a facade. Current-model identity and partial-render migration remain enforced.
- Runtime policy and estimator/solver settings are structured configuration with
  defaults equal to the qualified scenario. Reject malformed dimensions, invalid
  numeric values and unsupported modes before running an optimizer.
- Broken unused prototypes are retired only after consumer review; valid public
  helpers receive explicit dependencies. Original research remains recoverable.

## Interfaces and compatibility

`ADTMPCMissionRunner.run(mission=..., steps=...)`, `save_result`, `load_result`,
`DemoConfig`, `OrbexaDemo`, `ADTMPCMissionRenderer` and the console command remain
available. The new `simulation.mission_result` module owns `ADTMPCMissionResult`,
`load_adtmpc_mission_result` and `result_chaser_ids`. Model identity strings and
saved schemas remain compatible. Plain result records cross the runtime/render
boundary. New services accept dependencies explicitly and provide defaults.

## Qualification

Use actual nonlinear mission runs, comparison with immutable earlier raw
records, independent dense replay, saved-data rendering, editable/wheel build
and import examples. All evidence and temporary probes stay outside the repo;
no repository test or validation infrastructure is added or run. Preserve the
user's moves and record them through Git rename detection after hash checks.

## Physical limits

The maintained model uses physical LVLH translation in true anomaly, analytical
quadratic drag and prescribed target Euler motion. Contact, complete actuator
physics and full chaser attitude dynamics require additional identified models
and qualification; modular interfaces allow that future work without fabricating
physical validation in this refactor.
