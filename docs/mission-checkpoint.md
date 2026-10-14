# Numerical continuation of a saved approach

`MissionCheckpoint` continues a committed recorded mission under a changed solver source. It preserves the saved actual states, persistent nominal centers, component envelopes `phi` and `omega`, bandwidths, retained parameter belief, assignments, phase, terminal dwell, and reporter clock. A guarded restart also preserves the exact rational proof clock and every exact flow endpoint enclosure. No state is projected onto a reference or reset to a nominal center.

This is a numerical restart. It does not supply a safe successor when a solver fails, prove recursive feasibility, or make the retained run a fresh statistical comparison. A failed solve remains a failed attempt even when the corrected solver subsequently succeeds.

## Public interface

```python
from pathlib import Path
from orbexa.simulation import ADTMPCMissionRunner, MissionCheckpoint

# Reconstruct the original scenario using its original configuration, target
# epoch/domain, prior, deployment options, body geometry, and guard assumptions.
checkpoint = MissionCheckpoint.from_file(Path("saved/mission_data.json"))
checkpoint.validate_scenario(scenario, checkpoint.mission)
result = ADTMPCMissionRunner(scenario=scenario).resume(checkpoint, steps=20)
```

`steps` counts additional attempted MPC updates. `run(mission=..., steps=..., checkpoint=checkpoint)` provides the same interface with an explicit mission name. A checkpoint can be stored as plain JSON with `checkpoint.save(path)` and read with `MissionCheckpoint.load(path)`. A saved mission result is read with `from_file`; a serialized checkpoint is read with `load`.

New records contain `metadata.checkpoint_context`. An older result requires an explicit `source_context` decoded from its frozen configuration and source, passed to `from_file(..., source_context=context)`. The source context is a declaration that identifies the original runtime; a hash establishes identity, not the truth of an invented declaration. Do not infer historical settings from current defaults.

The maintained physical mission CLI supports the same bounded restart:

```powershell
orbexa-run-mission --config original-config.yaml --prior original-prior.json `
  --admission original-admission.json --chasers 8 `
  --resume saved-run/approach --steps 20 --output-dir continued-run
```

`--resume` accepts a mission result file or the directory containing `mission_data.json`. Supply the original configuration/prior/admission and vehicle options. `--steps` means additional updates and is capped by the original target preview domain. Omitting it uses the remaining bounded budget. For an older result, add `--source-context original-source-context.json`. The target bound scope is retained from the saved guard declaration or explicit legacy context; numerical sensing/model/target bounds still come from the supplied admission JSON and must agree. The CLI records a capped incomplete approach as incomplete and returns a nonzero status. Zero updates, an update budget beyond the original preview domain, and changed body geometry are rejected before reserving the output directory.

## Compatibility and finite domains

Every non-solver configuration value must match. Physical truth, target motion and geometry, original epoch and anomaly domain, known deployment/reference options, body radius, adaptation mode, planning horizon, outer interval and feedback substeps must match. The historical continuous guard cannot be added or removed. Controller/physical-model/tube-model versions must be supported. Unfinished nonlinear GEKKO single- and multi-agent records are accepted; a completed mission is not restarted.

A reconstructed rigid-body target retains its original finite anomaly domain. The caller must keep every new execution and preview query within that domain. Increasing an update cap does not authorize extrapolation or a changed target initial condition. `validate_scenario` checks domain identity; target queries retain their own domain checks. A CLI should reject an additional-update budget whose preview can leave the domain before starting execution.

Complete assignment, committed state/tube/control/belief/phase histories and every diagnostic appended by the current recorder are required. Incomplete optional histories are not fabricated after motion starts. Historical target kinematics must be finite, have the same event clock and physical-time record, and retain a compatible schema. The recorder preserves those historical samples verbatim and appends only new current-source samples.

## Guarded restoration

`CertifiedExecutionGuard.restore` accepts a fresh guard, the saved `continuous_execution_certificate`, fleet identities, reporter anomaly, dense hold width, checkpoint identity, and optional frozen-source context. It checks all compatibility decisions before changing guard state.

The original prior parameter box, measurement and additive-acceleration bounds, occupied-body radius, pair separation, actuator limits and target bound declarations must agree. The guard resumes from `exact_end_anomaly` and `endpoint_boxes[*].exact_lower/exact_upper`. It retains admitted/rejected counters, last rejection, start event and clock discrepancy. The next proof begins at the exact saved event. The rounded reporter clock continues to label the numerical engine; its discrepancy must remain within the guard's declared clock allowance.

The next measured state box is intersected with the previous certified endpoint. A numerical point never replaces that endpoint. An empty intersection refuses the next hold. These statements remain conditional on the original sensing/model/target/input assumptions and on authenticity of the saved evidence.

New guard metadata records `input_lower`, `input_upper`, `last_hold_width` and `target_bound_context`. For an older guarded record missing declarations, provide the following context in addition to its frozen `checkpoint_context`:

```python
source_context = {
    "source_reference": "immutable source result / manifest / archive",
    "checkpoint_context": original_scenario_context,
    "guard_context": {
        "source_reference": "identified frozen driver and configuration",
        "input_lower": [-100.0, -100.0, -100.0],
        "input_upper": [100.0, 100.0, 100.0],
        "target_bound_context": original_static_target_bounds,
    },
}
```

`original_static_target_bounds` is the original `RotatingCylinderBounds.as_dict()` restricted to `radius`, `half_length`, `angular_rate_per_anomaly`, `center_speed_per_anomaly`, `orientation_matrix_error`, `frame`, and `scope`. Its pose matrix is evaluated at the current event by the unchanged target; it is not a stored initial pose reused at another time. All seven static declarations must compare exactly. Missing actuator or target declarations refuse restoration rather than borrowing present-day values.

## Deliberate restart policies and provenance

Recorded results do not contain complete historical control horizons or the exact dense measurement/control window. The resumed planner therefore starts without warm plans, including neighbor warm plans. The adaptation window starts with the saved current primary state and event; exact old dense samples are not reconstructed from displayed mean outer inputs. The retained belief is unchanged. Membership/point-update scheduling continues from the recorded committed update count.

The recorder keeps every original solve attempt and cost. Proposals from a failed unexecuted fleet cycle are retained under `source_uncommitted_planning`, while the displayed committed nominal/profile histories remain aligned with executed intervals. The old failure message and original success flag, result hash, checkpoint identity/version, original and resumed source hashes, reset policies and scope appear in `numerical_restart_history`. Repeated restarts retain earlier segments. No dynamic callable or pickle object is serialized.

## Qualification evidence

Outside-repository probes on 2026-10-14 exercised a real two-update nonlinear guarded stop, checkpoint save/load, and two additional updates. Actual state, center, envelopes, bandwidth, belief, assignments and history prefixes were unchanged; the first resumed flow began at the exact old endpoint, and admitted-hold counts continued from 64 to 128. Changed body, input, measurement and horizon assumptions refused. A separate independent probe retained a zero-motion solver failure and then executed one actual guarded update from another checkpoint, preserving both restart segments and the original failed attempt. The maintained CLI also completed a real two-update prefix followed by one additional update, preserving the original target domain, policy, kinematics and exact clock; zero/over-domain/changed-body requests refused before creating outputs. A separately labeled legacy-metadata compatibility fixture retained the real prefix and executed one update with its original target bound scope and explicit source context; this checks backward declaration handling rather than supplying an additional historical mission run.

A frozen older eight-agent record stopped after 421 committed updates at a solver residual-verification failure. The corrected backend completed all 16 solves for two more fleet updates, reaching 423 committed updates, while preserving the original states/controller histories and failed attempts. Its original source file hash remained unchanged. This older case used its original prescribed target and point-body assumptions; it is not evidence for the separate gravity-gradient finite-body mission.

Independent negatives also covered changed retained model, widened actuator limits, sensing/forcing/target-rate changes, stale clocks, missing endpoints, malformed target clocks/times and missing diagnostic fields. These are bounded software and numerical continuation qualifications. They do not establish formal fallback viability, hardware timing, sensing validity or mission completion.
