# Physical terminal endpoint handoff

`PhysicalTerminalHandoff` connects a completed physical terminal execution to
attachment, attitude replay and propellant accounting. The originating
`ADTMPCMissionResult` is preserved. The terminal segment stays a separate record;
it does not manufacture optimizer solutions, nominal plans, SMID observations,
belief updates or MPC dwell intervals.

```python
from orbexa.simulation import PhysicalTerminalHandoff
from orbexa.simulation.approach.records.results import MissionResultStore
from orbexa.simulation.completion.coordinator import CompleteMissionOptions, CompleteMissionRunner

approach = MissionResultStore.load(approach_path)
handoff = PhysicalTerminalHandoff.from_files(approach_path, terminal_result_path)
handoff.verify_approach(approach)
runner = CompleteMissionRunner(options=CompleteMissionOptions(input_component_bound=100))
completion = runner.complete(approach, terminal_handoff=handoff)
```

The maintained CLI accepts the same source pair:

```powershell
orbexa-complete-mission APPROACH/mission_data.json `
  --terminal-execution TERMINAL/terminal-result.json.gz `
  --input-component-bound 100 --output-dir NEW_OUTPUT_DIRECTORY
```

The bound `100` must cover every held translation input, including actuator
uncertainty. It is checked against the explicit combined input intervals and
recorded means/command centers; sparse means alone cannot establish a held
input bound. Installed nozzles, fuel and attitude gates still apply.

## Verification boundary

`from_files(approach_path, terminal_result_path)` accepts plain or gzip terminal
JSON produced by `PhysicalTerminalRunner`. It checks the raw originating
approach SHA-256, its full semantic history, the physical target epoch and
model, assignments, initial prior and final nominal parameter context. It
requires the original exact guarded endpoint clock and exact endpoint boxes.
The initial actual states remain identical, and the initial measurement box
is intersected with those endpoint boxes. The rounded source reporter clock
is retained separately and must fit the declared reference query budget.

The verifier reconstructs each global bound builder and rederives all terminal
inequalities. It then re-evaluates each held command, proves each continuous
uncertain flow again with Arb, and proves the complete fleet's swept separation.
Recorded commands, exact start/hold clocks, request boxes, fingerprints,
geometry and their joint hash must reproduce. Every saved numerical endpoint
must lie inside its corresponding proved endpoint box. The final observation
must refine the last flow endpoints and satisfy robust terminal membership.
An adjacent `continuous-certificates.jsonl.gz`, when present, must match every
hold in the execution record. Saved success labels alone do not qualify a
handoff.

The terminal body envelope must enclose the original recorded physical body
envelope. Its input interval must lie inside the original declared input
interval. Downstream physical buses are additionally checked against each
verified terminal body envelope, and the fuel budget must cover the complete
combined interval through the exact terminal end event.

This focused version requires one common Lambda, a completed full declared
finite window, and deterministic reproduction using the producer's default
Arb stepping policy with the saved precision. It refuses unguarded inputs,
unsupported proof policies, missing declarations and incomplete stages. It
never replaces a refused endpoint with a reference state.

## Actual endpoint and reference reserve

The public endpoint fields are `states`, `nominal_centers`, `omega`, `phi`,
`lambda_gain`, `capture_truth` and the rational `exact_end_anomaly`.
`nominal_centers` means the evaluated physical reference in this interface;
these values are not MPC nominal centers.

Let the exact physical reference be `g`, its evaluated value be `g_tilde`, and
the declared component evaluation errors be `E_p` and `E_v`. The terminal proof
uses `|p-g_p| <= Omega` and
`|v-g_v + Lambda*(p-g_p)| <= Phi`. Capture operates around `g_tilde`, so its
outward rounded enclosing widths are

```
Omega_capture = Omega + E_p
Phi_capture   = Phi + E_v + Lambda*E_p.
```

The actual endpoint is unchanged. The original true-reference widths and
position tolerance remain recorded separately in `controller_bounds`; the
capture widths conservatively reserve evaluation error. The capture model
still checks whole-enclosure reach, speed, body clearance and pair separation,
as well as actual attitude/spin and state membership, before the idealized
contact and rigid attachment event.

`trajectory` is a separate `PhysicalTerminalTrajectory` for attitude replay and
fuel accounting. It concatenates the original actual state/target sample prefix
with the actual terminal samples. Its controls contain source applied-input
means followed by actual held terminal command centers, with that distinction
recorded. Its exact end clock includes the terminal stage. Its composite clock
metadata is explicitly an accounting boundary; the original approach and
terminal proof records remain separate. It is not an `ADTMPCMissionResult`.

`as_dict()` returns compact endpoint, controller bounds and source/journal
identities. It does not duplicate the full raw proof record or original
trajectory. `verify_approach(approach)` rejects a changed originating history or
mutation of the previously verified endpoint/timeline before downstream use.

## Scope

These checks establish source correspondence and recheck the conditional
finite-window certificates. Initial physical spin, sensing, reference/model
evaluation, geometry and process bounds retain their declared assumption
status. Numerical target kinematics and the separate bounded chaser attitude
replay do not become validated physical sensing or a hardware execution proof.
Attachment remains an explicit idealized contact interface on the selected
bare-hull surface points. No universal terminal feasibility or real-time
hardware theorem is inferred.
