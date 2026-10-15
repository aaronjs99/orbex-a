# ORBEX-A

ORBEX-A implements adaptive dynamic tube MPC (ADTMPC) for compact multi-chaser
deployment and rendezvous with an uncontrolled spacecraft. The physical mission
continues through finite-reach attachment, external-thruster detumbling and
finite-burn orbital lowering to a descending entry interface. Elliptical relative
dynamics, quadratic drag, nonlinear GEKKO/IPOPT geometry, adaptive uncertainty
sets and held-feedback dynamic tubes remain the approach controller.

The implementation separates dynamics, uncertainty, planning, continuous-motion
admission, terminal tracking, spacecraft attitude, contact gates, propulsion and
recorded-data presentation. The mathematical claims and their assumptions are
collected in [the theory](docs/theory.md).

The Envisat study uses eleven explicit bus, solar-wing and appendage components
and eight candidate structural patches. Planning, exclusion, capture and replay
consume the same geometry. [Its model and sources](docs/envisat.md) distinguish
published dimensions and mass/inertia from inferred layout and unknown current
state. The completed eight-chaser recording is a **cylinder baseline**, not
Envisat mission qualification.

The approach plant is the paper's analytical relative-motion approximation.
An independent Cartesian SI dynamics service includes nonlinear central gravity,
full vector quadratic drag and optional J2. Its discrepancy replay identifies
model bias outside the representative approach's declared allowance; the saved
approach certificates therefore qualify the analytical model. See
[orbital model fidelity](docs/dynamics-fidelity.md) for the equations, measured
differences and the assumptions required before using a richer truth plant.

`python run.py --help` works directly from a source checkout, including after
moving the folder, without installing the package or its numerical dependencies.
Mission execution still requires the runtime dependencies below. The checkout
launcher and installed console command use the same options.

## Install

```bash
pip install -e .
```

ADTMPC missions use the coupled nonlinear GEKKO/IPOPT controller. Ordinary MPC
and other solver adapters remain available through the Python API.

## Run a spacecraft mission study

Install continuously validated flow, local Studio and MP4 dependencies:

```bash
python -m pip install -e ".[certification,studio,video]"
```

Run an Envisat study with its explicit prior, deployment and assumed
sensing/model envelopes in [config](config/README.md). Supply the study's
[arrival-policy declaration](docs/arrival-policy.md), including sourced reference
errors and a validity domain covering the planned terminal suffix:

```bash
orbexa-run-mission --config config/envisat.yaml --prior config/envisat-prior.json --admission config/envisat-admission.json --arrival-policy path/to/arrival-policy.json --output-dir ../orbexa-output/envisat-eight
```

This is an actual nonlinear optimization run. The update cap is a resource bound;
it does not imply success. The physical target obeys quaternion/Euler rigid-body
dynamics in seconds with gravity-gradient torque. Each bus has a finite volume,
starts in a common compact region, and tracks its assigned rotating COM goal.
Every admitted translational feedback hold is enclosed continuously under the
declared bounds before the fleet commits it. A failed solve or unresolved hold
is retained as a failed mission rather than being labeled unreachable.

An unfinished saved approach can be numerically restarted with the same physical
configuration, prior, bounds and target domain:

```bash
orbexa-run-mission --config config/envisat.yaml --prior config/envisat-prior.json --admission config/envisat-admission.json --arrival-policy path/to/arrival-policy.json --resume ../orbexa-output/envisat-eight/approach --output-dir ../orbexa-output/resumed-eight --steps 20
```

The cap now counts additional updates. The original states, tubes, beliefs and
exact proof clock persist; unavailable warm plans and dense adaptation history
reset explicitly. Older records need identified frozen-source context through
`--source-context`. This [numerical restart](docs/mission-checkpoint.md) is separate
from certified backup control after a solver failure.

Introducing arrival into a historical prefix additionally requires
`--transition-arrival-policy`; its complete original configuration must match.
Use `--historical-terminal-policy` for the identified distance-gated reproduction.
Missing arrival declarations refuse before reserving an output directory.

After approach completion, the runner conservatively debits translation fuel,
replays bounded chaser attitude control, and supplies those actual attitudes and
remaining tanks to attachment, detumbling and deorbit. Completed stages are saved
immediately in a new directory. The entry endpoint is a modeled handoff at
120 km altitude; atmospheric breakup and disposal compliance require separate
models and evidence.

An already saved physical COM approach can be completed without repeating MPC:

```bash
orbexa-complete-mission ../orbexa-output/envisat-eight/approach/mission_data.json --output-dir ../orbexa-output/completed-eight --input-component-bound 100000
```

When a verified physical terminal window follows that exact approach, add
`--terminal-execution ../orbexa-output/terminal/terminal-result.json.gz`.
`PhysicalTerminalHandoff` binds the source record, original endpoint, exact clock
and every held-flow/geometry proof before exposing the actual terminal endpoint.
Attitude replay and propellant accounting include that extension. Capture uses
its final reference-centered enclosure; the original MPC history remains intact.

The input bound must cover **every actual held command**, including ancillary
feedback. Composite completion binds installed bus, fuel and nozzle properties
to the design recorded during approach. The Envisat concept uses eight 30 kg
dry / 100 kg fuel buses, a 0.55 m body radius, 0.85 m COM standoff and 100 N
translation channels. Its 2.4°/s spin is a historical-range study input, not a
current Envisat estimate. Actuator, fuel, constant-inertia and contact gates
can still refuse completion.

Select the paper-scale cylinder profile explicitly when reproducing it:

```bash
orbexa-run-mission --target-model cylinder-baseline --config config/default.yaml --prior config/physical-prior.json --admission config/physical-admission.json --output-dir ../orbexa-output/cylinder-eight --steps 550
```

See [physical lifecycle](docs/mission-lifecycle.md),
[chaser attitude](docs/chaser-attitude.md),
[continuous safety](docs/continuous-safety.md),
[fixed-deadline arrival](docs/arrival-policy.md),
[terminal tracking and backup](docs/terminal-tracking.md), and
[derived physical terminal bounds](docs/physical-terminal.md).

## Mission Studio

Open the [published Mission Studio](https://orbex-a-mission-studio.aaronjs99.chatgpt.site)
for the recorded 3D missions, plots, videos and illustrated theory. The Site is
private and uses ChatGPT sign-in. Envisat approach evidence and completed
cylinder-baseline mission results retain their separate outcomes.

`MissionStudio` exports explicitly selected saved missions into a local browser
workspace: a black-space 3D replay, synchronized goal/tube/reserve charts,
camera controls, finite-body envelopes, recorded video, mathematical documents
and a separate two-view post-capture replay. `LifecycleRenderer` produces
scientific plots, GIF and MP4 from the same SI state records. Browser video
export stays on the local computer. No controller is run inside the viewer.

The renderer, equation fonts and licenses are included in the package, so the
Studio needs no third-party network requests. Serve the exported directory with
an ordinary local HTTP server; open its `studio/` page. Raw source hashes,
sample identities, modeled assumptions, failed outcomes and synthetic fixtures
remain visible. Generated evidence belongs outside this checkout.
The workspaces are Mission, Plots and Theory. A phase selector shares approach
and attached-flight replay without duplicate navigation.

## Generate ADTMPC Mission Artifacts

Default full nonlinear run from the repository root:

```bash
python run.py
```

The default YAML scenario uses 8 chasers for the multi-agent mission. Select a
different scenario with `--config path/to/scenario.yaml`; its `num_chasers` sets
the fleet size, and its orbit parameters set the simulated plant truth while
the controller starts from a separate uncertain belief. The checkout default
is resolved relative to the package location, so launching from another working
directory works. Wheel installations must supply `--config` if the checkout
scenario is unavailable.

Equivalent console entry point:

```bash
orbexa-generate-demo \
  --output results \
  --data-output data \
  --steps 450 \
  --mission all \
  --primary-solver gekko
```

Here `--steps` is the maximum number of receding-horizon MPC updates. The
nonlinear default jointly optimizes nominal motion, dynamic tube states, and
feedback bandwidth with GEKKO/IPOPT. It carries the nominal center and tube
between updates and applies ancillary feedback within each control interval.
The default scenario uses 12 prediction nodes, one applied nominal interval,
and 32 ancillary feedback substeps. Phase horizons come from YAML; API callers
can override `horizon_steps`. The coupled policy replans every outer interval.
Fresh nonlinear primary runs return failure if the goal is not reached.

This writes:

- `results/<session_id>/index.html`: the session gallery, with mission selection, animations, interactive replays, plot filters, and full-resolution images
- `results/<session_id>/single/nonlinear/`
- `results/<session_id>/multi/nonlinear/`
- raw mission JSON under `data/<session_id>/`
- editable run-note templates at `results/<session_id>/README.md` and `data/<session_id>/README.md`
- portable `results/latest.json` and `data/latest.json` references to the newest generated session; directory symlinks are also created when supported

Each mission folder contains its own gallery, `manifest.json`, `replay.html`,
`trajectory.html`, `tube_trajectory.html`, `diagnostics.html`, `trajectory.gif`,
optional `trajectory.mp4`, and the existing control, safety, tube, target, and
parameter diagnostics. Matched tube views show position/sliding-error containment,
terminal accuracy including the entire tube, bandwidth allocation, and ancillary
input-reserve use. Other views show docking errors, the measurable drag
difference and retained marginal ambiguity, emitted-point SMID errors, a body-frame
docking close-up PNG/GIF, and a minimum pairwise-spacing heatmap for fleets.
`insights_manifest.json` records source sample indices, playback rate, fallback
assumptions, and reasons for unavailable views. Replays use recorded states;
playback is sample-based and does not represent physical time. Unsupported views
remain absent for incomplete or older saved data. Rendered sessions also expose
`results/<session_id>/demo.gif`, using a symlink when available and a copy otherwise.

Open the session's `index.html` in a browser. All interactive views include their
plotting runtime and work offline. Partial saved-data rendering retains the
other rendered mission families through `gallery.json` or their older manifests.

Playable MP4 output uses FFmpeg on the system path or the local executable
provided by `pip install -e ".[video]"`. Without an available encoder, the report
links to the GIF and `manifest.json` records why MP4 was skipped.

Generated session artifacts stay out of git. Add a curated preview intentionally when a representative mission artifact is available.

Physical constraint margins concern the actual vehicle. Tube-expanded margins
concern each persistent nominal center and its own matched-node tube. The
cylinder constraint values use radial and axial length margins in meters.
Parameter plots distinguish point estimates from the retained uncertainty box.
Local optimization may update a point estimate; only the outward-rounded interval
RK4 membership contractor removes parameter values. Unprocessed boxes are retained
when its budget expires. Exact held ancillary substep inputs are used for adaptation.

Solver results contain `T` state samples, including the initial state, and
`T - 1` control intervals. Control column `k` drives state `k` to state `k + 1`.
Backend adapters normalize their grid conventions before returning results.
The coupled mission reserves every ancillary-input term inside the optimization.
Saturation, reserve exceedance, measured containment failure, and inconsistent
membership stop the mission and preserve the available evidence. It does not
change a nominal command through an unaccounted docking projection. The assigned
side corridor is deliberately conservative and retains nonlinear cylinder
exclusion. Fleet planning constrains contemporaneous nominal paths and both tubes.

Moving-reference preview is enabled by default. State previews have shape
`(n, T)` and feedforward previews `(m, T - 1)`. The explicit arrival policy fixes
an absolute deadline and constrains the remaining terminal suffix inside the
required reference-centered region. Future nodes reserve a planning interior;
fixed node zero and actual completion keep the outer criteria. Completion requires the exact
whole-box norm `norm(abs(nominal position error) + Omega)` to fit 2 mm, and
`norm(abs(nominal velocity error) + Phi + Lambda*Omega)` to fit 0.08 m/rad,
at five consecutive outer endpoints.
The paper demo's requested goal remains its configured 5 mm point/COM standoff;
the finite-bus physical profile explicitly selects 0.165 m. A local Riccati
terminal penalty aids performance. The separate reference-centered terminal
controller requires a valid derived bound certificate and real entry membership;
ordinary MPC dwell does not establish its invariant-set conditions.

The physical-coordinate model uses meters, derivatives per true anomaly, and
anomaly-acceleration inputs. Physical acceleration increments are `qdot² * u`;
the approach itself does not integrate mass/nozzle dynamics. The complete runner
checks a conservative force/fuel allowance and separately propagates bounded
attitude and post-latch propulsion. The default
orbit now has a 300 km perigee at eccentricity 0.18. Its analytical quadratic-drag
approximation and prescribed Euler target motion are explicitly limited models.
See [the derivation and focused literature review](docs/adtmpc-design.md) for frame
conventions, moving-goal/obstacle distinctions, and evidence boundaries.

Nonlinear geometry is constrained at prediction nodes and checked more densely
during execution. Continuous held-flow admission and stored backup continuation
are explicit services with stricter assumptions and entry conditions; sampled
simulation alone does not establish them. Neither service guarantees that every
nonlinear performance optimization will be feasible. Physical latch loads,
uncertain release/navigation and atmospheric demise remain separate engineering
questions.
Saved-data mission rendering requires the current physical model and coupled
controller metadata. Retired runs remain in external recovery archives.

`run.py` defaults to the ADTMPC mission workflow. Useful options:

- `--mission single|multi|all`
- `--steps N`
- `--output results`
- `--data-output data`
- `--session-id NAME` (fresh runs require an unused, portable folder name)
- `--config PATH`
- `--primary-solver gekko`
- `--clean-generated` (explicitly removes only the selected generated roots, including their saved session notes; source and scenario inputs are protected)
- `--from-data`

Regenerate plots from saved data without rerunning solvers:

```bash
orbexa-generate-demo \
  --output results \
  --data-output data \
  --mission all \
  --from-data
```

Use `--session-id NAME` and `--mission single|multi` to select a saved physical
ADTMPC session. Rendering uses its data and does not load `--config` or construct
a mission runner. Every requested family is checked before existing artifacts
or notes are changed. Partial rendering retains other current model families.

Fresh runs never overwrite an existing session. Session names are single folder
names; `latest`, `latest.json`, path traversal, and Windows device names are
reserved. Existing ordinary `latest` folders and editable run notes are retained.
Raw mission data, manifests, and latest references are written atomically.

Incomplete or solver-failed primary missions save their raw data and visual
gallery before returning a nonzero status. Empty and one-state saved results
show their available evidence and omit unsupported motion views. Use
`--no-require-primary-success` deliberately to allow a zero exit status for
bounded diagnostic runs; their galleries still identify unmet mission criteria.

Generated plots, videos, JSON outputs, caches, and local paper PDFs are ignored by git.

## Repository Layout

Source code and CLIs live under `scripts/`, mapped to the `orbexa` Python
package for stable imports and console commands. Generated data belongs under
`data/`; rendered artifacts belong under `results/`. Package console entry
points are installed by `pip install -e .`.

Paper figures remain under `paper/IEEE Aerospace Conference 2025/assets/`.
The user's moved layout is preserved. See [the architecture guide](docs/architecture.md)
for module responsibilities, dependency injection, public APIs and recorded-data rendering.

The active checkout contains the maintained physical-coordinate implementation,
scenario settings, and current design notes. Imported historical manuscripts,
diagrams and prototypes are preserved in verified external recovery archives
and Git history, outside the active project tree.

## Citation

```bibtex
@inproceedings{johnsabu2025orbexa,
  title={Capturing Tumbling Objects in Orbit with Adaptive Tube Model Predictive Control},
  author={Aaron John Sabu and Brett T. Lopez},
  booktitle={IEEE Aerospace Conference},
  year={2025}
}
```

## License

GPLv2. See `LICENSE`.
