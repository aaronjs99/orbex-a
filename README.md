# ORBEX-A

Adaptive dynamic tube MPC for eight chasers deployed in a common cluster 300 m
from a tumbling spacecraft. The mission sequence is approach, finite attachment,
assembly detumbling and orbital lowering to a descending 120 km interface.
The target study uses Envisat bus, solar-wing and appendage geometry.

The maintained runtime separates these responsibilities:

| Component | Implementation |
| --- | --- |
| Physical translation | Cartesian central gravity, inclined J2 and co-rotating vector quadratic drag |
| Target rotation | Quaternion/Euler rigid-body dynamics with gravity-gradient torque |
| Navigation | Bounded range/bearing, pose/rate and own-GNSS products with bias, acquisition time, latency and dropout |
| ADTMPC | GEKKO/IPOPT nominal motion, dynamic tube and feedback bandwidth; sensing-aware held-input reserve |
| Adaptation | Initial osculating eccentricity inference and nonlinear set-membership parameter exclusion |
| Motion admission | Independent interval flows, every spacecraft component and peer, finite backup and fuel reservation |
| Attachment | Finite positioners, compliant pad contact, friction, finite bonds and settling/load gates |
| Post-contact flight | Actual assembled mass/inertia, external thrusters, propellant and environmental dynamics |

[Nonlinear mission theory](docs/nonlinear-mission.md) connects the equations and
assumptions. The commanded reference orbit frame has a fixed clock; it is not
the perturbed target LVLH. Carter–Humi dynamics remain the optimizer's nominal
surrogate. Independently bounded nonlinear residuals and physical flows bridge
that surrogate to the declared truth model.

[Control foundations](docs/control-foundations.md) teaches the full chain from
tracking and high-order barriers to ADTMPC, adaptation, coordination and contact.
The [swarm consolidation record](docs/swarm-consolidation.md) explains the
optional planning tools, source preservation and measured component costs.

The COM uncertainty tube stays outside the hull. A finite carried mechanism
closes the remaining gap after the whole terminal tube meets the position and
physical-speed gates. Completion does not require forcing the true spacecraft
to an uncertainty tube's edge.

## Run

Python 3.10 or later:

```bash
python -m pip install -e ".[certification,studio,video]"
python run.py --case central --adaptation adaptive --bandwidth dynamic --output-dir ../orbexa-output/central-adaptive-dynamic
```

The movable checkout launcher supplies `config/envisat.yaml`,
`config/spacecraft/envisat.json` and `config/nonlinear-study.json`. Override those
paths explicitly when needed. `python -S run.py --help` needs no scientific
packages. Installed use is equivalent:

```bash
orbexa-run-nonlinear --config config/envisat.yaml --spacecraft config/spacecraft/envisat.json --study config/nonlinear-study.json --case central --adaptation adaptive --bandwidth dynamic --output-dir ../orbexa-output/central-adaptive-dynamic
```

Each invocation selects one declared case and arm. Run all four adaptive/frozen
belief × dynamic/fixed bandwidth arms for each case to form a matched study.
A new output directory is required. `--steps` limits execution without shortening
the prospective reference domain. A cap or a refused stage returns a nonzero
status and retains the actual prefix; it never means mission completion.

Before motion, each process freezes its Python source, explicit inputs and
resolved study declaration. It streams complete admission witnesses before each
applied hold, then records completion or assumption violation. Outputs include
`source-manifest.json`, `study-manifest.json`, `approach/`, `completion/`,
`nonlinear-certificates.jsonl.zst` and `study-metrics.json`. Comparisons require
matching source and input hashes, the complete declared case/arm grid, and all
failed or capped outcomes. Planner-only and full attempted-update timings are
separate; offline simulation does not establish a flight deadline.

The lossless Zstd journal retains the complete ordered JSON witnesses. Each
acknowledged event is flushed and fsynced before motion; closure adds a checksum
and the manifest. Closed readers reject incomplete frames, checksum errors,
trailing data and decoder windows above 128 MiB. Earlier gzip archives remain
readable. An interrupted byte prefix is retained as an aborted run.

## Mission Studio

[Open the published Studio](https://orbex-a-mission-studio.aaronjs99.chatgpt.site/studio/).
Its saved record identifies its own source, model and completed stages. Publishing
a viewer does not upgrade an older recording's dynamics or proof scope.

For live local simulation:

```bash
python serve.py
```

Open `http://127.0.0.1:8881/studio/`. Release distance, spacing and target-spin
knobs start fresh native solver runs through the loopback API on port 8882.
Only committed samples update the scene. Run, pause and reset retain each job's
source and output directory. An already running process keeps its frozen source.
The browser is a viewer; it does not run GEKKO inside the page.

If those ports serve an earlier job, start alongside it with
`python serve.py --ui-port 8893 --api-port 8894` and open the printed Studio URL.
The hosted viewer checks ports 8882 and 8894 and requires the v2 sensed nonlinear
handshake before accepting a state or enabling commands. Each local viewer
uses its explicitly configured API port.

Mission, Plots and Theory share one recorded clock. True-size tube outlines and
a magnified error-space view show the small uncertainty geometry without changing
physical bounds. `MissionStudio`, `MissionPlots`, `NonlinearMissionPlots` and
`LifecycleRenderer` derive portable replays, figures and video from saved data.
Missing measurements remain unavailable. Decorative stars and the dispenser
illustration are display context.

## Models and scope

The [sensor contract](docs/nonlinear-navigation.md),
[sensed tube derivation](docs/sensed-tube.md),
[finite feedback fallback](docs/fleet-fallback.md),
[finite contact model](docs/finite-contact.md), and
[Envisat geometry sources](docs/envisat.md) state the implemented bounds.
The rigid geometry and contact parameters are a research model. Raw image
perception, flexible appendages, hardware qualification, atmospheric demise and
regulatory disposal approval are outside this simulation. Chaser attitude is a
causal rotational replay; approach translation uses a declared force-servo
error envelope. The model does not claim a coupled thruster-resolved flight test.

Source lives in `scripts/`, installed as the `orbexa` package. Dynamics,
estimation, control, solver backends, mission orchestration and rendering use
separate interfaces. Generated evidence stays outside the checkout. Paper source
and figures retain the user's layout in `paper/IEEE Aerospace Conference 2025/`;
analytical reproduction APIs remain available separately from the current run
entry. See [architecture](docs/architecture.md) and [scenario inputs](config/README.md).

## Citation

```bibtex
@inproceedings{johnsabu2025orbexa,
  title={Capturing Tumbling Objects in Orbit with Adaptive Tube Model Predictive Control},
  author={Aaron John Sabu and Brett T. Lopez},
  booktitle={IEEE Aerospace Conference},
  year={2025}
}
```

GPLv2. See `LICENSE`.
