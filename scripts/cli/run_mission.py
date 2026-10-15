"""Run compact physical ADTMPC deployment through modeled disposal."""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
import numpy as np

from orbexa.core.config import SimulationConfig
from orbexa.estimation.belief import ParameterBelief
from orbexa.simulation.adtmpc_mission import ADTMPCMissionRunner
from orbexa.simulation.complete_mission import CompleteMissionOptions, CompleteMissionRunner, CompleteMissionStore
from orbexa.simulation.mission_checkpoint import MissionCheckpoint
from orbexa.simulation.mission_result import MissionResultStore
from orbexa.simulation.physical_approach import ContinuousAdmissionOptions, PhysicalApproachOptions, PhysicalApproachScenario


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Explicit scenario YAML")
    parser.add_argument("--prior", type=Path, required=True, help="JSON with feasible_sets and estimates, independently selected from truth")
    parser.add_argument("--admission", type=Path, required=True, help="JSON with the declared ContinuousAdmissionOptions sensing/model/target envelopes")
    parser.add_argument("--output-dir", type=Path, required=True, help="A new directory; stage records survive subsequent failures")
    parser.add_argument("--steps", type=int, help="Maximum new control updates: 550 for a fresh run, or the remaining original preview domain on restart; a cap is not completion")
    parser.add_argument("--resume", type=Path, help="Numerically continue an unfinished mission_data.json or its directory; preserve the committed prefix")
    parser.add_argument("--source-context", type=Path, help="Explicit frozen-source context for legacy restart records missing checkpoint/guard declarations")
    parser.add_argument("--chasers", type=int, default=8)
    parser.add_argument('--target-model',choices=('composite','cylinder-baseline'),default='composite',
        help='Explicit component spacecraft by default; cylinder is retained for paper-baseline reproduction')
    parser.add_argument('--spacecraft',type=Path,help='Explicit geometry JSON; default config sibling spacecraft/envisat.json')
    parser.add_argument('--deployment',type=Path,help='Explicit deployment-options JSON; composite also checks the configuration-name sibling')
    parser.add_argument("--com-standoff-m", type=float)
    parser.add_argument("--body-radius-m", type=float)
    parser.add_argument("--initial-target-spin", type=float, nargs=3, metavar=("X", "Y", "Z"),
        help='Body rad/s; composite default is a declared2.4degree/s historical-range study, not a current state estimate')
    parser.add_argument("--disable-gravity-gradient", action="store_true")
    args = parser.parse_args(argv)
    if args.source_context is not None and args.resume is None:
        parser.error("--source-context requires --resume")
    journal = None
    try:
        config = SimulationConfig.load(str(args.config))
        config.num_chasers = args.chasers
        config.__post_init__()
        prior = ParameterBelief(**json.loads(args.prior.read_text(encoding="utf-8")))
        admission = ContinuousAdmissionOptions(**json.loads(args.admission.read_text(encoding="utf-8")))
        input_bound = max(abs(float(bound[key])) for bound in config.mpc.input_bounds for key in ("lower", "upper"))
        design=None
        if args.target_model=='composite':
            from orbexa.simulation.spacecraft_design import SpacecraftMissionDesign
            from orbexa.simulation.capture import ChaserPhysicalProperties
            geometry=args.spacecraft or args.config.parent/'spacecraft'/'envisat.json'
            radius=.55 if args.body_radius_m is None else args.body_radius_m
            standoff=.85 if args.com_standoff_m is None else args.com_standoff_m
            design=SpacecraftMissionDesign.from_json(geometry,com_standoff_m=standoff,
                chaser=ChaserPhysicalProperties(dry_mass_kg=30.,fuel_kg=100.,body_radius_m=radius,gripper_offset_m=.8))
            spin=tuple(args.initial_target_spin or [0.,0.,math.radians(2.4)])
        else:
            if args.spacecraft is not None:
                raise ValueError('--spacecraft requires the composite target model')
            radius=.15 if args.body_radius_m is None else args.body_radius_m
            standoff=.165 if args.com_standoff_m is None else args.com_standoff_m
            spin=tuple(args.initial_target_spin or [.0002,.0003,.00015])
        checkpoint, domain, target_scope = None, None, None
        steps = (config.num_update_steps if design is not None else 550) if args.steps is None else args.steps
        if args.resume is not None:
            source_path = args.resume / MissionResultStore.filename if args.resume.is_dir() else args.resume
            source_context = None if args.source_context is None else json.loads(args.source_context.read_text(encoding="utf-8"))
            checkpoint = MissionCheckpoint.from_file(source_path, source_context=source_context)
            source_result = checkpoint.result
            domain = source_result.target.get("anomaly_domain")
            if not isinstance(domain, (list, tuple)) or len(domain) != 2:
                raise ValueError("Physical restart requires the original target anomaly domain")
            remaining = math.floor((float(domain[1]) - checkpoint.anomaly) / config.anom_step
                                   - max(config.mpc.horizon_steps.values())) - 1
            if args.steps is None:
                steps = remaining
            if steps > remaining:
                raise ValueError(f"Restart permits at most {remaining} updates inside the original target preview domain")
            saved_guard = source_result.metadata.get("continuous_execution_certificate") or {}
            target_context = saved_guard.get("target_bound_context") or (source_context or {}).get("guard_context", {}).get("target_bound_context", {})
            target_scope = target_context.get("scope")
        if steps < 1:
            raise ValueError("At least one new update must fit the requested run and target preview domain")
        def certificate_sink(record):
            journal.write(json.dumps(record, allow_nan=False) + "\n")
        target=None
        if design is not None:
            config.target.docking_standoff=standoff
            config.target.inertia=np.asarray(design.geometry.metadata['mass_properties']['inertia_kg_m2'])
            end=(max(steps,config.num_update_steps)+max(config.mpc.horizon_steps.values())+2)*config.anom_step
            target=design.target(config,initial_body_spin_rad_s=spin,
                anomaly_domain=(0.,end) if domain is None else domain,
                gravity_gradient=not args.disable_gravity_gradient)
        deployment=None
        deployment_path=args.deployment
        if design is not None and deployment_path is None:
            candidate=args.config.with_name(args.config.stem+'-deployment.json')
            if candidate.is_file():
                deployment_path=candidate
        if deployment_path is not None:
            if design is not None:
                from orbexa.simulation.composite_deployment import CompositeDeploymentOptions
                deployment=CompositeDeploymentOptions(**json.loads(deployment_path.read_text(encoding='utf-8')))
            else:
                from orbexa.simulation.deployment import CompactDeploymentOptions
                deployment=CompactDeploymentOptions(**json.loads(deployment_path.read_text(encoding='utf-8')))
        profile = PhysicalApproachScenario(config, initial_belief=prior, maximum_updates=steps,
            options=PhysicalApproachOptions(body_radius_m=radius, com_standoff_m=standoff,
                initial_body_spin_rad_s=spin, gravity_gradient=not args.disable_gravity_gradient),
            admission=admission, certificate_sink=certificate_sink, target_anomaly_domain=domain,
            target_bound_scope=target_scope,target=target,deployment_options=deployment,
            vehicle_design=None if design is None else design.as_dict())
        if checkpoint is not None:
            checkpoint.validate_scenario(profile.scenario, "single" if args.chasers == 1 else "multi")
        from orbexa.simulation.capture import ChaserPhysicalProperties
        from orbexa.simulation.chaser_attitude import ChaserAttitudeTracker
        tracker = (ChaserAttitudeTracker(default_chaser=ChaserPhysicalProperties(body_radius_m=radius))
                   if design is None else None)
        approach_runner = ADTMPCMissionRunner(scenario=profile.scenario)
        runner = (design.completion_runner(input_component_bound=input_bound,approach_runner=approach_runner)
            if design is not None else CompleteMissionRunner(options=CompleteMissionOptions(input_component_bound=input_bound),
            approach_runner=approach_runner, attitude_tracker=tracker,
            ))
        store = CompleteMissionStore(args.output_dir)
        journal = gzip.open(store.directory / "continuous-certificates.jsonl.gz", "wt", encoding="utf-8")
        runner.stage_sink=store.record
        result = (runner.run(mission="single" if args.chasers == 1 else "multi", steps=steps)
                  if checkpoint is None else runner.complete(approach_runner.resume(checkpoint, steps=steps)))
        output = store.finish(result)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, ImportError) as error:
        parser.error(str(error))
    finally:
        if journal is not None:
            journal.close()
    print(json.dumps(dict(success=result.success, message=result.message, output=str(output)), indent=2))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
