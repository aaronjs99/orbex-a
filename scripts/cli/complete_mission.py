"""Complete a qualified physical approach using configurable lifecycle models."""
from __future__ import annotations

import argparse
from dataclasses import replace
from functools import partial
import json
from pathlib import Path

from orbexa.simulation.contact.capture import CaptureModel, CaptureOptions, ChaserPhysicalProperties, TargetPhysicalProperties
from orbexa.simulation.approach.physical.attitude import ChaserAttitudeActuators, ChaserAttitudeTracker
from orbexa.simulation.completion.coordinator import CompleteMissionOptions, CompleteMissionRunner, CompleteMissionStore, bind_approach_body_geometry
from orbexa.simulation.attached.disposal.simulator import DeorbitOptions, DeorbitSimulator, ThrustPointingPolicy
from orbexa.simulation.attached.detumbling import DetumblingOptions, DetumblingSimulator, RateDampingPolicy
from orbexa.simulation.completion.lifecycle import MissionLifecycle, MissionLifecycleStore
from orbexa.simulation.approach.records.results import MissionResultStore
from orbexa.simulation.attached.propulsion import ThrusterSystem
from orbexa.simulation.attached.disposal.axis import AxialBurnPlanner
from orbexa.simulation.attached.disposal.preview import FiniteBurnThrottlePlanner


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("approach_result", type=Path, help="Qualified physical mission_data.json or its directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--terminal-execution", type=Path, help="Successful physical terminal-result JSON or gzip bound to this exact approach; capture uses its literal final endpoint")
    parser.add_argument("--assume-attitude-matched", action="store_true", help="Explicit idealization; otherwise actual bounded chaser attitude tracking is required")
    parser.add_argument("--input-component-bound", type=float, help="Declared bound on EVERY actual held approach command in m/rad²; required for conservative translation fuel accounting")
    parser.add_argument("--approach-point-is-capture-reference", action="store_true", help="Alternate explicitly declared tip-reference input; COM remains the default")
    parser.add_argument("--target-mass-kg", type=float)
    parser.add_argument("--chaser-dry-mass-kg", type=float)
    parser.add_argument("--chaser-fuel-kg", type=float, help="Initially installed fuel for attitude replay; remaining-at-capture fuel only with --assume-attitude-matched")
    parser.add_argument("--chaser-body-radius-m", type=float)
    parser.add_argument("--gripper-offset-m", type=float)
    parser.add_argument("--capture-reach-m", type=float)
    parser.add_argument("--capture-speed-m-s", type=float, default=.001)
    parser.add_argument("--translation-force-n", type=float)
    parser.add_argument("--attitude-force-n", type=float)
    parser.add_argument('--attitude-moment-arm-m',type=float)
    parser.add_argument("--specific-impulse-s", type=float)
    parser.add_argument("--rate-tolerance-rad-s", type=float, default=1e-5)
    parser.add_argument("--detumbling-dwell-s", type=float, default=30.)
    parser.add_argument("--target-perigee-altitude-m", type=float, default=80000.)
    parser.add_argument("--entry-interface-altitude-m", type=float, default=120000.)
    parser.add_argument("--gravity-gradient", action="store_true", default=None, help="Include assembly gravity-gradient torque; default inherits approach target setting")
    args = parser.parse_args(argv)
    if not args.assume_attitude_matched and args.input_component_bound is None:
        parser.error("--input-component-bound is required for actual attitude replay and approach fuel accounting")
    if not args.assume_attitude_matched and args.approach_point_is_capture_reference:
        parser.error("Actual attitude replay requires center-of-mass approach trajectories")
    if args.terminal_execution is not None and args.approach_point_is_capture_reference:
        parser.error("A physical terminal handoff is a literal center-of-mass record")
    path = args.approach_result / MissionResultStore.filename if args.approach_result.is_dir() else args.approach_result
    try:
        approach = MissionResultStore.load(path)
        design=None
        defaults=dict(target_mass_kg=1000.,chaser_dry_mass_kg=8.,chaser_fuel_kg=4.,
            chaser_body_radius_m=.15,gripper_offset_m=.16,capture_reach_m=.01,
            translation_force_n=10.,attitude_force_n=.2,attitude_moment_arm_m=.1,specific_impulse_s=300.)
        if approach.target.get('shape')=='composite_spacecraft':
            from orbexa.core.geometry.composite import SpacecraftGeometry
            from orbexa.simulation.completion.configuration.design import SpacecraftMissionDesign
            record=approach.metadata.get('scenario',{}).get('vehicle_design')
            if not isinstance(record,dict):
                raise ValueError('Composite completion requires its recorded vehicle_design; installed bus/fuel/nozzle properties cannot be inferred from appearance')
            design=SpacecraftMissionDesign.from_dict(SpacecraftGeometry.from_dict(approach.target['geometry']),record)
            defaults.update(target_mass_kg=design.target_mass_kg,chaser_dry_mass_kg=design.chaser.dry_mass_kg,
                chaser_fuel_kg=design.chaser.fuel_kg,chaser_body_radius_m=design.chaser.body_radius_m,
                gripper_offset_m=design.chaser.gripper_offset_m,capture_reach_m=design.arm_reach_m,
                translation_force_n=design.translation_force_n,attitude_force_n=design.attitude_nozzle_force_n,
                attitude_moment_arm_m=design.attitude_moment_arm_m,specific_impulse_s=design.specific_impulse_s)
        for key,value in defaults.items():
            supplied=getattr(args,key)
            if supplied is None:
                setattr(args,key,value)
            elif design is not None and supplied!=value:
                raise ValueError(f'{key} conflicts with the installed vehicle design recorded during approach')
        handoff = None
        if args.terminal_execution is not None:
            from orbexa.simulation.approach.terminal.handoff import PhysicalTerminalHandoff
            handoff = PhysicalTerminalHandoff.from_files(path, args.terminal_execution)
        # Use the physical target inertia already declared by its approach;
        # target mass is a separately supplied representative property.
        inertia = approach.target.get("inertia_kg_m2")
        if inertia is None:
            parser.error("Approach must declare physical target inertia; prescribed Euler records cannot supply it")
        chaser = ChaserPhysicalProperties(args.chaser_dry_mass_kg, args.chaser_fuel_kg,
                                          args.chaser_body_radius_m, args.gripper_offset_m)
        capture_options = CaptureOptions(arm_reach_m=args.capture_reach_m, max_relative_speed_m_s=args.capture_speed_m_s,
                assume_attitude_matched=args.assume_attitude_matched,
                assume_approach_point_is_capture_reference=args.approach_point_is_capture_reference,
                gravity_gradient=args.gravity_gradient)
        detumbling = DetumblingSimulator(options=DetumblingOptions(rate_tolerance_rad_s=args.rate_tolerance_rad_s,
            dwell_s=args.detumbling_dwell_s),policy=RateDampingPolicy(20. if design is None else design.detumbling_gain_nm_s,
                compensate_gravity_gradient=design is not None))
        deorbit = DeorbitSimulator(options=DeorbitOptions(target_perigee_altitude_m=args.target_perigee_altitude_m,
            entry_interface_altitude_m=args.entry_interface_altitude_m),pointing_policy=ThrustPointingPolicy(
                .2 if design is None else design.pointing_attitude_gain_nm,
                20. if design is None else design.pointing_rate_gain_nm_s,
                compensate_gravity_gradient=design is not None),
                burn_axis_planner=None if design is None else AxialBurnPlanner(),
                burn_throttle_planner=None if design is None else FiniteBurnThrottlePlanner())
        propulsion = partial(ThrusterSystem.for_assembly, translation_force_n=args.translation_force_n,
                             attitude_force_n=args.attitude_force_n, specific_impulse_s=args.specific_impulse_s,
                             moment_arm_m=args.attitude_moment_arm_m)
        if args.assume_attitude_matched:
            args.output_dir.mkdir(parents=True, exist_ok=False)
            geometry = bind_approach_body_geometry(approach, {}, chaser,
                capture_reference=args.approach_point_is_capture_reference, terminal_handoff=handoff)
            lifecycle = MissionLifecycle(capture=CaptureModel(target=TargetPhysicalProperties(args.target_mass_kg, inertia),
                default_chaser=chaser, options=capture_options), detumbling=detumbling, deorbit=deorbit,
                propulsion_factory=propulsion)
            result = lifecycle.run(approach, terminal_handoff=handoff)
            result = replace(result, configuration=dict(result.configuration, approach_geometry_binding=geometry))
            output = MissionLifecycleStore.save(result, args.output_dir)
        else:
            store = CompleteMissionStore(args.output_dir)
            tracker = ChaserAttitudeTracker(default_chaser=chaser,
                actuators=ChaserAttitudeActuators(nozzle_force_limit_n=args.attitude_force_n,
                                                 moment_arm_m=args.attitude_moment_arm_m,
                                                 specific_impulse_s=args.specific_impulse_s))
            result = CompleteMissionRunner(options=CompleteMissionOptions(input_component_bound=args.input_component_bound,
                target_mass_kg=args.target_mass_kg, translation_force_n=args.translation_force_n,
                specific_impulse_s=args.specific_impulse_s), attitude_tracker=tracker,
                capture_options=capture_options, detumbling=detumbling, deorbit=deorbit,
                propulsion_factory=propulsion, stage_sink=store.record).complete(approach, terminal_handoff=handoff)
            output = store.finish(result)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(dict(success=result.success, message=result.message, output=str(output)), indent=2))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
