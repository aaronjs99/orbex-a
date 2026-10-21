"""Explicit saved analytical continuation with authenticated source accounting."""
from __future__ import annotations

import gzip
import json
import math
from fractions import Fraction
from orbexa.layout import PACKAGE_ROOT
from orbexa.simulation.approach.arrival.composition import _arrival_policy
from orbexa.simulation.recording.artifacts import _sha, _write_json
from orbexa.simulation.recording.native import native_controller as _native_controller
from orbexa.simulation.service.jobs.resume import _append_resume_journal, _resume_source


def _execute_historical_analytical_job(settings, parameters, directory, messages, run_event):
    """Explicit retained analytical restart; it never supplies a fresh run."""
    from dataclasses import replace
    import numpy as np
    from orbexa.core.configuration.model import SimulationConfig
    from orbexa.estimation.adaptation.belief import ParameterBelief
    from orbexa.simulation.approach.api import ADTMPCMissionRunner
    from orbexa.simulation.approach.deployment.component import CompositeDeploymentOptions
    from orbexa.simulation.completion.coordinator import CompleteMissionStore
    from orbexa.simulation.approach.records.results import MissionResultStore
    from orbexa.simulation.approach.physical.scenario import ContinuousAdmissionOptions, PhysicalApproachOptions, PhysicalApproachScenario
    from orbexa.simulation.completion.configuration.design import SpacecraftMissionDesign
    from orbexa.visualization.data.studio import StudioDataset
    from orbexa.simulation.approach.records.checkpoint import MissionCheckpoint, physical_target_from_snapshot, plain
    if not settings.historical_analytical_resume or settings.resume_path is None:
        raise ValueError('Historical analytical execution requires its explicitly selected saved source')
    TrackedController = _native_controller(directory, messages)
    inputs = {name: str(getattr(settings, name)) for name in
        ("config_path", "prior_path", "admission_path", "deployment_path", "spacecraft_path")}
    _write_json(directory / "source-inputs.json", dict(paths=inputs,
        sha256={name: _sha(path) for name, path in inputs.items()},
        runtime_python_sha256={str(path.relative_to(PACKAGE_ROOT)): _sha(path)
            for path in PACKAGE_ROOT.rglob("*.py")},
        parameters=parameters.as_dict(), maximum_updates=settings.maximum_updates,
        origin='historical_analytical_checkpoint', execution_model='historical_analytical_resume',
        resume_source=None if settings.resume_path is None else str(settings.resume_path),
        transfer_duration_rule=".18*sqrt(release_distance_m/300)rad; unchanged hardware/input caps"))
    cfg = SimulationConfig.load(str(settings.config_path))
    if cfg.solver.remote:
        raise ValueError('The live service requires local GEKKO/IPOPT; remote solvers are not enabled')
    cfg.num_chasers = 8
    cfg.__post_init__()
    prior = ParameterBelief(**json.loads(settings.prior_path.read_text(encoding="utf-8")))
    admission = ContinuousAdmissionOptions(**json.loads(settings.admission_path.read_text(encoding="utf-8")))
    deployment = CompositeDeploymentOptions(**json.loads(settings.deployment_path.read_text(encoding="utf-8")))
    buffer = parameters.cluster_spacing_m - cfg.mpc.min_chaser_separation - 2*deployment.reference_tube_radius_m
    if buffer <= 0:
        raise ValueError("Requested spacing cannot fit the unchanged pair/tube limits")
    deployment = replace(deployment, carrier_center_lvlh_m=(0., 0., parameters.release_distance_m),
        carrier_transfer_duration_rad=.18*math.sqrt(parameters.release_distance_m/300.), separation_buffer_m=buffer)
    design = SpacecraftMissionDesign.from_json(settings.spacecraft_path)
    cfg.target.docking_standoff = design.com_standoff_m
    cfg.target.inertia = np.asarray(design.geometry.metadata['mass_properties']['inertia_kg_m2'])
    spin = (0., 0., math.radians(parameters.target_spin_deg_s))
    end = (max(settings.maximum_updates, cfg.num_update_steps) + max(cfg.mpc.horizon_steps.values()) + 2) * cfg.anom_step
    checkpoint, target_scope, additional_updates = None, None, settings.maximum_updates
    if settings.resume_path is None:
        target = design.target(cfg, initial_body_spin_rad_s=spin, anomaly_domain=(0., end), gravity_gradient=True)
    else:
        from orbexa.simulation.approach.arrival.policy import exact_outer_step
        _resume_source(settings.resume_path, parameters)
        manifest = json.loads((directory / 'snapshot-manifest.json').read_bytes())
        for name in ('resume_path', 'resume_journal'):
            row = manifest['inputs'][name]
            if _sha(directory / row['frozen']) != row['sha256']:
                raise ValueError('Frozen resume input identity changed before checkpoint construction')
        checkpoint = MissionCheckpoint.from_file(settings.resume_path, source_context={
            'source_reference': manifest['inputs']['resume_path']['original']})
        saved = checkpoint.source_record
        target = physical_target_from_snapshot(saved['target'])
        guard = saved['metadata']['continuous_execution_certificate']
        target_scope = guard['target_bound_context']['scope']
        remaining_cap = saved['metadata']['checkpoint_context']['configuration']['num_update_steps'] - checkpoint.committed_updates
        remaining_domain = ((Fraction(target.anomaly_domain[1])-Fraction(guard['exact_end_anomaly']))
            // exact_outer_step(cfg)) - max(cfg.mpc.horizon_steps.values()) - 1
        additional_updates = min(settings.maximum_updates, remaining_cap, remaining_domain)
        if additional_updates < 1:
            raise ValueError('No new update fits the original total cap and target preview domain')
    options = PhysicalApproachOptions(body_radius_m=design.chaser.body_radius_m,
        com_standoff_m=design.com_standoff_m, initial_body_spin_rad_s=spin, gravity_gradient=True)
    with gzip.open(directory / "continuous-certificates.jsonl.gz", "wt", encoding="utf-8") as journal:
        def certificate_sink(record):
            journal.write(json.dumps(record, allow_nan=False) + "\n")
        profile = PhysicalApproachScenario(cfg, initial_belief=prior, maximum_updates=additional_updates,
            options=options, target=target, deployment_options=deployment, admission=admission,
            certificate_sink=certificate_sink, target_bound_scope=target_scope,
            vehicle_design=design.as_dict(), controller_factory=TrackedController)
        profile.scenario.arrival_policy = _arrival_policy(profile, prior, admission, directory)
        if checkpoint is not None:
            checkpoint.validate_scenario(profile.scenario, 'multi')
            checkpoint.save(directory / 'source-checkpoint.json')
            prefix = _append_resume_journal(checkpoint, settings.resume_path.parent / 'continuous-certificates.jsonl.gz', journal)
            journal.flush()
            if any(_sha(directory / manifest['inputs'][name]['frozen']) != manifest['inputs'][name]['sha256']
                    for name in ('resume_path', 'resume_journal')):
                raise ValueError('Frozen resume identity changed during prefix validation')
            _write_json(directory / 'resume-source.json', dict(checkpoint_identity=checkpoint.identity,
                source_record_sha256=checkpoint.source_record_sha256,
                source_canonical_sha256=checkpoint.source_canonical_sha256,
                frozen_inputs={name:manifest['inputs'][name] for name in ('resume_path', 'resume_journal')},
                prefix=prefix, additional_update_cap=additional_updates,
                original_total_update_cap=remaining_cap+checkpoint.committed_updates,
                target_anomaly_domain=list(target.anomaly_domain),
                scope='Numerical continuation with the public strict checkpoint; original failed attempts/source history retained. Cold plans and dense adaptation-window reset are explicit.'))
        def progress(event):
            kind = event.get("kind", event.get("type", "committed"))
            projection = event.get("projection")
            if projection is None and event.get("result") is not None:
                projection = StudioDataset(event["result"]).to_dict()
            if projection is None:
                raise ValueError("Runner live callback must provide committed projection or result")
            _write_json(directory / "committed-projection.json", projection)
            journal.flush()
            payload = dict(kind="ready" if kind == "start" else "committed", projection=projection,
                update=event.get("update", max(0, len(projection.get("q", []))-1)),
                anomaly=projection["q"][-1], phase=projection.get("phases", ["unknown"])[-1])
            messages.put(payload)
            if not run_event.is_set():
                messages.put(dict(kind="paused"))
                run_event.wait()
                messages.put(dict(kind="running"))
        profile.scenario.progress_sink = progress
        runner = ADTMPCMissionRunner(scenario=profile.scenario)
        approach = (runner.run(mission="multi", steps=additional_updates) if checkpoint is None
            else runner.resume(checkpoint, steps=additional_updates))
        MissionResultStore.save(approach, directory / "approach")
        # Close the complete source journal before authenticating and replaying
        # its interval proofs for conservative held-command fuel accounting.
        journal.close()
        projection = StudioDataset(approach).to_dict()
        _write_json(directory / "committed-projection.json", projection)
        messages.put(dict(kind="committed", projection=projection, update=len(approach.anom_history)-1,
            anomaly=approach.anom_history[-1], phase=approach.sample_phase_history[-1]))
        store = CompleteMissionStore(directory / "completion")
        def stage_sink(stage, result):
            store.record(stage, result)
            messages.put(dict(kind="stage", message="Physical completion: " + stage))
        bound = max(abs(float(b[k])) for b in cfg.mpc.input_bounds for k in ("lower", "upper"))
        accountant = None
        if approach.success and approach.metadata.get('mission_complete') is True:
            from orbexa.simulation.completion.accounting.held import HeldInputPropellantAccountant
            raw_path = directory / 'approach' / MissionResultStore.filename
            journal_path = directory / 'continuous-certificates.jsonl.gz'
            accountant = HeldInputPropellantAccountant.from_files(raw_path, [journal_path],
                expected_approach_sha256=_sha(raw_path), expected_journal_sha256=[_sha(journal_path)],
                producer_package_root=PACKAGE_ROOT)
            messages.put(dict(kind='stage', message='Verifying held-command propellant bounds'))
        completion = design.completion_runner(input_component_bound=bound, stage_sink=stage_sink,
            translation_accountant=accountant).complete(approach)
        store.finish(completion)
        payload = dict(kind="finished", success=bool(completion.success), message=completion.message,
            projection=projection)
        if completion.lifecycle is not None:
            from orbexa.visualization.renderers.lifecycle import LifecycleDataset
            payload['lifecycle'] = LifecycleDataset(completion.lifecycle).to_dict()
            _write_json(directory / "lifecycle-projection.json", payload['lifecycle'])
        _write_json(directory / "worker-result.json", {key: value for key, value in payload.items()
            if key not in ('projection', 'lifecycle')})
        messages.put(payload)
