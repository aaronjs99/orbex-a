"""Sensed nonlinear mission composition and committed live projections."""

from __future__ import annotations

import math
from dataclasses import asdict

from orbexa.common.layout import PACKAGE_ROOT
from orbexa.evidence.artifacts import _sha, _write_json
from orbexa.evidence.native import native_controller as _native_controller


def _execute_nonlinear_job(settings, parameters, directory, messages, run_event):
    """Fresh observed nonlinear run, durable proofs and ordinary finite capture."""
    from dataclasses import replace

    import numpy as np
    from orbexa.common.intervals import IntervalBox
    from orbexa.evidence.projections.studio import StudioDataset
    from orbexa.evidence.records.archive import NonlinearProofArchive
    from orbexa.evidence.records.results import MissionResultStore
    from orbexa.mission.lifecycle.configuration.design import (
        SpacecraftMissionDesign,
    )
    from orbexa.mission.lifecycle.coordinator import CompleteMissionStore
    from orbexa.mission.lifecycle.nonlinear.coordinator import (
        complete_nonlinear_mission,
    )
    from orbexa.mission.profile import (
        IndependentTruthCase,
        NonlinearStudyOptions,
    )
    from orbexa.mission.runner import NonlinearMissionRunner
    from orbexa.models.configuration.model import SimulationConfig

    cfg = SimulationConfig.load(str(settings.config_path))
    if cfg.solver.remote:
        raise ValueError(
            "The live service requires local GEKKO/IPOPT; remote solvers are not enabled"
        )
    cfg.num_chasers = 8
    cfg.__post_init__()
    design = SpacecraftMissionDesign.from_json(settings.spacecraft_path)
    cfg.target.docking_standoff = design.com_standoff_m
    cfg.target.inertia = np.asarray(
        design.geometry.metadata["mass_properties"]["inertia_kg_m2"]
    )
    options = replace(
        NonlinearStudyOptions(),
        release_distance_m=parameters.release_distance_m,
        release_pitch_m=parameters.cluster_spacing_m,
        nominal_spin_deg_s=parameters.target_spin_deg_s,
    )
    # These physical realization values are fixed prospectively, independent
    # of the observed controller prior and the success/failure of this run.
    case = IndependentTruthCase(
        "live-prospective-study-v1",
        0.0012,
        0.026,
        0.036,
        parameters.target_spin_deg_s,
        3817,
    )
    prior = IntervalBox.from_center_radius(
        (0.0, 0.0, math.radians(parameters.target_spin_deg_s)), (0.002, 0.002, 0.008)
    )
    inputs = {
        name: str(getattr(settings, name))
        for name in (
            "config_path",
            "spacecraft_path",
        )
    }
    _write_json(
        directory / "source-inputs.json",
        {
            "paths": inputs,
            "sha256": {name: _sha(path) for name, path in inputs.items()},
            "runtime_python_sha256": {
                str(path.relative_to(PACKAGE_ROOT)): _sha(path)
                for path in PACKAGE_ROOT.rglob("*.py")
            },
            "parameters": parameters.as_dict(),
            "maximum_updates": settings.maximum_updates,
            "origin": "fresh_nonlinear_parameters",
            "execution_model": "sensed_nonlinear",
            "nonlinear_study_options": options.as_dict(),
            "independent_truth_case": asdict(case),
            "initial_spin_prior": prior.as_dict(),
            "scope": "Shared declared nonlinear physics/sensor/safety profile; requested study knobs and nonzero observed spin prior; no flight qualification",
        },
    )
    controller_factory = _native_controller(directory, messages)
    with NonlinearProofArchive(
        directory / "nonlinear-certificates.jsonl.zst"
    ) as archive:

        def progress(event):
            projection = event.get("projection")
            if projection is None and event.get("result") is not None:
                projection = StudioDataset(event["result"]).to_dict()
            if projection is None:
                raise ValueError(
                    "Nonlinear runner must provide an actual committed projection"
                )
            _write_json(directory / "committed-projection.json", projection)
            archive.flush()
            messages.put(
                {
                    "kind": "ready" if event.get("kind") == "start" else "committed",
                    "projection": projection,
                    "update": event.get(
                        "update", max(0, len(projection.get("q", [])) - 1)
                    ),
                    "anomaly": projection["q"][-1],
                    "phase": projection.get("phases", ["unknown"])[-1],
                    "message": "Actual sensed nonlinear mission states",
                }
            )
            if not run_event.is_set():
                messages.put({"kind": "paused"})
                run_event.wait()
                messages.put({"kind": "running"})

        def checkpoint(update, runner, outcome):
            archive.flush()
            if archive.holds != len(runner.recorder.hold_records):
                raise ValueError(
                    "Durable completed-hold archive does not match the committed mission index"
                )
            # Both owners confirmed these lists are evidence-only. Their
            # full records already have durable archive receipts at this point.
            runner.executor.journal.clear()
            if runner.executor.runtime is not None:
                runner.executor.runtime.events.clear()
            _write_json(
                directory / "archive-progress.json",
                {
                    "update": update,
                    "entries": archive.entries,
                    "completed_holds": archive.holds,
                    "event_kinds": archive.kinds,
                    "retained_executor_holds": len(runner.executor.journal),
                    "retained_runtime_events": 0
                    if runner.executor.runtime is None
                    else len(runner.executor.runtime.events),
                    "execution_exact_q": str(runner.truth_target.current_q),
                    "proof_scope": "Producer witnesses streamed before motion",
                },
            )

        runner = NonlinearMissionRunner(
            cfg,
            design,
            case=case,
            options=options,
            progress_sink=progress,
            certificate_sink=archive.append,
            checkpoint_sink=checkpoint,
            controller_factory=controller_factory,
            initial_target_spin_prior=prior,
            streamed_hold_index=True,
            retain_journal=False,
        )
        approach = runner.run(mission="multi", steps=settings.maximum_updates)
        source = archive.close()
    # Publish the closed archive receipt after recording; no archive identity
    # depends on the result that later names it.
    approach.metadata["proof_archive"] = source.as_dict()
    approach.metadata["live_execution_route"] = "sensed_nonlinear"
    manifest = directory / "snapshot-manifest.json"
    approach.metadata["live_source_binding"] = {
        "source_inputs_sha256": _sha(directory / "source-inputs.json"),
        "snapshot_manifest_sha256": _sha(manifest) if manifest.is_file() else None,
        "frozen_package": str(PACKAGE_ROOT),
        "scope": "Closed archive receipt and immutable job source/input manifests; recorded after production to avoid self-reference",
    }
    MissionResultStore.save(approach, directory / "approach")
    projection = StudioDataset(approach).to_dict()
    _write_json(directory / "committed-projection.json", projection)
    messages.put(
        {
            "kind": "committed",
            "projection": projection,
            "update": len(approach.anom_history) - 1,
            "anomaly": approach.anom_history[-1],
            "phase": approach.sample_phase_history[-1],
        }
    )
    store = CompleteMissionStore(directory / "completion")

    def stage_sink(stage, result):
        store.record(stage, result)
        messages.put({"kind": "stage", "message": "Physical completion: " + stage})

    completion = complete_nonlinear_mission(
        approach, nonlinear_runner=runner, hold_records=source, stage_sink=stage_sink
    )
    # The early incomplete result deliberately performs no post-approach stage;
    # retain its original source in the same result directory nevertheless.
    if not approach.success or not approach.metadata.get("mission_complete", False):
        store.record("approach", approach)
    store.finish(completion)
    payload = {
        "kind": "finished",
        "success": bool(completion.success),
        "message": completion.message,
        "projection": projection,
    }
    if completion.lifecycle is not None:
        from orbexa.figures.renderers.lifecycle import LifecycleDataset

        payload["lifecycle"] = LifecycleDataset(completion.lifecycle).to_dict()
        _write_json(directory / "lifecycle-projection.json", payload["lifecycle"])
    _write_json(
        directory / "worker-result.json",
        {
            key: value
            for key, value in payload.items()
            if key not in ("projection", "lifecycle")
        },
    )
    messages.put(payload)
