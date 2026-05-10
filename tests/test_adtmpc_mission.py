from unittest.mock import patch
from pathlib import Path

import numpy as np

from orbexa.control.mpc_controller import MPCController
from orbexa.control import rotating_body_point_velocity, rotating_docking_point
from orbexa.core.config import SimulationConfig
from orbexa.simulation.adtmpc_mission import ADTMPCMissionRunner
from orbexa.solvers import SolverResult
from orbexa.visualization.demo import DemoConfig, OrbexaDemo, ADTMPCMissionRenderer


def _fake_solve_step(
    self,
    initial_state,
    final_state,
    control_input_0,
    start_anom,
    anom_step,
    num_steps,
    dynamics,
    bounds,
    **kwargs,
):
    states = np.repeat(np.asarray(initial_state, dtype=float)[:, None], num_steps, axis=1)
    target = np.asarray(final_state, dtype=float)
    for idx in range(num_steps):
        blend = idx / max(num_steps - 1, 1)
        states[:, idx] = (1.0 - blend) * states[:, idx] + blend * target
    controls = np.zeros((3, num_steps))
    controls[0, :] = 0.01
    return SolverResult(
        success=True,
        state_trajectory=states,
        control_trajectory=controls,
        cost=1.0,
        solve_time=0.01,
        solver_info={"fake": True, "target_params": kwargs.get("target_params")},
    )


def test_config_drives_adtmpc_target_geometry_and_single_assignment():
    config = SimulationConfig.load()
    runner = ADTMPCMissionRunner(config=config)

    assert runner.target.radius == 0.8
    assert runner.target.height == 0.6
    assert runner.target.half_length == 0.3
    assert config.num_chasers == 8
    assert np.allclose(config.target.initial_angular_velocity, [-0.09, 0.225, -0.045])
    assert config.target.radius == runner.target.radius
    assert config.target.height == runner.target.height
    assert config.target.half_length == runner.target.half_length
    assert np.isclose(
        runner.target.rendezvous_radius,
        np.sqrt(0.8**2 + 0.3**2) * (1.0 + config.target.tolerance),
    )

    chaser = runner._single_chaser()[0]
    expected_azimuth = runner.target.approach_azimuth(chaser.initial_state)
    wrapped_error = np.mod(chaser.docking_azimuth - expected_azimuth + np.pi, 2.0 * np.pi) - np.pi
    assert np.isclose(wrapped_error, 0.0)
    assert np.isclose(np.linalg.norm(chaser.docking_point_body[:2]), 0.8 + config.target.docking_standoff)
    inflated_radius = runner.target.radius * (1.0 + runner.target.tolerance)
    assert 0.0 < np.linalg.norm(chaser.docking_point_body[:2]) - inflated_radius < 0.01
    assert abs(chaser.docking_point_body[2]) <= 0.8 * config.target.half_length

    eps = 1.0e-6
    point = chaser.docking_point_body
    orientation = runner.target.orientation_at(0.4)
    finite_difference = (
        rotating_docking_point(point, runner.target.orientation_at(0.4 + eps))
        - rotating_docking_point(point, runner.target.orientation_at(0.4 - eps))
    ) / (2.0 * eps)
    analytic = rotating_body_point_velocity(
        point,
        orientation,
        runner.target.angular_velocity_at(0.4),
    )
    assert np.allclose(analytic, finite_difference, atol=1.0e-8)


def test_single_adtmpc_mission_keeps_truth_fixed_while_belief_smid_updates():
    runner = ADTMPCMissionRunner(
        solver_backend="gekko",
        approximation="nonlinear",
        horizon_steps=4,
        actuation_steps=1,
        smid_window=2,
    )

    with patch.object(MPCController, "solve_step", _fake_solve_step):
        result = runner.run(mission="single", steps=2)

    assert result.truth["eccentricity"] == 0.18
    assert result.truth["alpha"] == 2.0e-7
    assert result.truth["beta"] == 4.5e-7
    assert result.initial_belief["feasible_sets"] != {}
    for key, estimate in result.initial_belief["estimates"].items():
        lower, upper = result.initial_belief["feasible_sets"][key]
        assert lower <= estimate <= upper
    assert result.smid_records
    assert result.phase_history == ["rendezvous", "rendezvous"]
    assert result.sample_phase_history == ["rendezvous", "rendezvous", "rendezvous"]
    assert not result.success
    assert result.metadata["mission_complete"] is False
    assert result.metadata["max_mpc_updates"] == 2
    assert result.actual_trajectories["chaser_1"]
    assert result.nominal_trajectories["chaser_1"]
    assert max(result.tube_radius_history) > 0.0


def test_nonlinear_path_passes_rendezvous_and_docking_constraints_to_solver():
    calls = []

    def capture_solve(*args, **kwargs):
        calls.append(kwargs)
        return _fake_solve_step(*args, **kwargs)

    runner = ADTMPCMissionRunner(
        solver_backend="gekko",
        approximation="nonlinear",
        horizon_steps=4,
        actuation_steps=1,
        smid_window=2,
    )

    with patch.object(MPCController, "solve_step", capture_solve), patch.object(
        runner,
        "_rendezvous_goal_satisfied",
        side_effect=[False, False, False, True],
    ), patch.object(
        runner,
        "_docking_goal_satisfied",
        side_effect=[False, True],
    ):
        result = runner.run(mission="single", steps=3)

    assert result.success
    operations = [call["target_params"]["operation"] for call in calls]
    assert operations[:2] == ["rendezvous", "rendezvous"]
    assert operations[-1] == "docking"
    assert calls[0]["target_params"]["active_safety_model"] == "bounding_sphere"
    assert calls[-1]["target_params"]["active_safety_model"] == "rotating_cylinder_union"
    assert calls[-1]["target_params"]["angular_velocity"] == [-0.09, 0.225, -0.045]
    assert all(call["target_params"]["tube_radius"] >= 0.0 for call in calls)
    assert len(result.sample_phase_history) == len(result.anom_history)
    assert result.sample_phase_history[-1] == "docking"
    assert result.active_target_margins
    assert all(
        min(values) >= -runner.target_safety_tolerance
        for values in result.active_target_margins.values()
    )


def test_multi_chaser_assignments_and_pairwise_constraints_are_executable():
    calls = []

    def capture_solve(*args, **kwargs):
        calls.append(kwargs)
        return _fake_solve_step(*args, **kwargs)

    runner = ADTMPCMissionRunner(
        solver_backend="gekko",
        approximation="nonlinear",
        horizon_steps=4,
        actuation_steps=1,
        smid_window=2,
    )

    with patch.object(MPCController, "solve_step", capture_solve):
        result = runner.run(mission="multi", steps=1)

    assert not result.success
    assert result.metadata["mission_complete"] is False
    assert len(result.actual_trajectories) == runner.config.num_chasers
    docking_points = list(result.docking_points.values())
    assert len({tuple(point) for point in docking_points}) == runner.config.num_chasers
    assignments = result.metadata["chaser_assignments"]
    assert len({entry["docking_candidate_index"] for entry in assignments}) == runner.config.num_chasers
    for entry in assignments:
        point = np.asarray(entry["docking_point_body"], dtype=float)
        normal = np.asarray(entry["docking_normal_body"], dtype=float)
        assert entry["docking_surface"] == "cylinder_side"
        assert np.isclose(np.linalg.norm(normal), 1.0)
        assert abs(point[2]) <= 0.8 * runner.target.half_length
        assert np.linalg.norm(point[:2]) > runner.target.radius
    assert any(call.get("pairwise_constraints") for call in calls)
    assert result.pairwise_spacing[-1]
    assert min(result.pairwise_spacing[-1].values()) > 0.35


def test_renderer_and_from_data_regenerate_required_artifacts(tmp_path):
    runner = ADTMPCMissionRunner(
        solver_backend="scipy",
        approximation="linearized",
        horizon_steps=4,
        actuation_steps=1,
        smid_window=2,
    )

    with patch.object(MPCController, "solve_step", _fake_solve_step):
        result = runner.run(mission="single", steps=1)

    session_id = "test_session"
    output_dir = tmp_path / "results" / session_id / "single" / "nonlinear"
    data_dir = tmp_path / "data" / session_id / "single" / "nonlinear"
    data_file = runner.save_result(result, data_dir)

    with patch("orbexa.visualization.demo.shutil.which", return_value=None):
        manifest = ADTMPCMissionRenderer(output_dir).render(result, data_file=data_file)

    required = {
        "index.html",
        "trajectory.html",
        "diagnostics.html",
        "tube_trajectory.html",
        "trajectory.mp4",
        "trajectory.gif",
        "mission_data.json",
        "actual_nominal_trajectories.png",
        "tube_geometry.png",
        "control_effort.png",
        "rendezvous_margin.png",
        "docking_cylinder_margin.png",
        "tightened_rendezvous_margin.png",
        "tightened_docking_cylinder_margin.png",
        "active_target_margin.png",
        "smid_fss_widths.png",
        "parameter_estimates_vs_truth.png",
        "target_attitude.png",
        "target_angular_velocity.png",
        "multi_chaser_spacing.png",
        "manifest.json",
    }
    assert required.issubset({path.name for path in manifest.artifacts})
    manifest_payload = (output_dir / "manifest.json").read_text(encoding="utf-8")
    assert "rotating cylinder target" in manifest_payload
    trajectory_html = (output_dir / "trajectory.html").read_text(encoding="utf-8")
    assert "rotating target cylinder" in trajectory_html
    assert "rendezvous sphere" in trajectory_html
    assert "docking points" in trajectory_html

    with patch("orbexa.visualization.demo.shutil.which", return_value=None):
        manifests = OrbexaDemo(
            DemoConfig(
                output_dir=tmp_path / "results",
                data_dir=tmp_path / "data",
                session_id=session_id,
                mission="single",
                from_data=True,
            )
        ).run()

    assert manifests[0].output_dir == output_dir
    assert (tmp_path / "results" / session_id / "README.md").exists()
    assert (tmp_path / "data" / session_id / "README.md").exists()
    assert (output_dir / "index.html").exists()
    assert (data_dir / "mission_data.json").exists()

    generated_results = tmp_path / "generated_results"
    generated_data = tmp_path / "generated_data"
    with patch.object(MPCController, "solve_step", _fake_solve_step), patch(
        "orbexa.visualization.demo.shutil.which", return_value=None
    ):
        generated = OrbexaDemo(
            DemoConfig(
                output_dir=generated_results,
                data_dir=generated_data,
                session_id="fresh_session",
                mission="single",
                steps=1,
                require_primary_success=False,
            )
        ).run()

    assert generated[0].output_dir == generated_results / "fresh_session" / "single" / "nonlinear"
    assert (generated_results / "latest").is_symlink()
    assert (generated_data / "latest").is_symlink()

    multi_results = tmp_path / "multi_results"
    multi_data = tmp_path / "multi_data"
    with patch.object(MPCController, "solve_step", _fake_solve_step), patch(
        "orbexa.visualization.demo.shutil.which", return_value=None
    ), patch.object(ADTMPCMissionRenderer, "_render_gif") as gif_render:
        def write_fake_gif(result, manifest, *, speed_multiplier=2.0):
            path = manifest.output_dir / "trajectory.gif"
            path.write_bytes(b"GIF89a")
            manifest.add(path)

        gif_render.side_effect = write_fake_gif
        OrbexaDemo(
            DemoConfig(
                output_dir=multi_results,
                data_dir=multi_data,
                session_id="multi_session",
                mission="multi",
                steps=1,
                require_primary_success=False,
            )
        ).run()

    demo_gif = multi_results / "multi_session" / "demo.gif"
    assert demo_gif.is_symlink()
    assert demo_gif.resolve() == (multi_results / "multi_session" / "multi" / "nonlinear" / "trajectory.gif").resolve()
    assert "demo.gif" in (multi_results / "multi_session" / "README.md").read_text(encoding="utf-8")
