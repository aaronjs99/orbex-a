from unittest.mock import patch

import numpy as np

from orbexa.control.mpc_controller import MPCController
from orbexa.core.config import SimulationConfig
from orbexa.simulation.paper_system import PaperSystemRunner
from orbexa.solvers import SolverResult
from orbexa.visualization.demo import DemoConfig, OrbexaDemo, PaperSystemRenderer


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


def test_config_drives_paper_target_geometry_and_single_assignment():
    config = SimulationConfig.load()
    runner = PaperSystemRunner(config=config)

    assert runner.target.radius == 0.8
    assert runner.target.height == 0.6
    assert runner.target.half_length == 0.3
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
    assert np.isclose(chaser.docking_point_body[2], 0.0)


def test_single_paper_system_keeps_truth_fixed_while_belief_smid_updates():
    runner = PaperSystemRunner(
        solver_backend="gekko",
        approximation="nonlinear",
        horizon_steps=4,
        smid_window=2,
    )

    with patch.object(MPCController, "solve_step", _fake_solve_step):
        result = runner.run(mission="single", steps=2)

    assert result.truth["eccentricity"] == 0.18
    assert result.truth["alpha"] == 2.0e-7
    assert result.truth["beta"] == 4.5e-7
    assert result.initial_belief["feasible_sets"] != {}
    assert result.smid_records
    assert result.phase_history == ["rendezvous", "rendezvous"]
    assert result.sample_phase_history == ["rendezvous", "rendezvous", "rendezvous"]
    assert not result.success
    assert result.metadata["mission_complete"] is False
    assert result.metadata["max_mpc_updates"] == 2
    assert result.actual_trajectories["chaser_1"]
    assert result.nominal_trajectories["chaser_1"]


def test_nonlinear_path_passes_rendezvous_and_docking_constraints_to_solver():
    calls = []

    def capture_solve(*args, **kwargs):
        calls.append(kwargs)
        return _fake_solve_step(*args, **kwargs)

    runner = PaperSystemRunner(
        solver_backend="gekko",
        approximation="nonlinear",
        horizon_steps=4,
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

    runner = PaperSystemRunner(
        solver_backend="gekko",
        approximation="nonlinear",
        horizon_steps=4,
        smid_window=2,
    )

    with patch.object(MPCController, "solve_step", capture_solve):
        result = runner.run(mission="multi", steps=1)

    assert not result.success
    assert result.metadata["mission_complete"] is False
    assert len(result.actual_trajectories) == 3
    docking_points = list(result.docking_points.values())
    assert len({tuple(point) for point in docking_points}) == 3
    assignments = result.metadata["chaser_assignments"]
    assert len({entry["docking_candidate_index"] for entry in assignments}) == 3
    for entry in assignments:
        point = np.asarray(entry["docking_point_body"], dtype=float)
        normal = np.asarray(entry["docking_normal_body"], dtype=float)
        assert entry["docking_surface"] == "cylinder_side"
        assert np.isclose(np.linalg.norm(normal), 1.0)
        assert np.isclose(point[2], 0.0)
        assert np.linalg.norm(point[:2]) > runner.target.radius
    assert any(call.get("pairwise_constraints") for call in calls)
    assert result.pairwise_spacing[-1]
    assert min(result.pairwise_spacing[-1].values()) > 0.35


def test_renderer_and_from_data_regenerate_required_artifacts(tmp_path):
    runner = PaperSystemRunner(
        solver_backend="scipy",
        approximation="linearized",
        horizon_steps=4,
        smid_window=2,
    )

    with patch.object(MPCController, "solve_step", _fake_solve_step):
        result = runner.run(mission="single", steps=1)

    output_dir = tmp_path / "results" / "single" / "nonlinear"
    data_dir = tmp_path / "data" / "single" / "nonlinear"
    data_file = runner.save_result(result, data_dir)

    with patch("orbexa.visualization.demo.shutil.which", return_value=None):
        manifest = PaperSystemRenderer(output_dir).render(result, data_file=data_file)

    required = {
        "index.html",
        "trajectory.html",
        "trajectory.mp4",
        "mission_data.json",
        "actual_nominal_trajectories.png",
        "tube_geometry.png",
        "control_effort.png",
        "rendezvous_margin.png",
        "docking_cylinder_margin.png",
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
                mission="single",
                from_data=True,
            )
        ).run()

    assert manifests[0].output_dir == output_dir
    assert (output_dir / "index.html").exists()
    assert (data_dir / "mission_data.json").exists()
