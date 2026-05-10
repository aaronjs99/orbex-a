import numpy as np

from orbexa.control.constraints import (
    CylinderConstraint,
    rendezvous_margin,
    rotating_docking_point,
)
from orbexa.control.dynamic_tube_model import (
    calc_delta,
    input_tightening_from_profile,
    propagate_tube_profile,
)
from orbexa.control.linearization import (
    linearize_cylinder_constraint,
    linearize_rendezvous_constraint,
)
from orbexa.core.config import SimulationConfig
from orbexa.core.dynamics import orbital_ellp_drag
from orbexa.estimation.adaptor import SMIDAdaptor


def test_drag_aware_dynamics_reduce_to_cwh_appendix_case():
    mean_motion = 0.001
    mu = 3.986004418e14
    semi_major_axis = (mu / mean_motion**2) ** (1.0 / 3.0)
    anom_step = 0.1

    drag_matrices, _, _ = orbital_ellp_drag(
        anom_step=anom_step,
        mean_motion=mean_motion,
        eccentricity=0.0,
        alpha=0.0,
        beta=0.0,
        mu=mu,
        semi_major_axis=semi_major_axis,
    )
    A_func, B_func, _, _, d_func = drag_matrices

    expected = np.array(
        [
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 2, 0],
            [0, 3, 0, -2, 0, 0],
            [0, 0, -1, 0, 0, 0],
        ],
        dtype=float,
    )

    np.testing.assert_allclose(np.asarray(A_func(0.0), dtype=float), expected)
    np.testing.assert_allclose(B_func()[3:, :], np.eye(3))
    np.testing.assert_allclose(np.asarray(d_func(0.0), dtype=float), np.zeros(6))


def test_config_normalizes_weights_and_exposes_paper_runtime_params():
    config = SimulationConfig.load()

    assert config.mpc.Q.shape == (6, 6)
    assert config.mpc.R.shape == (3, 3)

    dynamics_params = config.dynamics_params()
    assert dynamics_params["mean_motion"] > 0
    assert dynamics_params["specific_angular_momentum"] > 0

    target_params = config.target.collision_params(operation="rendezvous")
    assert target_params["rendezvous_radius"] > 0


def test_rendezvous_and_docking_constraints_match_paper_union_logic():
    assert rendezvous_margin(np.array([2.0, 0.0, 0.0]), target_radius=1.0) > 0
    assert rendezvous_margin(np.array([0.5, 0.0, 0.0]), target_radius=1.0) < 0

    cylinder = CylinderConstraint(
        radius=1.0,
        half_length=2.0,
        orientation=np.zeros(3),
    )
    assert not cylinder.is_satisfied(np.array([0.5, 0.0, 0.0]))
    assert cylinder.is_satisfied(np.array([1.5, 0.0, 0.0]))
    assert cylinder.is_satisfied(np.array([0.0, 0.0, 2.5]))

    docking_point = rotating_docking_point(np.array([1.0, 0.0, 0.0]), np.zeros(3))
    np.testing.assert_allclose(docking_point, np.array([1.0, 0.0, 0.0]))


def test_tube_uncertainty_bound_is_zero_for_identical_feasible_set():
    delta = calc_delta(
        t=0.0,
        t_p=0.0,
        x=np.ones(6),
        anom_step=0.1,
        mean_motion=0.001,
        e_range=(0.1, 0.1),
        a_range=(1.0e-7, 1.0e-7),
        b_range=(2.0e-7, 2.0e-7),
    )

    np.testing.assert_allclose(delta, np.zeros(3))


def test_tube_profile_expands_with_larger_feasible_set():
    narrow = propagate_tube_profile(
        start_anom=0.0,
        num_steps=8,
        anom_step=0.1,
        mean_motion=0.001,
        lambda_gain=[0.1, 0.1, 0.1],
        alpha=[0.2, 0.2, 0.2],
        phi_0=[0.1, 0.1, 0.1],
        eccentricity_range=(0.1, 0.1),
        aRange=(1.0e-7, 1.0e-7),
        bRange=(1.0e-7, 1.0e-7),
    )
    wide = propagate_tube_profile(
        start_anom=0.0,
        num_steps=8,
        anom_step=0.1,
        mean_motion=0.001,
        lambda_gain=[0.1, 0.1, 0.1],
        alpha=[0.2, 0.2, 0.2],
        phi_0=[0.1, 0.1, 0.1],
        eccentricity_range=(0.0, 0.3),
        aRange=(0.0, 5.0e-7),
        bRange=(0.0, 5.0e-7),
    )

    assert wide.max_position_radius >= narrow.max_position_radius
    input_tightening = input_tightening_from_profile(wide, [0.1, 0.1, 0.1])
    assert input_tightening.shape == (3,)
    assert np.all(input_tightening >= 0.0)


def test_linearized_constraints_match_nonlinear_margin_at_expansion_point():
    rendezvous = linearize_rendezvous_constraint(
        np.array([2.0, 0.0, 0.0]),
        target_radius=1.0,
    )
    assert np.isclose(rendezvous.margin(np.array([2.0, 0.0, 0.0])), 1.0)

    cylinder = CylinderConstraint(
        radius=1.0,
        half_length=2.0,
        orientation=np.zeros(3),
    )
    linearized = linearize_cylinder_constraint(
        np.array([1.5, 0.0, 0.0]),
        cylinder,
        active="radial",
    )
    radial_margin, _ = cylinder.margins(np.array([1.5, 0.0, 0.0]))
    assert np.isclose(linearized.margin(np.array([1.5, 0.0, 0.0])), radial_margin)


def _synthetic_smid_window():
    config = SimulationConfig.load()
    dt = config.anom_step
    truth = {"eccentricity": 0.18, "alpha": 2.0e-7, "beta": 4.5e-7}
    context = {
        "mean_motion": config.orbit.mean_motion,
        "mu": config.orbit.mu,
        "semi_major_axis": config.orbit.semi_major_axis,
        "time_periapsis": config.orbit.time_periapsis,
    }
    params = dict(context)
    params.update(truth)
    params["specific_angular_momentum"] = np.sqrt(
        config.orbit.mu
        * config.orbit.semi_major_axis
        * (1.0 - truth["eccentricity"] ** 2)
    )
    matrices, _, _ = orbital_ellp_drag(anom_step=dt, **params)
    A_func, B_func, _, _, d_func = matrices
    state = np.asarray(config.orbit.initial_conditions, dtype=float).copy()
    control = np.array([0.01, -0.005, 0.002], dtype=float)
    states = [state.copy()]
    controls = []
    anom = 0.0
    for _ in range(4):
        A_val = np.asarray(A_func(anom, 0.0), dtype=float)
        d_val = np.asarray(d_func(anom, 0.0), dtype=float)
        state = state + (A_val @ state + B_func() @ control + d_val) * dt
        states.append(state.copy())
        controls.append(control.copy())
        anom += dt
    return context, np.asarray(states), np.asarray(controls), dt


def test_real_smid_shrinks_verified_feasible_set():
    context, states, controls, dt = _synthetic_smid_window()
    initial_fss = {
        "eccentricity": (0.02, 0.38),
        "alpha": (0.0, 5.5e-7),
        "beta": (0.0, 8.55e-7),
    }
    estimates = {key: np.mean(value) for key, value in initial_fss.items()}

    adaptor = SMIDAdaptor(error_bound=0.15, max_iter=90)
    fss, updated_estimates, record = adaptor.update(
        feasible_sets=initial_fss,
        estimates=estimates,
        states=states,
        controls=controls,
        start_anom=0.0,
        anom_step=dt,
        dynamics_context=context,
    )

    assert record.accepted
    assert record.verification_error <= 0.15
    assert any(
        fss[key][1] - fss[key][0] < initial_fss[key][1] - initial_fss[key][0]
        for key in initial_fss
    )
    assert all(fss[key][0] <= updated_estimates[key] <= fss[key][1] for key in fss)


def test_failed_smid_verification_preserves_previous_fss():
    context, states, controls, dt = _synthetic_smid_window()
    initial_fss = {
        "eccentricity": (0.02, 0.38),
        "alpha": (0.0, 5.5e-7),
        "beta": (0.0, 8.55e-7),
    }
    estimates = {key: np.mean(value) for key, value in initial_fss.items()}

    adaptor = SMIDAdaptor(error_bound=1.0e-12, max_iter=60)
    fss, updated_estimates, record = adaptor.update(
        feasible_sets=initial_fss,
        estimates=estimates,
        states=states,
        controls=controls,
        start_anom=0.0,
        anom_step=dt,
        dynamics_context=context,
    )

    assert not record.accepted
    assert fss == initial_fss
    assert updated_estimates == estimates
