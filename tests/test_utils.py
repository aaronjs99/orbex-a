import pytest
import numpy as np
import os
import time
from pathlib import Path
from queue import Queue
from orbexa.utils import io_utils, math_utils


class TestMathUtils:
    def test_calc_distance(self):
        p1 = np.array([1, 2, 3])
        p2 = np.array([4, 6, 8])
        dist = math_utils.calc_distance(p1, p2)
        expected = np.linalg.norm(p1 - p2)
        assert np.isclose(dist, expected)

    def test_gen_init_state(self):
        x0 = math_utils.gen_init_state(num_chasers=2, rX=10, rV=1)
        assert x0.shape == (12,)
        assert np.all(np.isfinite(x0))

    def test_gen_skew_sym_mat(self):
        v = np.array([1.0, 2.0, 3.0])
        skew = math_utils.gen_skew_sym_mat(v)
        expected = np.array([[0, -3, 2], [3, 0, -1], [-2, 1, 0]])
        assert np.allclose(skew, expected)
        assert np.allclose(skew, -skew.T)

    def test_tait_bryan_to_rotation_matrix(self):
        angles = np.array([0, 0, 0])
        rot = math_utils.tait_bryan_to_rotation_matrix(angles)
        assert np.allclose(rot, np.eye(3))

        # Test 90 deg z-rotation
        angles = np.array([0, 0, np.pi / 2])
        rot = math_utils.tait_bryan_to_rotation_matrix(angles)
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        assert np.allclose(rot, expected, atol=1e-7)

    def test_discretize_signature(self):
        dt = 0.1
        A = np.eye(2)
        B = np.ones((2, 1))
        Ad, Bd, Cd, Dd = math_utils.discretize(dt, A, B)
        assert Ad.shape == (2, 2)
        assert Bd.shape == (2, 1)
        assert Cd.shape == (1, 2)
        assert Dd.shape == (1, 1)

    def test_cylinder_inside_constraint(self):
        limits = {"r_T": 1.0, "l_T": 2.0, "tolerance": 0.0}
        r_in = np.array([0.5, 0.5, 1.0])
        val = math_utils.cylinder_inside_constraint(r_in, limits)
        assert val <= 0
        r_out_rad = np.array([1.1, 0.0, 0.0])
        val = math_utils.cylinder_inside_constraint(r_out_rad, limits)
        assert val > 0

    def test_pyramidal_constraint(self):
        x0 = np.array([10, 0, 0])
        xf = np.array([1, 0, 0])
        mu = {"mu_x": 0.1, "mu_y": 0.1}
        A, B, pol = math_utils.pyramidal_constraint(x0, xf, mu)
        assert A.ndim == 2
        assert A.shape[1] == 3
        assert B.ndim == 1
        assert A.shape[0] == B.shape[0]

    def test_calc_local_occlusion_cost(self):
        x = np.array([0, 0, 0])
        X = [np.array([1, 0, 0]), np.array([-1, 0, 0])]
        w = [1.0, 1.0]
        v = 100.0
        cost = math_utils.calc_local_occlusion_cost(x, w, v, X)
        expected_cost = (100.0 * 81.0 - 2.0) / (2 * 4)
        assert np.isclose(cost, expected_cost)

    def test_pyramidal_constraint_geometry(self):
        x0 = np.array([10, 0, 0])
        xf = np.array([1, 0, 0])
        mu = {"mu_x": 1.0, "mu_y": 1.0}
        A, B, _ = math_utils.pyramidal_constraint(x0, xf, mu)
        x_mid = (x0 + xf) / 2.0
        residuals = A @ x_mid - B
        assert np.all(residuals <= 1e-7)
        x_out = np.array([5.0, 10.0, 0.0])
        residuals_out = A @ x_out - B
        assert np.any(residuals_out > 1e-7)


class TestIOUtils:
    def test_convert_numpy_to_list(self):
        data = {
            "a": np.array([1, 2, 3]),
            "b": {"c": np.array([4.5])},
            "d": [np.int64(10), np.float64(20.5)],
        }
        converted = io_utils.convert_numpy_to_list(data)
        assert isinstance(converted["a"], list)
        assert converted["a"] == [1, 2, 3]
        assert isinstance(converted["b"]["c"], list)
        assert isinstance(converted["d"][0], int)

    def test_save_and_load_data(self, tmp_path):
        fpath = tmp_path / "test_data.json"
        data = {"key": "value", "arr": [1, 2, 3]}
        success = io_utils.save_data(fpath, data)
        assert success is True or success == 1
        assert fpath.exists()
        loaded = io_utils.load_data(fpath)
        assert loaded == data

    def test_latest_data_file(self, tmp_path):
        (tmp_path / "old.json").touch()
        time.sleep(0.01)
        (tmp_path / "new.json").touch()
        latest = io_utils.latest_data_file(tmp_path, suffixes=[".json"])
        assert Path(latest).name == "new.json"

    def test_create_filename(self, tmp_path):
        fname = io_utils.create_filename(tmp_path, ".txt")
        p = Path(fname)
        assert p.parent == tmp_path
        assert p.suffix == ".txt"
        assert "_" in p.stem

    def test_thread_worker(self):
        q = Queue()
        def success_func(): return "ok"
        def fail_func(): raise ValueError("oops")
        io_utils.thread_worker(q, success_func)
        ok, val = q.get()
        assert ok is True
        assert val == "ok"
        io_utils.thread_worker(q, fail_func)
        ok, err = q.get()
        assert ok is False
        assert isinstance(err, ValueError)
