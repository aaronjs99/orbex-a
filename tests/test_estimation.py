import pytest
import numpy as np
from orbexa.control.dynamic_tube_model import calc_delta, calc_d
from orbexa.estimation.adaptor import run_adaptor_op

# Check if Gekko available
try:
    from gekko import GEKKO

    gekko_available = True
except ImportError:
    gekko_available = False


@pytest.mark.skipif(not gekko_available, reason="GEKKO not installed")
class TestDynamicTube:
    def test_calc_delta_signature(self):
        # Mock GEKKO model
        m = GEKKO(remote=False)
        m.time = np.linspace(0, 1, 5)

        # calc_delta(t, t_p, x_nom, m=m, ...)
        # It expects x_nom to be list of GEKKO vars or array
        # We'll just check if function exists and imports worked
        assert callable(calc_delta)
        assert callable(calc_d)


class TestAdaptor:
    def test_adaptor_signature(self):
        assert callable(run_adaptor_op)

    def test_gen_adaptor_data(self):
        from orbexa.estimation.adaptor import gen_adaptor_data

        assert callable(gen_adaptor_data)


class TestEnclosures:
    def test_enclosures_funcs(self):
        from orbexa.estimation import enclosures

        assert callable(enclosures.min_enclosing_ellipsoid)
        assert callable(enclosures.max_inscribed_ellipsoid)
