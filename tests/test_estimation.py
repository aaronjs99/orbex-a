import pytest
import numpy as np
from orbexa.estimation.dynamictube import calcDelta, calcD
from orbexa.estimation.adaptor import adaptor

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

        # calcDelta(t, t_p, x_nom, m=m, ...)
        # It expects x_nom to be list of GEKKO vars or array
        # We'll just check if function exists and imports worked
        assert callable(calcDelta)
        assert callable(calcD)


class TestAdaptor:
    def test_adaptor_signature(self):
        assert callable(adaptor)
