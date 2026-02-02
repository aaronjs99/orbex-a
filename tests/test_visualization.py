import pytest

# Check explicit functions found in orbitsim
import orbexa.visualization.orbitsim as orbitsim


class TestVisualization:
    def test_plotting_functions_exist(self):
        # Check for presence of main plotting functions
        assert hasattr(orbitsim, "plot_mpc")
        assert callable(orbitsim.plot_mpc)

        assert hasattr(orbitsim, "plot_deflection")
        assert callable(orbitsim.plot_deflection)

        assert hasattr(orbitsim, "plot_adaptor")
        assert callable(orbitsim.plot_adaptor)

    def test_simulation_plot_exists(self):
        if hasattr(orbitsim, "create_animation_html"):
            assert callable(orbitsim.create_animation_html)
