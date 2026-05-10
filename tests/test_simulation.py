from orbexa.simulation import runner
from orbexa.simulation import simulator


class TestSimulation:
    def test_runner_funcs_exist(self):
        assert callable(runner.run_simulation)
        # _run_single_mode is private, so we usually don't test it directly
        # unless we want to access it via runner._run_single_mode
        if hasattr(runner, "_run_single_mode"):
            assert callable(runner._run_single_mode)

    def test_plotting_functions_exist(self):
        # Check for presence of main plotting functions
        assert hasattr(simulator, "plot_mpc")
        assert callable(simulator.plot_mpc)

        assert hasattr(simulator, "plot_deflection")
        assert callable(simulator.plot_deflection)

        assert hasattr(simulator, "plot_adaptor")
        assert callable(simulator.plot_adaptor)

    def test_simulation_plot_exists(self):
        if hasattr(simulator, "create_animation_html"):
            assert callable(simulator.create_animation_html)
