import pytest
import numpy as np
from orbexa.planning.deflection import target_deflect

# Check other modules
try:
    from orbexa.planning import taskalloc
    from orbexa.planning import distopt
    from orbexa.planning import optimobserve
except ImportError as e:
    pytest.fail(f"Failed to import planning modules: {e}")


def test_deflection_import():
    assert callable(target_deflect)


def test_distopt_funcs():
    from orbexa.planning import distopt

    assert callable(distopt.DistOptAgent)


def test_taskalloc_funcs():
    from orbexa.planning import taskalloc

    assert callable(taskalloc.gen_neighbors)


def test_optimobserve_funcs():
    try:
        from orbexa.planning import optimobserve

        # If it has specific functions, test them here
        pass
    except ImportError:
        pass
