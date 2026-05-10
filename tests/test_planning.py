import pytest
import numpy as np
from orbexa.planning.deflection import target_deflect

# Check other modules
try:
    from orbexa.planning import task_allocation
    from orbexa.planning import distributed_optim
    from orbexa.planning import observation
except ImportError as e:
    pytest.fail(f"Failed to import planning modules: {e}")


def test_deflection_import():
    assert callable(target_deflect)


def test_distributed_optim_api():
    from orbexa.planning import distributed_optim

    # New OOP surface
    assert hasattr(distributed_optim, "TaskAllocationSystem")
    assert hasattr(distributed_optim, "MaxDistanceNeighbors")
    assert hasattr(distributed_optim, "DistributedOptimizationAgent")

    # Demo entrypoint
    assert callable(distributed_optim.run_demo)


def test_task_allocation_api():
    from orbexa.planning import task_allocation

    assert hasattr(task_allocation, "TaskAllocationSystem")
    assert hasattr(task_allocation, "GreedySwapPolicy")
    assert hasattr(task_allocation, "DistributedAuctionPolicy")


def test_observation_reward_increases_with_viewpoint_separation():
    from orbexa.planning import observation

    close = observation.calc_total_local_observation(
        num_chasers=2,
        r=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        R=np.ones(3),
        shape=None,
    )
    spread = observation.calc_total_local_observation(
        num_chasers=2,
        r=np.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.0]),
        R=np.ones(3),
        shape=None,
    )

    assert spread > close
