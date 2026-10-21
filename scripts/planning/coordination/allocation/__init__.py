# /***********************************************************
# *                                                         *
# * Copyright (c) 2021                                      *
# *                                                         *
# * Dept. of Electrical Engineering                         *
# * Indian Institute of Technology Bombay                   *
# *                                                         *
# * Authors: Aaron John Sabu                                *
# * Contact: aaronjs@ucla.edu                               *
# *                                                         *
# ***********************************************************/


"""Task allocation contracts, networks and policies with lazy public exports."""
from __future__ import annotations

from importlib import import_module
import logging


logger = logging.getLogger(__name__)


_EXPORTS = {'Vec': 'orbexa.planning.coordination.allocation.contracts',
 'AgentState': 'orbexa.planning.coordination.allocation.contracts',
 'Network': 'orbexa.planning.coordination.allocation.contracts',
 'Policy': 'orbexa.planning.coordination.allocation.contracts',
 'StaticNeighborNetwork': 'orbexa.planning.coordination.allocation.networks',
 'KNNNetwork': 'orbexa.planning.coordination.allocation.networks',
 'RadiusNetwork': 'orbexa.planning.coordination.allocation.networks',
 'GreedySwapPolicy': 'orbexa.planning.coordination.allocation.policies.greedy',
 'LocalHungarianPolicy': 'orbexa.planning.coordination.allocation.policies.hungarian',
 'DistributedAuctionPolicy': 'orbexa.planning.coordination.allocation.policies.auction',
 'TaskAllocationSystem': 'orbexa.planning.coordination.allocation.system',
 'make_agents': 'orbexa.planning.coordination.allocation.system',
 'plot_assignments': 'orbexa.planning.coordination.allocation.demo',
 'demo_main': 'orbexa.planning.coordination.allocation.demo'}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value
