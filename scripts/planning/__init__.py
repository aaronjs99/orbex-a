# /***********************************************************
# *                                                         *
# * Copyright (c) 2026                                      *
# *                                                         *
# * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
# * University of California, Los Angeles                   *
# *                                                         *
# * Authors: Aaron John Sabu                                *
# * Contact: aaronjs@ucla.edu                               *
# *                                                         *
# ***********************************************************/

"""
ORBEX-A Planning Module

Mission planning, task allocation, and observation optimization.
"""

from importlib import import_module

_EXPORTS = {
    "TaskAllocationSystem": 'orbexa.planning.coordination.allocation.system',
    'CompositeAttachmentPlanner': 'orbexa.planning.rendezvous.routes',
    'AttachmentRoute': 'orbexa.planning.rendezvous.routes',
    'assign_composite_docking_targets': 'orbexa.planning.rendezvous.composite',
    "target_deflect": 'orbexa.planning.objectives.deflection',
    "observe_optimizer": 'orbexa.planning.objectives.observation',
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
