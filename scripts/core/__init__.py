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

"""Explicit core models and structured configuration with lazy public exports."""

from importlib import import_module

_EXPORTS = {
    'DynamicsModel': 'orbexa.core.dynamics.orbital',
    'orbital_ellp_drag': 'orbexa.core.dynamics.orbital',
    'orbital_ellp_undrag': 'orbexa.core.dynamics.orbital',
    'orbital_circ_undrag': 'orbexa.core.dynamics.orbital',
    'cwh_equations': 'orbexa.core.dynamics.orbital',
    'orbital_params': 'orbexa.core.dynamics.orbital',
    'triple_integrator': 'orbexa.core.dynamics.orbital',
    'Spacecraft': 'orbexa.core.spacecraft.base',
    'Target': 'orbexa.core.spacecraft.base',
    'RigidBodyCylinderTarget': 'orbexa.core.targets.cylinder',
    'RigidBodySpacecraftTarget': 'orbexa.core.targets.spacecraft',
    'SpacecraftGeometry': 'orbexa.core.geometry.composite',
    'OBBComponent': 'orbexa.core.geometry.composite',
    'AttachmentPatch': 'orbexa.core.geometry.composite',
    'InertialGravity': 'orbexa.core.dynamics.inertial',
    'QuadraticDrag': 'orbexa.core.dynamics.inertial',
    'LVLHFrame': 'orbexa.core.dynamics.inertial',
    'InertialDynamics': 'orbexa.core.dynamics.inertial',
    'InertialPropagationResult': 'orbexa.core.dynamics.inertial',
    'Chaser': 'orbexa.core.spacecraft.base',
    'SimulationConfig': 'orbexa.core.configuration.model',
    'OrbitConfig': 'orbexa.core.configuration.model',
    'MPCConfig': 'orbexa.core.configuration.model',
    'TargetConfig': 'orbexa.core.configuration.model',
    'TubeMPCConfig': 'orbexa.core.configuration.model',
    'MissionPolicyConfig': 'orbexa.core.configuration.model',
    'AdaptationConfig': 'orbexa.core.configuration.model',
    'SolverConfig': 'orbexa.core.configuration.model',
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
