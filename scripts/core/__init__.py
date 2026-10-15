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
    'DynamicsModel': 'orbexa.core.dynamics',
    'orbital_ellp_drag': 'orbexa.core.dynamics',
    'orbital_ellp_undrag': 'orbexa.core.dynamics',
    'orbital_circ_undrag': 'orbexa.core.dynamics',
    'cwh_equations': 'orbexa.core.dynamics',
    'orbital_params': 'orbexa.core.dynamics',
    'triple_integrator': 'orbexa.core.dynamics',
    'Spacecraft': 'orbexa.core.spacecraft',
    'Target': 'orbexa.core.spacecraft',
    'RigidBodyCylinderTarget': 'orbexa.core.rigid_body_target',
    'RigidBodySpacecraftTarget': 'orbexa.core.spacecraft_target',
    'SpacecraftGeometry': 'orbexa.core.spacecraft_geometry',
    'OBBComponent': 'orbexa.core.spacecraft_geometry',
    'AttachmentPatch': 'orbexa.core.spacecraft_geometry',
    'InertialGravity': 'orbexa.core.inertial_dynamics',
    'QuadraticDrag': 'orbexa.core.inertial_dynamics',
    'LVLHFrame': 'orbexa.core.inertial_dynamics',
    'InertialDynamics': 'orbexa.core.inertial_dynamics',
    'InertialPropagationResult': 'orbexa.core.inertial_dynamics',
    'Chaser': 'orbexa.core.spacecraft',
    'SimulationConfig': 'orbexa.core.config',
    'OrbitConfig': 'orbexa.core.config',
    'MPCConfig': 'orbexa.core.config',
    'TargetConfig': 'orbexa.core.config',
    'TubeMPCConfig': 'orbexa.core.config',
    'MissionPolicyConfig': 'orbexa.core.config',
    'AdaptationConfig': 'orbexa.core.config',
    'SolverConfig': 'orbexa.core.config',
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
