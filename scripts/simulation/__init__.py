# /***********************************************************
# *                                                         *
# * Copyright (c) 2026                                      *
# *                                                         *
# * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
# * University of California, Los Angeles                   *
# *                                                         *
# * Authors: Aaron John Sabu, Brett T. Lopez                *
# * Contact: {aaronjs, btlopez}@ucla.edu                    *
# *                                                         *
# ***********************************************************/

"""Mission APIs, resolved lazily so result loading does not import the engine."""
from importlib import import_module

_EXPORTS = {
    'ChaserConfig': ('mission_models', 'ChaserConfig'),
    'MissionPhase': ('mission_models', 'MissionPhase'),
    'ADTMPCMissionResult': ('mission_result', 'ADTMPCMissionResult'),
    'MissionResultStore': ('mission_result', 'MissionResultStore'),
    'load_adtmpc_mission_result': ('mission_result', 'load_adtmpc_mission_result'),
    'result_chaser_ids': ('mission_result', 'result_chaser_ids'),
    'ADTMPCMissionRunner': ('adtmpc_mission', 'ADTMPCMissionRunner'),
    'MissionScenario': ('mission_scenario', 'MissionScenario'),
    'CoupledTubeMission': ('coupled_mission', 'CoupledTubeMission'),
    'OrbitalPlantTruth': ('orbexa.core.orbital_parameters', 'OrbitalPlantTruth'),
    'ParameterBelief': ('orbexa.estimation.belief', 'ParameterBelief'),
    'TumblingCylinderTarget': ('orbexa.core.target_geometry', 'TumblingCylinderTarget'),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    module = import_module(module if module.startswith('orbexa.') else '.' + module, __name__)
    value = getattr(module, attribute)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
