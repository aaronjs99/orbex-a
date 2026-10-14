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
    'MissionCheckpoint': ('mission_checkpoint', 'MissionCheckpoint'),
    'MissionScenario': ('mission_scenario', 'MissionScenario'),
    'CoupledTubeMission': ('coupled_mission', 'CoupledTubeMission'),
    'OrbitalPlantTruth': ('orbexa.core.orbital_parameters', 'OrbitalPlantTruth'),
    'ParameterBelief': ('orbexa.estimation.belief', 'ParameterBelief'),
    'TumblingCylinderTarget': ('orbexa.core.target_geometry', 'TumblingCylinderTarget'),
    'CompactDeploymentOptions': ('deployment', 'CompactDeploymentOptions'),
    'CompactArcDeployment': ('deployment', 'CompactArcDeployment'),
    'CompactDeploymentFleet': ('deployment', 'CompactDeploymentFleet'),
    'CompactDeploymentPolicy': ('deployment_policy', 'CompactDeploymentPolicy'),
    'PhysicalApproachScenario': ('physical_approach', 'PhysicalApproachScenario'),
    'PhysicalApproachOptions': ('physical_approach', 'PhysicalApproachOptions'),
    'ContinuousAdmissionOptions': ('physical_approach', 'ContinuousAdmissionOptions'),
    'CompleteMissionRunner': ('complete_mission', 'CompleteMissionRunner'),
    'CompleteMissionOptions': ('complete_mission', 'CompleteMissionOptions'),
    'CompleteMissionResult': ('complete_mission', 'CompleteMissionResult'),
    'CompleteMissionStore': ('complete_mission', 'CompleteMissionStore'),
    'ApproachPropellantBudget': ('complete_mission', 'ApproachPropellantBudget'),
    'PhysicalTerminalRunner': ('physical_terminal_execution', 'PhysicalTerminalRunner'),
    'PhysicalTerminalExecutionResult': ('physical_terminal_execution', 'PhysicalTerminalExecutionResult'),
    'TerminalObservation': ('physical_terminal_execution', 'TerminalObservation'),
    'PhysicalTerminalHandoff': ('terminal_handoff', 'PhysicalTerminalHandoff'),
    'PhysicalTerminalTrajectory': ('terminal_handoff', 'PhysicalTerminalTrajectory'),
    'CertifiedExecutionGuard': ('certified_execution', 'CertifiedExecutionGuard'),
    'CaptureModel': ('capture', 'CaptureModel'),
    'CaptureOptions': ('capture', 'CaptureOptions'),
    'CaptureTruth': ('capture', 'CaptureTruth'),
    'ChaserPhysicalProperties': ('capture', 'ChaserPhysicalProperties'),
    'TargetPhysicalProperties': ('capture', 'TargetPhysicalProperties'),
    'ChaserAttitudeTracker': ('chaser_attitude', 'ChaserAttitudeTracker'),
    'ChaserAttitudeOptions': ('chaser_attitude', 'ChaserAttitudeOptions'),
    'ThrusterSystem': ('propulsion', 'ThrusterSystem'),
    'AttachedAssembly': ('assembly', 'AttachedAssembly'),
    'AssemblyState': ('assembly', 'AssemblyState'),
    'DetumblingSimulator': ('detumbling', 'DetumblingSimulator'),
    'DetumblingOptions': ('detumbling', 'DetumblingOptions'),
    'DeorbitSimulator': ('deorbit', 'DeorbitSimulator'),
    'DeorbitOptions': ('deorbit', 'DeorbitOptions'),
    'MissionLifecycle': ('lifecycle', 'MissionLifecycle'),
    'MissionLifecycleResult': ('lifecycle', 'MissionLifecycleResult'),
    'MissionLifecycleStore': ('lifecycle', 'MissionLifecycleStore'),
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
