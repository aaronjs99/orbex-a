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
    'ChaserConfig': ('orbexa.simulation.approach.models', 'ChaserConfig'),
    'MissionPhase': ('orbexa.simulation.approach.models', 'MissionPhase'),
    'ADTMPCMissionResult': ('orbexa.simulation.approach.records.results', 'ADTMPCMissionResult'),
    'MissionResultStore': ('orbexa.simulation.approach.records.results', 'MissionResultStore'),
    'load_adtmpc_mission_result': ('orbexa.simulation.approach.records.results', 'load_adtmpc_mission_result'),
    'result_chaser_ids': ('orbexa.simulation.approach.records.results', 'result_chaser_ids'),
    'ADTMPCMissionRunner': ('orbexa.simulation.approach.api', 'ADTMPCMissionRunner'),
    'MissionCheckpoint': ('orbexa.simulation.approach.records.checkpoint', 'MissionCheckpoint'),
    'ArrivalPolicyConfig': ('orbexa.simulation.approach.arrival.policy', 'ArrivalPolicyConfig'),
    'ArrivalCommitment': ('orbexa.simulation.approach.arrival.policy', 'ArrivalCommitment'),
    'MissionPolicyTransition': ('orbexa.simulation.approach.arrival.transition', 'MissionPolicyTransition'),
    'MissionScenario': ('orbexa.simulation.approach.scenario', 'MissionScenario'),
    'CoupledTubeMission': ('orbexa.simulation.approach.runner', 'CoupledTubeMission'),
    'OrbitalPlantTruth': ('orbexa.core.dynamics.parameters', 'OrbitalPlantTruth'),
    'ParameterBelief': ('orbexa.estimation.adaptation.belief', 'ParameterBelief'),
    'TumblingCylinderTarget': ('orbexa.core.targets.prescribed', 'TumblingCylinderTarget'),
    'CompactDeploymentOptions': ('orbexa.simulation.approach.deployment.arc', 'CompactDeploymentOptions'),
    'CompactArcDeployment': ('orbexa.simulation.approach.deployment.arc', 'CompactArcDeployment'),
    'CompactDeploymentFleet': ('orbexa.simulation.approach.deployment.arc', 'CompactDeploymentFleet'),
    'CompactDeploymentPolicy': ('orbexa.simulation.approach.deployment.arc_policy', 'CompactDeploymentPolicy'),
    'PhysicalApproachScenario': ('orbexa.simulation.approach.physical.scenario', 'PhysicalApproachScenario'),
    'PhysicalApproachOptions': ('orbexa.simulation.approach.physical.scenario', 'PhysicalApproachOptions'),
    'SpacecraftMissionDesign': ('orbexa.simulation.completion.configuration.design', 'SpacecraftMissionDesign'),
    'CompositeDeploymentOptions': ('orbexa.simulation.approach.deployment.component', 'CompositeDeploymentOptions'),
    'CompositePatchDeployment': ('orbexa.simulation.approach.deployment.component', 'CompositePatchDeployment'),
    'CompositeDeploymentPolicy': ('orbexa.simulation.approach.deployment.component_policy', 'CompositeDeploymentPolicy'),
    'ContinuousAdmissionOptions': ('orbexa.simulation.approach.physical.scenario', 'ContinuousAdmissionOptions'),
    'CompleteMissionRunner': ('orbexa.simulation.completion.coordinator', 'CompleteMissionRunner'),
    'CompleteMissionOptions': ('orbexa.simulation.completion.coordinator', 'CompleteMissionOptions'),
    'CompleteMissionResult': ('orbexa.simulation.completion.coordinator', 'CompleteMissionResult'),
    'CompleteMissionStore': ('orbexa.simulation.completion.coordinator', 'CompleteMissionStore'),
    'ApproachPropellantBudget': ('orbexa.simulation.completion.coordinator', 'ApproachPropellantBudget'),
    'PhysicalTerminalRunner': ('orbexa.simulation.approach.terminal.execution', 'PhysicalTerminalRunner'),
    'PhysicalTerminalExecutionResult': ('orbexa.simulation.approach.terminal.execution', 'PhysicalTerminalExecutionResult'),
    'TerminalObservation': ('orbexa.simulation.approach.terminal.execution', 'TerminalObservation'),
    'terminal_source_lineage': ('orbexa.simulation.approach.terminal.execution', 'terminal_source_lineage'),
    'PhysicalTerminalHandoff': ('orbexa.simulation.approach.terminal.handoff', 'PhysicalTerminalHandoff'),
    'PhysicalTerminalTrajectory': ('orbexa.simulation.approach.terminal.handoff', 'PhysicalTerminalTrajectory'),
    'CertifiedExecutionGuard': ('orbexa.simulation.approach.physical.admission', 'CertifiedExecutionGuard'),
    'CaptureModel': ('orbexa.simulation.contact.capture', 'CaptureModel'),
    'CaptureOptions': ('orbexa.simulation.contact.capture', 'CaptureOptions'),
    'CaptureTruth': ('orbexa.simulation.contact.capture', 'CaptureTruth'),
    'ChaserPhysicalProperties': ('orbexa.simulation.contact.capture', 'ChaserPhysicalProperties'),
    'TargetPhysicalProperties': ('orbexa.simulation.contact.capture', 'TargetPhysicalProperties'),
    'ChaserAttitudeTracker': ('orbexa.simulation.approach.physical.attitude', 'ChaserAttitudeTracker'),
    'ChaserAttitudeOptions': ('orbexa.simulation.approach.physical.attitude', 'ChaserAttitudeOptions'),
    'ThrusterSystem': ('orbexa.simulation.attached.propulsion', 'ThrusterSystem'),
    'AttachedAssembly': ('orbexa.simulation.attached.assembly.model', 'AttachedAssembly'),
    'AssemblyState': ('orbexa.simulation.attached.assembly.model', 'AssemblyState'),
    'DetumblingSimulator': ('orbexa.simulation.attached.detumbling', 'DetumblingSimulator'),
    'DetumblingOptions': ('orbexa.simulation.attached.detumbling', 'DetumblingOptions'),
    'DeorbitSimulator': ('orbexa.simulation.attached.disposal.simulator', 'DeorbitSimulator'),
    'DeorbitOptions': ('orbexa.simulation.attached.disposal.simulator', 'DeorbitOptions'),
    'MissionLifecycle': ('orbexa.simulation.completion.lifecycle', 'MissionLifecycle'),
    'MissionLifecycleResult': ('orbexa.simulation.completion.lifecycle', 'MissionLifecycleResult'),
    'MissionLifecycleStore': ('orbexa.simulation.completion.lifecycle', 'MissionLifecycleStore'),
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
