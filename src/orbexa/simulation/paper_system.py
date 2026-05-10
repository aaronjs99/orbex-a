"""Compatibility shim for older ADTMPC mission imports.

New code should import :mod:`orbexa.simulation.adtmpc_mission`.  The aliases in
this module keep older notebooks and scripts working while the public naming is
standardized around ADTMPC missions.
"""

from orbexa.simulation.adtmpc_mission import (  # noqa: F401
    ADTMPCMissionResult,
    ADTMPCMissionRunner,
    ChaserConfig,
    MissionPhase,
    OrbitalPlantTruth,
    ParameterBelief,
    TumblingCylinderTarget,
    load_adtmpc_mission_result,
    result_chaser_ids,
)

PaperSystemResult = ADTMPCMissionResult
PaperSystemRunner = ADTMPCMissionRunner
PaperTruth = OrbitalPlantTruth
load_paper_system_result = load_adtmpc_mission_result
