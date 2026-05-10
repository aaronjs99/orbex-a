"""Compatibility shim for older demo imports.

The maintained mission implementation is :mod:`orbexa.simulation.adtmpc_mission`.
This module intentionally contains only aliases for older scripts.
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
)

PaperMissionResult = ADTMPCMissionResult
PaperDemoRunner = ADTMPCMissionRunner
PaperTruth = OrbitalPlantTruth
load_paper_demo_result = load_adtmpc_mission_result
