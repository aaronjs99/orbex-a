#!/usr/bin/env python3
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

"""Compatibility imports for the retired demo-first module.

The paper system now lives in :mod:`orbexa.simulation.paper_system`.  These
aliases keep older imports from silently reaching the removed ablation code.
"""

from orbexa.simulation.paper_system import (  # noqa: F401
    ChaserConfig,
    MissionPhase,
    PaperSystemResult,
    PaperSystemRunner,
    PaperTruth,
    ParameterBelief,
    TumblingCylinderTarget,
    load_paper_system_result,
)

PaperMissionResult = PaperSystemResult
PaperDemoRunner = PaperSystemRunner
