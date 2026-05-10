#!/usr/bin/env python3
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

"""Plot legacy ORBEX-A JSON mission data."""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from orbexa.simulation.simulator import MPCPlotConfig, plot_mpc
from orbexa.utils.io_utils import load_data

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ORBEX-A MPC results")
    parser.add_argument("file", help="Path to JSON data file")
    parser.add_argument(
        "-o",
        "--output",
        help="Output directory for plots",
        default=None,
    )
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        logger.error("File not found: %s", file_path)
        sys.exit(1)

    logger.info("Loading data from: %s", file_path)
    data = load_data(file_path)
    if "state_history" not in data or "input_history" not in data:
        logger.error("Invalid data format: missing state_history or input_history")
        sys.exit(1)

    actual_states = np.array(data["state_history"])
    actual_inputs = np.array(data["input_history"])
    anom_history = np.array(data["anom_history"])
    anom_step = anom_history[1] - anom_history[0] if len(anom_history) > 1 else 0.01

    target_folder = Path(args.output) if args.output else Path("results") / file_path.stem
    plot_config = MPCPlotConfig(
        anom_step=anom_step,
        target_folder=target_folder,
        filename_sim=file_path.stem,
    )

    logger.info("Generating plots in: %s", target_folder)
    plot_mpc(
        act_states=actual_states,
        act_inputs=actual_inputs,
        nom_states=np.empty((0, 6)),
        nom_inputs=np.empty((0, 3)),
        fin_states=np.empty((0, 6)),
        tgt_states=np.empty((0, 6)),
        x_f_list=[],
        cfg=plot_config,
        plot_flags={"plot_act": True, "plot_nom": False},
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
