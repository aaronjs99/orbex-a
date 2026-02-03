#!/usr/bin/env python3
"""
Plot Results Script

Usage:
    python plot_results.py path/to/data.json
"""

import argparse
import sys
import numpy as np
import logging
from pathlib import Path

from orbexa.simulation.simulator import plot_mpc, MPCPlotConfig
from orbexa.utils.io_utils import load_data

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Plot ORBEX-A MPC Results")
    parser.add_argument("file", help="Path to JSON data file")
    parser.add_argument(
        "-o", "--output", help="Output directory for plots", default=None
    )
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        sys.exit(1)

    logger.info(f"Loading data from: {file_path}")
    data = load_data(file_path)

    # Extract data
    if "state_history" not in data or "input_history" not in data:
        logger.error("Invalid data format: Missing state_history or input_history")
        sys.exit(1)

    # Convert lists back to numpy arrays
    actual_states = np.array(data["state_history"])
    actual_inputs = np.array(data["input_history"])
    anom_history = np.array(data["anom_history"])

    # Determine anom_step from anomaly history if possible
    anom_step = 0.01
    if len(anom_history) > 1:
        anom_step = anom_history[1] - anom_history[0]

    # Defaults for other arrays if not present (legacy data support)
    nominal_states = np.empty((0, 6))
    nominal_inputs = np.empty((0, 3))
    final_states = np.empty((0, 6))
    target_states = np.empty((0, 6))
    x_f_list = []

    # Configure plotting
    target_folder = Path(args.output) if args.output else file_path.parent

    plot_config = MPCPlotConfig(
        anom_step=anom_step, target_folder=target_folder, filename_sim=file_path.stem
    )

    logger.info(f"Generating plots in: {target_folder}")

    # Call plot_mpc
    # We pass explicit arrays. nom_states matches simple MPC mode (no separate nominal)
    # unless we logged it separately.
    plot_mpc(
        act_states=actual_states,
        act_inputs=actual_inputs,
        nom_states=nominal_states,
        nom_inputs=nominal_inputs,
        fin_states=final_states,
        tgt_states=target_states,
        x_f_list=x_f_list,
        cfg=plot_config,
        plot_flags={"plot_act": True, "plot_nom": False},  # Simple visualization
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
