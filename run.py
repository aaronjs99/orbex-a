#!/usr/bin/env python3
# /***********************************************************
# *                                                         *
# * Copyright (c) 2026                                      *
# *                                                         *
# * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
# * University of California, Los Angeles                   *
# *                                                         *
# * Authors: Aaron John Sabu                                *
# * Contact: {aaronjs, btlopez}@ucla.edu                    *
# *                                                         *
# ***********************************************************/

"""ORBEX-A command-line entry point.

By default this wrapper generates ADTMPC mission artifacts.  The older
single-mode simulation runner remains available through ``--workflow legacy``.
"""

import argparse
import logging
import sys

from orbexa.simulation import run_simulation, run_all_modes
from orbexa.visualization.demo import DemoConfig, OrbexaDemo


def setup_logging(verbose: bool, quiet: bool):
    """
    Configure the root logger.

    Format: [LEVEL] Message
    """
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    # Define custom format to match user request: [LOG] [INFO] etc...
    # Though standard is just [INFO], user asked for [LOG] [INFO]...
    # Re-reading: "everything is displayed using a logger [LOG] [INFO] [WARN] [ERROR] etc..."
    # I'll stick to a standard informative format: "[%(levelname)s] %(message)s"
    # capturing the essence.

    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main():
    parser = argparse.ArgumentParser(
        description="ORBEX-A ADTMPC mission runner",
        epilog=(
            "Examples: python run.py --mission all --run-linearized; "
            "python run.py --workflow legacy --mode mpc --steps 15"
        ),
    )
    parser.add_argument(
        "--workflow",
        choices=["adtmpc", "legacy"],
        default="adtmpc",
        help="Run the ADTMPC mission artifact generator or the older mode simulator.",
    )
    parser.add_argument(
        "--mission",
        choices=["single", "multi", "all"],
        default="all",
        help="ADTMPC mission family to run.",
    )
    parser.add_argument("-n", "--steps", type=int, default=450)
    parser.add_argument("--output", default="results")
    parser.add_argument("--data-output", default="data")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--primary-solver", choices=["gekko", "scipy", "casadi"], default=None)
    parser.add_argument("--secondary-solver", choices=["gekko", "scipy", "casadi"], default="scipy")
    parser.add_argument("--run-linearized", action="store_true")
    parser.add_argument("--linearized-steps", type=int, default=20)
    parser.add_argument("--clean-generated", action="store_true")
    parser.add_argument("--from-data", action="store_true")
    parser.add_argument("--no-require-primary-success", action="store_true")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument(
        "-m",
        "--mode",
        choices=["oc", "mpc", "tube", "adtmpc", "all"],
        default="mpc",
        help="Legacy simulator mode used only with --workflow legacy.",
    )
    parser.add_argument(
        "-s", "--solver", choices=["gekko", "scipy", "casadi"], default="gekko"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    group.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress info logging"
    )

    args = parser.parse_args()

    setup_logging(args.verbose, args.quiet)
    logger = logging.getLogger(__name__)

    primary_solver = args.primary_solver or args.solver
    logger.info(
        "ORBEX-A | Workflow: %s, Steps: %s, Solver: %s",
        args.workflow,
        args.steps,
        primary_solver,
    )

    try:
        if args.workflow == "adtmpc":
            manifests = OrbexaDemo(
                DemoConfig(
                    output_dir=args.output,
                    data_dir=args.data_output,
                    session_id=args.session_id,
                    steps=args.steps,
                    mission=args.mission,
                    primary_solver=primary_solver,
                    secondary_solver=args.secondary_solver,
                    run_linearized=args.run_linearized,
                    linearized_steps=args.linearized_steps,
                    from_data=args.from_data,
                    clean_generated=args.clean_generated,
                    require_primary_success=not args.no_require_primary_success,
                    fps=args.fps,
                )
            ).run()
            for manifest in manifests:
                logger.info("ADTMPC mission written to %s", manifest.output_dir)
        elif args.mode == "all":
            run_all_modes(args.steps, args.solver)
        else:
            result = run_simulation(args.mode, args.steps, args.solver)
            if not result.success:
                sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
