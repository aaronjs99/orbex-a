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

"""
ORBEX-A Entry Point

Thin CLI wrapper for running simulations.
All logic is in orbexa.simulation module.
"""

import argparse
import logging
import sys

from orbexa.simulation import run_simulation, run_all_modes


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
        description="ORBEX-A Spacecraft Rendezvous Simulation",
        epilog="Modes: oc (optimal control), mpc, tube, adtmpc, all",
    )
    parser.add_argument(
        "-m", "--mode", choices=["oc", "mpc", "tube", "adtmpc", "all"], default="mpc"
    )
    parser.add_argument("-n", "--steps", type=int, default=15)
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

    logger.info(
        f"ORBEX-A | Mode: {args.mode.upper()}, Steps: {args.steps}, Solver: {args.solver}"
    )

    try:
        if args.mode == "all":
            run_all_modes(args.steps, args.solver)
        else:
            # We don't need to pass verbose boolean anymore if we use logging globally
            # But run_simulation might still expect it for now. We will check/update that signature next.
            # Ideally run_simulation just uses logging.getLogger().
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
