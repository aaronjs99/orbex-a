#!/usr/bin/env python3
"""Shared command line for checkout, module, and installed launchers."""

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

from orbexa.sessions import validate_session_id
from orbexa.layout import project_root


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main(argv: Optional[Sequence[str]] = None, *, default_root: Optional[Path] = None) -> int:
    checkout = project_root()
    root = Path(default_root) if default_root is not None else (
        checkout if (checkout / "config" / "default.yaml").is_file() else Path.cwd()
    )
    parser = argparse.ArgumentParser(description="Run ORBEX-A ADTMPC missions or render saved sessions")
    parser.add_argument("--mission", choices=["single", "multi", "all"], default="all")
    parser.add_argument("-n", "--steps", type=_positive_int, default=450,
                        help="Maximum receding-horizon MPC updates")
    parser.add_argument("--output", type=Path, default=root / "results")
    parser.add_argument("--data-output", type=Path, default=root / "data")
    parser.add_argument("--config", type=Path, default=None,
                        help="Scenario YAML; defaults to the checkout config/default.yaml. Unused with --from-data.")
    parser.add_argument("--session-id", type=lambda value: validate_session_id(value, allow_latest=True),
                        help="New unique session name, or a saved session to render")
    parser.add_argument("--primary-solver", choices=["gekko"], default="gekko")
    parser.add_argument("--clean-generated", action="store_true",
                        help="Explicitly remove the selected generated roots, including session notes")
    parser.add_argument("--from-data", action="store_true",
                        help="Render saved raw data without running MPC or loading a scenario")
    parser.add_argument("--no-require-primary-success", action="store_true",
                        help="Allow a zero exit status for incomplete diagnostic runs")
    parser.add_argument("--fps", type=_positive_int, default=8)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="store_true")
    verbosity.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.from_data and args.clean_generated:
        parser.error("--from-data cannot be combined with --clean-generated")
    if args.session_id is not None and args.session_id.lower() == "latest" and not args.from_data:
        parser.error("latest is reserved for selecting a saved session")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("orbexa")
    try:
        # Help and invalid arguments need only the Python standard library.
        from orbexa.visualization.demo import DemoConfig, OrbexaDemo

        manifests = OrbexaDemo(DemoConfig(
            output_dir=args.output, data_dir=args.data_output,
            config_path=args.config, session_id=args.session_id,
            steps=args.steps, mission=args.mission,
            primary_solver=args.primary_solver,
            from_data=args.from_data, clean_generated=args.clean_generated,
            require_primary_success=not args.no_require_primary_success, fps=args.fps,
        )).run()
        for manifest in manifests:
            logger.info("Mission artifacts written to %s", manifest.output_dir)
        return 0
    except KeyboardInterrupt:
        logger.warning("Interrupted; existing saved sessions are retained.")
        return 130
    except Exception as error:
        logger.error("ORBEX-A mission failed: %s", error, exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
