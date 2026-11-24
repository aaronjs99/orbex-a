"""run_nonlinear.py: Run one explicit arm of a predeclared sensed nonlinear 2-by-2 study.

Copyright: Aaron John Sabu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from orbexa.common.layout import ProjectLayout


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run one explicit arm of a predeclared sensed nonlinear 2-by-2 study."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--spacecraft", type=Path, required=True)
    parser.add_argument(
        "--study",
        type=Path,
        required=True,
        help="Prospective study JSON with explicit options and all cases",
    )
    parser.add_argument(
        "--case",
        required=True,
        help="Exact predeclared case_id; no case is selected automatically",
    )
    parser.add_argument("--adaptation", choices=("adaptive", "frozen"), required=True)
    parser.add_argument("--bandwidth", choices=("dynamic", "fixed"), required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New arm directory; defaults to data/nonlinear/<case>/<arm>/<UTC run id>",
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="Positive execution cap; does not shorten the prospective reference domain",
    )
    parser.add_argument(
        "--proof-deps",
        type=Path,
        help="Optional explicit python-flint dependency directory",
    )
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = ProjectLayout.current().run_directory(
            "nonlinear", args.case, f"{args.adaptation}-{args.bandwidth}"
        )
    try:
        if args.proof_deps is not None:
            path = args.proof_deps.expanduser().resolve()
            if not path.is_dir():
                raise ValueError("Missing explicit proof dependency directory")
            sys.path.insert(0, str(path))
        from orbexa.studies.api import StudyArmRequest, run_study_arm

        result = run_study_arm(
            StudyArmRequest(
                args.config,
                args.spacecraft,
                args.study,
                args.case,
                args.adaptation,
                args.bandwidth,
                args.output_dir,
                args.steps,
            )
        )
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0 if result["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
