"""figures.py: command line that renders figures from recorded runs into results/.

``orbexa-figures mission RUN`` renders one run's mission plots and, when a
completed lifecycle was recorded, its lifecycle plots and replay data.
``orbexa-figures comparison NAME ARM...`` renders a matched comparison of study
arms. Outputs default to ``results/missions/...`` and ``results/comparisons/NAME``.

Copyright: Aaron John Sabu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from orbexa.common.layout import ProjectLayout


class FiguresCommand:
    """Parse ``orbexa-figures`` arguments and render the requested figures."""

    def __call__(self, argv=None) -> int:
        """Run the command.

        Inputs: ``argv`` as for ``argparse``; ``None`` reads ``sys.argv``.
        Outputs: prints a JSON summary and returns 0, or exits through
        ``argparse`` with a message when a source is missing or unreadable.
        """
        parser = argparse.ArgumentParser(
            description="Render figures from recorded runs into results/."
        )
        commands = parser.add_subparsers(dest="command", required=True)
        mission = commands.add_parser("mission", help="Render one recorded run")
        mission.add_argument("run", type=Path, help="Run directory under data/")
        mission.add_argument("--output-dir", type=Path)
        comparison = commands.add_parser(
            "comparison", help="Render a matched comparison of study arms"
        )
        comparison.add_argument("name", help="Folder name under results/comparisons/")
        comparison.add_argument("arms", type=Path, nargs="+")
        comparison.add_argument("--output-dir", type=Path)
        args = parser.parse_args(argv)
        layout = ProjectLayout.current()
        try:
            if args.command == "mission":
                output = args.output_dir or layout.results_directory(
                    "missions", args.run
                )
                summary = self.mission(args.run, output)
            else:
                output = (
                    args.output_dir or layout.results_root / "comparisons" / args.name
                )
                summary = self.comparison(args.arms, output)
        except (OSError, ValueError, KeyError) as error:
            parser.error(str(error))
        print(json.dumps(summary, indent=2, allow_nan=False))
        return 0

    @staticmethod
    def mission(run: Path, output: Path) -> dict:
        """Render mission plots, and lifecycle plots when a lifecycle was recorded.

        Outputs: a summary with the output folder, written artifacts and the
        artifacts skipped for missing data.
        """
        from orbexa.figures.plots.nonlinear import NonlinearMissionPlots
        from orbexa.figures.renderers.lifecycle import LifecycleRenderer

        output = Path(output)
        plots = NonlinearMissionPlots.from_directory(run, output / "plots")
        written = [str(path) for path, _label in plots.render()]
        skipped = dict(plots.skipped_artifacts)
        if plots.lifecycle is not None:
            lifecycle = LifecycleRenderer(output / "lifecycle")
            lifecycle.render(plots.lifecycle)
            written.append(str(output / "lifecycle"))
            skipped.update(lifecycle.skipped_artifacts)
        return {"output_dir": str(output), "artifacts": written, "skipped": skipped}

    @staticmethod
    def comparison(arms, output: Path) -> dict:
        """Render the matched comparison of the given study arm directories.

        Outputs: a summary with the output folder and written artifacts.
        """
        from orbexa.figures.data.comparison import MatchedNonlinearComparison

        report = MatchedNonlinearComparison(arms, output)
        written = report.render()
        return {"output_dir": str(output), "artifacts": [str(p) for p in written or []]}


main = FiguresCommand()
