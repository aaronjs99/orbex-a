"""Compatibility entrypoint for the modular saved-data artifact workflow."""

from orbexa.simulation.approach.records.results import ADTMPCMissionResult
from orbexa.visualization.data.manifest import ADTMPCMissionManifest
from orbexa.visualization.renderers.mission import ADTMPCMissionRenderer
from orbexa.visualization.workflow.config import DemoConfig
from orbexa.visualization.workflow.workflow import OrbexaDemo, clean_generated_outputs

DemoManifest = ADTMPCMissionManifest
MissionDemo = ADTMPCMissionResult

__all__ = [
    "ADTMPCMissionManifest",
    "ADTMPCMissionRenderer",
    "DemoConfig",
    "DemoManifest",
    "MissionDemo",
    "OrbexaDemo",
    "clean_generated_outputs",
    "main",
]


def main() -> int:
    from orbexa.cli.generate_demo import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
