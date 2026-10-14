"""Compatibility entrypoint for the modular saved-data artifact workflow."""

from orbexa.simulation.mission_result import ADTMPCMissionResult
from orbexa.visualization.demo_config import DemoConfig
from orbexa.visualization.demo_workflow import OrbexaDemo, clean_generated_outputs
from orbexa.visualization.manifest import ADTMPCMissionManifest
from orbexa.visualization.mission_renderer import ADTMPCMissionRenderer

DemoManifest = ADTMPCMissionManifest
MissionDemo = ADTMPCMissionResult

__all__ = ["DemoConfig", "OrbexaDemo", "ADTMPCMissionRenderer", "ADTMPCMissionManifest",
           "DemoManifest", "MissionDemo", "clean_generated_outputs", "main"]


def main() -> int:
    from orbexa.cli.generate_demo import main as cli_main
    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
