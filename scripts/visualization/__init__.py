"""Lazy visualization exports: gallery and replay imports are runtime-independent."""

from importlib import import_module

_EXPORTS = {
    "DemoConfig": ("demo_config", "DemoConfig"),
    "DemoManifest": ("manifest", "ADTMPCMissionManifest"),
    "ADTMPCMissionManifest": ("manifest", "ADTMPCMissionManifest"),
    "MissionDemo": ("orbexa.simulation.mission_result", "ADTMPCMissionResult"),
    "OrbexaDemo": ("demo_workflow", "OrbexaDemo"),
    "ADTMPCMissionRenderer": ("mission_renderer", "ADTMPCMissionRenderer"),
    "MissionGallery": ("mission_gallery", "MissionGallery"),
    "MissionStudio": ("mission_studio", "MissionStudio"),
    "StudioEntry": ("mission_studio", "StudioEntry"),
    "StudioDataset": ("studio_data", "StudioDataset"),
    "LifecycleDataset": ("lifecycle_renderer", "LifecycleDataset"),
    "LifecycleRenderer": ("lifecycle_renderer", "LifecycleRenderer"),
    "MissionInsights": ("mission_insights", "MissionInsights"),
    "MissionReplay": ("mission_replay", "MissionReplay"),
    "MatchedTubeDiagnostics": ("tube_diagnostics", "MatchedTubeDiagnostics"),
    "MissionSamples": ("samples", "MissionSamples"),
    "MissionGeometry": ("geometry", "MissionGeometry"),
    "MissionPlots": ("mission_plots", "MissionPlots"),
    "MissionInteractive": ("mission_interactive", "MissionInteractive"),
    "MissionAnimation": ("mission_animation", "MissionAnimation"),
    "MissionAnimationScene": ("mission_animation", "MissionAnimationScene"),
    "MissionCatalog": ("mission_catalog", "MissionCatalog"),
    "clean_generated_outputs": ("demo_workflow", "clean_generated_outputs"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    module = import_module(module_name if module_name.startswith("orbexa.") else f"{__name__}.{module_name}")
    value = getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
