"""Lazy visualization exports: gallery and replay imports are runtime-independent."""

from importlib import import_module

_EXPORTS = {
    "DemoConfig": ('orbexa.visualization.workflow.config', "DemoConfig"),
    "DemoManifest": ('orbexa.visualization.data.manifest', "ADTMPCMissionManifest"),
    "ADTMPCMissionManifest": ('orbexa.visualization.data.manifest', "ADTMPCMissionManifest"),
    "MissionDemo": ('orbexa.simulation.approach.records.results', "ADTMPCMissionResult"),
    "OrbexaDemo": ('orbexa.visualization.workflow.workflow', "OrbexaDemo"),
    "ADTMPCMissionRenderer": ('orbexa.visualization.renderers.mission', "ADTMPCMissionRenderer"),
    "MissionGallery": ('orbexa.visualization.publishing.gallery', "MissionGallery"),
    "MissionStudio": ('orbexa.visualization.publishing.studio', "MissionStudio"),
    "StudioEntry": ('orbexa.visualization.publishing.studio', "StudioEntry"),
    "StudioDataset": ('orbexa.visualization.data.studio', "StudioDataset"),
    "LifecycleDataset": ('orbexa.visualization.renderers.lifecycle', "LifecycleDataset"),
    "LifecycleRenderer": ('orbexa.visualization.renderers.lifecycle', "LifecycleRenderer"),
    "MissionInsights": ('orbexa.visualization.plots.insights', "MissionInsights"),
    "MissionReplay": ('orbexa.visualization.renderers.replay', "MissionReplay"),
    "MatchedTubeDiagnostics": ('orbexa.visualization.plots.tubes', "MatchedTubeDiagnostics"),
    "MissionSamples": ('orbexa.visualization.data.samples', "MissionSamples"),
    "MissionGeometry": ('orbexa.visualization.data.geometry', "MissionGeometry"),
    "MissionPlots": ('orbexa.visualization.plots.mission', "MissionPlots"),
    "MissionInteractive": ('orbexa.visualization.renderers.interactive', "MissionInteractive"),
    "MissionAnimation": ('orbexa.visualization.renderers.animation', "MissionAnimation"),
    "MissionAnimationScene": ('orbexa.visualization.renderers.animation', "MissionAnimationScene"),
    "MissionCatalog": ('orbexa.visualization.publishing.catalog', "MissionCatalog"),
    "clean_generated_outputs": ('orbexa.visualization.workflow.workflow', "clean_generated_outputs"),
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
