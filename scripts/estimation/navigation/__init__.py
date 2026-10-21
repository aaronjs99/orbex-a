"""Bounded sensor and navigation services with lazy public exports."""

from importlib import import_module

_EXPORTS = {
    'intersect': 'orbexa.estimation.navigation.boxes',
    'midpoint': 'orbexa.estimation.navigation.boxes',
    'StudySensorProfile': 'orbexa.estimation.navigation.records',
    'SensorPacket': 'orbexa.estimation.navigation.records',
    'NavigationEstimate': 'orbexa.estimation.navigation.records',
    'SensorSuite': 'orbexa.estimation.navigation.sensors',
    'BoundedNavigationFilter': 'orbexa.estimation.navigation.filter',
    'infer_target_ephemeris': 'orbexa.estimation.navigation.target_pose',
    'propagate_pose_estimate': 'orbexa.estimation.navigation.target_pose',
    'PhysicalSlotNoise': 'orbexa.estimation.navigation.noise',
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
