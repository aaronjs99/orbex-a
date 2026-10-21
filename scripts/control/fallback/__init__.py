"""Finite feedback, whole-fleet geometry and transactional admission APIs."""
from importlib import import_module

_EXPORTS = {
    'POLYNOMIAL_PEER_METHOD': 'orbexa.control.fallback.polynomial',
    'PolynomialReferenceSpan': 'orbexa.control.fallback.polynomial',
    'PolynomialReferenceSource': 'orbexa.control.fallback.polynomial',
    'PolynomialPeerBounds': 'orbexa.control.fallback.polynomial',
    'FLEET_SWEEP_METHOD': 'orbexa.control.fallback.geometry',
    'OccupiedGeometryBounds': 'orbexa.control.fallback.geometry',
    'OccupiedProjection': 'orbexa.control.fallback.geometry',
    'SeparationRecord': 'orbexa.control.fallback.geometry',
    'FleetGeometryCertificate': 'orbexa.control.fallback.geometry',
    'FleetGeometryOracle': 'orbexa.control.fallback.geometry',
    'FLEET_FALLBACK_METHOD': 'orbexa.control.fallback.plan',
    'FleetFeedbackSegment': 'orbexa.control.fallback.plan',
    'FleetFallbackCertificate': 'orbexa.control.fallback.plan',
    'FleetFallbackPlan': 'orbexa.control.fallback.plan',
    'PerformanceAttempt': 'orbexa.control.fallback.records',
    'PerformanceAdmissionRefusal': 'orbexa.control.fallback.records',
    'PerformanceProposal': 'orbexa.control.fallback.records',
    'HeldFleetDecision': 'orbexa.control.fallback.records',
    'CommandResourceBound': 'orbexa.control.fallback.records',
    'FleetAdmissionRuntime': 'orbexa.control.fallback.runtime',
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
