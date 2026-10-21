# /***********************************************************
# *                                                         *
# * Copyright (c) 2026                                      *
# *                                                         *
# * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
# * University of California, Los Angeles                   *
# *                                                         *
# * Authors: Aaron John Sabu, Brett T. Lopez                *
# * Contact: {aaronjs, btlopez}@ucla.edu                    *
# *                                                         *
# ***********************************************************/

"""Solver-independent control helpers, loaded only when requested."""

from importlib import import_module

_EXPORTS = {
    'MPCController': 'orbexa.control.mpc.controller',
    'MissionResult': 'orbexa.control.mpc.controller',
    'ORBEXAProblemConfig': 'orbexa.control.mpc.problem',
    'build_mpc_problem': 'orbexa.control.mpc.problem',
    'build_from_dynamics': 'orbexa.control.mpc.problem',
    'ancillary_controller': 'orbexa.control.tubes.model',
    'calc_delta': 'orbexa.control.tubes.model',
    'calc_d': 'orbexa.control.tubes.model',
    'TubeProfile': 'orbexa.control.tubes.model',
    'TerminalRegion': 'orbexa.control.terminal.region',
    'input_tightening_from_profile': 'orbexa.control.tubes.model',
    'propagate_tube_profile': 'orbexa.control.tubes.model',
    'tighten_box_bounds': 'orbexa.control.tubes.model',
    'tighten_target_params': 'orbexa.control.tubes.model',
    'AffineConstraint': 'orbexa.control.mpc.linearization',
    'linearize_cylinder_constraint': 'orbexa.control.mpc.linearization',
    'linearize_rendezvous_constraint': 'orbexa.control.mpc.linearization',
    'CylinderConstraint': 'orbexa.control.mpc.constraints',
    'collision_params_from_target_config': 'orbexa.control.mpc.constraints',
    'rendezvous_margin': 'orbexa.control.mpc.constraints',
    'rotating_body_point_velocity': 'orbexa.control.mpc.constraints',
    'rotating_docking_point': 'orbexa.control.mpc.constraints',
    'target_frame_position': 'orbexa.control.mpc.constraints',
    'IntervalBox': 'orbexa.control.certification.reachability',
    'OrbitalModelBounds': 'orbexa.control.certification.reachability',
    'HeldInputRequest': 'orbexa.control.certification.reachability',
    'ReachabilityConfig': 'orbexa.control.certification.reachability',
    'ValidatedFlowEncloser': 'orbexa.control.certification.reachability',
    'SweptCollisionGuard': 'orbexa.control.certification.reachability',
    'RotatingCylinderBounds': 'orbexa.control.certification.reachability',
    'RotatingCompositeBounds': 'orbexa.control.certification.geometry',
    'CompositeSweptCollisionGuard': 'orbexa.control.certification.geometry',
    'CompositeTerminalGeometry': 'orbexa.control.certification.geometry',
    'CompositePointSeparation': 'orbexa.control.certification.geometry',
    'CompositePhysicalTerminalBoundsBuilder': 'orbexa.control.terminal.composite',
    'ComponentwiseCompositePhysicalTerminalBoundsBuilder': 'orbexa.control.terminal.componentwise',
    'ComponentwiseTerminalJetBounds': 'orbexa.control.terminal.body_jets',
    'bound_componentwise_terminal_jets': 'orbexa.control.terminal.body_jets',
    'RigidBodyRotationBounds': 'orbexa.control.terminal.rotation_bounds',
    'bound_rigid_body_rotation': 'orbexa.control.terminal.rotation_bounds',
    'BoundScope': 'orbexa.control.terminal.tracking',
    'TerminalTrackingBounds': 'orbexa.control.terminal.tracking',
    'TerminalGeometryBounds': 'orbexa.control.terminal.tracking',
    'TerminalTrackingController': 'orbexa.control.terminal.tracking',
    'TerminalReferenceSample': 'orbexa.control.terminal.tracking',
    'TerminalFeedforwardPolicy': 'orbexa.control.terminal.tracking',
    'TerminalHoldReferences': 'orbexa.control.terminal.tracking',
    'certify_terminal_tracking': 'orbexa.control.terminal.tracking',
    'ContinuationBuilder': 'orbexa.control.fallback.continuation',
    'ContinuationSupervisor': 'orbexa.control.fallback.continuation',
    'HeldControl': 'orbexa.control.fallback.continuation',
    'OrbitalClockBounds': 'orbexa.control.terminal.physical',
    'bound_orbital_clock': 'orbexa.control.terminal.physical',
    'PhysicalTerminalErrorBounds': 'orbexa.control.terminal.physical',
    'PhysicalTerminalBoundsBuilder': 'orbexa.control.terminal.physical',
    'PhysicalTerminalController': 'orbexa.control.terminal.physical',
    'PhysicalTerminalFleet': 'orbexa.control.terminal.physical',
    'PhysicalTerminalResult': 'orbexa.control.terminal.physical',
    'PhysicalTerminalAdmission': 'orbexa.control.terminal.physical',
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
