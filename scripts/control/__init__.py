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
    'MPCController': 'orbexa.control.mpc_controller',
    'MissionResult': 'orbexa.control.mpc_controller',
    'ORBEXAProblemConfig': 'orbexa.control.mpc_problem_builder',
    'build_mpc_problem': 'orbexa.control.mpc_problem_builder',
    'build_from_dynamics': 'orbexa.control.mpc_problem_builder',
    'ancillary_controller': 'orbexa.control.dynamic_tube_model',
    'calc_delta': 'orbexa.control.dynamic_tube_model',
    'calc_d': 'orbexa.control.dynamic_tube_model',
    'TubeProfile': 'orbexa.control.dynamic_tube_model',
    'TerminalRegion': 'orbexa.control.terminal_region',
    'input_tightening_from_profile': 'orbexa.control.dynamic_tube_model',
    'propagate_tube_profile': 'orbexa.control.dynamic_tube_model',
    'tighten_box_bounds': 'orbexa.control.dynamic_tube_model',
    'tighten_target_params': 'orbexa.control.dynamic_tube_model',
    'AffineConstraint': 'orbexa.control.linearization',
    'linearize_cylinder_constraint': 'orbexa.control.linearization',
    'linearize_rendezvous_constraint': 'orbexa.control.linearization',
    'CylinderConstraint': 'orbexa.control.constraints',
    'collision_params_from_target_config': 'orbexa.control.constraints',
    'rendezvous_margin': 'orbexa.control.constraints',
    'rotating_body_point_velocity': 'orbexa.control.constraints',
    'rotating_docking_point': 'orbexa.control.constraints',
    'target_frame_position': 'orbexa.control.constraints',
    'IntervalBox': 'orbexa.control.reachability',
    'OrbitalModelBounds': 'orbexa.control.reachability',
    'HeldInputRequest': 'orbexa.control.reachability',
    'ReachabilityConfig': 'orbexa.control.reachability',
    'ValidatedFlowEncloser': 'orbexa.control.reachability',
    'SweptCollisionGuard': 'orbexa.control.reachability',
    'RotatingCylinderBounds': 'orbexa.control.reachability',
    'RotatingCompositeBounds': 'orbexa.control.composite_geometry',
    'CompositeSweptCollisionGuard': 'orbexa.control.composite_geometry',
    'CompositeTerminalGeometry': 'orbexa.control.composite_geometry',
    'CompositePointSeparation': 'orbexa.control.composite_geometry',
    'CompositePhysicalTerminalBoundsBuilder': 'orbexa.control.composite_terminal',
    'ComponentwiseCompositePhysicalTerminalBoundsBuilder': 'orbexa.control.componentwise_terminal',
    'ComponentwiseTerminalJetBounds': 'orbexa.control.rigid_body_terminal',
    'bound_componentwise_terminal_jets': 'orbexa.control.rigid_body_terminal',
    'RigidBodyRotationBounds': 'orbexa.control.rigid_body_bounds',
    'bound_rigid_body_rotation': 'orbexa.control.rigid_body_bounds',
    'BoundScope': 'orbexa.control.terminal_tracking',
    'TerminalTrackingBounds': 'orbexa.control.terminal_tracking',
    'TerminalGeometryBounds': 'orbexa.control.terminal_tracking',
    'TerminalTrackingController': 'orbexa.control.terminal_tracking',
    'TerminalReferenceSample': 'orbexa.control.terminal_tracking',
    'TerminalFeedforwardPolicy': 'orbexa.control.terminal_tracking',
    'TerminalHoldReferences': 'orbexa.control.terminal_tracking',
    'certify_terminal_tracking': 'orbexa.control.terminal_tracking',
    'ContinuationBuilder': 'orbexa.control.continuation',
    'ContinuationSupervisor': 'orbexa.control.continuation',
    'HeldControl': 'orbexa.control.continuation',
    'OrbitalClockBounds': 'orbexa.control.physical_terminal',
    'bound_orbital_clock': 'orbexa.control.physical_terminal',
    'PhysicalTerminalErrorBounds': 'orbexa.control.physical_terminal',
    'PhysicalTerminalBoundsBuilder': 'orbexa.control.physical_terminal',
    'PhysicalTerminalController': 'orbexa.control.physical_terminal',
    'PhysicalTerminalFleet': 'orbexa.control.physical_terminal',
    'PhysicalTerminalResult': 'orbexa.control.physical_terminal',
    'PhysicalTerminalAdmission': 'orbexa.control.physical_terminal',
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
