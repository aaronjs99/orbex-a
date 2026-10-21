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

"""Dense sampled ancillary feedback on one shared fleet clock."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, TYPE_CHECKING
import numpy as np

from orbexa.control.tubes.model import advance_tube, boundary_layer_feedback
from orbexa.control.tubes.sample_hold import sample_hold_bounds, TUBE_MODEL_VERSION
from orbexa.simulation.numerics.integration import rk4_step

if TYPE_CHECKING:
    from orbexa.estimation.adaptation.mission import MissionAdaptation
    from orbexa.simulation.approach.planning import ChaserPlan, PlanningStep
    from orbexa.simulation.approach.state import CoupledMissionState
    from orbexa.simulation.approach.models import MissionPhase
    from orbexa.simulation.approach.scenario import MissionScenario


@dataclass
class ExecutionStep:
    start_anomaly: float
    end_anomaly: float
    success: bool = True
    messages: List[str] = field(default_factory=list)
    reserve_ratios: Dict[str, float] = field(default_factory=dict)
    applied: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    ancillary: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    containment: Dict[str, float] = field(default_factory=dict)
    clearance: Dict[str, float] = field(default_factory=dict)
    min_spacing: float = np.inf
    sampled_tube_contained: bool = True
    max_nominal_discretization_residual: float = 0.0
    nominal_center_defect: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def committed(self) -> bool:
        return self.end_anomaly != self.start_anomaly


class DenseFeedbackExecutor:
    """Check all inputs, propose all transitions, then commit the complete fleet."""

    def __init__(self, scenario: MissionScenario, adaptation: MissionAdaptation):
        self.scenario = scenario
        self.config = scenario.config
        self.adaptation = adaptation
        self.substeps = self.config.tube.feedback_substeps
        self.dense_anom_step = self.config.anom_step / self.substeps

    def _stage_preflight(self, key, state, plan):
        """Refuse inconsistent sampled-tube data before applying its input.

        The tolerance matches the existing sampled containment arithmetic
        check. It is a consistency diagnostic, not a floating-point proof.
        """
        tube, spec = plan.tube, plan.spec
        count, tolerance = spec.horizon_steps, 1e-8
        if tube.get('tube_model_version') != TUBE_MODEL_VERSION:
            raise ValueError(f'{key}: unsupported sampled tube model version')
        if not np.array_equal(spec.lambda_gain, np.asarray(self.config.tube.sliding_gains)):
            raise ValueError(f'{key}: planned and executed sliding gains disagree')
        tau = float(tube['hold_width'])
        width_tolerance = 16 * np.spacing(max(tau, self.dense_anom_step))
        if (not np.isfinite(tau) or tau <= 0 or
                abs(tau - spec.hold_width) > width_tolerance or
                self.dense_anom_step > tau + width_tolerance):
            raise ValueError(f'{key}: execution feedback hold exceeds the planned maximum')
        shapes = dict(phi=(count,3), omega=(count,3), alpha=(count-1,3),
            phi_peak=(count-1,3), omega_peak=(count-1,3), input_reserve=(count-1,3),
            nominal_enclosure=(count-1,6), base_forcing=(count-1,3), hold_forcing=(count-1,3),
            disturbance_bound=(count-1,3), velocity_peak=(count-1,3), error_peak=(count-1,6),
            velocity_derivative_bound=(count-1,3), error_derivative_bound=(count-1,6),
            sliding_derivative_bound=(count-1,3), A_variation=(count-1,3,6))
        data = {}
        for name, shape in shapes.items():
            value = np.asarray(tube[name], dtype=float)
            if value.shape != shape or not np.all(np.isfinite(value)) or np.any(value < 0):
                raise ValueError(f'{key}: invalid sampled-tube {name}, expected nonnegative {shape}')
            data[name] = value[0].copy()
        if not np.array_equal(data['A_variation'], spec.A_variation[0]):
            raise ValueError(f'{key}: nominal coefficient variation changed after planning')
        alpha = data['alpha']
        if np.any(alpha < spec.alpha_bounds[0]) or np.any(alpha > spec.alpha_bounds[1]):
            raise ValueError(f'{key}: planned bandwidth is outside its allowed interval')
        if (np.max(np.abs(data['phi'] - state.phi[key])) > tolerance or
                np.max(np.abs(data['omega'] - state.omega[key])) > tolerance or
                np.max(np.abs(spec.phi_initial - state.phi[key])) > tolerance or
                np.max(np.abs(spec.omega_initial - state.omega[key])) > tolerance or
                np.max(np.abs(np.asarray(plan.result.state_trajectory)[:,0] - state.centers[key])) > tolerance):
            raise ValueError(f'{key}: sampled tube does not start at the current nominal center and envelope')
        base = spec.delta_A[0] @ data['nominal_enclosure'] + spec.delta_d[0] + spec.eta
        expected = sample_hold_bounds(spec.A_abs[0], spec.delta_A[0], spec.A_variation[0],
            base, data['phi_peak'], data['omega_peak'], alpha, spec.lambda_gain, tau)
        for name, value in expected.items():
            if np.max(np.abs(data[name] - value)) > tolerance:
                raise ValueError(f'{key}: solved {name} is inconsistent with the held-input bound')
        next_phi, next_omega = advance_tube(state.phi[key], state.omega[key], alpha,
            spec.lambda_gain, data['disturbance_bound'], self.config.anom_step,
            phi_peak=data['phi_peak'])
        for name, initial, endpoint in (('phi', state.phi[key], next_phi),
                                         ('omega', state.omega[key], next_omega)):
            peak = data[f'{name}_peak']
            if (np.max(np.maximum(initial, endpoint) - peak) > tolerance or
                    np.max(np.abs(endpoint - np.asarray(tube[name])[1])) > tolerance):
                raise ValueError(f'{key}: solved {name} endpoints and stage peak disagree')
        error = state.states[key] - state.centers[key]
        if (np.max(np.abs(error[:3]) - state.omega[key]) > tolerance or
                np.max(np.abs(error[3:] + spec.lambda_gain * error[:3]) - state.phi[key]) > tolerance):
            raise ValueError(f'{key}: current state is outside the planned initial tube')
        return data

    def advance(self, state: CoupledMissionState, planning: PlanningStep,
                phase: MissionPhase, truth_matrices: tuple) -> ExecutionStep:
        scenario, cfg = self.scenario, self.config
        q = state.anomaly
        ids = list(state.states)
        plans, matrices, params = planning.plans, planning.matrices, planning.parameters
        h, lam = self.dense_anom_step, np.asarray(cfg.tube.sliding_gains)
        outcome = ExecutionStep(start_anomaly=q, end_anomaly=q,
            reserve_ratios={key: 0.0 for key in ids}, applied={key: [] for key in ids},
            ancillary={key: [] for key in ids}, containment={key: np.inf for key in ids},
            clearance={key: np.inf for key in ids})
        try:
            stage = {key: self._stage_preflight(key, state, plans[key]) for key in ids}
        except (ValueError, KeyError, TypeError, IndexError) as error:
            outcome.success = False
            outcome.messages.append(f'Held-input tube preflight rejected the plan: {error}')
            return outcome
        phi_start = {key: state.phi[key].copy() for key in ids}
        omega_start = {key: state.omega[key].copy() for key in ids}
        for inner in range(self.substeps):
            current = q + inner * h
            proposed = {}
            proposed_centers, proposed_phi, proposed_omega, proposed_bandwidth = {}, {}, {}, {}
            inputs, feedbacks = {}, {}
            for key in ids:
                sol = plans[key]
                u_nom = np.asarray(sol.result.control_trajectory)[:, 0]
                alpha = stage[key]['alpha']
                a = np.asarray(matrices[0](current, params['time_periapsis']), dtype=float)[3:]
                feedback = boundary_layer_feedback(a, sol.spec.delta_A[0], state.states[key],
                    state.centers[key], state.phi[key], alpha, lam)
                u = u_nom + feedback
                reserve = stage[key]['input_reserve']
                outcome.reserve_ratios[key] = max(outcome.reserve_ratios[key],
                    float(np.max(np.abs(feedback) / np.maximum(reserve, 1e-12))))
                if np.max(np.abs(feedback) - reserve) > 1e-5:
                    outcome.success = False
                    outcome.messages.append(f'{key}: ancillary feedback exceeded the planned input reserve')
                    break
                clipped = scenario.safety.clip_control(u)
                if np.max(np.abs(clipped - u)) > 1e-8:
                    outcome.success = False
                    outcome.messages.append(f'{key}: ancillary input would saturate; tube policy interrupted')
                    break
                inputs[key], feedbacks[key] = u, feedback
            if not outcome.success:
                break
            if scenario.execution_guard is not None:
                admitted = scenario.execution_guard.prepare(state.states, inputs, current, h)
                if not admitted.admissible:
                    outcome.success = False
                    outcome.messages.append('Continuous-flow admission refused the hold: ' + admitted.reason)
                    break
            for key in ids:
                sol = plans[key]
                u_nom = np.asarray(sol.result.control_trajectory)[:, 0]
                alpha = stage[key]['alpha']
                u = inputs[key]
                proposed[key] = rk4_step(state.states[key], u, current, h, truth_matrices, scenario.truth.time_periapsis)
                proposed_centers[key] = rk4_step(state.centers[key], u_nom, current, h, matrices, params['time_periapsis'])
                # Use the exact solved total forcing and fixed stage Phi peak
                # from the same comparison as the outer shooting equations.
                proposed_phi[key], proposed_omega[key] = advance_tube(phi_start[key], omega_start[key], alpha, lam,
                    stage[key]['disturbance_bound'], (inner + 1) * h, phi_peak=stage[key]['phi_peak'])
                if (np.max(proposed_phi[key] - stage[key]['phi_peak']) > 1e-8 or
                        np.max(proposed_omega[key] - stage[key]['omega_peak']) > 1e-8):
                    outcome.success = False
                    outcome.messages.append(f'{key}: dense envelope exceeded the solved stage peak before state commit')
                    break
                proposed_bandwidth[key] = alpha
            if not outcome.success:
                break
            state.commit(current + h, states=proposed, centers=proposed_centers,
                         phi=proposed_phi, omega=proposed_omega, bandwidth=proposed_bandwidth)
            outcome.end_anomaly = current + h
            if scenario.execution_guard is not None:
                scenario.execution_guard.commit(current + h)
            self.adaptation.append_transition(state.states[ids[0]], inputs[ids[0]], current + h)
            for key in ids:
                outcome.applied[key].append(inputs[key])
                outcome.ancillary[key].append(feedbacks[key])
                error = state.states[key] - state.centers[key]
                margin = min(float(np.min(state.omega[key] - np.abs(error[:3]))),
                             float(np.min(state.phi[key] - np.abs(error[3:] + lam * error[:3]))))
                clearance = scenario.safety.active_margin_for_state(state.states[key], current + h, phase)
                outcome.containment[key] = min(outcome.containment[key], margin)
                outcome.clearance[key] = min(outcome.clearance[key], clearance)
                outcome.sampled_tube_contained &= margin >= -1e-8
                if margin < -1e-8 or clearance < -scenario.safety.target_safety_tolerance:
                    outcome.success = False
                    outcome.messages.append(f'{key}: dense containment or clearance violated at q={current+h:.9g}')
            for i, key in enumerate(ids):
                for other in ids[i + 1:]:
                    distance = float(np.linalg.norm(state.states[key][:3] - state.states[other][:3]))
                    outcome.min_spacing = min(outcome.min_spacing, distance)
                    if distance < scenario.safety.min_chaser_separation - scenario.safety.target_safety_tolerance:
                        outcome.success = False
                        outcome.messages.append('dense pairwise spacing violated')
            if not outcome.success:
                break
        if outcome.committed and outcome.success:
            for key in ids:
                sol = plans[key]
                defect = state.centers[key] - np.asarray(sol.result.state_trajectory)[:, 1]
                outcome.nominal_center_defect[key] = defect.copy()
                residual = np.linalg.norm(defect)
                outcome.max_nominal_discretization_residual = max(outcome.max_nominal_discretization_residual, float(residual))
        return outcome
