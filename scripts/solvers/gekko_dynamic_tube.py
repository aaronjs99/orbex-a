"""Coupled nominal and dynamic-tube multiple shooting for GEKKO/IPOPT.

The independent variable is true anomaly. Controls and bandwidths are held
over each interval; nominal motion uses RK4 and positive tube comparison
systems use exponential updates. No complementarity abs/max operators are
used. The caller supplies interval-valid coefficient and disturbance bounds.
"""

import time
from dataclasses import replace
from numbers import Real
import numpy as np
from scipy.linalg import expm
from gekko import GEKKO

from orbexa.solvers.base import SolverResult
from orbexa.control.dynamic_tube_model import held_tube_stage_seed, advance_tube
from orbexa.control.sample_hold import sample_hold_expressions, TUBE_MODEL_VERSION
from orbexa.control.terminal_region import TerminalRegion
from orbexa.utils.math_utils import tait_bryan_to_rotation_matrix
from orbexa.utils.solver_options import gekko_memory_exponent, composite_geometry_mode


class GekkoDynamicTube:
    """Optional six-state, three-acceleration coupled MPC formulation."""

    def __init__(self, problem, config):
        self.problem = problem
        self.config = config
        self.max_memory = gekko_memory_exponent(config.get("max_memory", 4))
        self.composite_geometry_mode = composite_geometry_mode(config.get("composite_geometry_mode", "free"))
        self.spec = problem.extra_params["dynamic_tube"]
        self.explicit_terminal_policy = any(self.spec.get(key) is not None
            for key in ("terminal_start_index", "terminal_region", "terminal_planning_fraction"))
        reserve = config.get("terminal_solve_reserve", 1e-6)
        if isinstance(reserve, bool) or not isinstance(reserve, Real):
            raise ValueError("terminal_solve_reserve must be a finite real in [0,1)")
        try:
            reserve = float(reserve)
        except (ValueError, OverflowError) as error:
            raise ValueError("terminal_solve_reserve must be a finite real in [0,1)") from error
        if not np.isfinite(reserve) or not 0 <= reserve < 1:
            raise ValueError("terminal_solve_reserve must be a finite real in [0,1)")
        self.terminal_solve_reserve = reserve if self.explicit_terminal_policy else 0.
        fraction = self.spec.get("terminal_planning_fraction", 1.)
        if fraction is None:
            fraction = 1.
        if (isinstance(fraction, bool) or not isinstance(fraction, Real) or not np.isfinite(float(fraction))
                or not 0 < fraction <= 1):
            raise ValueError("Terminal planning fraction must be a finite real in (0,1]")
        self.terminal_planning_fraction = float(fraction)
        self.terminal_start_index = self.spec.get("terminal_start_index")
        if self.terminal_start_index is None:
            self.terminal_start_index = problem.num_steps - 1
        if (isinstance(self.terminal_start_index, bool) or
                not isinstance(self.terminal_start_index, (int, np.integer)) or
                not 0 <= self.terminal_start_index < problem.num_steps):
            raise ValueError("Terminal suffix start must be an integer index inside the prediction horizon")
        self.terminal_region = self.spec.get("terminal_region")
        if self.terminal_region is not None:
            if isinstance(self.terminal_region, dict):
                self.terminal_region = TerminalRegion.from_dict(self.terminal_region)
            if not isinstance(self.terminal_region, TerminalRegion) or self.terminal_region.frame != 'physical-lvlh-v1':
                raise ValueError("Terminal region must use the declared physical-lvlh-v1 state frame")
            start = problem.extra_params.get("start_anom", 0.)
            if any(not self.terminal_region.contains_anomaly(start + k * problem.anom_step)
                   for k in range(self.terminal_start_index, problem.num_steps)):
                raise ValueError("Terminal suffix lies outside its reference error declaration domain")
        if self.explicit_terminal_policy and self.spec.get("enforce_terminal") is not True:
            raise ValueError("Explicit terminal suffix/region requires enabled terminal constraints")
        self.model = GEKKO(remote=config.get("remote", False))
        self.residuals = []
        self.residual_descriptions = []
        self.composite_geometry = None
        self.constraint_context = 'formulation'
        try:
            self._build()
        except Exception:
            self.model.cleanup()
            raise

    def _array(self, key, shape, nonnegative=False):
        value = np.asarray(self.spec[key], dtype=float)
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError("dynamic_tube.%s must be finite with shape %s" % (key, shape))
        if nonnegative and np.any(value < 0):
            raise ValueError("dynamic_tube.%s must be nonnegative" % key)
        return value

    def _eq(self, left, right=0.0):
        residual = self.model.Intermediate(left - right)
        self.model.Equation(residual == 0)
        self.residuals.append((residual, True))
        self.residual_descriptions.append(self.constraint_context)

    def _ge(self, left, right=0.0):
        residual = self.model.Intermediate(left - right)
        self.model.Equation(residual >= 0)
        self.residuals.append((residual, False))
        self.residual_descriptions.append(self.constraint_context)

    def _abs(self, value, seed=0.0):
        bound = self.model.Var(value=abs(float(seed)), lb=0.0)
        self._ge(bound, value)
        self._ge(bound, -value)
        # Tight epigraphs improve conditioning when an uncertainty row is zero.
        self.model.Minimize(1.0e-6 * bound)
        return bound

    def _norm(self, values):
        # The tiny positive term makes the derivative finite at a zero tube.
        return self.model.sqrt(sum(value * value for value in values) + 1.0e-20)

    @staticmethod
    def _dot(matrix, vector):
        return [sum(float(a) * b for a, b in zip(row, vector) if a != 0) for row in matrix]

    @staticmethod
    def _bound(bound, key):
        value = bound.get(key)
        if value is None:
            return -np.inf if key == "lower" else np.inf
        return float(value)

    def _coefficients(self, q):
        p, extra = self.problem, self.problem.extra_params
        tp = extra.get("time_periapsis", 0.0)
        a_func = extra.get("dynamics_callable")
        a = np.asarray(a_func(q, tp) if callable(a_func) else p.dynamics_matrix, dtype=float)
        d_func = extra.get("disturbance_callable")
        d_raw = extra.get("disturbance_vector")
        d = np.asarray(d_func(q, tp) if callable(d_func) else
                       np.zeros(6) if d_raw is None else d_raw, dtype=float)
        if a.shape != (6, 6) or d.shape != (6,) or not np.all(np.isfinite(a)) or not np.all(np.isfinite(d)):
            raise ValueError("Dynamic tube requires finite six-state dynamics")
        return a, d

    def _rk4(self, state, control, coefficients):
        h = self.problem.anom_step
        b = np.asarray(self.problem.input_matrix, dtype=float)
        def rhs(x, pair):
            a, d = pair
            ax, bu = self._dot(a, x), self._dot(b, control)
            return [self.model.Intermediate(ax[i] + bu[i] + float(d[i])) for i in range(6)]
        k1 = rhs(state, coefficients[0])
        k2 = rhs([state[i] + h * k1[i] / 2 for i in range(6)], coefficients[1])
        k3 = rhs([state[i] + h * k2[i] / 2 for i in range(6)], coefficients[1])
        k4 = rhs([state[i] + h * k3[i] for i in range(6)], coefficients[2])
        return [state[i] + h * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) / 6 for i in range(6)]

    def _guess(self, key, shape, fallback):
        value = self.problem.extra_params.get(key)
        if value is None:
            return fallback
        value = np.asarray(value, dtype=float)
        if value.shape != shape and value.shape == shape[::-1]:
            value = value.T
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError("%s must have shape %s" % (key, shape))
        return value

    def _numeric_step(self, state, control, coefficients):
        """Dynamics-feasible initial iterate for the held-input RK4 equations."""
        h = self.problem.anom_step
        bu = np.asarray(self.problem.input_matrix) @ control
        def rhs(x, pair):
            return pair[0] @ x + bu + pair[1]
        k1 = rhs(state, coefficients[0])
        k2 = rhs(state+h*k1/2, coefficients[1])
        k3 = rhs(state+h*k2/2, coefficients[1])
        k4 = rhs(state+h*k3, coefficients[2])
        return state+h*(k1+2*k2+2*k3+k4)/6

    def _build(self):
        p, m, spec = self.problem, self.model, self.spec
        if p.num_states != 6 or p.num_inputs != 3 or p.dynamics_type != "continuous":
            raise ValueError("Dynamic tube requires continuous six-state/three-input dynamics")
        if p.extra_params.get("use_anomaly_scaling", False):
            raise ValueError("Dynamic tube dynamics must already use true anomaly")
        expected_b = np.vstack((np.zeros((3, 3)), np.eye(3)))
        if not np.allclose(p.input_matrix, expected_b):
            raise ValueError("Dynamic tube inputs must be acceleration coordinates, B=[0;I]")
        h, count = float(p.anom_step), p.num_steps
        if not np.isfinite(h) or h <= 0:
            raise ValueError("Dynamic tube anomaly step must be positive")
        lam = self._array("lambda_gain", (3,), True)
        if np.any(lam <= 0):
            raise ValueError("Dynamic tube lambda gains must be positive")
        phi0 = self._array("phi_initial", (3,), True)
        omega0 = self._array("omega_initial", (3,), True)
        alpha0 = self._array("alpha_initial", (3,), True)
        initial_state = np.asarray(p.initial_state, dtype=float)
        if initial_state.shape != (6,) or not np.all(np.isfinite(initial_state)):
            raise ValueError("Dynamic tube initial state must contain six finite values")
        initial_envelope = np.r_[omega0, phi0 + lam * omega0]
        if not np.all(np.isfinite(initial_envelope)):
            raise ValueError("Dynamic tube initial state envelope must be finite")
        # Initial values become parameters below. Validate their complete box
        # explicitly so parameterization cannot bypass a declared state bound.
        for i, bound in enumerate((p.state_bounds or [])[:6]):
            lower, upper = self._bound(bound, "lower"), self._bound(bound, "upper")
            if np.isnan(lower) or np.isnan(upper) or lower > upper:
                raise ValueError("Dynamic tube state bounds must be ordered and not NaN")
            if initial_state[i] - initial_envelope[i] < lower or initial_state[i] + initial_envelope[i] > upper:
                raise ValueError("Initial dynamic-tube state box violates a declared state bound")
        da = self._array("delta_A", (count-1, 3, 6), True)
        dd = self._array("delta_d", (count-1, 3), True)
        eta = self._array("eta", (3,), True)
        a_abs = self._array("A_abs", (count-1, 3, 6), True)
        a_variation = self._array("A_variation", (count-1, 3, 6), True)
        hold_width = float(spec["hold_width"])
        if not np.isfinite(hold_width) or not 0 < hold_width <= h:
            raise ValueError("Dynamic tube hold_width must be positive and no larger than the outer step")
        lower_alpha, upper_alpha = map(float, spec["alpha_bounds"])
        if not (0 < lower_alpha <= upper_alpha < np.inf):
            raise ValueError("Dynamic tube alpha_bounds must be finite and strictly positive")
        rate = float(spec.get("alpha_rate_limit", np.inf))
        if np.isnan(rate) or rate < 0:
            raise ValueError("Dynamic tube alpha_rate_limit must be nonnegative")
        if np.any(alpha0 < lower_alpha) or np.any(alpha0 > upper_alpha):
            raise ValueError('Initial bandwidth must belong to its allowed interval')
        fixed_bandwidth = rate == 0 or lower_alpha == upper_alpha
        tube_weight = float(spec.get("tube_weight", 1.0))
        bandwidth_weight = float(spec.get("bandwidth_weight", 1.0e-4))
        if min(tube_weight, bandwidth_weight) < 0 or not np.isfinite(tube_weight + bandwidth_weight):
            raise ValueError("Dynamic tube objective weights must be finite and nonnegative")
        start = float(p.extra_params.get("start_anom", 0.0))
        self.coefficients = [tuple(self._coefficients(start + (k+s)*h) for s in (0, .5, 1))
                             for k in range(count-1)]
        if "d_abs" in spec:
            d_abs = self._array("d_abs", (count-1, 3), True)
        elif callable(p.extra_params.get("disturbance_callable")):
            raise ValueError("dynamic_tube.d_abs interval bounds are required for callable disturbance")
        else:
            d_abs = np.array([np.abs(c[0][1][3:]) for c in self.coefficients])
        # The position equations must be p'=v for the comparison construction.
        for k, stages in enumerate(self.coefficients):
            for a, d in stages:
                if not np.allclose(a[:3], np.hstack((np.zeros((3,3)), np.eye(3)))) or not np.allclose(d[:3], 0):
                    raise ValueError("Dynamic tube requires position derivative equal to anomaly velocity")
                if np.any(np.abs(a[3:]) > a_abs[k] + 1e-10) or np.any(np.abs(d[3:]) > d_abs[k] + 1e-10):
                    raise ValueError("Dynamic tube interval coefficient bounds do not cover nominal samples")
        input_fallback = p.reference_inputs.copy()
        if p.input_reference is None and "u_0" in p.extra_params:
            input_fallback = np.repeat(np.asarray(p.extra_params["u_0"],dtype=float)[:,None],count-1,axis=1)
        uguess = self._guess("control_initial_guess", (3, count-1), input_fallback)
        xguess = np.zeros((6,count))
        xguess[:,0] = p.initial_state
        for k in range(count-1):
            xguess[:,k+1] = self._numeric_step(xguess[:,k],uguess[:,k],self.coefficients[k])
        xguess = self._guess("state_initial_guess", (6,count), xguess)
        xguess[:,0] = p.initial_state
        # Initialize all tube variables from the same comparison equations as
        # the NLP. Repeating the initial tube at every node creates a large,
        # avoidable infeasibility just when the hard terminal set is armed.
        transitions, phi_guess, omega_guess = [], np.zeros((count,3)), np.zeros((count,3))
        phi_peak_guess, omega_peak_guess = np.zeros((count-1,3)), np.zeros((count-1,3))
        self.seed_diagnostics = []
        phi_guess[0], omega_guess[0] = phi0, omega0
        alpha_guess = np.clip(alpha0,lower_alpha,upper_alpha)
        for k in range(count-1):
            comparison = np.vstack((np.hstack((np.zeros((3,3)),np.eye(3))),a_abs[k]))
            augmented = np.zeros((12,12))
            augmented[:6,:6], augmented[:6,6:] = comparison, np.eye(6)
            transition = expm(augmented*h)
            transitions.append(transition)
            source = np.abs(p.input_matrix) @ np.abs(uguess[:,k]) + np.concatenate((np.zeros(3),d_abs[k]))
            enclosure = transition[:6,:6] @ np.abs(xguess[:,k]) + transition[:6,6:] @ source
            w_guess = da[k] @ enclosure + dd[k] + eta
            try:
                phi_guess[k+1], omega_guess[k+1], phi_peak_guess[k], omega_peak_guess[k], _ = held_tube_stage_seed(
                    phi_guess[k], omega_guess[k], alpha_guess, lam, a_abs[k], da[k],
                    a_variation[k], w_guess, h, hold_width)
            except ValueError as error:
                # A failed fixed point at this *initial* bandwidth is not a
                # proof that another admissible bandwidth is infeasible. Keep
                # a finite initial iterate; all corrected equations and hard
                # constraints remain present in the actual optimization.
                phi_guess[k+1], omega_guess[k+1] = advance_tube(
                    phi_guess[k], omega_guess[k], alpha_guess, lam, w_guess, h)
                phi_peak_guess[k] = np.maximum(phi_guess[k], phi_guess[k+1])
                omega_peak_guess[k] = np.maximum(omega_guess[k], omega_guess[k+1])
                self.seed_diagnostics.append(dict(stage=k, initialization='finite_uncoupled_initial_iterate',
                                                  reason=str(error)))
        # These values are prescribed data, not optimization directions. In
        # particular, Omega0==0 together with its nonnegative variable bound
        # creates a redundant active constraint and singular norm curvature.
        self.states = [[m.Param(value=float(initial_state[i])) if k == 0 else
                        m.Var(value=float(xguess[i,k])) for i in range(6)] for k in range(count)]
        self.inputs = [[m.Var(value=float(uguess[i,k])) for i in range(3)] for k in range(count-1)]
        self.phi = [[m.Param(value=float(phi0[i])) if k == 0 else
                     m.Var(value=float(phi_guess[k,i]), lb=0) for i in range(3)] for k in range(count)]
        self.omega = [[m.Param(value=float(omega0[i])) if k == 0 else
                       m.Var(value=float(omega_guess[k,i]), lb=0) for i in range(3)] for k in range(count)]
        # A prescribed gain is a parameter. Equal variable bounds plus two
        # zero-rate inequalities create redundant active NLP constraints.
        self.alpha = [[m.Param(value=float(alpha0[i])) if fixed_bandwidth else
                       m.Var(value=float(alpha0[i]), lb=lower_alpha, ub=upper_alpha)
                       for i in range(3)] for _ in range(count-1)]
        self.bandwidth_mode = 'fixed_parameter' if fixed_bandwidth else 'optimized_dynamic_bandwidth'
        self.reserve = []
        self.disturbance_bounds = []
        self.nominal_enclosures = []
        self.stage_terms = {name: [] for name in (
            "base_forcing", "hold_forcing", "velocity_peak", "error_peak",
            "velocity_derivative_bound", "error_derivative_bound", "sliding_derivative_bound")}
        self.phi_peaks, self.omega_peaks = [], []
        for k in range(count-1):
            self.constraint_context = f'nominal_dynamics[{k}]'
            x, u, alpha = self.states[k], self.inputs[k], self.alpha[k]
            predicted = self._rk4(x, u, self.coefficients[k])
            for i in range(6):
                self._eq(self.states[k+1][i], predicted[i])
            self.constraint_context = f'absolute_epigraphs[{k}]'
            absx = [self._abs(x[i], xguess[i,k]) for i in range(6)]
            absu = [self._abs(u[i], uguess[i,k]) for i in range(3)]
            # Positive comparison system bounds the entire continuous interval,
            # rather than only the discrete RK4 start/mid/end values.
            transition = transitions[k]
            source = self._dot(np.abs(p.input_matrix), absu)
            source = [source[i] + (float(d_abs[k,i-3]) if i >= 3 else 0.) for i in range(6)]
            e_state = self._dot(transition[:6,:6], absx)
            e_source = self._dot(transition[:6,6:], source)
            enclosure = [m.Intermediate(e_state[i]+e_source[i]) for i in range(6)]
            w0 = [m.Intermediate(sum(float(da[k,i,j])*enclosure[j] for j in range(6))
                                 + float(dd[k,i]+eta[i])) for i in range(3)]
            self.nominal_enclosures.append(enclosure)
            phi_peak, omega_peak = [], []
            self.constraint_context = f'tube_peaks_and_position_decay[{k}]'
            for i in range(3):
                peak = m.Var(value=float(phi_peak_guess[k,i]), lb=0)
                self._ge(peak, self.phi[k][i])
                self._ge(peak, self.phi[k+1][i])
                phi_peak.append(peak)
                omega_decay = float(np.exp(-h*lam[i]))
                self._eq(self.omega[k+1][i], omega_decay*self.omega[k][i]
                         + (1-omega_decay)*peak/float(lam[i]))
                radius = m.Var(value=float(omega_peak_guess[k,i]), lb=0)
                self._ge(radius, self.omega[k][i])
                self._ge(radius, self.omega[k+1][i])
                omega_peak.append(radius)
                if np.isfinite(rate) and not fixed_bandwidth:
                    previous = float(alpha0[i]) if k == 0 else self.alpha[k-1][i]
                    self._ge(rate*h, alpha[i]-previous)
                    self._ge(rate*h, previous-alpha[i])
                m.Minimize(h*bandwidth_weight*alpha[i]**2
                           + h*(tube_weight+1e-4)*(peak**2+radius**2))
            # One coupled set of peaks supplies rho, the comparison forcing
            # and the complete actuator reserve. The bounded-term residual
            # remains valid at Phi=0 without a reciprocal or smallness test.
            terms = sample_hold_expressions(a_abs[k], da[k], a_variation[k], w0,
                phi_peak, omega_peak, alpha, lam, hold_width, intermediate=m.Intermediate)
            w, reserve = terms['disturbance_bound'], terms['input_reserve']
            self.disturbance_bounds.append(w)
            self.phi_peaks.append(phi_peak)
            self.omega_peaks.append(omega_peak)
            for name in self.stage_terms:
                self.stage_terms[name].append(terms[name])
            for i in range(3):
                self.constraint_context = f'sliding_decay_and_input_reserve[{k}]'
                decay = m.exp(-h*alpha[i])
                self._eq(self.phi[k+1][i], decay*self.phi[k][i] + (1-decay)*w[i]/alpha[i])
                margin = reserve[i]
                if p.input_bounds and i < len(p.input_bounds):
                    lower = self._bound(p.input_bounds[i], "lower")
                    upper = self._bound(p.input_bounds[i], "upper")
                    if np.isfinite(lower):
                        self._ge(u[i]-margin, lower)
                    if np.isfinite(upper):
                        self._ge(upper, u[i]+margin)
            self.reserve.append(reserve)
        self.lambda_gain = lam
        self._constraints()
        for k in range(count):
            if k < count-1:
                errors = [self.states[k][i]-float(p.reference_states[i,k]) for i in range(6)]
                m.Minimize(h*sum(float(p.state_cost_matrix[i,j])*errors[i]*errors[j]
                                 for i in range(6) for j in range(6) if p.state_cost_matrix[i,j] != 0))
            # Nodewise tube regularization includes the endpoint. State stage
            # costs stop before it; the Riccati tail already starts there.
            m.Minimize(h*tube_weight*sum(self.phi[k][i]**2+self.omega[k][i]**2 for i in range(3)))
        for k in range(count-1):
            errors = [self.inputs[k][i]-float(p.reference_inputs[i,k]) for i in range(3)]
            m.Minimize(h*sum(float(p.input_cost_matrix[i,j])*errors[i]*errors[j]
                             for i in range(3) for j in range(3) if p.input_cost_matrix[i,j] != 0))
        if p.terminal_cost_matrix is not None:
            errors = [self.states[-1][i]-float(p.reference_states[i,-1]) for i in range(6)]
            m.Minimize(sum(float(p.terminal_cost_matrix[i,j])*errors[i]*errors[j]
                           for i in range(6) for j in range(6) if p.terminal_cost_matrix[i,j] != 0))
        m.options.IMODE = 3
        m.options.SOLVER = 3
        # IPOPT otherwise relaxes bounds on its internal inequality slacks.
        # Projecting those slacks back after the solve can spoil the original
        # nonlinear terminal constraint despite a successful native status.
        # These are hard tube/geometry bounds, so keep them unrelaxed.
        m.solver_options = ["bound_relax_factor 0"]
        m.options.MAX_ITER = self.config.get("max_iter", 3000)
        m.options.MAX_MEMORY = self.max_memory
        # Keep the optimizer's feasibility tolerance below the existing 1e-8
        # runtime comparison replay check. This is numerical consistency, not
        # an integration or floating-point enclosure certificate.
        m.options.OTOL = self.config.get("otol", 1e-9)
        m.options.RTOL = self.config.get("rtol", 1e-9)
        m.options.SCALING = self.config.get('scaling', 1)

    def _constraints(self):
        p, m, spec = self.problem, self.model, self.spec
        lam = np.asarray(spec['lambda_gain'], dtype=float)
        target = p.extra_params.get("target_params") or {}
        center = np.asarray(target.get("center", np.zeros(3)), dtype=float)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("Target center must be a finite three-vector")
        rotations = target.get("rotation_matrices")
        orientations = target.get("orientations")
        if rotations is None:
            if orientations is None:
                angles = np.asarray(target.get("orientation", np.zeros(3)), dtype=float)
                rates = np.asarray(target.get("angular_velocity", np.zeros(3)), dtype=float)
                orientations = angles[None,:]+np.arange(p.num_steps)[:,None]*p.anom_step*rates[None,:]
            orientations = np.asarray(orientations, dtype=float)
            if orientations.shape != (p.num_steps,3):
                raise ValueError("target orientations must have shape (T,3)")
            rotations = np.array([tait_bryan_to_rotation_matrix(a) for a in orientations])
        rotations = np.asarray(rotations, dtype=float)
        if rotations.shape != (p.num_steps,3,3) or not np.all(np.isfinite(rotations)):
            raise ValueError("target rotation_matrices must have shape (T,3,3)")
        if not np.allclose(rotations.transpose(0,2,1) @ rotations,np.eye(3),atol=1e-7) or not np.allclose(np.linalg.det(rotations),1.,atol=1e-7):
            raise ValueError("Target rotation matrices must be proper orthogonal matrices")
        if target.get("shape") == "composite":
            from orbexa.solvers.composite_geometry import GekkoCompositeGeometry, GekkoReferenceCorridorGeometry
            geometry_class = (GekkoReferenceCorridorGeometry if self.composite_geometry_mode == "reference_corridor"
                              else GekkoCompositeGeometry)
            self.composite_geometry = geometry_class(target, rotations, center)
        pairs = p.extra_params.get("pairwise_constraints") or target.get("pairwise_constraints") or []
        pair_data = []
        for pair in pairs:
            positions = np.asarray(pair["reference_positions"], dtype=float)
            if positions.shape == (3,):
                positions = np.tile(positions,(p.num_steps,1))
            radii = np.broadcast_to(np.asarray(pair.get("tube_radius",0.),dtype=float),(p.num_steps,))
            separation = float(pair["min_separation"])
            if positions.shape != (p.num_steps,3) or not np.all(np.isfinite(positions)) or not np.all(np.isfinite(radii)) or np.any(radii < 0) or not np.isfinite(separation) or separation < 0:
                raise ValueError("Pairwise reference_positions/tube_radius have invalid shape or values")
            pair_data.append((positions,radii,separation))
        for k, x in enumerate(self.states):
            self.constraint_context = f'geometry_and_state_bounds[{k}]'
            omega = self.omega[k]
            velocity = [self.phi[k][i]+float(self.lambda_gain[i])*omega[i] for i in range(3)]
            envelope = omega+velocity
            for i, bound in enumerate((p.state_bounds or [])[:6]):
                lower, upper = self._bound(bound,"lower"), self._bound(bound,"upper")
                if np.isfinite(lower):
                    self._ge(x[i]-envelope[i],lower)
                if np.isfinite(upper):
                    self._ge(upper,x[i]+envelope[i])
            radius_tube = self._norm(omega)
            pos = [x[i]-float(center[i]) for i in range(3)]
            if self.composite_geometry is not None:
                self.composite_geometry.add_node(self, k, pos, omega, p.reference_states[:3,k])
                self.constraint_context = f'geometry_and_state_bounds[{k}]'
            elif target.get("operation") == "rendezvous":
                radius = float(target.get("rendezvous_radius",target.get("target_radius",0.)))
                self._ge(sum(v*v for v in pos),(radius+radius_tube)**2)
            elif target.get("operation") == "docking" and target.get("shape") == "cylinder":
                body = self._dot(rotations[k].T,pos)
                radius = float(target["target_radius"])
                half_length = float(target["target_half_length"])
                safety_reserve = float(target.get("safety_reserve",0.))
                if target.get("robust_side_corridor",False):
                    normal = np.asarray(target["docking_normal_body"],dtype=float)
                    if normal.shape != (3,) or np.linalg.norm(normal[:2]) <= 0:
                        raise ValueError("Side corridor needs a nonzero docking_normal_body")
                    normal = np.array([normal[0],normal[1],0.])/np.linalg.norm(normal[:2])
                    box_support = sum(abs(float(value))*omega[i]
                                      for i,value in enumerate(rotations[k] @ normal))
                    clearance = box_support + safety_reserve
                    self._ge(sum(float(normal[i])*body[i] for i in range(3)),radius+clearance)
                else:
                    # A unit support-plane normal supplies the nonlinear union
                    # (side OR cap) without abs2/max2 complementarity operators.
                    guess = rotations[k].T @ (p.reference_states[:3,k]-center)
                    norm_guess = np.linalg.norm(guess)
                    guess = guess/norm_guess if norm_guess > 1e-12 else np.array([1.,0.,0.])
                    normal = [m.Var(value=float(guess[i]),lb=-1,ub=1) for i in range(3)]
                    self._eq(sum(n*n for n in normal),1.)
                    nz_abs = self._abs(normal[2],guess[2])
                    support = radius*self._norm(normal[:2])+half_length*nz_abs
                    world_normal = self._dot(rotations[k], normal)
                    box_support = sum(self._abs(world_normal[i],(rotations[k] @ guess)[i])*omega[i]
                                      for i in range(3))
                    clearance = box_support + safety_reserve
                    self._ge(sum(normal[i]*body[i] for i in range(3)),support+clearance)
            for positions,radii,separation in pair_data:
                delta = [x[i]-float(positions[k,i]) for i in range(3)]
                self._ge(sum(v*v for v in delta),(separation+float(radii[k])+radius_tube)**2)
            for constraint in p.extra_params.get("affine_constraints") or []:
                if hasattr(constraint,"normal"):
                    row = np.asarray(constraint.normal,dtype=float)
                    offset = float(constraint.offset)
                else:
                    row = np.asarray(constraint["normal"],dtype=float)
                    offset = float(constraint["offset"])
                robust = sum(abs(float(row[i]))*envelope[i] for i in range(len(row)))
                self._ge(sum(float(row[i])*x[i] for i in range(len(row)))+offset,robust)
        if spec.get("enforce_terminal",False):
            for k in range(self.terminal_start_index, p.num_steps):
                # Fixed data at node zero must satisfy the outer admission set;
                # every future node reserves the explicit planning interior for
                # numerical carrying and the independent endpoint enclosure.
                fraction = self.terminal_planning_fraction if k > 0 else 1.
                native_fraction = 1. - self.terminal_solve_reserve if k > 0 else 1.
                self.constraint_context = ('normalized_terminal_pose[%s]' % k if self.explicit_terminal_policy
                                           else 'normalized_terminal_pose')
                velocity = [self.phi[k][i] + lam[i] * self.omega[k][i] for i in range(3)]
                for offset,key,tube in ((0,"terminal_position_tolerance",self.omega[k]),
                                         (3,"terminal_velocity_tolerance",velocity)):
                    tolerance = spec.get(key)
                    if tolerance is None or not np.isfinite(float(tolerance)) or float(tolerance) <= 0:
                        raise ValueError("Hard terminal constraints require positive %s" % key)
                    tolerance = float(tolerance) * fraction * native_fraction
                    error = [self.states[k][offset+i]-float(p.reference_states[offset+i,k]) for i in range(3)]
                    # Exact box worst-case Euclidean error, with convex absolute
                    # epigraphs. Squaring avoids the zero-norm derivative.
                    absolute_error = [self._abs(error[i]) for i in range(3)]
                    self._ge(1.,sum(((absolute_error[i]+tube[i])/float(tolerance))**2 for i in range(3)))
                if self.terminal_region is not None:
                    self.constraint_context = 'required_terminal_region[%s]' % k
                    region = self.terminal_region.scaled(fraction)
                    ep = [self.states[k][i] - float(p.reference_states[i,k]) for i in range(3)]
                    ev = [self.states[k][i+3] - float(p.reference_states[i+3,k]) for i in range(3)]
                    for i in range(3):
                        position = self._abs(ep[i]) + self.omega[k][i] + float(region.reference_state_error_abs[i])
                        sliding = (self._abs(ev[i] + float(region.lambda_gain[i]) * ep[i]) + self.phi[k][i]
                            + abs(float(region.lambda_gain[i]) - lam[i]) * self.omega[k][i]
                            + float(region.reference_state_error_abs[i+3]
                                    + region.lambda_gain[i] * region.reference_state_error_abs[i]))
                        self._ge(1., position / (float(region.omega[i])*native_fraction))
                        self._ge(1., sliding / (float(region.phi[i])*native_fraction))

    @staticmethod
    def _values(rows):
        return np.array([[float(value.value[0]) for value in row] for row in rows])

    def _terminal_suffix_readback(self, states, tube):
        """Recheck the printed full tube at every explicitly requested suffix node."""
        if not self.explicit_terminal_policy:
            return None
        spec, region = self.spec, self.terminal_region
        lam = np.asarray(spec['lambda_gain'], dtype=float)
        nodes, violation = [], 0.
        for k in range(self.terminal_start_index, self.problem.num_steps):
            fraction = self.terminal_planning_fraction if k > 0 else 1.
            position_tolerance = float(spec['terminal_position_tolerance']) * fraction
            velocity_tolerance = float(spec['terminal_velocity_tolerance']) * fraction
            reference = self.problem.reference_states[:, k]
            dp = float(np.linalg.norm(np.abs(states[:3,k] - reference[:3]) + tube['omega'][k]))
            dv = float(np.linalg.norm(np.abs(states[3:,k] - reference[3:]) + tube['phi'][k] + lam*tube['omega'][k]))
            violation = max(violation, (dp / position_tolerance)**2 - 1., (dv / velocity_tolerance)**2 - 1.)
            row = dict(index=k, whole_position_bound_m=dp, whole_velocity_bound_m_per_rad=dv,
                planning_fraction=fraction, position_tolerance_m=position_tolerance,
                velocity_tolerance_m_per_rad=velocity_tolerance)
            if region is not None:
                node_region = region.scaled(fraction)
                position, sliding = node_region.tube_margins(states[:,k], reference, tube['phi'][k], tube['omega'][k], lam)
                row.update(position_region_margins=[float(value) for value in position],
                           sliding_region_margins=[float(value) for value in sliding])
                violation = max(violation, *(float(-value / bound) for value, bound in zip(position, node_region.omega)),
                                *(float(-value / bound) for value, bound in zip(sliding, node_region.phi)))
            row['native_solve_fraction'] = fraction * (1. - self.terminal_solve_reserve if k > 0 else 1.)
            nodes.append(row)
        return dict(start_index=int(self.terminal_start_index), checked_nodes=nodes,
                    maximum_normalized_violation=violation,
                    planning_fraction=self.terminal_planning_fraction,
                    fixed_node_zero_policy='Unchanged outer admission; all future suffix nodes use the planning interior',
                    native_solve_reserve=dict(value=self.terminal_solve_reserve, future_only=True,
                        stricter_native_set=self.terminal_solve_reserve > 0.,
                        original_readback_unchanged=True, universal_serialization_guarantee=False),
                    required_region=None if region is None else region.as_dict(),
                    scope='Numerical whole-tube suffix/entry screen; continuous flow and physical terminal certification are separate')

    def _solve_once(self):
        started = time.time()
        try:
            self.model.solve(disp=self.config.get("disp",False))
            states = self._values(self.states).T
            controls = self._values(self.inputs).T
            tube = {"phi":self._values(self.phi), "omega":self._values(self.omega),
                    "alpha":self._values(self.alpha), "input_reserve":self._values(self.reserve),
                    "disturbance_bound":self._values(self.disturbance_bounds),
                    "nominal_enclosure":self._values(self.nominal_enclosures),
                    "phi_peak":self._values(self.phi_peaks), "omega_peak":self._values(self.omega_peaks),
                    "A_variation":np.asarray(self.spec['A_variation']).copy(),
                    "hold_width":float(self.spec['hold_width'])}
            tube.update({name:self._values(rows) for name,rows in self.stage_terms.items()})
            residuals = [(float(value.value[0]),equality) for value,equality in self.residuals]
            violation = max([abs(value) if equality else max(0.,-value) for value,equality in residuals]+[0.])
            residual_details = sorted((dict(index=i,context=self.residual_descriptions[i],
                equality=equality,residual=value,violation=abs(value) if equality else max(0.,-value))
                for i,(value,equality) in enumerate(residuals)), key=lambda x:x['violation'],reverse=True)[:8]
            geometry_info = None
            if self.composite_geometry is not None:
                geometry_margins = self.composite_geometry.numerical_margins(states, tube["omega"])
                geometry_info = self.composite_geometry.summary(geometry_margins)
                violation = max(violation, -geometry_info["minimum_support_margin_m"])
            terminal_info = self._terminal_suffix_readback(states, tube)
            if terminal_info is not None:
                violation = max(violation, terminal_info['maximum_normalized_violation'])
            lower_alpha, upper_alpha = map(float,self.spec["alpha_bounds"])
            violation = max(violation,float(np.max(-tube["phi"])),float(np.max(-tube["omega"])),
                            float(np.max(lower_alpha-tube["alpha"])),
                            float(np.max(tube["alpha"]-upper_alpha)))
            finite = all(np.all(np.isfinite(value)) for value in [states,controls]+list(tube.values()))
            finite = finite and all(np.isfinite(value) for value,_ in residuals)
            tube["tube_model_version"] = TUBE_MODEL_VERSION
            tolerance = float(self.config.get("residual_tolerance",1e-8))
            success = finite and violation <= tolerance and self.model.options.APPSTATUS == 1
            return SolverResult(success=success,state_trajectory=states,control_trajectory=controls,
                                cost=float(self.model.options.objfcnval),solve_time=time.time()-started,
                                message="Coupled dynamic-tube optimization successful" if success else
                                "Coupled dynamic-tube residual verification failed",
                                solver_info={"status":self.model.options.APPSTATUS,
                                             "iterations":self.model.options.ITERATIONS,
                                             "max_constraint_violation":violation,
                                             "largest_constraint_residuals":residual_details,
                                             "bandwidth_mode":self.bandwidth_mode,
                                             "composite_geometry_mode":self.composite_geometry_mode,
                                             "composite_geometry":geometry_info,
                                             "residual_tolerance":tolerance,
                                             "comparison_replay_tolerance":1e-8,
                                             "rtol":float(self.model.options.RTOL),
                                             "otol":float(self.model.options.OTOL),
                                             "native_solver_options":list(self.model.solver_options),
                                             "terminal_residual_scaling":"squared box error divided by squared physical tolerance",
                                             "tube_seed_diagnostics":list(self.seed_diagnostics),
                                             "constraint_sampling":"nonlinear geometry and state constraints at prediction nodes",
                                              **({"terminal_suffix":terminal_info} if terminal_info is not None else {}),
                                             "dynamic_tube":tube})
        except Exception as error:
            return SolverResult(success=False,solve_time=time.time()-started,
                                message="Coupled GEKKO dynamic-tube solve failed: %s" % error,
                                solver_info={"composite_geometry_mode":self.composite_geometry_mode})

    def solve(self):
        """Bounded initialization and numerical-accuracy retries after failure.

        One cold initialization retains the same solver options. A final
        residual-only refinement tightens numerical accuracy; neighbor paths,
        physical constraints, uncertainty bounds and admission stay identical.
        A rejected residual check is a failed attempt too; neither optimizer
        status nor retrying implies feasibility.
        """
        started = time.time()
        warm_keys = ("state_initial_guess", "control_initial_guess")
        has_warm_start = any(key in self.problem.extra_params for key in warm_keys)
        attempts = []

        def record(result, initialization):
            attempts.append(dict(initialization=initialization,success=bool(result.success),
                                 solve_time=float(result.solve_time),message=result.message,
                                 max_constraint_violation=result.solver_info.get("max_constraint_violation"),
                                 largest_constraint_residuals=result.solver_info.get('largest_constraint_residuals', [])))

        result = self._solve_once()
        record(result,"caller_warm_start" if has_warm_start else "dynamics_rollout")
        if not result.success and has_warm_start and self.config.get("cold_start_retry",True):
            # Rebuilding avoids inheriting IPOPT's failed iterate or stale
            # epigraph slacks. This is bounded to one alternate initialization.
            extra = {key:value for key,value in self.problem.extra_params.items() if key not in warm_keys}
            try:
                candidate = GekkoDynamicTube(replace(self.problem,extra_params=extra),self.config)
                retry = candidate._solve_once()
                record(retry,"fresh_dynamics_rollout")
                self.model.cleanup()
                self.__dict__.update(candidate.__dict__)
                result = retry
            except Exception as error:
                attempts.append(dict(initialization="fresh_dynamics_rollout",success=False,
                                     solve_time=0.0,message="Alternate initialization failed: %s" % error,
                                     max_constraint_violation=None))
                result.message += "; alternate initialization failed: %s" % error
        if (not result.success and result.message == 'Coupled dynamic-tube residual verification failed'
                and self.config.get('residual_refinement_retry', True)):
            # Refine a numerically rejected candidate under the same physical
            # constraints. Tightening solver accuracy never relaxes admission.
            extra = dict(self.problem.extra_params,
                state_initial_guess=result.state_trajectory, control_initial_guess=result.control_trajectory)
            refined_config = dict(self.config, rtol=min(float(self.config.get('rtol',1e-9)),1e-12),
                                  otol=min(float(self.config.get('otol',1e-9)),1e-12),scaling=0)
            try:
                candidate = GekkoDynamicTube(replace(self.problem,extra_params=extra),refined_config)
                refined = candidate._solve_once()
                record(refined,'strict_residual_refinement')
                self.model.cleanup()
                self.__dict__.update(candidate.__dict__)
                result = refined
            except Exception as error:
                attempts.append(dict(initialization='strict_residual_refinement',success=False,
                    solve_time=0.,message=f'Residual refinement failed: {error}',max_constraint_violation=None))
        result.solve_time = time.time()-started
        result.solver_info["attempts"] = attempts
        result.solver_info["initialization_retry_count"] = len(attempts)-1
        return result

    def cleanup(self):
        self.model.cleanup()
