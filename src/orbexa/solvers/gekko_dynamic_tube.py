"""Coupled nominal and dynamic-tube multiple shooting for GEKKO/IPOPT.

The independent variable is true anomaly. Controls and bandwidths are held
over each interval; nominal motion uses RK4 and positive tube comparison
systems use exponential updates. No complementarity abs/max operators are
used. The caller supplies interval-valid coefficient and disturbance bounds.
"""

import time
from dataclasses import replace
import numpy as np
from scipy.linalg import expm
from gekko import GEKKO

from orbexa.solvers.base import SolverResult
from orbexa.utils.math_utils import tait_bryan_to_rotation_matrix


class GekkoDynamicTube:
    """Optional six-state, three-acceleration coupled MPC formulation."""

    def __init__(self, problem, config):
        self.problem = problem
        self.config = config
        self.spec = problem.extra_params["dynamic_tube"]
        self.model = GEKKO(remote=config.get("remote", False))
        self.residuals = []
        self._build()

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

    def _ge(self, left, right=0.0):
        residual = self.model.Intermediate(left - right)
        self.model.Equation(residual >= 0)
        self.residuals.append((residual, False))

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
        da = self._array("delta_A", (count-1, 3, 6), True)
        dd = self._array("delta_d", (count-1, 3), True)
        eta = self._array("eta", (3,), True)
        a_abs = self._array("A_abs", (count-1, 3, 6), True)
        lower_alpha, upper_alpha = map(float, spec["alpha_bounds"])
        if not (0 < lower_alpha <= upper_alpha < np.inf):
            raise ValueError("Dynamic tube alpha_bounds must be finite and strictly positive")
        rate = float(spec.get("alpha_rate_limit", np.inf))
        if np.isnan(rate) or rate < 0:
            raise ValueError("Dynamic tube alpha_rate_limit must be nonnegative")
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
            alpha_decay = np.exp(-h*alpha_guess)
            phi_guess[k+1] = alpha_decay*phi_guess[k] + (-np.expm1(-h*alpha_guess))*w_guess/alpha_guess
            omega_decay = np.exp(-h*lam)
            omega_guess[k+1] = omega_decay*omega_guess[k] + (-np.expm1(-h*lam))*np.maximum(phi_guess[k],phi_guess[k+1])/lam
        self.states = [[m.Var(value=float(xguess[i,k])) for i in range(6)] for k in range(count)]
        self.inputs = [[m.Var(value=float(uguess[i,k])) for i in range(3)] for k in range(count-1)]
        self.phi = [[m.Var(value=float(phi_guess[k,i]), lb=0) for i in range(3)] for k in range(count)]
        self.omega = [[m.Var(value=float(omega_guess[k,i]), lb=0) for i in range(3)] for k in range(count)]
        self.alpha = [[m.Var(value=float(np.clip(alpha0[i],lower_alpha,upper_alpha)),
                             lb=lower_alpha, ub=upper_alpha) for i in range(3)] for _ in range(count-1)]
        self.reserve = []
        self.disturbance_bounds = []
        self.nominal_enclosures = []
        for i in range(6):
            self._eq(self.states[0][i], float(p.initial_state[i]))
        for i in range(3):
            self._eq(self.phi[0][i], float(phi0[i]))
            self._eq(self.omega[0][i], float(omega0[i]))
        for k in range(count-1):
            x, u, alpha = self.states[k], self.inputs[k], self.alpha[k]
            predicted = self._rk4(x, u, self.coefficients[k])
            for i in range(6):
                self._eq(self.states[k+1][i], predicted[i])
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
            w = [m.Intermediate(sum(float(da[k,i,j])*enclosure[j] for j in range(6))
                                + float(dd[k,i]+eta[i])) for i in range(3)]
            self.disturbance_bounds.append(w)
            self.nominal_enclosures.append(enclosure)
            phi_peak, omega_peak = [], []
            for i in range(3):
                decay = m.exp(-h*alpha[i])
                self._eq(self.phi[k+1][i], decay*self.phi[k][i] + (1-decay)*w[i]/alpha[i])
                peak = m.Var(value=float(max(phi_guess[k,i],phi_guess[k+1,i])), lb=0)
                self._ge(peak, self.phi[k][i])
                self._ge(peak, self.phi[k+1][i])
                phi_peak.append(peak)
                omega_decay = float(np.exp(-h*lam[i]))
                self._eq(self.omega[k+1][i], omega_decay*self.omega[k][i]
                         + (1-omega_decay)*peak/float(lam[i]))
                radius = m.Var(value=float(max(omega_guess[k,i],omega_guess[k+1,i])), lb=0)
                self._ge(radius, self.omega[k][i])
                self._ge(radius, self.omega[k+1][i])
                omega_peak.append(radius)
                if np.isfinite(rate):
                    previous = float(alpha0[i]) if k == 0 else self.alpha[k-1][i]
                    self._ge(rate*h, alpha[i]-previous)
                    self._ge(rate*h, previous-alpha[i])
                m.Minimize(h*bandwidth_weight*alpha[i]**2
                           + h*(tube_weight+1e-4)*(peak**2+radius**2))
            velocity_peak = [phi_peak[i]+float(lam[i])*omega_peak[i] for i in range(3)]
            error_peak = omega_peak+velocity_peak
            reserve = []
            for i in range(3):
                # |Av+Lambda| <= |Av|+Lambda. A_abs encloses every stage.
                feedback = sum(float(a_abs[k,i,j])*omega_peak[j] for j in range(3))
                feedback += sum(float(a_abs[k,i,j+3]+(lam[i] if i == j else 0))*velocity_peak[j]
                                for j in range(3))
                uncertainty = sum(float(da[k,i,j])*error_peak[j] for j in range(6))
                margin = m.Intermediate(feedback + uncertainty + alpha[i]*phi_peak[i])
                reserve.append(margin)
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
            errors = [self.states[k][i]-float(p.reference_states[i,k]) for i in range(6)]
            m.Minimize(h*sum(float(p.state_cost_matrix[i,j])*errors[i]*errors[j]
                             for i in range(6) for j in range(6) if p.state_cost_matrix[i,j] != 0))
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
        m.options.MAX_ITER = self.config.get("max_iter", 3000)
        m.options.MAX_MEMORY = self.config.get("max_memory", 512)
        m.options.OTOL = self.config.get("otol", 1e-7)
        m.options.RTOL = self.config.get("rtol", 1e-7)

    def _constraints(self):
        p, m, spec = self.problem, self.model, self.spec
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
            if target.get("operation") == "rendezvous":
                radius = float(target.get("rendezvous_radius",target.get("target_radius",0.)))
                self._ge(sum(v*v for v in pos),(radius+radius_tube)**2)
            elif target.get("operation") == "docking" and target.get("shape") == "cylinder":
                body = self._dot(rotations[k].T,pos)
                radius = float(target["target_radius"])
                half_length = float(target["target_half_length"])
                clearance = radius_tube+float(target.get("safety_reserve",0.))
                if target.get("robust_side_corridor",False):
                    normal = np.asarray(target["docking_normal_body"],dtype=float)
                    if normal.shape != (3,) or np.linalg.norm(normal[:2]) <= 0:
                        raise ValueError("Side corridor needs a nonzero docking_normal_body")
                    normal = np.array([normal[0],normal[1],0.])/np.linalg.norm(normal[:2])
                    self._ge(body[0]**2+body[1]**2,(radius+clearance)**2)
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
            for offset,key,tube in ((0,"terminal_position_tolerance",self.omega[-1]),
                                    (3,"terminal_velocity_tolerance",velocity)):
                tolerance = spec.get(key)
                if tolerance is None or not np.isfinite(float(tolerance)) or float(tolerance) <= 0:
                    raise ValueError("Hard terminal constraints require positive %s" % key)
                error = [self.states[-1][offset+i]-float(p.reference_states[offset+i,-1]) for i in range(3)]
                self._ge(float(tolerance),self._norm(error)+self._norm(tube))

    @staticmethod
    def _values(rows):
        return np.array([[float(value.value[0]) for value in row] for row in rows])

    def _solve_once(self):
        started = time.time()
        try:
            self.model.solve(disp=self.config.get("disp",False))
            states = self._values(self.states).T
            controls = self._values(self.inputs).T
            tube = {"phi":self._values(self.phi), "omega":self._values(self.omega),
                    "alpha":self._values(self.alpha), "input_reserve":self._values(self.reserve),
                    "disturbance_bound":self._values(self.disturbance_bounds),
                    "nominal_enclosure":self._values(self.nominal_enclosures)}
            residuals = [(float(value.value[0]),equality) for value,equality in self.residuals]
            violation = max([abs(value) if equality else max(0.,-value) for value,equality in residuals]+[0.])
            lower_alpha, upper_alpha = map(float,self.spec["alpha_bounds"])
            violation = max(violation,float(np.max(-tube["phi"])),float(np.max(-tube["omega"])),
                            float(np.max(lower_alpha-tube["alpha"])),
                            float(np.max(tube["alpha"]-upper_alpha)))
            finite = all(np.all(np.isfinite(value)) for value in [states,controls]+list(tube.values()))
            finite = finite and all(np.isfinite(value) for value,_ in residuals)
            tolerance = float(self.config.get("residual_tolerance",1e-5))
            success = finite and violation <= tolerance and self.model.options.APPSTATUS == 1
            return SolverResult(success=success,state_trajectory=states,control_trajectory=controls,
                                cost=float(self.model.options.objfcnval),solve_time=time.time()-started,
                                message="Coupled dynamic-tube optimization successful" if success else
                                "Coupled dynamic-tube residual verification failed",
                                solver_info={"status":self.model.options.APPSTATUS,
                                             "iterations":self.model.options.ITERATIONS,
                                             "max_constraint_violation":violation,
                                             "constraint_sampling":"nonlinear geometry and state constraints at prediction nodes",
                                             "dynamic_tube":tube})
        except Exception as error:
            return SolverResult(success=False,solve_time=time.time()-started,
                                message="Coupled GEKKO dynamic-tube solve failed: %s" % error)

    def solve(self):
        """Retry a failed warm start once using a fresh dynamics-feasible seed.

        Neighbor trajectories, terminal sets, uncertainty bounds, and solver
        tolerances stay identical. A rejected residual check is a failed
        attempt too; neither optimizer status nor retrying implies feasibility.
        """
        started = time.time()
        warm_keys = ("state_initial_guess", "control_initial_guess")
        has_warm_start = any(key in self.problem.extra_params for key in warm_keys)
        attempts = []

        def record(result, initialization):
            attempts.append(dict(initialization=initialization,success=bool(result.success),
                                 solve_time=float(result.solve_time),message=result.message,
                                 max_constraint_violation=result.solver_info.get("max_constraint_violation")))

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
        result.solve_time = time.time()-started
        result.solver_info["attempts"] = attempts
        result.solver_info["initialization_retry_count"] = len(attempts)-1
        return result

    def cleanup(self):
        self.model.cleanup()
