"""Separate executed nonlinear target truth and observation-driven preview.

Geometry is composed from an existing declared target. Its analytical motion
is never delegated: Cartesian ephemeris, physical time and gravity-gradient
torque are propagated with the declared nonlinear model/reference clock.
"""
from __future__ import annotations

import copy
import numpy as np
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

from .attitude import inertia_matrix, body_rate_derivative, quaternion_to_euler
from .rigid_body_motion import RigidBodyMotion
from .nonlinear_relative import NonlinearRelativeDynamics
from .reference_frame import exact


@dataclass(frozen=True)
class NonlinearTargetProposal:
    start_q: object
    end_q: object
    initial_state: tuple
    solution: object
    model_fingerprint: str
    clock_fingerprint: str

    def state_at(self,q):
        if not float(self.start_q) <= float(q) <= float(self.end_q):
            raise ValueError('Numerical target proposal queried outside its hold')
        return np.asarray(self.solution.sol(float(q)),dtype=float)


def _skew(v):
    x,y,z = v
    return np.array(((0,-z,y),(z,0,-x),(-y,x,0)))


class _NonlinearTargetQueries:
    reference_frame_id = 'reference-orbit-frame-v1'

    def _initialize(self, template, clock, model, initial_q):
        self._template, self.reference_clock, self.model = template, clock, model
        self._dynamics = NonlinearRelativeDynamics(model, clock)
        self.inertia = inertia_matrix(template.inertia).copy()
        self.inertia.setflags(write=False)
        self.gravity_gradient = bool(template.gravity_gradient)
        self.initial_quaternion = np.asarray(template.initial_quaternion, dtype=float).copy()
        self.initial_body_angular_velocity = np.asarray(template.initial_body_angular_velocity, dtype=float).copy()
        self.eccentricity, self.alpha = float(clock.eccentricity), float(clock.alpha)
        self.mu, self.semi_major_axis = float(clock.mu_m3_s2), float(clock.semi_major_axis_m)
        self.anomaly_domain = tuple(map(float, clock.q_domain))
        self.initial_q, self.current_q = exact(initial_q), exact(initial_q)
        self._pieces = []
        # The template is an immutable geometry declaration. Rebuilding its
        # exact occupied vertices/hash at every numerical clearance sample is
        # unnecessary; callers still receive independent mutable records.
        self._collision_declarations = {}

    def __getattr__(self, name):
        # Only geometry/static declarations are delegated. Motion methods all
        # resolve on this class or explicitly call the local dynamic state.
        allowed = {'geometry','mass_kg','radius','half_length','height','tolerance',
            'docking_standoff','bounding_sphere_radius','rendezvous_radius','shape',
            'approach_azimuth','approach_position_body','approach_axial_coordinate',
            'side_docking_geometry','docking_points_body'}
        if name not in allowed:
            raise AttributeError(name)
        return getattr(self._template, name)

    @property
    def initial_orientation(self):
        return quaternion_to_euler(self.initial_quaternion)

    @property
    def attitude_model(self):
        return 'nonlinear_ephemeris_central_gravity_gradient' if self.gravity_gradient else 'nonlinear_ephemeris_torque_free'

    def _query(self, q):
        value = float(q)
        if not np.isfinite(value):
            raise ValueError('Target query must be finite')
        self.reference_clock.check_domain((q,q))
        return value

    def anomaly_clock(self, q):
        point = self.reference_clock.point(q)
        return point.rate_s_inv, point.rate_s_inv*point.kappa

    def _full_state_at(self, q):
        value = self._query(q)
        for lo, hi, solution in reversed(self._pieces):
            if lo <= value <= hi:
                state = np.asarray(solution(value), dtype=float)
                if not np.all(np.isfinite(state)):
                    raise RuntimeError('Nonlinear target state is nonfinite')
                return state
        if value == float(self.current_q) and hasattr(self,'_current_state'):
            return self._current_state.copy()
        raise ValueError('Target query exceeds executed truth or observed preview coverage')

    def _state_at(self, q):
        state = self._full_state_at(q)
        quaternion = state[6:10]/np.linalg.norm(state[6:10])
        return float(state[13]), quaternion, state[10:13].copy()

    def _body_torque_state(self, state):
        if not self.gravity_gradient:
            return np.zeros(3)
        radius = np.linalg.norm(state[:3])
        radial_body = Rotation.from_quat(state[6:10]).inv().apply(state[:3]/radius)
        mu = float(self.model.mu_m3_s2[0])
        return 3*mu/radius**3*np.cross(radial_body,self.inertia@radial_body)

    def _rhs(self, q, state):
        point = self.reference_clock.point(q)
        quaternion, omega = state[6:10], state[10:13]
        qdot = .5*np.r_[quaternion[3]*omega+np.cross(quaternion[:3],omega),-np.dot(quaternion[:3],omega)]
        wdot = body_rate_derivative(omega,self.inertia,self._body_torque_state(state))
        return np.r_[self._dynamics.target_rhs(q,state[:6]),qdot/point.rate_s_inv,
                     wdot/point.rate_s_inv,1/point.rate_s_inv]

    def _integrate(self, start, end, state):
        self.reference_clock.check_domain((start,end))
        if end <= start:
            raise ValueError('Nonlinear target propagation must advance a finite interval')
        result = solve_ivp(self._rhs,(float(start),float(end)),state,method='DOP853',dense_output=True,
            max_step=min(.002,float(end-start)),rtol=1e-11,atol=1e-12)
        if not result.success or not np.all(np.isfinite(result.y)):
            raise RuntimeError('Nonlinear target propagation failed: '+str(result.message))
        return result

    def physical_time_at(self,q):
        return self._state_at(q)[0]

    def inertial_quaternion_at(self,q):
        return self._state_at(q)[1]

    def orbital_position_inertial_at(self,q):
        return self._full_state_at(q)[:3].copy()

    def inertial_state_at(self,q):
        return self._full_state_at(q)[:6].copy()

    def body_angular_velocity_at(self,q):
        return self._state_at(q)[2]

    def body_torque_at(self,q):
        return self._body_torque_state(self._full_state_at(q))

    def _body_torque(self,q,inertial_quaternion):
        state = self._full_state_at(q).copy()
        state[6:10] = inertial_quaternion
        return self._body_torque_state(state)

    def rotation_matrix_at(self,q):
        inertial = Rotation.from_quat(self.inertial_quaternion_at(q)).as_matrix()
        return np.asarray(self.reference_clock.point(q).rotation).T@inertial

    def quaternion_at(self,q):
        return Rotation.from_matrix(self.rotation_matrix_at(q)).as_quat()

    def rotation_matrices(self,q):
        rotation,omega = self.rotation_matrix_at(q),self.body_angular_velocity_at(q)
        rate,rate_prime = self.anomaly_clock(q)
        omega_dot = body_rate_derivative(omega,self.inertia,self.body_torque_at(q))
        k,kp = _skew(omega/rate),_skew((omega_dot-omega*rate_prime)/rate**2)
        zz = _skew((0,0,1))
        return rotation,-zz@rotation+rotation@k,zz@zz@rotation-2*zz@rotation@k+rotation@(k@k+kp)

    orientation_at = RigidBodyMotion.orientation_at
    angular_velocity_at = RigidBodyMotion.angular_velocity_at
    docking_state = RigidBodyMotion.docking_state
    docking_acceleration = RigidBodyMotion.docking_acceleration
    inertial_angular_momentum_at = RigidBodyMotion.inertial_angular_momentum_at

    def recorded_kinematics(self,anomalies):
        result = RigidBodyMotion.recorded_kinematics(self,anomalies)
        result['target_inertial_states'] = np.array([self.inertial_state_at(q) for q in anomalies])
        result['target_inertial_position_m'] = result['target_inertial_states'][:,:3]
        result['target_inertial_velocity_m_s'] = result['target_inertial_states'][:,3:]
        result['reference_anomaly_rate_s_inv'] = result['anomaly_rate'].copy()
        return result

    def collision_params(self,operation,anom):
        if operation not in self._collision_declarations:
            self._collision_declarations[operation] = copy.deepcopy(
                self._template.collision_params(operation,float(self.initial_q)))
        result = copy.deepcopy(self._collision_declarations[operation])
        result.update(rotation_matrix=self.rotation_matrix_at(anom).tolist(),
            attitude_model=self.attitude_model,reference_frame_id=self.reference_frame_id,
            reference_clock_fingerprint=self.reference_clock.fingerprint)
        return result

    def assign_docking_targets(self,initial_states):
        if getattr(self._template,'geometry',None) is not None:
            from orbexa.planning.composite_assignment import assign_composite_docking_targets
            return assign_composite_docking_targets(self,initial_states)
        return self._template.assign_docking_targets(initial_states)

    def as_dict(self):
        result = dict(self._template.as_dict())
        result.update(attitude_model=self.attitude_model,motion_model=self.attitude_model,
            attitude_reference_frame='body_to_commanded_reference_orbit_frame',
            reference_frame_id=self.reference_frame_id,clock_model='commanded_reference_clock',
            reference_clock=self.reference_clock.as_dict(),clock_fingerprint=self.reference_clock.fingerprint,
            physical_plant=self.model.as_dict(),nonlinear_target_role=self.role,
            integration=dict(method='DOP853',rtol=1e-11,atol=1e-12,max_anomaly_step=.002),
            inertial_reference='Declared fixed initial-reference axes; no absolute Earth-orientation epoch claim',
            lvlh_frame_rotation='Commanded reference C(q), not actual perturbed LVLH',
            clock_parameters=dict(eccentricity=self.eccentricity,alpha=self.alpha,mu=self.mu,
                semi_major_axis=self.semi_major_axis,theta0=float(self.reference_clock.phase)),
            initial_q=str(self.initial_q),current_q=str(self.current_q),
            assumptions=['Declared rigid inertia and component geometry',
                'Central spherical gravity-gradient torque only; J2/atmosphere act on translation',
                'Numerical target motion does not certify pose errors',
                'Preview consumes observations and declared nominal physics, never future truth'])
        return result


class NonlinearTargetTruth(_NonlinearTargetQueries):
    role = 'executed_numerical_truth'

    def __init__(self,template,clock,point_model,initial_target_state,*,initial_q=0,initial_time_s=0):
        self._initialize(template,clock,point_model,initial_q)
        state = np.asarray(initial_target_state,dtype=float)
        if state.shape != (6,) or not np.all(np.isfinite(state)):
            raise ValueError('Truth target ephemeris must have six finite SI components')
        inertial_rotation = np.asarray(clock.point(initial_q).rotation)@Rotation.from_quat(self.initial_quaternion).as_matrix()
        self._current_state = np.r_[state,Rotation.from_matrix(inertial_rotation).as_quat(),
            self.initial_body_angular_velocity,float(initial_time_s)]

    def advance(self,end_q):
        return self.commit_interval(self.propose_interval(end_q))

    def propose_interval(self,end_q):
        end = exact(end_q)
        result = self._integrate(self.current_q,end,self._current_state)
        return NonlinearTargetProposal(self.current_q,end,tuple(self._current_state),result,
            self.model.fingerprint,self.reference_clock.fingerprint)

    def commit_interval(self,proposal):
        if (not isinstance(proposal,NonlinearTargetProposal) or proposal.start_q!=self.current_q
                or proposal.initial_state!=tuple(self._current_state)
                or proposal.model_fingerprint!=self.model.fingerprint
                or proposal.clock_fingerprint!=self.reference_clock.fingerprint):
            raise ValueError('Numerical target proposal is stale or changed model/clock')
        self._pieces.append((float(self.current_q),float(proposal.end_q),proposal.solution.sol))
        self._current_state,self.current_q = proposal.solution.y[:,-1].copy(),proposal.end_q
        return self._current_state.copy()


class NonlinearTargetPreview(_NonlinearTargetQueries):
    role = 'observation_driven_numerical_preview'

    def __init__(self,template,clock,model_bounds,initial_target_box,*,initial_q=0,preview_width_q=1,initial_time_s=0):
        self.physical_model_bounds = model_bounds
        self._initialize(template,clock,model_bounds.midpoint_model(),initial_q)
        self.preview_width_q = exact(preview_width_q)
        self.observation_fingerprints = ()
        self.pose_operator_error = 2
        initial_state = np.array([float((a+b)/2) for a,b in zip(initial_target_box.lower,initial_target_box.upper)])
        inertial_rotation = np.asarray(clock.point(initial_q).rotation)@Rotation.from_quat(self.initial_quaternion).as_matrix()
        self._current_state = np.r_[initial_state,Rotation.from_matrix(inertial_rotation).as_quat(),
            self.initial_body_angular_velocity,float(initial_time_s)]
        self._predict(self.current_q+self.preview_width_q)

    def _predict(self,end):
        result = self._integrate(self.current_q,end,self._current_state)
        self._pieces.append((float(self.current_q),float(end),result.sol))

    def update_observation(self,navestimate,current_q=None,target_ephemeris_box=None,preview_end_q=None):
        if (navestimate.status != 'VALID' or navestimate.clock_fingerprint != self.reference_clock.fingerprint
                or navestimate.model_fingerprint != self.physical_model_bounds.fingerprint):
            raise ValueError('A valid same-clock navigation observation is required')
        q = exact(navestimate.exact_event_q if current_q is None else current_q)
        if q != exact(navestimate.exact_event_q) or q < self.current_q:
            raise ValueError('Observed preview event cannot be relabeled or run backward')
        ephemeris = navestimate.target_ephemeris_box if target_ephemeris_box is None else target_ephemeris_box
        if ephemeris.dimension != 6:
            raise ValueError('Observed target ephemeris has six SI states')
        rotation = np.asarray(navestimate.target_orientation_matrix,dtype=float)
        inertial = np.asarray(self.reference_clock.point(q).rotation)@rotation
        quaternion = Rotation.from_matrix(inertial).as_quat()
        omega = np.array([float((a+b)/2) for a,b in zip(navestimate.target_angular_velocity_box.lower,
                                                       navestimate.target_angular_velocity_box.upper)])
        state = np.array([float((a+b)/2) for a,b in zip(ephemeris.lower,ephemeris.upper)])
        physical_time = float(sum((navestimate.physical_time_box.lower[0],navestimate.physical_time_box.upper[0]))/2)
        # Truncate obsolete future previews; past intervals remain reproducible
        # estimates, not a reconstructed truth trajectory.
        self._pieces = [(lo,min(hi,float(q)),solution) for lo,hi,solution in self._pieces if lo < float(q)]
        self._current_state,self.current_q = np.r_[state,quaternion,omega,physical_time],q
        self.pose_operator_error = navestimate.target_orientation_operator_error
        self.observation_fingerprints = tuple(navestimate.packet_ids)
        self._predict(q+self.preview_width_q if preview_end_q is None else exact(preview_end_q))
        return self
