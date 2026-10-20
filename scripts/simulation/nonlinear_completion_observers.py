"""Causal navigation prediction and declared synthetic contact observations."""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

from orbexa.core.attitude import body_rate_derivative
from orbexa.core.nonlinear_relative import ExponentialAtmosphere, NonlinearModelBounds, acceleration, parameters
from orbexa.core.reference_frame import FloatArithmetic, exact
from orbexa.simulation.contact import ContactBodyState
from orbexa.simulation.mission_checkpoint import digest
from orbexa.simulation.nonlinear_completion_contracts import _box, _require


class PastNavigationAttitudeObserver:
    """Causal past accepted pose/rate prediction and bounded own-bus sensing."""
    def __init__(self, approach, *, model, clock):
        self.clock = clock
        # Prediction uses the public prior, never the independently chosen
        # point realization that is supplied to the physical continuation.
        public = approach.metadata['nonlinear_physical_bounds']
        atmosphere = ExponentialAtmosphere(**{key: public['atmosphere'][key] for key in
            ('rho0_kg_m3', 'reference_radius_m', 'scale_height_m', 'rotation_rad_s', 'wind_m_s', 'declaration')})
        prior = NonlinearModelBounds(atmosphere=atmosphere, **{key: public[key] for key in
            ('mu_m3_s2', 'j2', 'body_radius_m', 'gravity_reference_radius_m', 'j2_axis', 'sigma_target',
             'sigma_chaser', 'radius_domain_m', 'relative_position_limit_m', 'relative_velocity_limit_m_per_rad', 'declaration')})
        _require(digest(prior.as_dict()) == digest(public), 'public navigation model source changed')
        self.public_model_id, self.model = prior.fingerprint, prior.midpoint_model()
        self.inertia = np.asarray(approach.target['inertia_kg_m2'], dtype=float)
        self.gravity_gradient = bool(approach.target.get('gravity_gradient', False))
        self.par = parameters(self.model, FloatArithmetic())
        self.records = []
        for record in approach.metadata['navigation_history']:
            if record.get('status') != 'VALID':
                continue
            _require(record['clock_fingerprint'] == clock.fingerprint and
                     record['model_fingerprint'] == self.public_model_id and
                     record['frame'] == 'reference-orbit-frame-v1', 'navigation model, clock or frame source changed')
            box, ephemeris = _box(record['physical_time_box'], 1), _box(record['target_ephemeris_box'], 6)
            rate = _box(record['target_angular_velocity_box'], 3)
            q = exact(record['exact_q'])
            rotation = np.asarray(clock.point(q).rotation)@np.asarray(record['target_orientation_matrix'], dtype=float)
            state = np.r_[[float((lo+hi)/2) for lo, hi in zip(ephemeris.lower, ephemeris.upper)],
                Rotation.from_matrix(rotation).as_quat(),
                [float((lo+hi)/2) for lo, hi in zip(rate.lower, rate.upper)]]
            self.records.append((float(box.upper[0]), float((box.lower[0]+box.upper[0])/2), state,
                                 int(record.get('accepted_packet_count', 0)), str(q)))
        _require(self.records and all(a[0] <= b[0] for a, b in zip(self.records, self.records[1:])),
                 'ordered accepted past navigation history is required')
        self.calls, self.max_age_s, self.used_indices = 0, 0., set()
        self.prior_calls = 0
        self._cached = None

    def _rhs(self, time_s, state):
        rotation = Rotation.from_quat(state[6:10]).as_matrix()
        torque = np.zeros(3)
        if self.gravity_gradient:
            radius = np.linalg.norm(state[:3])
            radial = rotation.T@state[:3]/radius
            torque = 3*self.par['mu']/radius**3*np.cross(radial, self.inertia@radial)
        q, w = state[6:10], state[10:13]
        return np.r_[state[3:6], acceleration(tuple(state[:6]), self.model, self.par, FloatArithmetic(), self.par['sigma_target']),
            .5*np.r_[q[3]*w+np.cross(q[:3], w), -q[:3]@w], body_rate_derivative(w, self.inertia, torque)]

    def __call__(self, time_s, snapshot):
        eligible = [index for index, record in enumerate(self.records) if record[0] <= time_s+1e-10]
        _require(eligible, 'no past accepted target observation is available at this replay time')
        index = eligible[-1]
        available, epoch, initial, packets, q = self.records[index]
        if self._cached is not None and self._cached[0] == index and self._cached[1] <= time_s:
            _, epoch, initial = self._cached
        state = initial.copy()
        if time_s > epoch+1e-12:
            solved = solve_ivp(self._rhs, (epoch, time_s), state, method='DOP853', rtol=1e-10, atol=1e-12,
                               max_step=min(.5, time_s-epoch))
            _require(solved.success and np.all(np.isfinite(solved.y)), 'past observed target prediction failed')
            state = solved.y[:, -1]
        self._cached = (index, time_s, state.copy())
        result = dict(time_s=float(time_s), target_rotation_body_to_eci=Rotation.from_quat(state[6:10]).as_matrix(),
            target_angular_velocity_body_rad_s=state[10:13],
            target_angular_acceleration_body_rad_s2=self._rhs(time_s, state)[10:13], chasers={})
        for number, (key, sensed) in enumerate(snapshot['chasers'].items()):
            phase = .7*number+.03*time_s
            direction = np.array((np.cos(phase), np.sin(phase), 0.))
            rotation = Rotation.from_quat(sensed['quaternion_body_to_eci'])*Rotation.from_rotvec(.0003*direction)
            result['chasers'][key] = dict(quaternion_body_to_eci=rotation.as_quat(),
                angular_velocity_body_rad_s=np.asarray(sensed['angular_velocity_body_rad_s'])+1e-5*direction)
        self.calls += 1
        self.prior_calls += int(packets == 0)
        self.max_age_s = max(self.max_age_s, time_s-self.records[index][1])
        self.used_indices.add(index)
        return result

    def as_dict(self):
        return dict(method='past-accepted-navigation-rigid-body-extrapolation-v1',
            rule='latest VALID navigation record whose physical-time upper endpoint is no later than the replay query',
            future_truth_or_interpolation_used=False, own_bus_orientation_error_norm_rad=.0003,
            own_bus_rate_error_norm_rad_s=1e-5, calls=self.calls, maximum_prediction_age_s=self.max_age_s,
            initial_declared_prior_calls=self.prior_calls,
            prediction_model='midpoint of public declared nonlinear bounds, independent of selected truth',
            public_model_fingerprint=self.public_model_id, prediction_model_fingerprint=self.model.fingerprint,
            used_navigation_records=len(self.used_indices), source_navigation_sha256=digest(self.records),
            scope='Observed-center rigid-body/GG prediction; decoupled replay, no coupled translational-attitude flight claim')



class BiasedContactObserver:
    """Truth-side synthetic sensor realization; commands see only its outputs."""
    def __init__(self):
        self.calls = 0

    def __call__(self, time_s, snapshot):
        common_p, common_v = np.array((.35, -.4, .2)), np.array((.001, -.001, 0.))
        target = snapshot['target']
        target_rotation = Rotation.from_quat(target.quaternion_body_to_eci)*Rotation.from_rotvec((.001, 0., 0.))
        measured = dict(target=ContactBodyState(time_s, target.position_eci_m+common_p,
            target.velocity_eci_m_s+common_v, target_rotation.as_quat(),
            target.angular_velocity_body_rad_s+np.array((0., 1e-4, 0.)), target.fuel_kg), chasers={},
            contact_gaps_m={key: value+.001 for key, value in snapshot['contact_gaps_m'].items()},
            contact_wrenches_eci={}, positioner_encoders={})
        for number, (key, state) in enumerate(snapshot['chasers'].items()):
            sign = (-1.)**number
            direction = np.array((np.cos(.7*number), np.sin(.7*number), 0.))
            camera_p = np.array((sign*.0015, -.002, .0015))
            rotation = Rotation.from_quat(state.quaternion_body_to_eci)*Rotation.from_rotvec(.0003*direction)
            measured['chasers'][key] = ContactBodyState(time_s, state.position_eci_m+common_p+camera_p,
                state.velocity_eci_m_s+common_v+5e-5*direction, rotation.as_quat(), state.angular_velocity_body_rad_s+1e-5*direction,
                state.fuel_kg)
            encoder = snapshot['positioner_encoders'][key]
            measured['positioner_encoders'][key] = dict(
                extension_body_m=(np.asarray(encoder['extension_body_m'])+np.array((1e-6, -1e-6, 1e-6))).tolist(),
                extension_velocity_body_m_s=(np.asarray(encoder['extension_velocity_body_m_s'])+1e-7*direction).tolist())
            wrench = snapshot['contact_wrenches_eci'][key]
            measured['contact_wrenches_eci'][key] = dict(force_n=np.asarray(wrench['force_n'])+np.array((.003, -.002, .003)),
                couple_nm=np.asarray(wrench['couple_nm'])+np.array((.0001, -.0001, .0001)))
        self.calls += 1
        return measured

    def as_dict(self):
        return dict(method='declared-biased-synthetic-contact-observations-v1', calls=self.calls,
            common_gnss_position_bias_eci_m=(.35, -.4, .2), common_gnss_velocity_bias_eci_m_s=(.001, -.001, 0.),
            individual_camera_position_bias_norm_upper_m=.003, target_pose_error_rad=.001,
            individual_relative_velocity_bias_norm_upper_m_s=.0001,
            target_rate_error_norm_rad_s=1e-4, own_bus_pose_error_norm_rad=.0003,
            own_bus_rate_error_norm_rad_s=1e-5, local_range_bias_m=.001,
            wrench_force_bias_eci_n=(.003, -.002, .003), wrench_couple_bias_eci_nm=(.0001, -.0001, .0001),
            encoder_position_bias_body_m=(1e-6, -1e-6, 1e-6), encoder_velocity_bias_norm_m_s=1e-7,
            deterministic_support=self.observation_support().as_dict(),
            scope='Synthetic bounded processed products; no raw perception, calibrated material sensing or hardware claim')

    def observation_support(self):
        from orbexa.simulation.contact_observation import ContactObservationBounds
        return ContactObservationBounds()
