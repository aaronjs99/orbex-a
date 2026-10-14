# Uncontrolled rigid-body target

`RigidBodyCylinderTarget` supplies an optional, unactuated attitude model through
the existing `MissionScenario(target=...)` interface. Its physical body spin
evolves according to Euler's rigid-body equation. The default is torque-free;
`gravity_gradient=True` adds central-field gravity-gradient torque using the
instantaneous orbital radius and attitude. Unactuated motion does not imply
zero environmental torque. The existing
`TumblingCylinderTarget` remains a prescribed Euler-angle trajectory.

The collision shape remains a finite cylinder. The inertia is an independent
body-frame parameter; the envelope does not imply a particular mass distribution,
spacecraft structure or flight asset. A spacecraft drawn inside this envelope is
a representative concept unless a specific asset and its dimensions are supplied.

## Frames and units

The independent variable is unwrapped orbital anomaly, named `q`, in radians.
Physical time is `t`, in seconds measured from the first anomaly in the declared
domain. A prime denotes differentiation with respect to `q`; a dot denotes
differentiation with respect to `t`. The anomaly rate is `nu = dq/dt`, in rad/s.
The orbital radius is `r_o`, and its anomaly derivative is `r_o_prime`. The orbit's
drag parameter is `alpha`. The reference angular-momentum integration constant
is `h_o`, in m²/s, computed from the supplied gravitational parameter,
semi-major axis and eccentricity. The clock uses the same analytical inverse-radius
quadratic-drag orbit as the translational model:

$$
\frac{dt}{dq}=\frac{1}{\nu},\qquad
\nu(q)=\frac{h_o\exp(-\alpha q)}{r_o(q)^2},\qquad
\nu'(q)=\nu(q)\left(-\alpha-2\frac{r_o'(q)}{r_o(q)}\right).
$$

Orbital phase is zero, consistently with current mission truth.
See [the orbital model](theory.md) for the radius formula and its approximation.
The target attitude preview uses the supplied clock as known. The mission's
orbital parameter belief affects translational prediction and tubes; it does
not enclose error in that attitude preview or its time conversion.

The initial anomaly is `q_a`. An inertial basis is fixed coincident with LVLH
at `q_a`. LVLH has axes opposite the transverse direction, radially outward and
orbit normal. In the planar model this frame rotates positively by `q-q_a`
about the orbit normal. The inertial attitude matrix is `R_I`, mapping body
coordinates into the fixed inertial basis. The mission attitude is `R`, mapping
body coordinates into LVLH. The orbit-normal unit vector is `e_3=[0,0,1]^T`,
and `Z=[e_3]_cross` is its cross-product matrix. Thus:

$$
R(q)=\exp[-(q-q_a)Z]R_I(q).
$$

This frame transformation is necessary for physical torque-free motion. A target
with zero inertial spin counterrotates in LVLH as the spacecraft moves around
its orbit.

The scalar-last unit quaternion is `eta=[v_eta,s_eta]`, where `v_eta` contains
three vector components and `s_eta` is the scalar component. Internally it maps
body coordinates into the fixed inertial basis. `initial_quaternion` specifies
body-to-LVLH orientation at `q_a`, when those two bases coincide. Public
`quaternion_at` reports body-to-LVLH orientation; `inertial_quaternion_at` reports
body-to-inertial orientation. Both preserve the propagated sign rather than
folding every quaternion into a positive-scalar hemisphere.

The body-frame inertia matrix is `J`, in kg m². It must be finite, symmetric,
positive definite and physically realizable. Its sorted principal moments are
`lambda_1`, `lambda_2` and `lambda_3`. A physical mass distribution obeys the
following triangle condition because every principal moment is a sum of two
nonnegative coordinate second moments:

$$
0<\lambda_1\le\lambda_2\le\lambda_3,\qquad
\lambda_3\le\lambda_1+\lambda_2.
$$

The implementation allows relative eigensolver roundoff of `64*eps`, where
`eps` is floating-point machine epsilon, at that
boundary. SPD alone does not satisfy this requirement; for example, principal
moments `[1,1,3]` are rejected, whereas the planar boundary `[1,1,2]` is allowed.

The vector `omega` is angular velocity relative to inertial space, expressed in
body coordinates, in rad/s. The environmental body-frame torque is `tau_B`,
in N m. There is no target attitude-control actuation. The integrated equations
are:

$$
\dot\omega=J^{-1}(\tau_B-\omega\times J\omega),\qquad
\dot\eta=\frac12
\begin{bmatrix}s_\eta\omega+v_\eta\times\omega\\-v_\eta^T\omega\end{bmatrix}.
$$

The standard torque-free equation and conservation of inertial angular momentum
are recovered when `tau_B=0`. They follow from rigid-body angular-momentum balance. See the
[MIT rigid-body dynamics notes](https://ocw.mit.edu/courses/8-09-classical-mechanics-iii-fall-2014/d9bac33f6c60b304dc0398e99b327102_MIT8_09F14_full.pdf).
Quaternion ordering and the intrinsic uppercase `XYZ` Euler chart follow
[SciPy's quaternion](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_quat.html)
and [Euler conventions](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_euler.html).

For gravity-gradient mode, the central-body gravitational parameter is `mu`,
in m³/s². The radial unit vector in the fixed inertial basis is `rhat_I`, and
`rhat_B` is that vector expressed in body coordinates. The unit vector
`e_2=[0,1,0]^T` is the positive radial axis at the initial anomaly. The orbital
position vector is `r_I`, in meters. The current spherical central-field torque
is:

$$
\hat r_I=\exp[(q-q_a)Z]e_2,\qquad
r_I=r_o(q)\hat r_I,\qquad
\hat r_B=R_I^T\hat r_I=R^T e_2,
$$

$$
\tau_B=\frac{3\mu}{r_o(q)^3}
  \left[\hat r_B\times(J\hat r_B)\right].
$$

The radial direction and the instantaneous radius come from the same analytical
orbit and clock as the attitude propagation. No circular-orbit mean-motion
substitution is used for the eccentric case. The fixed orbital-plane inertial
basis is an Earth-centered inertial choice up to a constant rotation; an
absolute node/inclination orientation does not change spherical-field torque.
This is the leading gravity-gradient approximation for a small rigid body in
a spherical central field. The formula and required radius, inertia and attitude
inputs follow [NASA SP-8024, sections 2.5 and 4.1.6](https://ntrs.nasa.gov/api/citations/19700014113/downloads/19700014113.pdf).
ESA's [GODOT gravity-gradient torque interface](https://godot.io.esa.int/docs/api_reference/generated/generated/generated/godot.core.astro.gravityGradientTorque.html)
likewise uses the central-body-relative position and inertia in the same frame.

The inertial spin angular momentum about the target center of mass is `H_I`,
in kg m²/s. The rotational kinetic energy is `E_rot`, in joules; inertial torque
is `tau_I`, in N m, and rotational power is `P_rot`, in watts. The correct
balances for either mode are:

$$
H_I=R_IJ\omega,\quad E_{\rm rot}=\tfrac12\omega^TJ\omega,\quad
\tau_I=R_I\tau_B,\quad P_{\rm rot}=\omega^T\tau_B,
$$

$$
\dot H_I=\tau_I,\qquad \dot E_{\rm rot}=P_{\rm rot}.
$$

Gravity-gradient mode generally changes both inertial spin momentum and
rotational kinetic energy. Its checks compare those changes to torque impulse
and torque work; they do not assert torque-free conservation.

The implementation integrates `[t,eta,omega]` in anomaly, dividing the physical
attitude derivatives by `nu`. Each instance constructs one DOP853 dense
trajectory with default relative tolerance `1e-11`, absolute tolerance `1e-13`
and maximum anomaly step `0.05` rad. Its initial quaternion is normalized;
returned quaternion samples are normalized independently. Queries do not
restart, extend or mutate the trajectory. Arrays supplied at construction
are copied, and stored initial arrays are read-only.

## Docking-point derivatives

The normalized inertial body rate is `b=omega/nu`, in radians per anomaly
radian. Its cross-product matrix is `K=[b]_cross`; `K_prime` is its anomaly
derivative. The following expressions account for rigid-body acceleration,
orbital clock variation and the rotating LVLH frame:

$$
b'=\frac{\dot\omega-\omega\nu'}{\nu^2},\qquad
K'=[b']_\times,
$$

$$
R'=-ZR+RK,\qquad
R''=Z^2R-2ZRK+R(K^2+K').
$$

The desired body-fixed standoff point is `g_B`, in meters. Its LVLH position is
`p_g`; `v_g` is anomaly velocity in m/rad, and `a_g` is anomaly acceleration in
m/rad². The public state and acceleration are:

$$
p_g=Rg_B,\qquad v_g=R'g_B,\qquad a_g=R''g_B.
$$

No Euler finite differences or constant-Euler-rate acceleration formula enters
these methods. The existing controller and reference-preview services consume
them directly.

For compatibility, `orientation_at(q)` returns an Euler chart with
`R = Rx(roll) Ry(pitch) Rz(yaw)`. `angular_velocity_at(q)` returns derivatives
of those Euler angles **per anomaly radian**, as required by the existing target
protocol. It never returns physical body spin. The relative body rate used
for that chart is `b-R^T e_3`, accounting for LVLH rotation. The chart raises a
`ValueError` when `abs(cos(pitch)) <= 1e-8`. Quaternion and docking kinematics
remain valid there. Legacy consumers requesting Euler rates must handle that
explicit exception; proper-matrix queries and the collision payload remain available.

## API

```python
import numpy as np
from orbexa.core import RigidBodyCylinderTarget
from orbexa.core.config import SimulationConfig
from orbexa.estimation.belief import ParameterBelief
from orbexa.simulation.mission_scenario import MissionScenario
from orbexa.simulation.coupled_mission import CoupledTubeMission

config = SimulationConfig.load()
target = RigidBodyCylinderTarget.from_config(
    config,
    initial_body_angular_velocity=np.array([0.0002, 0.0003, 0.00015]),
    gravity_gradient=True,  # Optional; the default remains torque-free.
)
# Explicit illustrative prior, independent of reading the plant's truth object.
belief = ParameterBelief(
    feasible_sets={
        "eccentricity": (0.14, 0.21),
        "alpha": (1e-7, 3e-7),
        "beta": (2.25e-7, 6.75e-7),
    },
    estimates={"eccentricity": 0.16, "alpha": 1.5e-7, "beta": 3.5e-7},
)
scenario = MissionScenario(config=config, target=target, initial_belief=belief)
result = CoupledTubeMission(scenario).run(mission="single", steps=200)
samples = target.recorded_kinematics(result.anom_history)
```

The spin above is an illustrative slow-debris input, not an observed or
representative population measurement. `from_config` requires this explicit
rad/s input and deliberately ignores `target.initial_angular_velocity`, whose
legacy units are Euler derivatives per anomaly. It takes geometry, inertia and
the analytical orbital clock from the configuration. Optional `initial_quaternion`
and `inertia` arguments override those two initial values.
If a scenario injects orbital truth different from that configuration, construct
the target with the matching clock parameters explicitly. `MissionScenario`
does not reconcile two independently supplied clocks.

The direct constructor requires `radius`, `half_length`, `initial_quaternion`,
`initial_body_angular_velocity`, `inertia`, `eccentricity`, `alpha`, `mu` and
`semi_major_axis`. It also accepts `anomaly_domain`, `tolerance`,
`docking_standoff`, `integration_rtol`, `integration_atol` and
`integration_max_step` and boolean `gravity_gradient`. `from_config` accepts
that flag and the three integration options.

`anomaly_domain=(q_a,q_b)` has finite, strictly increasing bounds. All queries
must be finite scalars within this closed domain; no extrapolation is allowed.
The direct constructor defaults to `[0,2*pi]`. `from_config` defaults to zero
through `max(total_anom, num_update_steps*anom_step)` plus the longest configured
prediction horizon. For a longer mission, explicitly supply a domain that
includes every update and every preview node.

The additional inspection methods are:

| Method | Return |
| --- | --- |
| `physical_time_at(q)` | Elapsed physical seconds from `q_a` |
| `anomaly_clock(q)` | `(nu, nu_prime)` |
| `quaternion_at(q)` | Scalar-last body-to-LVLH unit quaternion |
| `inertial_quaternion_at(q)` | Scalar-last body-to-inertial unit quaternion |
| `body_angular_velocity_at(q)` | Inertial body spin in body components, rad/s |
| `body_torque_at(q)` | Applied environmental torque in body components, N m |
| `orbital_position_inertial_at(q)` | Central-body-to-target position in the fixed inertial basis, m |
| `inertial_angular_momentum_at(q)` | Spin angular momentum about the target COM in inertial components, kg m²/s |
| `rotation_matrix_at(q)` | Proper body-to-LVLH rotation without an Euler chart |
| `rotation_matrices(q)` | `(R, R_prime, R_double_prime)` in LVLH |
| `recorded_kinematics(q_values)` | Independent arrays sampled from this same target |

Recorded arrays preserve caller order and include `anomaly`, `physical_time`,
`quaternions`, `inertial_quaternions`, `inertial_rotations`,
`body_angular_velocity`, `rotations`, `rotation_primes`,
`rotation_double_primes`, `anomaly_rate` and `anomaly_rate_prime`.
They also include `body_angular_acceleration`, `body_torque`, `inertial_torque`,
`inertial_angular_momentum`, `rotational_kinetic_energy`, `rotational_power`,
`orbital_position_inertial`, `orbital_radius` and `radial_unit_body`. All have one
entry per requested anomaly. The inertia and its principal moments are retained
in target metadata rather than repeated at every sample.
Use the declared quaternion frame when rendering or checking conservation.
In particular, angular momentum in the rotating LVLH basis is not constant.

`as_dict()` distinguishes `torque_free_rigid_body` from
`gravity_gradient_rigid_body` and declares the torque option, physically
realizable inertia and principal moments, initial physical spin,
quaternion ordering, frames, clock, domain, integration settings and assumptions.
It retains the existing cylinder-envelope collision and assignment fields.
`collision_params` supplies the proper `rotation_matrix` and the compatible
Euler orientation, omitting the unnecessary Euler-rate field. A planner can
build its full matrix preview from `rotation_matrix_at` without a rate-chart
singularity. `angular_velocity_at` retains the legacy chart behavior explicitly.

## Scope

The attitude model assumes a rigid body with constant physical inertia, known
initial attitude/spin and a planar orbit. The default omits all external torque;
gravity-gradient mode includes only the spherical central-field tidal torque.
Both omit aerodynamic, magnetic and solar-radiation torques, flexible motion,
fuel slosh and contact. The orbit drives the attitude through the known radius
and radial direction; attitude back-reaction on the prescribed analytical orbit
is neglected. This remains a limited environmental model, not a complete
prediction of a particular uncontrolled object in LEO.

This target module supplies pre-contact unactuated attitude and the free-flight
standoff reference. Attachment, coupled target/chaser momentum exchange, active
detumbling and orbital disposal require their own physical models and event
conditions; reaching the standoff is not any of those events. The maintained
[physical mission lifecycle](mission-lifecycle.md) supplies those separate
stages and preserves their individual gates and outcomes.

Torque-free conservation, gravity-gradient torque impulse/work balance,
quaternion normalization, proper rotations and independent physical-seconds
and numerical derivative checks support the implemented kinematics. They do not
establish fidelity to any specific debris
object, robustness to attitude-estimation error, hardware behavior or a
continuous-time collision certificate. Mission success remains the existing
recorded terminal-tube dwell condition; a failed solve or incomplete run is
retained as that outcome.
