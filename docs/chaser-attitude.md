# Free-flying chaser attitude before capture

`orbexa.simulation.approach.physical.attitude` propagates the attitude of each finite chaser bus over a saved approach in physical seconds. Translation alone supplies no bus orientation or spin. This module starts the buses from declared attitudes, applies bounded external nozzle couples, retains the complete numerical attitude and actuator histories, and exposes the final physical states to the capture model.

The Envisat mission uses a representative 130 kg initial wet bus of radius 0.55 m, with 30 kg dry mass and 100 kg initial propellant. Its constant initial wet inertia is 15.73 kg m² about each axis. These are declared vehicle-concept properties. The bus starts with zero inertial angular velocity and a body-frame rotation-vector offset `[0.4, -0.3, 0.2]` rad from its assigned mounting attitude. That offset has magnitude 0.5385 rad; alignment is simulated.

## Reference and frames

Let the rotation matrix `R_T` carry target-body vectors into ECI. Let `n` be a recorded assigned outward surface normal, and let `t` be the normalized cross product of target-body z with `n`. The mounting matrix `M` has columns `n`, `t`, and `n × t`. Chaser x therefore points outward, and a gripper on chaser −x points toward the target. The desired chaser-to-ECI matrix `R_d`, desired chaser-body angular velocity `ω_d`, and desired body angular acceleration `α_d` are

$$
M=[n\quad t\quad n\times t],\qquad
R_d=R_T M,\qquad
\omega_d=M^T\omega_T,\qquad
\alpha_d=M^T\dot\omega_T.
$$

The target record must provide aligned, increasing anomaly and physical-time samples, unit body-to-ECI quaternions in scalar-last `xyzw` order, and physical target body angular velocities. Angular accelerations must also be recorded for a forced target. An older explicitly torque-free rigid-body record can instead recover them from its declared inertia and Euler's equation. Missing gravity-gradient acceleration is rejected.

Between saved target nodes, the module uses shortest-arc SLERP for orientation and componentwise linear interpolation for the independently recorded spin and acceleration. Every saved approach node is retained exactly on the output timeline. The SLERP derivative need not equal the separately interpolated physical spin, and its acceleration need not equal the supplied acceleration. The maximum endpoint spin discrepancy against each SLERP arc rate is recorded. This is a declared numerical tracking reference, not a certified target-truth enclosure or a new target attitude integrator.

## Physical feedback and actuation

Let `R` be the actual chaser-to-ECI matrix, `ω` its inertial angular velocity expressed in the actual chaser frame, and `J` its constant wet inertia. The transport matrix `A = RᵀR_d` maps desired-frame components into the actual chaser frame. The attitude error `e_R` is the axial vector of the skew matrix shown below, and the rate error `e_ω` compares actual spin against transported desired spin. The operator `vee` maps a skew matrix to its axial three-vector. The positive natural frequency `ν_c` and damping ratio `ζ` are controller settings, defaulting to 0.25 rad/s and 0.9.

$$
e_R=\tfrac12\operatorname{vee}(R_d^TR-R^TR_d),\qquad
e_\omega=\omega-A\omega_d,
$$

$$
\tau_{\rm request}
=\omega\times J\omega
+J[-\nu_c^2 e_R-2\zeta\nu_c e_\omega
-\omega\times(A\omega_d)+A\alpha_d].
$$

This is the geometric rigid-body tracking structure of [Lee, Leok and McClamroch](https://arxiv.org/abs/1003.2005), applied to the rotational subsystem with an explicit spacecraft actuator model. The implementation uses held feedback, saturation and a sampled interpolated reference. The paper's continuous ideal-controller stability result is not automatically a theorem for this implementation, and no global saturated tracking guarantee is claimed.

Each of three independent torque axes has a pure couple. For axis `i`, two equal signed nozzle commands act in opposite directions at opposite offsets of length `l`. The Envisat concept uses `l = 0.4 m` and active-nozzle magnitude at most `F_max = 3 N`; signed commands mean that a colocated opposed nozzle is selected when the sign changes. The requested nozzle command is clipped before forces and torque are reconstructed from the actual geometry. The ideal pair cancels net force, and the maximum axis torque is `2lF_max = 2.4 N m`.

Let `f_i` be the signed force at each location of the axis-i couple. Let `I_sp` be the specified impulse, defaulting to 300 s, and let `g_0 = 9.80665 m/s²` be standard gravity. The actual torque and total mass flow are

$$
\tau_i=2lf_i,\qquad |f_i|\le F_{\max},\qquad
\dot m_{\rm fuel}
=\frac{2\sum_i|f_i|}{g_0I_{sp}}
=\frac{\sum_i|\tau_i|}{lg_0I_{sp}}.
$$

No reaction-wheel momentum removal or unbounded virtual torque is used. The six signed commands, actual applied torque, net force, mass flow and saturation status are retained for every hold. Continuous throttling and ideal matched nozzles are explicit assumptions. No plume obstruction, minimum impulse bit, power, thermal, alignment-error, sensor-error or external-disturbance model is included.

## Propagation, fuel and finite-body scope

Let `q = (v,s)` be the actual body-to-ECI quaternion, with vector part `v` and scalar part `s`. Within a control hold, the applied body torque `τ` is constant. Quaternion kinematics and Euler's rigid-body equation are integrated with RK4 and quaternion normalization:

$$
\dot q=\tfrac12(s\omega+v\times\omega,\ -v^T\omega),\qquad
J\dot\omega=\tau-\omega\times J\omega.
$$

Feedback is recomputed at most every 0.5 s and at every original approach sample. Internal integration steps are at most 0.25 s by default. No Euler chart is used. The integration also accumulates ECI angular impulse and rotational work, permitting numerical checks of angular momentum and work-energy balance. These checks are numerical evidence, not validated ODE enclosures.

The caller must provide a nonnegative translational propellant debit for every bus. It is reserved before attitude propagation and carried into capture with the attitude fuel debit. The complete runner can derive this allowance from authenticated held-input interval proofs, using the [per-hold bound](mission-lifecycle.md#held-command-propellant-bound). Insufficient installed fuel is rejected; exhaustion during tracking stops before the first unexecutable hold and returns a failed result with the common fleet prefix.

Inertia remains the initial wet-bus inertia during this replay. The total mass debit, including translation and attitude propellant, is restricted to 2% of initial wet mass by default. Exceeding that scope fails before the next hold. This bounds the permitted mass-change fraction; it is not a proof that an unknown tank layout has the same fractional inertia error. Capture receives the remaining physical mass and spherical specific inertia. A larger-depletion or tank-resolved simulation requires a variable-inertia open-system attitude model.

Attitude and translation remain decoupled. The saved approach positions remain unchanged. The spherical bus envelope is orientation independent, and nozzles must lie inside it. The gripper is assumed stowed during approach; the attitude replay does not establish safe arm deployment or the swept geometry of extended appendages. Actual COM positions must separately satisfy the finite-body approach and capture gates.

## Capture interface

The tracker returns success only when the entire supplied timeline was propagated and every final orientation and inertial spin error passes its configured limit. Defaults are 0.01 rad and 10⁻⁵ rad/s. The spin comparison rotates the actual body rate into ECI, so the returned rate stays in the actual chaser frame even when alignment is imperfect.

```python
from orbexa.simulation.approach.physical.attitude import ChaserAttitudeTracker
from orbexa.simulation.contact.capture import CaptureModel, CaptureOptions

# Supply this allowance from the chosen translation/propulsion study.
attitudes = ChaserAttitudeTracker().track(
    approach,
    translation_fuel_debit_kg={key: allowance[key]
                               for key in approach.actual_trajectories},
)
if not attitudes.success:
    raise RuntimeError(attitudes.message)

capture = CaptureModel(
    chasers=attitudes.capture_chasers,
    options=CaptureOptions(max_spin_error_rad_s=1e-5),
).capture(approach, chaser_attitudes=attitudes.final_attitudes)
```

`final_attitudes` supplies `(body-to-ECI quaternion, actual body angular velocity)` for each bus. `capture_chasers` supplies corresponding physical properties with both fuel debits removed. A failed result also retains its last states for diagnosis; callers must require `success` before handing them to a lifecycle stage. `to_dict()` serializes the time axis, full node and hold histories, controller/actuator settings, numerical balance audits, assumptions and final handoff data.

The capture model independently checks translation qualification, geometry, tube bounds, relative latch speed, attitude and spin. Attitude-tracking success alone does not establish a successful capture.

## External numerical evidence

Qualification artifacts belong outside the repository. The initial bounded study replayed the saved 4,851.9 s physical single-target approach. From 0.5385 rad initial orientation error and zero bus spin, the simulated bus reached 1.75 × 10⁻⁶ rad orientation error and 1.64 × 10⁻⁷ rad/s spin error. Attitude use was 6.62 × 10⁻⁵ kg with an explicit 0.02 kg translation allowance. The largest nozzle force was 0.0129 N. These are attitude-replay results for that specific saved history; the old point-mass translation remains unsuitable for finite-bus capture.

A separate synthetic eight-bus fixture placed COM references at 0.165 m side standoff, simulated all eight misaligned buses, and exercised a physical-attitude handoff through the rigid-latch capture gates. This fixture verifies interfaces and modeled balances. It is not an eight-agent ADTMPC approach result or hardware capture evidence. The physical controller-run evidence must be assessed separately.

An independent DOP853 integration of a non-spherical rigid body under a held torque checked the propagation kernel, angular impulse and work-energy equations. Separate boundary probes covered nozzle saturation, fuel exhaustion without committing an unexecutable hold, invalid inertia, missing forced-target acceleration, malformed timing/quaternions and fuel-allowance scope. Passing these probes does not certify continuous collision avoidance, real thruster performance or contact mechanics.
