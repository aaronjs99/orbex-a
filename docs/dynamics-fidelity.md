# Dynamics fidelity and the nonlinear replay boundary

**The current `1e-5 m/rad²` componentwise guard allowance does not qualify the matched full vector quadratic-drag plant at the evaluated saved states.** The largest evaluated component discrepancy is `0.033936410695 m/rad²`, approximately 3,394 times that allowance. Increasing a configured number from these finite samples would not establish a continuous bound. The current approach, contractor, tube and Arb evidence therefore retains its declared analytical Carter–Humi model scope.

The independent `orbexa.core.inertial_dynamics` service supplies nonlinear Cartesian propagation for studying that gap. It is maintained, reusable numerical propagation, with explicit central gravity, optional J2, atmosphere and drag declarations. It has not been substituted into the controller, parameter contractor, proof backend or completed mission records. Its numerical success does not itself certify a flow, collision avoidance, attachment or closed-loop success.

## Force model and frame

For spacecraft $i$, let $r_i\in\mathbb R^3$ be its inertial position in metres, $v_i\in\mathbb R^3$ its inertial velocity in metres per second, $R_i=\lVert r_i\rVert_2$ its radius, and $\mu>0$ the body's gravitational parameter. The independent variable is SI time $t$. The declared applied inertial acceleration is $a_{u,i}$. The service integrates


$$

 \dot r_i=v_i,\qquad
 \dot v_i=-\frac{\mu r_i}{R_i^3}+a_{J2}(r_i)+a_{D,i}(t,r_i,v_i)+a_{u,i}.

$$


For J2, let $J_2$ be the unnormalized zonal coefficient, $R_e>0$ its reference radius, $n_i=r_i/R_i$, $k_I$ the declared unit symmetry axis fixed in the inertial frame, and $s_i=n_i^T k_I$. The optional acceleration is


$$

 a_{J2}(r_i)=\frac{3J_2\mu R_e^2}{2R_i^4}
       \left[(5s_i^2-1)n_i-2s_i k_I\right].

$$


This is the usual J2-only Cartesian expression, expressed for an arbitrary fixed axis. The sign convention is $J_2=-C_{20}$ for an unnormalized zonal coefficient. A nonzero J2 requires an explicit reference radius; the service does not infer an Earth orientation or gravity-field product. The equation agrees with the primary implementation in [Orekit's J2-only perturbation](https://raw.githubusercontent.com/CS-SI/Orekit/13.1.2/src/main/java/org/orekit/forces/gravity/J2OnlyPerturbation.java).

For drag, let $\rho(t,r)\ge0$ be the declared density in kilograms per cubic metre, $v_{\mathrm{air}}(t,r)$ the declared inertial atmosphere velocity, and $\sigma_i=C_{D,i}A_i/m_i\ge0$ the spacecraft's isotropic drag area per mass in square metres per kilogram. Define $w_i=v_i-v_{\mathrm{air}}(t,r_i)$. The service uses the full vector law


$$

 a_{D,i}=-\tfrac12\rho(t,r_i)\sigma_i\lVert w_i\rVert_2w_i.

$$


Both speed and direction use the atmosphere-relative velocity. A missing atmosphere-velocity callback explicitly means a stationary atmosphere in the declared inertial frame. This implements the [NASA drag equation](https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/drag-equation/) in vector form, consistent with an [isotropic orbital drag model](https://raw.githubusercontent.com/CS-SI/Orekit/13.1.2/src/main/java/org/orekit/forces/drag/IsotropicDrag.java). It is not an atmosphere estimator: density, winds, rotation, area, mass and their provenance remain caller declarations. The present isotropic coefficient is fixed for each propagation; attitude-dependent aerodynamics and fuel-driven mass changes need a further model.

For the relative frame, let $e_r=r_t/\lVert r_t\rVert$, $e_n=(r_t\times v_t)/\lVert r_t\times v_t\rVert$, and $e_\theta=e_n\times e_r$, where subscript $t$ denotes the target. The frame matrix is


$$

 C=[-e_\theta\ \ e_r\ \ e_n].

$$


Thus relative $x$ points opposite transverse motion, $y$ points radially outward and $z$ points along orbit normal, matching the maintained elliptical dynamics. `LVLHFrame` computes both $C$ and its physical-time derivative $\dot C$ from the supplied target state and acceleration, including normal-vector motion under out-of-plane forces. If $p$ is relative position and $w=\dot p$ is relative velocity **in m/s**, its transformation is


$$

 r_c=r_t+Cp,\qquad v_c=v_t+Cw+\dot Cp.

$$


The inverse includes the same rotation-rate term. For the analytical state `[p, dp/dq]`, conversion first requires $w=\nu p'$, with $\nu=dq/dt$ from an explicitly chosen clock. Feeding `dp/dq` directly to this API is a unit error. The service does not infer the analytical anomaly clock from a perturbed orbit.

## Public propagation API

The public classes are lazy exports from `orbexa.core`:

| Class | Contract |
| --- | --- |
| `InertialGravity(mu_m3_s2, j2=0, reference_radius_m=None, symmetry_axis=(0,0,1), minimum_radius_m=0)` | Explicit central body and optional fixed-axis J2. Radius at or below the declared minimum is refused; this is a model boundary, not an impact model. |
| `QuadraticDrag(cda_over_mass_m2_kg, density_kg_m3, description, atmosphere_velocity_m_s=None)` | Density and optional atmosphere callbacks receive `(time_s, inertial_position_m)`; a nonempty declaration is required. |
| `LVLHFrame(target_state, target_acceleration_m_s2)` | `to_inertial` and `from_inertial` transform arrays with shape `(N,6)`; relative velocities are physical m/s. |
| `InertialDynamics(gravity, drag_models=())` | No drag models means gravity only; otherwise supply exactly one drag model or `None` for every propagated spacecraft. |
| `InertialPropagationResult` | Status, reason, reached sample times/states, counts, force/integration context and `success`; an incomplete prefix is never successful. |

`accelerations(time_s, states, applied_acceleration=None)` returns an `(N,3)` array in m/s². Cartesian states are `(N,6)` arrays of `[r_I, v_I]`; a single `(6,)` state is accepted. `propagate` has the following interface:

```python
result = model.propagate(
    initial_states, sample_times_s,
    max_step_s=1.0,                 # required, finite SI integration step
    rtol=1e-11,
    atol=(1e-6, 1e-6, 1e-6, 1e-9, 1e-9, 1e-9),
    max_steps=100000,
    applied_acceleration=None,
    control_description=None,
)
```

Sample times must be finite and strictly increasing, with at least two entries. Their first and last entries define the integration span; the initial state belongs to the first entry. An applied-acceleration callback receives `(time_s, current_states_copy)` and returns an `(N,3)` inertial m/s² array. It requires a nonempty `control_description`. Callbacks must be pure functions; a controller should issue and record its holds outside this propagation service, rather than mutate a controller during adaptive integrator evaluations.

For example, the following is a declared illustrative atmosphere, not measured Earth weather:

```python
import numpy as np
from orbexa.core import InertialGravity, QuadraticDrag, LVLHFrame, InertialDynamics

gravity = InertialGravity(
    3.986004418e14, j2=1.08262668e-3,
    reference_radius_m=6378137.0, minimum_radius_m=6378137.0,
    symmetry_axis=(0.0, 0.0, 1.0),
)
omega_air = np.array([0.0, 0.0, 7.292115e-5])
def density(time_s, position_m):
    altitude_m = np.linalg.norm(position_m) - 6378137.0
    return 2e-11 * np.exp(-(altitude_m - 300000.0) / 50000.0)
def atmosphere(time_s, position_m):
    return np.cross(omega_air, position_m)
declaration = "Illustrative fixed exponential density and co-rotating atmosphere"
drags = [QuadraticDrag(sigma, density, declaration, atmosphere)
         for sigma in (0.02, 0.045)]
model = InertialDynamics(gravity, drags)
target = np.array([0.0, 7000000.0, 0.0, -7546.0, 0.0, 0.0])
target_accel = model.accelerations(0.0, np.vstack((target, target)))[0]
frame = LVLHFrame(target, target_accel)
chaser = frame.to_inertial([[0.0, 4.0, 0.0, 0.0, 0.0, 0.0]])[0]
result = model.propagate(np.vstack((target, chaser)), np.linspace(0.0, 60.0, 121),
                         max_step_s=1.0, max_steps=10000)
if not result.success:
    print(result.status, result.reason, result.times_s[-1])
```

The integrator is SciPy DOP853 with dense sampling, an explicit accepted-step budget and explicit SI tolerances. These are numerical controls, not interval error bounds; see the [official SciPy integration documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html). Invalid initial declarations refuse before propagation. Expected runtime force/callback failures return a failure status, all requested samples reached, and the last accepted integration point. Step exhaustion does the same. No mutual gravity, contact, attitude, sensor, density uncertainty or mass-depletion dynamics are inferred.

## Measured qualification and discrepancy

The outside-checkout qualification retained source hashes, raw numerical arrays, force declarations and failure outcomes. The service source used for these measurements has SHA-256 `b2fc20c116c5f3f7b08ea49f27ef2db8344ac8a053b36d49990ee875576ecdaa`.

Independent service checks include a complete no-drag orbit against a separately solved Kepler equation, with semi-major axis `8,144,068.6585 m` and $\mu=3.986004418\times10^{14}\,\mathrm{m^3/s^2}$. At 721 samples, `max_step_s=20`, `rtol=1e-12`, and SI absolute tolerances `1e-7 m` / `1e-10 m/s`, maximum position differences were `5.292e-8 m` for the circular case and `1.464e-7 m` for eccentricity `0.18`. Energy and angular-momentum relative drift stayed below `2.89e-15` and `1.50e-15`, respectively. These are numerical agreement measurements, not certified global errors.

Further independent checks found J2 acceleration agreement with a finite-difference potential gradient within `2.23e-10 m/s²`, J2 covariance under a rotated symmetry axis within `5.01e-15 m/s²`, drag dissipation agreement within `2.23e-16` relative error, and a nonplanar frame-derivative finite-difference difference below `3.89e-15 s⁻¹`. Invalid force inputs, unmatched drag counts, degenerate frames, an over-tight binary64 tolerance, a finite step budget and an intentionally failing force callback were also exercised; refusals retained their numerical prefixes.

For the model discrepancy, the retained source is the immutable 418-update stopped prefix saved in `physical-fleet-eight-resumed/checkpoint.json`, whose file SHA-256 is `2e58f7e1ecf61cb59fa5bcca00750d25b22fa6d45a95357a33b2022b091d4930`. It has eight chasers, $e=0.18$, $\alpha=2\times10^{-7}$, $\beta=4.5\times10^{-7}$, and final recorded anomaly approximately `6.56593 rad`. The comparison deliberately retains that stopped prefix and does not relabel it as a successful run.

To isolate the paper approximation, the full vector drag uses $\rho(r)=K/r$, a stationary inertial atmosphere and $\sigma_t=2\alpha/K$, $\sigma_c=2\beta/K$, with $K=2\times10^{-5}\,\mathrm{kg/m^2}$. Thus its inertial target and chaser forces are exactly $-\alpha\lVert v_t\rVert v_t/r_t$ and $-\beta\lVert v_c\rVert v_c/r_c$. This is a mathematical normalization, not a calibrated physical atmosphere. The primary [Carter–Humi paper](https://doi.org/10.2514/2.5010) starts from the vector quadratic-drag law and makes small-relative-motion and radial-speed approximations; the maintained relative factory is an analytical approximation to that law.

At 25 predetermined saved nodes, all eight chasers were compared instantaneously in a common physical LVLH frame and clock:

| Force variant | Evaluated states | Maximum component discrepancy, m/rad² |
| --- | ---: | ---: |
| Central gravity, no drag | 200 | `1.417892e-5` |
| Equal full vector drag constants | 200 | `1.427436e-5` |
| Full vector drag with declared target/chaser difference | 200 | **`3.393641e-2`** |

The maximum differential-drag norm corresponds to `2.602475e-8 m/s²` at its sampled orbital rate. SI acceleration can be small while an anomaly-acceleration allowance is violated; the units and clock conversion cannot be skipped.

Five further initial nodes were fixed before the replays: `0, 104, 209, 313, 417`, spanning initial, quarter, half, three-quarter and final-prefix phases. Each replay propagated the target and eight chasers together, for one saved outer interval and for 300 seconds. Both models received the **same physical inertial acceleration history**, formed from the saved outer mean command. The source's unsaved dense feedback was not reconstructed. The independent target's actual frame was used when converting its propagated relative position and velocity. Maximum differences over all five phases and eight agents were:

| Force variant | One saved interval, relative position | 300 s, relative position | 300 s, physical relative velocity |
| --- | ---: | ---: | ---: |
| Central gravity, no drag | `6.852e-9 m` | `5.037e-7 m` | `3.295e-9 m/s` |
| Matched full vector differential drag | `4.108e-6 m` | **`1.11350e-3 m`** | `7.230e-6 m/s` |
| Same drag plus declared J2 | `5.483e-6 m` | **`1.46073e-3 m`** | `9.385e-6 m/s` |

The J2 variant declares `J2=1.08262668e-3`, radius `6,378,137 m`, and fixed axis `(0,0.6,0.8)`; it adds a deliberately unmodeled force rather than isolating the drag approximation. Absolute target-orbit divergence is recorded separately and reaches hundreds of metres even when the relative discrepancy remains at millimetre scale. These open-loop, finite-time results do not establish a nonlinear-truth closed-loop mission or a uniform safety allowance.

The outside evidence bundle `work/completion-2026-10-14/dynamics-fidelity/` contains `service-qualification.json`, `model-discrepancy.json`, `matched-segment-samples.npz`, and the exact source/declaration information. Generated evidence and figures remain outside the maintained source tree.

## A conditional continuous discrepancy bound

This subsection supplies an analytical route to a bound, **not a certificate that its hypotheses hold in the mission**. It covers central gravity and stationary inverse-radius vector drag only. J2, atmosphere rotation/wind, general density laws, unknown input orientation and uncertain sensing require additional terms or an augmented validated plant.

Assume the full target moves in a fixed plane under central gravity and drag $-\alpha\lVert v_t\rVert v_t/r$. Use its actual increasing polar-angle clock $q$, with $\nu=dq/dt>0$. Define $r=\lVert r_t\rVert$, $k=r'/r$, $c=\mu/(r^3\nu^2)$, and the fixed skew matrix


$$

 Z=\begin{bmatrix}0&-1&0\\1&0&0\\0&0&0\end{bmatrix}.

$$


Let $p,p'$ be the actual relative state in the stated LVLH frame. Define $w_t=(-r,rk,0)^T$, $w=w_t+p'+Zp$, $s=\lVert(0,r,0)^T+p\rVert_2$, and $L=\sqrt{1+k^2}$. The target's exact angular-momentum equation gives $\dot\nu/\nu^2=-\alpha L-2k$. The full relative anomaly equation, before applied control, is therefore


$$

\begin{split}
 p''_{\mathrm{full}}={}&c(0,r,0)^T-c(r/s)^3[(0,r,0)^T+p]\\
 &-2Zp'+2k(Zp+p')-Z^2p
   -[\beta\lVert w\rVert_2/s-\alpha L]w.
\end{split}

$$


The maintained Carter–Humi coefficients, evaluated with these **same actual** $r,k,c$, give


$$

 p''_{\mathrm{CH}}=c\operatorname{diag}(-1,2,-1)p
       -2Zp'+2k(Zp+p')-Z^2p-(\beta-\alpha)w.

$$


In particular, even at $p=p'=0$, their difference is


$$

 (\beta-\alpha)r(\sqrt{1+k^2}-1)(1,-k,0)^T.

$$


This bias vanishes for a circular target clock or equal drag constants, but need not vanish at a nonzero eccentric radial speed. It explains why centering the tracking tube or merely shrinking relative position does not remove this model error.

For a uniform regional bound, declare positive radius bounds $R_-\le r\le R_+$, $0<c\le C_+$, $|k|\le K$, $\lVert p\rVert_2\le P<R_-$, and $\lVert p'\rVert_2\le V$. Let $\Gamma\ge|\beta-\alpha|$, $B_+\ge\beta\ge0$, $L_*\ge\sqrt{1+K^2}$, and $D=P+V$, treating radians as dimensionless in the inequalities. Euclidean acceleration-error bounds for the gravity and drag remainders are


$$

\begin{split}
 E_G&=\frac{3C_+P^2}{R_-}(1-P/R_-)^{-4},\\
 E_D&=\Gamma(L_*-1)(R_+L_*+D)
      +B_+(D+L_*P)\frac{R_-L_*+D}{R_--P}.
\end{split}

$$


For completeness, the directional Hessian of central gravity at radius $R$, along a unit direction $h$, is $\mu R^{-4}[6(n^Th)h+(3-15(n^Th)^2)n]$, with $n$ the unit radius vector. Its norm is at most $6\mu/R^4$; Taylor's integral remainder yields $E_G$. For drag, the reverse triangle inequality gives $|\lVert w\rVert/s-L|\le(D+L_*P)/(r-P)$, while $\lVert w\rVert\le rL_*+D$. Separating the omitted $L-1$ term yields $E_D$. Thus $\lVert p''_{\mathrm{full}}-p''_{\mathrm{CH}}\rVert_2\le E_G+E_D$; the same number bounds each component.

The implemented reference orbit uses its own $r_a,k_a,c_a$. To compare with those coefficients at the **same actual clock**, also require continuous bounds $|r-r_a|\le E_r$, $|k-k_a|\le E_k$, $|c-c_a|\le E_c$, with both orbits in the declared radius/curvature region. Their additional coefficient discrepancy is bounded by


$$

 E_O=2E_cP+2E_k(P+V)+\Gamma[(1+K)E_r+R_+E_k].

$$


If physical input is scaled with the analytical rate $\nu_a^2$ rather than the actual $\nu^2$, let $C_->0$ bound $c_a$ below, $\lVert u\rVert_2\le U$, and require $E_c<C_-$, $E_r<R_-$. Because $\nu_a^2/\nu^2=cr^3/(c_ar_a^3)$, define


$$

 H_+=(1+E_c/C_-)(1+E_r/R_-)^3,\quad
 H_-=(1-E_c/C_-)(1-E_r/R_-)^3,

$$


so an additional input discrepancy bound is $E_U=U\max(H_+-1,1-H_-)$. This assumes the input direction is expressed in the exact actual LVLH frame; a frame estimate needs its own bound. Numerical process error and sensing/reference uncertainty are additional, separate allowances.

As an explicitly conditional illustration, choose radii `[6.5e6,1e7] m`, $|k|\le1/4$, curvatures `[0.65,1.5]`, $P=20\,\mathrm m$, $V=100\,\mathrm{m/rad}$, $U=174\,\mathrm{m/rad^2}$, $\alpha\in[1.5,2.5]\times10^{-7}$, $\beta\in[3.5,5.5]\times10^{-7}$, $E_r=10\,\mathrm m$, and $E_k=E_c=10^{-5}$. Using the exact rational outward bound $L_*=51539/50000$, whose square exceeds $1+(1/4)^2$, exact rational arithmetic evaluates

| Conditional term | Upper bound, m/rad² |
| --- | ---: |
| $E_G$ | `0.000276926486` |
| $E_D$ | `0.126990831775` |
| $E_O$ | `0.002845` |
| $E_U$ | `0.003480013591` |
| Sum, rounded upward for display | `0.133592771851` |

**Those target/reference/clock hypotheses have not been proven or enforced by the maintained mission guard. This number is not a default allowance, and is not a certificate for the saved mission.** The outside `continuous-bound-qualification.json` retains the exact rational values. A separate 512-state numerical comparison cross-checked the compact full relative equation against the independent inertial force/frame transform, with maximum component disagreement `8.01e-9 m/rad²`; those samples check the algebra, rather than prove continuous domain containment.

## What remains before a nonlinear safety claim

The current Arb backend exposes the maintained analytical `enclose_orbital(HeldInputRequest, OrbitalModelBounds)` and constant `enclose_affine` inclusions. Its orbital request accepts a componentwise `acceleration_disturbance` box. A rigorously bounded additive discrepancy could use that interface **only after** all operating-region, model, input, target and clock hypotheses are bound to and checked throughout each proposed hold. Existing candidate relative boxes can help bound $p,p'$; they do not supply a validated full target orbit, its deviation from the analytical orbit, or its physical clock.

Central gravity and the matched vector-drag discrepancy are continuously boundable away from zero radius, as above. Arb arithmetic can also enclose their algebraic norm, square-root and inverse expressions over a positive-radius domain. The maintained backend does not yet provide the augmented nonlinear inertial/target vector field and validated physical-time propagation needed to do that automatically. Its present physical-time and target-kinematic bounds use the analytical orbit. A larger relative forcing box alone does not correct those clocks, target geometry, or the parameter contractor's model assumptions.

A complete bridge therefore needs either a validated nonlinear target/clock enclosure plus checked regional remainders for every hold, or an augmented nonlinear validated flow implementation covering the target, chasers, declared atmosphere/J2, applied input, and uncertainty. Both require a corresponding consistent sensing/contractor interpretation and failure continuation. Until that work is admitted and exercised, the analytical mission evidence and the independent nonlinear numerical replay evidence must retain their distinct scopes.
