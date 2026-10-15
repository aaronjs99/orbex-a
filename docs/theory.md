# ORBEX-A mathematical foundations

This document states the model, derives the implemented feedback and tube,
and separates mathematical inclusion from numerical evidence. Read it together
with the [equation audit](theory-audit.md), [deployment formulation](deployment.md)
and [research review](research-review.md). The approach task is translational
free-flight rendezvous at a specified standoff from a rotating target. The
physical profile supplies quaternion rigid-body motion; modeled attachment,
detumbling and orbital lowering have their own [SI dynamics](mission-lifecycle.md).

## 1. Coordinates, clocks and approximation

The independent variable is **unwrapped true anomaly**, denoted by $q$ and measured
in radians. A prime means differentiation with respect to $q$; a dot means
differentiation with respect to physical seconds. The relative position $p$ is a
three-vector in meters in a right-handed local orbital frame. Its axes are
opposite the transverse orbital direction, radially outward and orbit normal.
On an eccentric orbit, the first axis is not opposite the full inertial velocity.
The anomaly velocity $v=p'$ has units m/rad. The state $x=[p^T,v^T]^T$ has six
components. The control $u$ is an anomaly acceleration in m/rad².

The target radius from the central body is $r_o$, its anomaly rate is $\nu$, its
initial specific angular momentum is $h_o$, the gravitational parameter is
$\mu$, and the target drag constant is $\alpha$. The semi-major axis is $a$ and
eccentricity is $\epsilon$. The orbital phase is $q_0$. These names distinguish
orbital radius from a tube radius and drag from feedback bandwidth.

$$
h_o^2=\mu a(1-\epsilon^2),\qquad
r_o(q)=\frac{h_o^2(1+4\alpha^2)}
 {\mu[\exp(2\alpha q)+\epsilon\cos(q-q_0)]},\qquad
\nu(q)=\frac{h_o\exp(-\alpha q)}{r_o(q)^2}.
\tag{1}
$$

The denominator must remain positive over every uncertainty and prediction
interval. The fixed-axis convention is authoritative in the interval model.
The physical velocity is $\dot p=\nu v$. A thruster acceleration increment
$a_{\rm thr}$ maps to $u=a_{\rm thr}/\nu^2$. The complete physical relative
acceleration additionally contains $\dot\nu v$; one must not label $p''$ as m/s².

$$
\dot p=\nu p',\qquad \ddot p=\nu^2p''+\dot\nu p',
\qquad a_{\rm thr}=\nu^2u.
\tag{2}
$$

The translational model follows the **pre-transformation Carter–Humi
approximation**: small relative separation, small relative inertial speed,
moderate eccentricity, an inverse-radius drag-density law, and neglected
difference between the two spacecraft's orbital angular rates. This last
assumption is consequential: the model is not the exact first variation of a
general quadratic-drag two-spacecraft plant. The physical-coordinate coefficient
signs agree with their equations 19–21 after setting the density factor to
inverse radius. [Carter and Humi (2002)](https://doi.org/10.2514/2.5010).

Define the logarithmic radius derivative $k=r_o'/r_o$, the drag difference
$\gamma=\beta-\alpha$ where $\beta$ is the chaser drag constant, the curvature
$c=\mu r_o\exp(2\alpha q)/h_o^2$, and the damping coefficient $b=2k-\gamma$.
Let $A_p$ and $A_v$ be the three-by-three position and velocity acceleration
blocks, and $d_a$ the affine anomaly-acceleration vector. The model is

$$
p'=v,\qquad v'=A_p p+A_v v+d_a+u,
\tag{3}
$$

$$
A_p=\begin{bmatrix}1-c&-b&0\\b&1+2c&0\\0&0&-c\end{bmatrix},\quad
A_v=\begin{bmatrix}b&2&0\\-2&b&0\\0&0&b\end{bmatrix},\quad
d_a=\begin{bmatrix}\gamma r_o\\-\gamma r_o'\\0\end{bmatrix}.
\tag{4}
$$

At zero drag and zero eccentricity, $k=0,c=1$ and these reduce to HCW in the
declared axis convention. Common nonzero drag does not produce the exact
general quadratic-drag relative damping: that difference is approximation bias,
not a parameter-identification achievement. No parameter box can automatically
absorb unmodeled atmosphere, J2, sensor errors, actuator dynamics or contact.
The independent [Cartesian dynamics comparison](dynamics-fidelity.md) measures
this bias using nonlinear central gravity and full vector drag. Its evaluated
discrepancy exceeds the representative approach's extra acceleration allowance.
The analytical-model inclusion claims below therefore do not certify that richer
plant. Evaluated samples are not a uniform model-error bound.

## 2. Moving reference and occupied target

The body-to-LVLH rotation is $\mathcal R(q)$, a proper rotation matrix.
For the prescribed-target profile it is a product of active rotations
$R_x(\varphi)R_y(\vartheta)R_z(\psi)$ with prescribed Euler-angle functions of
anomaly. Their derivatives are Euler rates per radian of anomaly, not body
angular velocity in rad/s. The physical target instead propagates a quaternion
and physical body spin, then derives proper rotation matrices and their anomaly
derivatives including orbital-frame motion; no Euler chart is needed by geometry.
The body-fixed desired standoff point is $g_B$.
The desired state is $x_g=[p_g^T,v_g^T]^T$; $a_g$ is its anomaly acceleration.

$$
p_g=\mathcal Rg_B,\qquad v_g=\mathcal R'g_B,\qquad
a_g=\mathcal R''g_B,\qquad
u_g=a_g-\widehat A_a x_g-\widehat d_a.
\tag{5}
$$

Here $\widehat A_a=[\widehat A_p\ \widehat A_v]$ and $\widehat d_a$ use the
controller's nominal parameter point. Equation (5) supplies preview and nominal
feedforward under the model. Desired pose and occupied geometry use the same
rotation, anomaly and physical frame. They have different optimization roles:
the goal attracts the nominal state; the solid target excludes the entire
admissible position tube. A moving goal is not interchangeable with an obstacle.

For a body cylinder, let $r_c$ be its radius and $\ell_c$ its half-length. Its
world occupied set is $\mathcal R(q)K_B$. A unit body-frame separating normal
$n_B$ has world direction $n_W=\mathcal R n_B$. Its support is
$h_K(n_B)=r_c\sqrt{n_{B1}^2+n_{B2}^2}+\ell_c|n_{B3}|$.
This support description handles side and cap geometry without pretending that
every position belongs to one fixed approach halfspace.

## 3. Uncertainty sets and point estimates

The uncertain parameter vector is $\theta=(\epsilon,\alpha,\beta)$. The retained
parameter box $\Theta_k$ is a closed Cartesian product at update $k$.
The nominal point $\widehat\theta_k$ must belong to that box. A point estimate
and a feasible set are separate objects: a successful local fit is not an
outer enclosure of all consistent parameters.

The acceleration-coefficient difference is $\Delta A_a=A_a(\theta)-\widehat A_a$;
the affine difference is $\Delta d_a=d_a(\theta)-\widehat d_a$.
The nonnegative arrays $\delta_A$ and $\delta_d$ bound their entrywise absolute
values over the complete anomaly interval and parameter box. Their shapes are
three-by-six and three. Parameter dependence is nonlinear. Evaluating selected
corners, or selecting the corner with largest matrix norm, does not establish an
entrywise envelope throughout the box. The implementation uses interval
coefficient/derivative bounds and a mean-value enclosure with domain checks.

Membership observations consist of two endpoint state boxes, a held input and
an interval width. Let $\mathcal F_{\theta}^{\rm RK4}$ be the specified held-input
RK4 transition and $X_j$ the state measurement box at node $j$. A candidate
parameter box $B$ may be pruned only if its outward evaluated image has empty
intersection with the next observation box:

$$
\mathcal F^{\rm RK4}_{B}(X_j,u_j)\cap X_{j+1}=\varnothing.
\tag{6}
$$

The contractor retains all unresolved and unevaluated boxes when its work
budget ends, then returns their outer hull intersected with the prior box.
Consequently nested boxes preserve every parameter consistent with the declared
interval transition and measurement bounds, conditional on the enclosure
arithmetic. A retained parameter need not fit every observation: interval
over-approximation can prevent exclusion. This guarantee concerns the recorded
RK4 model; it is not a theorem about arbitrary continuous physical plants.

`MissionScenario(initial_belief=...)` allows an independently specified controller
prior. Historical demonstrations used config-centered synthetic priors, exact
states and exact target preview. Those runs do not establish robustness to
sensing errors or independent full identification of all three parameters.

## 4. Sliding coordinates and the continuous-feedback result

The nominal state is $z$, the actual-minus-nominal error is $e=x-z$, and its
position and anomaly-velocity parts are $e_p,e_v$. The positive diagonal sliding
gain is $\Lambda$ with units 1/rad. The sliding error is $s=e_v+\Lambda e_p$.
The component half-widths $\Omega\ge0$ and $\Phi\ge0$ bound position and sliding
error respectively; the implied velocity half-width is
$V=\Phi+\Lambda\Omega$. The nonnegative ancillary bandwidth $\kappa$ also has
units 1/rad and is distinct from both drag constants.

$$
\mathcal T(\Phi,\Omega)=
 \{e:\ |e_p|\le\Omega,\ |e_v+\Lambda e_p|\le\Phi\},
\qquad |e_v|\le\Phi+\Lambda\Omega.
\tag{7}
$$

The nominal and actual models share the input matrix. For continuous exact
feedback, the boundary-layer ancillary input is

$$
u_a^*(q)=-\widehat A_a(q)e-\Lambda e_v
 -[\delta_A|e|+\kappa\Phi]\odot\operatorname{sat}(s\oslash\Phi).
\tag{8}
$$

Products and division marked by $\odot,\oslash$ are componentwise. Saturation is
clipped to the interval $[-1,1]$. At zero layer width, the implementation uses
the sign convention; any contained component then has zero sliding error.

Let $Z$ enclose the absolute nominal state throughout one held-nominal-input
interval, and let $\eta$ be a genuinely bounded additional anomaly acceleration.
Define $w_0=\delta_A Z+\delta_d+\eta$. For the continuously recomputed input,
the upper right Dini derivative at a positive sliding boundary obeys
$D^+|s_i|\le-\kappa_i\Phi_i+w_{0i}$. Thus the positive comparison

$$
\Phi'=-\kappa\Phi+w_0,\qquad
\Omega'=-\Lambda\Omega+\Phi
\tag{9}
$$

preserves error containment under the stated exact continuous assumptions.
This is the relevant boundary-layer construction behind DTMPC; its general
idea is established literature. [Lopez, Slotine and How (2019)](https://arxiv.org/abs/1907.06553).
Equation (9) alone does **not** justify holding equation (8) between samples.

## 5. Derived correction for held ancillary feedback

This section gives the bounded-term correction used by the revised controller.
The outer interval width is $h>0$ and the maximum feedback hold is $\tau>0$, both
in radians. Bandwidth is constant on that outer interval. The candidate peak
half-widths $\bar\Phi,\bar\Omega$ enclose the comparison over it. The matrices
$\bar A$ and $A_{\rm var}$ are nonnegative: $\bar A$ encloses
$|\widehat A_a|$, while $A_{\rm var}$ bounds
$|\widehat A_a(q)-\widehat A_a(q_j)|$ within any feedback hold starting at $q_j$.
An outer-interval coefficient width is a conservative $A_{\rm var}$; it is
already a change and must not be multiplied by $\tau$ again.

Define the velocity-error peak $\bar V$, the complete state-error peak $E$, and
the ancillary input reserve $R_u$, all componentwise bounds:

$$
\bar V=\bar\Phi+\Lambda\bar\Omega,\qquad
E=\begin{bmatrix}\bar\Omega\\\bar V\end{bmatrix},\qquad
R_u=(\bar A+\delta_A)E+\Lambda\bar V+\kappa\bar\Phi.
\tag{10}
$$

The reserve includes nominal drift cancellation, sliding position feedback,
the uncertain-error term, and bandwidth damping. Nominal controls must satisfy
$u_{\min}+R_u\le u_{\rm nom}\le u_{\max}-R_u$. An empty interval is infeasible;
saturating a controller that requires more reserve does not preserve its proof.

Before a hypothetical first tube exit, define derivative bounds $D_v,D_e,D_s$
for velocity error, complete error and sliding error:

$$
D_v=(\bar A+\delta_A)E+w_0+R_u,\quad
D_e=\begin{bmatrix}\bar V\\D_v\end{bmatrix},\quad
D_s=D_v+\Lambda\bar V.
\tag{11}
$$

For a hold starting inside the tube, changes satisfy
$|e(q)-e(q_j)|\le\tau D_e$ and $|s(q)-s(q_j)|\le\tau D_s$.
The uncertain-error drift and held robust saturation input each have magnitude
at most $\delta_AE$; bound them separately rather than claiming cancellation.
The resulting hold-residual forcing is

$$
\rho=2\delta_AE+A_{\rm var}E
 +\tau[\bar A D_e+\Lambda D_v+\kappa D_s],\qquad
w=w_0+\rho.
\tag{12}
$$

Every term has units m/rad². There is no division by a potentially vanishing
layer and no unspoken smallness condition. The price is conservatism, especially
when parameter uncertainty or the tube is large. Candidate peaks, forcing and
reserve must be solved together; adding equation (12) after solving would break
the actuator and geometry constraints.

**Conditional inclusion argument.** Write the held controller as its sampled
linear cancellation/damping plus a robust term bounded by $\delta_AE$.
Add and subtract the continuous linear cancellation at the current state.
The difference is bounded by the coefficient-change and derivative terms in
equation (12). Therefore throughout the candidate tube,
$D^+|s|\le-\kappa|s|+w$.
The scalar comparison principle gives $|s|\le\Phi$ until any proposed first
exit, where it contradicts outward crossing. In turn
$D^+|e_p|\le-\Lambda|e_p|+\Phi$, so the position comparison contains the error.
The argument requires initial containment, fixed shared clock/context, the
specified parameter box, valid coefficient/nominal enclosures, input reserve,
and an additional acceleration bound that actually covers the claimed model.

For constant $w,\kappa$ the sliding comparison is exact. The positive position
surrogate uses the constant stage peak $\bar\Phi$ and is conservative. Starting
values are $\Phi_0,\Omega_0$; $\delta q$ is elapsed anomaly within the stage:

$$
\Phi(\delta q)=e^{-\kappa\delta q}\Phi_0
 +\frac{1-e^{-\kappa\delta q}}{\kappa}w,\qquad
\Omega(\delta q)=e^{-\Lambda\delta q}\Omega_0
 +\frac{1-e^{-\Lambda\delta q}}{\Lambda}\bar\Phi.
\tag{13}
$$

Zero gains use their continuous limits. Each component of these constant-source
comparisons is monotone, so endpoint peak constraints suffice for their stage
peaks. The runtime reuses the **stored solved** forcing, peaks, hold width and
reserve, with a preflight consistency check before applying any input.

The original continuous-only formula has a concrete counterexample at the
default hold: in the circular orbit-normal mode, choose position error 0.003 m,
anomaly-velocity error −0.058 m/rad, layer width 0.002 m/rad, sliding gain 20/rad,
bandwidth 80/rad and hold $\pi/(200\cdot32)$ rad. The initial sliding error is
on its allowed boundary. Exact held motion exceeds the old next envelope by
$8.8961\times10^{-7}$ m/rad. The corrected bound contains it. This disproves a
general transfer from continuous feedback; it does not assert that a particular
historical mission left its tube.

## 6. Discretization and numerical claims

The optimizer shoots one outer RK4 step; the executor propagates smaller held
RK4 steps. The actual center used for feedback remains the persistent runtime
nominal center. A recorded six-component center defect compares it to the
optimizer's next node; its position and velocity components have different
units. The old mixed Euclidean norm is a legacy diagnostic, not a physical
distance or an enclosure-transfer theorem.

The held-input comparison in section 5 is a continuous ODE inclusion result.
Finite RK4 and host floating-point arithmetic introduce additional gaps.
The configured numerical reserve is an **assumed extra acceleration allowance**,
not a derived global integration remainder. Successful dense replay and outward
endpoint rounding support numerical implementation checks without making a
machine-verified interval ODE certificate. A stronger certificate requires a
validated integrator or a proved local defect and propagation bound.

The optional [validated-flow layer](continuous-safety.md) supplies that continuous
ODE enclosure independently of the optimizer and dense RK4 replay. It does not
retroactively validate an ordinary run that omitted the guard, or prove that the
analytical relative model includes every environmental force.

## 7. Geometry, whole-tube accuracy and terminal cost

At a prediction node, the nominal position is $z_p$. The position error box has
half-width $\Omega$. Its support in direction $n_W$ is exactly
$h_{\rm box}(n_W)=|n_W|^T\Omega$. The nonnegative geometry reserve is $\sigma$.
A sufficient tube-cylinder separating constraint is

$$
n_B^T\mathcal R^Tz_p\ge h_K(n_B)+|\mathcal Rn_B|^T\Omega+\sigma,
\qquad \|n_B\|_2=1.
\tag{14}
$$

For a fixed assigned side approach, its outward body normal is fixed. For the
general side-or-cap union a separating normal is an optimization variable.
Equation (14) uses exact box support; replacing it by $\|\Omega\|_2$ would
enclose the box in a ball and add avoidable conservatism. The optimizer remains
nonconvex and local solver success is not global optimality.

For two different chasers, let their nominal centers be $z_i,z_j$, their
position radii be $r_i=\|\Omega_i\|_2,r_j=\|\Omega_j\|_2$, and their required
separation be $d_{ij}$. The implemented sufficient node condition is
$\|z_{pi}-z_{pj}\|_2\ge d_{ij}+r_i+r_j$. Fleet plans are sequential and
prioritized, then jointly checked against both new tubes; they are not one
globally optimized fleet NLP. Node and dense endpoint checks alone do not exclude
collisions between samples. The optional validated-flow execution guard instead
checks an enclosure of every complete held interval, including finite bus
volumes and target rotation; see [continuous-motion safety](continuous-safety.md).

For terminal accuracy, let $a_p=z_p-p_g$ and $a_v=z_v-v_g$ be nominal goal
errors. The requested position and anomaly-velocity tolerances are
$\varepsilon_p,\varepsilon_v>0$. The exact worst-case Euclidean norm of an
axis-aligned box is attained at a corner aligned with the signs of its center:

$$
\max_{|e_p|\le\Omega}\|a_p+e_p\|_2
 =\big\||a_p|+\Omega\big\|_2,\qquad
\max_{|e_v|\le V}\|a_v+e_v\|_2
 =\big\||a_v|+V\big\|_2.
\tag{15}
$$

Require those bounds to be at most $\varepsilon_p,\varepsilon_v$. This is
tighter than adding two norms and requires the entire tube to reach the
specified pose tolerance. No reference shift asks the actual vehicle to reach
the inward edge of a displaced tube. The implemented dwell rule requires
consecutive admissible outer endpoints together with successful sampled safety
checks. It is an acceptance rule. Persistent tracking uses a separate
reference-centered terminal controller and explicit admissible-set membership;
see [terminal tracking](terminal-tracking.md) and the
[eccentric rigid-target bound construction](physical-terminal.md).

A symmetric nonzero free-flight tube centered on a solid surface necessarily
penetrates the solid. Therefore standoff must cover the relevant tube support,
geometry inflation and reserve. Zero-standoff contact requires a new hybrid
contact/capture model and an appropriate one-sided or contact admissible set.

The running quadratic cost uses the outer-step factor $h$. Let $Q,R$ be running
state/input weights, $A_d,B_d$ a frozen discrete model, and $P$ its local Riccati
tail. The consistent construction is $P=\operatorname{DARE}(A_d,B_d,hQ,hR)$.
DARE homogeneity gives $P=h\operatorname{DARE}(A_d,B_d,Q,R)$. Omitting $h$
overweights the tail by $1/h$, about 63.662 at the paper profile's step and
1000 at the Envisat profile's 0.001 rad step. A stronger tail can be intentional
tuning, but should be an explicit multiplier. This local
penalty supplies no terminal invariance theorem for the moving nonlinear task.

## 8. What the present formulation establishes

| Claim | Status and required scope |
|---|---|
| Continuous sliding-tube inclusion with continuously evaluated feedback | Conditional on exact model, initial containment and valid uncertainty/forcing bounds. |
| Held-feedback inclusion using equations (10)–(13) | Conditional continuous-ODE comparison with declared stage peaks, maximum hold, reserve and model bounds. |
| Contractor preserves consistent parameters | Conditional on its specified held-input interval RK4 map, measurement boxes and enclosure arithmetic. |
| Nodewise tube/target exclusion and pair separation | Sufficient geometry constraints at checked nodes, subject to feasible numerical solution. |
| Whole-tube standoff tolerance at accepted endpoints | Direct component-box bound, conditional on error containment; consecutive endpoints are explicitly counted. |
| Fixed-deadline whole-tube arrival | Accepted joint plans anchor an immutable absolute deadline and required reference-centered suffix region. Actual endpoint gates remain separate; this does not guarantee that a local NLP succeeds. |
| Numerical mission completion | Evidence for each saved run and its exact source/configuration; not a universal theorem. |
| Continuous held-input orbital flow and finite-body collision exclusion | Implemented Arb Picard enclosures and swept separating geometry. Conditional on the declared initial/model/input/clock/target-pose bounds; unresolved holds refuse execution. An RK4 trajectory is not substituted for this certificate. |
| Admissible continuation on performance-solver failure | Implemented stored backup-chain supervisor. Admission requires swept safety and a successor enclosure inside the next certified entry set. Request/flow/geometry identities are bound together; stale or incomplete certificates refuse admission. |
| Robust reference-centered terminal tracking | Exact rational sufficient conditions establish invariance over their declared scope. Eccentric-orbit and rigid-target coefficient bounds are derived continuously on a finite anomaly domain. Entry membership is mandatory; ordinary endpoint dwell does not imply it. |
| Recursive feasibility of every nonlinear performance optimization | Not claimed. The continuation supervisor preserves an already certified backup; it does not prove that every GEKKO problem has a feasible solution or that the ordinary approach has such a backup. |
| Preserved eight-agent joint tail and physical-terminal successor | A source-bound saved candidate passed a separate full-prior 352-hold swept-set proof and all eight complete endpoint boxes strictly enter the declared midpoint terminal region. This finite prospective post-hoc proof does not wire an operational fallback into the ordinary mission. |
| Attachment, detumbling and orbital delivery | Implemented distinct physical stages with finite-bus/reach/attitude gates, inelastic momentum balance, external nozzle forces, propellant depletion and descending entry-event localization. Successful runs establish numerical evidence under the recorded representative assumptions. |
| Real-time eight-agent execution, sensing accuracy, physical release/contact and atmospheric demise | Require separate engineering or experimental evidence. Offline simulation and interval arithmetic do not establish these properties. |

### How the proof and execution boundaries fit together

The validated-flow layer encloses the **continuous orbital ODE** directly,
using the original declared parameter prior independently of a contracted RK4
membership box. Its endpoint and swept boxes cover the complete actual held
command. The target adapter must supply a valid rotation/operator-error and
motion bound; a numerical quaternion integrator alone does not supply that proof.

The continuation layer is deliberately stricter than a collision check. Before
admitting a performance command, it proves both the current swept interval and
membership of its successor in a stored admissible continuation. A rejected,
malformed or failed performance solve therefore selects the previously certified
backup. This conditional recursive safety argument applies only after a valid
backup chain has been admitted. `CertifiedExecutionGuard` on an ordinary approach
does not manufacture that chain: a solver failure there preserves the last
committed evidence and reports failure.

The terminal layer measures error relative to the physical reference, rather
than the last MPC shooting center. Its finite-domain construction bounds orbital
coefficients and their derivatives with Arb and uses rotation-invariant port
distances. Cylinder terminal bounds retain the scalar first-exit spin comparison.
Composite terminal bounds select the tighter of an energy comparison and an
independently validated scalar first-exit bound; the composite approach rotation
rate uses the energy and orbital-clock certificate. It checks
held-feedback forcing, input authority, whole-box pose tolerance and finite-body
geometry together. Membership of the real endpoint enclosure is checked against
the declared reference-evaluation box; the implementation never projects a
refused endpoint into the set.

The componentwise composite construction additionally uses full-inertia
rigid-body port jets and a typed midpoint feedforward law. Its finite-window
certificate bounds actual hold widths, query-clock/evaluation error and
held forcing without replacing the true reference derivative by a smaller one.
See [physical terminal bounds](physical-terminal.md),
[fixed-deadline arrival](arrival-policy.md), and
[authenticated orbital endpoints](orbital-held-flow.md). The saved joint-tail
proof transports the original spin enclosure to its successor without a future
spin measurement. It does not retroactively establish that a backup was available
when the original mission executed, or prove general recursive feasibility.

The matched empirical study uses an independently fixed prior, compact physical
deployment and the same truth/reference for adaptive, genuinely frozen-belief,
and fixed-bandwidth cases. Numerical failures and measured offline deadline
misses remain outcomes. A separate source cohort is required after a backend
repair, so an improved solver is not confused with an adaptation benefit. See
[the research review](research-review.md) and the saved source-qualified report.
