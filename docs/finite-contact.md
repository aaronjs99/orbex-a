# Finite compliant capture

The ordinary `CaptureModel` path admits a qualified physical approach, deploys chaser-carried positioners over finite time, integrates contact between the target and the entire fleet, and completes a finite load-bounded lock. It requires no target docking fixture. Pad compatibility, strength and actuator properties are declared study assumptions. They are not established material properties of present-day Envisat or a qualified flight gripper.

The general six-degree-of-freedom contact-force/moment scope is described by Glaese and Tobbe, [Contact dynamics math model, NASA-CR-179147 (1986)](https://ntrs.nasa.gov/citations/19870016368). The code's compliant adhesive-pad law and numerical parameters are separate representative assumptions. The target hull and buses remain rigid, and the finite-bandwidth positioners have negligible moving mass. Positioner reaction work is included explicitly.

`CaptureOptions.idealized_instantaneous=True` selects the old instantaneous model for explicit reproduction. It is false by default. A successful ordinary lifecycle requires a successful `attachment.audit.finite_contact` report and retains its actual body histories. Old instantaneous results do not become finite-contact evidence by changing a label.

## Physical model

All target and chaser translations, quaternions, body rates and remaining fuel are integrated simultaneously in physical seconds. The target responds to every contact force. The admitted approach endpoint supplies the literal starting COM positions and velocities. An explicit `CaptureTruth` supplies the inertial target boundary; the adapter does not reconstruct a different orbit when it is provided.

Each pad has an outward surface gap `g`, compression `d=max(-g,0)`, normal force `k_n*d+c_n*max(-g_dot,0)` while touching, and regularized dissipative friction opposing tangential slip. Adhesion is exactly zero before touch and preload dwell. A bond stores its actual engagement deformation/orientation reference, ramps over finite time, and contributes bounded translational and rotational springs/dampers. Tensile, shear, moment, compression, positioner reach/speed/acceleration/force/power, fuel and geometry violations refuse capture and retain the recorded prefix.

For an engaged pad, the already declared bounded force and tip-couple
measurements also constrain its carried strain through the spring/damper
equations. This same-clock intersection preserves every state compatible
with the prior and declared sensing errors; it introduces no more accurate sensor. Zero activation
supplies no inverse strain information, and contradictory measurements refuse
before motion. The principal rotation-log branch remains an explicit gate.

The prediction also carries the adhesive wrench as a redundant state. Let
\(\delta\) be displacement from the retained engagement rest pose,
\(u=\dot\delta\) its target-frame rate, \(\theta=\log Q\) the relative
rotation vector and \(\sigma\) the relative angular velocity. For the declared
translation and rotation stiffness/damping coefficients, write

\[
B=k\delta+cu,\qquad M=k_r\theta+c_r\sigma,
\qquad F_{\rm adhesive}=-\alpha B,\quad C_{\rm adhesive}=-\alpha M.
\]

Here \(\alpha\) is bond activation. Bounded measured force/couple supports
intersect these same equations when activation is strictly positive. The
held prediction carries their exact derivatives,

\[
\dot B=ku+c\dot u,\qquad
\dot M=k_rJ_l^{-1}(\theta)\sigma+c_r\dot\sigma,
\]

where \(J_l^{-1}\) maps relative angular velocity to rotation-log rate on the
admitted principal branch. This retains measured spring/damper correlation
through the hold; independently rebuilding their sums from broad boxes loses
that information. Strain, uncertain rest poses, energy and all strength gates
remain active. A newly engaged stress-free spring still has damping wrench
\(B=cu\), \(M=c_r\sigma\); it is not initialized to zero wrench.

The deformation and damping use the rotating target frame. Action and reaction have consistent spatial moments, including the finite separation of compliant anchors. This prevents artificial angular momentum from tangential force. The ledger records external impulse/work, propellant momentum/energy flux, positioner work, engagement work, spring energy and compression/friction/bond dissipation. A translating nonrotating central-gravity reference keeps orbital-scale coordinates out of the short-range balance subtraction.

Before adhesion, bounded thrusters provide physical stationkeeping and attitude control using the supplied observations and measured pad loads. Their commands, saturation and propellant use are recorded. Their free-flight commands fade to zero as the bond engages; they do not continually cancel the bond's damping. A finite search accommodates uncertainty in global hull pose, using local range and force channels to approach slowly and establish preload.

`handoff_rule="load_bounded_finite_lock"` permits a handoff after every pad is
fully engaged for the declared dwell and the continuous contact, geometry,
fuel and interface-load gates pass. This establishes eligibility for the
next phase. Attachment success still requires the complete finite lock to
pass. `handoff_rule="settled_passive"` additionally requires the original
small relative-speed/spin and residual-energy gates before that handoff; its old receipts
retain that interpretation. The active rule makes no claim of passive
adhesive convergence.

The lock receives the complete carried state, tip velocity, mounting and
strain-energy bounds. Over its explicit duration \(T\), it brakes relative
translation and rotation with \(g(t)=1-t/T\) and displacement profile
\(f(t)=t-t^2/(2T)\). Write total body-frame angular momentum as
\(H=I(t)\Omega+h_{\mathrm{internal}}(t)\), where \(I\) is the moving assembly
inertia, \(\Omega\) its common spin and \(h_{\mathrm{internal}}\) the declared
relative-motion momentum. The common spin obeys

\[
\dot\Omega=I^{-1}\{\tau_{\mathrm{external}}-\dot I\Omega
 -\dot h_{\mathrm{internal}}-\Omega\times(I\Omega+h_{\mathrm{internal}})\}.
\]

The lock preserves each uncertain initial relative rotation \(A_{j0}\) as
an immutable parameter and integrates an attitude increment \(D_j\), with
\(A_j(t)=D_j(t)A_{j0}\) and \(D_j(0)=I\). The identity increment is a
coordinate choice; the physical attitude remains uncertain. Positioner
extension and its derivatives retain the same initial extension and speed
graph instead of subtracting independent orbital-position boxes. In
particular, for the initial reference-frame lever \(\ell_j\), relative spin
\(\sigma_j\) and initial extension rate \(\dot x_{j0}\), the displacement
profile derivative has the exact initial identity
\(d_f(0)=A_{j0}\dot x_{j0}+\sigma_j\times\ell_j\).
The incremental derivative subtracts this baseline before interval
evaluation. Dropping either term would change the mechanism dynamics.
The carried scalar pad gap intersects its same-event geometric cover;
neither attitude, gap, extension nor twist is reacquired from truth.

Acquired contact history must retain its complete last proof and chain
identity, exact accepted duration, original physical clock origin and equal
target/chaser/lock boundary clocks. A supplied initial-graph fixture requires
an explicit marker and cannot claim an acquired chain. These local value
checks accompany the independent full-history/source audit; they do not
replace it.

The complete lock is proved prospectively from its immutable initial state,
resolved configuration and source before any lock motion. The executed
integration must reproduce that same preflight. Newton–Euler reactions,
mechanical power, travel, geometry, fuel and physical load bounds gate every
held interval. A later preflight refusal therefore starts no lock motion.
The fixed-mass lock uses the exact COM identities \(\sum m_i r_i=0\) and
\(\sum m_i u_i=0\) before interval evaluation. Subtracting one common central
acceleration from every component preserves COM acceleration, reactions,
torque and work; the position-dependent gravity difference and the original
environment/domain checks remain. This avoids fictitious reaction bounds
from independently evaluating a shared orbital acceleration.
Remaining pad spring energy has a whole-set ledger, retaining the uncertain
rest poses and carried strains. The final actual poses define the rigid assembly, with
zero residual relative velocity at the modeled lock endpoint. No planned
patch or nominal trajectory replaces those poses. This is an ideal internal
encoder/servo mechanism model; actual mechanism tracking and material
qualification are separate obligations.

For an actual tip force \(F\), tip couple \(C\), moving target-body anchor
\(a\), positioner extension \(x\), and body rotations \(R_t,R_j\), the
interface power is

\[
P=F^T(R_t\dot a-R_j\dot x)+C^T(\omega_j-\omega_t).
\]

The lock bounds this power and its absolute time integral over every held
step. The active mechanism explicitly permits bidirectional work. The former
0.1 mJ tolerance bounded energy deleted by an instantaneous velocity
projection; it is not an actuator energy capacity and does not bound the
finite active maneuver. It remains a gate for the passive policy. Finite
work and power accounting still do not qualify a real actuator's battery,
thermal capacity, servo tracking or attachment material.

## Representative configuration

`FiniteContactOptions` is injectable through `CaptureModel(contact_options=...)` and `CompleteMissionOptions.finite_contact`. Its ordinary starting values include:

| Property | Value |
| --- | --- |
| Maximum duration / internal step | 60 s / 5 ms |
| Extension speed / acceleration | 20 mm/s / 20 mm/s² |
| Touch speed / local range-error reserve | 1 mm/s / 2 mm |
| Surface search depth / pad radius | 20 mm / 25 mm |
| Normal compression stiffness / damping | 2000 N/m / 700 N s/m |
| Maximum pad compression / preload | 3 mm / 1 N |
| Friction coefficient / regularization speed | 0.5 / 1 mm/s |
| Bond translation stiffness / damping | 20000 N/m / 1000 N s/m |
| Bond rotation stiffness / damping | 500 N m/rad / 300 N m s/rad |
| Bond tensile / shear / moment capacity | 250 N / 250 N / 5 N m |
| Preload / engagement / settling dwell | 0.5 s / 1 s / 2 s |
| Final relative speed / spin | 0.1 mm/s / 10 microrad/s |
| Passive residual lock kinetic-energy tolerance | 0.1 mJ |
| Finite lock duration | 0.25 s |

The standoff, gripper offset, arm reach, remaining mass and actuator limits still come from the vehicle/mission configuration. A 0.85 m COM standoff with a 0.8 m gripper offset begins with a 50 mm tip gap. These parameters must be sized against the actual endpoint enclosure, pose error, relative speed and subsequent loads. Numerical successful contact does not establish hardware strength.

## Observation and environment callbacks

`capture(..., contact_feedback=callback)` calls `callback(time_s, snapshot)`. The snapshot has `target: ContactBodyState`, `chasers: {id: ContactBodyState}`, `contact_gaps_m: {id: outward_gap}`, and `contact_wrenches_eci: {id: {force_n, couple_nm}}`. The callback supplies estimated/measured versions of those channels. It changes commands, never physical plant state. Body estimates must name the exact fleet and current physical clock. With no callback, exact feedback is explicitly recorded as an assumption. Omitted injected wrench channels mean zero measured force; omitted range channels require slow search over the declared surface uncertainty.

`capture(..., contact_environment=callback)` and `CompleteMissionRunner.complete(..., contact_environment=callback)` use `callback(time_s, bodies_by_name)`, where names are flat `target` plus every actual chaser id. It must return every body's `acceleration_eci_m_s2` beyond central gravity and optional intrinsic `torque_body_nm` beyond configured gravity-gradient torque. Body torques use each body's own axes. The callback retains the caller's physical ECI basis, including an explicitly inclined J2 axis if applicable. It survives into `AttachedAssembly`, controlled detumbling/burn RHS calls, deorbit coasts and joint-load reconstruction, so J2/vector-drag need not switch off at capture.

The separate saved-trajectory attitude replay accepts `observation_callback` on `ChaserAttitudeTracker.track`; `CompleteMissionRunner.complete(attitude_observations=...)` forwards it. It supplies sensed target and bus attitudes/rates to held feedback, records the observations, and keeps physical replay integration/final gates distinct. It remains a decoupled rotational replay with its documented inertia/depletion scope.

## Nonlinear mission continuation

`simulation.nonlinear_completion.complete_nonlinear_mission(approach, nonlinear_runner)` composes the ordinary post-approach stages from the live runner's executed truth, original commanded clock, declared vehicle and full admitted hold journal. Saved inputs instead require explicit typed `model`, `clock`, `design`, and full `hold_records`; an optional explicit `CaptureTruth` must equal the retained actual boundary. Incomplete approaches stay incomplete. The adapter uses the declared 40 mm whole-position profile, 10 mm/s relative speed, 10 mrad orientation and 1 mrad/s spin admission limits; it retains the configured 100 mm mechanism reach and the stricter physical contact settling gates.

`NonlinearTranslationAccountant` checks the source result identity, the complete exact contiguous hold clock, every complete decision identity and commit, source actuator/sensor IDs, all component and peer geometry coverage, delivered-input containment and actual saved endpoint alignment. It rebuilds conservative fuel allowances from each saved applied-input box and the original clock's outward physical duration. It does not infer commands from outer means, use the analytical CH journal, or re-execute the independent interval witnesses. Those full witnesses remain producer evidence. The compact `fleet_hold_records` omit geometry certificates; retain the full `executor.journal` or the producer certificate archive as well. Serialize with numeric-aware utilities such as `mission_checkpoint.plain`; converting NumPy arrays with `default=str` loses the physical boundary.

`NonlinearEnvironment` evaluates the same declared point central/J2/co-rotating-vector-drag law and supplies only its acceleration beyond central gravity. The original inclined physical axes remain unchanged. Default CdA/m stays constant as declared in the approach; optional explicit CdA values divide by each component's actual dry plus fuel mass. Its numerical radius domain extends to an 80 km altitude event bracket, with unchanged force parameters, while the mission stops at the descending 120 km interface. This extension does not establish atmospheric disposal physics.

`PastNavigationAttitudeObserver` uses the latest past `VALID` navigation record whose physical-time upper endpoint is no later than the replay query. Before delivered packets it uses and counts the declared initial navigation prior. It propagates the observed pose/rate center with rigid-body/GG dynamics and the public physical bounds' midpoint model, independently of the selected point truth, never interpolating future target truth. It adds bounded own-bus errors of 0.3 mrad and 10 microrad/s. The finite-contact sensor realization supplies a shared GNSS position bias below 1 m, individual camera position bias below 3 mm, 1 mrad target pose and 0.1 mrad/s target-rate errors, a 1 mm local range bias, and explicit pad wrench biases. These are synthetic processed-product assumptions. Actual states still determine every physical gate. The completed result exports their contracts and environment continuity in `completion_audit`.

## Evidence limits

All occupied target components, spherical bus envelopes, finite pad/positioner
segments and peer buses remain active during contact. `ContactStepEncloser`
now gates each held step using outward Arb interval Picard inclusion for the
declared coupled contact ODE. It retains unilateral contact branches,
regularized friction, growing bonds, fuel and positioner states. Whole-step
component, bus, arm and pad separation and physical load/travel limits must
pass before numerical propagation. Exact rational clocks preserve adjacency,
engagement and settling dwell. The carried endpoint set also covers the
numerical endpoint; sampled guards remain additional diagnostics.

After lock, the allocator checks joint loads **before** each controlled step.
At fixed assembly state $z$, each joint wrench is affine in the signed nozzle
commands $c$: $w_i(c)=b_i(z)+B_i(z)c$. The bias retains the actual environment,
gravity gradient, angular velocity and component inertia. Tensile/compressive
half-space limits and shear/moment norm limits define a convex strength set.
After the ordinary hardware-bounded allocation and fuel scaling, the allocator
tests the zero-command and candidate loads. If zero is within the declared
0.98 utilization reserve, bisection finds a common scale $0\le\gamma\le1$
with every $w_i(\gamma c)$ within that reserve. It then independently recomputes
the full actual loads, applied wrench and fuel rates. An unsafe zero-command
state, unavailable reserve or failed final check refuses propagation. A refused
ray is not a proof that every other command distribution is infeasible.

The unchanged physical capacities, including the 5 N m moment limit, remain
the acceptance thresholds. The 0.98 fraction is a stricter command-selection
reserve at the current state. The selected command additionally passes
`AttachmentLoadEncloser`: whole-step fuel, COM/inertia, angular and pose
bounds enclose joint loads throughout that hold. The typed nonlinear
environment retains inclined J2, vector drag and co-rotation; unsupported
callbacks or altered model declarations refuse. The pose envelope includes
the complete held angular-rate bound, including an interior spin peak.
Actual endpoints, controlled histories and recorded zero-thrust coast states
remain numerical diagnostics.

`TerminalContactBoundary` intersects the authenticated navigation and terminal
tube sets in the declared frame and transforms them at the original clock.
The contact chart stores the full absolute target ephemeris and genuinely
target-relative chaser translations. The exact target-local translation is a
coordinate zero. Target acceleration appears in the absolute ephemeris and
is subtracted from each relative acceleration; local momentum/work ledgers
include the accelerating-origin terms. Shared target-position error is
retained once.

Redundant rotation-matrix and pad-gap states preserve observed dependencies:
\(\dot R=R[\omega]_\times\) and the exact geometric gap derivative accompany
the quaternion/body equations. Supported current measurements intersect the
carried set; they never reset it to the physical point. The fixed-normal
tangent projector retains exact normal cancellation, and friction is bounded
by its analytic \(\|F_f\|\le\mu F_n\) cone. Finite proof refinements and
numerical integration substeps remain separate from the held sensor/control
interval.

`AssemblyEnclosure` carries the post-lock state, uncertain geometry and
inertia through controlled and zero-thrust intervals. Bounded chaser-carried
gyros/star trackers condition the common assembly state; the target supplies
no receiver. Multiple deterministic sensor supports are intersected without
assuming independent noise or averaging away their biases. Quiet dwell and
deorbit rate gates concern the complete carried state/flow.

A successful robust entry handoff needs a contiguous coast bracket whose
entire initial position set is above the interface and whose entire final
set is at or below it. Every intervening swept set must satisfy
\(r^Tv<0\). Powered split-depletion steps invalidate a continuous bracket
because their COM coordinates can jump; a fresh zero-depletion coast must
establish it. A point root from the numerical integrator is insufficient.

These implementations and local qualifications remain conditional on their
supplied initial sets and declared rigid/contact/mechanism models. An
uninterrupted source-bound full mission and its independent archive audit
remain necessary; component fixtures cannot replace them.

The model does not resolve arm moving mass, flexible appendages, adhesive microstructure, thermal/material compatibility, plume interaction, manufactured mechanism qualification or atmospheric demise. Exact-state reproduction, sensor-biased numerical boundaries, full mission execution and hardware evidence remain distinct.
