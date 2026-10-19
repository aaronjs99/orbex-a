# Nonlinear fleet mission

The maintained mission separates physical truth, measurements, navigation,
ADTMPC, continuous admission and finite contact. Its scope is a declared
simulation study. Every accepted run must retain its exact source, profile,
truth case, observations, failed attempts and applied commands.

## Coordinates and physical plant

Let $C(q)$ map the commanded orbit frame into fixed inertial axes,
$C'=CZ$, $Zp=(-p_y,p_x,0)$ and $\nu(q)=dq/dt>0$. The scheduling clock is
fixed; parameter adaptation never changes it. Define

$$
p=C^T(r_c-r_t),\qquad w=p',\qquad
v_c-v_t=\nu C(w+Zp).
$$

Both spacecraft propagate Cartesian central gravity, inclined-axis J2 and
vector drag in a co-rotating exponential atmosphere:

$$
a(r,v)=-\frac{\mu r}{\|r\|^3}
+\frac{3J_2\mu R_E^2}{2\|r\|^5}
 \left[\left(5\frac{(k^Tr)^2}{\|r\|^2}-1\right)r-2(k^Tr)k\right]
-\frac{\rho(r)\sigma}{2}\|v_{rel}\|v_{rel},
\quad v_{rel}=v-\omega_E\times r-v_{wind}.
$$

Here $k$ is the declared unit Earth axis and $\sigma=C_DA/m$. These parameters,
the atmosphere and the initial Cartesian state have explicit finite bounds.
With $\kappa=\nu'/\nu$, the exact relative-coordinate equations are

$$
p'=w,\qquad
w'=\frac{C^T(a_c-a_t)}{\nu^2}-2Zw-Z^2p
-\kappa(w+Zp)+u.
$$

The command is held in these coordinates: $a_{thr}=\nu^2Cu$.
The target quaternion and Euler rigid-body equations use physical seconds and
central gravity-gradient torque. The coordinates are **reference orbit axes**,
not the perturbed target LVLH. The model includes neither flexible wings nor
unmodeled aerodynamic, magnetic or solar-pressure torques.

## What ADTMPC controls

GEKKO/IPOPT optimizes the Carter–Humi nominal surrogate, its dynamic tube and
feedback bandwidth together. An Arb interval calculation bounds the complete
nonlinear-minus-surrogate residual over the prospective regional domain.
The optimizer is restricted to that domain; the executed nonlinear sweep must
also remain inside it. A pointwise sampled model discrepancy is insufficient.

For $e=x-\bar x$, $s=e_v+\Lambda e_p$, the tube is
$|e_p|\le\Omega$, $|s|\le\Phi$. The implemented feedback uses the navigation
estimate and its bounded error. Its [input reserve and forcing](sensed-tube.md)
include sensing and the maximum feedback hold. The comparison dynamics are

$$
\Phi'=-\alpha\Phi+F_{total},\qquad
\Omega'=-\Lambda\Omega+\Phi.
$$

The numerical nominal endpoint has a separate Picard/variation-of-constants
certificate. Its full endpoint radius is transferred into $\Phi,\Omega$.
Replacing it by an unbounded integration reset would break the argument.

Navigation stores a rectangular enclosure, while the sliding tube relates
position and velocity. After a proved performance hold, the physical state
belongs to both sets; unrelated corners of the navigation rectangle need not
belong to the sliding tube. An authenticated, exact-clock tube witness carries
the previous comparison bound and nominal endpoint transfer into the next hold.
Sensing support, the new nonlinear residual region and input reserve are still
checked. Initial or reset tubes require whole-navigation-box containment.
The executor independently reevaluates the nonnegative forcing algebra with
exact rationals and advances the scalar comparison with outward Arb arithmetic.
Changed or missing witness identities cannot bypass the initial membership gate.

The prospective NLP also reserves each numerical nominal-center transfer.
For a declared dense hold $h$, endpoint budgets $\epsilon_p,\epsilon_v$ and
$\epsilon_s=\epsilon_v+\Lambda\epsilon_p$, its prediction adds

$$
F_{num}=\frac{\alpha\epsilon_s}{1-e^{-\alpha h}},\qquad
G_{num}=\frac{\Lambda\epsilon_p}{1-e^{-\Lambda h}}
$$

to the $\Phi$ and $\Omega$ comparison forcing, respectively. Each constant
forcing integrates to the corresponding endpoint jump over one hold; positivity
then bounds their repeated accumulation. The zero-rate limits are
$\epsilon_s/h$ and $\epsilon_p/h$. These are numerical reset allowances, not
physical disturbances or extra thrust. Physical forcing and actuator reserve
use the resulting larger stage peaks. Execution checks the actual certified
nominal defect against the budgets, the exact hold schedule and strict peaks;
a refused transfer selects the admitted feedback. The current profile declares
1 nm position and $10^{-8}$ m/rad velocity budgets per nominal transfer.

The nonlinear profile also declares prospective peak padding of $10^{-6}$
m/rad for $\Phi$ and $10^{-7}$ m for $\Omega$. Both endpoint-to-peak
inequalities in the NLP include this padding. The enlarged peaks enter the
same forcing, input-reserve and geometry constraints; strict runtime checks
remain unchanged. This keeps native feasibility and decimal-output errors
from turning an active equality into a smaller-than-required physical peak.
Absent or zero padding retains the analytical reproduction formulation.

Independent per-chaser NLP candidates use the preceding shared fleet preview.
Every peer remains constrained. A simultaneous planned-tube check and the
independent whole-fleet admission service decide whether the candidate can run.
Solver setup errors, numerical failures and rejected candidates remain in the
record. An optimizer failure is not a proof of physical infeasibility.

The declared component corridor chooses its support planes around a nominal
RK4 rollout of the guessed held controls from the current state. It does not
choose them around the moving goal: a goal-side plane can exclude the fixed
initial state. A supplied warm state plan is validated but projected from the
current state, since a previously optimized plan may have been rejected in
favor of fallback. This corridor still restricts the free-normal feasible set;
seed separation does not prove horizon feasibility or continuous safety. All
component supports and independent whole-hold checks remain active.

## Sensing and adaptation

Only the sensor generator receives truth. The navigator consumes priors,
timestamped range/bearing, target pose/rate, own-chaser GNSS and proved input
flows. Delayed packets are replayed through their actual acquisition history.
Velocity follows position histories and enclosed acceleration, rather than a
hidden true-velocity measurement. The processed products and error supports
are [study declarations](nonlinear-navigation.md), not calibrated hardware.

Initial GNSS/relative observations contract initial **osculating** eccentricity.
It is not a constant orbital element under J2. Nonlinear set membership acts on
physical $\sigma_t,\sigma_c$ candidates. Only proved disjointness discards a
candidate; unresolved hypotheses survive. Low drag observability may leave the
drag box unchanged, and the experiment must report that result.

The maintained linked estimator keeps a bounded history per agent. With a
fixed affine actuator matrix $M_i$, bias $b_i$, declared command $u_k$ and
bounded evaluation remainder $r_k$, it uses
$u_k^{applied}=M_i u_k+b_i+r_k$. Every observation in a surviving branch must
share the same $(M_i,b_i)$. Candidate-conditioned target, physical-time and
relative-state endpoints intersect the next contiguous observation. Covering
splits retain all unresolved or unevaluated nuisance cells; exclusion requires
every covering branch to be proved inconsistent. Gaps reset state links,
without inventing a new actuator. Bounded histories and proof budgets bound
work; they do not guarantee informative data or a timely contraction.

Raw measured packets and their actual acquisition events are archived before
consumption. Delivery attempts/results are separate events. Their source hashes,
physical-time slots, channel identities and eligibility can therefore be audited
without mistaking a later acceptance timestamp for acquisition.

## Moving goal, moving obstacle, finite contact

One observed target pose generates both a moving COM goal and the occupied
component union. Goal tracking attracts the trajectory. Obstacle exclusion
keeps every finite bus and peer separated throughout an admitted hold. These
are different constraints despite their common pose prediction.

A nonzero uncertainty tube cannot generally guarantee that every state touches
one sharp surface boundary. The COM instead tracks an exterior reference. The
whole terminal tube, including moving-goal pose/rate error, must fit the declared
position and velocity tolerances for consecutive endpoints. A fixed arrival
deadline must be anchored before an endpoint can start entry or dwell. Once
anchored, it cannot silently recede or switch off after a retreat.
Whole-set entry must occur by that exact deadline. Every checked endpoint at
or after the deadline must remain admissible until the five-endpoint dwell
finishes; a late first entry or a lost dwell fails the mission. The saved
exact-clock history is checked again before contact can start.
The inertial error screen $\nu_u(P+V)\le0.01$ is retained, and a shared
rotating-tip speed certificate also connects planning to capture. This extra
check closes a handoff gap: the capture gate adds separate nominal and tube norms,
the hull's rotational motion and a gripper spin-error reserve. A norm of
componentwise sums does not generally dominate those separate norms.

Let $\hat g=(\hat g_p,\hat g_v)$ be the same-event numerical goal,
$N_p=\|\bar p-\hat g_p\|$, $N_v=\|\bar v-\hat g_v\|$,
$T_p=\|\Omega\|$ and $T_v=\|\Phi+|\Lambda|\Omega\|$.
The positive clock bound is $\nu_u$ and the target body-rate box gives
$W\ge\|\omega_t\|$. Here $\omega_t$ is expressed in the reference axes;
proper physical rotations preserve the body-rate norm.
For declared gripper length $\ell$ and admitted chaser spin mismatch
$\delta\omega_{max}$, define

$$
S_0=\nu_u(N_v+T_v)+(\nu_u+W)(N_p+T_p)
    +\delta\omega_{max}\ell,
\qquad
Q_{split}=\nu_u E_v+(\nu_u+W)E_p,
$$

where $E_p,E_v$ bound the observed goal error. Then
$S_0+Q_{split}$ is sufficient for the unchanged capture speed gate. It
explicitly retains all four nominal/tube norms.

Rigid-goal kinematics supplies a second bound on the same goal residual.
Let $\hat R$ map body axes into reference axes and let $\hat\omega_b$ be the
numerical preview body rate. Let
$\|R-\hat R\|_2\le\gamma$, and let
$\Delta\omega_b\ge\|\omega_b-\hat\omega_b\|$ follow from the declared
body-rate box. With the original point-clock interval, compute

$$
\epsilon_\omega=\gamma\|\hat\omega_b\|+\Delta\omega_b,
\qquad
D=\sup_\nu\left\|
\nu(\hat g_v+Z\hat g_p)
 -(\hat R\hat\omega_b)\times\hat g_p\right\|,
\qquad
Q_{kin}=\epsilon_\omega\|\hat g_p\|+D.
$$

The exact physical goal satisfies
$\nu(g_v+Zg_p)-\omega_t\times g_p=0$.
The residual $D$ keeps the actual rounding defects of the numerical preview;
it does not assume that the rounded matrix is exactly orthogonal. Both
$Q_{split}$ and $Q_{kin}$ bound the same rigid-relative goal velocity, so their
minimum is valid when their clock, goal and source bindings agree. Admission
requires

$$
S_0+\min(Q_{split},Q_{kin})\le0.01\;\mathrm{m/s}.
$$

The nominal endpoint constraint uses four nonnegative norm epigraphs at each
required terminal node. Its constants come from that node's validated attitude
forecast. Actual endpoint admission and saved-arrival verification use the
same certificate with the accepted same-event navigation support. Forecast
uncertainty is not replaced by an assumed future valid measurement. These
speed conditions do not establish contact settling, bond strength or terminal
invariance; the subsequent physical gates remain necessary.

Reach uses the same separate position norms. Let $d$ be the assigned COM
standoff, $L$ the positioner reach and $\theta_{max}$ the admitted mounting
orientation error. With $P_{split}=N_p+T_p+E_p$, sufficient conditions are

$$
|d-\ell|+\ell\theta_{max}+P_{split}\le L,
\qquad d-\ell-P_{split}\ge0.
$$

The first inequality bounds the nominal gripper gap, orientation displacement
and entire position enclosure. The second keeps the entire initial tip outside
the surface: a proper rotation cannot increase its inward normal projection
beyond $\ell$. The existing norm epigraphs also enforce these conditions in
the MPC, and the endpoint/verifier recompute the same source-bound reach
certificate. With $d=0.85$ m, $\ell=0.8$ m, $L=0.1$ m and
$\theta_{max}=0.01$ rad, the split position allowance is 0.042 m. The original
40 mm whole-position gate still applies. Patch edges, other components, bus
clearance and physical contact settling retain their separate required gates.

![Contact speed budget across the forecast and sensing supports](https://orbex-a-mission-studio.aaronjs99.chatgpt.site/theory/figures/contact-speed-budget.svg)

The dotted line is the unchanged 10 mm/s capture limit. This saved-forecast
algebraic screen illustrates the effect of preserving rigid-goal correlation;
some forecast nodes still exceed the limit even with zero nominal tracking
error. It is not a mission or optimizer result.
[Figure data and provenance](https://orbex-a-mission-studio.aaronjs99.chatgpt.site/theory/figures/contact-speed-budget-data.json).

The 0.85 m COM standoff, 0.8 m tip offset and 0.1 m mechanism reach leave a
finite closing gap. The actual nine-body [contact simulation](finite-contact.md)
extends the mechanism, resolves compliant touch and finite adhesion, checks
load/reach/settling gates, and preserves actual body poses. Detumbling and
finite-burn orbital lowering use the resulting assembly and remaining tanks.
A descending 120 km interface is a disposal handoff, not an atmospheric-demise
or regulatory-compliance result.

## Safe continuation and evidence

An admitted [fleet refuge](fleet-fallback.md) is installed before motion.
Performance requires a continuous nonlinear flow, all component/peer sweeps,
entry into an admitted successor, and fuel for the complete finite backup.
If the current stored suffix cannot contain the candidate endpoint, a separate
builder may construct a new complete backup from the full nonlinear endpoint
enclosures and proved target/time endpoint. This is a prospective set, not a
future measurement. Close-range attitude prediction starts at the actual
current observed pose and spans the entire new suffix. Runtime admission
independently rechecks endpoint entry, source identities and remaining fuel;
only committing the admitted hold installs that successor. A refused builder
leaves the existing feedback available.
Failure selects the stored feedback policy while physical time advances.
Braking references account for current momentum; close-range candidates use
bounded target attitude and every appendage. A refused route stays refused.
Finite fuel and sensing-blackout duration remain explicit limits.

Full admission witnesses are durably archived before numerical motion. Applied
completion or assumption-violation events follow with exact clocks. Lightweight
result indexes bind those immutable records. A failed sensor/model assumption
cannot become a successful endpoint or a silently paused plant.

The journal uses lossless long-window Zstd compression with an explicit
128 MiB decoder-window cap. It preserves the complete ordered JSON payloads
and semantic identities; compression supplies no new mathematical guarantee.
Acknowledgement follows event flush and file fsync. A closed reader additionally
requires one completed checksummed frame with no trailing bytes. Interrupted
prefixes remain aborted evidence rather than complete proof archives.

### Accepted-hold induction

Let $\mathcal N_k$ be the current navigation enclosure,
$\mathcal T_k$ the carried physical sliding tube and $\mathcal B_k$ the
installed finite backup entry. Start with whole-set entry into $\mathcal B_0$.
The first performance tube, and every reset after fallback, must contain the
whole navigation box. Later performance holds may carry authenticated physical
tube membership; fallback clears that witness. Admission requires

$$
x_k\in\mathcal N_k\cap\mathcal T_k
\ \Longrightarrow\
x(q)\in\mathcal X_k(q),\quad
\mathcal X_k(q_{k+1})\subseteq\mathcal B_{k+1}.
$$

Here $\mathcal X_k$ is the independently enclosed fleet state sweep. On each
common proof window, certified lower separations must exceed every target
component clearance and the larger of physical body separation and the
configured peer center threshold. Same-time relative-reference bounds may
prove peer separation even when separate swept AABBs overlap; separation of
entire hold unions is not required.
The sensed comparison, actual command-evaluation bound and certified nominal
transfer also carry $\mathcal T_{k+1}$. Before motion, the candidate must satisfy
hardware and physical force limits, and its hold fuel charge plus complete
successor reserve must fit the current fuel. A refused performance transaction executes the
previously admitted feedback, whose invariant tube and geometry cover that
hold. Induction therefore covers a finite committed prefix, conditional on
the declared plant, sensing, actuation and clock bounds. This closes the
node-to-sweep and failure-continuation obligations for those accepted holds.
It does not imply indefinite holding after the finite backup ends, feasibility
of every future NLP, or a contact invariant. Finite contact has its separate
declared dynamics and whole-step local contact/load gates, with unresolved
uncertainty transfers and rigid-lock impulse certification described in
[finite contact](finite-contact.md).

The matched experiment is adaptive/frozen belief crossed with dynamic/fixed
bandwidth, on independently selected truth/noise cases. Each pair uses the same
prior, vehicle, cost, reference, sensor profile, safety rules and resource cap.
All arms start at 1000/rad; fixed arms hold that value, while dynamic arms may
use 160–1200/rad. This common baseline was selected before the independent
cases from a development terminal bound screen, including a prospective
2.5 mm position / 0.5 m/rad velocity sensing budget. The screen left about
0.41 m/rad of whole-velocity margin at 1000/rad; 800/rad left only 0.01 m/rad.
That local sufficient bound does not guarantee future navigation support or
global feasibility: larger observed sets can still force a certified retreat.

![Terminal velocity bound versus bandwidth, for three position sensing supports](https://orbex-a-mission-studio.aaronjs99.chatgpt.site/theory/figures/terminal-budget.svg)

The recorded development screen includes target-goal prediction error and the
same input limits. At 3.5 mm position support, none of the three screened
bandwidths satisfies the velocity gate. These are local bound calculations,
not mission outcomes. [Published figure data and provenance](https://orbex-a-mission-studio.aaronjs99.chatgpt.site/theory/figures/terminal-budget-data.json).

Completion, failures, tube widths, effort, retained membership, force/fuel and
deadline misses all belong in the results. Offline proof/solve time is recorded;
it does not establish real-time eight-agent execution.

Dynamic tube optimization is established by
[Lopez, Slotine and How (2019)](https://arxiv.org/abs/1907.06553).
Adaptive tube feasibility requires additional estimator/terminal conditions;
see [Köhler et al.](https://arxiv.org/abs/1911.02899).
Moving-reference interception with adaptive terminal sets also has finite-time
results for linear discrete-time bounded-disturbance systems:
[Quartullo et al.](https://arxiv.org/html/2410.08807v1).
Those assumptions do not establish nonlinear fleet contact or disposal.
Tumbling Envisat tube MPC also predates this project:
[Buckner and Lampariello (2018)](https://elib.dlr.de/125251/).
Finite contact dynamics have a long modeling history, including
[NASA-CR-179147](https://ntrs.nasa.gov/citations/19870016368).
This implementation does not inherit those works' theorems merely by citing them.
