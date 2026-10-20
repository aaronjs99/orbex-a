# Attached flight

Attachment changes the plant. Free-flight ADTMPC brings each finite chaser
into its contact entry set. A finite contact and lock model must then finish
before the simulator admits a rigid assembly. Its state, uncertain geometry,
remaining propellant and carried endpoint enclosure become the initial
conditions for detumbling and disposal.

## Preserve the common environmental acceleration

For member $i$, let $m_i$ be its current mass, $p_i$ its attachment-frame
position and $M=\sum_i m_i$ the total mass. The assembly center is
$c=\sum_i m_i p_i/M$ and $r_i=p_i-c$ is the member's offset from it. Let
$R$ map assembly axes to inertial axes, and let $a_i$ be that member's
noncentral environmental acceleration at its own position, velocity and fuel.
Choose a virtual reference acceleration $a_0$ using the same typed
environment and uncertain parameters. Then

$$
\sum_i m_i r_i=0,\qquad
\frac{1}{M}\sum_i m_i a_i
=a_0+\frac{1}{M}\sum_i m_i(a_i-a_0),
$$

$$
\tau_{env}
=\sum_i r_i\times m_iR^Ta_i
=\sum_i r_i\times m_iR^T(a_i-a_0).
$$

These are exact identities for every physical member of the uncertain
family. Uniform acceleration changes the assembly's translation but has no
moment about its center. Evaluating the differences first prevents interval
arithmetic from assigning an independent large common acceleration to each
member's torque. Distinct drag laws and member velocities remain distinct;
the reference does not replace them. Both each member and the virtual
reference must satisfy the environment's declared domain.

The identity uses the physical unit-rotation and mass-center graph, rather
than unrelated corners of a rectangular enclosure. The maintained rigid
model freezes mass during each integration segment and applies its declared
half-depletion maps on either side. Those maps reform the center and inertia
at the new masses. This describes that split numerical model; it is not an
exact continuous propellant-slosh or flexible-body model.

## Prove the load for the command that will run

Let $u_0$ be the bounded nozzle command from the allocator and
$\gamma\in\{1,.75,.5,.25,.125,0\}$ a declared common scale. Before any
motion, the allocator tries $u_\gamma=\gamma u_0$ in order. A candidate can
run only when a whole-held-flow enclosure proves every attachment's force
and moment below the declared capacity fraction, with the same initial
uncertainty, geometry, environment, tank and hardware limits.

The proof is repeated for each scale. Smaller commands do not automatically
imply smaller total loads: environmental and residual-motion loads remain.
The zero command also needs its own proof. If every candidate refuses,
propagation refuses. Achieved wrench, residual and propellant flow are
recomputed from the selected command, including any admitted zero command.
The finite search is configurable through
`ThrusterSystem.attachment_backoff_scales`; its first and last values must
be one and zero, and intermediate values must decrease strictly.

## Finish detumbling with a whole-set dwell

Let $W_k$ enclose angular velocity throughout held interval $k$, let
$\omega_{tol}$ be the configured angular-speed limit and let $T_{quiet}$ be
the configured dwell. Interval $k$ contributes to quiet time only when

$$
\sup_{\omega\in W_k}\|\omega\|_2\le\omega_{tol}.
$$

An interval that is not wholly quiet resets quiet time; a refused
propagation stops the stage. A small numerical endpoint alone cannot finish
detumbling. Disposal starts only after consecutive proved
intervals cover $T_{quiet}$ and the assembly carries their final enclosure.
The descending entry interface likewise needs a whole-set event bracket;
an osculating perigee estimate or a point trajectory crossing is insufficient.

## Keep the orbital sweep directed

Let $P_0,V_0$ be the initial inertial position and velocity boxes, $h$ the
held duration, and $A$ a proved bound on continuous acceleration. Let $S$
bound the total variation of the split-depletion COM position jumps, and
$W$ bound angular speed across every previously proved substep and the
current trial. COM velocity jumps total at most $WS$. Double integration
then gives the componentwise cover

$$
P([0,h])\subseteq P_0+[0,h]V_0+[-d,d]^3,
\qquad d=\frac{Ah^2}{2}+S+hWS.
$$

The existing radial and velocity bootstraps must still prove that $A$ and
$W$ apply throughout the hold. With no propellant depletion, $S=0$.
Retaining the velocity direction avoids treating a tangential coast as
independent maximum-speed travel along every inertial axis. Member offsets,
rotation, attachment loads and the original environment-domain checks
remain active. An unresolved bootstrap or genuinely outside-domain state
still refuses; this cover supplies no new physical operating domain.

## Evidence boundary

Closed contact, lock, load-allocation and rigid-flow components qualify
their exact sources and supplied initial sets. They do not establish that
the 300 m mission reached those sets. Mission completion requires the
source-bound approach, actual contact and lock, detumbling dwell and entry
event in one authenticated chain. Flexible appendages, plume interaction,
physical adhesive and actuator qualification, and atmospheric demise remain
outside the maintained model.
