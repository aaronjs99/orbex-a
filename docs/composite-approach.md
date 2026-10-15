# Component-aware spacecraft approach

The Envisat study uses the union of the declared bus, solar wing, yoke, antenna,
reflector and payload envelopes. The same immutable geometry fingerprint binds
the target, assignment, nonlinear node constraints and continuous safety guard.
An enclosing sphere may set a camera or automatic release scale; it is not the
collision model. Dimensions, inferred placements, uncertainty and source scope
remain in `config/spacecraft/envisat.json`. These are study geometry declarations,
not flight CAD, current pose measurements or qualified structural interfaces.

## Nonlinear node constraints

Let $p$ be the nominal chaser center in LVLH, $c_0$ the target center, and $R$ the
declared target-body-to-LVLH matrix at a prediction node. A component has body
center $c$, component-to-body matrix $C$, positive half extents $h$, and an
isotropic geometry/placement uncertainty radius $u$. Its occupied set is
$c_0+R(c+C[-h,h]+B(u))$, where $B(u)$ is a Euclidean ball. The chaser's
position error box has nonnegative half widths $\Omega$; its finite body is
enclosed by a world-frame ball of radius $r$. Let $d$ be the sum of requested
physical clearance and declared pose reserves. For each component and node,
a unit LVLH normal $n$ must satisfy

$$
\lVert n\rVert_2^2=1,\qquad
n^\mathsf{T}(p-c_0-Rc)\geq
h^\mathsf{T}|C^\mathsf{T}R^\mathsf{T}n|
+u\lVert R^\mathsf{T}n\rVert_2
+\Omega^\mathsf{T}|n|+r+d.
$$

The right side contains the support of the complete component and the complete
chaser position box. Geometry uncertainty is included once. Using a unit world
normal also preserves the finite-body distance meaning when a supplied
rotation matrix has a small permitted numerical departure from orthogonality.
The component is treated as its declared affine image. A rotation-operator
error reserve uses the maximum norm of its nominal body-frame vertices plus
`u`; translation uncertainty is added separately.

Each component gets its own normal. This represents exclusion of every member
of the union and remains a nonconvex nonlinear GEKKO/IPOPT problem. Absolute
values use nonnegative epigraphs, without complementarity or binary switches.
A declared attachment face can add a fixed outward support plane during
docking. It supplements all union constraints. Normalization, original solver
residuals and component support margins are rechecked numerically after solving.
These are node checks; continuous execution still requires the independent
composite swept-flow admission.

## Optional reference support corridor

The solver setting `composite_geometry_mode` defaults to `free`. Explicitly
selecting `reference_corridor` prescribes a unit normal $\bar n$ for each
component and node from the same reference-position witness used to initialize
the free formulation. With these normals fixed, the node inequality becomes

$$
\bar n^\mathsf{T}(p-c_0-Rc)\geq
h^\mathsf{T}|C^\mathsf{T}R^\mathsf{T}\bar n|
+u\lVert R^\mathsf{T}\bar n\rVert_2
+\Omega^\mathsf{T}|\bar n|+r+d.
$$

The component and pose support terms are numerical constants; the position-box
support still depends on the optimized tube. Every component remains active,
including solar-wing and antenna envelopes. The six absolute-value epigraph
objective terms per component/node retain their fixed constant contributions.
Nominal dynamics, adaptive bandwidth, tube comparison, input reserves, pairwise
separation, state and terminal constraints retain their original formulation.
This remains a nonlinear GEKKO/IPOPT problem. The original normalized numerical
component-margin gate and independent whole-hold safety admission both remain.

This option **restricts** the feasible set. For a unit box centered at the
origin with 0.1 m bus radius and 0.2 m position-box half widths, a reference at
$(2,0,0)$ selects a positive-$x$ witness. The safe alternative $(0,2,0)$ has a
positive-$y$ margin of 0.7 m but a fixed positive-$x$ margin of $-1.3$ m. The
corridor therefore excludes that detour. Face changes can also create abrupt
halfspace changes under bounded dynamics. Neither recursive feasibility nor
whole-route completion follows from selecting this option. There is no
automatic fallback between modes.

Construction refuses invalid modes, poses, references, or reference witnesses
that do not separate the finite initial predicted tube from every component.
Such a refusal is not a proof of physical infeasibility. Solver diagnostics
record the mode, the restriction, the witness policy/count/fingerprint and the
original component-margin check. Witness fingerprints bind the complete
geometry, node poses, reference points, tube seeds and numerical support data.
They identify numerical inputs; they are not interval arithmetic certificates.

An outside replay of the exact retained Envisat second-chaser problem used
132 fixed witnesses for all 11 components at 12 nodes. It reduced explicit
decisions from 1551 to 363 and solved in 2.36 s versus 44.87 s for the qualified
free-normal representation on the same machine. Projecting the result back to
all 2526 original free-normal residuals gave a maximum violation of
$8.48\times10^{-11}$, below the unchanged $10^{-8}$ numerical gate, and retained
the original restricted objective. This is one recorded optimization problem,
not an end-to-end mission or a general runtime guarantee. The shipped Envisat
study explicitly selects `solver.composite_geometry_mode: reference_corridor`,
matching its source-qualified continuation. The generic solver default remains
`free`; select it explicitly to reproduce the unrestricted formulation. This
choice changes the declared planning policy and must stay in saved configuration.

## Finite cluster, patch assignment and staged routes

`CompositeAttachmentPlanner` assigns the supplied structural-face patches by
finite-graph route length. A patch is eligible only when its staging-to-final
COM segment excludes every component after finite-body inflation. The final
COM point is the patch point plus the explicit normal standoff. This does not
assert that a panel supports attachment loads or supplies a docking fixture.

For numerical route screening, each component is transformed by `C⁻¹`. A
Euclidean padding radius contributes its value times each row norm of `C⁻¹`
to the corresponding local half extent. The padding includes bus radius,
reference tube, clearance and that component's geometry uncertainty. Slab
intersection then conservatively screens each straight segment against every
inflated component. Component corners and an outer box provide route-search
candidates; the outer box never replaces the component collision checks.

The compact release grid has finite center separation. One reference moves at
a time; the others remain at their release or staging body-frame points. Each
segment is also screened against all stationary fleet points with the declared
pair distance plus both reference-tube radii. A common rigid rotation preserves
the theoretical pair distances. A failed matching or exhausted visibility graph
means this bounded search did not find a route; it is not an unreachability
proof. The controller and whole-fleet safety guard evaluate actual motion.

Each route segment uses a quintic interpolation with zero first and second
derivatives at both endpoints. Segment durations enforce the declared
body-frame reference speed and acceleration bounds. Full reference derivatives
include `R`, `R'`, `R''` and the Coriolis term `2 R' p_body'`. These scheduling
bounds do not establish actuator feasibility: a rapidly tumbling target can
require much more acceleration in LVLH. The saved sampled feedforward and
actual optimization outcomes expose that distinction.

Docking begins only after all references are staged and the current fleet
passes whole-tube pose, component-clearance and pair-distance gates. The
nonlinear docking controller then tracks the assigned COM goals. Terminal
completion, attachment, detumbling and deorbit retain their separate admission
and resource conditions.

## Prospective Envisat profile

`config/envisat.yaml` and its adjacent prior, deployment and admission JSON files
declare an eight-bus study with 0.55 m bus radii, 0.85 m COM standoff and 1.3 m
minimum pair-center distance. The historical reported 2.4 degree/s spin
magnitude is used as an explicit axial initial condition; it is not present
telemetry. The small-eccentricity orbit, drag values, independent prior,
navigation/pose assumptions and actuator bounds are explicit study inputs.
Slow-spin reference studies and earlier cylinder results cannot establish
completion of this full-spin composite mission.
