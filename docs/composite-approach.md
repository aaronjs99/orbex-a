# Component-aware spacecraft approach

The Envisat study uses the union of the declared bus, solar wing, yoke, antenna,
reflector and payload envelopes. The same immutable geometry fingerprint binds
the target, assignment, nonlinear node constraints and continuous safety guard.
An enclosing sphere may set a camera or automatic release scale; it is not the
collision model. Dimensions, inferred placements, uncertainty and source scope
remain in `config/spacecraft/envisat.json`. These are study geometry declarations,
not flight CAD, current pose measurements or qualified structural interfaces.

## Nonlinear node constraints

Let `p` be the nominal chaser center in LVLH, `c0` the target center, and `R` the
declared target-body-to-LVLH matrix at a prediction node. A component has body
center `c`, component-to-body matrix `C`, positive half extents `h`, and an
isotropic geometry/placement uncertainty radius `u`. Its occupied set is
`c0 + R(c + C[-h,h] + B(u))`, where `B(u)` is a Euclidean ball. The chaser's
position error box has nonnegative half widths `Omega`; its finite body is
enclosed by a world-frame ball of radius `r`. Let `d` be the requested physical
clearance and declared pose reserves. For each component and node, a unit
LVLH normal `n` must satisfy

```
||n||² = 1
nᵀ(p - c0 - R c) >= hᵀ|CᵀRᵀn| + u ||Rᵀn|| + Omegaᵀ|n| + r + d.
```

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
