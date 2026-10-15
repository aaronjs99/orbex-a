# Continuous safety for a composite spacecraft

The occupied target is the union of every component in the explicit
`SpacecraftGeometry`. Its bus, solar array, yoke, radar antenna, reflector,
mast, instrument enclosures and interface structure participate in exclusion.
A deployment/camera radius does not replace that union. Unsupported geometry
refuses admission.

The Envisat asset in `config/spacecraft/envisat.json` combines published
dimensions and mass properties with separately identified layout, articulation
and modeling assumptions. Its provenance and uncertainty are part of the
geometry identity. Unknown flight articulation, deformation and damage are
not established by the declared study radii; the certificates are conditional
on the complete occupied geometry actually satisfying that declaration.

## Whole-hold separation

For component `k`, let `V_k` be its exact expanded body-frame vertices and
`C_k=conv(V_k)` their convex hull. The core geometry service already includes
its declared isotropic dimension/placement uncertainty in those vertices,
including the finite-precision component-frame correction. The proof layer
does not add that uncertainty a second time.

Let `R(q)` be the true common body-to-LVLH rotation and `c(q)` the target-frame
origin. Let `M` be its evaluated rotation at the exact hold start, with operator
error at most `epsilon_R`. Let `L_R` and `L_c` bound relative angular speed and
origin translation speed per anomaly throughout the hold. A segment ends at
anomaly offset `tau`. Its rotation discrepancy is bounded by

```
delta_R = min(2, L_R*tau) + epsilon_R.
```

The component radius `rho_k` is an outward bound on
`max(v in V_k) ||v||`. It includes an appendage's offset from the target COM;
an offset solar array is not treated as rotating only about its own center.

Let `B_p` enclose the chaser COM over the complete continuous flow segment,
`B_c` enclose `c` at the hold start, `r` be the physical bus bounding radius,
and `d` the requested clearance. For any nonzero separating normal `n`, the
following strict inequality proves exclusion of the entire finite bus from
this component over that complete segment:

```
inf n·(B_p - B_c) - max(v in V_k) n·M*v
    - ||n||*(r + d + rho_k*delta_R + L_c*tau) > 0.
```

The support maximum and box projection are exact rational calculations.
Arb encloses the normal/component norms and the final margin; only a strictly
positive interval grants admission. Candidate normals can come from numerical
closest points or component/world axes, because their final support and
margin are independently proved. An unsuccessful search means `UNRESOLVED`;
it does not establish a collision or physical infeasibility.

Every segment of every flow must pass for every component. Peer exclusion
separately checks every overlapping pair of continuous flow partitions with
their finite body radii, clearance and prescribed minimum center separation.
Flow coverage, exact common clocks, frame and fingerprints are checked before
geometry. Missing segments or missing/unproved flows cannot pass.

`RotatingCompositeBounds` carries the complete geometry, evaluated matrix,
matrix-error budget, rates, center box and frame/scope. Its geometry identity
is serialized in the collision certificate. `CertifiedExecutionGuard` selects
the composite guard explicitly, includes this identity in restart context,
and refuses a changed or removed appendage inside a committed proof prefix.
Exact accumulated clock corrections are included in pose/center uncertainty.

### Derived finite-domain rotation rate

`bound_rigid_body_rotation(start_anomaly, end_anomaly, parameters, *, inertia,
initial_spin, gravity_gradient=True, precision_bits=160)` returns a typed
`RigidBodyRotationBounds`. The initial spin box applies at the specified start
anomaly. It uses Arb orbital radius/clock bounds and the same exact inertia and
energy comparison described below; it does not take a maximum from the target
preview. Its whole-domain relative angular-rate upper bound is
`1 + spin_norm_upper / clock.anomaly_rate.lower[0]`. The added one is the exact
planar LVLH basis rotation per anomaly radian. Additional torque models are
unsupported by this helper.

The composite `PhysicalApproachScenario` uses this derived rate over the full
declared target preview domain, with the target's known initial spin and fixed
inertia. The target clock, initial spin and gravity-gradient option must agree
with the explicit approach configuration/options, and its preview must contain
the initial anomaly zero. The guard records the complete rate derivation in
`rotation_rate_bounds` and rejects holds outside its finite domain. The
`ContinuousAdmissionOptions` numerical rotation operator error remains a
caller-declared assumption; a derived rate does not certify DOP853 attitude
integration accuracy. Caller rate declarations remain supported for explicitly
constructed `RotatingCompositeBounds` without a derived certificate, and for
the separately qualified cylinder path.

For the declared Envisat study tensor, initial axial spin of 2.4 degrees/s,
`a=7178137 m`, `e=.001`, `alpha=2e-9` and anomaly domain `[0,1.214]`, the
source-qualified helper gives relative rate at most `124.238590 rad/rad` and
spin norm at most `.128027116 rad/s`. These conservative energy/clock values
are conditional on those inputs and gravity-gradient-only torque. A smaller
declared rate is not disproved by this sufficient upper bound.

## Reference clearance and terminal tracking

`CompositeTerminalGeometry.point_separation(point_body)` proves distance from
a body-fixed reference point to every component using the same exact supports.
For each part, a positive normalized support gap lower-bounds the Euclidean
distance. The minimum of those part bounds lower-bounds distance to the entire
union. A common true rigid rotation preserves these distances over the whole
terminal interval. The caller's solid-position error is then reserved, and the
terminal inequalities erode the remaining distance by the complete position
tube, physical bus radius and clearance.

`CompositePhysicalTerminalBoundsBuilder` uses that service. It records a
distinct composite method, geometry fingerprint and per-component distance
evidence. The held controller, orbital interval derivatives, uncertainty
prior, sensing/evaluation contracts and inward/input/whole-set tolerance
inequalities remain explicit. The inherited cylinder dimension fields are
inactive compatibility data in this subclass; no composite proof uses them.

The composite spin comparison uses the fixed physical inertia tensor `J`.
Let `j_min` lower-bound its smallest eigenvalue, `E=omega^T J omega/2` be its
rotational energy and `V=sqrt(2E/j_min)`. Let `tau_max` bound the gravity-gradient
torque throughout the analytical orbit interval, and `T` bound physical time
from the entry event, including the expanded reference-query interval. Then

```
||omega|| <= V,
|dV/dt| <= tau_max/j_min,
||omega(t)|| <= sqrt(Q_entry_upper/j_min) + (tau_max/j_min)*T.
```

Here `Q_entry_upper` is an Arb upper enclosure of `omega^T J omega` over the
explicit entry-spin box, including off-diagonal inertia. The absolute comparison
also covers backward query displacement; positive regularization establishes
the zero-energy limit without dividing by zero. The torque bound is
`tau_max <= 3*mu*(j_max-j_min)/(2*r_min^3)`. Additional environmental/control
torques require their own valid bounds; they are not silently included.
This finite-domain energy bound avoids the scalar quadratic first-exit branch
failure for a highly anisotropic spacecraft. When the separately validated
scalar first-exit bound is tighter on a short window, the implementation uses
the minimum of the two valid bounds and records both certificates and its
selection. It does not ensure feasibility of
the subsequent input or held-feedback inequalities; those can still refuse.

## Runtime and source binding

```python
from orbexa.control import RotatingCompositeBounds, CompositeTerminalGeometry
from orbexa.control import CompositePhysicalTerminalBoundsBuilder
from orbexa.control import bound_rigid_body_rotation, IntervalBox, OrbitalModelBounds
from orbexa.core.spacecraft_geometry import SpacecraftGeometry

geometry = SpacecraftGeometry.from_json(geometry_path)
clearance = CompositeTerminalGeometry(geometry).point_separation(body_point)
rate = bound_rigid_body_rotation(
    q_initial, q_final, OrbitalModelBounds(**target_clock_parameters),
    inertia=target_inertia,
    initial_spin=IntervalBox.from_center_radius(initial_body_spin, spin_error),
    gravity_gradient=True,
)
pose = RotatingCompositeBounds(
    geometry=geometry,
    orientation_matrix_at_start=body_to_lvlh,
    orientation_matrix_error=declared_matrix_and_clock_error,
    angular_rate_per_anomaly=rate.relative_angular_rate_upper,
    rotation_rate_bounds=rate,
)
```

`PhysicalTerminalRunner` requires a composite physical target and composite
builders to share the identical full geometry. Its per-hold journal links the
actual held command, uncertain flow and each-part swept certificate. The source
snapshot factory, checkpoint context and `PhysicalTerminalHandoff` reconstruct
the explicitly identified shape, bind its complete geometry and refuse a
cylinder/composite mismatch. The original approach stays unchanged; a new
composite target cannot inherit the earlier cylinder mission's certificates.

The earlier cylinder records remain a qualified analytical baseline for their
own declared geometry. New generic composite service fixtures, prospective
Envisat terminal inequalities, initialized terminal replays and actual approach
handoffs are different evidence scopes. None establishes a universal recursive
feasibility, failure-continuation, hardware sensing or flight collision theorem.
