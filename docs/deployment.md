# Carrier deployment and compact free flight

The current 300 m Envisat mission uses `NonlinearMissionRunner`: eight buses
start in one separated payload-sized cluster, follow component-aware routes to
assigned attachment patches, and require global nonlinear fleet admission and
a finite fuel-backed backup before motion. See the [sensed mission](nonlinear-mission.md).

`CarrierTransferReference` blends their common centre in the commanded
reference-orbit frame before the proximity routes begin. In this composition
the historical `center_lvlh_m` field names reference-frame coordinates, not
the perturbed target's physical LVLH. The distant centre is never multiplied
by target rotation. Only the small local dispenser offsets rotate, so release
does not acquire a spurious angular-rate times 300 m velocity.

Let $C$ denote the distant reference-frame centre, $c$ the close-route
body-frame centre, $\delta_i$ a fixed dispenser offset, $R(q)$ the observed
target preview's body-to-reference rotation, and $s(q)$ a quintic progress
function over the declared transfer duration. The parallel reference is

$$p_i=(1-s)C+R(q)(sc+\delta_i).$$

Both derivatives of $s$ vanish at each end. Position, anomaly velocity and
acceleration therefore join the delayed body-frame proximity routes
continuously. Pair differences are $R(q)(\delta_i-\delta_j)$, preserving the
compact grid's reference spacing. This does not prove target clearance or
control feasibility: all eleven components, actual controls and continuous
swept admission remain binding for every executed hold. The displayed carrier
is initial release context; launch and mechanical ejection are not simulated.

## Component-aware release

`CompositePatchDeployment` (`mission/approach/deployment/component.py`) turns the
payload cluster into per-chaser start states and staged routes to the assigned
attachment patches. It blends the cluster with the same quintic progress used by
`CarrierTransferReference`, keeps every pair at or beyond the configured minimum
spacing, and refuses a cluster or route that cannot satisfy it. It never expands
the launch radius or reduces the minimum spacing to repair a requested packing.
The reference geometry is a proposal: input bounds, ancillary reserves, tubes,
pair constraints and target exclusion remain with the planner and the admission
layer, and no deployment geometry certifies them.

