# Scenario inputs

`envisat.yaml` supplies the prospective eight-chaser composite study.
`spacecraft/envisat.json` declares the occupied components;
`envisat-deployment.json` declares routes from one compact release cluster.
The separate `envisat-prior.json` does not read plant truth to center the belief.
`envisat-admission.json` identifies assumed sensing/model/evaluation envelopes.
The physical profile derives a conservative target rotation-rate bound from
rigid-body energy and the orbital clock. These inputs are not measured flight
sensor accuracy or a current target-state estimate.

The installed concept has eight 130 kg wet chasers (30 kg dry and 100 kg fuel),
0.55 m body radius, 0.85 m COM standoff, 0.8 m gripper offset, 0.1 m closure
reach, 100 N translation channels and a declared 300 s specific impulse.
Target mass remains the published study value 7827.867 kg. Stages refuse
inadequate limits; target mass is not reduced to obtain success. See
[Envisat sources and assumptions](../docs/envisat.md).

`default.yaml` supplies the paper-scale analytical orbital translation, controller
and prescribed-target demo profile. `orbexa-run-mission` composes the physical
gravity-gradient target, compact finite-bus deployment and post-capture stages
around that configuration with `--target-model cylinder-baseline`.

`physical-prior.json` is an explicitly fixed controller prior and initial point.
The physical runner does not read plant truth to center or enlarge this prior.
Changing the plant may make it inconsistent, which must remain a failure.

`physical-admission.json` declares representative **assumed** sensing and model
envelopes: position error ±1 µm, anomaly-velocity error ±1e-4 m/rad, additive
anomaly acceleration ±1e-5 m/rad², target rotation operator error at most 1e-8,
and full target-relative angular-rate norm at most 16 per radian of anomaly.
These are inputs to conditional continuous-flow admission, not measured sensor
performance or an independently validated quaternion integration error.
The additive allowance also does not qualify the independent nonlinear vector-drag
plant: the [orbital fidelity study](../docs/dynamics-fidelity.md) measured a larger
model discrepancy. This profile is evidence for the analytical relative model;
switching plants requires valid model and identification enclosures for that plant.

Vehicle mass, fuel, body volume, standoff, gripper reach and nozzle limits are
explicit physical-stage options. The cylinder baseline profile is a
1000 kg target and eight 8 kg dry/4 kg fuel buses, each enclosed in a 0.15 m ball,
with 0.165 m COM standoff and 0.16 m inward gripper offset. See
[the physical lifecycle](../docs/mission-lifecycle.md) for the full model scope.
