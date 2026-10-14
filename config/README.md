# Scenario inputs

`default.yaml` supplies the paper-scale analytical orbital translation, controller
and prescribed-target demo profile. `orbexa-run-mission` composes the physical
gravity-gradient target, compact finite-bus deployment and post-capture stages
around that controller configuration.

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
explicit physical-stage options. The maintained representative profile is a
1000 kg target and eight 8 kg dry/4 kg fuel buses, each enclosed in a 0.15 m ball,
with 0.165 m COM standoff and 0.16 m inward gripper offset. See
[the physical lifecycle](../docs/mission-lifecycle.md) for the full model scope.
