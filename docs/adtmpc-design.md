# ADTMPC tracking, docking, and physical-coordinate contract

## Verdict

The earlier mission's successful physical-clearance samples did not establish
dynamic-tube containment. At Git revision `458a031`, the saved default single
mission exceeded its matching next-node position radius on 144 of 181 intervals.
The nominal center was reset to the actual state, feedback was evaluated only
at that reset, and docking disabled tube expansion. The saved horizon maximum
radius also obscured the relevant next-node radius. These are implementation
defects, not necessary behavior of ADTMPC.

The current nonlinear mission jointly optimizes nominal motion, tube states,
and feedback bandwidth with GEKKO/IPOPT. It carries the nominal center and tube
between updates, applies ancillary feedback at configured substeps, and accepts
the final pose only when the entire current position/velocity tube fits its
tolerances for consecutive endpoint checks. It stops on failed containment,
input reserves, physical clearance, or inconsistent parameter membership.

## Conventions and assumptions

`physical-lvlh-v1` uses meters in a right-handed local frame: x is opposite the
transverse orbital direction, y is radial outward, and z is orbit normal.
The independent variable is unwrapped true anomaly. State components 3–5 are
position derivatives with respect to anomaly; they are not meters per second.
Inputs are anomaly acceleration, not Newtons. Physical acceleration increments
are obtained by multiplying an input by the square of the anomaly rate.

The reference follows an analytical eccentric orbit with the Carter–Humi drag
approximation. The relative equations are a local linearization in physical
coordinates. The historical transformed equations remain explicitly available
as `orbital_ellp_drag_legacy_transformed`; they cannot be combined directly with
meter-valued collision geometry. See [Carter and Humi, 2002](https://doi.org/10.2514/2.5010).
The model is not a calibrated atmosphere, an orbital propagator with J2, or a
contact-mechanics model. Target attitude remains prescribed Euler motion per
anomaly. Its inertia is not used to claim torque-free tumbling.

## Symbols, shapes, units, and roles

| Symbol | Meaning and role | Shape / units |
|---|---|---|
| q | Unwrapped true anomaly, the independent variable | scalar, rad |
| x, z | Actual and nominal state, each position followed by anomaly velocity | 6, m and m/rad |
| e_p, e_v | Actual minus nominal position and anomaly velocity | 3 each |
| Lambda | Positive diagonal sliding gain | 3, 1/rad |
| s | Sliding error, combining velocity and position tracking error | 3, m/rad |
| Phi | Componentwise nonnegative sliding-error envelope | 3, m/rad |
| Omega | Componentwise nonnegative position-error envelope | 3, m |
| kappa | Optimized ancillary bandwidth; unrelated to target drag coefficient | 3, 1/rad |
| Delta_A, Delta_d | Nonnegative coefficient-error envelopes over the full parameter box | 3×6 and 3 |
| w | Nominal-trajectory uncertainty bound, including configured numerical reserve | 3, m/rad² |

## Corrected tube construction

Define the sliding error before using its envelope:

```
s = e_v + Lambda e_p
|e_p| <= Omega
|e_v| <= Phi + Lambda Omega
Phi' = -kappa Phi + w
Omega' = -Lambda Omega + Phi
```

Uncertainty is evaluated on an enclosure of the nominal trajectory over each
held-input interval. Interval forward differentiation and a mean-value bound
cover nonlinear eccentricity and drag dependence throughout the parameter box;
one largest-norm corner is insufficient. Positive exponential tube updates
replace forward-Euler clipping. The ancillary input reserve includes all nominal
cancellation, sliding feedback, and uncertainty-feedback terms. Empty admissible
sets cause failure.

The point estimator may use local optimization. Only the outward-rounded
interval RK4 membership contractor may remove parameter values. It preserves
all unevaluated boxes when its work budget expires. Its enclosure guarantee is
conditional on the recorded held-input RK4 model and configured measurement
error, rather than a claim about arbitrary physical plants.

## Moving goal and moving obstacle

A target can supply both, but they are different objects. Its desired docking
pose and derivative define the moving tracking reference. Its occupied volume
defines the moving exclusion set. Both must use the same attitude trajectory,
frame, clock, and geometry. The desired reference is never shifted outward to
make success depend on the actual vehicle occupying the inward tube edge.

For terminal acceptance, the nominal position error plus the Euclidean radius
of Omega must fit the position tolerance. The nominal velocity error plus the
Euclidean radius of `Phi + Lambda Omega` must fit the velocity tolerance. A
nonzero symmetric free-flight tube centered on a solid target's surface cannot
lie wholly outside that target. The default goal is consequently a specified
5 mm standoff; physical capture/contact requires a separate model and transition.

## Focused literature review and implementation implications

SciSpace was used for semantic discovery and method extraction on 2026-10-13.
Primary papers were then inspected; generated index summaries were not treated
as evidence. This is a focused design review, not an exhaustive novelty search.

| Primary work | Supported result | Implication here |
|---|---|---|
| [Lopez, Slotine and How, Dynamic Tube MPC for Nonlinear Systems, 2019](https://arxiv.org/abs/1907.06553) | Boundary-layer feedback links uncertainty, bandwidth, and dynamic tube geometry. | Optimize bandwidth with nominal motion and tube evolution, and reserve the complete feedback input. |
| [Morozov et al., Performance Analysis of Adaptive Dynamic Tube MPC, 2021](https://arxiv.org/abs/2012.12403) | Experiments on a pendulum assess dynamic tubes and SMID adaptation. Its adaptation construction assumes linearly parameterized uncertain dynamics. | Those performance numbers and guarantees do not transfer directly to eccentric orbital parameters. ORBEX-A uses interval membership for their nonlinear parameter dependence. |
| [Limon et al., On the Design of Robust Tube-Based MPC for Tracking, 2008](https://idus.us.es/bitstreams/18ad8ca8-e9a7-4782-8920-b0899d0cf769/download) | Admissible references and an invariant tracking terminal set preserve feasibility for the stated linear-system setting. | A terminal cost alone is insufficient; requested pose accuracy and tube admissibility need explicit conditions. |
| [Quartullo et al., Robust Variable-Horizon MPC with Adaptive Terminal Constraints, 2024 preprint](https://arxiv.org/abs/2410.08807) | A moving-reference terminal-set mechanism reduces final-distance conservatism while proving feasibility/convergence under its stated LTI assumptions. | Your terminal-distance concern already has relevant literature. Fixed-horizon nonlinear ORBEX-A does not inherit that theorem. |

The defensible research direction is joint terminal accuracy, collision clearance,
uncertainty contraction, and bandwidth allocation for the moving docking geometry.
These ingredients are not, individually, a novelty claim. A publication claim
needs a sharper comparison and a proof or an explicitly experimental scope.

## Evidence limits

GEKKO checks nonlinear geometry at prediction nodes. Runtime checks containment,
clearance, and separation at the configured feedback substeps. These observations
do not establish continuous-time obstacle avoidance or recursive feasibility.
The local terminal Riccati penalty is a performance aid. The terminal region
implemented here is an admissibility and acceptance set, not a proved invariant
set for the complete nonlinear moving-target system.

The optimizer's outer RK4 step and the nominal center's smaller runtime RK4
steps have a recorded numerical difference. Results report that difference and
the configured numerical reserve. A platform-independent interval integration
certificate, actuator dynamics, sensed target uncertainty, torque-free attitude,
and physical capture remain separate fidelity/guarantee extensions.
