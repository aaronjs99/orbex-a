# Envisat geometry and physical assumptions

ORBEX-A's Envisat asset is an explicit union of eleven occupied components: the main bus, solar wing and yoke, ASAR antenna and mount, communications mast and reflector, three payload envelopes, and the existing launch interface. The asset is [config/spacecraft/envisat.json](../config/spacecraft/envisat.json). Each component participates in composite planning and exclusion; replacing this union with a cylinder would change the problem.

This is a source-dimensioned study model of a real spacecraft. It is **not flight CAD, a reconstruction of its current shape, or proof that the declared uncertainty radii enclose the actual vehicle**. Placement, some dimensions and fixed appendage articulation are inferred and identified in the JSON. Small protrusions, blankets, flexible motion and present damage are not independently enclosed.

## Dimensions and layout

The following dimensions are full lengths in metres, before the separately declared isotropic modeling radius. Centers and component rotations in the asset define the exact study pose.

| Occupied component | Full dimensions (m) | Basis and remaining assumptions |
| :-- | :-- | :-- |
| Main bus | 10.02 × 2.75 × 2.075 | Published length/width; the larger published body depth is retained. |
| Solar array | 14.028 × 4.972 × 0.010 | Published dimensions; placement and 30° roll are study assumptions. |
| Array yoke | 3.00 × 0.24 × 0.24 | Published boom length; section and pose inferred. |
| ASAR antenna | 1.30 × 10.00 × 0.16 | Published aperture; thickness, placement and 26.19° tilt inferred. |
| ASAR mount | 0.45 × 1.15 × 0.65 | Inferred occupied mounting envelope. |
| Ka-band mast | 0.18 × 0.18 × 2.00 | Published length; section and location inferred. |
| Ka-band reflector | 0.90 × 0.90 × 0.32 | Published aperture diameter; enclosing thickness and pose inferred. |
| Optical payload envelope | 2.00 × 2.10 × 0.75 | Inferred grouped instrument envelope. |
| Nadir payload envelope | 1.80 × 2.45 × 0.70 | Inferred grouped instrument envelope. |
| Side payload envelope | 1.65 × 0.65 × 1.65 | Inferred grouped instrument envelope. |
| Launch-interface enclosure | 0.22 × 2.30 × 2.30 | Inferred cuboid around existing structure; no added docking port. |

Bus and array numbers come from [Peters et al., Table 1 and Figure 1 (2018)](https://hanspeterschaub.info/PapersPrivate/Peters2018.pdf). Its 1.6 m bus depth differs from the 2.075 m depth in [Corso, Table 4.4 (2016)](https://elib.dlr.de/109429/1/CORSO_Mario_Master_Thesis_elib.pdf); the asset uses 2.075 m. Corso's Table 4.1 also supplies the 3 m boom. The ASAR aperture is documented by the [ESA instrument team, Bulletin 106, p. 48](https://www.esa.int/esapub/bulletin/bullet106/bul106_3.pdf). The communications dimensions are in [ESA BR-250](https://www.esa.int/esapub/br/br250/br250.pdf).

ESA's rounded deployed envelopes differ: [BR-250](https://www.esa.int/esapub/br/br250/br250.pdf) gives approximately 26 × 10 × 5 m, whereas the [mission FAQ](https://www.esa.int/Applications/Observing_the_Earth/Envisat_FAQs) gives 25 × 10 × 7 m. These are not precise component coordinates. The asset does not silently treat either rounded envelope as exact geometry.

## Mass, inertia and body frame

The nominal mass is 7827.867 kg. The inertia tensor $J$, in kg m² about the center of mass, is

$$
J=\begin{bmatrix}
17023.3&397.1&-2171.1\\
397.1&124825.7&344.2\\
-2171.1&344.2&129112.2
\end{bmatrix}.
$$

These are published removal-study values from [Peters et al., Table 1](https://hanspeterschaub.info/PapersPrivate/Peters2018.pdf), independently cross-checked against [Corso, Table 4.4](https://elib.dlr.de/109429/1/CORSO_Mario_Master_Thesis_elib.pdf). They are not present-day flight measurements. Neither source supplies a current confidence interval; the asset leaves mass and inertia uncertainty unspecified. Collision boxes do not imply a uniform density or a replacement inertia tensor.

The canonical frame is `target_body_COM`. Its axes remain parallel to the published interface-origin axes; the origin translates by the reported center of mass $(-3.905,-0.009,0.003)$ m. There is no principal-axis rotation. Component coordinates, patch coordinates, attitude and the full tensor must all use this frame.

## Unprepared attachment candidates

Eight `bus_upper_01`–`bus_upper_08` patches lie on the nominal +Z bus face. Their x coordinates are −2.4, −0.8, 0.8 and 2.4 m, their y coordinates are −0.740 and 0.758 m, and z is 1.0345 m. Each rectangle has half-widths 0.22 × 0.20 m with a 0.03 m edge margin. They are algorithmic bare-hull contact candidates. No fixture has been installed on the target, and no claim establishes that blankets, radiators or every panel can carry attachment loads.

Published e.deorbit work preferred the existing launch-adapter ring for its known stiffness, strength and accessibility; other appendages were unsuitable interfaces. This supports retaining the ring in the geometry, not asserting that arbitrary bus panels are mechanically qualified. [Jaekel et al., robotic subsystem and interface-selection study (2018)](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2018.00100/full)

A stand-off query checks clearance to **every** component. Nominal surface contact and the uncertainty-expanded collision solid are different declarations: the physical bus must remain outside the latter while its declared reach can meet the former. Contact mechanics, surface condition and structural load limits require their own model.

## Spin and articulation are dated assumptions

Laser-ranging observations gave an inertial period of 134.74 s on 25 September 2013, approximately 2.672°/s. This is a dated observation, not a current spin estimate. [Kucharski, Kirchner and Koidl, ILRS workshop paper (2013)](https://ilrs.gsfc.nasa.gov/lw18/docs/papers/Session4/13-02-15-Kirchner.pdf)

Later radar evidence matters: Fraunhofer's 2022 presentation reports 2021 observations with complex tumbling and no clear periodicity, differing from the stable-axis behavior seen during 2013–2017. A passage fit of approximately 1.68°/s was reported, with an explicit warning that constant angular velocity did not describe the passage. [Fraunhofer FHR, ESA study final presentation, PDF pp. 67–68 / slides 10–11 (2022)](https://nebula.esa.int/sites/default/files/neb_tec_studies/2745/public/GT17-152GR_FP.pdf)

The ESA/GMV 2017 detumbling study used 1–5°/s main-spin design cases and a fixed solar-panel pose assumption; those are design inputs. [GMV executive summary, Section 6 (2017)](https://nebula.esa.int/sites/default/files/neb_tec_study/1222/public/C4000113022ExS.pdf) A mission must identify its chosen entry-spin box and appendage pose. This repository does not infer current Envisat attitude, solar articulation or rigid-body validity from historical measurements.

## Numerical queries and continuous proof

`SpacecraftGeometry` freezes the component declarations and their provenance. `signed_distance` and `closest_point` are numerical candidate queries. Inside overlapping solids, the union query is an occupancy indicator, not the exact distance required to exit all components.

For continuous support calculations, `vertices_body_m` contains exact rational corners of an enclosing box. Each component's declared isotropic uncertainty is already included once. The expansion also accounts for the binary64 rotation's measured departure from orthogonality. A consumer must not add that uncertainty a second time, omit a component, or reinterpret cached display vertices as exact occupied vertices. Serialization verifies the physical declarations, cached corners and supplied fingerprint.

The composite guard uses a common rigid target pose with explicit matrix, angular-rate and translation bounds, tests each occupied component, and binds the geometry fingerprint into the certificate. It proves exclusion from the declared study geometry under those assumptions. It cannot prove that an unmeasured appendage pose or omitted physical protrusion lies inside the model. Existing cylindrical baseline runs remain evidence for their original geometry only; Envisat results require their own saved source, configuration and complete component certificates.
