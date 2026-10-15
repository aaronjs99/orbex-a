# Analytical orbital holds and separate endpoint refinement

`orbexa.control.orbital_hold` proves one held-input family for the maintained six-state analytical orbital model. It constructs a genuine continuous Picard proof, then a separate variation-of-constants (VoC) endpoint witness. It performs no mission execution, fleet admission, terminal handoff or fallback dispatch.

Import the service directly. Arb remains optional and is imported only at the proof boundary. If it is unavailable, the result is unproved with status `UNAVAILABLE`. A failed enclosure has status `UNRESOLVED`; it never grants admission.

## Explicit context and request

`OrbitalHoldContext` binds the complete retained `OrbitalModelBounds`, exact anomaly domain, original three-axis process disturbance, physical input limits, original model-declaration SHA-256, and full `ReachabilityConfig` with at least 128 proof bits. The service never replaces that prior with an estimated point or contracted box. The caller must establish that the declaration contains the physical plant; its hash provides identity, not evidence of physical accuracy.

`OrbitalHoldRequest` declares the complete initial six-state box, a box of **constant** input vectors, one exact start and positive hold width, and a separate box for possibly time-varying actuator-delivery error. State units are metres and metres per anomaly radian; input/process units are metres per anomaly radian squared. Uncertain clocks, jitter, nonlinear inertial/J2 plants and arbitrary coefficient callbacks are outside this v1 method.

The constant input box must cover the actual held binary command and any permitted constant input variation. The service checks that the constant input plus varying actuator-delivery box fits the original physical actuator limits. It adds varying delivery to the original process box once for Picard and once in the separate endpoint residual. The original process declaration remains visible and unchanged. An actuator error already allocated to that declaration must not be silently declared as a second independent error.

For existing exact boxes and declarations, the API is:

```python
from orbexa.control.orbital_hold import (
    OrbitalHoldContext, OrbitalHoldRequest, OrbitalHoldProofService,
)
from orbexa.control.reachability import ReachabilityConfig

context = OrbitalHoldContext(
    model=retained_prior,
    anomaly_domain=(exact_domain_start, exact_domain_end),
    input_limits=physical_input_limits,
    acceleration_disturbance=original_process_box,
    model_source_sha256=original_model_declaration_sha256,
    reachability_config=ReachabilityConfig(precision_bits=192),
)
service = OrbitalHoldProofService(context)
request = OrbitalHoldRequest(
    initial_state=complete_entry,
    constant_input=constant_command_box,
    start_anomaly=exact_start,
    hold_width=exact_width,
    varying_actuator_error=varying_delivery_box,
)
proof = service.prove(request)
if not proof.proven:
    raise ValueError(proof.as_dict()["reason"])

# Geometry uses the unchanged original continuous swept witness.
original_flow = proof.picard_flow
# A successor uses this separate endpoint and its witness identity.
complete_successor = proof.endpoint

# Stored evidence is reconstructed, not trusted by its saved status.
restored = service.rebuild(proof.as_dict())
```

The names supplied by the caller above are independently justified exact boxes, clocks and declarations, not sampled maxima. This example selects no physical command and advances no state.

## Endpoint theorem

Let $q$ denote anomaly, $h>0$ the exact hold width, and $x$ the six-state position/anomaly-derivative vector. Let $\theta$ range over the original retained physical prior. Let $F(q,\theta)$ and $b(q,\theta)$ be the shared analytical orbital matrix and affine term, and let $B=[0;I]$ inject acceleration. Let $u_c$ be one unknown constant vector in the constant-input box $U_c$. Let $e_v(q)$ be varying actuator error in $E_v$. Let $w(q)$ be the six-state lift $[0;d(q)]$ of the original three-axis process acceleration, and let $W$ be its lifted box. The physical equation is

\[
x'=F(q,\theta)x+b(q,\theta)+Bu_c+Be_v(q)+w(q).
\]

The authentic Picard certificate supplies a whole-hold swept box $Y$ and a Picard endpoint $P$. The same private coefficient evaluator supplies both proofs; no input-dependent model or alternate nominal point is substituted.

Choose any exact constant matrix $F_0$ and exact vector $b_0$. The implementation uses exact rational midpoints of the start coefficient enclosures under the full prior. Let $r_{\max}$ bound the componentwise magnitude of

\[
r=(F-F_0)Y+(b-b_0)+BE_v+W
\]

over the complete hold and prior. Define $E=\exp(F_0h)$ and $G=\int_0^h\exp(F_0s)\,ds$. The top blocks of the exponential of $h\begin{bmatrix}F_0&I\\0&0\end{bmatrix}$ produce $E$ and $G$ without inverting $F_0$. Entrywise comparison gives $|\exp(F_0t)|\le\exp(|F_0|t)$, so a sufficient endpoint is

\[
V=EX_0+G(b_0+BU_c)
+[-h\exp(|F_0|h)r_{\max},\ h\exp(|F_0|h)r_{\max}].
\]

The returned endpoint is the exact intersection $V\cap P$. Empty intersection refuses. The residual and matrix exponentials use outward Arb arithmetic. Narrower bounds reduce enclosure wrapping; they do not reduce physical uncertainty or establish small tracking error.

The constant-input quantifier matters. For a planar rotation generator and one full rotation, $G=0$: every constant input has zero endpoint from zero initial state. A varying input that rotates opposite the transition can instead yield a nonzero accumulated endpoint. Thus a varying actuator box cannot be treated solely as a constant through $GU_c$; it belongs in the residual.

## Proof identities and saved data

The original `FlowCertificate`, Picard segments, swept box, physical velocity/time bounds and fingerprint remain unchanged. The endpoint has distinct method `arb-voc-affine-orbital-endpoint-v1` and schema `orbexa.orbital_endpoint.v1`. A composed `orbexa.orbital_hold.v1` record binds both identities, the exact requests, full context, actual supported source-file hashes, precision/backend identity and complete VoC witness.

`prove` owns the supported Picard encloser. It accepts neither an arbitrary flow nor an unvalidated coefficient/provider callback. `rebuild` checks the expected context/source/method, re-proves the original request, recomputes the endpoint witness and compares the complete evidence. A typed object, `PROVEN` flag, fingerprint or source hash alone is not an authentication boundary. Evidence constructors exist for records; only fresh service construction/reconstruction establishes the supported proof.

An optional `swept_entry` may be a larger complete initial box for the original Picard proof. The service requires it to contain the proposed entry and constructs all other request/model fields identically. The original swept set therefore encloses the smaller trajectory family. This is exact request-family inclusion; two independently regenerated interval boxes need not be numerically nested.

Instances bind readable supported source files to the identities recorded when the modules were loaded. Both existing and newly constructed services refuse changed files under already loaded code. Different context, prior, domain, input/process declaration, precision, implementation or backend requires a separately qualified source/context transition; it is not silently accepted as an old proof. Real-plant containment and delivery assumptions remain conditional.

Existing execution guards, scalar continuation, mission/checkpoint schemas and fleet policies are not changed to consume this endpoint. A later integration must preserve all eight original swept witnesses, jointly prove every component and pair, maintain an exact cursor and whole-successor containment, and independently admit the physical terminal successor. This module establishes none of those larger outcomes.
