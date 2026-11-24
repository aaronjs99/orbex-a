# docs

Guides for the models, proofs and mission stages, and the design record. Start with [mission readiness](mission-readiness.md) and the [architecture map](architecture.md).

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `superpowers/` | Design specs and implementation plans. | none | none |
| `adtmpc-design.md` | ADTMPC tracking, docking, and physical-coordinate contract | `continuous-safety.md`, `mission-lifecycle.md`, `physical-terminal.md`, `research-review.md`, `theory-audit.md`, `theory.md` | none |
| `architecture.md` | ORBEX-A architecture | `finite-contact.md`, `mission-lifecycle.md`, `nonlinear-mission.md`, `nonlinear-navigation.md`, `nonlinear-solver.md` | none |
| `arrival-policy.md` | Arrival with a fixed deadline | none | `theory.md` |
| `attached-flight.md` | Attached flight | none | `nonlinear-mission.md` |
| `chaser-attitude.md` | Free-flying chaser attitude before capture | `mission-lifecycle.md` | `mission-lifecycle.md` |
| `composite-approach.md` | Component-aware spacecraft approach | none | `nonlinear-solver.md` |
| `composite-continuous-safety.md` | Continuous safety for a composite spacecraft | none | none |
| `configuration.md` | Configure a mission | `mission-readiness.md` | `mission-readiness.md` |
| `continuous-safety.md` | Continuous held-input reachability and swept safety | none | `adtmpc-design.md`, `terminal-tracking.md`, `theory.md` |
| `control-foundations.md` | ORBEX-A: control foundations for adaptive orbital capture | `finite-contact.md`, `fleet-fallback.md`, `mission-lifecycle.md`, `nonlinear-mission.md`, `sensed-tube.md`, `terminal-handoff.md`, +1 more | none |
| `deployment.md` | Carrier deployment and compact free flight | `nonlinear-mission.md` | `theory.md` |
| `envisat.md` | Envisat geometry and physical assumptions | none | `mission-lifecycle.md` |
| `finite-contact.md` | Finite compliant capture | none | `architecture.md`, `control-foundations.md`, `mission-lifecycle.md`, `mission-readiness.md`, `nonlinear-mission.md`, `theory.md` |
| `fleet-fallback.md` | Finite nonlinear fleet feedback fallback | none | `control-foundations.md`, `mission-readiness.md`, `nonlinear-mission.md`, `theory.md` |
| `mission-lifecycle.md` | Physical mission lifecycle | `chaser-attitude.md`, `envisat.md`, `finite-contact.md`, `nonlinear-mission.md` | `adtmpc-design.md`, `architecture.md`, `chaser-attitude.md`, `control-foundations.md`, `theory.md`, `uncontrolled-target.md` |
| `mission-readiness.md` | Mission readiness | `configuration.md`, `finite-contact.md`, `fleet-fallback.md`, `nonlinear-mission.md`, `nonlinear-navigation.md`, `research-review.md` | `configuration.md`, `research-review.md` |
| `nonlinear-mission.md` | Nonlinear fleet mission | `attached-flight.md`, `finite-contact.md`, `fleet-fallback.md`, `nonlinear-navigation.md`, `sensed-tube.md` | `architecture.md`, `control-foundations.md`, `deployment.md`, `mission-lifecycle.md`, `mission-readiness.md`, `theory.md` |
| `nonlinear-navigation.md` | Nonlinear reference motion and bounded navigation | none | `architecture.md`, `mission-readiness.md`, `nonlinear-mission.md`, `theory.md` |
| `nonlinear-solver.md` | Nonlinear solver acceptance and failure handling | `composite-approach.md`, `terminal-tracking.md` | `architecture.md`, `research-review.md` |
| `orbital-held-flow.md` | Analytical orbital holds and separate endpoint refinement | none | `theory.md` |
| `physical-terminal.md` | Finite terminal bounds for the physical uncontrolled target | `terminal-tracking.md`, `theory.md` | `adtmpc-design.md`, `terminal-tracking.md`, `theory.md` |
| `research-review.md` | Research review and implementation decisions | `mission-readiness.md`, `nonlinear-solver.md` | `adtmpc-design.md`, `mission-readiness.md`, `theory.md` |
| `sensed-tube.md` | ADTMPC with bounded observation error | none | `control-foundations.md`, `nonlinear-mission.md`, `theory.md` |
| `terminal-handoff.md` | Physical terminal endpoint handoff | none | `control-foundations.md` |
| `terminal-tracking.md` | Terminal tracking and admissible continuation | `continuous-safety.md`, `physical-terminal.md` | `nonlinear-solver.md`, `physical-terminal.md`, `theory.md` |
| `theory-audit.md` | Equation and implementation audit | `theory.md` | `adtmpc-design.md`, `theory.md` |
| `theory.md` | ORBEX-A mathematical foundations | `arrival-policy.md`, `continuous-safety.md`, `deployment.md`, `finite-contact.md`, `fleet-fallback.md`, `mission-lifecycle.md`, +8 more | `adtmpc-design.md`, `control-foundations.md`, `physical-terminal.md`, `theory-audit.md`, `uncontrolled-target.md` |
| `uncontrolled-target.md` | Uncontrolled rigid-body target | `mission-lifecycle.md`, `theory.md` | none |
