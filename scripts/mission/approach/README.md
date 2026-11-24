# `mission/approach`

Approach-stage models, deployment, references, arrival, attitude and terminal handoff.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `arrival/` | Arrival policy configuration and commitments. | `control/terminal` | `mission/approach` |
| `deployment/` | Fleet deployment from the payload cluster. | `control/mpc`, `mission/approach`, `mission/approach/references`, `mission/rendezvous`, `numpy` | `mission` |
| `physical/` | Chaser attitude tracking during approach. | `mission/attached/assembly`, `mission/contact`, `models/dynamics`, `models/geometry`, `numpy`, `scipy` | `mission/lifecycle`, `mission/lifecycle/configuration`, `studies` |
| `references/` | Phase reference and horizon policies. | `control/mpc`, `mission/approach`, `models/configuration`, `models/targets`, `numpy` | `mission/approach`, `mission/approach/deployment` |
| `terminal/` | Terminal execution and the physical terminal handoff. | `common`, `control/terminal`, `evidence/records`, `mission/contact`, `models/geometry`, `models/targets`, +3 more | `mission/contact`, `mission/lifecycle` |
| `execution.py` | Dense sampled ancillary feedback on one shared fleet clock. | `numpy` | `mission/execution` |
| `fleet.py` | Fleet initialization, separate from mission policy and numerical execution. | `models`, `models/configuration`, `models/targets`, `numpy` | `mission/approach/deployment`, `scenario` |
| `models.py` | Plain chaser assignments and phase records shared by runtime services. | `numpy` | `fleet`, `mission/approach/deployment`, `mission/approach/references`, `mission/records`, `planning`, `safety`, +1 more |
| `planning.py` | Coupled planning of all chasers with the existing dynamic-tube backend. | `common`, `control/mpc`, `control/tubes`, `estimation/adaptation`, `mission/approach/arrival`, `models`, +4 more | `mission`, `mission/records` |
| `safety.py` | Measured physical safety and matched-tube planning margins. | `models`, `models/configuration`, `models/geometry`, `models/targets`, `numpy` | `mission`, `mission/execution`, `scenario` |
| `scenario.py` | Explicit dependencies and policy services for a coupled tube mission. | `estimation/adaptation`, `fleet`, `mission/approach/arrival`, `mission/approach/references`, `models/configuration`, `models/dynamics`, +4 more | `mission/approach/deployment`, `mission/records`, `planning` |
| `state.py` | Persistent nominal centers and tube envelopes on the shared fleet clock. | `mission/approach/arrival`, `models`, `models/configuration`, `numpy` | `mission`, `mission/approach/references`, `mission/records`, `planning` |
