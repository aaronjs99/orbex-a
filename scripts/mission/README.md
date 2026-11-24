# `mission`

Mission stages and their orchestration: approach, contact, attached flight and lifecycle.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `approach/` | Approach-stage models, deployment, references, arrival, attitude and terminal handoff. | `common`, `control/mpc`, `control/terminal`, `control/tubes`, `estimation/adaptation`, `evidence/records`, +11 more | `mission`, `mission/contact`, `mission/execution`, `mission/lifecycle`, `mission/lifecycle/configuration`, `mission/records`, +1 more |
| `attached/` | Assembled-spacecraft flight after attachment. | `common`, `evidence`, `mission/contact`, `mission/contact/dynamics`, `mission/lifecycle/environment`, `models/dynamics`, +4 more | `mission/approach/physical`, `mission/contact`, `mission/contact/dynamics`, `mission/lifecycle`, `mission/lifecycle/configuration`, `mission/lifecycle/environment` |
| `contact/` | Finite contact, capture and lock. | `common`, `mission/approach/terminal`, `mission/attached/assembly`, `mission/lifecycle/environment`, `models/dynamics`, `models/geometry`, +4 more | `mission/approach/physical`, `mission/approach/terminal`, `mission/attached/assembly`, `mission/lifecycle`, `mission/lifecycle/configuration`, `mission/lifecycle/environment`, +1 more |
| `coordination/` | Multi-chaser coordination. | `common/geometry`, `matplotlib`, `numpy`, `scipy` | none |
| `execution/` | Held-command admission, physical commit and sensing delivery for each hold. | `common`, `control`, `control/fallback`, `control/tubes`, `estimation/membership`, `estimation/navigation`, +7 more | `mission` |
| `lifecycle/` | Mission lifecycle from approach boundary to entry interface. | `common`, `common/io`, `control/fallback`, `control/terminal`, `evidence/records`, `mission/approach/physical`, +14 more | `mission`, `mission/attached`, `mission/attached/assembly`, `mission/contact`, `mission/contact/lock`, `service/jobs`, +1 more |
| `online/` | Independent online execution: clocks, decisions, worker planning and observation. | `common`, `control/fallback`, `estimation/navigation`, `mission/safety`, `proofs` | none |
| `records/` | Mission records collected during execution: per-chaser arrays and the nonlinear recorder. | `control/fallback`, `control/tubes`, `estimation/adaptation`, `evidence/projections`, `evidence/records`, `mission/approach`, +2 more | `mission` |
| `rendezvous/` | Rendezvous routes around the target. | `models/geometry`, `numpy`, `scipy` | `mission/approach/deployment` |
| `safety/` | Finite fallback refuges, candidates and backup planning for the fleet. | `common`, `control/fallback`, `estimation/navigation`, `models/dynamics`, `models/geometry`, `numpy`, +3 more | `mission`, `mission/execution`, `mission/lifecycle/nonlinear`, `mission/online` |
| `planning.py` | GEKKO ADTMPC adapter with explicit nonlinear regional residuals and sensing. | `common`, `control/tubes`, `mission/approach`, `models/dynamics`, `numpy`, `proofs` | `runner` |
| `profile.py` | Explicit prospective cases for sensed, nonlinear multi-spacecraft missions. | `estimation/adaptation`, `estimation/navigation`, `mission/approach`, `mission/approach/deployment`, `mission/lifecycle/configuration`, `mission/safety`, +2 more | `runner`, `service/jobs`, `studies` |
| `runner.py` | Observation-driven nonlinear ADTMPC mission orchestration. | `common`, `control/fallback`, `control/terminal`, `control/tubes`, `estimation/adaptation`, `estimation/navigation`, +12 more | `service/jobs`, `studies` |
