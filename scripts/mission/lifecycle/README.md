# `mission/lifecycle`

Mission lifecycle from approach boundary to entry interface.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `configuration/` | Prospective vehicle and completion profiles. | `mission/approach/physical`, `mission/attached`, `mission/attached/assembly`, `mission/attached/disposal`, `mission/contact`, `mission/lifecycle`, +3 more | `mission`, `service/jobs`, `studies` |
| `environment/` | Physical environment models for completion. | `common`, `mission/attached/assembly`, `mission/contact`, `mission/lifecycle/nonlinear`, `models/dynamics`, `numpy`, +1 more | `mission/attached`, `mission/attached/assembly`, `mission/contact`, `mission/contact/lock`, `mission/lifecycle/nonlinear` |
| `nonlinear/` | Authenticated approach boundary and nonlinear lifecycle composition. | `common`, `control/fallback`, `control/terminal`, `evidence/records`, `mission/contact`, `mission/contact/sensing`, +6 more | `mission`, `mission/lifecycle`, `mission/lifecycle/environment`, `service/jobs`, `studies` |
| `accounting.py` | Conservative physical resource accounting from authenticated nonlinear holds. | `common`, `coordinator`, `evidence/records`, `mission/attached/assembly`, `mission/lifecycle/nonlinear`, `models/dynamics` | `mission/lifecycle/nonlinear` |
| `coordinator.py` | Compose a recorded physical approach with attitude, latch and disposal stages. | `common/io`, `control/terminal`, `evidence/records`, `mission/approach/physical`, `mission/approach/terminal`, `mission/attached`, +5 more | `accounting`, `mission/lifecycle/configuration`, `mission/lifecycle/nonlinear`, `service/jobs`, `studies` |
| `stages.py` | Reusable qualified-approach → attachment → detumbling → deorbit composition. | `common/io`, `mission/approach/terminal`, `mission/attached`, `mission/attached/disposal`, `mission/contact` | `coordinator` |
