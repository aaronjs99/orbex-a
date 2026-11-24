# `models`

Physical models: clocks and frames, dynamics, geometry, targets and configuration.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `configuration/` | Structured scenario configuration and loading. | `common`, `common/io`, `numpy` | `mission/approach`, `mission/approach/references`, `models/dynamics`, `models/targets`, `service/jobs`, `studies` |
| `dynamics/` | Translational, attitude and rigid-body dynamics with their clocks and frames. | `common`, `common/geometry`, `matplotlib`, `models/configuration`, `numpy`, `python-flint`, +1 more | `control`, `control/fallback`, `control/terminal`, `control/tubes`, `estimation/adaptation`, `estimation/membership`, +25 more |
| `geometry/` | Composite spacecraft geometry, attachment patches and mounting. | `common/geometry`, `models/dynamics`, `numpy`, `scipy` | `evidence/projections`, `mission/approach`, `mission/approach/physical`, `mission/approach/terminal`, `mission/contact`, `mission/lifecycle/configuration`, +5 more |
| `targets/` | Target truth and observed target previews. | `common/geometry`, `models/configuration`, `models/dynamics`, `models/geometry`, `numpy`, `scipy` | `mission`, `mission/approach`, `mission/approach/references`, `mission/approach/terminal`, `mission/lifecycle/configuration` |
