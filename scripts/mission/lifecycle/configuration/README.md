# `mission/lifecycle/configuration`

Prospective vehicle and completion profiles.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `design.py` | Injectable vehicle sizing for composite-target mission studies. | `mission/approach/physical`, `mission/attached`, `mission/attached/assembly`, `mission/attached/disposal`, `mission/contact`, `mission/lifecycle`, +4 more | `service/jobs`, `studies` |
| `profile.py` | Explicit contact, detumbling and disposal settings for any target design. | `mission/attached`, `mission/attached/disposal`, `mission/contact` | `design`, `mission` |
