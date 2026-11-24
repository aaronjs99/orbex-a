# `models/configuration`

Structured scenario configuration and loading.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `loader.py` | YAML decoding into structured configuration, loaded on demand. | `common`, `common/io`, `model`, `numpy`, `types` | `model` |
| `model.py` | Simulation configuration composition and stable public model exports. | `loader`, `numpy`, `types` | `loader`, `mission/approach`, `mission/approach/references`, `models/dynamics`, `models/targets`, `service/jobs`, +1 more |
| `types.py` | Structured numeric and policy configuration contracts. | `common`, `numpy` | `loader`, `model` |
