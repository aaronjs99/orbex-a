# `figures/renderers`

Lifecycle and replay renderers.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `lifecycle.py` | Saved lifecycle projection and scientific plots, without mission execution. | `common/io`, `evidence/records`, `imageio-ffmpeg`, `matplotlib`, `numpy`, `scipy` | `cli`, `service/jobs` |
