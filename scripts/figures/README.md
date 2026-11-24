# `figures`

Figures, renderers and Studio export from recorded mission data.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `data/` | Projections of recorded data for figures and the Studio. | `figures`, `figures/plots`, `matplotlib`, `numpy` | `cli` |
| `plots/` | Static mission figures. | `evidence/records`, `figures`, `matplotlib`, `models/dynamics`, `numpy` | `cli`, `figures/data` |
| `publishing/` | Studio export and publishing. | `common`, `evidence/projections`, `evidence/records`, `mistune` | `cli` |
| `renderers/` | Lifecycle and replay renderers. | `common/io`, `evidence/records`, `imageio-ffmpeg`, `matplotlib`, `numpy`, `scipy` | `cli`, `service/jobs` |
| `style.py` | Lightweight, consistent identity and plot styles; no Matplotlib setup. | none | `figures/data`, `figures/plots` |
