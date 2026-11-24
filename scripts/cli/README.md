# `cli`

Command-line entrypoints for ORBEX-A.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `figures.py` | command line that renders figures from recorded runs into results/. | `common`, `figures/data`, `figures/plots`, `figures/renderers` | none |
| `run_nonlinear.py` | Run one explicit arm of a predeclared sensed nonlinear 2-by-2 study. | `common`, `studies` | none |
| `serve_studio.py` | Serve the local Studio at127.0.0.1:8881/studio/ and real API at127.0.0.1:8882. | `common`, `figures/publishing`, `service` | none |
