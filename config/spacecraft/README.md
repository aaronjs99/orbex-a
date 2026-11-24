# config/spacecraft

Vehicle geometry for one mission design. Each file declares the occupied components, the attachment patches, the vehicle mass properties and the chaser propulsion.

## File Structure

| File | Relevance | Dependencies | Used by |
|---|---|---|---|
| `envisat.json` | Eleven occupied components, eight attachment patches, vehicle mass and propulsion for the default Envisat study. | `models/geometry/composite.py` parses it | `run.py`, `serve.py`, `mission/lifecycle/configuration/design.py` |
