# ORBEX-A Agent Rules

## Project Direction

- Keep the implementation anchored to the paper: elliptical relative dynamics with quadratic drag, MPC rendezvous/docking constraints, dynamic tubes, SMID adaptation, and multi-chaser extensions are the priority.
- Prefer modular, reusable interfaces over one-off experiments or hardcoded paths. The `orbexa` Python package is mapped to the physical `scripts/` directory; keep core behavior there and use focused modules with explicit interfaces.
- Generalize algorithms around explicit interfaces for dynamics, constraints, solvers, estimation, tube propagation, and mission orchestration. Avoid baking one paper experiment, one solver, or one target geometry into shared code.
- Prefer object-oriented design for project-level concepts: missions, controllers, solvers, estimators, constraints, renderers, demos, and datasets should be classes or dataclasses when they carry state or policy. Keep small stateless math helpers as functions.
- Generated plots, simulation outputs, caches, and local data should stay out of commits. Clean ignored artifacts periodically and only with explicit intent.
- Figures are generated results. Write them into `results/<topic>/` from recorded data. Do not track figure images, paper PDFs or other generated media.
- Keep `scripts/` as the `orbexa` package and `run.py`/`serve.py` outside it, matching the owner's ROS 2 package layout.
- Follow the layer order in `docs/superpowers/specs/2026-11-24-restructure-design.md`. Imports only go downward. Package `__init__` files hold no eager re-exports, so callers import from the defining module.

## Code Standards

- Keep configuration structured and injectable. Avoid global parameter modules and hidden runtime state.
- Maintain solver boundaries: problem formulation should stay separate from GEKKO/CasADi/SciPy backend details.
- Keep the nonlinear IPOPT/GEKKO path available for nonconvex collision and docking constraints. Linearized or QP-friendly constraints may be added as optional approximations, but should not silently replace the nonlinear formulation.
- Preserve user work in the tree. Do not revert unrelated changes or delete outputs unless the user explicitly asks.
- Use subagents for broad discovery or parallel review when the task is wide enough to benefit from them, then integrate the findings into the main implementation.

## Repository maintenance

- The maintainer does not keep test or validation infrastructure in this repository. Do not add or run it.
- Preserve paper scope and runtime solver boundaries when simplifying the source tree.

## Owner rules

These come from the owner's standing instructions. The sections above win where they differ.

### Workflow

- Work end to end without asking: edit, check, commit, open a pull request, squash-merge it into `master` and delete the branch. Keep only `master`. Ask only when a decision belongs to the owner.
- Commits name Aaron John Sabu as author and carry a co-author trailer for the assistant that made the change. Never print or share the owner's email address.
- A pull request description states what changed, the evidence and what is not shown. Report software checks (ruff, imports, wheel) and simulation evidence (named runs under `results/`) separately. Do not call an unmeasured value known. Say plainly what was not run.
- Never force-push and never loosen a check to get a merge through.

### Code

- Behavior lives in classes. A callable is a small class with `__call__` and a module-level instance of the same name. Small stateless math helpers stay functions, as Project Direction says.
- Keep it lean: no filler, dead code, speculative options or comments that restate code. Add no gate that does not protect real data or block real progress.
- One job per class. Keep pure logic separate from solver backends and file I/O.
- Format with `ruff format` and lint with `ruff check`, using the 88-column settings in `pyproject.toml`.
- Name each signal, measurement and operation directly. Give physical values their units in names or docstrings (m, m/s, rad, s, kg, N).
- Refactors must not change behavior. Compare every string constant, message and regex with the old version, and keep decorators and exact conditions.

### Documentation

- Every source file starts with a header docstring. Its first line names the file and its responsibility, and its last line is `Copyright: Aaron John Sabu`.
- Every class, function and method has a factual docstring written from the code. Add `Inputs:` and `Outputs:` when the signature does not state values, units, state changes or failures.
- The root README uses these sections in order: Overview, Responsibilities, Requirements, Installation, Usage, Validation, File Structure, License. Validation describes recorded evidence; it does not add test infrastructure.
- Every maintained folder has a README with a File Structure table of its immediate entries. Columns: File, Relevance, Dependencies, Used by.
- Update the guide a change affects.

