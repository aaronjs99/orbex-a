# ORBEX-A Agent Rules

## Project Direction

- Keep the implementation anchored to the paper: elliptical relative dynamics with quadratic drag, MPC rendezvous/docking constraints, dynamic tubes, SMID adaptation, and multi-chaser extensions are the priority.
- Prefer modular, reusable interfaces over one-off scripts or hardcoded experiment paths. Core project behavior belongs in `src/orbexa`; scripts should exist only when they are vital project entrypoints.
- Generalize algorithms around explicit interfaces for dynamics, constraints, solvers, estimation, tube propagation, and mission orchestration. Avoid baking one paper experiment, one solver, or one target geometry into shared code.
- Prefer object-oriented design for project-level concepts: missions, controllers, solvers, estimators, constraints, renderers, demos, and datasets should be classes or dataclasses when they carry state or policy. Keep small stateless math helpers as functions.
- Generated plots, simulation outputs, caches, and local data should stay out of commits. Clean ignored artifacts periodically and only with explicit intent.
- Do not track local paper PDFs unless explicitly requested. Keep source paper assets, figures, and implementation code separate from downloaded manuscripts.

## Code Standards

- Keep configuration structured and injectable. Avoid global parameter modules and hidden runtime state.
- Maintain solver boundaries: problem formulation should stay separate from GEKKO/CasADi/SciPy backend details.
- Keep the nonlinear IPOPT/GEKKO path available for nonconvex collision and docking constraints. Linearized or QP-friendly constraints may be added as optional approximations, but should not silently replace the nonlinear formulation.
- Preserve user work in the tree. Do not revert unrelated changes or delete outputs unless the user explicitly asks.
- Use subagents for broad discovery or parallel review when the task is wide enough to benefit from them, then integrate the findings into the main implementation.

## Testing Standards

- Pytests should be concept-level and vital. They should validate paper claims, architecture contracts, and end-to-end behavior on small deterministic scenarios.
- Avoid tests that only check imports, signatures, or every small implementation detail unless that check protects an important public contract.
- Good tests for this project include dynamics equivalence/invariants, collision constraint behavior, tube monotonicity, SMID range shrinkage, solver-agnostic MPC behavior, and multi-chaser assignment/cooperation concepts.
