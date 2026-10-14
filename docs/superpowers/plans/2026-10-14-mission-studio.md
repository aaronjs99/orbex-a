# Mission Studio Implementation Plan

> **For agentic workers:** Use executing-plans for the presentation, and independent subagents for broad theory and system reviews. Keep all executable examples outside this repository.

**Goal:** Build a numerical 3D replay workspace with synchronized measurements, media and accessible theory.

**Architecture:** Export `MissionSamples` into a portable plain dataset. A static modular browser application owns display and interaction only; recorded outcomes and raw results remain authoritative.

**Tech Stack:** Python 3.10+, NumPy, static HTML/CSS/ES modules, pinned Three.js, SVG charts, native HTML video.

**Spec:** `docs/superpowers/specs/2026-10-14-mission-studio-design.md`

## Global constraints

- Preserve the physical `scripts/` and conference-asset relocations.
- Do not add or run repository test infrastructure.
- Keep generated outputs and executable examples outside the checkout.
- Retain recorded sample identities and own-chaser tube geometry.
- Serve local web assets without runtime third-party requests.

## Task 1: Saved-data interface

- [x] Add `StudioDataset(result).to_dict()` in `scripts/visualization/studio_data.py`.
- [x] Require usable finite states and a strictly increasing anomaly history;
  share alignment and reconstruction notes with `MissionSamples`.
- [x] Project actual/nominal states, target rotations, docking points, diagnostic
  histories, elapsed time and outcome; label sample vs interval metrics.
- [x] Add `MissionStudio(output_dir).export(entries, *, theory_documents)` in
  `scripts/visualization/mission_studio.py`, accepting explicit result records
  and media links within its output root. Copy assets and emit catalog JSON.
- [x] Run outside saved-data examples and compare arrays exactly with the source.

## Task 2: Browser workspace

- [x] Implement the semantic shell and shared design tokens in `studio/`.
- [x] Add `MissionScene` for numerical target/chaser/tube geometry and camera
  lifecycle, disposing old scenarios before replacement.
- [x] Add `MetricChart` for labeled source-derived curves and a selected-sample
  cursor. Handle zeros/missing fields without fabricated measurements.
- [x] Add one recorded-sample clock, scenario/chaser selection, seek/step/play,
  paths/tubes switches, camera presets, fullscreen and raw-data download.
- [x] Add native video selection/playback plus evidence and rendered theory views.
- [x] Package all static assets and the pinned renderer's license.

## Task 3: Delivery

- [x] Export current qualified cases into the existing loopback gallery root,
  preserve the evidence gallery and show Mission Studio at the primary URL.
- [x] Verify actual browser interaction and desktop/mobile views; compare final
  screenshot and concept, recording scientific deviations in an outside ledger.
- [ ] Rebuild the wheel and inspect package assets/imports. Refresh the offline
  bundle, measured report and PR description after final counts are known.
- [ ] Commit and push the reviewed source and documentation; retain generated
  media outside Git.
