# Mission Studio design

## Purpose

Present recorded ORBEX-A missions as a professional simulation replay workspace,
with direct access to videos, measurements, raw evidence and theory. The current
saved-data gallery remains available as the evidence library. The viewer does
not run a controller or generate scientific states in the browser.

## Information and rendering contract

- The source is an `ADTMPCMissionResult`, aligned through `MissionSamples`.
- A single selected recorded sample controls target orientation, chaser
  positions, nominal centers, componentwise tube boxes, metrics and chart cursors.
- Playback advances among recorded samples. It does not invent interpolated
  states or turn display smoothing into additional safety evidence.
- Physical geometry stays in meters. Chaser glyphs have no inferred physical
  attitude or volume; a declared finite body is displayed only from recorded
  safety metadata. Tube boxes use each chaser's own nominal center and Omega.
- Missing optional diagnostics display as unavailable. Unequal tracks stop at
  the shared recorded sample count and retain an explanatory note.
- Completed, incomplete and failed missions have distinct visible outcomes.
- Video playback has its own native transport and is explicitly separate from
  the recorded-state timeline. A video is not relabeled as a live simulation.
- Export is static and works without external requests when served locally.
  The original offline gallery remains usable from the distributed bundle.

## Interface

Use the complete Mission Studio concept generated on 2026-10-14 as the visual
reference: dark navy workspace, restrained teal accent, large 3D canvas, narrow
inspector rail, one transport band and three synchronized charts. The top-level
views are Replay, Videos, Evidence and Theory. Scenario and chaser selection,
camera presets, paths/tubes visibility, playback, stepping, seeking, full screen,
and downloads must work. Keyboard controls and mobile layouts remain legible.

Intentional scientific corrections to concept artwork: use actual saved labels,
timestamps only when recorded, actual sample count, actual anomaly domain and
actual metric values. The cylinder remains the numerical collision envelope.
A representative detailed spacecraft is contained inside it; Envelope exposes
the distinction. The user's later black-space request supersedes the concept's
visible navy grid by default. No generated bitmap represents a simulation result.

## Ownership

- `studio_data.py`: plain saved-data projection and derived metric semantics.
- `mission_studio.py`: catalog, path-safe static export and asset copying.
- `studio/index.html`, `studio.css`: semantic shell and responsive design tokens.
- `studio/app.js`: user interactions and one playback clock.
- `studio/scene.js`: Three.js scene lifecycle, cameras and numerical geometry.
- `studio/charts.js`: sample-aligned quantitative charts.
- `studio/lifecycle.js`: synchronized assembly and Earth-orbit views, recorded
  SI clock, fuel/spin/altitude inspection and a local composite video export.
- `studio/recorder.js`: browser video capture and origin-checked local download.
- Vendored pinned Three.js runtime and MIT license: offline 3D dependency.

The existing Python static-export architecture dictates a static application;
introducing a separate frontend build service is unnecessary. Ship web assets as
package data so wheel users can create the same workspace.

## Verification and limits

Use saved single/fleet/fast cases plus malformed/partial examples outside the
checkout. Compare every exported state, rotation and tube to source samples.
Inspect actual browser playback, scenario switching, seeking, chaser selection,
camera controls, chart synchronization, videos, theory links and narrow layout.
Inspect the concept and final browser screenshots together. Rebuild the wheel
and verify included assets. Do not add repository test infrastructure.

The viewer reports model evidence; it does not certify continuous-time safety,
recursive feasibility, physical capture or flight readiness.
