# Legacy simulation sources

This directory retains useful original material from `vectr-ucla/mas_space_sim`,
superproject commit `080cb1f561be39ba01e591495501165413c4332c`, consolidated on 2026-10-13.
The modular runtime remains in `src/orbexa`. Source correspondence establishes
this as an older research precursor; Git ancestry with ORBEX-A was not established.

- `manuscripts/tumbling-capture-atmpc/`: original paper source, class, bibliography,
  all chapter source, and the complete figure dependencies of `main.tex`.
- `manuscripts/uclamsthesis/`: original thesis source, class/style files,
  bibliography, and the complete figure dependencies of `main.tex`.
- `diagrams/`: editable mission, geometry, architecture and SMID diagrams, including
  the locally modified `smid.drawio`, with original reference images.
- `source/`: six historical experiment/prototype sources and their original parameters.
  These are references, retain original flat-module imports, and are not package entrypoints.
  Restore the full original checkout from the recovery archive to investigate them.

The manuscript dependency check found every active include, bibliography, class/style
file and figure. It did not compile either manuscript. Template/example documents
with missing placeholders and generated manuscript PDFs were not promoted.
Prototype consensus and separation code is unfinished and is not evidence of a
working distributed mission. Original results and plots are historical data.

Paper submodule: `623389960643ffc28bd4055981ce16544cf14dd1`, with the local
`assets/smid.png` edit preserved. Thesis submodule:
`6664404e9c85c5fcb35ef49d3b2b1f2bbe34c4ed`, with the local
`assets/target-cylinder.png` edit preserved. The thesis stash
`520b86c19d6806d5a69f21c9fa3225198b0e3e39` is retained in the recovery archive;
it contains an alternative SMID image and image deletions and was not applied.

## Full recovery

Archive: `C:\Users\aaron\Documents\Codex\2026-10-13\c-users-aaron-desktop-aaronjs99-orbex\outputs\mas_space_sim-recovery-2026-10-13.zip`

Archive SHA-256: `7240c598936b2d5aa0afbc4caabd67e48cd4a1d850bdbaabd7597750c94ae509`

The verified archive contains all 2,399 files from the original
working tree, including `.git`, both submodules' objects/refs/reflogs, the stash,
original outputs, and deleted tracked diagrams recoverable from Git history.
The adjacent `.manifest.json` records every original file's SHA-256. Extract the ZIP
to a separate folder to recover the complete checkout and relative submodule Git pointers.
Nothing from the archive was applied over existing ORBEX-A runtime code.
