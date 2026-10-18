# Swarm CLF-CBF Playground

Selected algorithmic ideas and implementation structure were adapted from
[GRANDE-Robots/swarm-clf-cbf-playground](https://github.com/GRANDE-Robots/swarm-clf-cbf-playground),
master `07073759b14e2afef96f744362b5a78577020d5f` and the unmerged
`feature/rl-cbf` head `1324b890cda49d664b797413cdb1bb95a6a85fad`.

Affected modules: `control/barrier.py`, `control/command_projection.py`,
`estimation/residual_gp.py`, `planning/consensus.py`, `planning/connectivity.py`,
`planning/trajectory_hints.py`, `planning/ppo_proposal.py`,
`planning/policy_features.py` and `planning/training.py` under `scripts/`.
ORBEX-A supplies its own dynamics, units, mathematical corrections, solver
boundary, physical admission and mission orchestration. See
[the incorporation record](docs/swarm-consolidation.md).

The source license is reproduced below. It does not change ORBEX-A's project license.

```text
MIT License

Copyright (c) 2026 Aaron John Sabu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
