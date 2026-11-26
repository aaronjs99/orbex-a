# Swarm CLF-CBF Playground

Selected algorithmic ideas and implementation structure were adapted from
[GRANDE-Robots/swarm-clf-cbf-playground](https://github.com/GRANDE-Robots/swarm-clf-cbf-playground),
master `07073759b14e2afef96f744362b5a78577020d5f` and the unmerged
`feature/rl-cbf` head `1324b890cda49d664b797413cdb1bb95a6a85fad`.

Earlier ORBEX-A versions adapted barrier, command-projection, residual-GP,
consensus, connectivity, trajectory-hint and policy-training modules from that
source. Those modules are no longer in the tree. `solvers/proposals.py` and
`solvers/proposal_seed.py` keep only the affine-prediction and proposal-seeding
interface, which retains structure adapted from that source. ORBEX-A supplies
its own dynamics, units, solver boundary, physical admission and mission
orchestration.

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
