"""Reproducible teaching figures, explicitly separate from mission evidence."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import numpy as np


@dataclass(frozen=True)
class ControlFoundationsFigures:
    """Illustrative geometry plus measured GP/ADMM component calculations.

    No simulated mission completion, physical safety, or timing claim is made.
    Generated assets belong in an ignored or explicitly supplied output folder.
    """
    dpi: int = 180

    def export(self, directory):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle
        from orbexa.control.barrier import BrakingEnvelope
        from orbexa.control.command_projection import project_command
        from orbexa.estimation.residual_gp import ResidualGaussianProcess
        from orbexa.planning.consensus import ConsensusADMM

        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        colors = dict(ink='#142b45', teal='#078b87', red='#c44852', gold='#bc7b16')
        files = []
        with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10,
                             'axes.spines.top': False, 'axes.spines.right': False,
                             'axes.labelcolor': colors['ink'], 'text.color': colors['ink']}):
            figure, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
            speed = np.linspace(0, 3, 200)
            gaps = [BrakingEnvelope.evaluate(0, v, 1, adverse_acceleration_m_s2=.2,
                                             latency_s=.4).required_gap_m for v in speed]
            axes[0].fill_between(speed, gaps, 10, color=colors['teal'], alpha=.12)
            axes[0].plot(speed, gaps, color=colors['teal'], lw=2, label='Required gap with delay')
            axes[0].plot(speed, speed**2/2, color=colors['ink'], ls='--', label='Ideal braking only')
            axes[0].set(xlabel='Closing speed [m/s]', ylabel='Surface gap [m]', ylim=(0, 10),
                        title='1D braking viability\n$ a_{away}=1$, $a_w=0.2$ m/s²; delay 0.4 s')
            axes[0].legend(frameon=False, fontsize=8)
            qp = project_command([.9, .7], [[1, 0]], [.25], [-1, -1], [1, 1])
            if not qp.accepted:
                raise ValueError('Illustrative QP did not pass numerical readback')
            axes[1].add_patch(Rectangle((-1, -1), 1.25, 2, color=colors['teal'], alpha=.12))
            axes[1].add_patch(Rectangle((-1, -1), 2, 2, fill=False, color=colors['ink']))
            axes[1].axvline(.25, color=colors['teal'], lw=2)
            axes[1].scatter(.9, .7, color=colors['red'], label='Nominal preference')
            axes[1].scatter(*qp.command, color=colors['teal'], marker='s', label='Hard-row QP proposal')
            axes[1].annotate('', xy=qp.command, xytext=(.9, .7),
                             arrowprops=dict(arrowstyle='->', color=colors['ink']))
            axes[1].set(xlabel=r'$u_x$ [m/s²]', ylabel=r'$u_y$ [m/s²]',
                        xlim=(-1.15, 1.15), ylim=(-1.15, 1.15), title='Projection in command space')
            axes[1].set_aspect('equal'); axes[1].legend(frameon=False, fontsize=8, loc='lower left')
            for center, color, label in [((0, .1), colors['teal'], 'Whole tube fits'),
                                          ((.85, -.15), colors['red'], 'Nominal fits; tube does not')]:
                axes[2].add_patch(Circle(center, .34, color=color, alpha=.18))
                axes[2].add_patch(Circle(center, .34, fill=False, color=color, lw=2))
                axes[2].plot(*center, marker='o', color=color, label=label)
            axes[2].add_patch(Rectangle((-1, -.6), 2, 1.2, fill=False, color=colors['ink'], lw=2))
            axes[2].set(xlabel='Patch-relative transverse error [m]', ylabel='Patch-relative axial error [m]',
                        xlim=(-1.4, 1.4), ylim=(-.9, .9), title='Capture set must contain the tube')
            axes[2].set_aspect('equal'); axes[2].legend(frameon=False, fontsize=8, loc='lower left')
            for axis in axes: axis.grid(alpha=.15)
            figure.suptitle('Illustrative local geometry — no orbital mission result', fontweight='bold')
            figure.savefig(output/'geometry.png', dpi=self.dpi); plt.close(figure)
            files.append('geometry.png')

            gp = ResidualGaussianProcess(1, 1, capacity=32, length_scale=.6, amplitude=.8, noise=.08)
            x = np.linspace(-1.5, 1.5, 10)
            y = .4*np.sin(2*x)+.03*np.cos(7*x)
            for xi, yi in zip(x, y): gp.add([xi], [yi])
            gp.fit(); queries = np.linspace(-2.5, 2.5, 160)
            posterior = [gp.predict([q]) for q in queries]
            mean = np.array([p.mean[0] for p in posterior])
            radius = np.array([p.projected_radius([1], beta=2) for p in posterior])
            centers = [np.array([2., -1.]), np.array([-2., 5.])]
            prox = [lambda v, rho, c=c: (c+rho*v)/(1+rho) for c in centers]
            consensus = ConsensusADMM(absolute_tolerance=1e-9, relative_tolerance=1e-9).solve(prox, np.zeros(2))
            figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
            axes[0].fill_between(queries, mean-radius, mean+radius, color=colors['teal'], alpha=.18,
                                 label='Posterior ±2σ (statistical)')
            axes[0].plot(queries, mean, color=colors['teal'], label='GP mean')
            axes[0].scatter(x, y, color=colors['ink'], s=18, label='Illustrative observations')
            axes[0].set(xlabel='Normalized feature', ylabel='Residual acceleration [m/s²]',
                        title='Unobserved regions retain uncertainty')
            history = np.asarray(consensus.residual_history)
            axes[1].semilogy(np.arange(1, len(history)+1), history[:, 0], color=colors['teal'], label='Primal residual')
            axes[1].semilogy(np.arange(1, len(history)+1), history[:, 1], color=colors['gold'], label='Dual residual')
            axes[1].set(xlabel='Consensus iteration', ylabel='Residual norm',
                        title=f'Actual convex component solve: {consensus.iterations} iterations')
            for axis in axes: axis.grid(alpha=.15); axis.legend(frameon=False, fontsize=8)
            figure.suptitle('Statistical learning and numerical coordination — neither certifies flight safety', fontsize=11)
            figure.savefig(output/'learning.png', dpi=self.dpi); plt.close(figure)
            files.append('learning.png')

            from scipy.linalg import expm
            t = np.linspace(0, 20, 401); dt = t[1]-t[0]
            figure, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
            for label, alpha, color in [('α = 1 /s', 1., colors['ink']), ('α = 2 /s', 2., colors['teal'])]:
                state = np.array([.12, .08]); values = [state.copy()]
                # State order [Omega, Phi], lambda=1 /s; D=0.02 m/s².
                augmented = np.array([[-1., 1., 0.], [0., -alpha, .02], [0., 0., 0.]])
                flow = expm(augmented*dt)
                for _ in t[1:]: state = (flow @ np.r_[state, 1.])[:2]; values.append(state.copy())
                values = np.asarray(values)
                axes[0].plot(t, values[:, 0], color=color, label=label)
                axes[1].plot(t, alpha*values[:, 1], color=color, label=label)
            axes[0].set(xlabel='Physical time [s]', ylabel='Position radius Ω [m]', title='Ideal continuous comparison')
            axes[1].set(xlabel='Physical time [s]', ylabel='αΦ [m/s²]', title='Bandwidth also demands feedback effort')
            for axis in axes: axis.grid(alpha=.15); axis.legend(frameon=False)
            figure.suptitle('Illustrative constant forcing — held sensing terms and full reserve omitted', fontsize=11)
            figure.savefig(output/'tube.png', dpi=self.dpi); plt.close(figure)
            files.append('tube.png')

        record = dict(schema='orbexa.control-foundations-figures.v1',
                      scope='Teaching examples and component calculations; no mission qualification',
                      consensus=dict(iterations=consensus.iterations, optimum=consensus.consensus.tolist()),
                      files=[dict(path=f, sha256=hashlib.sha256((output/f).read_bytes()).hexdigest()) for f in files])
        (output/'figures.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
        panels = ''.join(f'<figure><img src="{name}" alt="{name[:-4]} teaching figure"></figure>' for name in files)
        (output/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1"><title>ORBEX-A · Control foundations</title>'
            '<style>body{max-width:1280px;margin:40px auto;padding:0 24px;background:#f4f7fa;color:#142b45;'
            'font:16px system-ui}h1{font-size:28px}p{color:#486078}figure{margin:24px 0;background:white;'
            'border-radius:12px;overflow:hidden;border:1px solid #dde5ed}img{width:100%;display:block}</style>'
            '<h1>ORBEX-A · Control foundations</h1><p>Teaching examples and actual component calculations. '
            'These figures do not depict a completed spacecraft mission.</p>'+panels+'</html>', encoding='utf-8')
        return record
