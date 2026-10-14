"""Plotly views of saved paths, safety histories and matched tube geometry."""
from __future__ import annotations

from html import escape
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from orbexa.visualization.geometry import MissionGeometry
from orbexa.visualization.samples import MissionSamples, recorded_array
from orbexa.visualization.style import chaser_color


class MissionInteractive:
    REQUIRED_FIELDS = {
        "trajectory.html": ("anom_history", "actual_trajectories"),
        "diagnostics.html": ("anom_history", "at least one saved tube/control/margin/spacing history"),
        "tube_trajectory.html": ("anom_history", "actual_trajectories", "tube_diagnostics.<chaser>.nominal_center/omega"),
    }

    def __init__(self, result, output_dir: Path, *, samples=None, geometry=None):
        self.result, self.output_dir = result, Path(output_dir)
        self.samples = samples if samples is not None else MissionSamples(result)
        self.geometry = geometry if geometry is not None else MissionGeometry(self.samples)
        self.artifacts, self.skipped_artifacts = [], {}

    def render(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts, self.skipped_artifacts = [], {}
        self.trajectory()
        self.diagnostics()
        self.tube_trajectory()
        return list(self.artifacts)

    def _write(self, fig, filename, label):
        path = self.output_dir / filename
        html = fig.to_html(full_html=True, include_plotlyjs=True, auto_play=False)
        notes = " ".join(self.samples.notes)
        if self.geometry.cylinder_mesh(max(0, len(self.samples.q) - 1)) is None:
            notes += " Saved target geometry or pose unavailable; the target is omitted."
        if notes:
            html = html.replace("<body>", '<body><p style="font:14px system-ui;padding:12px;color:#526171">' + escape(notes) + "</p>", 1)
        path.write_text(html, encoding="utf-8")
        self.artifacts.append((path, label))

    def _paths(self, fig, *, horizons=True):
        for number, identifier in enumerate(self.samples.ids):
            positions, color = self.samples.states[:, number, :3], chaser_color(number)
            fig.add_trace(go.Scatter3d(x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
                                      mode="lines+markers", name=f"{identifier} actual",
                                      line=dict(color=color, width=4), marker=dict(color=color, size=2)))
            if horizons:
                for horizon, nominal in enumerate(self.samples.nominal_horizons(identifier)):
                    fig.add_trace(go.Scatter3d(x=nominal[:, 0], y=nominal[:, 1], z=nominal[:, 2], mode="lines",
                                              name=f"{identifier} nominal horizons", showlegend=horizon == 0,
                                              opacity=0.22, line=dict(color=color, dash="dot")))

    def _geometry(self, fig, *, tube=False):
        final = len(self.samples.q) - 1
        cylinder = self.geometry.cylinder_mesh(final)
        if cylinder is not None:
            fig.add_trace(go.Surface(x=cylinder[0], y=cylinder[1], z=cylinder[2], name="rotating target cylinder",
                                     showscale=False, opacity=0.32 if tube else 0.35,
                                     colorscale=[[0, "#7B8A99"], [1, "#7B8A99"]]))
        sphere = self.geometry.sphere_mesh()
        if sphere is not None:
            fig.add_trace(go.Surface(x=sphere[0], y=sphere[1], z=sphere[2], name="rendezvous sphere",
                                     showscale=False, opacity=0.08 if tube else 0.12,
                                     colorscale=[[0, "#0072B2"], [1, "#0072B2"]]))
        goals = self.geometry.docking_points(final)
        if goals:
            values = np.asarray(list(goals.values()))
            fig.add_trace(go.Scatter3d(x=values[:, 0], y=values[:, 1], z=values[:, 2], mode="markers",
                                       name="docking points", marker=dict(size=5, color="#D55E00", symbol="diamond")))

    def trajectory(self):
        if not len(self.samples.q):
            self.skipped_artifacts["trajectory.html"] = "No complete usable recorded state trajectories; see saved-data insight notes."
            return
        fig = go.Figure()
        self._paths(fig)
        self._geometry(fig)
        for index in self.samples.phase_switch_indices():
            positions = self.samples.states[index, :, :3]
            fig.add_trace(go.Scatter3d(x=positions[:, 0], y=positions[:, 1], z=positions[:, 2], mode="markers",
                                       name="rendezvous-to-docking switch", marker=dict(size=6, color="#D55E00", symbol="x")))
        fig.update_layout(title=f"ORBEX-A ADTMPC Mission ({self.result.mission}, {self.result.approximation})",
                          scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"), margin=dict(l=0, r=0, t=40, b=0))
        self._write(fig, "trajectory.html", "Interactive trajectory with cylinder target, rendezvous sphere, docking points, and phase switch markers.")

    def diagnostics(self):
        q = recorded_array(self.result.anom_history).reshape(-1)
        fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.035,
                            subplot_titles=("Tube radius", "Control norm", "Physical target margins", "Tube-tightened target margins", "Pairwise spacing"))
        def series(row, label, values):
            values = recorded_array(values)
            if values.ndim != 1:
                return
            count = min(len(q), len(values))
            if count and np.any(np.isfinite(q[:count]) & np.isfinite(values[:count])):
                fig.add_trace(go.Scatter(x=q[:count], y=values[:count], mode="lines", name=label), row=row, col=1)
        series(1, "tube max_tube", self.result.tube_radius_history)
        for identifier in self.samples.ids:
            matched = self.samples.matched_tube(identifier)
            if matched is not None:
                series(1, f"tube {identifier}", np.linalg.norm(matched[1], axis=1))
        for identifier, values in self.result.controls.items():
            values = recorded_array(values)
            if values.ndim == 2 and values.shape[1] >= 3:
                series(2, f"{identifier} control", np.linalg.norm(values[:, :3], axis=1))
        for row, prefix, field in (
            (3, "physical rendezvous", "rendezvous_margins"), (3, "physical cylinder", "docking_cylinder_margins"),
            (3, "active physical", "active_target_margins"), (4, "tightened rendezvous", "tightened_rendezvous_margins"),
            (4, "tightened cylinder", "tightened_docking_cylinder_margins"),
        ):
            for identifier, values in getattr(self.result, field).items():
                series(row, f"{prefix} {identifier}", values)
        spacing = self.result.pairwise_spacing or []
        keys = sorted({key for item in spacing if isinstance(item, dict) for key in item})
        for key in keys:
            series(5, f"spacing {key}", [item.get(key, np.nan) if isinstance(item, dict) else np.nan for item in spacing])
        if not len(fig.data):
            self.skipped_artifacts["diagnostics.html"] = "No usable aligned saved tube, control, safety or spacing histories are available."
            return
        for row in (3, 4):
            fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.55, row=row, col=1)
        for index in self.samples.phase_switch_indices():
            for row in range(1, 6):
                fig.add_vline(x=float(self.samples.q[index]), line_dash="dot", line_color="#D55E00", opacity=0.45, row=row, col=1)
        fig.update_layout(title=f"ADTMPC diagnostics ({self.result.mission}, {self.result.approximation})", height=1050,
                          hovermode="x unified", margin=dict(l=40, r=20, t=70, b=40))
        fig.update_xaxes(title_text="true anomaly", row=5, col=1)
        self._write(fig, "diagnostics.html", "Interactive safety, tube, control, and spacing diagnostics.")

    def tube_trajectory(self):
        matched = {identifier: self.samples.matched_tube(identifier) for identifier in self.samples.ids}
        usable = {identifier: values for identifier, values in matched.items() if values is not None}
        if not usable:
            self.skipped_artifacts["tube_trajectory.html"] = "No aligned per-chaser matched nominal centers and omega were recorded; fleet-maximum tubes around actual states are not a matched-tube substitute."
            return
        fig = go.Figure()
        self._paths(fig)
        for identifier, (centers, omega) in usable.items():
            color = chaser_color(self.samples.ids.index(identifier))
            fig.add_trace(go.Scatter3d(x=centers[:, 0], y=centers[:, 1], z=centers[:, 2], mode="lines",
                                       name=f"{identifier} matched nominal center", line=dict(color=color, width=3, dash="dash")))
            indices = self.samples.frame_indices(12)
            for number, index in enumerate(indices):
                vertices, faces = self.geometry.tube_box_mesh(centers[index], omega[index])
                fig.add_trace(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                                        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], color=color, opacity=0.10,
                                        name=f"{identifier} matched omega box", showlegend=number == 0,
                                        hovertemplate=f"{escape(identifier)} · recorded sample {index}<br>own Ω = {omega[index].tolist()}<extra></extra>"))
        self._geometry(fig, tube=True)
        missing = [identifier for identifier, values in matched.items() if values is None]
        note = "Boxes use each chaser's own recorded component bounds around its matched nominal center."
        if missing:
            note += " Matched tube geometry unavailable for: " + ", ".join(missing) + "."
        fig.add_annotation(text=escape(note), x=0.5, y=0, xref="paper", yref="paper", showarrow=False)
        fig.update_layout(title=f"ADTMPC tube trajectory ({self.result.mission}, {self.result.approximation})",
                          scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"), margin=dict(l=0, r=0, t=40, b=50))
        self._write(fig, "tube_trajectory.html", "Interactive trajectory with actual path, nominal horizons, target geometry, and sparse tube envelopes.")
