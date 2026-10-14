"""Self-contained animated Plotly replay of recorded mission states."""

from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from orbexa.simulation.adtmpc_mission import ADTMPCMissionResult
from orbexa.visualization.mission_insights import MissionSamples, chaser_color, chaser_label


class MissionReplay:
    """Replay saved samples without constructing a mission runner or scenario.

    Up to 128 evenly spaced recorded states, plus phase transitions, are animation
    stops. Full recorded trail prefixes and the final sample remain visible.
    Playback is sample-based, with no invented intermediate motion.
    """

    def __init__(self, result: ADTMPCMissionResult):
        self.result = result
        self.samples = MissionSamples(result)

    def _traces(self, index: int):
        s = self.samples
        traces = []
        for number, identifier in enumerate(s.ids):
            points = s.states[:index + 1, number, :3]
            traces.append(go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="lines",
                line=dict(color=chaser_color(number), width=5), name=escape(chaser_label(identifier)),
                hovertemplate="x %{x:.4g}<br>y %{y:.4g}<br>z %{z:.4g}<extra>%{fullData.name}</extra>",
            ))
        points = s.states[index, :, :3]
        traces.append(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers",
            marker=dict(size=6, color=[chaser_color(number) for number in range(len(s.ids))]),
            text=[escape(chaser_label(identifier)) for identifier in s.ids], name="Recorded positions", showlegend=False,
            hovertemplate="%{text}<br>x %{x:.4g}<br>y %{y:.4g}<br>z %{z:.4g}<extra></extra>",
        ))
        geometry = s.cylinder()
        if geometry is not None and s.rotations is not None:
            radius, half_length = geometry
            angle, heights = np.meshgrid(np.linspace(0, 2 * np.pi, 36), [-half_length, half_length])
            cylinder = np.stack((radius * np.cos(angle), radius * np.sin(angle), heights))
            surface = (s.rotations[index] @ cylinder.reshape(3, -1)).reshape(cylinder.shape)
            traces.append(go.Surface(
                x=surface[0], y=surface[1], z=surface[2], opacity=0.30,
                colorscale=[[0, "#7B8A99"], [1, "#7B8A99"]], showscale=False, name="Saved target exclusion boundary", hoverinfo="skip",
            ))
        if s.goals_body and s.rotations is not None:
            identifiers = [identifier for identifier in s.ids if identifier in s.goals_body]
            goals = np.array([s.rotations[index] @ s.goals_body[identifier] for identifier in identifiers])
            traces.append(go.Scatter3d(
                x=goals[:, 0], y=goals[:, 1], z=goals[:, 2], mode="markers", name="Assigned docking points",
                marker=dict(size=4, color=[chaser_color(s.ids.index(identifier)) for identifier in identifiers], symbol="diamond"),
                text=[escape(chaser_label(identifier)) + " goal" for identifier in identifiers], hovertemplate="%{text}<extra></extra>",
            ))
        return traces

    def _title(self, index: int) -> str:
        s = self.samples
        phase = str(s.phases[index]).capitalize() if s.phases[index] != "unknown" else "Phase unavailable"
        return "{} · Recorded sample {} / {} · {} · q = {:.4f} rad".format(escape(s.title), index + 1, len(s.q), escape(phase), s.q[index])

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        s = self.samples
        count = len(s.q)
        fig = go.Figure()
        selected = []
        if count:
            selected = np.unique(np.linspace(0, count - 1, min(128, count)).astype(int)).tolist()
            selected = sorted(set(selected + [index for index in range(1, count) if s.phases[index] != s.phases[index - 1]]))
            fig.add_traces(self._traces(0))
            if count > 1:
                fig.frames = [go.Frame(name=str(index), data=self._traces(index), layout=dict(title=dict(text=self._title(index)))) for index in selected]
            finite = np.abs(s.states[:, :, :3])
            finite = finite[np.isfinite(finite)]
            geometry = s.cylinder()
            limit = max(float(np.max(finite)) if len(finite) else 0.0, max(geometry) if geometry else 0.0, 0.1) * 1.12
            fig.update_layout(title=self._title(0), scene=dict(
                xaxis=dict(title="LVLH x", range=[-limit, limit]),
                yaxis=dict(title="LVLH y", range=[-limit, limit]),
                zaxis=dict(title="LVLH z", range=[-limit, limit]), aspectmode="cube",
                camera=dict(eye=dict(x=1.45, y=1.45, z=0.9)),
            ))
        else:
            fig.update_layout(title="Recorded mission replay unavailable")
            fig.add_annotation(text="No aligned recorded trajectory samples are available.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        fig.update_layout(
            template="plotly_white", height=700, margin=dict(l=10, r=10, t=100, b=70),
            font=dict(family="Arial, sans-serif", color="#22303A"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), uirevision="saved-mission-replay",
        )
        if count > 1:
            immediate = dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))
            fig.update_layout(
                updatemenus=[dict(type="buttons", direction="left", x=0, y=-0.06, showactive=False, buttons=[
                    dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=100, redraw=True), transition=dict(duration=0), fromcurrent=True, mode="immediate")]),
                    dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
                ])],
                sliders=[dict(active=0, x=0.20, len=0.78, y=-0.05, currentvalue=dict(prefix="Recorded sample index "), steps=[
                    dict(label=str(index), method="animate", args=[[str(index)], immediate]) for index in selected
                ])],
            )
        status = "Saved mission criteria passed." if self.result.success else "Saved mission criteria did not all pass. " + str(self.result.message or "")
        playback = "{} of {} recorded samples are replay stops; trails include intervening recorded states. No interpolation. Playback is not physical time.".format(len(selected), count)
        if count == 1:
            playback = "One recorded state; no motion was recorded."
        notes = status + " " + playback + " Positions use saved simulation coordinates. " + " ".join(s.notes)
        html = fig.to_html(full_html=True, include_plotlyjs=True, auto_play=False, config=dict(responsive=True, displaylogo=False))
        banner = '<div style="font:14px/1.5 system-ui,sans-serif;color:#526171;padding:16px 22px;background:#f2f5f8;border-bottom:1px solid #dce3e9">' + escape(notes) + '</div>'
        html = html.replace("<body>", "<body>" + banner, 1)
        path.write_text(html, encoding="utf-8")
        return path
