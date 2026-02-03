# /***********************************************************
# *                                                         *
# * Copyright (c) 2026                                      *
# *                                                         *
# * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
# * University of California, Los Angeles                   *
# *                                                         *
# * Authors: Aaron John Sabu, Brett T. Lopez                *
# * Contact: {aaronjs, btlopez}@ucla.edu                    *
# *                                                         *
# ***********************************************************/

# PACKAGE IMPORTS
import random
import numpy as np
import plotly.offline as ptyplt
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Sequence, List, Tuple, Dict

logger = logging.getLogger(__name__)

from orbexa.utils import (
    create_filename,
    latest_data_file,
    load_data,
    tait_bryan_to_rotation_matrix,
)

# --- Dataclasses ---


@dataclass(frozen=True)
class MPCSeries:
    """Encapsulates time-series data for MPC plotting."""

    t: np.ndarray
    position: np.ndarray  # (N, 3)
    velocity: np.ndarray  # (N, 3)
    control_input: Optional[np.ndarray] = None  # (N, 3)
    pos_ref1: Optional[np.ndarray] = None  # (N, 3)
    pos_ref2: Optional[np.ndarray] = None  # (N, 3)

    def slice(self, start: int) -> "MPCSeries":
        slice_index = slice(start, None)
        return MPCSeries(
            t=self.t[slice_index],
            position=self.position[slice_index],
            velocity=self.velocity[slice_index],
            control_input=(
                self.control_input[slice_index]
                if self.control_input is not None
                else None
            ),
            pos_ref1=self.pos_ref1[slice_index] if self.pos_ref1 is not None else None,
            pos_ref2=self.pos_ref2[slice_index] if self.pos_ref2 is not None else None,
        )


@dataclass(frozen=True)
class PlotFlags:
    """Configuration flags for plotting."""

    plot_act: bool = True
    plot_act_sim: bool = False
    plot_act_con: bool = False
    plot_nom: bool = True
    plot_nom_sim: bool = False
    plot_nom_con: bool = False
    # Deflection flags
    plot_target: bool = False
    plot_position: bool = False
    plot_force: bool = False


@dataclass(frozen=True)
class TargetLimits:
    """Target geometric limits."""

    target_radius: float = 0.0
    target_height: float = 0.0


@dataclass
class MPCPlotConfig:
    """Configuration for MPC plotting."""

    anom_step: float
    time_periapsis: float = 0.0
    dock_index: Optional[int] = None
    target_folder: Optional[Path] = None
    filename_sim: str = "mpc_test"
    target_limits: Optional[TargetLimits] = None

    @classmethod
    def from_kwargs(cls, anom_step: float, **kwargs):
        """Create config from legacy kwargs."""
        tf = kwargs.get("target_folder")
        if tf:
            tf = Path(tf)

        tlimits = None
        if "target_limits" in kwargs:
            tlimits = TargetLimits(
                r_T=kwargs["target_limits"].get("r_T", 0.0),
                l_T=kwargs["target_limits"].get("l_T", 0.0),
            )

        return cls(
            anom_step=anom_step,
            time_periapsis=kwargs.get("time_periapsis", 0.0)
            or kwargs.get("t_periapsis", 0.0),
            dock_index=kwargs.get("dock_index"),
            target_folder=tf,
            filename_sim=kwargs.get("filename_sim", "mpc_test"),
            target_limits=tlimits,
        )


# --- Helper Functions ---


def as_arrays(*seqs):
    """Convert sequence of inputs to numpy arrays."""
    return [np.asarray(s) for s in seqs]


def split_state(state_array: np.ndarray):
    """Split state array into position and velocity components."""
    array = np.asarray(state_array)
    if array.size == 0:
        return np.array([]), np.array([])

    # If state is 1D (N,) where N>=6
    if array.ndim == 1:
        if array.shape[0] >= 6:
            return array[:3], array[3:6]
        return array, np.array([])

    # If state is 2D (N, M)
    if array.ndim == 2:
        if array.shape[1] >= 6:
            return array[:, :3], array[:, 3:6]
        if array.shape[1] >= 3:
            return array[:, :3], np.array([])

    return array, np.array([])


def norms(array: np.ndarray) -> np.ndarray:
    """Compute norms of vectors in array. Expects (N,3) or (3,)."""
    array = np.asarray(array)
    if array.size == 0:
        return np.array([])

    if array.ndim == 1:
        if array.shape != (3,):
            raise ValueError(f"Expected (3,) for single vector, got {array.shape}")
        return np.array([np.linalg.norm(array)])

    if array.shape[1] != 3:
        raise ValueError(f"Expected (N,3) for vectors, got {array.shape}")

    return np.linalg.norm(array, axis=1)


def outpath(folder: Optional[Path], name: str) -> Optional[Path]:
    """Resolve output path, creating directory if needed."""
    if folder is None:
        return None
    folder.mkdir(parents=True, exist_ok=True)
    return folder / name


def sim_html_path(cfg: MPCPlotConfig, suffix: str) -> str:
    """Generate HTML path for simulation."""
    base = cfg.target_folder if cfg.target_folder else Path("../plots")
    base.mkdir(parents=True, exist_ok=True)
    # Maintain legacy naming: filename_sim usually acts as prefix
    # If filename_sim is absolute/relative path string passed by user...
    # But usually it is just a name like "mpc_test".
    return str(base / f"{cfg.filename_sim}{suffix}")


def print_min_max(values, label):
    """Log min/max of sequence."""
    if len(values) > 0:
        logger.info(f"{label}: Min = {np.min(values)}, Max = {np.max(values)}")


# --- Plotting Functions ---


def create_animation_figure(
    x_positions,
    y_positions,
    z_positions,
    labels,
    *,
    lines=False,
    markers=True,
    point_list=None,
    shapes=None,
) -> go.Figure:
    """Create a Plotly 3D animation figure."""
    traces = []
    num_agents = len(labels)

    # Validation
    if not (len(x_positions) == len(y_positions) == len(z_positions) == num_agents):
        # Warn and adjust to minimum length for safety
        min_length = min(
            len(x_positions), len(y_positions), len(z_positions), num_agents
        )
        logger.warning(f"Mismatch in position array lengths, using {min_length} agents")
        num_agents = min_length

    for agent in range(num_agents):
        if lines:
            traces.append(
                go.Scatter3d(
                    mode="lines",
                    x=x_positions[agent],
                    y=y_positions[agent],
                    z=z_positions[agent],
                    marker=dict(size=3),
                    name=labels[agent],
                )
            )
        if markers:
            traces.append(
                go.Scatter3d(
                    mode="markers",
                    x=x_positions[agent],
                    y=y_positions[agent],
                    z=z_positions[agent],
                    marker=dict(size=3, color="darkblue"),
                    name=labels[agent],
                )
            )

    if point_list:
        for i, point in enumerate(point_list):
            traces.append(
                go.Scatter3d(
                    mode="markers",
                    x=[point[0]],
                    y=[point[1]],
                    z=[point[2]],
                    marker=dict(size=6),
                    name=f"P{i+1}",
                )
            )

    if shapes:
        for shape in shapes:
            stype = shape.get("type")
            opacity = shape.get("opacity", 0.5)
            center = shape.get("center", [0, 0, 0])

            x_s, y_s, z_s = [], [], []

            if stype == "sphere":
                radius = shape["radius"]
                u, v = np.mgrid[0 : 2 * np.pi : 100j, 0 : np.pi : 50j]
                x_s = radius * np.cos(u) * np.sin(v) + center[0]
                y_s = radius * np.sin(u) * np.sin(v) + center[1]
                z_s = radius * np.cos(v) + center[2]

            elif stype == "ellipsoid":
                radii = shape["radii"]
                u, v = np.mgrid[0 : 2 * np.pi : 100j, 0 : np.pi : 50j]
                x_s = radii[0] * np.cos(u) * np.sin(v) + center[0]
                y_s = radii[1] * np.sin(u) * np.sin(v) + center[1]
                z_s = radii[2] * np.cos(v) + center[2]

            elif stype == "cylinder":
                radius = shape["radius"]
                length = shape["length"]
                u, v = np.mgrid[
                    0 : 2 * np.pi : 100j,
                    center[2] - length / 2.0 : center[2] + length / 2.0 : 20j,
                ]
                x_s = radius * np.cos(u) + center[0]
                y_s = radius * np.sin(u) + center[1]
                z_s = v

            if len(x_s) > 0:
                traces.append(
                    go.Surface(
                        x=x_s,
                        y=y_s,
                        z=z_s,
                        opacity=opacity,
                        showscale=False,
                    )
                )

    layout = go.Layout(
        font_color="white",
        paper_bgcolor="rgba(72,72,72,255)",
        plot_bgcolor="rgba(185,185,185,255)",
    )
    return go.Figure(data=traces, layout=layout)


def save_plotly_html(fig: go.Figure, filename: str) -> None:
    """Save Plotly figure to HTML."""
    fpath = Path(filename)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    ptyplt.plot(fig, filename=str(fpath), auto_open=False)


def create_animation_html(
    filename, x_positions, y_positions, z_positions, labels, *args, **kwargs
):
    """Legacy wrapper for creating animation HTML."""
    point_list = kwargs.get("point_list")
    shapes = kwargs.get("shape")
    lines = kwargs.get("lines", False)
    markers = kwargs.get("markers", True)

    fig = create_animation_figure(
        x_positions,
        y_positions,
        z_positions,
        labels,
        lines=lines,
        markers=markers,
        point_list=point_list,
        shapes=shapes,
    )
    save_plotly_html(fig, filename)


def plot_time_series(
    t: np.ndarray,
    states: Sequence[np.ndarray],
    control_inputs: Sequence[np.ndarray],
    y_outputs: Sequence[np.ndarray] = (),
    cost_history: Optional[np.ndarray] = None,
    labelX: Sequence[str] = (),
    labelU: Sequence[str] = (),
    labelY: Sequence[str] = (),
    labelC: Optional[str] = None,
    states_ref1: Sequence[np.ndarray] = (),
    labelX1: Sequence[str] = (),
    states_ref2: Sequence[np.ndarray] = (),
    labelX2: Sequence[str] = (),
    save_path: Optional[Path] = None,
    **kwargs,
) -> None:
    """
    Plot time series data.

    Supports legacy kwargs: 'filename', 'fName' for save_path.
    """
    # Legacy path handling
    if save_path is None:
        if "filename" in kwargs:
            save_path = Path(kwargs["filename"])
        elif "fName" in kwargs:
            save_path = Path(kwargs["fName"])

    # Updated plot_time_series to be flexible
    # Determine active components
    has_states = len(states) > 0
    has_inputs = len(control_inputs) > 0
    has_outputs = len(y_outputs) > 0 or (
        cost_history is not None
    )  # cost_history is legacy cost

    # Calculate rows needed by stacking: States (if any), Inputs (if any), Outputs (if any)

    n_state_plots = len(states)
    n_state_cols = 2
    n_state_rows = (n_state_plots + n_state_cols - 1) // n_state_cols

    # Total figure rows
    current_row = 0
    total_rows = n_state_rows + (1 if has_inputs else 0) + (1 if has_outputs else 0)
    if total_rows == 0:
        return

    plt.figure(figsize=(10, 4 * total_rows))
    cmap = plt.get_cmap("jet")

    # 1. State Plots
    for i in range(n_state_plots):
        # Index in total grid
        # We want to place these in the first n_state_rows * 2 slots?
        # Actually simplest is: subplot(total_rows, 2, ...)
        # But inputs/outputs need to span columns.

        # GridSpec is best but sticking to subplots:
        # We can pretend grid is (total_rows, 2)
        # States take (r, 1) and (r, 2) for r in 0..n_state_rows-1

        r_idx = i // 2
        c_idx = i % 2

        # subplot index = row * cols + col + 1
        # row = r_idx + current_row
        plot_idx = ((current_row + r_idx) * 2) + c_idx + 1

        plt.subplot(total_rows, 2, plot_idx)

        label = labelX[i] if i < len(labelX) else f"$x_{i}$"
        plt.plot(t, states[i], c=cmap(80 * i), label=label)

        if i < len(states_ref1):
            l1 = labelX1[i] if i < len(labelX1) else f"$x_{{1,{i}}}$"
            plt.plot(t, states_ref1[i], c=cmap(80 * i + 40), label=l1, linestyle="--")
        if i < len(states_ref2):
            l2 = labelX2[i] if i < len(labelX2) else f"$x_{{2,{i}}}$"
            plt.plot(t, states_ref2[i], c=cmap(80 * i + 80), label=l2, linestyle="-.")

        plt.legend()
        plt.ylabel("State")
        plt.xlabel("Time")

    if has_states:
        current_row += n_state_rows

    # 2. Input Plots
    if has_inputs:
        # Span both columns?
        # subplot(total_rows, 1, current_row + 1)
        # This usually works in matplotlib (mixing grids)
        plt.subplot(total_rows, 1, current_row + 1)

        for i in range(len(control_inputs)):
            u_vec = control_inputs[i]
            # Handle mismatch
            n_u = u_vec.shape[0]
            n_t = t.shape[0]
            t_plot = t
            if n_u < n_t:
                t_plot = t[:n_u]
            elif n_u > n_t:
                u_vec = u_vec[:n_t]

            label = labelU[i] if i < len(labelU) else f"$u_{i}$"
            plt.plot(t_plot, u_vec, c=cmap(150 + 80 * i), label=label)

        plt.legend()
        plt.ylabel("Input")
        plt.xlabel("Anomaly (rad)")
        current_row += 1

    # 3. Output Plots
    if has_outputs:
        plt.subplot(total_rows, 1, current_row + 1)

        for i in range(len(y_outputs)):
            y_vec = y_outputs[i]
            # Handle mismatch
            n_y = y_vec.shape[0]
            n_t = t.shape[0]
            t_plot = t
            if n_y < n_t:
                t_plot = t[:n_y]
            elif n_y > n_t:
                y_vec = y_vec[:n_t]

            label = labelY[i] if i < len(labelY) else f"$y_{i}$"
            plt.plot(t_plot, y_vec, c=cmap(180 + 96 * i), label=label)

        if cost_history is not None and (
            len(cost_history) > 0 if hasattr(cost_history, "__len__") else True
        ):
            label = labelC if labelC else "$c$"
            plt.plot(t, cost_history, c=cmap(180), label=label, linestyle=":")

        plt.legend()
        plt.ylabel("Output/Cost")
        plt.xlabel("Anomaly (rad)")
        current_row += 1

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_mpc(
    act_states,
    act_inputs,
    nom_states,
    nom_inputs,
    fin_states,
    target_thetas=None,
    cfg: Optional[MPCPlotConfig] = None,
    *args,
    **kwargs,
):
    """
    Plot MPC results including trajectories, inputs, and constraints.
    Supports both legacy kwargs and modern MPCPlotConfig.
    """
    # 1. Compatibility Layer
    if cfg is None:
        if anom_step is None:
            anom_step = 0.1
        cfg = MPCPlotConfig.from_kwargs(anom_step, **kwargs)

    if isinstance(plot_flags, dict):
        plot_flags = PlotFlags(**plot_flags)
    elif plot_flags is None:
        plot_flags = PlotFlags()

    # 2. Data Preparation
    # Convert to arrays
    (act_states, act_inputs, nom_states, nom_inputs, fin_states, tgt_states) = (
        as_arrays(
            act_states, act_inputs, nom_states, nom_inputs, fin_states, tgt_states
        )
    )

    # Split states (pos/vel)
    act_pos, act_vel = split_state(act_states)
    nom_pos, nom_vel = split_state(nom_states)
    fin_pos, fin_vel = split_state(fin_states)
    tgt_pos, tgt_vel = split_state(tgt_states)

    act_u = act_inputs
    nom_u = nom_inputs

    # Computed Norms (moved inside logic or computed on fly)
    # 3. Time vector
    start_anom = cfg.time_periapsis
    steps = act_states.shape[0] if act_states.ndim > 1 else 0
    t = np.linspace(start_anom, start_anom + (steps - 1) * cfg.anom_step, steps)

    # 4. Docking Slice
    dock_index = cfg.dock_index
    if dock_index is None:
        if "dock_index" in kwargs:
            dock_index = kwargs["dock_index"]
        else:
            dock_index = -1

    # Create Series
    act_series = MPCSeries(
        t=t,
        position=act_pos,
        velocity=act_vel,
        control_input=act_u,
        pos_ref1=fin_pos if fin_pos.size > 0 else None,
        pos_ref2=tgt_pos if tgt_pos.size > 0 else None,
    )

    nom_series = MPCSeries(
        t=t,
        position=nom_pos,
        velocity=nom_vel,
        control_input=nom_u,
        pos_ref1=fin_pos if fin_pos.size > 0 else None,
        pos_ref2=tgt_pos if tgt_pos.size > 0 else None,
    )

    def plot_mpc_series(series: MPCSeries, prefix: str, do_save=True, is_docking=False):
        if len(series.t) == 0:
            return

        state_x, state_y, state_z = series.position.T
        velocity_x, velocity_y, velocity_z = series.velocity.T

        u_list = []
        labelU = []
        # Conditional u logic
        if series.control_input is not None and series.control_input.size > 0:
            # Safer check: ensure (N,3)
            if series.control_input.ndim == 2 and series.control_input.shape[1] == 3:
                control_x, control_y, control_z = series.control_input.T
                u_list = [control_x, control_y, control_z]
                labelU = ["$u_X$", "$u_Y$", "$u_Z$"]
                u_norm = norms(series.control_input)
            else:
                # If u is present but bad shape (could happen with legacy data), ignore or flatten?
                # Ignoring to prevent crash.
                u_norm = None
        else:
            u_norm = None

        ref_state_list_1 = (
            list(series.pos_ref1.T)
            if series.pos_ref1 is not None and series.pos_ref1.size > 0
            else []
        )
        x2_list = (
            list(series.pos_ref2.T)
            if series.pos_ref2 is not None and series.pos_ref2.size > 0
            else []
        )

        labelX = ["$s_X$", "$s_Y$", "$s_Z$", "$v_X$", "$v_Y$", "$v_Z$"]

        y_list = [norms(series.position), norms(series.velocity)]
        labelY = ["$||s||$", "$||v||$"]

        if u_norm is not None:
            y_list.append(u_norm)
            labelY.append("$||u||$")

        labelX1 = ["$s_{X,F}$", "$s_{Y,F}$", "$s_{Z,F}$"]
        labelX2 = ["$s_{X,T}$", "$s_{Y,T}$", "$s_{Z,T}$"]

        file_name = f"{prefix}_states_{'docking' if is_docking else 'all_time'}.png"
        save_path = outpath(cfg.target_folder, file_name) if do_save else None

        file_name = f"{prefix}_states_{'docking' if is_docking else 'all_time'}.png"
        save_path = outpath(cfg.target_folder, file_name) if do_save else None

        # Logic: We split the plotting into 3 separate files if saving,
        # or combine them if just showing.

        base_name = f"{prefix}_{'docking' if is_docking else 'all_time'}"

        # 1. States
        if do_save:
            f_states = outpath(cfg.target_folder, f"{base_name}_states.png")
            plot_time_series(
                series.t,
                states=[state_x, state_y, state_z, velocity_x, velocity_y, velocity_z],
                control_inputs=[],  # No inputs
                y_outputs=[],  # No outputs
                labelX=labelX,
                states_ref1=ref_state_list_1,
                states_ref2=x2_list,
                labelX1=labelX1,
                labelX2=labelX2,
                save_path=f_states,
            )

        # 2. Inputs (if present)
        if do_save and u_list:
            f_inputs = outpath(cfg.target_folder, f"{base_name}_inputs.png")
            plot_time_series(
                series.t,
                states=[],  # No states
                control_inputs=u_list,
                y_outputs=[],
                labelU=labelU,
                save_path=f_inputs,
            )

        # 3. Norms/Outputs
        if do_save:
            f_norms = outpath(cfg.target_folder, f"{base_name}_norms.png")
            plot_time_series(
                series.t,
                states=[],
                control_inputs=[],
                y_outputs=y_list,
                labelY=labelY,
                save_path=f_norms,
            )

        # If not saving (show), we might spam windows.
        if not do_save:
            # Just plot combined? Or seq?
            # For show(), combined is better.
            plot_time_series(
                series.t,
                states=[state_x, state_y, state_z, velocity_x, velocity_y, velocity_z],
                control_inputs=u_list,
                y_outputs=y_list,
                states_ref1=ref_state_list_1,
                states_ref2=x2_list,
                labelX=labelX,
                labelU=labelU,
                labelY=labelY,
                labelX1=labelX1,
                labelX2=labelX2,
                save_path=None,
            )

    # 5. Execute Plots

    # --- Actual States ---
    if plot_flags.plot_act:
        if plot_flags.plot_act_sim:
            # 3D Animation
            create_animation_html(
                sim_html_path(cfg, "_act.html"),
                [act_pos[:, 0]],
                [act_pos[:, 1]],
                [act_pos[:, 2]],
                ["Chaser"],
                1,
                point_list=x_f_list,
            )

        # Time Series
        plot_mpc_series(act_series, "act", do_save=True, is_docking=False)

        # Docking
        if dock_index is not None and dock_index >= 0:
            dock_series = act_series.slice(dock_index)
            plot_mpc_series(dock_series, "act", do_save=True, is_docking=True)

    # --- Nominal States ---
    if plot_flags.plot_nom:
        if plot_flags.plot_nom_sim:
            create_animation_html(
                sim_html_path(cfg, "_nom.html"),
                [nom_pos[:, 0]],
                [nom_pos[:, 1]],
                [nom_pos[:, 2]],
                ["Nominal"],
                1,
                point_list=x_f_list,
                # Shape for sphere
                shape=[
                    {
                        "type": "sphere",
                        "radius": 10.0,
                        "center": [0, 0, 0],
                        "opacity": 0.25,
                    }
                ],
            )

        plot_mpc_series(nom_series, "nom", do_save=True, is_docking=False)
        if dock_index is not None and dock_index >= 0:
            dock_series_nom = nom_series.slice(dock_index)
            plot_mpc_series(dock_series_nom, "nom", do_save=True, is_docking=True)

    # --- Constraints ---
    # (Simplified Logic: Calculate constraints and plot)
    def plot_constraints(prefix):
        # Select data
        pos = act_pos if prefix == "act" else nom_pos
        # Constraints logic (Cylinder example)
        # Needs target_limits and target_thetas
        if not cfg.target_limits or not target_thetas:
            return

        con1, con2, con12 = [], [], []
        rT = cfg.target_limits.target_radius
        lT = cfg.target_limits.target_height

        for i, time_step in enumerate(t):
            if i >= len(target_thetas):
                break
            if i >= len(pos):
                break

            rot = tait_bryan_to_rotation_matrix(target_thetas[i])
            s_body = rot.T @ pos[i]

            c1 = s_body[0] ** 2 + s_body[1] ** 2 - rT**2
            c2 = s_body[2] ** 2 - lT**2
            c12 = 1.0 if (c1 < 0 and c2 < 0) else -1.0  # Boolean violation indicator

            con1.append(c1)
            con2.append(c2)
            con12.append(c12)

        file_name = f"constraints_{prefix}_states_all_time.png"
        save_path = outpath(cfg.target_folder, file_name)

        sXa, sYa, sZa = pos[:, 0], pos[:, 1], pos[:, 2]

        # Pass constraints as 'y' (Output) instead of inputs
        labelY = [f"$s_X^2+s_Y^2 - {rT}^2$", f"$s_Z^2 - {lT}^2$", "Violation"]

        plot_time_series(
            t[: len(con1)],
            states=[sXa, sYa, sZa],
            control_inputs=[],
            y_outputs=[np.array(con1), np.array(con2), np.array(con12)],
            labelX=["$s_X$", "$s_Y$", "$s_Z$"],
            labelY=labelY,
            save_path=save_path,
        )

    if plot_flags.plot_act_con:
        plot_constraints("act")
    if plot_flags.plot_nom_con:
        plot_constraints("nom")


def plot_adaptor(estim_lists, range_lists, orbit_params, *args, **kwargs):
    """Plot adaptive estimation results."""
    # Fix orbitParams -> orbit_params

    # Unwrap kwargs for range_params
    range_params = kwargs.get("rangeParams")
    if not range_params:
        # Warn or return?
        return

    dt = range_params["dt"]
    drange = range_params["data_range"]
    time_vec = dt * np.arange(drange)

    target_folder = kwargs.get("target_folder")

    # Plot Control Inputs (u_t) if present
    if "u_t" in orbit_params:
        u_t = orbit_params["u_t"]
        plt.figure(figsize=(10, 10))
        colors = ["b-", "g-", "r-"]
        for i in range(min(3, len(u_t))):
            plt.plot(time_vec, u_t[i], colors[i])
        plt.show()

    # Plot W if present
    if "W" in kwargs:
        W = kwargs["W"]
        plt.figure(figsize=(10, 10))
        colors = ["b-", "g-", "r-"]
        for i in range(min(3, W.shape[1])):
            plt.plot(time_vec, W[:, i], colors[i])
        plt.show()

    # Plot Estimates
    eEst, aEst, bEst = estim_lists
    eR, aR, bR = range_lists

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    # Eccentricity
    axs[0].plot(eEst, "b-")
    axs[0].plot(eR[0], "r-")
    axs[0].plot(eR[1], "r-")
    ref_e = orbit_params.get("eccentricity", 0)
    axs[0].plot([ref_e] * len(eEst), "r--")
    axs[0].set_title("Estimate of Eccentricity")

    # Alpha
    axs[1].plot(aEst, "b-")
    axs[1].plot(aR[0], "r-")
    axs[1].plot(aR[1], "r-")
    ref_a = orbit_params.get("drag_alpha", 0)
    axs[1].plot([ref_a] * len(aEst), "r--")
    axs[1].set_title("Estimate of Alpha")

    # Beta
    axs[2].plot(bEst, "b-")
    axs[2].plot(bR[0], "r-")
    axs[2].plot(bR[1], "r-")
    ref_b = orbit_params.get("drag_beta", 0)
    axs[2].plot([ref_b] * len(bEst), "r--")
    axs[2].set_title("Estimate of Beta")

    if target_folder:
        spath = Path(target_folder) / "param_est.png"
        spath.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(spath)
        plt.close()
    else:
        plt.show()


def plot_deflection(target, x_surface, r_agents, f_agents, plot_flags, *args, **kwargs):
    """Plot deflection mission."""
    # Renamed arguments to avoid shadowing: x -> x_surface, r -> r_agents, f -> f_agents

    # Unpack kwargs
    dt = kwargs["dt"]
    num_steps = kwargs.get("numSteps", 1)
    num_chasers = kwargs.get("num_chasers", 1)
    shape_params = kwargs.get("shapeParams", {})
    target_shape = kwargs.get("target_shape", "sphere")
    init_tl = kwargs.get("initial_time_lapse", 0)
    target_folder = kwargs.get("target_folder")

    # Use PlotFlags object
    if isinstance(plot_flags, dict):
        plot_flags = PlotFlags(**plot_flags)

    # Plot Target
    if plot_flags.plot_target:
        # Assuming target has plotStateHistory
        target.plotStateHistory(
            params={"sep_plots": False, "disp_plot": False}, target_folder=target_folder
        )

    # Plot Position
    if plot_flags.plot_position:
        fig = plt.figure()
        ax = plt.axes(projection="3d")

        # Surface generation
        x_surf, y_surf, z_surf = None, None, None

        if target_shape == "cylinder":
            cylHeight = shape_params["cylHeight"]
            cylRadius = shape_params["cylRadius"]
            cylCenter = shape_params["cylCenter"]
            u, v = np.mgrid[0 : 2 * np.pi : 30j, -1.0:1.0:30j]
            x_surf = cylCenter[0] + cylRadius * np.cos(u)
            y_surf = cylCenter[1] + cylRadius * np.sin(u)
            z_surf = cylCenter[2] + cylHeight * v

        elif target_shape == "ellipsoid":
            ellRadX = shape_params["ellRadX"]
            ellRadY = shape_params["ellRadY"]
            ellRadZ = shape_params["ellRadZ"]
            ellCenter = shape_params["ellCenter"]
            u, v = np.mgrid[0 : 2 * np.pi : 50j, 0 : np.pi : 50j]
            x_surf = ellCenter[0] + ellRadX * np.cos(u) * np.sin(v)
            y_surf = ellCenter[1] + ellRadY * np.sin(u) * np.sin(v)
            z_surf = ellCenter[2] + ellRadZ * np.cos(v)

        if x_surf is not None:
            ax.plot_surface(x_surf, y_surf, z_surf, cmap=plt.cm.YlGnBu_r)

        # Plot agents
        for agent in range(num_chasers):
            state = r_agents[agent]
            ax.scatter(state[0], state[1], state[2], label=f"$r_{agent}$")

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.legend()

        # Saving logic
        if target_folder:
            tf = Path(target_folder)
            tf.mkdir(parents=True, exist_ok=True)
            azims = [45 * i for i in range(8)]
            for i in range(8):
                ax.view_init(45, azims[i])
                plt.gcf().set_size_inches(plt.gcf().get_size_inches())
                plt.tight_layout()
                plt.savefig(tf / f"chaser_pos_ortho{i+1}.png")

                ax.view_init(0, azims[i])
                plt.savefig(tf / f"chaser_pos_vert{i+1}.png")

                ax.view_init(88, azims[i])
                plt.savefig(tf / f"chaser_pos_top{i+1}.png")
            plt.close()
        else:
            plt.show()

    # Plot Force
    if plot_flags.plot_force:
        plt.figure()
        t = np.linspace(init_tl, init_tl + (num_steps - 1) * dt, num_steps)
        for agent in range(num_chasers):
            force = f_agents[agent]
            for j in range(3):
                plt.plot(t, force[j], label=f"$f_{agent},{['x','y','z'][j]}$")
        plt.legend()
        if target_folder:
            # Save the plot to target folder
            os.makedirs(target_folder, exist_ok=True)
            plt.savefig(
                os.path.join(target_folder, "forces.png"), dpi=150, bbox_inches="tight"
            )
            plt.close()
        else:
            plt.show()


# Functions generate_orbit and import_orbit
def generate_orbit(constants, t, num_agents):
    mu, r_0, n, T = constants
    d_s = 10

    # Logic unchanged...
    rho_x = [(agent * (d_s / 2.0) / num_agents) for agent in range(1, num_agents + 1)]
    # ...
    # Return X, Y, Z lists

    # Re-implementing the logic briefly to ensure validity
    rho_y = 0
    rho_z = [agent * d_s / num_agents for agent in range(num_agents)]
    alpha_x = 0
    alpha_z = [(agent * np.pi / num_agents - np.pi / 2) for agent in range(num_agents)]

    random.shuffle(rho_x)
    random.shuffle(rho_z)
    random.shuffle(alpha_z)

    X = [
        [rho_x[agent] * np.sin(n * t[elem] + alpha_x) for elem in range(len(t))]
        for agent in range(num_agents)
    ]
    Y = [
        [
            rho_y + 2.0 * rho_x[agent] * np.cos(n * t[elem] + alpha_x)
            for elem in range(len(t))
        ]
        for agent in range(num_agents)
    ]
    Z = [
        [rho_z[agent] * np.sin(n * t[elem] + alpha_z[agent]) for elem in range(len(t))]
        for agent in range(num_agents)
    ]

    return X, Y, Z


def import_orbit():
    # Unchanged
    fname = latest_data_file("../results/")
    data = load_data(fname)
    ipData = data["ipData"]
    opData = data["opData"]
    x, u, y, d = opData
    return [x[0]], [x[1]], [x[2]], fname


# MAIN
if __name__ == "__main__":
    # Test logic
    import time  # Needed

    mu = 3.986004418 * (10**14)
    r_0 = (6371 + 400) * (10**3)
    n = np.sqrt(mu / r_0**3)
    T = 2 * np.pi / n

    constants = (mu, r_0, n, T)
    timeSeq = list(np.arange(0, T, T / 1000))

    remote = True  # Toggle
    if not remote:
        num_agents = 5
        X, Y, Z = generate_orbit(constants, timeSeq, num_agents)
        fname = create_filename("../plots/", ".html")
        create_animation_html(
            fname, X, Y, Z, [f"Agent {i+1}" for i in range(num_agents)]
        )
    else:
        # Mocking import logic - placeholder for remote data import
        logger.info("Remote mode enabled - import logic not implemented")
