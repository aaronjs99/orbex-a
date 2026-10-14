"""Saved target geometry shared by static, interactive and animated views."""
from __future__ import annotations

from typing import Optional
import numpy as np

from orbexa.visualization.samples import MissionSamples


class MissionGeometry:
    """Resolve geometry only from a result's own recorded or reconstructed pose."""

    def __init__(self, samples: MissionSamples):
        self.samples = samples

    def cylinder_mesh(self, index: int, *, theta_count: int = 48, z_count: int = 12):
        geometry = self.samples.cylinder()
        if geometry is None or self.samples.rotations is None or index < 0 or index >= len(self.samples.q):
            return None
        radius, half_length = geometry
        theta, height = np.meshgrid(np.linspace(0, 2 * np.pi, theta_count), np.linspace(-half_length, half_length, z_count))
        body = np.stack((radius * np.cos(theta), radius * np.sin(theta), height))
        return (self.samples.rotations[index] @ body.reshape(3, -1)).reshape(body.shape)

    def rendezvous_radius(self) -> Optional[float]:
        target = self.samples.target
        try:
            value = float(target.get("rendezvous_sphere_radius", target.get("rendezvous_radius", float("nan"))))
            if value > 0 and np.isfinite(value):
                return value
        except (TypeError, ValueError):
            pass
        return None

    @staticmethod
    def offset_sphere_mesh(center, radius, *, count: int = 18):
        theta, phi = np.meshgrid(np.linspace(0, 2 * np.pi, count), np.linspace(0, np.pi, max(count // 2, 4)))
        center = np.asarray(center, dtype=float)
        return (center[0] + radius * np.cos(theta) * np.sin(phi),
                center[1] + radius * np.sin(theta) * np.sin(phi),
                center[2] + radius * np.cos(phi))

    @staticmethod
    def tube_box_mesh(center, omega):
        """A matched component box, without substituting a fleet-wide radius."""
        signs = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                          [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])
        vertices = np.asarray(center) + signs * np.asarray(omega)
        faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
                          [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                          [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
        return vertices, faces

    def sphere_mesh(self):
        radius = self.rendezvous_radius()
        return None if radius is None else self.offset_sphere_mesh(np.zeros(3), radius, count=36)

    def docking_points(self, index: int):
        samples = self.samples
        if samples.rotations is None or index < 0 or index >= len(samples.q):
            return {}
        return {identifier: samples.rotations[index] @ point
                for identifier, point in samples.goals_body.items()}

    def plot_target(self, ax, index: int, *, label: bool = True):
        mesh = self.cylinder_mesh(index)
        if mesh is None:
            return None
        artist = ax.plot_surface(*mesh, color="#7B8A99", alpha=0.30, linewidth=0, shade=True)
        if label:
            ax.plot([], [], [], color="#7B8A99", linewidth=6, label="rotating target cylinder")
        return artist

    def plot_sphere(self, ax):
        mesh = self.sphere_mesh()
        if mesh is not None:
            ax.plot_wireframe(*mesh, color="#0072B2", alpha=0.16, linewidth=0.45)
            ax.plot([], [], [], color="#0072B2", alpha=0.45, label="rendezvous sphere")

    def plot_docking_points(self, ax, index: int):
        points = self.docking_points(index)
        if points:
            values = np.asarray(list(points.values()))
            ax.scatter(values[:, 0], values[:, 1], values[:, 2], color="#D55E00", marker="D", s=36, label="docking points")

    def limits(self):
        positions = self.samples.states[:, :, :3].reshape(-1, 3)
        radius = max(self.rendezvous_radius() or 0.0, max(self.samples.cylinder() or (0.0, 0.0)))
        points = [positions, np.array([[-radius] * 3, [radius] * 3])]
        if self.samples.goals_body:
            extent = max(float(np.linalg.norm(value)) for value in self.samples.goals_body.values())
            points.append(np.array([[-extent] * 3, [extent] * 3]))
        values = np.vstack(points)
        low, high = values.min(axis=0), values.max(axis=0)
        center = (low + high) / 2
        extent = max(float(np.max(high - low)) / 2, 0.1) * 1.12
        return center, extent
