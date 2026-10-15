"""Component-aware attachment assignment and finite-bus reference routes.

This bounded visibility search screens geometric references. It does not
establish dynamic feasibility, uncertain navigation or continuous execution;
the nonlinear controller and swept-flow admission retain those boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools

import numpy as np
from scipy.optimize import linear_sum_assignment


def _positive(value, name, *, zero=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(name + " must be a finite scalar with the required sign")
    value = float(value)
    if not np.isfinite(value) or value < 0 or (not zero and value == 0):
        raise ValueError(name + " must be a finite scalar with the required sign")
    return value


def _point(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(name + " must be a finite body-frame three-vector")
    return result.copy()


def segment_point_distance(start, end, point):
    delta = end - start
    squared = float(delta @ delta)
    progress = 0. if squared == 0 else float(np.clip((point - start) @ delta / squared, 0., 1.))
    return float(np.linalg.norm(start + progress * delta - point))


@dataclass(frozen=True)
class AttachmentRoute:
    chaser_index: int
    patch_index: int
    patch_id: str
    component_id: str
    surface_point_body_m: np.ndarray
    normal_body: np.ndarray
    final_com_point_body_m: np.ndarray
    staging_point_body_m: np.ndarray
    waypoints_body_m: np.ndarray
    release_order: int
    length_m: float


class CompositeAttachmentPlanner:
    """Assign explicit patches and route sequentially around all components.

    Each route is screened against every OBB enlarged coordinatewise by the
    bus, declared reference tube, clearance and its own geometry uncertainty.
    This enlargement is conservative. Other chasers are stationary body-frame
    points while one route moves, and segment-to-point distance screens their
    complete reference separation. No target-wide replacement shape is used.
    """

    def __init__(self, geometry, *, body_radius_m, reference_tube_radius_m,
                 clearance_m, minimum_pair_separation_m, route_buffer_m=.05):
        from orbexa.core.spacecraft_geometry import SpacecraftGeometry
        if not isinstance(geometry, SpacecraftGeometry):
            raise TypeError("Attachment routes require an explicit SpacecraftGeometry")
        self.geometry = geometry
        self.body_radius = _positive(body_radius_m, "Route bus radius")
        self.tube_radius = _positive(reference_tube_radius_m, "Route tube radius", zero=True)
        self.clearance = _positive(clearance_m, "Route clearance", zero=True)
        self.minimum_pair = _positive(minimum_pair_separation_m, "Route pair separation")
        self.buffer = _positive(route_buffer_m, "Route corner buffer")
        if self.minimum_pair < 2 * self.body_radius:
            raise ValueError("Route pair separation must enclose both finite buses")
        self.padding = self.body_radius + self.tube_radius + self.clearance
        self.pair_radius = self.minimum_pair + 2 * self.tube_radius
        self._edge_cache = {}
        self._components = []
        candidates = []
        for component in geometry.components:
            rotation = np.asarray(component.rotation_component_to_body)
            inverse = np.linalg.inv(rotation)
            inverse_row_norms = np.linalg.norm(inverse, axis=1)
            center = np.asarray(component.center_m)
            half = np.asarray(component.half_extents_m) + (self.padding + component.uncertainty_m) * inverse_row_norms
            self._components.append((inverse, center, half))
            corner_half = half + self.buffer * inverse_row_norms
            candidates.extend(center + rotation @ (corner_half * signs)
                              for signs in map(np.array, itertools.product((-1., 1.), repeat=3)))
        vertices = np.concatenate([np.asarray(c.nominal_vertices_body_m, dtype=float) for c in geometry.components])
        lower, upper = vertices.min(axis=0), vertices.max(axis=0)
        span = self.padding + max(c.uncertainty_m for c in geometry.components) + self.buffer
        candidates.extend(np.array([lower[i] - span if sign[i] < 0 else upper[i] + span for i in range(3)])
                          for sign in itertools.product((-1., 1.), repeat=3))
        self.waypoints = self._unique([p for p in candidates if self.segment_clear(p, p)])

    @staticmethod
    def _unique(points):
        result, seen = [], set()
        for point in points:
            key = tuple(map(float, point))
            if key not in seen:
                result.append(np.asarray(point, dtype=float)); seen.add(key)
        return result

    def segment_clear(self, start, end):
        """Numerical slab exclusion of each conservatively enlarged OBB."""
        start, end = _point(start, "Route segment start"), _point(end, "Route segment end")
        key = tuple(sorted((tuple(start), tuple(end))))
        if key in self._edge_cache:
            return self._edge_cache[key]
        for inverse, center, half in self._components:
            local = inverse @ (start - center)
            delta = inverse @ (end - start)
            lo, hi = 0., 1.
            for axis in range(3):
                if abs(delta[axis]) <= 1e-14:
                    if abs(local[axis]) > half[axis]:
                        lo, hi = 1., 0.; break
                else:
                    first, last = sorted(((-half[axis] - local[axis]) / delta[axis],
                                          (half[axis] - local[axis]) / delta[axis]))
                    lo, hi = max(lo, first), min(hi, last)
                    if lo > hi: break
            if lo <= hi:
                self._edge_cache[key] = False
                return False
        self._edge_cache[key] = True
        return True

    def shortest_route(self, start, goal, stationary_points=()):
        """Shortest path in this finite graph; None means search failure only."""
        start, goal = _point(start, "Release point"), _point(goal, "Staging point")
        stationary = tuple(_point(point, "Stationary fleet point") for point in stationary_points)
        def free(first, second):
            return self.segment_clear(first, second) and all(
                segment_point_distance(first, second, point) > self.pair_radius for point in stationary)
        if not free(start, start) or not free(goal, goal): return None
        if np.array_equal(start, goal): return np.array([start, goal])
        if free(start, goal): return np.array([start, goal])
        candidates = list(self.waypoints)
        for point in stationary:
            half = self.pair_radius + self.buffer
            candidates.extend(point + half * np.array(signs) for signs in itertools.product((-1., 1.), repeat=3))
        nodes = [start, goal] + self._unique([p for p in candidates if free(p, p)
                                            and not np.array_equal(p, start) and not np.array_equal(p, goal)])
        distances, previous, queue = [np.inf] * len(nodes), {}, [(0., 0)]
        distances[0] = 0.
        while queue:
            distance, index = heapq.heappop(queue)
            if distance != distances[index]: continue
            if index == 1:
                path = [1]
                while path[-1] != 0: path.append(previous[path[-1]])
                return np.asarray([nodes[i] for i in path[::-1]])
            for other, point in enumerate(nodes):
                if other == index: continue
                candidate = distance + float(np.linalg.norm(point - nodes[index]))
                if candidate < distances[other] and free(nodes[index], point):
                    distances[other], previous[other] = candidate, index
                    heapq.heappush(queue, (candidate, other))
        return None

    @staticmethod
    def length(route):
        return float(np.sum(np.linalg.norm(np.diff(route, axis=0), axis=1)))

    def plan(self, initial_points_body_m, *, com_standoff_m, staging_offset_m, patch_ids=None):
        starts = np.asarray(initial_points_body_m, dtype=float)
        if starts.ndim != 2 or starts.shape[1:] != (3,) or not len(starts) or not np.all(np.isfinite(starts)):
            raise ValueError("Supply the complete finite body-frame release cluster")
        standoff = _positive(com_standoff_m, "Attachment COM standoff")
        staging = _positive(staging_offset_m, "Attachment staging offset")
        if staging <= standoff:
            raise ValueError("Staging offset must exceed the final COM standoff")
        patches = list(self.geometry.attachment_patches)
        if patch_ids is not None:
            selected = tuple(patch_ids)
            if not selected or len(set(selected)) != len(selected):
                raise ValueError("Explicit patch IDs must be nonempty and unique")
            patches = [self.geometry.attachment_patch(key) for key in selected]
        candidates = []
        for patch in patches:
            final = np.asarray(patch.point_body_m) + standoff * np.asarray(patch.normal_body)
            goal = np.asarray(patch.point_body_m) + staging * np.asarray(patch.normal_body)
            if self.segment_clear(goal, final): candidates.append((patch, final, goal))
        if len(candidates) < len(starts):
            raise ValueError("Too few declared patches have a clear finite-bus normal corridor; increase standoff or choose accessible patches")
        costs = np.full((len(starts), len(candidates)), np.inf)
        for row, start in enumerate(starts):
            for col, (_, final, goal) in enumerate(candidates):
                route = self.shortest_route(start, goal)
                if route is not None: costs[row, col] = self.length(route)
        try: rows, cols = linear_sum_assignment(costs)
        except ValueError as error:
            raise ValueError("The finite route search found no complete patch matching; this is not a physical unreachability proof") from error
        assignment = dict(zip(rows.tolist(), cols.tolist()))
        for index, col in assignment.items():
            for other, other_col in assignment.items():
                if other <= index: continue
                for offset in (1, 2):
                    if np.linalg.norm(candidates[col][offset] - candidates[other_col][offset]) <= self.pair_radius:
                        raise ValueError("Selected attachment/staging points need a more widely separated explicit patch subset")
        positions, remaining, routes = starts.copy(), set(range(len(starts))), []
        while remaining:
            available = []
            for index in sorted(remaining):
                patch, final, goal = candidates[assignment[index]]
                route = self.shortest_route(starts[index], goal,
                                            [p for i, p in enumerate(positions) if i != index])
                if route is not None: available.append((self.length(route), index, route))
            if not available:
                raise ValueError("The finite sequential fleet route search exhausted its graph; no dynamic or physical infeasibility is inferred")
            length, index, waypoints = min(available, key=lambda item:(item[0],item[1]))
            patch, final, goal = candidates[assignment[index]]
            patch_index = next(i for i, original in enumerate(self.geometry.attachment_patches) if original.id == patch.id)
            routes.append(AttachmentRoute(index, patch_index, patch.id, patch.component_id,
                np.asarray(patch.point_body_m).copy(), np.asarray(patch.normal_body).copy(), final.copy(), goal.copy(),
                waypoints.copy(), len(routes), length))
            positions[index] = goal
            remaining.remove(index)
        return tuple(sorted(routes, key=lambda route:route.chaser_index))
