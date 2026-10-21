"""Capacity and geometry guards with their observed extrema and clearances."""
from __future__ import annotations

import numpy as np


class ContactGuards:
    """Apply declared endpoint/travel-reserve guards in their original order."""

    def __init__(self, *, fleet, options, geometry):
        self.fleet = fleet
        self.options = options
        self.geometry = geometry
        self.maxima = dict(force_n=0., moment_nm=0., compression_m=0., extension_m=0.,
                      positioner_power_w=0., internal_force_residual_n=0., internal_torque_residual_nm=0.,
                      quaternion_norm_error=0.)
        self.minima = dict(bus_component_clearance_m=np.inf, peer_bus_clearance_m=np.inf,
                      positioner_component_clearance_m=np.inf, positioner_peer_clearance_m=np.inf,
                      patch_edge_margin_m=np.inf)

    def cylinder_distance(self, point):
        geometry = self.geometry
        radial = np.linalg.norm(point[:2])-float(geometry['radius'])
        axial = abs(point[2])-float(geometry['half_length'])
        return float(np.linalg.norm(np.maximum([radial, axial], 0.))+min(max(radial, axial), 0.))

    def check(self, t, values, diag, dt):
        fleet, o = self.fleet, self.options
        members, count = fleet.members, fleet.chaser_count
        normals, unpack = fleet.normals, fleet.unpack
        geometry, cylinder_distance = self.geometry, self.cylinder_distance
        maxima, minima = self.maxima, self.minima
        b, tips = unpack(values)
        rt, wt = diag['rotations'][0], diag['spins'][0]
        segments = []
        for k, member in enumerate(members):
            j, row = k+1, diag['rows'][k]
            extension = float(np.linalg.norm(tips[k, :3]))
            maxima['extension_m'] = max(maxima['extension_m'], extension)
            maxima['force_n'] = max(maxima['force_n'], float(np.linalg.norm(row['force'])))
            maxima['moment_nm'] = max(maxima['moment_nm'], float(np.linalg.norm(row['couple'])))
            maxima['compression_m'] = max(maxima['compression_m'], row['compression'])
            maxima['positioner_power_w'] = max(maxima['positioner_power_w'], abs(row['positioner_power']))
            if extension > member.arm_reach_m+1e-10:
                return f'{member.id}: positioner_reach_exceeded'
            if np.linalg.norm(tips[k, 3:]) > o.extension_speed_m_s+o.extension_acceleration_m_s2*dt+1e-10:
                return f'{member.id}: positioner_speed_exceeded'
            if row['compression'] > o.max_compression_m:
                return f'{member.id}: pad_compression_exceeded'
            if np.linalg.norm(row['force']) > o.positioner_force_limit_n or abs(row['positioner_power']) > o.positioner_power_limit_w:
                return f'{member.id}: positioner_force_or_power_exceeded'
            if row['activation'] > 0:
                fn = float(row['force_patch']@normals[k])
                shear = np.linalg.norm(row['force_patch']-fn*normals[k])
                if (max(0., -fn) > row['activation']*o.tensile_limit_n or
                    shear > row['activation']*o.shear_limit_n+o.friction_coefficient*row['normal_force'] or
                    np.linalg.norm(row['couple']) > row['activation']*o.moment_limit_nm):
                    return f'{member.id}: adhesive_capacity_exceeded'
            com = rt.T@(b[j, :3]-b[0, :3])
            relative_speed = np.linalg.norm(b[j, 3:6]-b[0, 3:6]-np.cross(wt, b[j, :3]-b[0, :3]))
            # Both endpoint and an integration-step travel reserve are
            # checked. This is a declared velocity/acceleration sweep bound.
            travel = relative_speed*dt+.5*dt**2*(np.linalg.norm(diag['acceleration'][j]-diag['acceleration'][0])+
                                               2*np.linalg.norm(wt)*relative_speed)
            clearance = (geometry.signed_distance(com) if hasattr(geometry, 'components') else cylinder_distance(com))-member.properties.body_radius_m-travel
            minima['bus_component_clearance_m'] = min(minima['bus_component_clearance_m'], clearance)
            if clearance < o.minimum_bus_clearance_m:
                return f'{member.id}: bus_component_clearance_failed'
            tip_body = rt.T@(row['tip']-b[0, :3])
            contact_point = tip_body-row['gap']*normals[k]
            if hasattr(geometry, 'components'):
                patch = geometry.attachment_patch(member.patch_id)
                delta = contact_point-np.asarray(patch.point_body_m)
                margin = min(patch.half_extents_m[0]-patch.margin_m-abs(delta@patch.tangent_u_body),
                             patch.half_extents_m[1]-patch.margin_m-abs(delta@patch.tangent_v_body))-o.pad_radius_m-travel
                minima['patch_edge_margin_m'] = min(minima['patch_edge_margin_m'], float(margin))
                if margin < 0:
                    return f'{member.id}: pad_footprint_left_patch'
            arm_vector = row['tip']-b[j, :3]
            start = b[j, :3]+arm_vector/np.linalg.norm(arm_vector)*member.properties.body_radius_m
            # The thin pad is the only selected-face contact exception.
            end = row['tip']+rt@normals[k]*o.positioner_radius_m
            segments.append((start, end))
            mid_body = rt.T@((start+end)/2-b[0, :3])
            half = np.linalg.norm(end-start)/2
            if hasattr(geometry, 'components'):
                patch = geometry.attachment_patch(member.patch_id)
                for component in geometry.components:
                    if component.id == patch.component_id:
                        continue
                    margin = component.signed_distance(mid_body)-half-o.positioner_radius_m-travel
                    pad_margin = component.signed_distance(tip_body)-o.pad_radius_m-travel
                    margin = min(margin, pad_margin)
                    minima['positioner_component_clearance_m'] = min(minima['positioner_component_clearance_m'], margin)
                    if margin <= 0:
                        return f'{member.id}: positioner_obstructed_by_{component.id}'
        for k, member in enumerate(members):
            j = k+1
            for other in range(k+1, count):
                right = other+1
                margin = np.linalg.norm(b[j, :3]-b[right, :3])-member.properties.body_radius_m-members[other].properties.body_radius_m
                margin -= np.linalg.norm(b[j, 3:6]-b[right, 3:6])*dt+.5*dt**2*np.linalg.norm(diag['acceleration'][j]-diag['acceleration'][right])
                minima['peer_bus_clearance_m'] = min(minima['peer_bus_clearance_m'], float(margin))
                if margin < o.minimum_bus_clearance_m:
                    return f'{member.id}|{members[other].id}: peer_bus_clearance_failed'
            start, end = segments[k]
            direction = end-start
            for other, peer in enumerate(members):
                if other == k:
                    continue
                p = b[other+1, :3]
                closest = start+np.clip((p-start)@direction/(direction@direction), 0., 1.)*direction
                travel = (np.linalg.norm(diag['rows'][k]['tip_velocity']-b[other+1, 3:6])+o.extension_speed_m_s)*dt
                margin = np.linalg.norm(p-closest)-peer.properties.body_radius_m-max(o.pad_radius_m, o.positioner_radius_m)-travel
                minima['positioner_peer_clearance_m'] = min(minima['positioner_peer_clearance_m'], float(margin))
                if margin <= 0:
                    return f'{member.id}: positioner_obstructed_by_peer_{peer.id}'
        return None
