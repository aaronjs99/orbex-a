"""Common right-handed mounting frames for arbitrary structural surface normals."""
from __future__ import annotations
import numpy as np
from orbexa.core.attitude import finite_vector


def mounting_frame(normal_body):
    """Columns map chaser axes to target body, with chaser +x outward.

    Retain the previous z-reference convention away from its singular axis;
    select body y there. The identical frame is used by attitude and capture.
    """
    normal = finite_vector(normal_body, 'Assigned outward normal', 3)
    scale = float(np.max(np.abs(normal)))
    if scale == 0:
        raise ValueError('Mounting normal cannot be zero')
    normal /= scale
    normal /= np.linalg.norm(normal)
    reference = np.array([0.,0.,1.]) if abs(normal[2]) < .99 else np.array([0.,1.,0.])
    tangent = np.cross(reference,normal)
    tangent /= np.linalg.norm(tangent)
    return np.column_stack((normal,tangent,np.cross(normal,tangent)))
