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
import math
import numpy as np

try:
    from mayavi import mlab
    from mayavi.modules.surface import Surface

    MAYAVI_AVAILABLE = True
except ImportError:
    MAYAVI_AVAILABLE = False
    mlab = None
    Surface = None

try:
    from meshpy.tet import MeshInfo, build
    from meshpy.geometry import make_box

    MESHPY_AVAILABLE = True
except ImportError:
    MESHPY_AVAILABLE = False
    MeshInfo = None
    build = None
    make_box = None


# FUNCTION DEFINITIONS
def pointSet1():
    points = [
        (0, 0, 0),
        (4, 0, 0),
        (6, 2, 0),
        (9, 1, 0),
        (10, 4, 0),
        (8, 8, 0),
        (8, 10, 0),
        (6, 9, 0),
        (5, 6, 0),
        (4, 5, 0),
        (3, 7, 0),
        (0, 6, 0),
        (-1, 4, 0),
        (-2, 4, 0),
        (-2, 1, 0),
    ]
    more_points = []
    for point in points:
        more_points.append((point[0], point[1], point[2] + 2))
    points.extend(more_points)
    facets = [
        (0, 15, 29, 14),
        (0, 1, 16, 15),
        (1, 2, 17, 16),
        (2, 3, 18, 17),
        (3, 4, 19, 18),
        (4, 5, 20, 19),
        (5, 6, 21, 20),
        (6, 7, 22, 21),
        (7, 8, 23, 22),
        (8, 9, 24, 23),
        (9, 10, 25, 24),
        (10, 11, 26, 25),
        (11, 12, 27, 26),
        (12, 13, 28, 27),
        (13, 14, 29, 28),
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14),
        (15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29),
    ]

    return points, facets


def pointSet2():
    points = [
        (0, 0, 0),
        (4, 0, -1),
        (6, 2, -1),
        (9, 6, -2),
        (4, 5, -2),
        (2, 3, -1),
        (-1, 2, 0),
        (0, 0, 2),
        (4, 0, 4),
        (6, 2, 5),
        (8, 5, 6),
        (3, 5, 4),
        (2, 3, 3),
        (-1, 2, 2),
    ]
    facets = [
        (0, 1, 8, 7),
        (1, 2, 9, 8),
        (2, 3, 10, 9),
        (3, 4, 11, 10),
        (4, 5, 12, 11),
        (5, 6, 13, 12),
        (7, 0, 6, 13),
        (0, 1, 2, 3, 4, 5, 6),
        (7, 8, 9, 10, 11, 12, 13),
    ]

    return points, facets


def pointSet3():
    points = [
        (0, 0, 4),
        (6, 0, 3),
        (5, 2, 2),
        (11, 6, 3),
        (9, 9, 4),
        (3, 10, 4),
        (2, 5, 5),
        (-1, 3, 4),
        (0, -1, 0),
        (7, -1, -1),
        (11, 1, -1),
        (14, 1, 0),
        (16, 5, -1),
        (18, 8, -2),
        (17, 7, -2),
        (12, 7, -2),
        (12, 9, -3),
        (7, 9, -3),
        (6, 10, -2),
        (0, 9, -2),
        (0, 7, -2),
        (-4, 5, -1),
        (-4, 3, -1),
        (0, 1, 0),
        (3, 3, -8),
        (5, 5, -8),
        (7, 8, -9),
        (7, 10, -9),
        (6, 10, -9),
        (4, 8, -8),
        (3, 5, -8),
    ]
    facets = [
        (0, 8, 9, 10, 1),
        (1, 10, 11, 3, 2),
        (3, 11, 12, 13, 14, 15, 16, 4),
        (4, 16, 17, 18, 19, 5),
        (5, 19, 20, 21, 6),
        (6, 21, 22, 7),
        (7, 22, 23, 8, 0),
        (8, 9, 10, 25, 24),
        (10, 11, 12, 26, 25),
        (12, 13, 14, 15, 27, 26),
        (15, 16, 17, 28, 27),
        (17, 18, 19, 29, 28),
        (19, 20, 21, 30, 29),
        (21, 22, 23, 8, 24, 30),
        (0, 1, 2, 3, 4, 5, 6, 7),
        (8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23),
        (24, 25, 26, 27, 28, 29, 30),
    ]

    return points, facets


# MAIN PROGRAM
if __name__ == "__main__":
    mesh_info = MeshInfo()

    points, facets = pointSet2()

    mesh_info.set_points(points)
    mesh_info.set_facets(facets)

    debug = False
    mesh = build(mesh_info)
    if debug:
        print("Mesh Points")
        for i, p in enumerate(mesh.points):
            print("Point %d: %s" % (i, p))
        print()
        print("Mesh Elements")
        for i, e in enumerate(mesh.elements):
            print("Element %d: %s" % (i, e))
        print()
        print("Mesh Facets")
        for i, f in enumerate(mesh.facets):
            print("Facet %d: %s" % (i, f))
    mesh.write_vtk("../results/mesh/mesh.vtk")

    # Plot the mesh
    fig = mlab.figure(bgcolor=(0, 0, 0))
    engine = mlab.get_engine()
    vtk_file_reader = engine.open("../results/mesh/mesh.vtk")
    surf = Surface()
    engine.add_filter(surf, vtk_file_reader)
    # mlab.figure(bgcolor=(1, 1, 1))
    # mlab.triangular_mesh(mesh.points[:, 0], mesh.points[:, 1], mesh.points[:, 2], mesh.elements)
    mlab.show()
