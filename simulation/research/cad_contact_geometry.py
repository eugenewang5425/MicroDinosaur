"""Blender tessellation for CAD contacts: preserve concave polygons and holes.

The historical converter's fan triangulation is unsuitable for these booleans.
Use evaluated loop triangles; only read the CAD, never save or alter it.
"""
import bpy
import numpy as np


def triangles_world(obj):
    ev=obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh=ev.to_mesh()
    try:
        mesh.calc_loop_triangles()
        vertices=np.array([ev.matrix_world@v.co for v in mesh.vertices])
        faces=np.array([list(t.vertices) for t in mesh.loop_triangles])
        return vertices[faces]
    finally:
        ev.to_mesh_clear()
