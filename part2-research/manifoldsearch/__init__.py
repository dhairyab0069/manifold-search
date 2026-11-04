"""Manifold search: shortest paths over curved surfaces, and how well a coarse-to-fine search approximates them.

Part 2 of the project (independent research, computational geometry). Part 1 is the CS 5800 notebook.
"""
from .meshes import Mesh, bumpy_sphere, delaunay_sphere, grid_sphere, grid_torus, load_obj
from .baselines import astar, dijkstra
from .hierarchical import HierarchicalIndex, build_index
from .metrics import sphere_geodesic, stretch_summary

__all__ = [
    "Mesh", "grid_sphere", "grid_torus", "bumpy_sphere", "delaunay_sphere", "load_obj",
    "dijkstra", "astar", "HierarchicalIndex", "build_index", "sphere_geodesic", "stretch_summary",
]
