"""
3D representations and geometric processing.

This module contains functions for working with 3D data representations:
- Depth maps and surface normals
- Implicit functions (SDF)
- Voxel grids and point clouds
"""

from .depth_maps import depth_map
from .surface_normals import surface_normal
from .implicit_functions import (
    implicit_function_sphere,
    implicit_function_box, 
    implicit_function_plane
)
from .voxels import points_to_voxel, voxel_to_points
from .point_clouds import (
    point_cloud_from_depth,
    downsample_point_cloud,
    compute_point_normals
)

__all__ = [
    'depth_map',
    'surface_normal',
    'implicit_function_sphere',
    'implicit_function_box',
    'implicit_function_plane',
    'points_to_voxel',
    'voxel_to_points',
    'point_cloud_from_depth',
    'downsample_point_cloud',
    'compute_point_normals',
]