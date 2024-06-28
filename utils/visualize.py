import mcubes
import numpy as np
from typing import Callable


def extract_geometry(
    bound_min: np.ndarray,
    bound_max: np.ndarray,
    resolution: int,  # cell/m
    threshold: float,
    query_func: Callable,
    batch_size: int = 2**15,
):
    x, y, z = [
        np.linspace(l, u, resolution, dtype=np.float32)
        for l, u in zip(bound_min, bound_max)
    ]

    grid = np.meshgrid(x, y, z, indexing="ij")
    xyz = np.stack(grid, axis=-1).reshape(-1, 3)

    sd = []
    xyz = np.array_split(xyz, np.ceil(xyz.shape[0] / batch_size))
    for pts in xyz:
        sd.append(query_func(pts))
    sd = np.concatenate(sd).reshape([resolution] * 3)

    vertices, triangles = mcubes.marching_cubes(sd, threshold)
    vertices = vertices / (resolution - 1.0) * (bound_max - bound_min) + bound_min

    return vertices, triangles
