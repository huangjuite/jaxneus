import numpy as np
import mcubes
from tqdm import tqdm

from typing import Callable


def extract_geometry(
    bound_min: np.ndarray,
    bound_max: np.ndarray,
    resolution: int,  # cell/m
    threshold: float,
    query_func: Callable,
    batch_size: int = 2048,
):

    size = ((bound_max - bound_min) * resolution).astype(int)
    x, y, z = [
        np.linspace(l, u, s, dtype=np.float32)
        for l, u, s in zip(bound_min, bound_max, size)
    ]

    grid = np.meshgrid(x, y, z, indexing="ij")
    xyz = np.stack(grid, axis=-1).reshape(-1, 3)

    sd = []
    xyz = np.array_split(xyz, np.ceil(xyz.shape[0] / batch_size))
    pbar = tqdm(xyz)
    pbar.set_description("extracting mesh")
    for pts in pbar:
        sd.append(query_func(pts))
    sd = np.concatenate(sd).reshape(size)

    vertices, triangles = mcubes.marching_cubes(sd, threshold)
    vertices = vertices / resolution 

    return vertices, triangles
