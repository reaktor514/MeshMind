"""Marching-cubes isosurface extraction.

Originally part of TripoSR (https://github.com/VAST-AI-Research/TripoSR), MIT
licensed. Vendored into MeshMind and patched to drop the optional
``torchmcubes`` dependency in favour of ``skimage.measure.marching_cubes``,
which works out of the box on CPU-only environments.
"""

import numpy as np
import torch
import torch.nn as nn
from skimage import measure


class IsosurfaceHelper(nn.Module):
    points_range: tuple[float, float] = (0, 1)

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        raise NotImplementedError


class MarchingCubeHelper(IsosurfaceHelper):
    def __init__(self, resolution: int) -> None:
        super().__init__()
        self.resolution = resolution
        self._grid_vertices: torch.FloatTensor | None = None

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        if self._grid_vertices is None:
            x, y, z = (
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
            )
            x, y, z = torch.meshgrid(x, y, z, indexing="ij")
            verts = torch.cat(
                [x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1
            ).reshape(-1, 3)
            self._grid_vertices = verts
        return self._grid_vertices

    def forward(
        self,
        level: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        # ``level`` here is the negative of the signed-distance / density-
        # threshold field (see ``TSR.extract_mesh``). ``skimage`` extracts the
        # surface at ``level=0.0`` where positive values are *outside*.
        volume = -level.view(self.resolution, self.resolution, self.resolution)
        volume_np = volume.detach().cpu().numpy().astype(np.float32)
        v_min, v_max = float(volume_np.min()), float(volume_np.max())
        if not (v_min < 0.0 < v_max):
            empty_v = torch.zeros((0, 3), dtype=torch.float32, device=level.device)
            empty_f = torch.zeros((0, 3), dtype=torch.long, device=level.device)
            return empty_v, empty_f
        verts_np, faces_np, _normals, _values = measure.marching_cubes(
            volume_np, level=0.0
        )
        v_pos = torch.from_numpy(verts_np.astype(np.float32))
        t_pos_idx = torch.from_numpy(faces_np.astype(np.int64))
        # Match the channel ordering produced by torchmcubes.
        v_pos = v_pos[..., [2, 1, 0]]
        v_pos = v_pos / (self.resolution - 1.0)
        return v_pos.to(level.device), t_pos_idx.to(level.device)
