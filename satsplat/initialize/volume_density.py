import numpy as np
import torch

from .base_initializer import BaseInitializer, BasePipeline
from ..module.gaussians import GaussianModel
from ..geometryutil import compute_scale_knn
from typing import Optional


class VolumeDensityInitializer(BaseInitializer):
    def __init__(
        self,
        point_density=0.13,  # num points per cubic meter
        init_opacity=0.01,
        init_scale: Optional[float] = 1.0,  # radius in world units
        feature_dim=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.point_density = point_density
        self.init_scale = init_scale
        self.init_opacity = init_opacity
        self.feature_dim = feature_dim

    def initialize(self, pipeline: BasePipeline):
        dataset = pipeline.dataset

        min_xyz = torch.from_numpy(dataset.site_min_scene.astype(np.float32))  # (3,)
        max_xyz = torch.from_numpy(dataset.site_max_scene.astype(np.float32))  # (3,)
        extent_xyz = max_xyz - min_xyz  # (3,)

        volume_m3 = torch.prod(extent_xyz) * dataset.site_scale**3  # Volume of the scene in cubic meters
        N = int(volume_m3 * self.point_density)  # Number of points to initialize

        points = (
            torch.rand((N, 3), dtype=torch.float32) * extent_xyz + min_xyz
        )  # [N, 3] - Random points in the scene
        if self.init_scale is None:
            scales = compute_scale_knn(points)  # [N, 3] - Scale based on k-NN distance
        else:
            scales = torch.full((N, 3), self.init_scale / dataset.site_scale, dtype=torch.float32)

        colors = torch.ones(
            (N, pipeline.color_dim), dtype=torch.float32
        )  # [N, 3] - RGB values initialized to white

        feats = torch.zeros((N, pipeline.feature_dim), dtype=torch.float32)

        quats = torch.zeros((N, 4), dtype=torch.float32)
        quats[:, 0] = 1.0

        opacity = torch.full((N,), self.init_opacity)  # [N,]

        G = GaussianModel(
            utm_epsg=dataset.site_utm_epsg, offset=dataset.site_offset_utm, scale=dataset.site_scale
        )
        G.means = points
        G.scales = scales
        G.quats = quats
        G.opacities = opacity
        G.colors = colors
        G.features = feats

        return G
