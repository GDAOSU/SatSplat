import os
import torch
from torch import nn
from omegaconf import DictConfig

from gsplat.optimizers import SelectiveAdam
from ..ioutil import save_ply
from ..geometryutil import quat_to_rotation_matrix, rotation_matrix_to_quat

import logging

logger = logging.getLogger(__name__)


class GaussianModel(nn.Module):
    _mean_name = "means"
    _scale_name = "scales"
    _quat_name = "quats"
    _color_name = "colors"
    _opacity_name = "opacities"
    _feature_name = "features"

    def __init__(self, utm_epsg: int, offset: torch.Tensor, scale: torch.Tensor) -> None:
        super().__init__()
        self.utm_epsg = utm_epsg
        self.offset = offset
        self.scale = scale

        self.gaussians = nn.ParameterDict()
        self.optimizers = {}
        self.lr_schedulers = {}

        self.opacity_activation = torch.sigmoid
        self.opacity_inverse_activation = torch.logit

        self.scale_activation = torch.exp
        self.scale_inverse_activation = torch.log

    @property
    def means(self) -> torch.Tensor:
        return self.gaussians[self._mean_name]

    @means.setter
    def means(self, value: torch.Tensor):
        self.gaussians[self._mean_name] = value

    @property
    def scales(self) -> torch.Tensor:
        return self.scale_activation(self.gaussians[self._scale_name])

    @scales.setter
    def scales(self, value: torch.Tensor):
        self.gaussians[self._scale_name] = self.scale_inverse_activation(value)

    @property
    def quats(self) -> torch.Tensor:
        return self.gaussians[self._quat_name]

    @quats.setter
    def quats(self, value: torch.Tensor):
        self.gaussians[self._quat_name] = value

    @property
    def colors(self) -> torch.Tensor:
        return self.gaussians[self._color_name]

    @colors.setter
    def colors(self, value: torch.Tensor):
        self.gaussians[self._color_name] = value

    @property
    def opacities(self) -> torch.Tensor:
        return self.opacity_activation(self.gaussians[self._opacity_name])

    @opacities.setter
    def opacities(self, value: torch.Tensor):
        self.gaussians[self._opacity_name] = self.opacity_inverse_activation(value)

    @property
    def features(self) -> torch.Tensor:
        return self.gaussians[self._feature_name]

    @features.setter
    def features(self, value: torch.Tensor):
        self.gaussians[self._feature_name] = value

    def get_n_gaussians(self) -> int:
        return self.gaussians[next(iter(self.gaussians))].shape[0]

    @property
    def n_gaussians(self) -> int:
        return self.get_n_gaussians()

    def sort_scales(self):
        """
        Sort scales so that the minimum axis is placed last (largest two first).
        This is useful for 2DGS initialization where only the first two scales are used.
        Also handles the rotation quaternion correspondingly to maintain proper orientation.
        """
        # Get current scales (activated)
        current_scales = self.scales  # Shape: [N, 3]

        # Sort scales in descending order and get permutation indices
        sorted_scales, sort_indices = torch.sort(current_scales, dim=1, descending=True)

        # Update scales with sorted values
        self.scales = sorted_scales

        # Handle rotation quaternion permutation
        if self._quat_name in self.gaussians:
            current_quats = self.quats  # Shape: [N, 4] (w, x, y, z)

            # Create permutation matrices for each Gaussian
            # We need to permute the rotation matrix corresponding to each quaternion
            batch_size = current_quats.shape[0]

            # Convert quaternions to rotation matrices
            rotation_matrices = quat_to_rotation_matrix(current_quats)  # [N, 3, 3]

            # Apply permutation to each rotation matrix
            permuted_matrices = torch.zeros_like(rotation_matrices)
            for i in range(batch_size):
                perm = sort_indices[i]  # [3] - permutation for this Gaussian
                # Permute both rows and columns of the rotation matrix
                permuted_matrices[i] = rotation_matrices[i][perm][:, perm]

            # Convert back to quaternions
            new_quats = rotation_matrix_to_quat(permuted_matrices)
            self.quats = new_quats

        logger.info("Sorted scales for 2DGS initialization - minimum axis moved to last position")

    def save_ply(self, path: os.PathLike):
        save_ply(self.gaussians, path)

    def training_setup(self, lr_cfg: DictConfig):
        optimizer_class = None
        if lr_cfg.sparse_grad:
            optimizer_class = torch.optim.SparseAdam
        elif lr_cfg.visible_adam:
            optimizer_class = SelectiveAdam
        else:
            optimizer_class = torch.optim.Adam

        self.optimizers[self._mean_name] = optimizer_class(
            [
                {
                    "params": self.gaussians[self._mean_name],
                    "lr": lr_cfg.means_lr / self.scale,
                    "name": self._mean_name,
                }
            ],
            lr=0.0,
            eps=1e-15,
        )
        self.optimizers[self._scale_name] = optimizer_class(
            [{"params": self.gaussians[self._scale_name], "lr": lr_cfg.scales_lr, "name": self._scale_name}],
            lr=0.0,
            eps=1e-15,
        )
        self.optimizers[self._quat_name] = optimizer_class(
            [{"params": self.gaussians[self._quat_name], "lr": lr_cfg.quats_lr, "name": self._quat_name}],
            lr=0.0,
            eps=1e-15,
        )
        self.optimizers[self._opacity_name] = optimizer_class(
            [
                {
                    "params": self.gaussians[self._opacity_name],
                    "lr": lr_cfg.opacities_lr,
                    "name": self._opacity_name,
                }
            ],
            lr=0.0,
            eps=1e-15,
        )
        self.optimizers[self._color_name] = optimizer_class(
            [{"params": self.gaussians[self._color_name], "lr": lr_cfg.color_lr, "name": self._color_name}],
            lr=0.0,
            eps=1e-15,
        )
        if self._feature_name in self.gaussians:
            self.optimizers[self._feature_name] = optimizer_class(
                [
                    {
                        "params": self.gaussians[self._feature_name],
                        "lr": lr_cfg.feature_lr,
                        "name": self._feature_name,
                    }
                ],
                lr=0.0,
                eps=1e-15,
            )

    def optimizer_step(self):
        for optimizer in self.optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
