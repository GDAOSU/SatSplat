from datetime import datetime, timezone
import math
from typing import Optional, Dict, Tuple, Literal

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F


class AffineCamera(torch.nn.Module):
    idx: Tensor
    appearance_id: Tensor

    A: Tensor
    b: Tensor
    width: Tensor
    height: Tensor
    sun_dir: Tensor
    min_scene: Tensor
    max_scene: Tensor
    corners_scene: Tensor
    acquisition_date: Optional[datetime]

    @classmethod
    def sun_dir_from_datetime(cls, datetime: datetime, lon: float, lat: float) -> torch.Tensor:
        from ephem import Observer, Sun

        dt = datetime
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        obs = Observer()
        obs.lat = str(lat)
        obs.lon = str(lon)
        obs.date = dt
        sun = Sun()
        sun.compute(obs)
        sun_el_deg_date = math.degrees(float(sun.alt))
        sun_az_deg_date = math.degrees(float(sun.az))
        return cls.sun_dir_from_elev_azimuth_degrees(sun_az_deg_date, sun_el_deg_date)

    @classmethod
    def sun_dir_from_elev_azimuth_degrees(cls, azimuth_deg: float, elevation_deg: float) -> torch.Tensor:
        az = np.deg2rad(azimuth_deg)
        el = np.deg2rad(elevation_deg)
        x = np.cos(el) * np.sin(az)
        y = np.cos(el) * np.cos(az)
        z = np.sin(el)
        return np.array([x, y, z])

    @staticmethod
    def resample_by_depth(
        dst_cam: Dict,
        dst_depth: Tensor,
        src_cam: Dict,
        src_depth: Tensor,
        src_feat: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        # Mapping from dst space (u,v,d,1) to src space (u,v,d)
        M = src_cam["V"][..., :3, :] @ torch.linalg.inv(dst_cam["V"])  # [3, 4]

        # src_h, src_w = src_cam["shape"]
        src_depth = src_depth.reshape(src_cam["shape"])
        dst_h, dst_w = dst_cam["shape"]
        dst_depth = dst_depth.reshape(dst_cam["shape"])
        u, v = torch.meshgrid(
            torch.linspace(-1, 1, dst_w, device=dst_depth.device),
            torch.linspace(-1, 1, dst_h, device=dst_depth.device),
            indexing="xy",
        )
        dst_uvd1 = torch.stack([u, v, dst_depth, torch.ones_like(dst_depth)], dim=-1)
        projected_uvd1_on_src = dst_uvd1 @ M.T

        resampled_src_depth = F.grid_sample(
            input=src_depth[None, None],
            grid=projected_uvd1_on_src[..., :2].unsqueeze(0),
            mode="bilinear",
            align_corners=True,
        )[0, 0]

        resampled_src_feat = None
        if src_feat is not None:
            resampled_src_feat = F.grid_sample(
                input=src_feat.permute(2, 0, 1)[None],
                grid=projected_uvd1_on_src[..., :2].unsqueeze(0),
                mode="bilinear",
                align_corners=True,
            )[0].permute(1, 2, 0)

        result = {
            "projected_dst_depth": projected_uvd1_on_src[..., 2:3],  # (H, W, 1)
            "projected_dst_uv": projected_uvd1_on_src[..., :2],  # (H, W, 2)
            "resampled_src_depth": resampled_src_depth.unsqueeze(-1),  # (H, W, 1)
        }

        if src_feat is not None:
            result["resampled_src_feat"] = resampled_src_feat  # (H,W,C)

        return result

    def to_device(self, device):
        self.device = device
        for key, value in self.__dict__.items():
            if isinstance(value, Tensor):
                setattr(self, key, value.to(device))
        return self

    def __init__(
        self,
        width,
        height,
        A: np.array,
        b: np.array,
        sun_dir: np.array,
        min_scene: np.ndarray,
        max_scene: np.ndarray,
        idx: int = 0,
        appearance_id: float = 0,
        acquisition_date: datetime = None,
        device=None,
    ):
        super().__init__()  # Initialize torch.nn.Module properly
        self.device = device or "cpu"
        self.width = torch.tensor(width, dtype=torch.int32, device=device, requires_grad=False)
        self.height = torch.tensor(height, dtype=torch.int32, device=device, requires_grad=False)
        self.A = torch.tensor(A, dtype=torch.float32, device=device, requires_grad=False).reshape(3, 3)
        self.b = torch.tensor(b, dtype=torch.float32, device=device, requires_grad=False).reshape(3, 1)
        self.sun_dir = torch.tensor(
            sun_dir / sun_dir[-1], dtype=torch.float32, device=device, requires_grad=False
        ).reshape(3, 1)

        self.min_scene = torch.tensor(
            min_scene, dtype=torch.float32, device=device, requires_grad=False
        ).reshape(3)
        self.max_scene = torch.tensor(
            max_scene, dtype=torch.float32, device=device, requires_grad=False
        ).reshape(3)
        self.corners_scene = torch.tensor(
            [
                [min_scene[0], min_scene[1], min_scene[2]],
                [min_scene[0], max_scene[1], min_scene[2]],
                [max_scene[0], max_scene[1], min_scene[2]],
                [max_scene[0], min_scene[1], min_scene[2]],
                [min_scene[0], min_scene[1], max_scene[2]],
                [min_scene[0], max_scene[1], max_scene[2]],
                [max_scene[0], max_scene[1], max_scene[2]],
                [max_scene[0], min_scene[1], max_scene[2]],
            ],
            dtype=torch.float32,
            device=device,
            requires_grad=False,
        )  # [8, 3]

        self.idx = torch.tensor(idx, dtype=torch.int32, device=device, requires_grad=False)
        self.appearance_id = torch.tensor(
            appearance_id, dtype=torch.float32, device=device, requires_grad=False
        )

        self.acquisition_date = acquisition_date

    def compute_invert_z_depth_coefficients(self):
        with torch.no_grad():
            A_z = torch.tensor([0, 0, -1], dtype=torch.float32, device=self.device)
            b_z = self.max_scene[2]
            near = 0
            far = self.max_scene[2] - self.min_scene[2]
        return A_z, b_z, near, far

    def compute_orthogonal_depth_coefficients(self, A: torch.Tensor):
        """
        Args:
            A: [..., 2, 3]
        return:
            A_z: Linear term for output z [..., 1, 3]
            b_z: Bias term for output z [..., 1]
            near: Near clipping plane [..., 1]
            far: Far clipping plane [..., 1]
        """
        assert A.shape[-1] == 3
        with torch.no_grad():
            A_z = torch.cross(A[..., 0:1, :], A[..., 1:2, :], dim=-1)
            A_z /= torch.norm(A_z, dim=-1, keepdim=True)

            depth_prebias = torch.einsum("...j,nj->...n", A_z, self.corners_scene)
            b_z = -torch.min(depth_prebias, dim=-1).values
            near = 0
            far = torch.max(depth_prebias, dim=-1).values + b_z

        return A_z, b_z, near, far

    def get_main_camera(
        self,
        depth_func: Literal["invert_z", "orthogonal"] = "invert_z",
        deltaA: Optional[Tensor] = None,
        deltab: Optional[Tensor] = None,
    ) -> dict:

        # K: from uv (-1,1) to image (0,0)-(width, height)
        K = torch.tensor(
            [[self.width / 2, 0, self.width / 2], [0, self.height / 2, self.height / 2], [0, 0, 1]],
            device=self.device,
        )

        A_composed = self.A[..., :2, :3]
        b_composed = self.b[..., :2, :]
        if deltaA is not None:
            A_composed = A_composed + deltaA
        if deltab is not None:
            b_composed = b_composed + deltab

        if depth_func == "orthogonal":
            A_z, b_z, near, far = self.compute_orthogonal_depth_coefficients(A_composed)
        elif depth_func == "invert_z":
            A_z, b_z, near, far = self.compute_invert_z_depth_coefficients()

        V = torch.eye(4, device=self.device)
        V[..., :2, :3] = A_composed
        V[..., 2:3, :3] = A_z
        V[..., :2, 3:] = b_composed
        V[..., 2, 3] = b_z.squeeze(-1)

        return {
            "K": K,
            "V": V,
            # "Mxy1d": Mxy1d,
            # "Muv1d": Muv1d,
            "shape": (self.height, self.width),
            "near": near,
            "far": far * 2,
        }

    def get_sun_camera(
        self,
        transformScale: float = 1.5,
        depth_func: Literal["invert_z", "orthogonal"] = "invert_z",
        deltaA: Optional[Tensor] = None,
        deltab: Optional[Tensor] = None,
    ) -> dict:
        """
        transformScale: enlarge the rendered image by this scale factor, keep the resolution same. Thus, coverage being scaled.
        """
        offset_x = int(self.width * (transformScale - 1) / 2)
        offset_y = int(self.height * (transformScale - 1) / 2)
        new_width = self.width + 2 * offset_x
        new_height = self.height + 2 * offset_y

        # K: from uv (-1,1) to image (0,0)-(width, height)
        K = torch.tensor(
            [[new_width / 2, 0, new_width / 2], [0, new_height / 2, new_height / 2], [0, 0, 1]],
            device=self.device,
        )
        # Project all points to z=0 plane, yield an offset vector on 2d image that only affected by input z (altitude).
        A_composed = self.A[..., :2, :3].clone()
        b_composed = self.b[..., :2, :]

        if deltaA is not None:
            A_composed = A_composed + deltaA
        if deltab is not None:
            b_composed = b_composed + deltab

        sun_proj_offset = -torch.einsum("...ij,...jk->...ik", A_composed, self.sun_dir)
        A_composed[..., :2, 2:3] += sun_proj_offset

        if depth_func == "orthogonal":
            A_z, b_z, near, far = self.compute_orthogonal_depth_coefficients(A_composed)
        elif depth_func == "invert_z":
            A_z, b_z, near, far = self.compute_invert_z_depth_coefficients()

        V = torch.eye(4, device=self.device)
        V[..., :2, :3] = A_composed
        V[..., 2:3, :3] = A_z
        V[..., :2, 3:] = b_composed
        V[..., 2, 3] = b_z.squeeze(-1)

        V[..., 0, :] *= self.width / new_width
        V[..., 1, :] *= self.height / new_height

        return {
            "K": K,
            "V": V,
            "shape": (new_height, new_width),
            "near": near,
            "far": far * 2,
        }

    def get_random_camera(
        self,
        extent=0.05,
        transformScale: float = 1.5,
        depth_func: Literal["invert_z", "orthogonal"] = "invert_z",
        deltaA: Optional[Tensor] = None,
        deltab: Optional[Tensor] = None,
    ) -> dict:
        offset_x = int(self.width * (transformScale - 1) / 2)
        offset_y = int(self.height * (transformScale - 1) / 2)
        new_width = self.width + 2 * offset_x
        new_height = self.height + 2 * offset_y

        # K: from uv (-1,1) to image (0,0)-(width, height)
        K = torch.tensor(
            [[new_width / 2, 0, new_width / 2], [0, new_height / 2, new_height / 2], [0, 0, 1]],
            device=self.device,
        )

        random_offset = torch.clip(torch.randn((2, 1), device=self.device), -1, 1) * extent

        A_composed = self.A[..., :2, :3].clone()
        b_composed = self.b[..., :2, :]

        if deltaA is not None:
            A_composed = A_composed + deltaA
        if deltab is not None:
            b_composed = b_composed + deltab

        A_composed[..., :2, 2:3] += random_offset

        if depth_func == "orthogonal":
            A_z, b_z, near, far = self.compute_orthogonal_depth_coefficients(A_composed)
        elif depth_func == "invert_z":
            A_z, b_z, near, far = self.compute_invert_z_depth_coefficients()

        V = torch.eye(4, device=self.device)
        V[..., :2, :3] = A_composed
        V[..., 2:3, :3] = A_z
        V[..., :2, 3:] = b_composed
        V[..., 2, 3] = b_z.squeeze(-1)

        V[..., 0, :] *= self.width / new_width
        V[..., 1, :] *= self.height / new_height

        return {
            "K": K,
            "V": V,
            "shape": (new_height, new_width),
            "near": near,
            "far": far * 2,
        }
