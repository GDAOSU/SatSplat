from pathlib import Path
from typing import Literal, Optional, Tuple

import logging

import numpy as np
import numpy.typing as npt
import pyproj
import rasterio as rio
import rasterio.warp as rio_warp
import torch
from affine import Affine
from scipy import ndimage

from .base_initializer import BaseInitializer, BasePipeline
from ..dataset import SatDataset
from ..module.gaussians import GaussianModel
from ..geometryutil import quat_from_two_vectors, compute_scale_knn

logger = logging.getLogger(__name__)


class DSMInitializer(BaseInitializer):
    """
    Initializer that creates Gaussian splats from a DSM (Digital Surface Model) and ortho images.

    This initializer generates 3D Gaussian splats by:
    - Using DSM for 3D geometry (X, Y, Z coordinates)
    - Using ortho images for appearance (color)
    - Initializing scales based on DSM resolution
    - Setting identity rotations and uniform opacities

    Multiple ortho images can be provided - they will be aligned to the DSM grid and
    median-averaged if more than 2 are provided.

    If no ortho images are provided (empty list), all splats will be initialized with
    white color.

    Example usage:
        # Initialize from DSM and single ortho
        initializer = DSMInitializer(
            dsm_path="path/to/dsm.tif",
            ortho_paths=["path/to/ortho.tif"]
        )

        # Initialize from DSM and multiple orthos (median-averaged)
        initializer = DSMInitializer(
            dsm_path="path/to/dsm.tif",
            ortho_paths=["path/to/ortho1.tif", "path/to/ortho2.tif", "path/to/ortho3.tif"],
            default_scale=0.5  # Smaller scale factor for finer splats
        )

        # Initialize from DSM only with white colors
        initializer = DSMInitializer(
            dsm_path="path/to/dsm.tif",
            ortho_paths=[]  # Empty list = white colors
        )
    """

    def __init__(
        self,
        default_opacity: float = 0.9,
        default_scale: Optional[float] = None,
        num_samples: int = 50000,
        sampling_method: Literal["uniform", "area_weighted"] = "area_weighted",
        seed: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            dsm_path: Path to the DSM file
            ortho_paths: List of paths to ortho image files (empty list = white colors)
            default_opacity: Initial opacity value for all splats (default: 0.9)
            default_scale: Scale factor relative to DSM resolution (default: 1.0)
            num_samples: Number of surface samples to draw from the DSM grid
            sampling_method: Strategy for selecting DSM pixels ("uniform" or "area_weighted")
            seed: Random seed for surface sampling
        """
        super().__init__(**kwargs)
        self.default_scale = default_scale
        self.default_opacity = default_opacity
        self.num_samples = num_samples
        self.seed = seed
        valid_sampling_methods = {"uniform", "area_weighted"}
        if sampling_method not in valid_sampling_methods:
            raise ValueError(
                f"Unsupported sampling_method '{sampling_method}'. Supported methods: {sorted(valid_sampling_methods)}"
            )
        self.sampling_method = sampling_method

    @classmethod
    def _fill_missing_with_nearest(
        cls, data: npt.NDArray[np.floating], mask: npt.NDArray[np.bool_]
    ) -> npt.NDArray[np.floating]:
        """Nearest-neighbour fill for NaNs so gradient ops behave."""
        if mask.all():
            return data
        nearest_indices = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
        filled = data[tuple(nearest_indices)]
        return filled

    def _load_dsm(self):
        """Load DSM data and metadata."""
        with rio.open(self.dsm_path) as src:
            dsm = src.read(1)
            dsm[dsm == src.nodata] = np.nan
            dsmheight, dsmwidth = src.height, src.width
            dsmtf = src.transform
            dsmcrs = src.crs
            dsmres = src.res

        return dsm, dsmheight, dsmwidth, dsmtf, dsmcrs, dsmres

    def _load_and_align_orthos(self, pipeline: BasePipeline, dsmheight: int, dsmwidth: int, dsmtf, dsmcrs):
        """Load ortho images and align them to DSM grid. Returns white if no orthos provided."""
        # If no ortho paths provided, create white colors
        if not self.ortho_paths:
            logger.info("No ortho images provided, initializing with white colors")
            white_ortho = np.ones((dsmheight, dsmwidth, pipeline.color_dim), dtype=np.float32)
            return white_ortho

        dataset = pipeline.dataset
        ortho_stack = []

        for ortho_path in self.ortho_paths:
            with rio.open(ortho_path) as src:
                # Read and resample to align with dsm
                ortho = src.read(range(1, pipeline.color_dim + 1)).astype(np.float32)
                aligned_ortho = np.empty((ortho.shape[0], dsmheight, dsmwidth), dtype=np.float32)

                rio_warp.reproject(
                    source=ortho,
                    destination=aligned_ortho,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dsmtf,
                    dst_crs=dsmcrs,
                    resampling=rio.warp.Resampling.bilinear,
                )
                aligned_ortho = dataset.preprocess_img(aligned_ortho.transpose(1, 2, 0))
                ortho_stack.append(aligned_ortho)

        # Median-average if multiple orthos, otherwise use single ortho
        if len(ortho_stack) > 2:
            ortho = np.median(np.stack(ortho_stack, axis=0), axis=0)
            logger.info(f"Median-averaged {len(ortho_stack)} ortho images")
        else:
            ortho = ortho_stack[0]

        return ortho

    @classmethod
    def _compute_surface_metrics(
        cls, dsm: npt.NDArray[np.floating], transform: Affine
    ) -> dict[str, npt.NDArray[np.floating]]:
        """Compute per-pixel surface area and world-space normals for a DSM."""
        dsm = dsm.astype(np.float64, copy=False)
        mask = np.isfinite(dsm)
        dsm_filled = cls._fill_missing_with_nearest(dsm, mask)

        dz_drow, dz_dcol = np.gradient(dsm_filled, edge_order=2)

        # Invalidate gradients that touch nodata so downstream area/normal stay masked.
        row_valid = mask.copy()
        row_valid[:1, :] = False
        row_valid[-1:, :] = False
        row_valid[:-1, :] &= mask[1:, :]
        row_valid[1:, :] &= mask[:-1, :]
        dz_drow[~row_valid] = np.nan

        col_valid = mask.copy()
        col_valid[:, :1] = False
        col_valid[:, -1:] = False
        col_valid[:, :-1] &= mask[:, 1:]
        col_valid[:, 1:] &= mask[:, :-1]
        dz_dcol[~col_valid] = np.nan

        dx_col, dy_col = transform.a, transform.d
        dx_row, dy_row = transform.b, transform.e

        t_col = np.stack(
            (
                np.full_like(dsm, dx_col),
                np.full_like(dsm, dy_col),
                dz_dcol,
            ),
            axis=-1,
        )
        t_row = np.stack(
            (
                np.full_like(dsm, dx_row),
                np.full_like(dsm, dy_row),
                dz_drow,
            ),
            axis=-1,
        )

        cross = np.cross(t_col, t_row)
        area = np.linalg.norm(cross, axis=-1)

        normals = cross / (area[..., None] + 1e-12)
        normals[~mask] = np.nan
        flip_mask = normals[..., 2] < 0
        normals[flip_mask] *= -1

        area[~mask] = np.nan

        gsd_x = float(np.hypot(dx_col, dy_col))
        gsd_y = float(np.hypot(dx_row, dy_row))

        return {
            "area": area.astype(np.float32),
            "normals": normals.astype(np.float32),
            "mask": mask,
            "gsd": np.array([gsd_x, gsd_y], dtype=np.float32),
        }

    @classmethod
    def _build_weight_map(
        cls,
        area: npt.NDArray[np.floating],
        mask: npt.NDArray[np.bool_],
        sampling_method: Literal["uniform", "area_weighted"],
    ) -> npt.NDArray[np.floating]:
        """Construct sampling weights for the specified strategy."""
        weights = np.zeros_like(area, dtype=np.float64)
        valid = mask.copy()

        if sampling_method == "area_weighted":
            valid &= np.isfinite(area)
            weights[valid] = area[valid]
        elif sampling_method == "uniform":
            valid &= np.isfinite(area)
            weights[valid] = 1.0
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}")

        total = weights.sum()
        if total <= 0:
            raise ValueError("No valid pixels available to build a weight map.")

        weights /= total
        return weights.astype(np.float32)

    @classmethod
    def _pixel_to_world(
        cls, transform: Affine, rows: npt.NDArray[np.int64], cols: npt.NDArray[np.int64]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Map pixel indices (row, col) to map coordinates at pixel centre."""
        cols_f = cols.astype(np.float64) + 0.5
        rows_f = rows.astype(np.float64) + 0.5
        xs = transform.c + transform.a * cols_f + transform.b * rows_f
        ys = transform.f + transform.d * cols_f + transform.e * rows_f
        return xs, ys

    @classmethod
    def _sample_surface_points(
        cls,
        dsm: npt.NDArray[np.floating],
        transform: Affine,
        weight_map: npt.NDArray[np.floating],
        normals: npt.NDArray[np.floating],
        ortho: npt.NDArray[np.floating],
        num_samples: int,
        rng: Optional[np.random.Generator] = None,
    ) -> dict[str, npt.NDArray[np.floating]]:
        """Sample pseudo-uniform 3D surface points, normals, and colours."""
        if rng is None:
            rng = np.random.default_rng()

        valid = np.isfinite(weight_map) & (weight_map > 0)
        rows, cols = np.where(valid)
        if rows.size == 0:
            raise ValueError("Weight map does not contain any valid pixels.")

        probs = weight_map[rows, cols]
        probs = probs / probs.sum()

        replace = num_samples > rows.size
        sample_size = num_samples if replace else min(num_samples, rows.size)
        choice_idx = rng.choice(rows.size, size=sample_size, replace=replace, p=probs)
        sel_rows = rows[choice_idx]
        sel_cols = cols[choice_idx]

        xs, ys = cls._pixel_to_world(transform, sel_rows, sel_cols)
        zs = dsm[sel_rows, sel_cols]
        points = np.stack([xs, ys, zs], axis=-1)

        sampled_normals = normals[sel_rows, sel_cols]
        sampled_colors = ortho[sel_rows, sel_cols]

        return {
            "points": points.astype(np.float32),
            "normals": sampled_normals.astype(np.float32),
            "colors": sampled_colors.astype(np.float32),
            "rows": sel_rows,
            "cols": sel_cols,
        }

    @classmethod
    def _query_nearest_surface_attributes(
        cls,
        target_xy: Tuple[float, float],
        dsm: npt.NDArray[np.floating],
        transform: Affine,
        normals: npt.NDArray[np.floating],
        ortho: npt.NDArray[np.floating],
        weight_map: Optional[npt.NDArray[np.floating]] = None,
    ) -> Optional[dict[str, np.ndarray]]:
        """Return position, normal, colour, and (optional) weight for the nearest DSM pixel."""
        inv_transform = ~transform
        col_f, row_f = inv_transform * target_xy
        row = int(np.clip(np.round(row_f), 0, dsm.shape[0] - 1))
        col = int(np.clip(np.round(col_f), 0, dsm.shape[1] - 1))
        if not np.isfinite(dsm[row, col]):
            return None

        x, y = cls._pixel_to_world(transform, np.array([row]), np.array([col]))
        z = float(dsm[row, col])
        normal = normals[row, col]
        color = ortho[row, col]

        result = {
            "point": np.array([x[0], y[0], z], dtype=np.float32),
            "normal": normal.astype(np.float32),
            "color": color.astype(np.float32),
            "pixel_index": (row, col),
        }
        if weight_map is not None:
            result["weight"] = float(weight_map[row, col])
        return result

    def _generate_splats(
        self,
        dsm: np.ndarray,
        ortho: np.ndarray,
        dsmtf: Affine,
        dsmcrs: int,
        dataset: SatDataset,
        num_samples: int,
    ):
        """Generate 3D Gaussian splat parameters from DSM and ortho data."""
        metrics = self._compute_surface_metrics(dsm, dsmtf)
        area_map = metrics["area"]
        normal_map = metrics["normals"]
        mask = metrics["mask"]
        gsd = metrics["gsd"]

        weight_map = self._build_weight_map(area_map, mask, self.sampling_method)

        samples = self._sample_surface_points(
            dsm=dsm,
            transform=dsmtf,
            weight_map=weight_map,
            normals=normal_map,
            ortho=ortho,
            num_samples=num_samples,
            rng=np.random.default_rng(self.seed),
        )

        xyz = samples["points"]  # (N, 3)
        normal = samples["normals"]  # (N, 3)
        color = samples["colors"]  # (N, C)

        num_points = xyz.shape[0]
        if num_points == 0:
            raise ValueError("DSM sampling produced zero valid points. Check input DSM/ortho alignment.")

        # Initialize opacity
        opacity = np.full((num_points,), self.default_opacity, dtype=np.float32)

        if self.default_scale is None:
            scale = compute_scale_knn(torch.from_numpy(xyz)).numpy()
        else:
            # Interpret default_scale as a multiplier of the DSM ground sample distance so
            # callers can specify scale in meters regardless of dataset normalization.
            mean_gsd = float(np.mean(gsd))
            scale_value = float(self.default_scale) * mean_gsd
            scale = np.full((num_points, 3), scale_value, dtype=np.float32)

        # Initialize rotations (identity quaternions)
        # initialize quats by normal, make the up vector (0, 0, 1) rotate to normal
        quats = np.zeros((num_points, 4), dtype=np.float32)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        for i in range(num_points):
            n = normal[i]
            if not np.isfinite(n).all() or np.linalg.norm(n) < 1e-6:
                n = up
            q = quat_from_two_vectors(up, n)
            quats[i] = q

        # Convert to dataset's coordinate system if needed
        dsm_epsg = dsmcrs.to_epsg() if dsmcrs is not None else None
        target_epsg = getattr(dataset, "site_utm_epsg", None)
        if dsm_epsg is not None and target_epsg is not None and dsm_epsg != target_epsg:
            transformer = pyproj.Transformer.from_crs(dsmcrs, target_epsg, always_xy=True)
            xs, ys = transformer.transform(xyz[:, 0], xyz[:, 1])
            xyz[:, 0] = xs
            xyz[:, 1] = ys

        # Apply dataset scaling
        xyz = (xyz - dataset.site_offset_utm) / dataset.site_scale
        scale = scale / dataset.site_scale

        # Remove nan points and out of bound points
        inbounds_mask = np.all((xyz >= dataset.site_min_scene) & (xyz <= dataset.site_max_scene), axis=-1)
        valid_mask = inbounds_mask & (~np.isnan(xyz).any(axis=-1))

        xyz = xyz[valid_mask]
        quats = quats[valid_mask]
        color = color[valid_mask]
        scale = scale[valid_mask]
        opacity = opacity[valid_mask]

        return xyz, quats, color, scale, opacity

    def initialize(self, pipeline: BasePipeline) -> GaussianModel:
        """Initialize GaussianModel from DSM and ortho data."""
        dataset = pipeline.dataset
        assert dataset is not None, "Pipeline must have a dataset"
        assert dataset.init_dsm is not None, "Dataset must specify 'init_dsm' path for DSMInitializer"
        assert dataset.init_orthos is not None, "Dataset must specify 'init_orthos' paths for DSMInitializer"

        self.dsm_path = Path(dataset.init_dsm)
        self.ortho_paths = [Path(p) for p in dataset.init_orthos]

        logger.info(f"Loading DSM from: {self.dsm_path}")
        logger.info(f"Loading {len(self.ortho_paths)} ortho image(s)")

        # Load DSM for geometry
        dsm, dsmheight, dsmwidth, dsmtf, dsmcrs, dsmres = self._load_dsm()

        # Load and align ortho images for appearance
        ortho = self._load_and_align_orthos(pipeline, dsmheight, dsmwidth, dsmtf, dsmcrs)

        # Generate splat parameters
        xyz, quats, color, scale, opacity = self._generate_splats(
            dsm, ortho, dsmtf, dsmcrs, dataset, self.num_samples
        )

        logger.info(f"Generated {xyz.shape[0]} Gaussian splats from DSM/ortho")
        logger.info(f"Mean scale: {np.mean(scale):.4f}, DSM resolution: {np.mean(dsmres):.2f}")

        # Convert to tensors
        means_tensor = torch.from_numpy(xyz.astype(np.float32))
        colors_tensor = torch.from_numpy(color.astype(np.float32))
        opacities_tensor = torch.from_numpy(opacity.astype(np.float32))
        scales_tensor = torch.from_numpy(scale.astype(np.float32))
        quats_tensor = torch.from_numpy(quats.astype(np.float32))
        feats_tensor = torch.zeros((means_tensor.shape[0], pipeline.feature_dim), dtype=torch.float32)

        # Create GaussianModel
        G = GaussianModel(
            utm_epsg=dataset.site_utm_epsg, offset=dataset.site_offset_utm, scale=dataset.site_scale
        )

        # Set parameters
        G.means = means_tensor
        G.opacities = opacities_tensor
        G.scales = scales_tensor
        G.quats = quats_tensor
        G.colors = colors_tensor
        G.features = feats_tensor

        logger.info(f"Initialized GaussianModel with {G.n_gaussians} gaussians")
        return G
