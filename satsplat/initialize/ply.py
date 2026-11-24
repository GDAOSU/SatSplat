import numpy as np
import torch
from pathlib import Path
import logging

from .base_initializer import BaseInitializer, BasePipeline
from ..module.gaussians import GaussianModel
from ..ioutil import read_ply_vertex_data, sh_to_color
from ..geometryutil import compute_scale_knn


logger = logging.getLogger(__name__)


class PlyInitializer(BaseInitializer):
    """
    Initializer that loads Gaussian splats from a PLY file and mimics volume density initialization.

    The PLY file must contain:
    - xyz coordinates (required)
    - Optionally: rgb colors (0-255 if byte, 0-1 if float) OR f_dc_* spherical harmonics
    - Optionally: opacities (raw values, processed with sigmoid internally)
    - Optionally: scales (raw values, processed with exp internally)
    - Optionally: rotation quaternions (rot_0, rot_1, rot_2, rot_3)

    If optional attributes are missing, they will be initialized similar to volume density:
    - Colors: white
    - Opacities: default_opacity value
    - Scales: estimated from nearest neighbor distances
    - Rotations: random quaternions

    Example usage:
        # Initialize from PLY file with auto-detection
        initializer = PlyVolumeDensityInitializer(
            ply_path="path/to/splats.ply",
            default_opacity=0.01,
            default_scale=1.0
        )

        # Force RGB color mode
        initializer = PlyVolumeDensityInitializer(
            ply_path="path/to/splats.ply",
            color_mode="rgb"
        )

        # Use white colors regardless of PLY content
        initializer = PlyVolumeDensityInitializer(
            ply_path="path/to/splats.ply",
            color_mode="white"
        )
    """

    def __init__(
        self,
        default_opacity: float = 0.5,
        default_scale: float = 1.0,
        color_mode: str = "auto",  # "auto", "rgb", "sh", "white"
        **kwargs,
    ):
        """
        Args:
            ply_path: Path to the PLY file
            default_opacity: Default opacity value if not present in PLY
            default_scale: Default scale value if not present in PLY
            color_mode: How to handle colors - "auto" detects, "rgb" forces RGB, "sh" forces SH, "white" uses white
        """
        super().__init__(**kwargs)
        self.default_opacity = default_opacity
        self.default_scale = default_scale
        self.color_mode = color_mode

    def _load_ply_data(self):
        """Load and parse PLY file data."""
        vertex = read_ply_vertex_data(self.ply_path)
        attrib_names = set(vertex.names)

        # Extract xyz coordinates (required)
        means = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
        N = means.shape[0]

        # Extract colors
        colors = self._extract_colors(vertex, attrib_names, N)

        # Extract opacities
        opacities = self._extract_opacities(vertex, attrib_names, N)

        # Extract scales
        scales = self._extract_scales(vertex, attrib_names, N, means)

        # Extract rotations
        quats = self._extract_rotations(vertex, attrib_names, N)

        return means, colors, opacities, scales, quats

    def _extract_colors(self, vertex, attrib_names: set, N: int) -> np.ndarray:
        """Extract or generate color data."""
        if self.color_mode == "white":
            return np.ones((N, 3), dtype=np.float32)

        # Check for RGB colors
        has_rgb = all(name in attrib_names for name in ["red", "green", "blue"])

        # Check for spherical harmonics
        dc_names = sorted([name for name in attrib_names if name.startswith("f_dc_")])
        has_sh = len(dc_names) >= 3

        if self.color_mode == "auto":
            # Auto-detect: prefer SH over RGB
            if has_sh:
                self.color_mode = "sh"
            elif has_rgb:
                self.color_mode = "rgb"
            else:
                self.color_mode = "white"

        if self.color_mode == "sh" and has_sh:
            # Load spherical harmonics and convert to RGB
            sh_coeffs = np.stack([vertex[name] for name in dc_names], axis=1).astype(np.float32)
            colors = sh_to_color(sh_coeffs)
            # Clamp to valid range
            colors = np.clip(colors, 0.0, 1.0)
            return colors

        elif self.color_mode == "rgb" and has_rgb:
            # Load RGB colors
            rgb = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.float32)

            # Detect if colors are in 0-255 range (byte) or 0-1 range (float)
            if rgb.max() > 1.0:
                rgb = rgb / 255.0  # Convert byte to float

            return np.clip(rgb, 0.0, 1.0)
        else:
            # Default to white if no colors available
            logger.warning("Warning: No suitable colors found in PLY file, using white")
            return np.ones((N, 3), dtype=np.float32)

    def _extract_opacities(self, vertex, attrib_names: set, N: int) -> np.ndarray:
        """Extract opacity data."""
        if "opacity" in attrib_names:
            # PLY file contains raw opacity values (before sigmoid)
            raw_opacities = vertex["opacity"].astype(np.float32)
            return raw_opacities
        else:
            # Use default opacity (in logit space for sigmoid activation)
            return np.full(N, torch.logit(torch.tensor(self.default_opacity)).item(), dtype=np.float32)

    def _extract_scales(self, vertex, attrib_names: set, N: int, means: np.ndarray) -> np.ndarray:
        """Extract scale data."""
        scale_names = sorted([name for name in attrib_names if name.startswith("scale_")])

        if len(scale_names) >= 3:
            # PLY file contains raw scales (before exp)
            scales = np.stack([vertex[name] for name in scale_names[:3]], axis=1).astype(np.float32)
            return scales
        elif len(scale_names) == 1:
            # Isotropic scaling
            scale_val = vertex[scale_names[0]].astype(np.float32)
            scales = np.column_stack([scale_val, scale_val, scale_val])
            return scales
        else:
            scales = compute_scale_knn(means)
            return scales

    def _extract_rotations(self, vertex, attrib_names: set, N: int) -> np.ndarray:
        """Extract rotation quaternions."""
        rot_names = sorted([name for name in attrib_names if name.startswith("rot_")])

        if len(rot_names) >= 4:
            # PLY file contains quaternions
            quats = np.stack([vertex[name] for name in rot_names[:4]], axis=1).astype(np.float32)
            # Normalize quaternions
            quats = quats / (np.linalg.norm(quats, axis=1, keepdims=True) + 1e-8)
            return quats
        else:
            quats = np.zeros((N, 4), dtype=np.float32)
            quats[:, 0] = 1.0

            return quats

    def initialize(self, pipeline: BasePipeline) -> GaussianModel:
        """Initialize GaussianModel from PLY file data."""
        dataset = pipeline.dataset
        assert dataset is not None, "Pipeline must have a dataset"
        assert dataset.init_ply is not None, "Dataset must specify 'init_ply' path for PlyInitializer"

        self.ply_path = Path(dataset.init_ply)
        logger.info(f"Loading Gaussian splats from PLY file: {self.ply_path}")

        # Load PLY data
        means, colors, opacities, scales, quats = self._load_ply_data()
        logger.info(f"Loaded {means.shape[0]} Gaussian splats")
        logger.info(f"Color mode: {self.color_mode}")

        # crop or pad colors to match pipeline color_dim
        if colors.shape[-1] > pipeline.color_dim:
            colors = colors[:, : pipeline.color_dim]
            logger.info(f"Cropping colors to first {pipeline.color_dim} channels")
        elif colors.shape[-1] < pipeline.color_dim:
            pad_width = pipeline.color_dim - colors.shape[-1]
            # pad with mean values of existing colors
            pad_values = np.mean(colors, axis=0, keepdims=True).repeat(pad_width, axis=0)
            colors = np.concatenate([colors, pad_values], axis=0)
            logger.info(f"Padding colors with mean values to reach {pipeline.color_dim} channels")

        # Convert to tensors
        means_tensor = torch.from_numpy(means).float()
        colors_tensor = torch.from_numpy(colors).float()
        opacities_tensor = torch.from_numpy(opacities).float()
        scales_tensor = torch.from_numpy(scales).float()
        quats_tensor = torch.from_numpy(quats).float()

        # Create GaussianModel
        dataset = pipeline.dataset
        G = GaussianModel(
            utm_epsg=dataset.site_utm_epsg, offset=dataset.site_offset_utm, scale=dataset.site_scale
        )

        # Set parameters (setters handle activations automatically)
        G.means = means_tensor
        G.colors = colors_tensor
        G.opacities = torch.sigmoid(opacities_tensor)  # Convert raw to activated, setter will convert back
        G.scales = torch.exp(scales_tensor)  # Convert raw to activated, setter will convert back
        G.quats = quats_tensor
        G.features = torch.zeros((means.shape[0], pipeline.feature_dim), dtype=torch.float32)

        logger.info(f"Initialized GaussianModel with {G.n_gaussians} gaussians")
        return G
