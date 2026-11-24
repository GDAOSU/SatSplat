import os
import json
import logging
import math
from datetime import datetime, timezone
from typing import Optional, List, Tuple
from pathlib import Path
import warnings

import numpy as np
import pyproj
import rasterio as rio
import torch
from affine import Affine
from rasterio.warp import Resampling, reproject

from .camera import AffineCamera

warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)

logger = logging.getLogger(__name__)

UTM_GRID_TO_NS = {
    "X": "N",
    "W": "N",
    "V": "N",
    "U": "N",
    "T": "N",
    "S": "N",
    "R": "N",
    "Q": "N",
    "P": "N",
    "M": "S",
    "L": "S",
    "K": "S",
    "J": "S",
    "H": "S",
    "G": "S",
    "F": "S",
    "E": "S",
    "D": "S",
    "C": "S",
}

# https://github.com/pubgeo/dfc2019/blob/master/data/README.md#classification-labels
CLS_CATEGORYS = {
    2: "Ground",
    5: "Trees",
    6: "Buildings",
    9: "Water",
    17: "Bridge / elevated road",
    65: "Unlabeled",
}


class SatDataset(torch.utils.data.Dataset):
    """Dataset class for satellite images dataset."""

    def __init__(
        self,
        name: str,
        img_dir: os.PathLike,
        affine_models_path: os.PathLike,
        truth_dir: Optional[os.PathLike] = None,
        preload: bool = True,
        pixel_percentiles: Tuple[int, int] = (None, 100),
        init_ply: Optional[os.PathLike] = None,
        init_dsm: Optional[os.PathLike] = None,
        init_orthos: Optional[List[os.PathLike]] = None,
        init_ortho_dir: Optional[os.PathLike] = None,
        device="cpu",
    ):
        self.device = torch.device(device)
        self.name = name
        self.img_dir = Path(img_dir)
        self.pixel_percentiles = pixel_percentiles
        self.affine_models_path = Path(affine_models_path)
        self.truth_dir = Path(truth_dir) if truth_dir else None
        self.preload = preload

        self.init_ply = Path(init_ply) if init_ply else None
        self.init_dsm = Path(init_dsm) if init_dsm else None
        if init_orthos is not None and init_ortho_dir is not None:
            raise ValueError("Specify either init_orthos or init_ortho_dir.")
        self.init_orthos = [Path(path) for path in init_orthos or ()]
        if init_ortho_dir is not None:
            directory = Path(init_ortho_dir)
            self.init_orthos = sorted(directory.glob("*.tif"))
            if not self.init_orthos:
                raise FileNotFoundError(f"No initialization orthophotos in {directory}.")

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory {self.img_dir} does not exist.")
        if not self.affine_models_path.exists():
            raise FileNotFoundError(f"Affine models file {self.affine_models_path} does not exist.")

        if self.init_ply and not self.init_ply.exists():
            raise FileNotFoundError(f"Initial point cloud file {self.init_ply} does not exist.")
        if self.init_dsm and not self.init_dsm.exists():
            raise FileNotFoundError(f"Initial DSM file {self.init_dsm} does not exist.")
        for p in self.init_orthos:
            if not p.exists():
                raise FileNotFoundError(f"Initial ortho image file {p} does not exist.")

        ################
        # Load Images and Camera Models
        self.img_list = list(sorted([f.name for f in self.img_dir.glob("*.tif")]))
        self.load_affine_models()

        # Load Ground Truth
        self.load_groundtruth()

        # Create ortho camera
        self.create_ortho_camera()

    def channels(self):
        if all([i is not None for i in self.imgs]):
            return list(sorted(set([i.shape[-1] for i in self.imgs])))
        else:
            channels = []
            for n in self.img_list:
                file_path = self.img_dir / n
                with rio.open(file_path, "r") as f:
                    channels.append(f.count)
            return list(sorted(set(channels)))

    def create_ortho_camera(self):
        corners = torch.tensor(
            [
                [self.site_min_scene[0], self.site_min_scene[1], self.site_min_scene[2]],
                [self.site_max_scene[0], self.site_min_scene[1], self.site_min_scene[2]],
                [self.site_min_scene[0], self.site_max_scene[1], self.site_min_scene[2]],
            ],
            dtype=torch.float32,
        ).T.to(self.device)

        site_extent = self.site_max_scene - self.site_min_scene

        resx_list = []
        resy_list = []
        for cam in self.cams:
            main_cam = cam.get_main_camera()
            V = main_cam["V"]
            K = main_cam["K"]
            o = V[:3, :3] @ corners + V[:3, 3:4]  # Transform corners to camera space
            oo = K[:2, :2] @ o[:2, :] + K[:2, 2:3]  # Project corners to image space
            resx = site_extent[0] / (oo[:, 1] - oo[:, 0]).norm().item()
            resy = site_extent[1] / (oo[:, 2] - oo[:, 0]).norm().item()

            resx_list.append(resx)
            resy_list.append(resy)

        resx = np.median(resx_list)
        resy = np.median(resy_list)

        ortho_width = int(math.ceil(site_extent[0] / resx))
        ortho_height = int(math.ceil(site_extent[1] / resy))

        viewmat = torch.tensor(
            [
                [
                    2 / site_extent[0],
                    0,
                    0,
                    -(self.site_min_scene[0] + self.site_max_scene[0]) / site_extent[0],
                ],
                [
                    0,
                    -2 / site_extent[1],
                    0,
                    (self.site_min_scene[1] + self.site_max_scene[1]) / site_extent[1],
                ],
                [0, 0, -1, self.site_max_scene[2]],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        K = torch.tensor(
            [[ortho_width / 2, 0, ortho_width / 2], [0, ortho_height / 2, ortho_height / 2], [0, 0, 1]],
            dtype=torch.float32,
            device=self.device,
        )

        self.ortho_cam = dict(
            V=viewmat,
            K=K,
            near=0,
            far=site_extent[2] * 2,
            shape=(ortho_height, ortho_width),
        )

        A_scene2world = torch.tensor(
            [
                [self.site_scale, 0, self.site_offset_utm[0]],
                [0, self.site_scale, self.site_offset_utm[1]],
                [0, 0, 1],
            ],
            dtype=torch.float64,
            device="cpu",
        )
        A_scene2img = K.to("cpu", torch.float64) @ viewmat[[0, 1, 3]][:, [0, 1, 3]].to("cpu", torch.float64)
        A_img2world = A_scene2world @ torch.linalg.inv(A_scene2img)

        self.ortho_geotransform = Affine(
            A_img2world[0, 0],
            A_img2world[0, 1],
            A_img2world[0, 2],
            A_img2world[1, 0],
            A_img2world[1, 1],
            A_img2world[1, 2],
        )

        logger.info("Ortho camera created:")
        logger.info(f"  Median resolution: {resx:.5f} units/pixel, {resy:.5f} units/pixel")
        logger.info(f"  Ortho camera resolution: {ortho_width} x {ortho_height} pixels")

    def preprocess_img(self, img):
        if self.pixel_percentiles[0] is None:
            vmin = 0
            vmax = np.nanpercentile(img, self.pixel_percentiles[1], axis=(0, 1), keepdims=True)
        else:
            vmin, vmax = np.nanpercentile(img, self.pixel_percentiles, axis=(0, 1), keepdims=True)
        img = (img - vmin) / (vmax - vmin + 1e-8)
        return img

    def load_img(self, file_path):
        img = rio.open(file_path, "r").read().transpose(1, 2, 0).astype(np.float32)
        assert np.isnan(img).sum() == 0, f"Image {file_path} contains NaN values."

        img = self.preprocess_img(img)

        return img

    def load_affine_models(self):
        affine_models = json.loads(self.affine_models_path.read_text())
        image_names = set(self.img_list)
        if not image_names or image_names - {model["img"] for model in affine_models}:
            raise ValueError(f"Missing images or camera records for {self.name}.")

        self.imgs = [None] * len(self.img_list)
        self.img_meta = [None] * len(self.img_list)
        self.cams: list[AffineCamera] = [None] * len(self.img_list)
        self.img_shapes: List[Optional[Tuple[int, int]]] = [None] * len(self.img_list)

        self.site_lonlat = []
        self.site_utm_zone = []
        self.site_utm_ns = []
        self.site_offset_utm = []
        self.site_min_scene = []
        self.site_max_scene = []
        self.site_scale = []

        for m in affine_models:
            name = m["img"]
            if name not in self.img_list:
                continue
            idx = self.img_list.index(name)
            expected_shape = (int(m["height"]), int(m["width"]))
            self.img_shapes[idx] = expected_shape

            file_path = self.img_dir / name
            if not file_path.exists():
                raise FileNotFoundError(f"Image file {file_path} does not exist.")
            elif self.preload:
                img = self.load_img(file_path)
                if img.shape[:2] != expected_shape:
                    raise ValueError(f"Image dimensions do not match camera metadata for {name}.")
                self.imgs[idx] = torch.from_numpy(img).to(self.device)

            acquisition_date = datetime.strptime(m["acquisition_date"], "%Y%m%d%H%M%S").replace(
                tzinfo=timezone.utc
            )

            lon, lat = m["geojson"]["center"]
            self.site_lonlat.append((lon, lat))
            self.site_utm_zone.append(m["model"]["n"])
            self.site_utm_ns.append(UTM_GRID_TO_NS[m["model"]["l"]])
            self.site_offset_utm.append(m["model"]["center"])
            self.site_scale.append(m["model"]["scale"])
            min_world = m["model"]["min_world"]
            max_world = m["model"]["max_world"]
            self.site_min_scene.append(m["model"]["min_world"])
            self.site_max_scene.append(m["model"]["max_world"])

            self.img_meta[idx] = {
                "name": name,
                "acquisition_date": acquisition_date,
            }

            ######### Image camera parameters
            A_cam = np.array(m["model"]["coef_"])
            b_cam = np.array(m["model"]["intercept_"])

            sun_dir = AffineCamera.sun_dir_from_datetime(acquisition_date, lon=lon, lat=lat)
            cam = AffineCamera(
                width=m["width"],
                height=m["height"],
                A=A_cam,
                b=b_cam,
                sun_dir=sun_dir,
                min_scene=min_world,
                max_scene=max_world,
                idx=idx,
                appearance_id=idx,
                acquisition_date=acquisition_date,
                device=self.device,
            )

            self.cams[idx] = cam

        if len(set(self.site_utm_zone)) > 1:
            logger.critical("Multiple UTM zones detected in the dataset. This may lead to inconsistencies.")
        if len(set(self.site_utm_ns)) > 1:
            logger.critical(
                "Multiple UTM hemispheres detected in the dataset. This may lead to inconsistencies."
            )

        #############
        # Resolve site information
        self.site_utm_ns = self.site_utm_ns[0]
        self.site_utm_zone = self.site_utm_zone[0]
        self.site_utm_epsg = self.site_utm_zone + (32600 if self.site_utm_ns == "N" else 32700)

        self.site_offset_utm = np.array(self.site_offset_utm, dtype=np.float64)
        self.site_min_scene = np.array(self.site_min_scene, dtype=np.float64)
        self.site_max_scene = np.array(self.site_max_scene, dtype=np.float64)
        self.site_scale = np.array(self.site_scale, dtype=np.float64)

        if self.site_offset_utm.std(axis=0).max() > 1e-6:
            raise ValueError("Site UTM offsets vary significantly. This may lead to inconsistencies.")
        if self.site_min_scene.std(axis=0).max() > 1e-6:
            raise ValueError(
                "Site minimum scene coordinates vary significantly. This may lead to inconsistencies."
            )
        if self.site_max_scene.std(axis=0).max() > 1e-6:
            raise ValueError(
                "Site maximum scene coordinates vary significantly. This may lead to inconsistencies."
            )
        if self.site_scale.std() > 1e-6:
            raise ValueError("Site scales vary significantly. This may lead to inconsistencies.")

        self.site_offset_utm = self.site_offset_utm.mean(axis=0)
        self.site_min_scene = self.site_min_scene.mean(axis=0)
        self.site_max_scene = self.site_max_scene.mean(axis=0)
        self.site_scale = self.site_scale.mean()
        self.site_lonlat = pyproj.Transformer.from_crs(
            self.site_utm_epsg, "EPSG:4326", always_xy=True
        ).transform(self.site_offset_utm[0], self.site_offset_utm[1])

        logger.info("Site information:")
        logger.info(f"  UTM EPSG: {self.site_utm_epsg}")
        logger.info(f"  UTM Offset: {self.site_offset_utm}")
        logger.info(f"  LonLat: {self.site_lonlat}")
        logger.info(f"  Min Scene: {self.site_min_scene}")
        logger.info(f"  Max Scene: {self.site_max_scene}")
        logger.info(f"  Scale Scene: {self.site_scale}")

    def load_groundtruth(self):
        if self.truth_dir is None:
            self.has_GT = False
            self.has_GT_cls = False
            return

        gt_tif_path = None
        gt_txt_path = None
        cls_tif_path = None

        gt_tif_path = list(self.truth_dir.glob("*DSM.tif"))
        if len(gt_tif_path) == 0:
            raise FileNotFoundError(f"Ground truth DSM file not found in {self.truth_dir}")
        elif len(gt_tif_path) > 1:
            raise ValueError(f"Multiple ground truth DSM files found in {self.truth_dir}")
        else:
            gt_tif_path = gt_tif_path[0]
            gt_txt_path = gt_tif_path.with_suffix(".txt")
            gt_txt_path = gt_txt_path if gt_txt_path.exists() else None

        cls_tif_path = list(self.truth_dir.glob("*CLS.tif"))
        if len(cls_tif_path) == 0:
            logger.warning(
                f"Ground truth classification file not found in {self.truth_dir}. Proceeding without it."
            )
            self.has_GT_cls = False
            cls_tif_path = None
        elif len(cls_tif_path) > 1:
            raise ValueError(f"Multiple ground truth classification files found in {self.truth_dir}")
        else:
            cls_tif_path = cls_tif_path[0]

        logger.info(f"Ground truth DSM loaded from {gt_tif_path}")

        # Read DSM from tif
        with rio.open(gt_tif_path, "r") as gt_rst:
            GT_DSM = gt_rst.read(masked=True).astype(np.float32).filled(np.nan)
            GT_DSM = (GT_DSM - self.site_offset_utm[2]) / self.site_scale  # Scale the DSM
            self.GT_DSM = torch.from_numpy(GT_DSM).permute(1, 2, 0).to(self.device)
            logger.info(f"  Ground truth shape: {self.GT_DSM.shape}")

            if gt_txt_path and gt_txt_path.exists():
                gt_xmin, gt_ymin, gt_width, gt_gsd = np.loadtxt(gt_txt_path, dtype=np.float64, unpack=True)
                gt_height = gt_rst.height
                if gt_width != gt_rst.width:
                    raise ValueError(f"DSM grid metadata does not match {gt_tif_path}")
                gt_ymax = gt_ymin + gt_height * gt_gsd
                gt_geotransform = Affine(gt_gsd, 0, gt_xmin, 0, -gt_gsd, gt_ymax)
                gt_xmax = gt_xmin + gt_width * gt_gsd
                logger.info(f"  Override metadata from {gt_txt_path}")
                logger.info(f"  Ground truth Xmin: {gt_xmin}, Ymin: {gt_ymin}, GSD: {gt_gsd}")
            else:
                # Geotransform
                gt_height, gt_width = gt_rst.height, gt_rst.width
                gt_geotransform = gt_rst.transform

                gt_xmin = gt_geotransform.c
                gt_ymax = gt_geotransform.f
                gt_xmax = gt_geotransform.c + gt_geotransform.a * gt_width
                gt_ymin = gt_geotransform.f + gt_geotransform.e * gt_height
                logger.info("  Read metadata from TIF")
                logger.info(
                    f"  Ground truth Xmin: {gt_xmin}, Ymin: {gt_ymin}, Xmax: {gt_xmax}, Ymax: {gt_ymax}"
                )

        gt_scene_xmin = (gt_xmin - self.site_offset_utm[0]) / self.site_scale
        gt_scene_ymin = (gt_ymin - self.site_offset_utm[1]) / self.site_scale
        gt_scene_xmax = (gt_xmax - self.site_offset_utm[0]) / self.site_scale
        gt_scene_ymax = (gt_ymax - self.site_offset_utm[1]) / self.site_scale

        GT_viewmat = torch.tensor(
            [
                [
                    2 / (gt_scene_xmax - gt_scene_xmin),
                    0,
                    0,
                    -(gt_scene_xmax + gt_scene_xmin) / (gt_scene_xmax - gt_scene_xmin),
                ],
                [
                    0,
                    -2 / (gt_scene_ymax - gt_scene_ymin),
                    0,
                    (gt_scene_ymax + gt_scene_ymin) / (gt_scene_ymax - gt_scene_ymin),
                ],
                [0, 0, -1, self.site_max_scene[2]],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        GT_K = torch.tensor(
            [[gt_width / 2, 0, gt_width / 2], [0, gt_height / 2, gt_height / 2], [0, 0, 1]],
            dtype=torch.float32,
            device=self.device,
        )
        GT_shape = (int(gt_height), int(gt_width))
        GT_near = 0.0
        GT_far = float(self.site_max_scene[2] - self.site_min_scene[2])

        self.GT_cam = dict(
            V=GT_viewmat,
            K=GT_K,
            near=GT_near,
            far=GT_far * 2,
            shape=GT_shape,
        )
        self.GT_geotransform = gt_geotransform
        self.has_GT = True

        if cls_tif_path:
            logger.info(f"Ground truth classification loaded from {cls_tif_path}")
            gt_cls = np.zeros(GT_shape, dtype=np.uint16)
            with rio.open(cls_tif_path) as cls_rst:
                reproject(
                    source=rio.band(cls_rst, 1),
                    destination=gt_cls,
                    src_crs=cls_rst.crs or f"EPSG:{self.site_utm_epsg}",
                    dst_crs=f"EPSG:{self.site_utm_epsg}",
                    dst_transform=gt_geotransform,
                    dst_nodata=0,
                    resampling=Resampling.nearest,
                )
            self.GT_CLS = torch.from_numpy(gt_cls.astype(np.int32)).to(self.device)
            self.has_GT_cls = True

    def get_camera_direction(self, idx):
        A = self.cams[idx].A
        d = torch.cross(A[0], A[1], dim=-1)
        d = d / (d.norm() + 1e-6)
        return d

    def get_sun_direction(self, idx):
        return self.cams[idx].sun_dir

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if self.preload:
            img = self.imgs[idx]
        else:
            file_path = self.img_dir / self.img_list[idx]
            img = self.load_img(file_path)
            img = torch.from_numpy(img).to(self.device)

        sample = dict(
            idx=idx,
            img=img,
            img_meta=self.img_meta[idx],
            sat_camera=self.cams[idx],
            ortho_camera=self.ortho_cam,
            gt_camera=self.GT_cam if self.has_GT else None,
            gt_DSM=self.GT_DSM if self.has_GT else None,
        )
        return sample

    def to(self, device):
        self.device = torch.device(device)
        if self.preload and self.imgs:
            self.imgs = [img.to(self.device) if img is not None else None for img in self.imgs]

        for cam in self.cams:
            cam.to_device(self.device)
        if hasattr(self, "GT_DSM") and self.GT_DSM is not None:
            self.GT_DSM = self.GT_DSM.to(self.device)
        return self


# plot camera directions in polar coordinates
def plot_camera_directions(dataset, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        fig, _ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(10, 10))
    else:
        assert isinstance(ax, plt.PolarAxes), "ax must be a polar projection"
        _ax = ax

    _min_elevation = 90
    with torch.no_grad():
        for i in range(len(dataset)):
            direction = -dataset.get_camera_direction(i).cpu()
            theta = torch.atan2(direction[1], direction[0])
            r = torch.norm(direction[:2])
            elev = torch.rad2deg(torch.asin(direction[2]))
            _min_elevation = min(_min_elevation, elev)

            _ax.plot(theta, r, "o", markersize=10)

    _rticks_elevation = torch.arange(80, int(_min_elevation) - 15, -15)
    _rticks = torch.cos(torch.deg2rad(_rticks_elevation))
    _rlabels = [f"{i}\N{DEGREE SIGN}" for i in _rticks_elevation]
    _ax.set_rticks(_rticks, _rlabels)
    _ax.set_rmax(_rticks.max())
    _ax.set_theta_zero_location("N")
    _ax.set_theta_direction(-1)
    _ax.grid(True)

    _ax.set_title("Camera Directions")

    if _ax is None:
        plt.tight_layout()
        plt.show()


def plot_sun_directions(dataset, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        fig, _ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(10, 10))
    else:
        assert isinstance(ax, plt.PolarAxes), "ax must be a polar projection"
        _ax = ax

    _min_elevation = 90
    with torch.no_grad():
        for i in range(len(dataset)):
            direction = dataset.get_sun_direction(i).cpu()
            direction = direction / direction.norm()
            theta = torch.atan2(direction[1], direction[0])
            elev = torch.rad2deg(torch.asin(direction[2]))
            _min_elevation = min(_min_elevation, elev)
            r = torch.norm(direction[:2])
            _ax.plot(theta, r, "o", markersize=10)

    _rticks_elevation = torch.arange(80, int(_min_elevation) - 15, -15)
    _rticks = torch.cos(torch.deg2rad(_rticks_elevation))
    _rlabels = [f"{i}\N{DEGREE SIGN}" for i in _rticks_elevation]
    _ax.set_rticks(_rticks, _rlabels)
    _ax.set_rmax(_rticks.max())
    _ax.set_theta_zero_location("N")
    _ax.set_theta_direction(-1)
    _ax.grid(True)
    _ax.set_title("Sun Directions")

    if _ax is None:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    dataset = SatDataset(
        img_dir="doc/images/JAX_068",
        affine_models_path="doc/affine_models/JAX_068/affine_models.json",
        truth_dir="doc/truth/JAX_068",
    )
    print(dataset[0])
