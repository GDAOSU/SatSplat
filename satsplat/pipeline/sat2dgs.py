import logging
from collections import OrderedDict
from typing import Optional
from pathlib import Path
import yaml

from omegaconf import DictConfig

import numpy as np
import torch

from torchmetrics.image import StructuralSimilarityIndexMeasure

from gsplat import rasterization_2dgs

from .base_pipeline import BasePipeline
from ..dataset import SatDataset
from ..metric import dsmr
from ..strategy import DefaultStrategy, Strategy
from ..module import (
    AffineCameraOptModule,
    ColorCorrectionModule,
    SunlightModule,
    GaussianModel,
)

logger = logging.getLogger(__name__)


class Sat2DGS(BasePipeline):
    def save_checkpoint(self, outdir, export_ply=True, export_aux_yaml=True):
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        # Dump all parameters in PTH format for restore
        torch.save(self.state_dict(), outdir / "state_dict.pth")
        if export_ply:
            self.model.save_ply(outdir / "model.ply")

        if export_aux_yaml:
            aux_params_yaml = {
                "illumination_model": {
                    k: v.tolist() for k, v in self.illumination_model.state_dict().items()
                },
                "colorcorrection_model": {
                    k: v.tolist() for k, v in self.colorcorrection_model.state_dict().items()
                },
                "camopt": {k: v.tolist() for k, v in self.camopt.state_dict().items()},
            }
            with open(outdir / "aux_params.yaml", "w") as f:
                yaml.dump(aux_params_yaml, f)

    def __init__(
        self,
        dataset: SatDataset,
        color_dim: int,
        feature_dim: int = 0,
        strategy: Strategy = DefaultStrategy(),
        init_cfg: DictConfig = {},
        lr_cfg: DictConfig = {},
        schedule_cfg: DictConfig = {},
        render_cfg: DictConfig = {},
        loss_weights: DictConfig = {},
        checkpoint_state: Optional[dict] = None,
    ):
        super().__init__(dataset, color_dim, feature_dim)

        self.init_cfg = init_cfg
        self.lr_cfg = lr_cfg
        self.schedule_cfg = schedule_cfg
        self.render_cfg = render_cfg
        self.loss_weights = loss_weights
        self.strategy = strategy

        if checkpoint_state is None:
            self.model = self.init_cfg.initializer.initialize(self)
            scales = self.model.scales
            scales[..., -1] = 1e-6
            self.model.scales = scales
        else:
            self.model = GaussianModel(dataset.site_utm_epsg, dataset.site_offset_utm, dataset.site_scale)
            prefix = "model.gaussians."
            for key, tensor in checkpoint_state.items():
                if key.startswith(prefix):
                    self.model.gaussians[key.removeprefix(prefix)] = torch.nn.Parameter(tensor)

        N_IMG = len(self.dataset)
        self.illumination_model = SunlightModule(
            n=N_IMG,
            color_dim=self.color_dim,
            lr=self.lr_cfg.illumination_lr,
            init_ambient=self.init_cfg.init_ambient,
        )
        self.colorcorrection_model = ColorCorrectionModule(
            n=N_IMG, color_dim=self.color_dim, lr=self.lr_cfg.color_correction_lr
        )
        self.camopt = AffineCameraOptModule(n=N_IMG, lr=self.lr_cfg.camopt_lr)
        self.camopt.zero_init()

        self.visibility_threshold = self.render_cfg.visibility_threshold / self.dataset.site_scale
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        if checkpoint_state is not None:
            self.load_state_dict(checkpoint_state)

    def render_camera(
        self,
        camera_dict: dict,
        buffers: list = ["altitude", "color", "ED"],
        packed: bool = False,
        random_bg: bool = False,
        distloss: bool = False,
        surface_normal: bool = False,
        radius_clip: float = 0.0,
        eps2d: float = 0.3,
    ):
        assert not "ED" in buffers or not "D" in buffers, "ED and D are execlusive"
        V = camera_dict["V"]
        K = camera_dict["K"]
        shape = camera_dict["shape"]
        near = camera_dict["near"]
        far = camera_dict["far"]

        means = self.model.means
        quats = self.model.quats
        scales = self.model.scales
        opacities = self.model.opacities
        colors = self.model.colors

        feat_pkg = []
        bg_pkg = []
        feat_slice = {}
        feat_dim_offset = 0

        render_mode = ""

        if "altitude" in buffers:
            render_mode = "RGB"
            feat_pkg.append(means[..., -1:])
            buf_dim = feat_pkg[-1].shape[-1]
            feat_slice["altitude"] = slice(feat_dim_offset, feat_dim_offset + buf_dim)
            feat_dim_offset += buf_dim
            bg_pkg.append(
                torch.tensor([[self.dataset.site_min_scene[-1]]], dtype=torch.float32, device=means.device)
            )

        if "color" in buffers:
            render_mode = "RGB"
            feat_pkg.append(colors)
            buf_dim = feat_pkg[-1].shape[-1]
            feat_slice["color"] = slice(feat_dim_offset, feat_dim_offset + buf_dim)
            feat_dim_offset += buf_dim

            if random_bg:
                bg_value = torch.rand(1, self.color_dim, dtype=torch.float32, device=means.device)
            else:
                bg_value = torch.zeros(1, self.color_dim, dtype=torch.float32, device=means.device)
            bg_pkg.append(bg_value)

        if "ED" in buffers:
            if render_mode != "":
                render_mode += "+"
            render_mode += "ED"
            feat_slice["depth"] = slice(feat_dim_offset, feat_dim_offset + 1)
        elif "D" in buffers:
            if render_mode != "":
                render_mode += "+"
            render_mode += "D"
            feat_slice["depth"] = slice(feat_dim_offset, feat_dim_offset + 1)
        elif distloss or surface_normal:
            render_mode += "+ED"  # Default for normal from depth, and distloss

        if "RGB" in render_mode:
            feat_pkg = torch.cat(feat_pkg, dim=-1)
            bg_pkg = torch.cat(bg_pkg, dim=-1)
        else:
            feat_pkg = torch.empty((*means.shape[:-1], 0), dtype=torch.float32, device=means.device)
            bg_pkg = torch.empty((*means.shape[:-1], 0), dtype=torch.float32, device=means.device)

        V = V.unsqueeze(0)  # [1, 4, 4]
        K = K.unsqueeze(0)  # [1, 3, 3]

        rendered_pkg, rendered_alphas, rendered_normals, surf_normals, render_distort, render_median, info = (
            rasterization_2dgs(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=feat_pkg,
                viewmats=V,
                Ks=K,
                sh_degree=None,
                radius_clip=radius_clip,
                eps2d=eps2d,
                height=shape[0],
                width=shape[1],
                backgrounds=bg_pkg,
                near_plane=near,
                far_plane=far,
                render_mode=render_mode,
                packed=packed,
                distloss=distloss,
                camera_model="ortho",
            )
        )

        result = dict(
            alpha=rendered_alphas[0],  # [H, W, 1]
            normal=rendered_normals[0],  # [H, W, 3]
            median_depth=render_median[0],  # [H, W, 1]
            info=info,
        )

        if distloss:
            result["distloss"] = render_distort[0]  # [H, W, 1]
            result["normal_from_depth"] = surf_normals.reshape(*shape, 3)  # [H, W, 3]

        if surface_normal:
            result["normal_from_depth"] = surf_normals.reshape(*shape, 3)  # [H, W, 3]

        for key, sl in feat_slice.items():
            result[key] = rendered_pkg[0, ..., sl]

        return result

    @classmethod
    def shadow_from_depths(cls, sun_depth: torch.Tensor, cam_depth: torch.Tensor, density: int = 5):
        """Regarding depth, in shaded region, reprojected camera depth is greater than sun depth."""
        return torch.minimum(
            torch.exp(-density * (cam_depth - sun_depth)), torch.tensor(1.0, device=sun_depth.device)
        )

    @classmethod
    def shadow_from_difference(cls, diff: torch.Tensor, density: int = 5):
        """Regarding difference, in shaded region, reprojected camera depth is greater than sun depth."""
        return torch.minimum(torch.exp(-density * diff), torch.tensor(1.0, device=diff.device))

    @classmethod
    def shadow_translucency(cls, shadowmap: torch.Tensor):
        a = shadowmap
        b = shadowmap.clip(0.05, 0.95)
        return -(a * torch.log2(b) + (1 - a) * torch.log2(1 - b)).mean()

    def opacity_entropy(self, opacity: torch.Tensor):
        """Encourage opacity to be either 0 or 1, avoid uncertain values in the middle."""
        return (opacity * (1 - opacity)).mean()

    def opacity_sparsity(self, opacity: torch.Tensor):
        """Encourage opacity to be sparse, i.e. most opacities are 0."""
        return opacity.sum() / self.init_num_of_gaussians

    def on_train_start(self):

        self.model.training_setup(self.lr_cfg)

        if isinstance(self.strategy, DefaultStrategy):
            self.strategy_state = self.strategy.initialize_state(1.0 / self.dataset.site_scale)
        else:
            self.strategy_state = self.strategy.initialize_state()

        self.strategy.check_sanity(self.model.gaussians, self.model.optimizers)

        self.init_num_of_gaussians = self.model.n_gaussians

    def training_step(self, step, log_images=True):
        metrics = OrderedDict()
        rendered = OrderedDict()
        losses = OrderedDict()

        samp_idx = np.random.randint(0, len(self.dataset))
        item = self.dataset[samp_idx]

        cam_idx = item["idx"]
        gt_img = item["img"]

        deltaA, deltab = None, None
        if step >= self.schedule_cfg["camopt_start_iter"]:
            deltaA, deltab = self.camopt(cam_idx)

        main_camera = item["sat_camera"].get_main_camera(
            depth_func=self.render_cfg.depth_func, deltaA=deltaA, deltab=deltab
        )

        main_rendered = self.render_camera(
            main_camera,
            buffers=["color", self.render_cfg.depth_mode],
            random_bg=self.render_cfg.random_bg,
            surface_normal=True,
            distloss=True,
            radius_clip=self.render_cfg.radius_clip,
            eps2d=self.render_cfg.eps2d,
        )
        if step >= self.schedule_cfg["L_accumulated_opacity_start_iter"]:
            losses["accumulated_opacity"] = 1.0 - main_rendered["alpha"].mean()  # As solid as possible

        if step >= self.schedule_cfg["shadowmapping_start_iter"]:
            sun_camera = item["sat_camera"].get_sun_camera(
                transformScale=self.render_cfg.sun_camera_scale,
                depth_func=self.render_cfg.depth_func,
                deltaA=deltaA,
                deltab=deltab,
            )

            sun_rendered = self.render_camera(
                sun_camera,
                buffers=["color", self.render_cfg.depth_mode],
                random_bg=self.render_cfg.random_bg,
                radius_clip=self.render_cfg.radius_clip,
                eps2d=self.render_cfg.eps2d,
            )

            resample_result = item["sat_camera"].resample_by_depth(
                dst_cam=main_camera,
                src_cam=sun_camera,
                dst_depth=main_rendered["depth"],
                src_depth=sun_rendered["depth"],
                src_feat=sun_rendered["color"],
            )

            sun_depth_diff = resample_result["projected_dst_depth"] - resample_result["resampled_src_depth"]
            if step >= self.schedule_cfg["L_sun_resample_start_iter"]:
                sun_color_diff_map = resample_result["resampled_src_feat"] - main_rendered["color"]
                sun_visibility_map = (sun_depth_diff > -self.visibility_threshold) * (
                    resample_result["projected_dst_uv"].abs() < 1
                ).all(-1, keepdim=True)
                sun_visibility_map = sun_visibility_map.detach()
                if sun_visibility_map.any():
                    losses["sun_depth_resample"] = (
                        sun_depth_diff.abs() * sun_visibility_map
                    ).sum() / sun_visibility_map.sum()
                    losses["sun_color_resample"] = (
                        sun_color_diff_map.abs() * sun_visibility_map
                    ).sum() / sun_visibility_map.sum()

            shadow_map = self.shadow_from_difference(
                diff=sun_depth_diff, density=self.render_cfg.shadow_density * self.dataset.site_scale
            )

            corrected_albedo = self.colorcorrection_model(cam_idx, main_rendered["color"])
            illumination_map = self.illumination_model(cam_idx, shadow_map)
            rendered_img = illumination_map * corrected_albedo  # [H, W, 3]

            img_abs_error = (rendered_img - gt_img).abs()
            shadow_weight = 1.0 / illumination_map.detach().clamp(min=0.05)
            weighted_img_abs_error = img_abs_error * shadow_weight

            losses["image_l1"] = img_abs_error.mean()  # unweighted L1
            losses["image_weighted_l1"] = weighted_img_abs_error.mean()  # weighted

            losses["shadow_translucency"] = self.shadow_translucency(shadow_map)

            if log_images:
                rendered["img_abs_error"] = img_abs_error.detach().cpu().numpy()
                rendered["img_weighted_abs_error"] = weighted_img_abs_error.detach().cpu().numpy()
                rendered["ref"] = item["img"].detach().cpu().numpy()
                rendered["depth"] = main_rendered["depth"].detach().cpu().numpy()
                rendered["alpha"] = main_rendered["alpha"].detach().cpu().numpy()
                rendered["sun_depth"] = sun_rendered["depth"].detach().cpu().numpy()
                rendered["normal"] = main_rendered["normal"].detach().cpu().numpy()
                rendered["normal_from_depth"] = main_rendered["normal_from_depth"].detach().cpu().numpy()
                rendered["resampled_sun_depth"] = (
                    resample_result["resampled_src_depth"].detach().cpu().numpy()
                )

                rendered["illumination"] = illumination_map.detach().cpu().numpy()
                rendered["albedo"] = corrected_albedo.detach().cpu().numpy()
                rendered["shadowmap"] = shadow_map.detach().cpu().numpy()
                rendered["final"] = rendered_img.detach().cpu().numpy()
        else:
            rendered_img = main_rendered["color"]

            img_abs_error = (rendered_img - gt_img).abs()
            losses["image_l1"] = img_abs_error.mean()  # unweighted L1

            if log_images:
                rendered["img_abs_error"] = img_abs_error.detach().cpu().numpy()
                rendered["ref"] = item["img"].detach().cpu().numpy()
                rendered["alpha"] = main_rendered["alpha"].detach().cpu().numpy()
                rendered["depth"] = main_rendered["depth"].detach().cpu().numpy()
                rendered["albedo"] = main_rendered["color"].detach().cpu().numpy()
                rendered["normal"] = main_rendered["normal"].detach().cpu().numpy()
                rendered["normal_from_depth"] = main_rendered["normal_from_depth"].detach().cpu().numpy()
                rendered["final"] = rendered_img.detach().cpu().numpy()

        # Compute losses
        L_img_ssim = 1.0 - self.ssim(
            rendered_img.permute(2, 0, 1).unsqueeze(0), gt_img.permute(2, 0, 1).unsqueeze(0)
        )
        losses["image_ssim"] = L_img_ssim

        opacity = self.model.opacities
        losses["opacity_sparsity"] = self.opacity_sparsity(opacity)
        losses["opacity_entropy"] = self.opacity_entropy(opacity)

        # Compute normal losses
        if step >= self.schedule_cfg["L_normal_start_iter"]:
            rend_normal = main_rendered["normal"]
            surf_normal = main_rendered["normal_from_depth"] * main_rendered["alpha"].detach()
            normal_error = 1 - (rend_normal * surf_normal).sum(dim=-1)  # Cosine similarity
            losses["normal"] = normal_error.mean()

        if step >= self.schedule_cfg["L_distloss_start_iter"]:
            losses["distloss"] = main_rendered["distloss"].mean()

        loss = 0
        for k, v in losses.items():
            if k not in self.loss_weights:
                # Only show once
                logger.warning(f"Loss {k} not found in loss_weights, using default weight of 1.0")
                w = 1.0
            else:
                w = self.loss_weights[k]
            loss += w * v

        metrics["total_loss"] = loss.item()

        if step >= self.schedule_cfg["camopt_start_iter"]:
            # Regularize camera parameters
            loss_camopt = torch.abs(deltaA).mean() + torch.abs(deltab).mean()
            loss += loss_camopt
            metrics["camopt"] = loss_camopt.item()

        self.strategy.step_pre_backward(
            params=self.model.gaussians,
            optimizers=self.model.optimizers,
            state=self.strategy_state,
            step=step,
            info=main_rendered["info"],
        )
        # Optimize
        loss.backward()

        if step >= self.schedule_cfg["camopt_start_iter"]:
            self.camopt.optimizer_step()
        self.model.optimizer_step()
        self.illumination_model.optimizer_step()
        self.colorcorrection_model.optimizer_step()
        self.strategy.step_post_backward(
            params=self.model.gaussians,
            optimizers=self.model.optimizers,
            state=self.strategy_state,
            step=step,
            info=main_rendered["info"],
        )

        metrics["pts"] = self.model.n_gaussians
        metrics["scales_mean"] = self.model.scales.nanmean().item() * self.model.scale
        metrics["ambient_mean"] = self.illumination_model.ambient_mean()
        metrics["opacity_mean"] = self.model.opacities.mean().item()

        for k, v in losses.items():
            metrics[f"L_{k}"] = v.item()

        return metrics, rendered

    @torch.no_grad()
    def render_ortho(self, camera):
        rendered = self.render_camera(
            camera,
            buffers=["ED", "color"],
            random_bg=False,
            radius_clip=self.render_cfg.radius_clip,
            eps2d=self.render_cfg.eps2d,
        )
        altitude = self.dataset.site_max_scene[2] - rendered["depth"][..., 0]
        dsm = altitude.cpu().numpy() * self.dataset.site_scale + self.dataset.site_offset_utm[2]
        dsm[rendered["alpha"][..., 0].cpu().numpy() <= 0] = np.nan
        return dsm, rendered["color"].cpu().numpy()

    def evaluation_step(self, step):
        if not self.dataset.has_GT:
            return {}, {}

        dsm, ortho = self.render_ortho(self.dataset.GT_cam)
        gt = (
            self.dataset.GT_DSM[..., 0].cpu().numpy() * self.dataset.site_scale
            + self.dataset.site_offset_utm[2]
        )
        classes = self.dataset.GT_CLS.cpu().numpy() if self.dataset.has_GT_cls else None
        metrics, registered, error = dsmr.evaluate(gt, dsm, classes)
        return metrics, {"DSM": dsm, "ORTHO": ortho, "DSM_REGISTERED": registered, "DSM_ERR": error}
