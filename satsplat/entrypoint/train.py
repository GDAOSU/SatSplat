import logging
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from ..config import CONFIG_PATH
from ..pipeline.base_pipeline import BasePipeline
from ..vizutil import vizImg
from .render import export_products

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name="pers")
def main(cfg: DictConfig):
    for name in ("max_steps", "evaluate_every", "log_images_every", "save_checkpoint_every"):
        if not isinstance(cfg[name], int) or cfg[name] <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if not torch.cuda.is_available():
        raise RuntimeError("SatSplat training requires a CUDA-enabled NVIDIA GPU.")

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    logdir = Path(HydraConfig.get().runtime.output_dir)
    cfg.data_root = str(Path(cfg.data_root).expanduser().resolve())

    dataset = instantiate(cfg.dataset, device="cuda")
    channels = dataset.channels()
    if len(channels) != 1:
        raise ValueError(f"Images have inconsistent channel counts: {channels}")
    pipeline = instantiate(cfg.pipeline, dataset=dataset, color_dim=channels[0]).to("cuda")
    pipeline.on_train_start()
    (logdir / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))

    logger.info(
        "Training %s on %d views with %d initial splats", cfg.site, len(dataset), pipeline.model.n_gaussians
    )
    with SummaryWriter(logdir) as writer, trange(cfg.max_steps, desc=cfg.site, mininterval=1) as progress:
        for step in progress:
            iteration = step + 1
            log_images = step == 0 or iteration % cfg.log_images_every == 0
            metrics, rendered = pipeline.training_step(step, log_images=log_images)
            if iteration % cfg.evaluate_every == 0 and iteration < cfg.max_steps:
                evaluation, _ = pipeline.evaluation_step(step)
                metrics.update(evaluation)
            for key, value in metrics.items():
                if value is not None:
                    writer.add_scalar(f"metrics/{key}", value, iteration)

            progress.set_postfix(
                loss=f"{metrics['total_loss']:.3f}", splats=pipeline.model.n_gaussians, refresh=False
            )
            if log_images:
                log_training_images(writer, rendered, iteration)
                log_splat_histograms(writer, pipeline, iteration)
            if iteration % cfg.save_checkpoint_every == 0 and iteration < cfg.max_steps:
                pipeline.save_checkpoint(logdir / f"iter{iteration:06d}")

    final = logdir / f"final_iter{cfg.max_steps:06d}"
    pipeline.save_checkpoint(final)
    metrics = export_products(pipeline, final)
    logger.info("Saved checkpoint and georeferenced products to %s", final)
    logger.info("Final evaluation %s", metrics)


def log_training_images(writer: SummaryWriter, step_rendered: dict, step: int):
    """Log training images to tensorboard."""
    if "albedo" in step_rendered:
        writer.add_image("albedo", vizImg(step_rendered["albedo"]), step, dataformats="HWC")
    if "alpha" in step_rendered:
        writer.add_image(
            "alpha",
            vizImg(step_rendered["alpha"], cmap="magma", skip_normalize=True),
            step,
            dataformats="HWC",
        )
    if "new_view_visibility" in step_rendered:
        writer.add_image(
            "new_view_visibility",
            vizImg(step_rendered["new_view_visibility"], cmap="magma", skip_normalize=True),
            step,
            dataformats="HWC",
        )
    if "shadowmap" in step_rendered:
        writer.add_image("shadowmap", vizImg(step_rendered["shadowmap"]), step, dataformats="HWC")
    if "illumination" in step_rendered:
        writer.add_image("illumination", vizImg(step_rendered["illumination"]), step, dataformats="HWC")
    if "normal" in step_rendered:
        writer.add_image(
            "normal",
            vizImg((step_rendered["normal"] + 1) / 2.0, skip_normalize=True),
            step,
            dataformats="HWC",
        )
    if "normal_from_depth" in step_rendered:
        writer.add_image(
            "normal_from_depth",
            vizImg((step_rendered["normal_from_depth"] + 1) / 2.0, skip_normalize=True),
            step,
            dataformats="HWC",
        )
    if "final" in step_rendered:
        writer.add_image("final", vizImg(step_rendered["final"]), step, dataformats="HWC")
    if "ref" in step_rendered:
        writer.add_image("ref", vizImg(step_rendered["ref"]), step, dataformats="HWC")

    if "img_abs_error" in step_rendered:
        writer.add_image("img_abs_error", vizImg(step_rendered["img_abs_error"]), step, dataformats="HWC")
    if "img_weighted_abs_error" in step_rendered:
        writer.add_image(
            "img_weighted_abs_error", vizImg(step_rendered["img_weighted_abs_error"]), step, dataformats="HWC"
        )

    if "depth" in step_rendered:
        writer.add_image("depth", vizImg(step_rendered["depth"], cmap="viridis"), step, dataformats="HWC")
    if "DSM" in step_rendered:
        writer.add_image("DSM", vizImg(step_rendered["DSM"], perc=(2, 98)), step, dataformats="HWC")
    if "ORTHO" in step_rendered:
        writer.add_image("ORTHO", vizImg(step_rendered["ORTHO"]), step, dataformats="HWC")
    if "DSM_ERR" in step_rendered:
        writer.add_image(
            "DSM_ERR",
            vizImg(np.abs(step_rendered["DSM_ERR"]), perc=(0, 99), cmap="magma"),
            step,
            dataformats="HWC",
        )


def log_splat_histograms(writer: SummaryWriter, pipeline: BasePipeline, step: int):
    """Log splat parameter histograms to tensorboard."""
    writer.add_histogram("splats/scales", pipeline.model.scales, step)
    writer.add_histogram("splats/opacities", pipeline.model.opacities, step)
    writer.add_histogram("splats/means", pipeline.model.means, step)
    writer.add_histogram("splats/colors", pipeline.model.colors, step)
    writer.add_histogram("splats/quats", pipeline.model.quats, step)


if __name__ == "__main__":
    main()
