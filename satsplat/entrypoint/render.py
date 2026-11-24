"""Export georeferenced SatSplat rasters and DSM accuracy metrics."""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


def export_products(pipeline, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    dataset = pipeline.dataset
    if dataset.has_GT:
        metrics, products = pipeline.evaluation_step(None)
        transform = dataset.GT_geotransform
    else:
        dsm, ortho = pipeline.render_ortho(dataset.ortho_cam)
        metrics = {}
        products = {"DSM": dsm, "ORTHO": ortho}
        transform = dataset.ortho_geotransform

    profile = {
        "driver": "GTiff",
        "height": products["DSM"].shape[0],
        "width": products["DSM"].shape[1],
        "transform": transform,
        "crs": f"EPSG:{dataset.site_utm_epsg}",
        "compress": "deflate",
    }
    for key, filename in (
        ("DSM", "dsm.tif"),
        ("DSM_REGISTERED", "dsm_registered.tif"),
        ("DSM_ERR", "dsm_error.tif"),
    ):
        if key in products:
            with rasterio.open(
                output / filename, "w", **profile, count=1, dtype="float32", nodata=np.nan
            ) as raster:
                raster.write(products[key].astype(np.float32), 1)

    ortho = np.rint(np.clip(products["ORTHO"], 0, 1) * 255).astype(np.uint8)
    with rasterio.open(output / "ortho.tif", "w", **profile, count=ortho.shape[-1], dtype="uint8") as raster:
        raster.write(np.moveaxis(ortho, -1, 0))

    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to state_dict.pth in a training run")
    parser.add_argument("--config", type=Path, help="Training config.yaml (defaults to the checkpoint's run)")
    parser.add_argument("--data-root", type=Path, help="Replacement dataset root for a relocated benchmark")
    parser.add_argument(
        "--output", type=Path, help="Destination (defaults to a rendered directory beside the checkpoint)"
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("SatSplat rendering requires a CUDA-enabled NVIDIA GPU.")

    cfg = OmegaConf.load(args.config or args.checkpoint.parent.parent / "config.yaml")
    if args.data_root:
        old_root = Path(cfg.data_root)
        for field in ("img_dir", "affine_models_path", "truth_dir", "init_dsm", "init_ortho_dir"):
            value = cfg.dataset.get(field)
            if value:
                cfg.dataset[field] = str(args.data_root.resolve() / Path(value).relative_to(old_root))

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    dataset = instantiate(cfg.dataset, device="cuda", preload=False)
    build_pipeline = instantiate(cfg.pipeline, _partial_=True)
    pipeline = build_pipeline(dataset=dataset, color_dim=dataset.channels()[0], checkpoint_state=state).to(
        "cuda"
    )
    pipeline.eval()
    metrics = export_products(pipeline, args.output or args.checkpoint.parent / "rendered")
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
