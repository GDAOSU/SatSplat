"""Generate a small synthetic multiview scene for the SatSplat smoke test."""

import argparse
import json
from pathlib import Path

import numpy as np
from rasterio import open as open_raster
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter

SITE = "SYNTHETIC"
SIZE = 144
SCALE = 64
CENTER = [500000, 3318785, 30]
CRS = "EPSG:32617"


def surface(x, y):
    building = (abs(x + 0.24) < 0.23) & (abs(y - 0.12) < 0.22)
    annex = (abs(x - 0.32) < 0.14) & (abs(y + 0.3) < 0.16)
    z = 0.025 * np.sin(5 * x) + 0.02 * np.cos(5 * y) + 0.14 * building + 0.09 * annex
    stripe = 0.04 * np.sin(22 * x) * np.sin(19 * y)
    rgb = np.stack(
        (0.32 + stripe + 0.33 * building, 0.49 + stripe - 0.18 * annex, 0.30 - stripe + 0.27 * annex)
    )
    return z, rgb, building | annex


def write_tif(path, data, **georeference):
    path.parent.mkdir(parents=True, exist_ok=True)
    bands, height, width = data.shape
    with open_raster(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=bands,
        dtype=data.dtype,
        compress="deflate",
        **georeference,
    ) as image:
        image.write(data)


def as_rgb(values):
    return np.rint(values.clip(0, 1) * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Output dataset root")
    args = parser.parse_args()
    root = args.root / SITE
    root.mkdir(parents=True, exist_ok=False)
    grid = ((np.arange(SIZE) + 0.5) / SIZE) * 2 - 1
    x, y = np.meshgrid(grid, -grid)
    size_m = 2 * SCALE / SIZE
    georef = {"transform": from_origin(CENTER[0] - SCALE, CENTER[1] + SCALE, size_m, size_m), "crs": CRS}
    z, rgb, buildings = surface(x, y)
    dsm = (z * SCALE + CENTER[2]).astype(np.float32)
    write_tif(root / "truth" / f"{SITE}_DSM.tif", dsm[None], **georef)
    write_tif(root / "truth" / f"{SITE}_CLS.tif", np.where(buildings, 6, 2).astype(np.uint8)[None], **georef)
    initial = gaussian_filter(dsm, sigma=1.7).astype(np.float32)
    write_tif(root / "initialization" / "dsm.tif", initial[None], **georef)

    cameras = []
    tilts = [(0, 0), (0.32, 0.1), (-0.28, 0.18), (0.16, -0.29)]
    dates = ["20220615150000", "20220815160000", "20221015170000", "20220415180000"]
    for index, ((tx, ty), date) in enumerate(zip(tilts, dates, strict=True)):
        filename = f"view_{index:02d}.tif"
        z_view = z
        for _ in range(4):
            z_view, image, _ = surface(x - tx * z_view, y + ty * z_view)
        factors = np.array([1.0 + 0.03 * index, 1.0, 1.0 - 0.03 * index])[:, None, None]
        write_tif(root / "images" / filename, as_rgb(image * factors))
        write_tif(root / "initialization" / "orthos" / filename, as_rgb(rgb), **georef)
        cameras.append(
            {
                "img": filename,
                "width": SIZE,
                "height": SIZE,
                "acquisition_date": date,
                "geojson": {"center": [-81, 30]},
                "model": {
                    "coef_": [[1, 0, tx], [0, -1, ty], [0, 0, SCALE]],
                    "intercept_": [0, 0, CENTER[2]],
                    "center": CENTER,
                    "scale": SCALE,
                    "n": 17,
                    "l": "R",
                    "min_world": [-1, -1, -0.12],
                    "max_world": [1, 1, 0.35],
                },
            }
        )

    (root / "cameras.json").write_text(json.dumps(cameras, indent=2) + "\n")
    (args.root / "README.md").write_text(
        "# SatSplat synthetic test scene\n\n"
        "This scene contains generated geometry and pixel values. It is provided under "
        "CC0 1.0 Universal (https://creativecommons.org/publicdomain/zero/1.0/).\n"
    )


if __name__ == "__main__":
    main()
