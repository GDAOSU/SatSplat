"""Assemble SatSplat scenes from published geometry and upstream benchmark files."""

import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
from urllib.request import urlopen
import warnings
import zipfile

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.windows import Window
from scipy.ndimage import map_coordinates

EOGS_URL = "https://github.com/mezzelfo/EOGS/releases/download/dataset_v01/data.zip"
EOGS_SHA256 = "b73c51d1d189a4d87d4c16258d4a9d2d53ab68776a91c5539f63094fef8a7d12"
GEOMETRY_URL = (
    "https://github.com/GDAOSU/SatSplat/releases/download/v1.0.0/SatSplatDataset_PERS_geometry.tar.gz"
)
GEOMETRY_SHA256 = "d3ed8368a260bdce0fc286d2d2600af114a48d3088152bd41eefb159a507b72f"


def download(url, checksum, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        print(f"Downloading {url}", flush=True)
        try:
            with (
                urlopen(url, timeout=90) as response,
                tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as out,
            ):
                temporary = Path(out.name)
                digest = hashlib.sha256()
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    digest.update(chunk)
                    out.write(chunk)
            if digest.hexdigest() != checksum:
                raise ValueError(f"SHA-256 mismatch for {url}")
            temporary.replace(target)
        finally:
            if "temporary" in locals():
                temporary.unlink(missing_ok=True)
    else:
        with target.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != checksum:
                raise ValueError(f"SHA-256 mismatch for {target}. Remove the file and download again.")
    return target


class DfcFiles:
    """Find scenes in the user's official Track 3 archives or extracted folders."""

    def __init__(self, sources, stack):
        self.files = {}
        for path in sources:
            if path.is_dir():
                for item in sorted(path.rglob("*")):
                    if item.is_file():
                        self.files.setdefault(item.name, (None, item))
            else:
                archive = stack.enter_context(zipfile.ZipFile(path))
                for member in archive.infolist():
                    if not member.is_dir():
                        self.files.setdefault(Path(member.filename).name, (archive, member))

    def __contains__(self, name):
        return name in self.files

    @contextmanager
    def open(self, name):
        if name not in self.files:
            raise FileNotFoundError(f"{name} was not found in the supplied DFC2019 Track 3 data.")
        archive, member = self.files[name]
        with archive.open(member) if archive else member.open("rb") as source:
            yield source

    @contextmanager
    def raster(self, name):
        if name not in self.files:
            raise FileNotFoundError(f"{name} was not found in the supplied DFC2019 Track 3 data.")
        archive, member = self.files[name]
        if archive is None:
            with rasterio.open(member) as image:
                yield image
        else:
            with archive.open(member) as source, MemoryFile(source) as memory, memory.open() as image:
                yield image


def read_dfc_image(dfc, camera, crop_offset):
    filename = camera["img"]
    msi_name = filename.replace("_RGB.tif", "_MSI.tif")
    msi = msi_name in dfc
    with dfc.raster(msi_name if msi else filename) as image:
        rpc = image.tags(ns="RPC")
        origin = np.array([float(rpc["SAMP_OFF"]), float(rpc["LINE_OFF"])]) - crop_offset
        x, y = rounded = np.rint(origin).astype(int)
        width, height = camera["width"], camera["height"]
        if (
            np.max(np.abs(origin - rounded)) > 0.05
            or x < 0
            or y < 0
            or x + width > image.width
            or y + height > image.height
        ):
            raise ValueError(
                f"Source geometry does not match the PERS crop for {filename}: {origin.tolist()}."
            )
        rgb = image.read([5, 3, 2] if msi else [1, 2, 3], window=Window(x, y, width, height))

    if msi:
        low, high = np.percentile(rgb, [1, 99], axis=(1, 2), keepdims=True)
        rgb = (np.clip((rgb - low) / (high - low + 1e-8), 0, 1) * 255).astype(np.uint8)
    return rgb


def write_tiff(path, raster, **profile):
    count, height, width = raster.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        count=count,
        height=height,
        width=width,
        dtype=raster.dtype,
        compress="deflate",
        **profile,
    ) as output:
        output.write(raster)


def dsm_coordinates(dsm_path):
    with rasterio.open(dsm_path) as raster:
        height = raster.read(1, masked=True).astype(np.float64).filled(np.nan)
        row, col = np.indices(height.shape, dtype=np.float64)
        east, north = raster.transform * (col + 0.5, row + 0.5)
        grid = {"crs": raster.crs, "transform": raster.transform}
    return np.stack((east, north, height), axis=-1), grid


def project_ortho(rgb, camera, coordinates):
    model = camera["model"]
    scene = (coordinates - np.asarray(model["center"])) / model["scale"]
    uv = scene @ np.asarray(model["coef_"][:2]).T + np.asarray(model["intercept_"][:2])
    col = (uv[..., 0] + 1) * camera["width"] / 2 - 0.5
    row = (uv[..., 1] + 1) * camera["height"] / 2 - 0.5
    pixels = np.where(np.isfinite([row, col]), [row, col], -2)
    return np.stack([map_coordinates(band, pixels, order=1, mode="constant", cval=0) for band in rgb])


def copy_truth(opener, site, target, crs, has_grid_text):
    grid = {}
    if has_grid_text:
        name = f"{site}_DSM.txt"
        with opener(name) as source:
            (target / name).write_bytes(source.read())
        east, north, size, gsd = np.loadtxt(target / name)
        grid = {"crs": crs, "transform": from_origin(east, north + size * gsd, gsd, gsd)}

    for suffix in ("DSM", "CLS"):
        name = f"{site}_{suffix}.tif"
        with opener(name) as source:
            if not grid:
                with (target / name).open("wb") as out:
                    shutil.copyfileobj(source, out)
                continue
            with MemoryFile(source) as mem, mem.open() as raster:
                write_tiff(target / name, raster.read(), nodata=raster.nodata, **grid)


def prepare_scene(site, target, camera_list, layout, geometry, eogs, dfc):
    use_dfc = not site.startswith("IARPA_")
    target.joinpath("images").mkdir(parents=True)
    target.joinpath("initialization", "orthos").mkdir(parents=True)
    target.joinpath("truth").mkdir(parents=True)
    dsm_path = target / "initialization" / "dsm.tif"
    with geometry.extractfile(f"{site}/initialization/dsm.tif") as source, dsm_path.open("wb") as out:
        shutil.copyfileobj(source, out)
    coordinates, grid = dsm_coordinates(dsm_path)

    for camera in camera_list:
        filename = camera["img"]
        inputs = layout[filename]
        if use_dfc:
            rgb = read_dfc_image(dfc, camera, inputs["dfc_crop_rpc_offset"])
        else:
            with (
                eogs.open(f"images/{site}/{filename}") as source,
                MemoryFile(source) as mem,
                mem.open() as image,
            ):
                rgb = image.read()

        if rgb.shape != (3, camera["height"], camera["width"]) or not np.isfinite(rgb).all():
            raise ValueError(f"Unexpected RGB dimensions or non-finite values in {site}/{filename}.")
        write_tiff(target / "images" / filename, rgb)
        write_tiff(
            target / "initialization" / "orthos" / filename, project_ortho(rgb, camera, coordinates), **grid
        )

    def open_truth(name):
        return dfc.open(name) if use_dfc else eogs.open(f"truth/{site}/{name}")

    copy_truth(open_truth, site, target / "truth", grid["crs"], not site.startswith("IARPA_"))
    (target / "cameras.json").write_text(json.dumps(camera_list, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--sites",
        nargs="+",
        help="Scene names (defaults to IARPA scenes, or all sites with --dfc-source)",
    )
    parser.add_argument(
        "--cache", type=Path, help="Download directory (defaults to a sibling SatSplat-downloads folder)"
    )
    parser.add_argument("--geometry", type=Path, help="Use a local copy of the published geometry archive")
    parser.add_argument("--eogs-zip", type=Path, help="Use a local copy of the EOGS data.zip")
    parser.add_argument(
        "--dfc-source",
        type=Path,
        action="append",
        help="Use an official Track 3 zip or extracted directory for DFC2019 scenes. Repeat for additional files.",
    )
    args = parser.parse_args()
    cache = args.cache or args.destination.parent / "SatSplat-downloads"
    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

    with ExitStack() as stack:
        archive_path = args.geometry or download(
            GEOMETRY_URL, GEOMETRY_SHA256, cache / "SatSplatDataset_PERS_geometry_v1.0.0.tar.gz"
        )
        geometry = stack.enter_context(tarfile.open(archive_path, "r:gz"))
        layout = json.load(geometry.extractfile("inputs.json"))
        layout.pop("JAX_164", None)
        eogs_sites = [site for site in layout if site.startswith("IARPA_")]
        sites = list(dict.fromkeys(args.sites or (layout if args.dfc_source else eogs_sites)))
        missing = [site for site in sites if site not in layout]
        if missing:
            parser.error(f"Unknown scenes: {', '.join(missing)}")
        if not args.dfc_source and (missing := [site for site in sites if site not in eogs_sites]):
            parser.error(
                f"Supply the official DFC2019 Track 3 input via --dfc-source for {', '.join(missing)}"
            )
        dfc = DfcFiles(args.dfc_source, stack) if args.dfc_source else None
        needs_eogs = any(site in eogs_sites for site in sites)
        eogs = None
        if needs_eogs:
            eogs_path = args.eogs_zip or download(EOGS_URL, EOGS_SHA256, cache / "EOGS-dataset_v01.zip")
            eogs = stack.enter_context(zipfile.ZipFile(eogs_path))

        args.destination.mkdir(parents=True, exist_ok=True)
        for site in sites:
            output = args.destination / site
            if output.exists():
                parser.error(f"{output} already exists. Choose a new destination for this scene.")
            cameras = json.load(geometry.extractfile(f"{site}/cameras.json"))
            with tempfile.TemporaryDirectory(dir=args.destination) as temp:
                staged = Path(temp) / site
                prepare_scene(site, staged, cameras, layout[site], geometry, eogs, dfc)
                staged.rename(output)
            print(f"{site}: prepared {len(cameras)} views", flush=True)


if __name__ == "__main__":
    main()
