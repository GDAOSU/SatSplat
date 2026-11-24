"""Collect the inputs used by the PERS training and evaluation commands."""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

CAMERA_FIELDS = ("img", "width", "height", "acquisition_date", "geojson", "model")
MODEL_FIELDS = ("coef_", "intercept_", "center", "scale", "n", "l", "min_world", "max_world")


def copy_file(source: Path, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def package_scene(source: Path, target: Path, iarpa_source: Path):
    site = source.name
    inputs = iarpa_source / site if site.startswith("IARPA_") else source
    models = inputs / "affine_models"
    cameras = json.loads((models / "affine_models.json").read_text())
    cameras_by_name = {camera["img"]: camera for camera in cameras}
    if len(cameras_by_name) != len(cameras):
        raise ValueError(f"{site}: duplicate camera names")

    images = sorted((inputs / "images").glob("*.tif"))
    if not images:
        raise ValueError(f"{site}: no images in {inputs}")
    output_cameras = []
    for image in images:
        original = cameras_by_name[image.name]
        camera = {key: original[key] for key in CAMERA_FIELDS}
        camera["geojson"] = {"center": original["geojson"]["center"]}
        camera["model"] = {key: original["model"][key] for key in MODEL_FIELDS}
        output_cameras.append(camera)
        copy_file(image, target / "images" / image.name)
        copy_file(models / "asp" / "ortho" / image.name, target / "initialization" / "orthos" / image.name)

    (target / "cameras.json").write_text(json.dumps(output_cameras, indent=2) + "\n")
    copy_file(models / "asp" / "dem" / "ASP-DEM.tif", target / "initialization" / "dsm.tif")
    for suffix in ("DSM.tif", "CLS.tif", "DSM.txt"):
        raster = source / "truth" / f"{site}_{suffix}"
        if raster.is_file() or suffix != "DSM.txt":
            copy_file(raster, target / "truth" / raster.name)
    return {"site": site, "views": len(output_cameras)}


def sha256(path: Path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original SatSplatDataset_PERS directory")
    parser.add_argument("destination", type=Path, help="New directory to create")
    parser.add_argument(
        "--iarpa-source",
        required=True,
        type=Path,
        help="PERS IARPA inputs with matching cameras and ASP outputs",
    )
    args = parser.parse_args()
    if args.destination.exists():
        parser.error(f"Destination already exists: {args.destination}")
    sites = sorted(path for path in args.source.iterdir() if path.is_dir())
    if not sites:
        parser.error(f"No sites found in {args.source}")

    with tempfile.TemporaryDirectory(dir=args.destination.parent) as temp:
        staged = Path(temp) / args.destination.name
        manifest = []
        for source in sites:
            record = package_scene(source, staged / source.name, args.iarpa_source)
            manifest.append(record)
            print(f"{source.name}: {record['views']} views", flush=True)
        (staged / "scenes.json").write_text(json.dumps(manifest, indent=2) + "\n")
        shutil.copyfile(Path(__file__).resolve().parent.parent / "docs" / "dataset.md", staged / "README.md")
        checksums = [
            f"{sha256(path)}  {path.relative_to(staged).as_posix()}"
            for path in sorted(staged.rglob("*"))
            if path.is_file()
        ]
        (staged / "SHA256SUMS").write_text("\n".join(checksums) + "\n")
        staged.rename(args.destination)


if __name__ == "__main__":
    main()
