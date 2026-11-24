# SatSplat PERS reproduction data

The prepared benchmark data contain the 13 sites reported in Tables S1 and S2 of the paper, with 340 RGB views.

## Prepare benchmark scenes

The public geometry archive contains fitted affine cameras, stereo-derived coarse DSMs and DFC2019 crop offsets. The setup script reads seven JAX and three OMA scenes from the official DFC2019 Track 3 files. It retrieves the three IARPA scenes from the EOGS release and generates orthophotos from the coarse DSM and affine cameras locally. Automatic downloads are checked against SHA-256 digests. [Geometry release](https://github.com/GDAOSU/SatSplat/releases/tag/v1.0.0) · [EOGS dataset](https://github.com/mezzelfo/EOGS/releases/tag/dataset_v01)

```bash
pixi run --locked python scripts/fetch_dataset.py data/SatSplatDataset_PERS --dfc-source /path/to/Track3
```

With `--dfc-source` the default is all 13 scenes. Without it, the script defaults to IARPA_001, IARPA_002 and IARPA_003. Use `--sites` to select a subset. The EOGS archive is approximately 1.1 GB and is only downloaded when an IARPA scene is selected. Downloads are cached in a sibling `SatSplat-downloads` directory. To reuse a prior download, provide `--eogs-zip /path/to/data.zip` and optionally `--geometry /path/to/SatSplatDataset_PERS_geometry.tar.gz`.

For the DFC2019 scenes, download the Track 3 training MSI and reference files via IEEE DataPort. Pass the extracted directory or repeat `--dfc-source` for multiple zip files. The script reads the MSI bands and reproduces the PERS crop using the source RPC offsets. Original Track 3 RGB files with matching RPC offsets also work and can produce different scores. [IEEE DataPort Track 3](https://doi.org/10.21227/c6tm-vw12)

```bash
pixi run --locked python scripts/fetch_dataset.py data/SatSplatDataset_PERS \
  --dfc-source /path/to/Track3 \
  --sites JAX_068 OMA_212
```

Use a fresh destination when preparing a scene again. Upstream radiometry and locally generated orthophotos can change scores relative to the paper.

## On-disk layout

```text
SatSplatDataset_PERS/
├── scenes.json
├── SHA256SUMS
└── JAX_068/
    ├── cameras.json
    ├── images/
    │   └── JAX_068_001_RGB.tif
    ├── initialization/
    │   ├── dsm.tif
    │   └── orthos/
    │       └── JAX_068_001_RGB.tif
    └── truth/
        ├── JAX_068_CLS.tif
        ├── JAX_068_DSM.tif
        └── JAX_068_DSM.txt
```

- `images/` contains the cropped RGB training views. All supplied views are used for DSM reconstruction.
- `cameras.json` is a list with one entry per image. Each entry has `img`, integer `width` and `height`, UTC `acquisition_date` formatted as `YYYYMMDDhhmmss`, and longitude and latitude in `geojson.center`. The model has `coef_` (3×3) and `intercept_` (three entries). The first two output coordinates are normalized image coordinates with right and down positive. The input coordinates are UTM east, north and altitude after subtracting `model.center` and dividing by `model.scale`. Camera metadata also records the common UTM zone number `n`, latitude band `l` and normalized scene bounds `min_world` and `max_world`.
- `initialization/` contains the coarse stereo height raster in meters from ASP and locally generated orthoprojected RGB images for geometry and color initialization.
- `truth/` contains the reference DSM and classification rasters. For DFC2019, the text file describes minimum easting, minimum northing, pixel width and ground sample distance. IARPA reference grids come from the GeoTIFF metadata. Class 6 marks buildings. Class rasters are reprojected to the reference DSM grid by nearest-neighbor sampling for evaluation.

Lab-prepared copies include `SHA256SUMS`. Run `sha256sum -c SHA256SUMS` from the dataset root to verify those files.

## Data sources

The JAX and OMA scenes come from the original DFC2019 PERS data. The IARPA2016 scenes use the later PERS inputs with matching fitted cameras and stereo initialization from the wrapped dataset. The preparation script collects the files consumed by training and evaluation. Researchers with access to both processed source trees can run it with Python 3.12.

```bash
pixi run --locked python scripts/package_dataset.py /path/to/original/SatSplatDataset_PERS \
  /path/to/clean/SatSplatDataset_PERS --iarpa-source /path/to/wrapped_dataset/satsplat
```

The benchmark imagery and reference rasters retain the original providers' terms. The 2019 IEEE GRSS Data Fusion Contest terms restrict redistribution of its data packages. See the original sources for authorized access or use the CC0 synthetic example to test the software. [IEEE GRSS DFC2019](https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019) · [EOGS benchmark distribution](https://github.com/mezzelfo/EOGS/releases/tag/dataset_v01)

The authors would like to thank the Johns Hopkins University Applied Physics Laboratory and IARPA for providing the data used in this study, and the IEEE GRSS Image Analysis and Data Fusion Technical Committee for organizing the Data Fusion Contest.
