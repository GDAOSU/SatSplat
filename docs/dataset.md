# SatSplat PERS reproduction data

The prepared research data contain 14 sites and 363 RGB views. Tables S1 and S2 in the paper report 13 sites. JAX_164 is an additional site from the research dataset.

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

- `images/` contains the cropped 8-bit RGB training views. All supplied views are used for DSM reconstruction.
- `cameras.json` is a list with one entry per image. Each entry has `img`, integer `width` and `height`, UTC `acquisition_date` formatted as `YYYYMMDDhhmmss`, and longitude and latitude in `geojson.center`. The model has `coef_` (3×3) and `intercept_` (three entries). The first two output coordinates are normalized image coordinates with right and down positive. The input coordinates are UTM east, north and altitude after subtracting `model.center` and dividing by `model.scale`. Camera metadata also records the common UTM zone number `n`, latitude band `l` and normalized scene bounds `min_world` and `max_world`.
- `initialization/` contains the coarse stereo height raster in meters and orthoprojected RGB images for geometry and color initialization. These are the two products of ASP consumed by training.
- `truth/` contains the reference DSM and classification rasters. For DFC2019, the text file describes minimum easting, minimum northing, pixel width and ground sample distance. IARPA reference grids come from the GeoTIFF metadata. Class 6 marks buildings. Class rasters are reprojected to the reference DSM grid by nearest-neighbor sampling for evaluation.

Use `sha256sum -c SHA256SUMS` from the dataset root to verify every file.

## Data sources

The JAX and OMA scenes come from the original DFC2019 PERS data. The IARPA2016 scenes use the later PERS inputs with matching fitted cameras and stereo initialization from the wrapped dataset. The preparation script collects the files consumed by training and evaluation. Researchers with access to both processed source trees can run it with Python 3.12.

```bash
python3.12 scripts/package_dataset.py /path/to/original/SatSplatDataset_PERS \
  /path/to/clean/SatSplatDataset_PERS --iarpa-source /path/to/wrapped_dataset/satsplat
```

The benchmark imagery and reference rasters retain the original providers' terms. The 2019 IEEE GRSS Data Fusion Contest terms restrict redistribution of its data packages. See the original sources for authorized access or use the CC0 synthetic example to test the software. [IEEE GRSS DFC2019](https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019) · [EOGS benchmark distribution](https://github.com/mezzelfo/EOGS/releases/tag/dataset_v01)

The authors would like to thank the Johns Hopkins University Applied Physics Laboratory and IARPA for providing the data used in this study, and the IEEE GRSS Image Analysis and Data Fusion Technical Committee for organizing the Data Fusion Contest.
