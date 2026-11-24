# SatSplat

SatSplat reconstructs georeferenced digital surface models (DSMs) and orthophotos from multi-date satellite imagery. This repository accompanies the PERS paper by Shuang Song, Jiyong Kim, and Rongjun Qin. It implements affine 2D Gaussian splatting, online camera refinement, shadow-aware appearance, and surface-area-weighted initialization from a coarse stereo DSM.

[Project page](https://gdaosu.github.io/satsplat/) · [Interactive demo](https://gdaosu.github.io/satsplat_html/) · [PERS paper](https://www.ingentaconnect.com/content/asprs/pers/pre-prints/content-26-00025r2) · [arXiv and supplementary material](https://arxiv.org/abs/2606.28581)

## Install

Use Linux x86-64 with glibc 2.28 or newer, an NVIDIA driver that supports CUDA 12.6, and [Pixi](https://pixi.sh/latest/installation/) 0.84.2 or newer. The lock contains Python 3.12, the PyTorch CUDA 12.6 wheels, the CUDA compiler and development headers, GCC 11, and the exact affine gsplat commit. Budget several GB of disk space for the environment and build cache.

From the repository root, install the locked environment and compile gsplat against the installed PyTorch. The first compilation can take several minutes.

```bash
BUILD_NO_CUDA=1 pixi install --locked
pixi run --locked python -c 'import torch; from gsplat.cuda._backend import _C; assert torch.cuda.is_available() and _C is not None; print(torch.__version__, torch.cuda.get_device_name())'
```

`BUILD_NO_CUDA=1` defers extension compilation until the GPU environment is active. Pixi sets the CUDA and compiler paths for each command. The host supplies the NVIDIA driver.

## Run the smoke test

Generate the CC0 synthetic scene or extract the release archive under `data/`, then train for 40 steps. This exercises initialization, image formation, camera updates, final GeoTIFF export and DSM evaluation. [Download test data](https://github.com/GDAOSU/SatSplat-PERS/releases/download/v1.0.0/SatSplatSynthetic.tar.gz) · [SHA256](https://github.com/GDAOSU/SatSplat-PERS/releases/download/v1.0.0/SatSplatSynthetic.tar.gz.sha256)

```bash
pixi run --locked python scripts/make_smoke_dataset.py data/SatSplatSynthetic
pixi run --locked sstrain data_root=data/SatSplatSynthetic site=SYNTHETIC \
  max_steps=40 pipeline.schedule_cfg.camopt_start_iter=10 \
  pipeline.init_cfg.initializer.num_samples=5000 pipeline.strategy.cap_max=5000 \
  hydra.run.dir=runs/smoke
```

## Reproduce a benchmark site

The setup script uses the official DFC2019 Track 3 imagery and reference rasters for all JAX and OMA scenes. The IARPA scenes come from the EOGS dataset release. It combines these inputs with the published SatSplat geometry and generates initialization orthophotos locally. [Dataset setup and source attribution](docs/dataset.md)

```bash
pixi run --locked python scripts/fetch_dataset.py data/SatSplatExample \
  --dfc-source /path/to/Track3 --sites JAX_068
export SATSPLAT_DATA_ROOT="$PWD/data/SatSplatExample"
pixi run --locked sstrain site=JAX_068 seed=42 hydra.run.dir=runs/JAX_068
pixi run --locked ssrender runs/JAX_068/final_iter030000/state_dict.pth
```

The PERS preset runs for 30,000 updates and samples one view per step. It starts shadow mapping at the first update and enables affine camera refinement after 3,000 steps. The default seed is 42 and controls surface sampling, view selection and PyTorch randomness. GPU kernel reductions can introduce small numerical differences between runs. Density control settings, loss weights and the seed are recorded in each run's resolved `config.yaml`. Use `pixi run --locked sstrain --cfg job --resolve` to inspect the full setting. Put `dataset.preload=false` on the command line to read images on demand.

Prepare the 13 scenes, then use Hydra's multirun to train each scene in sequence.

```bash
pixi run --locked python scripts/fetch_dataset.py data/SatSplatDataset_PERS --dfc-source /path/to/Track3
export SATSPLAT_DATA_ROOT="$PWD/data/SatSplatDataset_PERS"
pixi run --locked sstrain --multirun seed=42 \
  site=JAX_004,JAX_068,JAX_168,JAX_214,JAX_251,JAX_260,JAX_280,OMA_203,OMA_212,OMA_315,IARPA_001,IARPA_002,IARPA_003
```

The repo starts from fitted cameras and stereo initialization rasters. Raw image preparation in the study used pansharpened imagery, RPC bundle adjustment, affine fitting and Ames Stereo Pipeline (ASP). Refer to Appendix B for the experimental setup. [Paper and supplement](https://arxiv.org/abs/2606.28581)

### Outputs and evaluation

- `model.ply`, `state_dict.pth`, and `aux_params.yaml` contain the trained splats and learned appearance and camera parameters.
- `dsm.tif` stores predicted heights in meters and `ortho.tif` stores RGB albedo on the same UTM grid. The final checkpoint also contains `dsm_registered.tif`, `dsm_error.tif`, and `metrics.json`. Training logs and TensorBoard events live in the run directory.
- Evaluation renders onto the reference DSM grid, searches for a horizontal translation by multiscale normalized correlation and fits a constant vertical shift by the median residual. Benchmark MAE excludes water and elevated road or bridge pixels (classes 9 and 17) and caps absolute residuals at 10 m. Building MAE uses class 6 on the same registered grid. `metrics.json` records the shifts and valid overlap count.
- `ssrender` re-exports from a trained checkpoint. Use `--data-root` if the benchmark directory moved, or `--output` to select an export directory. Intermediate checkpoints are model snapshots for rendering.
- Initial splat scales use SciPy's KD-tree search for the three closest neighbors. The updated CUDA/PyTorch stack, registration implementation, and sampling can shift metrics relative to the paper.

### Validation

All 13 scenes were prepared with the public setup script, using the official DFC2019 MSI and reference files for JAX and OMA and the EOGS archive for IARPA. Each scene completed the 30,000-step PERS preset with seed 42 in the locked Pixi environment on an NVIDIA RTX 6000 Ada. All values below are registered MAE in meters. Paper values are from Tables S1 and S2. [Supplementary material](https://arxiv.org/abs/2606.28581)

| Site | All classes | Paper | Buildings | Paper |
| --- | ---: | ---: | ---: | ---: |
| JAX_004 | 1.39 | 1.46 | 0.80 | 0.82 |
| JAX_068 | 0.95 | 1.04 | 0.85 | 0.99 |
| JAX_168 | 0.95 | 0.94 | 0.84 | 0.87 |
| JAX_214 | 1.35 | 1.49 | 0.98 | 1.19 |
| JAX_251 | 1.11 | 1.07 | 0.81 | 0.79 |
| JAX_260 | 1.59 | 1.48 | 0.88 | 1.01 |
| JAX_280 | 1.73 | 1.85 | 0.69 | 0.83 |
| OMA_203 | 1.05 | 1.10 | 0.53 | 0.59 |
| OMA_212 | 0.58 | 0.63 | 1.83 | 2.10 |
| OMA_315 | 1.21 | 1.19 | 1.60 | 1.42 |
| IARPA_001 | 1.55 | 1.63 | 0.87 | 0.88 |
| IARPA_002 | 1.94 | 2.09 | 0.84 | 1.02 |
| IARPA_003 | 2.08 | 2.04 | 1.23 | 1.68 |
| Mean (13 paper sites) | 1.35 | 1.39 | 0.98 | 1.09 |

### Custom prepared scenes

Create a folder for your site with `images/`, `cameras.json`, `initialization/dsm.tif`, `initialization/orthos/`, and optionally `truth/`. Use the dataset guide for camera and GeoTIFF conventions. For a scene without reference rasters pass `dataset.truth_dir=null` to export on the scene's native ortho grid. Images and cameras must agree on the filename and raster size.

## Citation

```bibtex
@article{song2026satsplat,
  title   = {SatSplat: Geometrically Accurate Gaussian Splatting for Satellite Imagery},
  author  = {Song, Shuang and Kim, Jiyong and Qin, Rongjun},
  journal = {Photogrammetric Engineering & Remote Sensing},
  year    = {2026},
  doi     = {10.14358/pers.26-00025r2}
}
```

The source code uses Apache-2.0. SatSplat builds on EOGS, 2D Gaussian Splatting, and gsplat. The benchmark data follow their providers' terms. [License](LICENSE) · [Notice](NOTICE) · [EOGS](https://github.com/mezzelfo/EOGS) · [gsplat](https://github.com/nerfstudio-project/gsplat)
