# SatSplat

SatSplat reconstructs georeferenced digital surface models (DSMs) and orthophotos from multi-date satellite imagery. This repository accompanies the PERS paper by Shuang Song, Jiyong Kim, and Rongjun Qin. It implements affine 2D Gaussian splatting, online camera refinement, shadow-aware appearance, and surface-area-weighted initialization from a coarse stereo DSM.

[Project page](https://gdaosu.github.io/satsplat/) · [Interactive demo](https://gdaosu.github.io/satsplat_html/) · [PERS paper](https://www.ingentaconnect.com/content/asprs/pers/pre-prints/content-26-00025r2) · [arXiv and supplementary material](https://arxiv.org/abs/2606.28581)

## Install

Use Linux x86-64, Python 3.12, an NVIDIA GPU with a CUDA 12.6-compatible driver, CUDA Toolkit 12.6 with `nvcc`, a C++ compiler supported by the toolkit, and [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.11.4. GCC 11 and the RTX 6000 Ada were used for validation. Budget several GB for the CUDA Python environment and build cache.

From the repository root, set `CUDA_HOME` to your CUDA 12.6 installation, then install the locked environment. `BUILD_NO_CUDA=1` installs the pinned gsplat source and lets it compile its CUDA extension against the active PyTorch on first use.

```bash
export CUDA_HOME=/usr/local/cuda-12.6
export MAX_JOBS=4
export TORCH_EXTENSIONS_DIR="$PWD/.venv/torch_extensions"
"$CUDA_HOME/bin/nvcc" --version

uv python install 3.12
BUILD_NO_CUDA=1 uv sync --locked
uv run --no-sync python -c 'import torch; from gsplat.cuda._backend import _C; assert torch.cuda.is_available() and _C is not None; print(torch.__version__, torch.cuda.get_device_name())'
```

For a CUDA toolkit installed under a Conda prefix, set `CUDA_HOME` to that prefix. If its CUDA headers are under `targets/x86_64-linux/include`, also set `CPATH` to include that directory. The first build can take several minutes. Use `MAX_JOBS=2` if compilation runs out of host memory. The lock selects the PyTorch CUDA 12.6 wheels and the exact SatSplat affine gsplat commit. Editable installs expose Hydra configurations and the `sstrain` and `ssrender` entry points.

## Run the smoke test

Generate the CC0 synthetic scene or extract the release archive under `data/`, then train for 40 steps. This exercises initialization, image formation, camera updates, final GeoTIFF export and DSM evaluation. [Download test data](https://github.com/GDAOSU/SatSplat-PERS/releases/download/v1.0.0/SatSplatSynthetic.tar.gz) · [SHA256](https://github.com/GDAOSU/SatSplat-PERS/releases/download/v1.0.0/SatSplatSynthetic.tar.gz.sha256)

```bash
uv run --no-sync python scripts/make_smoke_dataset.py data/SatSplatSynthetic
uv run --no-sync sstrain data_root=data/SatSplatSynthetic site=SYNTHETIC \
  max_steps=40 pipeline.schedule_cfg.camopt_start_iter=10 \
  pipeline.init_cfg.initializer.num_samples=5000 pipeline.strategy.cap_max=5000 \
  hydra.run.dir=runs/smoke
```

## Reproduce a benchmark site

The setup script downloads the published SatSplat geometry and the EOGS benchmark inputs, then generates initialization orthophotos locally. It prepares seven scenes by default. Use `--dfc-source` with the official Track 3 files for the other DFC2019 scenes. [Dataset setup and source attribution](docs/dataset.md) · [Geometry archive](https://github.com/GDAOSU/SatSplat-PERS/releases/tag/v1.0.1)

```bash
uv run --no-sync python scripts/fetch_dataset.py data/SatSplatDataset_PERS --sites JAX_068
export SATSPLAT_DATA_ROOT="$PWD/data/SatSplatDataset_PERS"
uv run --no-sync sstrain site=JAX_068 hydra.run.dir=runs/JAX_068
uv run --no-sync ssrender runs/JAX_068/final_iter030000/state_dict.pth
```

The PERS preset runs for 30,000 updates and samples one view per step. It starts shadow mapping at the first update and enables affine camera refinement after 3,000 steps. The seed, density control settings and paper loss weights are recorded in each run's resolved `config.yaml`. Use `sstrain --cfg job --resolve` to inspect the full setting. Put `dataset.preload=false` on the command line to read images on demand.

The published tables report these sites. Use Hydra's multirun to run each scene in sequence.

```bash
uv run --no-sync sstrain --multirun \
  site=JAX_004,JAX_068,JAX_168,JAX_214,JAX_251,JAX_260,JAX_280,OMA_203,OMA_212,OMA_315,IARPA_001,IARPA_002,IARPA_003
```

JAX_164 is also available in the prepared data. The repo starts from fitted cameras and stereo initialization rasters. Raw image preparation in the study used pansharpened imagery, RPC bundle adjustment, affine fitting and Ames Stereo Pipeline (ASP). Refer to Appendix B for the experimental setup. [Paper and supplement](https://arxiv.org/abs/2606.28581)

### Outputs and evaluation

- `model.ply`, `state_dict.pth`, and `aux_params.yaml` contain the trained splats and learned appearance and camera parameters.
- `dsm.tif` stores predicted heights in meters and `ortho.tif` stores RGB albedo on the same UTM grid. The final checkpoint also contains `dsm_registered.tif`, `dsm_error.tif`, and `metrics.json`. Training logs and TensorBoard events live in the run directory.
- Evaluation renders onto the reference DSM grid, searches for a horizontal translation by multiscale normalized correlation and fits a constant vertical shift by the median residual. Benchmark MAE excludes water and elevated road or bridge pixels (classes 9 and 17) and caps absolute residuals at 10 m. Building MAE uses class 6 on the same registered grid. `metrics.json` records the shifts and valid overlap count.
- `ssrender` re-exports from a trained checkpoint. Use `--data-root` if the benchmark directory moved, or `--output` to select an export directory. Intermediate checkpoints are model snapshots for rendering.
- Initial splat scales use SciPy's KD-tree search for the three closest neighbors. The updated CUDA/PyTorch stack, registration implementation, and sampling can shift metrics relative to the paper.

### Validation

All 14 prepared scenes passed a short GPU run through GeoTIFF export and evaluation. Two scenes completed the full 30,000-step preset with seed 42 on an RTX 6000 Ada. All values below are registered MAE in meters. Paper values are from Tables S1 and S2. [Supplementary material](https://arxiv.org/abs/2606.28581)

| Site | All classes | Paper | Buildings | Paper |
| --- | ---: | ---: | ---: | ---: |
| JAX_068 | 0.94 | 1.04 | 0.83 | 0.99 |
| IARPA_001 | 1.48 | 1.63 | 0.83 | 0.88 |

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
