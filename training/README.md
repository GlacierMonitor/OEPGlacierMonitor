# GlacierMonitor Part 4: U-Net training

The model trains on one labelled reference year and is then run on all 10 years.
Inputs (9 channels): B02, B03, B04, B08, B11, B12 (reflectance 0–1), NDSI, elevation, slope.
The data is normalised with training-tile statistics. The model is a U-Net with a ResNet34 encoder
(ImageNet weights, first conv adapted to 9 channels), trained with Dice+BCE loss in bf16 mixed precision.

## Setup (done on the RTX 4050 laptop)

```
py -3.14 -m venv venv
venv\Scripts\python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
venv\Scripts\python -m pip install -r requirements.txt
venv\Scripts\python gpu_check.py
```

Without an NVIDIA GPU everything still runs on the CPU (slowly), or you can use Colab.

## Dummy-data test (6 Oct)

| Test | Result |
|---|---|
| GPU memory, batch 8 → 32 | 0.8 → 2.2 GB of 6 GB; about 120 tiles/s from batch 12 up |
| Train on full-size dummy grid (108 train tiles) | val IoU 0.997 after about 20 epochs, about 1 s per epoch; 1.25 GB GPU, 2.2 GB RAM |
| Predict a full 3220×3940 scene | 2.7 GB RAM |
| Test tiles: U-Net vs NDSI > 0.4 | 0.9965 vs 0.874 (NDSI misses debris-covered ice) |

Dummy data is synthetic, so these numbers only show that the code works. They say nothing about accuracy on real data.

## What the code needs from the other parts

| File | From | Format |
|---|---|---|
| `S2_YYYYMMDD.tif` | Part 1 `export.py` | unchanged: 7-band uint16, B02…B12 + SCL, nodata 0, 10 m, EPSG:32645 |
| `CopDEM_GLO30_UTM45N.tif` | Part 1 `export.py` | unchanged: 30 m. Slope is computed here, and both layers are resampled to 10 m |
| `label_<year>.tif` | Part 3 (Krisha) | uint8 on the **same 10 m grid** as that year's S2 file: 1 glacier, 0 not glacier, 255 ignore/unsure |
| `tiles.csv` | Part 3 (Krisha) | columns `row, col, split` (+ optional `glacier`). Each row/col is a 256×256 tile's top-left pixel. Splits are `train`/`val`/`test`, and train tiles must not overlap val/test tiles |

The NDSI channel is computed from the S2 bands. A separate preprocessed NDSI file isn't needed.

## Run

```
venv\Scripts\python make_dummy.py --out dummy                    # synthetic data for testing
venv\Scripts\python train.py --s2 dummy\S2_20991024.tif --dem dummy\CopDEM_GLO30_UTM45N.tif ^
    --label dummy\label_2099.tif --tiles dummy\tiles.csv --out runs\dummy --weights none
venv\Scripts\python predict.py --model runs\dummy\best.pt --dem dummy\CopDEM_GLO30_UTM45N.tif ^
    --s2 dummy\S2_*.tif --out runs\dummy\pred
venv\Scripts\python predict.py --model runs\dummy\best.pt --dem dummy\CopDEM_GLO30_UTM45N.tif ^
    --s2 dummy\S2_20991024.tif --label dummy\label_2099.tif --tiles dummy\tiles.csv --split test ^
    --out runs\dummy\test
```

`train.py` writes `best.pt` (the weights plus the normalisation statistics and settings), along with `stats.json` and `log.csv`.
`predict.py` writes `<scene>_prob.tif`, `<scene>_mask.tif` and `areas.csv` (glacier km² per year). If
`--label` is given, it also reports test-tile IoU for both the U-Net and the NDSI > 0.4 baseline.

## Notes
- `--no-terrain` (train.py) leaves out elevation and slope, giving 7 inputs. Elevation and slope are the same
  every year, so with one labelled year the model might learn glacier position from terrain alone. Train
  both versions and compare. The choice is stored in `best.pt`, so `predict.py` needs no flag. `--dem` is
  still required: pixels outside the DEM stay invalid, so both versions are scored on the same pixels.
- SCL cloud masking is **off** by default (`--mask-cloud` turns it on), because SCL marks snow and debris
  as cloud before 2022 (see `decisions.md`). Cloud gap-filling belongs to Part 2.
- Areas count every pixel classed as glacier, so lakes and seasonal snow are included if the model labels them glacier.
  The ± half-pixel error margin is added in the results step.
