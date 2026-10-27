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
| `masks/mask_YYYYMMDD.tif` | Part 2 (Krisha, `masks.py`) | uint8 on the same 10 m grid, one per S2 date: 0 clear, 1 cloud, 2 cloud shadow, 3 terrain shadow, 255 no data |
| `tiles.csv` | Part 3 (Krisha) | columns `row, col, split` (+ optional `glacier`). Each row/col is a 256×256 tile's top-left pixel. Splits are `train`/`val`/`test`, and train tiles must not overlap val/test tiles |

The NDSI channel is computed from the S2 bands. A separate preprocessed NDSI file isn't needed.

`--masks <folder>` (train.py and predict.py) uses the Part 2 masks: cloud (1), cloud shadow (2) and no data (255)
become no data, so they are ignored in training and come out as 255 in the predicted masks. Terrain shadow (3) is
kept, because shadowed ice is still glacier. Each S2 file gets the mask with the same date; an `S2GEE_YYYY.tif` file
gets the single mask of that year. These masks replace `--mask-cloud`, which uses SCL and is wrong before 2022.

## Run

```
venv\Scripts\python make_dummy.py --out dummy                    # synthetic data for testing
venv\Scripts\python train.py --s2 dummy\S2_20991024.tif --dem dummy\CopDEM_GLO30_UTM45N.tif ^
    --label dummy\label_2099.tif --tiles dummy\tiles.csv --out runs\dummy --weights none --masks dummy\masks
venv\Scripts\python predict.py --model runs\dummy\best.pt --dem dummy\CopDEM_GLO30_UTM45N.tif ^
    --s2 dummy\S2_*.tif --out runs\dummy\pred --masks dummy\masks
venv\Scripts\python predict.py --model runs\dummy\best.pt --dem dummy\CopDEM_GLO30_UTM45N.tif ^
    --s2 dummy\S2_20991024.tif --label dummy\label_2099.tif --tiles dummy\tiles.csv --split test ^
    --out runs\dummy\test
```

`train.py` writes `best.pt` (the weights plus the normalisation statistics and settings), along with `stats.json` and `log.csv`.
`predict.py` writes `<scene>_prob.tif`, `<scene>_mask.tif` and `areas.csv` (glacier km² per year). If
`--label` is given, it also reports test-tile IoU for both the U-Net and the NDSI > 0.4 baseline.

## Random Forest baseline

```
venv\Scripts\python rf_baseline.py --s2 dummy\S2_20991024.tif --dem dummy\CopDEM_GLO30_UTM45N.tif ^
    --label dummy\label_2099.tif --tiles dummy\tiles.csv --out runs\rf_dummy --predict dummy\S2_*.tif
```

This is a per-pixel scikit-learn `RandomForestClassifier` on the same 9 inputs as the U-Net. It samples up to
50 000 labelled pixels per class (`--samples`) from the train tiles only, then maps the reference scene and
any `--predict` scenes. It prints val/test IoU next to the NDSI > 0.4 baseline, plus the importance of each
input. It writes `rf.joblib`, `scores.json` and `<scene>_prob.tif` / `<scene>_mask.tif` in the same format as
`predict.py`. On the full 3220 × 3940 grid, fitting takes about 2 s and mapping one scene about 12 s on the laptop CPU.

## Notes
- `--no-terrain` (train.py) leaves out elevation and slope, giving 7 inputs. Elevation and slope are the same
  every year, so with one labelled year the model might learn glacier position from terrain alone. Train
  both versions and compare. The choice is stored in `best.pt`, so `predict.py` needs no flag. `--dem` is
  still required: pixels outside the DEM stay invalid, so both versions are scored on the same pixels.
- SCL cloud masking is **off** by default (`--mask-cloud` turns it on), because SCL marks snow and debris
  as cloud before 2022 (see `decisions.md`). Cloud gap-filling belongs to Part 2.
- Approach B files (`S2GEE_YYYY.tif`, Earth Engine composites) have only the 6 bands and no SCL. They load
  as they are: Cloud Score+ already masked cloud in Earth Engine, so the SCL steps (and `--mask-cloud`) are
  skipped for them. The bands must be in the order B02, B03, B04, B08, B11, B12, either named that way or unnamed.
- Areas count every pixel classed as glacier, so lakes and seasonal snow are included if the model labels them glacier.
  The ± half-pixel error margin is added in the results step.

## Per-glacier areas

```
venv\Scripts\python glacier_areas.py --masks runs\dummy\pred\S2_*_mask.tif --glaciers dummy\glacier_ids.tif ^
    --lookup dummy\glacier_ids.csv --method unet --results results.csv
```

This needs a glacier-ID raster on the same 10 m grid (one integer per glacier, 0 elsewhere) plus a lookup CSV
with columns `value, glacier_id, glacier_name`. It writes area per glacier per year to the shared
`results.csv` (`glacier_id, glacier_name, year, preprocessing, method, area_km2`). Year and preprocessing
come from the file name: `S2_` files are A and `S2GEE_` files are B. Rows for the same glacier, year,
preprocessing and method are replaced, and all other rows are kept. If more than 5 % of a glacier has no
data (`--max-nodata`), its area is left empty instead of being undercounted. `make_dummy.py` also writes a
dummy `glacier_ids.tif` and `glacier_ids.csv` (one outline per connected glacier).
