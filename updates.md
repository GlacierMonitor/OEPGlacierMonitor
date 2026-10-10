# GlacierMonitor Part 1 – Progress updates

## 2026-10-10 – Part 2 started: cloud and shadow masks (approach A)

### Done
- `masks.py` written and run for all 10 years (method: `decisions.md` items 14–16). Outputs are in `GlacierMonitor/masks/`:
  - `mask_YYYYMMDD.tif` (0 clear, 1 cloud, 2 cloud shadow, 3 terrain shadow, 255 no data)
  - `compare_YYYYMMDD.tif` (cloud agreement: OmniCloudMask vs SCL)
  - QGIS `.qml` styles, previews and `mask_summary.csv`
- Results: cloud is 0–2.6% in every year except 2023 (6.4%). The 2023 and 2025 clouds are real and lie in the SW valleys,
  off the main glaciers. **2022 (reference) has 0% cloud.** Terrain shadow is 3–7% in October and 11–12% in November (2024, 2025).
- SCL (with the snow correction) calls 14–48% of each pre-2022 scene cloud, almost none of it real.

### QGIS review (by eye, `compare_*.tif` over the image)
- **2022:** no cloud; SCL-only patches are bright debris. No fixes.
- **2017:** orange (OmniCloudMask only) is just a rim around clouds both methods found, so it's the thin cloud edge. No fixes.
- **2021:** mask correct, no fixes. Cloud lies over the **upper parts of Changri Nup/Shar**, and possibly the top of
  Lhotse Shar and upper Khumbu (approximate positions). The Khumbu and Ngozumpa tongues, Imja and Ama Dablam are clear.
  → The 2021 per-glacier areas for those glaciers will be incomplete: flag them or gap-fill.
  Check exactly with `glaciers.gpkg` once it exists.
- **2023:** mask correct, no fixes. Real cloud covers the SW valleys (6.4%). Khumbu, Lobuche and Gokyo are clear;
  the cloud field reaches the **very tip of the Ngozumpa tongue** at most. Note for the 2023 Ngozumpa area.
- **2025:** mask correct, no fixes. Real cloud covers the SW valleys only. The blurry dark patches in the upper Ngozumpa
  icefall area also appear in 2022 and 2024, so they are crevassed ice, not haze. For labelling: this is glacier, even though it looks dark.
- **2018:** mask correct, no fixes. Cloud (0.4%) sits over the Everest–Lhotse massif / top of the Khumbu icefall (upper Khumbu);
  the Khumbu tongue, Ngozumpa and Imja are clear.
- **2016, 2019, 2020, 2024:** checked, no fixes (cloud < 1%). SCL-only areas are snow and debris, as expected.
- **Review complete (2026-10-10): all 10 masks accepted as made by OmniCloudMask, with no hand fixes needed.**
  Cloud on glaciers to note for the area step: 2021 (upper Changri Nup/Shar, maybe Lhotse Shar), 2018 (upper Khumbu),
  2023 (tip of the Ngozumpa tongue at most).

### Next
- Then: labels (2022 first, then 2016 as an edited copy) and `tiles.csv`.

## 2026-10-07 – export complete

### Decisions (details in `decisions.md` 11–13)
- **2022 = 2022-10-24** (was 10-29). The old `S2_20221029.tif` was removed from `raw/`.
- **2025 added = 2025-11-22**. It overrides the skip of 2026-10-05, at the team's request; heavy
  snow all season, so it is the least-bad date.
- **Landsat cloud rule = QA bit 3 only.** Bit 4 is mostly topographic shadow here; it is still logged.

### Exported (`GlacierMonitor/raw/`)
- `sentinel2/`: 10 files: S2_20161030, S2_20171015, S2_20181020, S2_20191015, S2_20201009,
  S2_20211014, **S2_20221024**, S2_20231009, S2_20241117, S2_20251122.
- `dem/CopDEM_GLO30_UTM45N.tif` (from 2026-10-05).
- `landsat_thermal/`: 10 LST files (2016-11-19, 2017-10-21, 2018-11-09, 2019-11-12, 2020-10-29,
  2021-11-17, 2022-10-27, 2023-10-06, 2024-11-01, 2025-10-11), with AOI cloud 3.6–23.0%, plus `landsat_log.csv`.
  2022-10-19 was unreadable on Planetary Computer, so 2022-10-27 was used instead.
- `scene_log.csv` (the fresh_snow column is still empty and goes to Snehi).

### Checks
- `verify.py`: **ALL CHECKS PASSED**. All 10 S2 files are 3940 × 3220 on an identical grid with 7 bands,
  and the offset check 2020 vs 2023 gives median diffs within ±51 DN.

### Reference year
- **2022, file `S2_20221024.tif`** (labels are drawn on this scene).

### Still to do
- ~~False-colour check~~ done 2026-10-08 in Python (QGIS would not install): 2020 vs 2022 aligned (edges match at Imja, Gokyo, Khumbu), no cloud on main glaciers. Images: `previews/check_*_2020_vs_2022.png`. Minor: thin tile seam on the S shore of Imja Tsho in 2020.
- ~~Upload to Drive~~ done 2026-10-08 (`raw/` in the team folder: 10 S2, DEM, 10 LST, scene_log). Image-selection sheet updated 2026-10-08 (2022 → 10-24, 2025 row filled).
- Hand off: Siddhi gets `S2_20221024.tif` for alignment, Snehi gets `scene_log.csv`, and Kavya gets the reference file name.

### 2026-10-08 – 10
- PR #6 merged into `main` by Kavya on 2026-10-10.
- Siddhi's `S2GEE_2022.tif` checked against `S2_20221024.tif`. It has the same grid with a 0 px shift, no holes, and a median NDSI on ice of 0.89.
  It has 20 bands instead of 6, though, so a re-export with only B02–B12 has been requested.
- Reviewed Kavya's Siamese plan.
  - Fix needed: if only `|difference|` reaches the decoder, the model cannot tell loss from gain, so use the signed difference.
  - Fix needed: start the encoder from the 7-input `--no-terrain` U-Net, because the 9-input U-Net weights won't load.
  - Labels: draw the early-year label by editing a copy of the 2022 label.
  - Proposed early year: 2016 (`S2_20161030.tif`), pending Snehi's fresh-snow notes.

## 2026-10-05 (paused)

### Done
- **Picks confirmed** (`chosen_dates.json`). Same as suggested, except **2022 = 2022-10-29**.
  **2025 skipped**: no suitable date, and 2025-11-22 is too snow-covered for glacier area.
- **Sentinel-2 exported** for 2016–2024 (9 files, `GlacierMonitor/raw/sentinel2/S2_YYYYMMDD.tif`).
- **DEM exported**: `dem/CopDEM_GLO30_UTM45N.tif`, elevation 3272–8737 m. The minimum is the Dudh Koshi
  valley at the south edge, and the maximum is a slightly blunted Everest summit at 30 m.
- **Checks (`verify.py`) passed** for S2 and DEM: EPSG:32645, 10 m / 30 m, all 9 S2 files on the
  same 3940 × 3220 grid, 7 bands with names. **Offset fix confirmed**: on pixels clear in both 2020
  and 2023, the band medians differ by only −51 to +30 DN (≈ +1000 if the fix had failed).
- **Scripts pushed** to https://github.com/GlacierMonitor/OEPGlacierMonitor (`main`, first commit).
  Later fixes (export retries/resume, Landsat mosaic fix, new offset check) are **not pushed yet**.
- Disk use so far: about 982 MB.

### Landsat thermal – not finished, needs a decision
- **Bug fixed:** odc-stac filled the area outside WRS row 040 with 0, which overwrote row 041 and
  made every year look like <90% coverage. Scenes are now mosaicked one by one, and coverage is 100%.
- **Specified cloud rule (QA bits 3 + 4) rejects every year checked so far (2016–2021)**: 39–62% cloud +
  shadow, while the USGS scene estimate is 2–20%. Most of it is **bit 4 "cloud shadow"** (26–39%),
  which is really topographic shadow in steep terrain with a low autumn sun, plus some snow flagged as cloud.

  | Year | Clearest date | Bits 3+4 % | Cloud (bit 3) % | Shadow (bit 4) % | USGS scene % |
  |---|---|---|---|---|---|
  | 2016 | 2016-10-18 | 56.7 | 20.7 | 36.0 | 8.9 |
  | 2017 | 2017-10-21 | 38.6 | 12.5 | 26.1 | 19.8 |
  | 2018 | 2018-11-09 | 54.4 | 20.2 | 34.2 | 2.3 |
  | 2019 | 2019-10-27 | 55.2 | 25.1 | 30.2 | 11.4 |
  | 2020 | 2020-10-29 | 56.7 | 21.3 | 35.4 | 2.4 |
  | 2021 | 2021-11-17 | 61.8 | 23.0 | 38.7 | 4.1 |

- **2022:** the file `LC08_L2SP_140041_20221019` returns HTTP 403 on Planetary Computer (row 040
  of the same date opens fine). `export.py` now skips unreadable dates instead of crashing.
- **Open decision:** keep the rule (no LST files at all), or drop bit 4 / use another threshold.

### To resume
1. Decide the Landsat cloud rule.
2. `venv\Scripts\python export.py landsat log`: S2 and DEM are skipped because they already exist.
3. `venv\Scripts\python verify.py`
4. Commit and push the script fixes. Fill the Google Sheet from `GlacierMonitor/raw/scene_log.csv`.

## 2026-09-27

### Done
- **Environment:** `./venv` (Python 3.12.10) with pystac-client 0.9.0, planetary-computer 1.0.0,
  odc-stac 0.5.3, rioxarray 0.23.0, rasterio 1.5.1, pandas 3.0.6, matplotlib 3.11.2.
- **Scripts written:** `config.py`, `search.py`, `export.py`, `verify.py` (see `decisions.md`).
  `export.py` and `verify.py` are written but **not run yet**; they wait for confirmed picks.
- **Step 1 (scan), complete:** all 116 Oct–Nov dates for 2016–2025 scanned at 60 m, results in
  `GlacierMonitor/raw/s2_scan.csv` (both cloud metrics, coverage, snow hint, tiles, baselines).
- **Found a problem with the SCL cloud rule** (details in `decisions.md`): it rejects every
  date for 2016–2021 because snow and bright debris are labelled as cloud.
- **Step 2 (previews), complete:** 34 PNGs in `GlacierMonitor/raw/previews/`. These are the top 3 per
  year plus 4 November alternatives for 2024–2025.
  - 2016–2021: ranked by the snow-corrected metric (no cut-off), because the SCL rule gives no candidates.
  - 2022–2025: specified SCL rule (<20% cloud, >95% cover).
- **Every preview reviewed by eye.** The automatic suggestion was wrong for 2018, 2019, 2024 and 2025
  (see table). Reviewed picks are saved in `chosen_dates.suggested.json`.

### Step 3: candidates and suggested picks (awaiting team confirmation)
`SCL` = specified cloud rule, `ndsi` = snow-corrected cloud %, `snow` = SCL class-11 hint.
All coverage is 99.9–100%.

| Year | Top candidates: SCL % / ndsi % / snow % | Suggested pick | Why |
|---|---|---|---|
| 2016 | 10-10: 55.4 / 32.0 / 13.4 · 10-30: 75.7 / 48.0 / 1.3 · 11-29: 71.0 / 48.9 / 3.4 | **2016-10-30** | Clear. 10-10 has fresh snow on valley floors + cloud E |
| 2017 | 10-10: 43.2 / 29.2 / 14.6 · 10-15: 56.5 / 29.8 / 2.2 · 10-30: 72.1 / 38.1 / 3.4 | **2017-10-15** | Clear. 10-10 has cloud E |
| 2018 | 10-05: 59.0 / 37.0 / 4.0 · 10-20: 65.6 / 42.9 / 4.9 · 10-25: 65.5 / 43.9 / 3.6 | **2018-10-20** | Clear. 10-05 (auto pick) has fresh snow + cloud E |
| 2019 | 10-15: 40.9 / 17.1 / 13.1 · 10-05: 37.0 / 22.7 / 25.9 · 10-25: 89.7 / 32.3 / 3.4 | **2019-10-15** | Clear. 10-25 (auto pick) heavily clouded; 10-05 fresh snow |
| 2020 | 10-09: 51.1 / 25.5 / 3.5 · 10-19: 41.1 / 30.8 / 17.1 · 10-14: 58.0 / 37.6 / 8.7 | **2020-10-09** | Clear. 10-14 / 10-19 cloud in S |
| 2021 | 10-14: 35.7 / 16.4 / 12.2 · 10-09: 54.8 / 33.4 / 8.8 · 10-04: 65.2 / 41.0 / 12.4 | **2021-10-14** | Mostly clear, small cloud patches |
| 2022 | 10-24: 6.7 / 4.6 / 41.5 · 10-14: 6.8 / 4.7 / 52.4 · 10-29: 9.9 / 7.6 / 37.8 | **2022-10-24** | Clear |
| 2023 | 10-09: 6.9 / 6.2 / 37.5 · 10-29: 7.8 / 4.4 / 37.4 · 10-14: 9.8 / 8.9 / 34.6 | **2023-10-09** | Clear. 10-29 has scene-wide haze SCL missed |
| 2024 | 10-28: 3.6 / 2.4 / 44.9 · 10-08: 3.8 / 2.7 / 68.6 · 10-13: 7.7 / 6.0 / 57.4 | **2024-11-17** (3.9 / 2.7 / 35.4) | All Oct dates have fresh snow on debris tongues; 11-17 clean |
| 2025 | 10-13: 1.4 / 0.4 / 86.9 · 10-25: 3.3 / 1.8 / 71.4 · 10-18: 5.4 / 3.8 / 78.0 | **2025-11-22** (4.7 / 3.2 / 64.5) | Heavy snow all season; least-bad date. Likely fresh snow = Y |

### Next
1. Team confirms the cloud rule for 2016–2021 and the picks: copy `chosen_dates.suggested.json`
   to `chosen_dates.json` and edit it if needed.
2. Then run `export.py` (S2, DEM, Landsat LST, scene log) and `verify.py`.

### Not yet produced
- No GeoTIFFs exported and no `scene_log.csv`.
