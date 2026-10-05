# GlacierMonitor Part 1 – Progress updates

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
