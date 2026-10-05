# GlacierMonitor Part 1 – Decisions

Record of choices made while running the data-collection step locally, with reasons.
Open items still need a decision from the team (see bottom).

## Settled

### 1. Run as local scripts, not Colab
- Colab / Google Drive code removed. Everything runs from `./venv` (Python 3.12).
- Outputs go to `./GlacierMonitor/raw/{sentinel2,dem,landsat_thermal,previews}`.
- Code split into reusable scripts:
  - `config.py`: fixed settings (bbox, CRS, bands, season, paths) and shared helpers
  - `search.py`: steps 1–3 (scan dates, candidate table, previews, suggested picks)
  - `export.py`: steps 4–7 (S2 export, DEM, Landsat LST, scene log)
  - `verify.py`: final checks (CRS, resolution, identical grid, bands, offset, disk use)

### 2. One fixed output grid instead of `bbox=` on every load
- The notebook passed `bbox=` + `resolution=` on each load. `config.geobox()` now builds one
  explicit grid in EPSG:32645 from the bbox, and every load uses it.
- **Why:** it guarantees every S2 file has identical width, height and transform (a required check).
- 10 m grid: 3940 × 3220 px, origin (460600, 3110570).

### 3. Cloud / cover / snow % measured strictly inside the lon/lat bbox
- The bbox is reprojected to UTM, and pixels outside the original lon/lat rectangle are masked out
  before computing percentages.

### 4. BOA offset removed per processing baseline, not per date
- Some dates mix baselines on Planetary Computer. For example, 2022–2023 have both 04.00/05.09 and
  reprocessed 05.10 copies of the same tiles, and 2016-11-29 mixes 02.12 and 03.00.
- `config.load_s2_date()` groups a date's tiles by baseline, subtracts 1000 only from tiles with
  baseline >= 04.00 (clipped to a minimum of 1, 0 stays nodata, SCL untouched), then mosaics.
- **Why:** the notebook subtracted 1000 from the whole mosaic if *any* tile was >= 04.00, which
  would wrongly shift old-baseline tiles.
- Previews use the same corrected data, so brightness is comparable across years.

### 5. Resampling
- S2 bands and SCL: nearest neighbour (keeps original DNs and SCL classes).
- DEM: bilinear (continuous surface).
- Landsat thermal: nearest.

### 6. Output format
- S2: 7-band uint16 (B02, B03, B04, B08, B11, B12, SCL), band names stored as band descriptions,
  nodata 0, deflate + predictor, 512 px tiles.
- DEM: float32, nodata −32767. LST: float32 Kelvin, nodata NaN.
- Written with rasterio directly (full control over descriptions, nodata, compression).

### 7. Landsat valid-pixel test hardened
- Valid = fill bit 0 not set **and** qa_pixel ≠ 0, so that areas outside a scene's footprint can
  never count as valid. Cloud = bit 3 (cloud) or bit 4 (cloud shadow), as specified.
- The chosen Landsat scene per year is logged to `landsat_thermal/landsat_log.csv`.

### 8. Picks are confirmed through a file
- `search.py` writes `chosen_dates.suggested.json`. After review, the team copies it to
  `chosen_dates.json` and edits it. `export.py` refuses to export S2 without `chosen_dates.json`.
- Notes per year can be added: `"2017": {"date": "2017-10-15", "notes": "haze SE"}`.
- `scene_log.csv` keeps any `fresh_snow` values already typed in when it is regenerated.

## Finding that affects the method

### SCL cloud classes are unreliable here before 2022
- On baselines 02.xx / 03.00 (2016–2021), Sen2Cor labels most **high-altitude snow/ice** and much
  **bright debris/rock** (e.g. the Ngozumpa tongue) as class 8/9 "cloud".
- Verified by eye: **2016-11-19** (SCL cloud 75.7%), **2017-10-15** (56.5%), **2020-10-24** (68.6%)
  and **2020-11-28** (78.2%) are all essentially cloud-free in both true colour and SWIR.
- As a result, **no date in 2016–2021 passes the specified rule**, not even the relaxed one
  (<30% cloud, >90% cover).
- From baseline 04.00 (from mid-Oct 2022) Sen2Cor behaves: clear dates score 3–10% cloud and snow is
  put in class 11. So **SCL cloud % and snow % are not comparable between pre-2022 and 2022+.**
- For the same reason, the `snow_%_hint` (class 11) is heavily underestimated for 2016–2021.

### Extra metric added (the specified rule is kept)
- `search.py` records both metrics for every date in `GlacierMonitor/raw/s2_scan.csv`:
  - `cloud_%`: the specified rule (SCL 3, 8, 9, 10). **Default for filtering.**
  - `cloud_ndsi_%`: the same, but SCL "cloud" pixels with NDSI = (B03−B11)/(B03+B11) > 0.4 are
    treated as snow. Chosen with `--metric scl_ndsi`.
- This correction removes the snow part of the false cloud but not the bright-debris part. Clear
  pre-2022 dates still score about 25–35%. It is useful for **ranking dates within a year**, not as
  an absolute threshold.
- A blue-band test (B02 > threshold) was tried and **rejected**: it did not separate a clear date
  (2017-10-15) from a partly cloudy one (2017-11-29).

### 9. "Big snow jump" is measured against the year's median
- A date counts as a likely fresh-snow date if its `snow_%_hint` is more than 10 points above the
  **median** of that year's candidates, and it is then skipped for the suggested pick.
- **Why:** the first version compared against the year's *minimum*. Snow cover naturally shrinks
  through Oct–Nov, so that rule pushed 2024 and 2025 to late-November picks. The median
  still catches real spikes (e.g. 2024-10-08 at 68.6% vs a median of about 41%).
- This is only the *suggestion*. The final call is made by eye on the SWIR preview.
- In practice the automatic suggestion was wrong for 4 of 10 years. Snow hidden under cloud makes
  cloudy dates look snow-free (2019-10-25), and pre-2022 SCL misses fresh snow (2018-10-05).
  All picks in `chosen_dates.suggested.json` were therefore **reviewed by eye**
  (see `updates.md`).

### 10. 2025 has unusually high snow cover all season
- SCL snow hint is 64–87% from 8 Oct 2025 onwards (2022–2024: about 30–50%). This is consistent
  with a heavy early-October snowfall. Whichever 2025 date is picked, it will probably
  show more snow than other years, so note it in `scene_log.csv`.

## Open – needs a team decision
1. **Cloud rule for 2016–2021.** Options:
   - (a) Rank by `cloud_ndsi_%` with no cut-off, and choose each year's date **by eye** from the
     previews (recommended).
   - (b) Keep the SCL rule and accept no candidates for 2016–2021 (not workable).
   - (c) Bring in a separate cloud mask (e.g. s2cloudless). This is more work and isn't on Planetary
     Computer.
2. **Fresh-snow judgement for 2016–2021** must come from the SWIR previews, not from `snow_%_hint`.
3. **Final pick per year**. Nothing is exported until `chosen_dates.json` is confirmed.
