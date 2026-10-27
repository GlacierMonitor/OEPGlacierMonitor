"""Glacier area per glacier per year from predicted masks, written to the shared results.csv.

Usage:
    venv/Scripts/python glacier_areas.py --masks predictions/S2_*_mask.tif \
        --glaciers glacier_ids.tif --lookup glacier_ids.csv --method unet --results results.csv

Inputs:
  - masks from predict.py: <stem>_mask.tif, 1 glacier / 0 not / 255 no data. The year comes from the
    stem (S2_YYYYMMDD or S2GEE_YYYY), and so does the preprocessing: S2_ = A, S2GEE_ = B.
  - glacier_ids.tif on the same 10 m grid: one integer per glacier, 0 elsewhere.
  - glacier_ids.csv with columns value, glacier_id, glacier_name (value = the integer in the raster).

Glacier pixels inside each outline are counted, so area outside the outline is not counted. If more than
--max-nodata % of an outline has no data, its area_km2 is left empty rather than undercounted. Rows already
in results.csv with the same glacier_id, year, preprocessing and method are replaced; all others are kept.
"""
import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

COLUMNS = ["glacier_id", "glacier_name", "year", "preprocessing", "method", "area_km2"]
KEY = ["glacier_id", "year", "preprocessing", "method"]
STEM = re.compile(r"^(S2GEE_(\d{4})|S2_(\d{4})\d{4})_mask$")


def scene_info(path):
    m = STEM.match(Path(path).stem)
    if not m:
        raise ValueError(f"{path}: expected S2_YYYYMMDD_mask.tif or S2GEE_YYYY_mask.tif")
    return (int(m.group(2)), "B") if m.group(2) else (int(m.group(3)), "A")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--masks", nargs="+", required=True, help="one or more *_mask.tif files (wildcards ok)")
    p.add_argument("--glaciers", required=True, help="glacier-ID GeoTIFF on the same grid")
    p.add_argument("--lookup", required=True, help="CSV with value, glacier_id, glacier_name")
    p.add_argument("--method", required=True, help="method name written to results.csv, e.g. unet")
    p.add_argument("--preprocessing", help="override the A/B read from the file names")
    p.add_argument("--results", required=True, help="shared results.csv (created if missing)")
    p.add_argument("--max-nodata", type=float, default=5.0,
                   help="leave area_km2 empty when more than this %% of a glacier's pixels have no data")
    args = p.parse_args()

    files = sorted({f for pat in args.masks for f in (glob.glob(pat) or [pat])})
    lookup = pd.read_csv(args.lookup)
    missing = {"value", "glacier_id", "glacier_name"} - set(lookup.columns)
    if missing:
        raise ValueError(f"{args.lookup} is missing columns {missing}")
    if lookup.value.duplicated().any() or (lookup.value <= 0).any():
        raise ValueError(f"{args.lookup}: values must be unique and > 0 (0 means no glacier)")

    with rasterio.open(args.glaciers) as src:
        ids = src.read(1).astype("int64")
        crs, transform = src.crs, src.transform
    unknown = set(np.unique(ids).tolist()) - {0} - set(lookup.value.tolist())
    if unknown:
        raise ValueError(f"{args.glaciers} has values not in {args.lookup}: {sorted(unknown)[:10]}")
    n = int(lookup.value.max()) + 1
    pixels = np.bincount(ids.ravel(), minlength=n)
    pixel_km2 = abs(transform.a * transform.e) / 1e6

    rows = []
    for f in files:
        year, prep = scene_info(f)
        prep = args.preprocessing or prep
        with rasterio.open(f) as src:
            mask = src.read(1)
            if mask.shape != ids.shape or src.crs != crs or not src.transform.almost_equals(transform):
                raise ValueError(f"{f} is not on the same grid as {args.glaciers}")
        glacier = np.bincount(ids.ravel(), weights=(mask == 1).ravel(), minlength=n)
        nodata = np.bincount(ids.ravel(), weights=(mask == 255).ravel(), minlength=n)
        gaps = []
        for v, gid, name in lookup[["value", "glacier_id", "glacier_name"]].itertuples(index=False):
            gap = 100 * nodata[v] / pixels[v] if pixels[v] else 0.0
            area = round(glacier[v] * pixel_km2, 4) if gap <= args.max_nodata else None
            rows.append({"glacier_id": gid, "glacier_name": name, "year": year, "preprocessing": prep,
                         "method": args.method, "area_km2": area})
            if area is None:
                gaps.append(f"{name or gid} {gap:.0f}%")
        total = glacier[1:].sum() * pixel_km2
        print(f"{Path(f).stem}: {year} {prep}, {len(lookup)} glaciers, {total:.2f} km2 inside outlines")
        if gaps:
            print(f"  area left empty, too much no data: {', '.join(gaps)}")

    new = pd.DataFrame(rows, columns=COLUMNS)
    if new.duplicated(KEY).any():
        raise ValueError("two masks give the same year and preprocessing; pass one scene per year")
    path = Path(args.results)
    if path.exists():
        old = pd.read_csv(path)
        if list(old.columns) != COLUMNS:
            raise ValueError(f"{path} has columns {list(old.columns)}, expected {COLUMNS}")
        old = old.merge(new[KEY], on=KEY, how="left", indicator=True)
        old = old[old._merge == "left_only"][COLUMNS]
        new = pd.concat([old, new], ignore_index=True)
    new.sort_values(["method", "preprocessing", "glacier_id", "year"]).to_csv(path, index=False)
    print(f"-> {path} ({len(rows)} rows written, {len(new)} in total)")


if __name__ == "__main__":
    main()
