"""Approach B: yearly Earth Engine composites on the approach-A grid (S2GEE_YYYY.tif, read by training/data.py).

    training\\venv\\Scripts\\python gee_export.py --project <cloud-project-id> --dry-run   # image counts per year only
    training\\venv\\Scripts\\python gee_export.py --project <cloud-project-id>             # start the 10 Drive exports

Recipe (paper methods, preprocessing B): every COPERNICUS/S2_HARMONIZED (top-of-atmosphere, L1C) image of 1 Oct - 30 Nov
that touches the study area; pixels with Cloud Score+ cs_cdf < 0.6 are masked; the per-pixel median of what is left.
TOA (not surface reflectance) on purpose: S2_SR_HARMONIZED starts in March 2017, so 2016 would be missing.
Output per year: 6 bands B02 B03 B04 B08 B11 B12 (uint16, reflectance x 10000, 0 = no clear image), EPSG:32645,
10 m, exactly the approach-A grid (S2_YYYYMMDD.tif: 3940 x 3220 px, top-left 460600 E 3110570 N), so the label,
masks and tiles line up pixel for pixel. 20 m bands (B11, B12) are resampled by Earth Engine's default nearest
neighbour, as in approach A. Exports run on Google's servers into the Drive folder --folder; image counts per year
are written to GlacierMonitor/s2gee_counts.csv (the paper reports them). Needs earthengine-api and a one-time
`earthengine authenticate`.
"""
import argparse
import csv
import time
from pathlib import Path

import ee

YEARS = range(2016, 2026)
SEASON = ("10-01", "11-30")                     # same as config.SEASON (config needs the STAC packages)
BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]          # Earth Engine names of B02 B03 B04 B08 B11 B12
CS_MIN = 0.6
WIDTH, HEIGHT = 3940, 3220
TRANSFORM = [10, 0, 460600, 0, -10, 3110570]             # same grid as S2_YYYYMMDD.tif
CRS = "EPSG:32645"


def region():
    x0, y1 = TRANSFORM[2], TRANSFORM[5]
    x1, y0 = x0 + WIDTH * TRANSFORM[0], y1 + HEIGHT * TRANSFORM[4]
    return ee.Geometry.Rectangle([x0, y0, x1, y1], proj=CRS, geodesic=False)


def composite(year, aoi):
    start, end = f"{year}-{SEASON[0]}", f"{year}-{SEASON[1]}"
    s2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(aoi).filterDate(start, ee.Date(end).advance(1, "day"))
    cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    clear = s2.linkCollection(cs, ["cs_cdf"]).map(lambda im: im.updateMask(im.select("cs_cdf").gte(CS_MIN)))
    # named B02..B12 so the GeoTIFF band descriptions match training/data.py S2_BANDS (Earth Engine says B2..B12)
    img = clear.select(BANDS).median().unmask(0).toUint16().rename(["B02", "B03", "B04", "B08", "B11", "B12"])
    return img, s2.size()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--project", required=True, help="Google Cloud project registered for Earth Engine")
    p.add_argument("--years", nargs="+", type=int, default=list(YEARS))
    p.add_argument("--folder", default="GlacierMonitor_S2GEE", help="Drive folder for the exports")
    p.add_argument("--dry-run", action="store_true", help="only count the images per year")
    a = p.parse_args()
    ee.Initialize(project=a.project)
    aoi = region()
    rows = []
    for year in a.years:
        img, n = composite(year, aoi)
        n = n.getInfo()
        rows.append({"year": year, "images": n})
        print(f"{year}: {n} S2_HARMONIZED images in {SEASON[0]}..{SEASON[1]}")
        if a.dry_run or n == 0:
            continue
        task = ee.batch.Export.image.toDrive(image=img, description=f"S2GEE_{year}", folder=a.folder,
                                             fileNamePrefix=f"S2GEE_{year}", crs=CRS, crsTransform=TRANSFORM,
                                             dimensions=f"{WIDTH}x{HEIGHT}", maxPixels=int(1e9), fileFormat="GeoTIFF")
        task.start()
        print(f"      export started: {task.id}")
        time.sleep(1)
    out = Path(__file__).resolve().parent / "GlacierMonitor" / "s2gee_counts.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, ["year", "images"]); w.writeheader(); w.writerows(rows)
    print(f"-> {out}" + ("" if a.dry_run else f"; follow the exports at https://code.earthengine.google.com/tasks"))


if __name__ == "__main__":
    main()
