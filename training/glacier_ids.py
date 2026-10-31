"""Glacier-ID raster from the inventory GeoPackage, for per-glacier areas and per-glacier test scores.

Usage:
    venv/Scripts/python glacier_ids.py --gpkg glaciers.gpkg --s2 S2_20221024.tif --out glacier_ids.tif

Writes glacier_ids.tif (uint16 on the S2 10 m grid: the outline's fid, 0 outside every outline) and
glacier_ids.csv next to it (value, glacier_id, glacier_name, debris). glacier_areas.py and score.py read both.
Where outlines overlap, the smaller glacier keeps the pixel, so a small glacier is not swallowed by a big one.
"""
import argparse
import csv
import sqlite3
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely import wkb


def gpkg_geom(blob):
    """Shapely geometry from a GeoPackage geometry blob (header + WKB)."""
    envelope = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[(blob[3] >> 1) & 7]
    return wkb.loads(bytes(blob[8 + envelope:]))


def read_outlines(path, layer):
    con = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
    try:
        cols = [r[1] for r in con.execute(f'pragma table_info("{layer}")')]
        want = [c for c in ("glacier_id", "glacier_name", "debris") if c in cols]
        rows = con.execute(f'select fid, geom, {", ".join(want) or "null"} from "{layer}" where geom is not null')
        return [{"value": fid, "geom": gpkg_geom(g), **dict(zip(want, rest))} for fid, g, *rest in rows]
    finally:
        con.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--gpkg", required=True, help="inventory GeoPackage (glaciers layer, EPSG:32645)")
    p.add_argument("--layer", default="glaciers")
    p.add_argument("--s2", required=True, help="any S2 file on the target 10 m grid")
    p.add_argument("--out", required=True, help="output glacier_ids.tif (the CSV is written next to it)")
    args = p.parse_args()

    outlines = read_outlines(args.gpkg, args.layer)
    if not outlines:
        raise SystemExit(f"{args.gpkg}: layer '{args.layer}' has no outlines")
    if max(o["value"] for o in outlines) > 65535:
        raise SystemExit("fid above 65535 does not fit in uint16")
    with rasterio.open(args.s2) as src:
        profile = src.profile
    shape = (profile["height"], profile["width"])

    # rasterize draws in order, so big first and small last: the smaller glacier wins overlaps
    order = sorted(outlines, key=lambda o: -o["geom"].area)
    ids = rasterize(((o["geom"], o["value"]) for o in order), out_shape=shape,
                    transform=profile["transform"], fill=0, dtype="uint16")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prof = dict(profile, count=1, dtype="uint16", nodata=0, compress="deflate", predictor=2)
    with rasterio.open(out, "w", **prof) as dst:
        dst.write(ids[None])
    counts = np.bincount(ids.ravel(), minlength=max(o["value"] for o in outlines) + 1)
    with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["value", "glacier_id", "glacier_name", "debris", "pixels"])
        for o in sorted(outlines, key=lambda o: o["value"]):
            w.writerow([o["value"], o.get("glacier_id") or "", o.get("glacier_name") or "",
                        o.get("debris") or "", int(counts[o["value"]])])
    on_grid = sum(counts[o["value"]] > 0 for o in outlines)
    print(f"{len(outlines)} outlines, {on_grid} on the grid -> {out}, {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
