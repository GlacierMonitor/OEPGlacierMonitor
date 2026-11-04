"""Two-person label agreement: the first drawing against a blind redraw of the same squares.

Usage (after the break, once the redraw files are back):
    venv/Scripts/python agreement.py --first labels_2022_kavya.gpkg --redraw labels_2022_redraw_siddhi.gpkg \
        --s2 S2_20221024.tif --glaciers glacier_ids.csv --out agreement_siddhi

Both files are make_labels.py drawings (layers glaciers, unsure, squares). The squares compared are the ones
in the redraw file's squares layer. Each drawing is rasterised on the S2 10 m grid inside each square
(glacier 1, not glacier 0); pixels that either person put in 'unsure' are left out of both, so the numbers
measure where both people committed to an answer.

Per square: glacier area by each person, the difference, IoU, and the boundary distance (for every edge
pixel of one drawing, the distance to the nearest edge pixel of the other, both ways; mean and 95th
percentile in metres). Per study glacier (with --glaciers, a glacier_ids.csv with glacier_id and
glacier_name): the area each person gave that glacier inside the squares, matched by the outlines'
glacier_id, so it is the label term of the per-glacier error budget.

Label sd: if two independent drawings differ by d, one drawing's error sd is |d| / sqrt(2). Over several
squares the pooled value is sqrt(mean(d^2) / 2). It is reported in km2 and as % of the mean area.

Writes <out>_squares.csv, <out>_glaciers.csv (with --glaciers) and prints a summary.
"""
import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from scipy import ndimage
from shapely import wkb

TILE = 256


def gpkg_geom(blob):
    """Shapely geometry from a GeoPackage geometry blob (header + WKB)."""
    envelope = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[(blob[3] >> 1) & 7]
    return wkb.loads(bytes(blob[8 + envelope:]))


def read_layer(path, layer, cols=()):
    con = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
    try:
        have = {r[1] for r in con.execute(f'pragma table_info("{layer}")')}
        if not have:
            return []
        sel = ", ".join(["geom"] + [c if c in have else "NULL" for c in cols])
        return [(gpkg_geom(r[0]), *r[1:]) for r in con.execute(f'select {sel} from "{layer}" where geom is not null')]
    finally:
        con.close()


def burn(shapes, shape, transform, dtype="uint8"):
    shapes = [(g, v) for g, v in shapes if not g.is_empty]
    if not shapes:
        return np.zeros(shape, dtype)
    return rasterize(shapes, out_shape=shape, transform=transform, fill=0, dtype=dtype)


def edge(mask):
    return mask & ~ndimage.binary_erosion(mask, border_value=1)


def boundary_distances(a, b, px_m):
    """Distances (m) from each edge pixel of a to the nearest edge pixel of b, and back."""
    ea, eb = edge(a), edge(b)
    if not ea.any() or not eb.any():
        return np.array([])
    da = ndimage.distance_transform_edt(~eb)[ea]
    db = ndimage.distance_transform_edt(~ea)[eb]
    return np.concatenate([da, db]) * px_m


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--first", required=True, help="the first drawing (labels_<year>_<labeller>.gpkg)")
    p.add_argument("--redraw", required=True, help="the blind redraw (labels_<year>_redraw_<who>.gpkg)")
    p.add_argument("--s2", required=True, help="the year's S2 image (only its grid is used)")
    p.add_argument("--glaciers", help="glacier_ids.csv (glacier_id, glacier_name) for per-glacier areas")
    p.add_argument("--out", required=True, help="output path prefix")
    args = p.parse_args()

    with rasterio.open(args.s2) as src:
        transform, crs = src.transform, src.crs
    px_m = abs(transform.a)
    px_km2 = abs(transform.a * transform.e) / 1e6

    squares = read_layer(args.redraw, "squares", ("row", "col", "name", "split"))
    if not squares:
        raise SystemExit(f"{args.redraw} has no squares layer")
    first_sq = {r[3] for r in read_layer(args.first, "squares", ("row", "col", "name"))}
    missing = [s[3] for s in squares if s[3] not in first_sq]
    if missing:
        print(f"  note: squares not in {Path(args.first).name}'s own squares: {', '.join(missing)}")

    study = {}
    if args.glaciers:
        lk = pd.read_csv(args.glaciers, keep_default_na=False)
        study = dict(zip(lk.glacier_id, lk.glacier_name))
    code = {gid: i for i, gid in enumerate(study, start=1)}

    drawings = {}
    for who, path in (("first", args.first), ("redraw", args.redraw)):
        drawings[who] = {"glaciers": read_layer(path, "glaciers", ("glacier_id",)),
                         "unsure": read_layer(path, "unsure")}

    sq_rows, gl_rows = [], []
    for geom, row, col, name, split in squares:
        win = Window(int(col), int(row), TILE, TILE)
        wt = rasterio.windows.transform(win, transform)
        shape = (TILE, TILE)
        ice, ids, unsure = {}, {}, np.zeros(shape, bool)
        for who, d in drawings.items():
            ice[who] = burn([(g, 1) for g, _ in d["glaciers"]], shape, wt).astype(bool)
            ids[who] = burn([(g, code[gid]) for g, gid in d["glaciers"] if gid in code], shape, wt, "uint16")
            unsure |= burn([(g[0], 1) for g in d["unsure"]], shape, wt).astype(bool)
        ok = ~unsure
        a, b = ice["first"] & ok, ice["redraw"] & ok
        inter, union = (a & b).sum(), (a | b).sum()
        dist = boundary_distances(a, b, px_m)
        sq_rows.append({"square": name, "split": split, "unsure_km2": unsure.sum() * px_km2,
                        "first_km2": a.sum() * px_km2, "redraw_km2": b.sum() * px_km2,
                        "diff_km2": (b.sum() - a.sum()) * px_km2,
                        "iou": inter / union if union else np.nan,
                        "disagree_km2": (a ^ b).sum() * px_km2,
                        "edge_mean_m": dist.mean() if dist.size else np.nan,
                        "edge_p95_m": np.percentile(dist, 95) if dist.size else np.nan})
        for gid, v in code.items():
            fa, fb = ((ids["first"] == v) & ok).sum(), ((ids["redraw"] == v) & ok).sum()
            if fa or fb:
                gl_rows.append({"square": name, "glacier": study[gid], "first_km2": fa * px_km2,
                                "redraw_km2": fb * px_km2})

    sq = pd.DataFrame(sq_rows)
    base = Path(args.out)
    base.parent.mkdir(parents=True, exist_ok=True)
    sq.round(5).to_csv(f"{base}_squares.csv", index=False)

    def label_sd(first, redraw):
        d = np.asarray(redraw) - np.asarray(first)
        sd = float(np.sqrt(np.mean(d ** 2) / 2)) if len(d) else np.nan
        mean = float(np.mean((np.asarray(first) + np.asarray(redraw)) / 2)) if len(d) else np.nan
        return sd, 100 * sd / mean if mean else np.nan

    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(sq.to_string(index=False))
    tot_a, tot_b = sq.first_km2.sum(), sq.redraw_km2.sum()
    sd, pct = label_sd(sq.first_km2, sq.redraw_km2)
    dis = sq.disagree_km2.sum()
    inter = (tot_a + tot_b - dis) / 2          # |a & b| = (|a| + |b| - |a ^ b|) / 2
    if tot_a:
        print(f"\nall {len(sq)} squares: first {tot_a:.3f} km2, redraw {tot_b:.3f} km2 "
              f"(difference {100 * (tot_b - tot_a) / tot_a:+.1f} %), IoU {inter / (inter + dis):.3f}")
    else:
        print("\nno glacier in the squares")
    print(f"label sd per square: {sd:.3f} km2 ({pct:.1f} % of a square's glacier area); "
          f"median edge distance {sq.edge_mean_m.median():.1f} m, median p95 {sq.edge_p95_m.median():.1f} m")
    print(f"unsure (either person): {sq.unsure_km2.sum():.3f} km2 left out")

    outs = [f"{base}_squares.csv"]
    if gl_rows:
        gl = pd.DataFrame(gl_rows).groupby("glacier")[["first_km2", "redraw_km2"]].sum().reset_index()
        gl["diff_km2"] = gl.redraw_km2 - gl.first_km2
        gl["diff_pct"] = 100 * gl.diff_km2 / ((gl.first_km2 + gl.redraw_km2) / 2)
        gl.round(5).to_csv(f"{base}_glaciers.csv", index=False)
        outs.append(f"{base}_glaciers.csv")
        print("\nper study glacier, inside the redrawn squares:")
        with pd.option_context("display.float_format", "{:.3f}".format):
            print(gl.to_string(index=False))
    print("-> " + ", ".join(outs))


if __name__ == "__main__":
    main()
