"""Score glacier maps against the label, and collect the scores of many runs into one table.

Usage:
    # one map (from predict.py or rf_baseline.py):
    venv/Scripts/python score.py run --pred runs/x/S2_20221024_mask.tif --s2 S2_20221024.tif \
        --dem CopDEM_GLO30_UTM45N.tif --label label_2022.tif --tiles tiles.csv \
        --glaciers glacier_ids.tif --lookup glacier_ids.csv --out runs/x/scores.json
    # the NDSI baseline instead of a map: --ndsi 0.4 in place of --pred (add --masks as for the model)
    # every scores.json under runs/matrix into one table:
    venv/Scripts/python score.py collect runs/matrix --out runs/matrix/results

Per split (val, test) it reports IoU, precision, recall, F1 and area bias (predicted minus labelled glacier
area, % of labelled). Pixels the label marks 255, or the map marks no data, are not scored.

With --glaciers/--lookup (from glacier_ids.py) it also gives:
  - test_strict: the test squares without any glacier the model has partly seen, that is a glacier whose
    outline has at least --min-shared pixels and --min-shared-frac of its in-square area in train or val
    squares. Each pixel belongs to the nearest outline within --buffer pixels, so a tongue that has moved
    since the outline was drawn still counts as that glacier. The left-out glaciers are listed.
  - per-glacier IoU on the test squares, for glaciers with at least --min-pixels labelled glacier pixels there.
  - recall on clean ice (labelled glacier with NDSI > 0.4) and on debris-covered ice (NDSI <= 0.4). The split
    is made with NDSI, so the NDSI baseline scores 1 and 0 here by construction; it is for comparing models.

`collect` writes <out>_runs.csv (one row per run), <out>_summary.csv (mean, sd and n over seeds for each
setting) and <out>_glaciers.csv (per-glacier IoU, one row per run and glacier).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

from data import CHANNELS, IGNORE, TILE, load_stack, read_label, read_mask, read_tiles

SPLITS = ("val", "test")
HEADLINE = ["val_iou", "test_iou", "test_strict_iou", "test_strict_f1", "test_strict_area_bias_pct",
            "test_strict_clean_recall", "test_strict_debris_recall"]


def metrics(pred, y, region):
    """IoU, precision, recall, F1 and area bias over the scored pixels of region."""
    s = region & (y != IGNORE) & (pred != IGNORE)
    p, t = (pred == 1) & s, (y == 1) & s
    tp = int((p & t).sum()); fp = int((p & ~t).sum()); fn = int((~p & t).sum())

    def ratio(a, b):
        return round(a / b, 4) if b else None
    prec, rec = ratio(tp, tp + fp), ratio(tp, tp + fn)
    return {"iou": ratio(tp, tp + fp + fn), "precision": prec, "recall": rec,
            "f1": ratio(2 * tp, 2 * tp + fp + fn),
            "area_bias_pct": round(100 * (fp - fn) / (tp + fn), 2) if tp + fn else None,
            "scored_px": int(s.sum()), "glacier_px": tp + fn}


def tile_region(tiles, split, shape):
    region = np.zeros(shape, bool)
    for r, c in tiles[tiles.split == split][["row", "col"]].itertuples(index=False):
        region[r:r + TILE, c:c + TILE] = True
    return region


def glacier_zones(path, profile, buffer):
    """(ids, zones): the outline raster, and each pixel's nearest outline within buffer pixels (0 if none)."""
    with rasterio.open(path) as src:
        ids = src.read(1).astype("int32")
        same = src.crs == profile["crs"] and src.transform.almost_equals(profile["transform"])
    if ids.shape != (profile["height"], profile["width"]) or not same:
        raise ValueError(f"{path} is not on the same grid as the image")
    dist, (ri, ci) = ndimage.distance_transform_edt(ids == 0, return_indices=True)
    zones = ids[ri, ci]
    zones[dist > buffer] = 0
    return ids, zones


def run(args):
    if (args.pred is None) == (args.ndsi is None):
        raise SystemExit("give either --pred (a *_mask.tif) or --ndsi (threshold for the baseline)")
    x, valid, profile = load_stack(args.s2, args.dem, args.mask_cloud)
    if args.masks:
        valid &= ~read_mask(args.s2, args.masks, profile)
    ndsi = x[CHANNELS.index("NDSI")]
    del x
    y = read_label(args.label, valid, profile)
    if args.pred:
        with rasterio.open(args.pred) as src:
            pred = src.read(1)
            same = src.crs == profile["crs"] and src.transform.almost_equals(profile["transform"])
        if pred.shape != y.shape or not same:
            raise SystemExit(f"{args.pred} is not on the same grid as {args.s2}")
        pred = np.where(valid, pred, IGNORE).astype("uint8")
    else:
        pred = np.where(valid, ndsi > args.ndsi, IGNORE).astype("uint8")

    tiles = read_tiles(args.tiles, y.shape)
    regions = {s: tile_region(tiles, s, y.shape) for s in ("train",) + SPLITS}
    out = {"config": dict(kv.split("=", 1) for kv in args.tag), "pred": args.pred or f"NDSI>{args.ndsi}"}
    for s in SPLITS:
        if regions[s].any():
            out[s] = metrics(pred, y, regions[s])

    if args.glaciers:
        names = pd.read_csv(args.lookup, keep_default_na=False).set_index("value")
        name = lambda v: (str(names.glacier_name.get(v, "")) if v in names.index else "") or f"fid {v}"
        ids, zones = glacier_zones(args.glaciers, profile, args.buffer)
        seen = regions["train"] | regions["val"]
        excluded = []
        for v in np.unique(zones[regions["test"]]):
            if v == 0:
                continue
            g = ids == v                  # the outline itself: the buffer only assigns pixels, it does not share
            n_seen, n_test = int((g & seen).sum()), int((g & regions["test"]).sum())
            if n_seen >= args.min_shared and n_seen >= args.min_shared_frac * (n_seen + n_test):
                excluded.append({"value": int(v), "name": name(v), "px_train_val": n_seen, "px_test": n_test})
        strict = regions["test"] & ~np.isin(zones, [e["value"] for e in excluded])
        out["excluded_from_test_strict"] = excluded
        out["test_strict"] = metrics(pred, y, strict)
        clean = ndsi > 0.4
        for kind, part in (("clean", clean), ("debris", ~clean)):
            # recall only: map pixels outside the labelled glacier cannot be called clean or debris ice
            m = metrics(pred, np.where(part, y, IGNORE), strict)
            out["test_strict"][f"{kind}_recall"] = m["recall"]
            out["test_strict"][f"{kind}_glacier_px"] = m["glacier_px"]
        per = []
        for v in np.unique(zones[regions["test"]]):
            m = metrics(pred, y, regions["test"] & (zones == v))
            if v and m["glacier_px"] >= args.min_pixels:
                per.append({"value": int(v), "name": name(v),
                            "debris": str(names.debris.get(v, "")) if "debris" in names else "",
                            "shared": int(v) in {e["value"] for e in excluded}, **m})
        out["per_glacier"] = per

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    for s in ("val", "test", "test_strict"):
        if s in out:
            m = out[s]
            extra = (f", clean ice recall {m['clean_recall']}, debris recall {m['debris_recall']}"
                     if "clean_recall" in m else "")
            print(f"{s:12s} IoU {m['iou']}  F1 {m['f1']}  area bias {m['area_bias_pct']}%{extra}")
    if args.glaciers:
        left = ", ".join(e["name"] for e in out["excluded_from_test_strict"]) or "none"
        print(f"left out of test_strict (also in train/val squares): {left}")
    print("->", args.out)


def collect(args):
    rows, glaciers = [], []
    for f in sorted(p for d in args.dirs for p in Path(d).rglob("scores.json")):
        s = json.loads(f.read_text())
        if "config" not in s:
            continue                      # rf_baseline.py's own scores.json, not a score.py one
        row = {**s["config"], "run": f.parent.as_posix()}
        for split in ("val", "test", "test_strict"):
            for k, v in s.get(split, {}).items():
                row[f"{split}_{k}"] = v
        row["excluded"] = "; ".join(e["name"] for e in s.get("excluded_from_test_strict", []))
        rows.append(row)
        glaciers += [{**s["config"], "run": row["run"], **g} for g in s.get("per_glacier", [])]
    if not rows:
        raise SystemExit("no score.py scores.json found")
    runs = pd.DataFrame(rows)
    keys = [c for c in runs.columns if c in rows[0] and c not in ("run", "seed", "excluded")
            and not any(c.startswith(p) for p in ("val_", "test_"))]
    metric_cols = [c for c in HEADLINE if c in runs]
    summary = runs.groupby(keys, dropna=False)[metric_cols].agg(["mean", "std", "count"])
    summary.columns = [f"{m}_{a}".replace("_count", "_n") for m, a in summary.columns]
    summary = summary.round(4).reset_index()

    base = Path(args.out)
    base.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(f"{base}_runs.csv", index=False)
    summary.to_csv(f"{base}_summary.csv", index=False)
    if glaciers:
        pd.DataFrame(glaciers).to_csv(f"{base}_glaciers.csv", index=False)
    show = summary[keys + [c for c in summary.columns if c.endswith("_mean") or c == "test_iou_n"]]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(show.to_string(index=False))
    print(f"-> {base}_runs.csv, {base}_summary.csv" + (f", {base}_glaciers.csv" if glaciers else ""))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="score one map")
    r.add_argument("--pred", help="*_mask.tif from predict.py or rf_baseline.py")
    r.add_argument("--ndsi", type=float, help="score the NDSI > threshold baseline instead of --pred")
    r.add_argument("--s2", required=True, help="the S2 scene the label belongs to")
    r.add_argument("--dem", required=True)
    r.add_argument("--masks", help="Part 2 masks folder; use the same setting as the run being scored")
    r.add_argument("--mask-cloud", action="store_true", help="SCL cloud masking, as in train.py")
    r.add_argument("--label", required=True); r.add_argument("--tiles", required=True)
    r.add_argument("--glaciers", help="glacier_ids.tif (glacier_ids.py)")
    r.add_argument("--lookup", help="glacier_ids.csv")
    r.add_argument("--buffer", type=int, default=30, help="pixels around an outline that still belong to it")
    r.add_argument("--min-shared", type=int, default=2000,
                   help="outline pixels in train/val squares for a glacier to count as seen")
    r.add_argument("--min-shared-frac", type=float, default=0.1,
                   help="share of the outline's in-square pixels that must lie in train/val squares")
    r.add_argument("--min-pixels", type=int, default=1000, help="labelled glacier pixels for a per-glacier score")
    r.add_argument("--tag", nargs="*", default=[], help="key=value settings stored with the scores")
    r.add_argument("--out", required=True, help="scores.json to write")
    c = sub.add_parser("collect", help="collect scores.json files into tables")
    c.add_argument("dirs", nargs="+")
    c.add_argument("--out", required=True, help="output path prefix")
    args = p.parse_args()
    if args.cmd == "run" and bool(args.glaciers) != bool(args.lookup):
        raise SystemExit("--glaciers and --lookup go together")
    (run if args.cmd == "run" else collect)(args)


if __name__ == "__main__":
    main()
