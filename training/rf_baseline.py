"""Random Forest baseline: per-pixel classifier on the same 9 inputs as the U-Net.

Usage:
    venv/Scripts/python rf_baseline.py --s2 S2_20221024.tif --dem CopDEM_GLO30_UTM45N.tif \
        --label label_2022.tif --tiles tiles.csv --out runs/rf2022
    # also map other years with the same forest:
    venv/Scripts/python rf_baseline.py ... --predict raw/sentinel2/S2_*.tif

Samples labelled pixels from the train tiles only (up to --samples per class), trains scikit-learn's
RandomForestClassifier, maps the reference scene (and any --predict scenes) and scores the val and test
tiles against the label, next to the NDSI > 0.4 baseline. Writes to --out: rf.joblib, scores.json and
<stem>_prob.tif / <stem>_mask.tif in the same format as predict.py, so glacier_areas.py reads them too.
"""
import argparse
import glob
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from data import CHANNELS, IGNORE, TILE, load_stack, read_label, read_tiles
from model import confusion, iou
from predict import PIXEL_KM2, write


def tile_region(tiles, split, shape):
    region = np.zeros(shape, bool)
    for r, c in tiles[tiles.split == split][["row", "col"]].itertuples(index=False):
        region[r:r + TILE, c:c + TILE] = True
    return region


def sample(x, y, region, per_class, rng):
    """Balanced random sample of labelled pixels inside region: (features, labels)."""
    picks = []
    for cls in (0, 1):
        idx = np.flatnonzero(region.ravel() & (y.ravel() == cls))
        if len(idx) == 0:
            raise ValueError(f"train tiles have no labelled pixels of class {cls}")
        picks.append(rng.choice(idx, min(per_class, len(idx)), replace=False))
    idx = np.concatenate(picks)
    return x.reshape(len(x), -1)[:, idx].T, y.ravel()[idx]


def predict_scene(rf, x, valid, chunk=500_000):
    """Glacier probability (H, W) float32; only valid pixels are classified."""
    flat = x.reshape(len(x), -1)
    idx = np.flatnonzero(valid.ravel())
    prob = np.zeros(valid.size, "float32")
    for i in range(0, len(idx), chunk):
        part = idx[i:i + chunk]
        prob[part] = rf.predict_proba(flat[:, part].T)[:, 1]
    return prob.reshape(valid.shape)


def to_rasters(prob, valid):
    """(mask, prob8) in predict.py's format: 1/0/255 and 0-100/255."""
    mask = (prob >= 0.5).astype("uint8"); mask[~valid] = 255
    prob8 = np.round(prob * 100).astype("uint8"); prob8[~valid] = 255
    return mask, prob8


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--s2", required=True); p.add_argument("--dem", required=True)
    p.add_argument("--label", required=True); p.add_argument("--tiles", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--predict", nargs="*", default=[], help="other S2 files to map (wildcards ok)")
    p.add_argument("--samples", type=int, default=50_000, help="training pixels per class")
    p.add_argument("--trees", type=int, default=200)
    p.add_argument("--min-leaf", type=int, default=5, help="min_samples_leaf (limits overfitting)")
    p.add_argument("--mask-cloud", action="store_true", help="ignore SCL cloud pixels (only reliable 2022+)")
    p.add_argument("--ndsi", type=float, default=0.4, help="NDSI threshold for the baseline comparison")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    x, valid, profile = load_stack(args.s2, args.dem, args.mask_cloud)
    y = read_label(args.label, valid, profile)
    tiles = read_tiles(args.tiles, valid.shape)
    train = tile_region(tiles, "train", valid.shape)
    if not train.any():
        raise SystemExit("tiles.csv has no 'train' tiles")

    fx, fy = sample(x, y, train, args.samples, rng)
    print(f"training on {len(fy)} pixels ({(fy == 1).sum()} glacier, {(fy == 0).sum()} not) "
          f"from {(tiles.split == 'train').sum()} train tiles")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=args.trees, min_samples_leaf=args.min_leaf, n_jobs=-1,
                                random_state=args.seed)
    rf.fit(fx, fy)
    print(f"fitted {args.trees} trees in {time.time() - t0:.0f}s")
    joblib.dump({"rf": rf, "channels": CHANNELS, "mask_cloud": args.mask_cloud, "s2": str(args.s2)},
                out / "rf.joblib", compress=3)

    t0 = time.time()
    prob = predict_scene(rf, x, valid)
    stem = Path(args.s2).stem
    mask, prob8 = to_rasters(prob, valid)
    write(out / f"{stem}_prob.tif", prob8, profile)
    write(out / f"{stem}_mask.tif", mask, profile)
    print(f"{stem}: glacier {(mask == 1).sum() * PIXEL_KM2:.2f} km2, mapped in {time.time() - t0:.0f}s")

    scores = {"importance": dict(zip(CHANNELS, np.round(rf.feature_importances_, 4).tolist()))}
    ndsi = x[CHANNELS.index("NDSI")]
    for split in ("val", "test"):
        region = tile_region(tiles, split, valid.shape)
        if not region.any():
            continue
        target = np.where(region, y, IGNORE)
        rf_iou = iou(*confusion(mask == 1, target))
        nd_iou = iou(*confusion((ndsi > args.ndsi) & valid, target))
        scores[split] = {"iou_rf": round(rf_iou, 4), f"iou_ndsi_{args.ndsi}": round(nd_iou, 4)}
        print(f"  IoU on '{split}' tiles: RF {rf_iou:.4f} | NDSI>{args.ndsi} {nd_iou:.4f}")
    print("  importance: " + ", ".join(f"{k} {v:.2f}" for k, v in scores["importance"].items()))

    ref = Path(args.s2).resolve()
    files = sorted({f for pat in args.predict for f in (glob.glob(pat) or [pat]) if Path(f).resolve() != ref})
    for f in files:
        xf, vf, pf = load_stack(f, args.dem, args.mask_cloud)
        mf, pr8 = to_rasters(predict_scene(rf, xf, vf), vf)
        write(out / f"{Path(f).stem}_prob.tif", pr8, pf)
        write(out / f"{Path(f).stem}_mask.tif", mf, pf)
        print(f"{Path(f).stem}: glacier {(mf == 1).sum() * PIXEL_KM2:.2f} km2")

    (out / "scores.json").write_text(json.dumps({**scores, **vars(args)}, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
