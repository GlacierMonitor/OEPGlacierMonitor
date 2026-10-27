"""Run a trained U-Net over whole scenes (any year) and optionally score it against labels.

Usage:
    venv/Scripts/python predict.py --model runs/ref2022/best.pt --dem CopDEM_GLO30_UTM45N.tif \
        --s2 raw/sentinel2/S2_*.tif --out predictions
    # score on the held-out glacier, U-Net vs NDSI baseline:
    venv/Scripts/python predict.py ... --label label_2022.tif --tiles tiles.csv --split test

For each S2 file writes <stem>_prob.tif (glacier probability 0-100) and <stem>_mask.tif
(1 glacier, 0 not, 255 no data) on the same 10 m grid, plus areas.csv.
"""
import argparse
import csv
import glob
from pathlib import Path

import numpy as np
import rasterio
import torch

from data import CHANNELS, IGNORE, TILE, load_stack, normalise, read_label, read_tiles, select_channels
from model import build_model, confusion, iou

PIXEL_KM2 = 10 * 10 / 1e6


@torch.no_grad()
def predict_scene(model, x, device, amp_dtype, stride=TILE // 2, batch=8):
    """Sliding-window glacier probability for a (C, H, W) normalised image."""
    _, h, w = x.shape
    # pad so that windows at the given stride exactly cover the image
    padded = [TILE + -(-max(n - TILE, 0) // stride) * stride for n in (h, w)]
    xp = np.pad(x, ((0, 0), (0, padded[0] - h), (0, padded[1] - w)), mode="reflect")
    H, W = xp.shape[1:]
    # taper the window edges so overlapping tiles blend without seams
    ramp = np.minimum(np.arange(TILE) + 1, TILE - np.arange(TILE)).astype("float32")
    weight = np.minimum.outer(ramp, ramp) / (TILE / 2)
    prob = np.zeros((H, W), "float32"); wsum = np.zeros((H, W), "float32")
    pos = [(r, c) for r in range(0, H - TILE + 1, stride) for c in range(0, W - TILE + 1, stride)]
    for i in range(0, len(pos), batch):
        chunk = pos[i:i + batch]
        xb = torch.from_numpy(np.stack([xp[:, r:r + TILE, c:c + TILE] for r, c in chunk])).to(device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            pb = torch.sigmoid(model(xb)[:, 0].float()).cpu().numpy()
        for (r, c), pr in zip(chunk, pb):
            prob[r:r + TILE, c:c + TILE] += pr * weight
            wsum[r:r + TILE, c:c + TILE] += weight
    return (prob / np.maximum(wsum, 1e-6))[:h, :w]


def write(path, arr, profile):
    prof = dict(profile, count=1, dtype="uint8", nodata=255, compress="deflate", predictor=2)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr[None])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", required=True); p.add_argument("--dem", required=True)
    p.add_argument("--s2", nargs="+", required=True, help="one or more S2 files (wildcards ok)")
    p.add_argument("--out", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--label", help="label GeoTIFF to score against (only for the S2 file it belongs to)")
    p.add_argument("--tiles", help="tiles.csv; with --split, score only those tiles")
    p.add_argument("--split", default="test")
    p.add_argument("--ndsi", type=float, default=0.4, help="NDSI threshold for the baseline comparison")
    args = p.parse_args()

    files = sorted({f for pat in args.s2 for f in (glob.glob(pat) or [pat])})
    if args.label and len(files) != 1:
        raise SystemExit("--label scores one scene: pass exactly one --s2 file with it")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    channels = ckpt.get("channels", CHANNELS)
    model = build_model(ckpt["encoder"], weights=None, in_channels=len(channels))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    print(f"model {args.model} (epoch {ckpt['epoch']}, val IoU {ckpt['val_iou']:.4f}), "
          f"inputs: {', '.join(channels)}")

    rows = []
    for f in files:
        stem = Path(f).stem
        x, valid, profile = load_stack(f, args.dem, ckpt.get("mask_cloud", False))
        ndsi = x[6].copy()
        x = select_channels(x, channels)
        prob = predict_scene(model, normalise(x, valid, ckpt["stats"]), device, amp_dtype)
        mask = (prob >= args.threshold).astype("uint8")
        mask[~valid] = 255
        prob8 = np.round(prob * 100).astype("uint8"); prob8[~valid] = 255
        write(out / f"{stem}_prob.tif", prob8, profile)
        write(out / f"{stem}_mask.tif", mask, profile)
        area = (mask == 1).sum() * PIXEL_KM2
        nodata_pct = 100 * (~valid).mean()
        row = {"scene": stem, "glacier_km2": round(area, 3), "no_data_%": round(nodata_pct, 2)}
        print(f"{stem}: glacier {area:.2f} km2, no data {nodata_pct:.1f}%")

        if args.label:
            y = read_label(args.label, valid, profile)
            region = np.zeros_like(valid)
            if args.tiles:
                t = read_tiles(args.tiles, valid.shape)
                for r, c in t[t.split == args.split][["row", "col"]].itertuples(index=False):
                    region[r:r + TILE, c:c + TILE] = True
            else:
                region[:] = True
            y = np.where(region, y, IGNORE)
            unet = iou(*confusion(mask == 1, y))
            base = iou(*confusion((ndsi > args.ndsi) & valid, y))
            where = f"'{args.split}' tiles" if args.tiles else "whole label"
            print(f"  IoU on {where}: U-Net {unet:.4f} | NDSI>{args.ndsi} {base:.4f}")
            row.update({"scored_on": where, "iou_unet": round(unet, 4), f"iou_ndsi_{args.ndsi}": round(base, 4)})
        rows.append(row)

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "areas.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print("->", out / "areas.csv")


if __name__ == "__main__":
    main()
