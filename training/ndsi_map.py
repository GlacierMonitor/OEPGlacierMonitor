"""NDSI-threshold glacier maps for whole scenes, in the same format as predict.py, so the NDSI baseline
gets an area time series like the models.

Usage:
    venv/Scripts/python ndsi_map.py --s2 raw/sentinel2/S2_*.tif --dem CopDEM_GLO30_UTM45N.tif \
        --masks masks --out runs/matrix/ndsi/years

Writes <stem>_mask.tif (1 glacier = NDSI > --threshold, 0 not, 255 no data) and areas.csv. No data is the
same as for the models: missing bands, outside the DEM, and with --masks / --mask-cloud the cloud pixels.
"""
import argparse
import csv
import glob
from pathlib import Path

import numpy as np

from data import CHANNELS, load_stack, read_mask
from predict import PIXEL_KM2, write


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--s2", nargs="+", required=True, help="one or more S2 files (wildcards ok)")
    p.add_argument("--dem", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--threshold", type=float, default=0.4)
    p.add_argument("--masks", help="Part 2 masks folder: cloud, cloud shadow -> no data")
    p.add_argument("--mask-cloud", action="store_true", help="SCL cloud masking instead")
    args = p.parse_args()

    files = sorted({f for pat in args.s2 for f in (glob.glob(pat) or [pat])})
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in files:
        stem = Path(f).stem
        x, valid, profile = load_stack(f, args.dem, args.mask_cloud)
        if args.masks:
            valid &= ~read_mask(f, args.masks, profile)
        mask = (x[CHANNELS.index("NDSI")] > args.threshold).astype("uint8")
        del x
        mask[~valid] = 255
        write(out / f"{stem}_mask.tif", mask, profile)
        area = (mask == 1).sum() * PIXEL_KM2
        rows.append({"scene": stem, "glacier_km2": round(area, 3), "no_data_%": round(100 * (~valid).mean(), 2)})
        print(f"{stem}: NDSI>{args.threshold} glacier {area:.2f} km2, no data {rows[-1]['no_data_%']}%")
    with open(out / "areas.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("->", out / "areas.csv")


if __name__ == "__main__":
    main()
