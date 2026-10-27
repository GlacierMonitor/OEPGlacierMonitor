"""Write synthetic test data in exactly the format Part 1 / Part 3 will deliver.

Only for testing the training code before the real data exists; the numbers mean nothing.

Usage:
    venv/Scripts/python make_dummy.py --out dummy            # 1280 x 1536 px, quick
    venv/Scripts/python make_dummy.py --out dummy_full --full   # real 3220 x 3940 grid

Writes S2_20991024.tif and S2_20251122.tif (glaciers shrunk), CopDEM_GLO30_UTM45N.tif,
label_2099.tif, tiles.csv (train/val/test in separate vertical strips = separate glaciers), and
glacier_ids.tif + glacier_ids.csv (one outline per connected glacier, like the RGI-based ID raster).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize, shapes
from rasterio.transform import from_origin

from data import S2_BANDS, TILE

CRS, ORIGIN = "EPSG:32645", (460600, 3110570)       # same grid origin as config.geobox(10)


def blobs(h, w, n, rng, scale):
    yy, xx = np.mgrid[0:h, 0:w].astype("float32")
    out = np.zeros((h, w), "float32")
    for _ in range(n):
        cy, cx, s = rng.uniform(0, h), rng.uniform(0, w), rng.uniform(0.5, 1.5) * scale
        out += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s ** 2))
    return out


def scene(h, w, glacier, debris, lake, rng):
    """Uint16 reflectance-like bands (offset already removed) + SCL."""
    rock = np.array([900, 1100, 1300, 2000, 2600, 2200], "float32")      # B02 B03 B04 B08 B11 B12
    ice = np.array([6500, 6300, 6000, 5200, 900, 600], "float32")
    deb = np.array([1000, 1200, 1350, 1900, 2300, 1900], "float32")
    water = np.array([700, 800, 600, 300, 150, 100], "float32")
    bands = np.where(glacier & ~debris, ice[:, None, None], rock[:, None, None])
    bands = np.where(debris, deb[:, None, None], bands)
    bands = np.where(lake, water[:, None, None], bands)
    bands = bands * rng.normal(1, 0.08, (1, h, w)) + rng.normal(0, 80, (6, h, w))
    bands = np.clip(bands, 1, 20000).astype("uint16")
    scl = np.where(glacier & ~debris, 11, 5).astype("uint16")
    scl = np.where(lake, 6, scl)
    bands[:, :, :8] = 0; scl[:, :8] = 0                                   # a strip of nodata
    return np.concatenate([bands, scl[None]])


def write(path, arr, transform, dtype, nodata, names):
    arr = arr[None] if arr.ndim == 2 else arr
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2],
                       count=arr.shape[0], dtype=dtype, crs=CRS, transform=transform, nodata=nodata,
                       compress="deflate", tiled=True, blockxsize=256, blockysize=256) as dst:
        dst.write(arr.astype(dtype))
        for i, n in enumerate(names, 1):
            dst.set_band_description(i, n)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out", default="dummy")
    p.add_argument("--full", action="store_true", help="use the real 3220 x 3940 grid")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    h, w = (3220, 3940) if args.full else (1280, 1536)

    # DEM on a 30 m grid covering the same area
    dh, dw = -(-h // 3), -(-w // 3)
    dem30 = 3500 + 4500 * blobs(dh, dw, 40, rng, dw / 12) / 2.5
    dem30 += rng.normal(0, 15, dem30.shape)
    dem30 = np.clip(dem30, 3500, 8500).astype("float32")
    dem = np.kron(dem30, np.ones((3, 3), "float32"))[:h, :w]

    field = blobs(h, w, 30, rng, w / 25)
    glacier = (field > 0.55) & (dem > 4800)
    debris = glacier & (dem < 5600)
    lake = (blobs(h, w, 6, rng, w / 120) > 0.8) & ~glacier
    shrunk = glacier & (field > 0.65)

    t10 = from_origin(*ORIGIN, 10, 10)
    write(out / "CopDEM_GLO30_UTM45N.tif", dem30, from_origin(*ORIGIN, 30, 30), "float32", -32767.0,
          ["elevation_m"])
    write(out / "S2_20991024.tif", scene(h, w, glacier, debris, lake, rng), t10, "uint16", 0,
          S2_BANDS + ["SCL"])
    write(out / "S2_20251122.tif", scene(h, w, shrunk, debris & shrunk, lake, rng), t10, "uint16", 0,
          S2_BANDS + ["SCL"])
    write(out / "label_2099.tif", glacier.astype("uint8"), t10, "uint8", 255, ["glacier"])

    # non-overlapping tiles; strips 0-60% train, 60-80% val, 80-100% test
    rows = []
    for r in range(0, h - TILE + 1, TILE):
        for c in range(0, w - TILE + 1, TILE):
            f = (c + TILE / 2) / w
            rows.append({"row": r, "col": c, "split": "train" if f < 0.6 else "val" if f < 0.8 else "test"})
    tiles = pd.DataFrame(rows)
    tiles.to_csv(out / "tiles.csv", index=False)

    # glacier outlines: each connected glacier of 2099 (>= 1000 px) gets its own integer ID
    polys = [g for g, _ in shapes(glacier.astype("uint8"), mask=glacier, connectivity=8, transform=t10)]
    ids = np.zeros((h, w), "int32")
    for g in polys:
        one = rasterize([(g, 1)], out_shape=(h, w), transform=t10, dtype="uint8")
        if one.sum() >= 1000:
            ids[(one == 1) & (ids == 0)] = ids.max() + 1
    write(out / "glacier_ids.tif", ids, t10, "int32", None, ["glacier_id"])
    pd.DataFrame({"value": range(1, ids.max() + 1),
                  "glacier_id": [f"DUMMY-{i:03d}" for i in range(1, ids.max() + 1)],
                  "glacier_name": [f"Dummy {i}" for i in range(1, ids.max() + 1)]}
                 ).to_csv(out / "glacier_ids.csv", index=False)
    print(f"{out}: {h}x{w} px, glacier {glacier.mean():.1%} (debris {debris.mean():.1%}), "
          f"tiles {tiles.split.value_counts().to_dict()}, {ids.max()} glacier outlines")


if __name__ == "__main__":
    main()
