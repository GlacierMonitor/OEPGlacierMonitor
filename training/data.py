"""Model inputs and training tiles for GlacierMonitor Part 4 (U-Net).

Inputs come straight from Part 1's export.py:
  - S2_YYYYMMDD.tif : 7-band uint16 (B02, B03, B04, B08, B11, B12, SCL), nodata 0, 10 m, EPSG:32645
  - CopDEM_GLO30_UTM45N.tif : float32 elevation, nodata -32767, 30 m grid of the same area

Labels and tiles (Part 3) are expected as:
  - label GeoTIFF on the same 10 m grid as the S2 file: 1 = glacier, 0 = not glacier, 255 = ignore
  - tiles.csv with columns row, col, split (train/val/test) and optionally glacier;
    row/col are the pixel offsets of each tile's top-left corner, tile size = TILE
"""
import json

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.warp import Resampling, reproject

S2_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
CHANNELS = S2_BANDS + ["NDSI", "elevation", "slope"]
TILE = 256
IGNORE = 255
SCL_BAD = (0, 1)                 # no data, saturated/defective
SCL_CLOUD = (3, 8, 9, 10)        # shadow, cloud medium/high, cirrus (unreliable before 2022)


def slope_degrees(dem, res):
    """Slope in degrees from a DEM array with square pixels of size res (m)."""
    dy, dx = np.gradient(dem, res)
    return np.degrees(np.arctan(np.hypot(dx, dy))).astype("float32")


def load_stack(s2_path, dem_path, mask_cloud=False):
    """Return (x, valid, profile) for one year.

    x is (9, H, W) float32 in CHANNELS order: reflectance 0-1, NDSI, elevation (m), slope (deg).
    valid is (H, W) bool: pixels with data (and, if mask_cloud, not SCL cloud/shadow).
    """
    with rasterio.open(s2_path) as src:
        s2 = src.read()
        profile = src.profile
        names = list(src.descriptions)
    if names[:7] != S2_BANDS + ["SCL"]:
        raise ValueError(f"{s2_path}: unexpected bands {names}")

    refl = s2[:6].astype("float32") / 10000.0
    scl = s2[6]
    valid = (s2[:6] > 0).all(0) & ~np.isin(scl, SCL_BAD)
    if mask_cloud:
        valid &= ~np.isin(scl, SCL_CLOUD)

    green, swir = refl[1], refl[4]
    ndsi = np.where(green + swir > 0, (green - swir) / (green + swir + 1e-6), 0).astype("float32")

    # Slope on the native 30 m DEM, then both layers bilinear onto the 10 m S2 grid
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype("float32")
        dem_nodata = src.nodata
        dem_profile = src.profile
        res = src.res[0]
    dem_ok = dem != dem_nodata
    dem_filled = np.where(dem_ok, dem, np.nan)
    slope = slope_degrees(np.where(dem_ok, dem, np.nanmean(dem_filled)), res)
    slope[~dem_ok] = np.nan

    h, w = profile["height"], profile["width"]
    terrain = np.full((2, h, w), np.nan, dtype="float32")
    for i, layer in enumerate((dem_filled, slope)):
        reproject(layer, terrain[i], src_transform=dem_profile["transform"], src_crs=dem_profile["crs"],
                  dst_transform=profile["transform"], dst_crs=profile["crs"],
                  src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.bilinear)
    valid &= np.isfinite(terrain).all(0)
    terrain = np.nan_to_num(terrain)

    x = np.concatenate([refl, ndsi[None], terrain]).astype("float32")
    return x, valid, profile


def read_label(path, valid, profile):
    with rasterio.open(path) as src:
        y = src.read(1).astype("uint8")
        same = src.crs == profile["crs"] and src.transform.almost_equals(profile["transform"])
    if y.shape != valid.shape:
        raise ValueError(f"label {y.shape} does not match image grid {valid.shape}")
    if not same:
        raise ValueError(f"{path} is not on the same grid as the image (CRS or transform differs)")
    y[~valid] = IGNORE
    return y


def read_tiles(path, shape):
    tiles = pd.read_csv(path)
    missing = {"row", "col", "split"} - set(tiles.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    h, w = shape
    bad = (tiles.row < 0) | (tiles.col < 0) | (tiles.row + TILE > h) | (tiles.col + TILE > w)
    if bad.any():
        raise ValueError(f"{bad.sum()} tiles fall outside the {h}x{w} image")
    tr = tiles[tiles.split == "train"][["row", "col"]].to_numpy()
    ot = tiles[tiles.split != "train"][["row", "col"]].to_numpy()
    if len(tr) and len(ot):
        d = np.abs(tr[:, None, :] - ot[None, :, :])
        if (d < TILE).all(2).any():
            raise ValueError(f"{path}: train tiles overlap val/test tiles")
    return tiles


# --- Normalisation (training-set statistics) ----------------------------------
def compute_stats(x, y, tiles):
    """Per-channel mean/std over labelled pixels of the training tiles only."""
    sums = np.zeros(len(CHANNELS)); sq = np.zeros(len(CHANNELS)); n = 0
    for r, c in tiles[["row", "col"]].itertuples(index=False):
        m = y[r:r + TILE, c:c + TILE] != IGNORE
        v = x[:, r:r + TILE, c:c + TILE][:, m].astype("float64")
        sums += v.sum(1); sq += (v ** 2).sum(1); n += m.sum()
    if n == 0:
        raise ValueError("training tiles have no labelled pixels")
    mean = sums / n
    std = np.sqrt(np.maximum(sq / n - mean ** 2, 1e-12))
    return {"channels": CHANNELS, "mean": mean.tolist(), "std": std.tolist()}


def normalise(x, valid, stats):
    mean = np.asarray(stats["mean"], dtype="float32")[:, None, None]
    std = np.asarray(stats["std"], dtype="float32")[:, None, None]
    out = (x - mean) / std
    out[:, ~valid] = 0.0
    return out


def save_stats(stats, path):
    path.write_text(json.dumps(stats, indent=2))


class TileDataset(torch.utils.data.Dataset):
    """256x256 tiles cut from one normalised image + label; random flips/rotations if augment."""

    def __init__(self, x, y, tiles, augment=False):
        self.x, self.y, self.augment = x, y, augment
        self.pos = tiles[["row", "col"]].to_numpy()

    def __len__(self):
        return len(self.pos)

    def __getitem__(self, i):
        r, c = self.pos[i]
        x = self.x[:, r:r + TILE, c:c + TILE]
        y = self.y[r:r + TILE, c:c + TILE]
        if self.augment:
            k = np.random.randint(4)
            x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k)
            if np.random.rand() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy()).long()
