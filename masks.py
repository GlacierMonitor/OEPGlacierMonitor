"""GlacierMonitor Part 2, approach A: cloud, cloud-shadow and terrain-shadow masks.

For every S2_YYYYMMDD.tif in GlacierMonitor/raw/sentinel2 this writes, on the same 10 m grid:
  masks/mask_YYYYMMDD.tif     uint8: 0 clear, 1 cloud, 2 cloud shadow, 3 terrain shadow, 255 no data
  masks/compare_YYYYMMDD.tif  uint8 cloud agreement, for review in QGIS:
                              0 neither, 1 OmniCloudMask only, 2 SCL only, 3 both, 255 no data
  masks/previews/mask_YYYYMMDD.png
  masks/mask_summary.csv      % of each class per year and the OmniCloudMask / SCL agreement

Cloud and cloud shadow come from OmniCloudMask (thick + thin cloud, shadow), run on B04, B03, B08.
SCL with the NDSI snow correction (decisions.md) is only a second opinion, written to compare_*.tif.
Terrain shadow uses the DEM and the scene's mean sun angles: a slope facing away from the sun, or a
pixel whose line to the sun is blocked by higher ground. It is a flag, not something to remove.

The raw OmniCloudMask result is cached in masks/ocm/, so a re-run skips the slow step (~5 min/scene on CPU).

  venv\\Scripts\\python masks.py                 # all years
  venv\\Scripts\\python masks.py --years 2021    # one year
"""
import argparse
import csv
import json

import matplotlib
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

import config as C

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MASK_DIR = C.ROOT / "GlacierMonitor" / "masks"
OCM_DIR = MASK_DIR / "ocm"
PREV_DIR = MASK_DIR / "previews"
SUN_FILE = MASK_DIR / "sun_angles.json"
DEM_FILE = C.DEM_DIR / "CopDEM_GLO30_UTM45N.tif"

CLEAR, CLOUD, CLOUD_SHADOW, TERRAIN_SHADOW, NODATA = 0, 1, 2, 3, 255
SCL_CLOUD_ONLY = [8, 9, 10]          # cloud medium, cloud high, cirrus (SCL 3 shadow not compared)
NDSI_SNOW = 0.4
MAX_SHADOW_M = 10000                 # how far to look towards the sun for higher ground


# --- Sun angles ------------------------------------------------------------
def sun_angles(dates):
    """Mean solar zenith / azimuth (degrees) per date from the STAC metadata, cached in a json file."""
    cache = json.loads(SUN_FILE.read_text()) if SUN_FILE.exists() else {}
    missing = [d for d in dates if d not in cache]
    if missing:
        catalog = C.open_catalog()
        for d in missing:
            iso = f"{d[:4]}-{d[4:6]}-{d[6:]}"
            items = list(C.retry(C.search, catalog, C.S2_COLLECTION, iso, iso))
            cache[d] = {"zenith": float(np.mean([i.properties["s2:mean_solar_zenith"] for i in items])),
                        "azimuth": float(np.mean([i.properties["s2:mean_solar_azimuth"] for i in items]))}
        SUN_FILE.write_text(json.dumps(cache, indent=1))
    return cache


# --- Terrain shadow --------------------------------------------------------
def terrain_shadow(dem, res, zenith, azimuth):
    """Bool array: self-shadow (slope faces away from the sun) or cast shadow (sun blocked)."""
    z, az = np.radians(zenith), np.radians(azimuth)
    # x east, y north, z up; array rows run south, so dz/dy = -dz/drow
    dz_drow, dz_dcol = np.gradient(dem, res)
    sun = np.array([np.sin(z) * np.sin(az), np.sin(z) * np.cos(az), np.cos(z)])
    illum = (-dz_dcol * sun[0] + dz_drow * sun[1] + sun[2]) / np.sqrt(dz_dcol ** 2 + dz_drow ** 2 + 1)
    shadow = illum <= 0

    # cast shadow: march towards the sun and compare with the sun-ray height
    tan_elev = np.tan(np.pi / 2 - z)
    step_r, step_c = -np.cos(az), np.sin(az)          # one pixel towards the sun, in row/col
    h, w = dem.shape
    for k in range(1, int(MAX_SHADOW_M / res) + 1):
        dr, dc = int(round(k * step_r)), int(round(k * step_c))
        if abs(dr) >= h or abs(dc) >= w:
            break
        src = dem[max(dr, 0):h + min(dr, 0), max(dc, 0):w + min(dc, 0)]
        dst = (slice(max(-dr, 0), h + min(-dr, 0)), slice(max(-dc, 0), w + min(-dc, 0)))
        shadow[dst] |= src > dem[dst] + k * res * tan_elev
    return shadow


def terrain_shadow_10m(profile, zenith, azimuth):
    with rasterio.open(DEM_FILE) as s:
        dem = s.read(1).astype("float64")
        dem[(dem == s.nodata) | ~np.isfinite(dem)] = np.nan
        dem = np.where(np.isnan(dem), np.nanmean(dem), dem)
        sh = terrain_shadow(dem, s.res[0], zenith, azimuth).astype("uint8")
        out = np.zeros((profile["height"], profile["width"]), "uint8")
        reproject(sh, out, src_transform=s.transform, src_crs=s.crs,
                  dst_transform=profile["transform"], dst_crs=profile["crs"], resampling=Resampling.nearest)
    return out.astype(bool)


# --- Cloud -----------------------------------------------------------------
def omnicloudmask(path, rgn, profile):
    """OmniCloudMask classes (0 clear, 1 thick cloud, 2 thin cloud, 3 shadow), cached as a GeoTIFF."""
    cache = OCM_DIR / path.name.replace("S2_", "ocm_")
    if not cache.exists():
        from omnicloudmask import predict_from_array
        pred = predict_from_array(rgn, inference_device="cpu")[0].astype("uint8")
        write(cache, pred, profile, nodata=None)
    with rasterio.open(cache) as s:
        return s.read(1)


def scl_cloud(scl, green, swir):
    """SCL cloud classes minus the pixels the NDSI snow correction calls snow."""
    ndsi = np.where(green + swir > 0, (green - swir) / (green + swir + 1e-6), 0)
    return np.isin(scl, SCL_CLOUD_ONLY) & ~(ndsi > NDSI_SNOW)


# --- Output ----------------------------------------------------------------
def write(path, arr, profile, nodata=NODATA, names=None):
    p = {"driver": "GTiff", "height": arr.shape[0], "width": arr.shape[1], "count": 1, "dtype": "uint8",
         "crs": profile["crs"], "transform": profile["transform"], "nodata": nodata,
         "compress": "deflate", "tiled": True, "blockxsize": 512, "blockysize": 512}
    with rasterio.open(path, "w", **p) as d:
        d.write(arr, 1)
        if names:
            d.set_band_description(1, names)


QML = """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.44" styleCategories="Symbology">
 <pipe>
  <rasterrenderer type="paletted" band="1" opacity="1" alphaBand="-1" nodataColor="">
   <rasterTransparency/>
   <colorPalette>
{entries}
   </colorPalette>
  </rasterrenderer>
 </pipe>
</qgis>
"""
# value, colour, alpha, label; values not listed (0 clear / neither) draw transparent
MASK_STYLE = [(1, "#ffff00", 255, "cloud"), (2, "#ff00ff", 255, "cloud shadow"), (3, "#3355ff", 130, "terrain shadow")]
COMPARE_STYLE = [(1, "#ff8000", 255, "OmniCloudMask only"), (2, "#4da6ff", 255, "SCL only"), (3, "#ffff00", 255, "both")]


def write_qml(tif, style):
    """Sidecar style: QGIS applies <name>.qml automatically when the raster is opened."""
    entries = "\n".join(f'    <paletteEntry value="{v}" color="{c}" alpha="{a}" label="{lab}"/>'
                        for v, c, a, lab in style)
    tif.with_suffix(".qml").write_text(QML.format(entries=entries))


def stretch(z):
    o = np.zeros(z.shape[1:] + (3,), "float32")
    for i in range(3):
        v = z[i][z[i] > 0]
        lo, hi = np.percentile(v, [2, 98]) if v.size else (0, 1)
        o[..., i] = np.clip((z[i] - lo) / (hi - lo), 0, 1)
    return o


def preview(path, fc, mask, compare, title):
    f = 4
    img = stretch(fc[:, ::f, ::f].astype("float32"))
    m, c = mask[::f, ::f], compare[::f, ::f]
    a = img.copy()
    a[m == TERRAIN_SHADOW] = 0.5 * a[m == TERRAIN_SHADOW] + 0.5 * np.array([0.2, 0.3, 1.0])
    a[m == CLOUD] = [1, 1, 0]
    a[m == CLOUD_SHADOW] = [1, 0, 1]
    b = img.copy()
    for val, col in ((1, [1, 0.5, 0]), (2, [0.3, 0.6, 1]), (3, [1, 1, 0])):
        b[c == val] = col
    fig, ax = plt.subplots(1, 3, figsize=(21, 6.5))
    for x, im, t in zip(ax, (img, a, b), (f"{title} false colour (B11/B08/B04)",
                                          "mask: yellow cloud, magenta cloud shadow, blue terrain shadow",
                                          "cloud: orange OCM only, light blue SCL only, yellow both")):
        x.imshow(im); x.set_title(t); x.axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=80); plt.close(fig)


# --- Main ------------------------------------------------------------------
def run(years):
    for d in (MASK_DIR, OCM_DIR, PREV_DIR):
        d.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in C.S2_DIR.glob("S2_*.tif") if int(f.stem[3:7]) in years)
    if not files:
        raise SystemExit(f"no S2 files for {years} in {C.S2_DIR}")
    sun = sun_angles([f.stem[3:] for f in files])
    rows = []
    for f in files:
        date = f.stem[3:]
        with rasterio.open(f) as s:
            b = dict(zip(s.descriptions, s.read()))
            profile = s.profile
        valid = b["B03"] > 0
        n = valid.sum()

        ocm = omnicloudmask(f, np.stack([b["B04"], b["B03"], b["B08"]]), profile)
        ocm_cloud = np.isin(ocm, [1, 2]) & valid
        scl = scl_cloud(b["SCL"], b["B03"].astype("float32"), b["B11"].astype("float32")) & valid
        terrain = terrain_shadow_10m(profile, sun[date]["zenith"], sun[date]["azimuth"]) & valid

        mask = np.full(valid.shape, CLEAR, "uint8")             # later classes win
        mask[terrain] = TERRAIN_SHADOW
        mask[(ocm == 3) & valid] = CLOUD_SHADOW
        mask[ocm_cloud] = CLOUD
        mask[~valid] = NODATA
        write(MASK_DIR / f"mask_{date}.tif", mask, profile,
              names="0 clear, 1 cloud, 2 cloud shadow, 3 terrain shadow, 255 no data")
        write_qml(MASK_DIR / f"mask_{date}.tif", MASK_STYLE)

        compare = (ocm_cloud.astype("uint8") + 2 * scl.astype("uint8"))
        compare[~valid] = NODATA
        write(MASK_DIR / f"compare_{date}.tif", compare, profile,
              names="0 neither, 1 OmniCloudMask only, 2 SCL only, 3 both")
        write_qml(MASK_DIR / f"compare_{date}.tif", COMPARE_STYLE)
        preview(PREV_DIR / f"mask_{date}.png", np.stack([b["B11"], b["B08"], b["B04"]]), mask, compare, date)

        pct = lambda m: round(100 * m.sum() / n, 2)  # noqa: E731
        either = ocm_cloud | scl
        row = {"date": date, "cloud_%": pct(mask == CLOUD), "cloud_shadow_%": pct(mask == CLOUD_SHADOW),
               "terrain_shadow_%": pct(mask == TERRAIN_SHADOW), "clear_%": pct(mask == CLEAR),
               "ocm_cloud_%": pct(ocm_cloud), "scl_cloud_%": pct(scl),
               "both_%": pct(ocm_cloud & scl), "ocm_only_%": pct(ocm_cloud & ~scl), "scl_only_%": pct(scl & ~ocm_cloud),
               "agreement_iou": round((ocm_cloud & scl).sum() / either.sum(), 3) if either.any() else 1.0,
               "sun_zenith": round(sun[date]["zenith"], 1), "sun_azimuth": round(sun[date]["azimuth"], 1)}
        rows.append(row)
        print(f"{date}: cloud {row['cloud_%']}%, cloud shadow {row['cloud_shadow_%']}%, terrain shadow "
              f"{row['terrain_shadow_%']}% | SCL cloud {row['scl_cloud_%']}%, agreement IoU {row['agreement_iou']}",
              flush=True)

    summary = MASK_DIR / "mask_summary.csv"
    old = {}
    if summary.exists():
        with open(summary, newline="") as fh:
            old = {r["date"]: r for r in csv.DictReader(fh)}
    old.update({r["date"]: r for r in rows})
    with open(summary, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(old[k] for k in sorted(old))
    print(f"-> {summary}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--years", type=int, nargs="+", default=list(C.YEARS))
    run(set(p.parse_args().years))


if __name__ == "__main__":
    main()
