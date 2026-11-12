"""GlacierMonitor paper section 5.3: glacier lakes and vegetation near villages, 2016-2025.

Works on the 10 Sentinel-2 images and cloud masks only (no trained model needed).

  venv\\Scripts\\python impacts.py

Writes:
  GlacierMonitor/results/lakes.csv          lake area per year (km2) for the lakes in LAKES, + frozen / cloud flags
  GlacierMonitor/results/village_ndvi.csv   NDVI per year in each village square (make_labels.PLACES) + whole-area reference
  GlacierMonitor/figures/F8_lakes_ndvi.png  both as one figure

Lakes: glacial lakes are milky (NIR/green up to ~0.55 in Imja Tsho), so lake water here is SWIR1 < 250, NIR/green < 0.6
and green < 3500 (snow, even shaded, is brighter or has more SWIR). Each lake's extent is the
connected water area (water in >= 2 of 10 years) that contains its seed point, grown by 30 m; its area in a year is the
water pixels inside that extent. A lake that is frozen or snow-covered in a year fails the water test, so 'ice_%' (bright
snow/ice inside the extent: NDSI > 0.4 and green >= 3500) is reported: a high value means the area for that year is too low, not that the lake shrank.
Vegetation: NDVI = (B08 - B04) / (B08 + B04) on pixels that are not glacier (inventory), water, snow/ice (NDSI > 0.4),
cloud, cloud shadow or no data. October-November NDVI also depends on grazing, harvest and snow, so trends are an
association with glacier change, not evidence of a cause.
"""
import sqlite3

import matplotlib
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.features import rasterize, shapes
from shapely.geometry import Point, shape
from shapely.ops import unary_union

import config as C
from make_labels import PLACES, TILE, gpkg_geom

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MASK_DIR = C.ROOT / "GlacierMonitor" / "masks"
RESULTS = C.ROOT / "GlacierMonitor" / "results"
FIGURES = C.ROOT / "GlacierMonitor" / "figures"
INVENTORY = C.ROOT / "GlacierMonitor" / "inventory" / "glaciers.gpkg"

# lon, lat of a point inside each lake, nearby village, and a radius (m): 0 = the one lake at the point; > 0 = every
# lake within that distance together (Spillway Lake is a cluster of ponds on the Ngozumpa tongue that merge over time)
LAKES = {
    "Imja Tsho": ((86.925, 27.900), "upstream of Dingboche / Pheriche", 0),
    "Gokyo lake (Dudh Pokhari)": ((86.695, 27.955), "Gokyo", 0),
    "Spillway Lake cluster (Ngozumpa)": ((86.7133, 27.9408), "Gokyo / Thangnak", 600),
}
LAKE_GROW_M = 30
MIN_WATER_YEARS = 2


def read_year(path):
    date = path.stem[3:]
    with rasterio.open(path) as s:
        b = dict(zip(s.descriptions, s.read().astype("float32")))
    with rasterio.open(MASK_DIR / f"mask_{date}.tif") as m:
        mask = m.read(1)
    ok = (b["B03"] > 0) & ~np.isin(mask, [1, 2, 255])
    water = (b["B11"] < 250) & (b["B08"] / (b["B03"] + 1) < 0.6) & (b["B03"] < 3500) & (b["B03"] > 0)
    ndsi = (b["B03"] - b["B11"]) / (b["B03"] + b["B11"] + 1e-6)
    ice = (ndsi > 0.4) & (b["B03"] >= 3500)
    ndvi = (b["B08"] - b["B04"]) / (b["B08"] + b["B04"] + 1e-6)
    return date, ok, water, ndsi > 0.4, ice, ndvi


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    files = sorted(C.S2_DIR.glob("S2_*.tif"))
    with rasterio.open(files[0]) as s:
        tf, crs, shp = s.transform, s.crs, (s.height, s.width)
    con = sqlite3.connect(INVENTORY.resolve().as_uri() + "?mode=ro", uri=True)
    glacier = rasterize([(gpkg_geom(g), 1) for (g,) in con.execute("select geom from glaciers")],
                        out_shape=shp, transform=tf, dtype="uint8").astype(bool)
    years = [read_year(f) for f in files]

    # --- lakes ---------------------------------------------------------------
    water_count = sum(w.astype("uint8") for _, _, w, _, _, _ in years)
    perennial = water_count >= MIN_WATER_YEARS
    polys = [shape(g) for g, v in shapes(perennial.astype("uint8"), mask=perennial, transform=tf)]
    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    rows = []
    for name, ((lon, lat), village, radius) in LAKES.items():
        seed = Point(*to_utm.transform(lon, lat))
        near = [p for p in polys if p.distance(seed) < max(radius, 300)]
        if not near:
            print(f"{name}: no water found near the seed point - check LAKES")
            continue
        if radius:
            lake = unary_union([p for p in near if p.area >= 500])
        else:
            lake = max(near, key=lambda p: p.area if p.distance(seed) == 0 else p.area / 10)
        extent = rasterize([(lake.buffer(LAKE_GROW_M), 1)], out_shape=shp, transform=tf, dtype="uint8").astype(bool)
        for date, ok, water, _, ice, _ in years:
            n = extent.sum()
            rows.append({"lake": name, "village": village, "date": date, "year": int(date[:4]),
                         "area_km2": round((water & extent).sum() * 1e-4, 3),
                         "ice_%": round(100 * (ice & extent).sum() / n, 1),
                         "cloud_%": round(100 * (~ok & extent).sum() / n, 1)})
        print(f"{name}: max extent {lake.area / 1e6:.3f} km2")
    lakes = pd.DataFrame(rows)
    lakes.to_csv(RESULTS / "lakes.csv", index=False)

    # --- vegetation ----------------------------------------------------------
    rows = []
    areas = {name: (slice(r * TILE, (r + 1) * TILE), slice(c * TILE, (c + 1) * TILE)) for (r, c), name in PLACES.items()}
    areas["whole area (reference)"] = (slice(None), slice(None))
    for date, ok, water, snow, _, ndvi in years:
        land = ok & ~glacier & ~water & ~snow
        for name, w in areas.items():
            m = land[w]
            v = ndvi[w][m]
            rows.append({"place": name, "date": date, "year": int(date[:4]),
                         "ndvi_median": round(float(np.median(v)), 3) if v.size else np.nan,
                         "vegetated_%": round(100 * float((v > 0.2).mean()), 1) if v.size else np.nan,
                         "usable_%": round(100 * m.mean(), 1)})
    veg = pd.DataFrame(rows)
    veg.to_csv(RESULTS / "village_ndvi.csv", index=False)

    # --- figure --------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    for name, g in lakes.groupby("lake"):
        ax[0].plot(g.year, g.area_km2, marker="o", label=name)
        bad = g[g["ice_%"] >= 10]
        ax[0].scatter(bad.year, bad.area_km2, s=120, facecolors="none", edgecolors="red", zorder=3)
    ax[0].set_title("Glacier lake area (red ring = >=10% frozen / snow-covered: area too low)")
    ax[0].set_ylabel("km²"); ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)
    for name, g in veg.groupby("place"):
        ref = name.startswith("whole")
        ax[1].plot(g.year, g.ndvi_median, marker="" if ref else ".", lw=3 if ref else 1, color="black" if ref else None,
                   alpha=1 if ref else 0.75, label=name)
    ax[1].set_title("Median NDVI on ice- and snow-free land, village squares (Oct–Nov)")
    ax[1].set_ylabel("NDVI"); ax[1].legend(fontsize=7.5, ncol=2); ax[1].grid(alpha=0.3)
    for a in ax:
        a.set_xticks(range(2016, 2026))
    plt.tight_layout()
    plt.savefig(FIGURES / "F8_lakes_ndvi.png", dpi=200)
    print(f"-> {RESULTS / 'lakes.csv'}, {RESULTS / 'village_ndvi.csv'}, {FIGURES / 'F8_lakes_ndvi.png'}")


if __name__ == "__main__":
    main()
