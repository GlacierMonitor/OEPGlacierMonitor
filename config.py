"""Shared settings and helpers for GlacierMonitor Part 1 (data collection).

Fixed project settings live here so search.py, export.py and verify.py all use
the same study area, CRS, grids and output folders.
"""
import os
import time
from pathlib import Path

import numpy as np
import planetary_computer
import pystac_client
from odc.geo.geobox import GeoBox
from odc.geo.geom import box

# Let GDAL retry flaky HTTP reads from Planetary Computer blob storage
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "3")

# --- Fixed settings (do not change) -----------------------------------------
# Study area: Khumbu, Ngozumpa, Lobuche, Changri Nup/Shar, Imja-Lhotse Shar, Ama Dablam
BBOX = (86.60, 27.83, 87.00, 28.12)          # lon_min, lat_min, lon_max, lat_max
CRS = "EPSG:32645"                           # UTM 45N
S2_COLLECTION = "sentinel-2-l2a"
S2_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
S2_ALL_BANDS = S2_BANDS + ["SCL"]
SCL_CLOUD = [3, 8, 9, 10]                    # shadow, cloud medium, cloud high, cirrus
SCL_SNOW = 11
YEARS = range(2016, 2026)
SEASON = ("10-01", "11-30")                  # 1 Oct - 30 Nov
BOA_OFFSET_BASELINE = "04.00"                # +1000 offset from 25 Jan 2022 onwards

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# --- Output folders ---------------------------------------------------------
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "GlacierMonitor" / "raw"
S2_DIR = OUT / "sentinel2"
DEM_DIR = OUT / "dem"
LST_DIR = OUT / "landsat_thermal"
PREVIEW_DIR = OUT / "previews"
CANDIDATES_CSV = OUT / "s2_candidates.csv"
SCENE_LOG = OUT / "scene_log.csv"
CHOSEN_FILE = ROOT / "chosen_dates.json"     # picks confirmed by the team


def make_dirs():
    for d in (S2_DIR, DEM_DIR, LST_DIR, PREVIEW_DIR):
        d.mkdir(parents=True, exist_ok=True)


def open_catalog():
    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def geobox(resolution):
    """Fixed output grid for the study area in UTM 45N.

    Using one explicit GeoBox (instead of bbox= on every load) guarantees that
    every file at a given resolution has identical width, height and transform.
    """
    aoi = box(*BBOX, crs="EPSG:4326").to_crs(CRS)
    return GeoBox.from_geopolygon(aoi, resolution=resolution, crs=CRS)


def retry(fn, *args, tries=3, wait=20, **kwargs):
    """Call fn, retrying on errors (e.g. dropped network reads) with a pause between tries."""
    for attempt in range(1, tries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == tries:
                raise
            print(f"   attempt {attempt} failed ({type(e).__name__}: {str(e)[:120]}) - retrying in {wait}s")
            time.sleep(wait)


def season(year):
    return f"{year}-{SEASON[0]}", f"{year}-{SEASON[1]}"


def search(catalog, collection, start, end, **kwargs):
    return catalog.search(collections=[collection], bbox=BBOX,
                          datetime=f"{start}/{end}", **kwargs).item_collection()


def needs_offset(item):
    """True if the S2 item's reflectance carries the +1000 BOA offset."""
    return item.properties.get("s2:processing_baseline", "00.00") >= BOA_OFFSET_BASELINE


def remove_offset(arr):
    """Subtract 1000 from reflectance DNs; keep 0 as nodata, clip valid to >= 1."""
    v = arr.astype("int32")
    return np.where(v > 0, np.clip(v - 1000, 1, None), 0).astype("uint16")


def aoi_mask(gbox):
    """Boolean mask (y, x) of pixels inside the lon/lat bbox on the given grid."""
    from odc.geo.xr import rasterize
    return rasterize(box(*BBOX, crs="EPSG:4326"), gbox).values.astype(bool)


def load_s2_date(items, bands, gbox):
    """Load one date as a single mosaic with the BOA offset already removed.

    Items are grouped by processing baseline so the offset is removed only
    from tiles that carry it, even if one day mixes old and new baselines.
    Returns (xarray.Dataset without time dim, sorted list of baselines).
    """
    from odc.stac import load

    baselines = sorted({it.properties.get("s2:processing_baseline", "00.00") for it in items})
    groups = [[it for it in items if needs_offset(it)],
              [it for it in items if not needs_offset(it)]]
    mosaic = None
    for offset, group in zip((True, False), groups):
        if not group:
            continue
        ds = load(group, bands=bands, geobox=gbox, groupby="solar_day",
                  resampling="nearest").isel(time=0)
        if offset:
            for b in bands:
                if b != "SCL":
                    ds[b] = (ds[b].dims, remove_offset(ds[b].values))
        if mosaic is None:
            mosaic = ds
        else:                                    # fill remaining nodata gaps
            for b in bands:
                mosaic[b] = mosaic[b].where(mosaic[b] != 0, ds[b])
    return mosaic, baselines
