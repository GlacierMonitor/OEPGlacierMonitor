"""Steps 4-7: export confirmed Sentinel-2 dates, Copernicus DEM, Landsat LST, scene log.

Usage:
    venv/Scripts/python export.py              # everything
    venv/Scripts/python export.py s2 log       # only some steps (s2, dem, landsat, log)
    venv/Scripts/python export.py s2 --years 2020 2023
    venv/Scripts/python export.py s2 --overwrite   # re-export files that already exist

Files that already exist are skipped, so an interrupted run can simply be restarted.

Sentinel-2 dates come from chosen_dates.json (team-confirmed picks), e.g.
    {"2016": "2016-10-24", "2017": {"date": "2017-10-19", "notes": "haze in SE"}}
"""
import argparse
import json

import numpy as np
import pandas as pd
import rasterio
from odc.stac import load

import config as C

LST_SCALE, LST_OFFSET = 0.00341802, 149.0
LS_MAX_CLOUD, LS_MIN_COVER = 30, 90
ALL_STEPS = ["s2", "dem", "landsat", "log"]
LS_PLATFORMS = {"platform": {"in": ["landsat-8", "landsat-9"]}}


def write_tif(path, data, gbox, nodata, names=None):
    """Write a (bands, y, x) array on the given grid as a deflate-compressed GeoTIFF."""
    data = data[None] if data.ndim == 2 else data
    profile = dict(driver="GTiff", width=gbox.shape.x, height=gbox.shape.y, count=data.shape[0],
                   dtype=data.dtype, crs=C.CRS, transform=gbox.transform, nodata=nodata,
                   compress="deflate", predictor=2 if data.dtype.kind in "iu" else 3,
                   tiled=True, blockxsize=512, blockysize=512)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        for i, n in enumerate(names or [], start=1):
            dst.set_band_description(i, n)
    return path


def read_chosen():
    if not C.CHOSEN_FILE.exists():
        raise SystemExit(f"{C.CHOSEN_FILE.name} not found - confirm the picks from search.py first "
                         "(copy chosen_dates.suggested.json and edit it).")
    raw = json.loads(C.CHOSEN_FILE.read_text())
    return {int(y): (v if isinstance(v, dict) else {"date": v}) for y, v in raw.items() if v}


# --- Step 4: Sentinel-2 ----------------------------------------------------
def export_s2(catalog, date, overwrite=False):
    items = C.search(catalog, C.S2_COLLECTION, date, date)
    if len(items) == 0:
        raise RuntimeError(f"no Sentinel-2 items for {date}")
    baselines = sorted({it.properties.get("s2:processing_baseline", "00.00") for it in items})
    path = C.S2_DIR / f"S2_{date.replace('-', '')}.tif"
    if overwrite or not path.exists():
        gbox = C.geobox(10)
        ds, _ = C.load_s2_date(items, C.S2_ALL_BANDS, gbox)
        arr = np.stack([ds[b].values.astype("uint16") for b in C.S2_ALL_BANDS])
        write_tif(path, arr, gbox, nodata=0, names=C.S2_ALL_BANDS)
    return path, baselines, sorted(it.id for it in items)


def run_s2(catalog, years, overwrite=False):
    chosen = read_chosen()
    rows = []
    for y in years:
        if y not in chosen:
            print(f"S2 {y}: no confirmed date - skipped")
            continue
        date = chosen[y]["date"]
        path, bl, ids = C.retry(export_s2, catalog, date, overwrite)
        print(f"S2 {y}: {date} baseline {', '.join(bl)} -> {path.name}")
        rows.append({"year": y, "date": date, "tile_ids": "; ".join(ids),
                     "processing_baseline": ", ".join(bl), "file": path.name})
    return rows


# --- Step 5: DEM -------------------------------------------------------------
def run_dem(catalog, overwrite=False):
    path = C.DEM_DIR / "CopDEM_GLO30_UTM45N.tif"
    if path.exists() and not overwrite:
        with rasterio.open(path) as src:
            v = src.read(1, masked=True)
        print(f"DEM: {path.name} exists; elevation {v.min():.0f} - {v.max():.0f} m")
        return path
    items = catalog.search(collections=["cop-dem-glo-30"], bbox=C.BBOX).item_collection()
    gbox = C.geobox(30)
    dem = C.retry(lambda: load(items, bands=["data"], geobox=gbox, resampling="bilinear",
                               groupby="solar_day").data.max("time").values.astype("float32"))
    nodata = -32767.0
    dem[~np.isfinite(dem)] = nodata
    write_tif(path, dem, gbox, nodata=nodata, names=["elevation_m"])
    v = dem[dem != nodata]
    print(f"DEM: {len(items)} tiles -> {path.name}; elevation {v.min():.0f} - {v.max():.0f} m "
          "(expect roughly 3500-8800)")
    return path


# --- Step 6: Landsat LST ---------------------------------------------------
def ls_candidates(catalog, year):
    items = C.search(catalog, "landsat-c2-l2", *C.season(year), query=LS_PLATFORMS)
    if len(items) == 0:
        return pd.DataFrame()
    gbox = C.geobox(90)
    inside = C.aoi_mask(gbox)
    qa = load(items, bands=["qa_pixel"], geobox=gbox, groupby="solar_day").qa_pixel
    rows = []
    for t, q in zip(qa.time.values, qa.values.astype("uint16")):
        valid = ((q & 1) == 0) & (q != 0) & inside                       # bit 0 = fill
        nv = valid.sum()
        if nv == 0:
            continue
        cloud = ((((q >> 3) & 1) | ((q >> 4) & 1)) == 1) & valid         # bit 3 cloud, 4 shadow
        rows.append({"date": str(t)[:10], "cloud_%": round(100 * cloud.sum() / nv, 1),
                     "cover_%": round(100 * nv / inside.sum(), 1)})
    df = pd.DataFrame(rows)
    return df[df["cover_%"] > LS_MIN_COVER].sort_values("cloud_%") if len(df) else df


def export_lst(catalog, date, overwrite=False):
    items = C.search(catalog, "landsat-c2-l2", date, date, query=LS_PLATFORMS)
    path = C.LST_DIR / f"LST_{date.replace('-', '')}.tif"
    if overwrite or not path.exists():
        gbox = C.geobox(30)
        dn = load(items, bands=["lwir11"], geobox=gbox, groupby="solar_day").lwir11.isel(time=0).values
        kelvin = np.where(dn > 0, dn * LST_SCALE + LST_OFFSET, np.nan).astype("float32")
        write_tif(path, kelvin, gbox, nodata=np.nan, names=["LST_kelvin"])
    return path, sorted(it.id for it in items)


def run_landsat(catalog, years, overwrite=False):
    rows = []
    for y in years:
        df = C.retry(ls_candidates, catalog, y)
        if df.empty:
            print(f"LST {y}: no Landsat 8/9 scene with >{LS_MIN_COVER}% coverage - skipped")
            rows.append({"year": y, "status": "no scene"})
            continue
        best = df.iloc[0]
        if best["cloud_%"] > LS_MAX_CLOUD:
            print(f"LST {y}: clearest {best.date} has {best['cloud_%']}% cloud - skipped")
            rows.append({"year": y, "date": best.date, "cloud_%_aoi": best["cloud_%"],
                         "status": "too cloudy"})
            continue
        path, ids = C.retry(export_lst, catalog, best.date, overwrite)
        print(f"LST {y}: {best.date} cloud {best['cloud_%']}% -> {path.name}")
        rows.append({"year": y, "date": best.date, "cloud_%_aoi": best["cloud_%"],
                     "cover_%": best["cover_%"], "scene_ids": "; ".join(ids),
                     "status": "ok", "file": path.name})
    pd.DataFrame(rows).to_csv(C.LST_DIR / "landsat_log.csv", index=False)
    return rows


# --- Step 7: scene log -------------------------------------------------------
def write_log(s2_rows):
    chosen = read_chosen()
    scan = pd.read_csv(C.OUT / "s2_scan.csv")
    cands = pd.read_csv(C.CANDIDATES_CSV) if C.CANDIDATES_CSV.exists() else pd.DataFrame()
    old = pd.read_csv(C.SCENE_LOG, dtype=str) if C.SCENE_LOG.exists() else pd.DataFrame()
    rows = []
    for r in s2_rows:
        s = scan[scan.date == r["date"]]
        c = cands[cands.date == r["date"]] if len(cands) else cands
        notes = [chosen[r["year"]].get("notes", "")]
        if len(c) and bool(c.iloc[0]["relaxed"]):
            notes.append("relaxed limits (cloud<30%, cover>90%)")
        if len(s):
            notes.append(f"NDSI-corrected cloud {s.iloc[0]['cloud_ndsi_%']}%; "
                         f"SCL snow hint {s.iloc[0]['snow_%_hint']}%")
        prev = old[old.year == str(r["year"])] if len(old) else old
        rows.append({"year": r["year"], "date": r["date"], "tile_ids": r["tile_ids"],
                     "processing_baseline": r["processing_baseline"],
                     "cloud_%_aoi": float(s.iloc[0]["cloud_%"]) if len(s) else None,
                     # keep anything the team already typed in
                     "fresh_snow": prev.iloc[0]["fresh_snow"] if len(prev) and prev.iloc[0]["date"] == r["date"] else "",
                     "notes": "; ".join(n for n in notes if n),
                     "file": r["file"]})
    log = pd.DataFrame(rows)
    if len(old):
        log = pd.concat([old[~old.year.isin(log.year.astype(str))], log.astype({"year": str})])
        log = log.sort_values("year", key=lambda s: s.astype(int))
    log.to_csv(C.SCENE_LOG, index=False)
    print("Scene log ->", C.SCENE_LOG)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("steps", nargs="*", choices=ALL_STEPS, help="default: all steps")
    p.add_argument("--years", type=int, nargs="+", default=list(C.YEARS))
    p.add_argument("--overwrite", action="store_true", help="re-export files that already exist")
    args = p.parse_args()
    args.steps = args.steps or ALL_STEPS

    C.make_dirs()
    catalog = C.open_catalog()
    s2_rows = []
    if "s2" in args.steps or "log" in args.steps:
        if "s2" in args.steps:
            s2_rows = run_s2(catalog, args.years, args.overwrite)
        else:   # log only: rebuild rows from files already on disk
            for y, v in read_chosen().items():
                if y not in args.years:
                    continue
                items = C.search(catalog, C.S2_COLLECTION, v["date"], v["date"])
                s2_rows.append({"year": y, "date": v["date"],
                                "tile_ids": "; ".join(sorted(it.id for it in items)),
                                "processing_baseline": ", ".join(sorted(
                                    {it.properties.get("s2:processing_baseline", "") for it in items})),
                                "file": f"S2_{v['date'].replace('-', '')}.tif"})
    if "dem" in args.steps:
        run_dem(catalog, args.overwrite)
    if "landsat" in args.steps:
        run_landsat(catalog, args.years, args.overwrite)
    if "log" in args.steps and s2_rows:
        write_log(s2_rows)


if __name__ == "__main__":
    main()
