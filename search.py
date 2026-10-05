"""Step 1-3: list Sentinel-2 candidate dates per year and save previews.

Usage:
    venv/Scripts/python search.py                        # all years, SCL cloud rule
    venv/Scripts/python search.py --years 2019 2020
    venv/Scripts/python search.py --metric scl_ndsi      # snow-corrected cloud %
    venv/Scripts/python search.py --rescan               # ignore cached scan
    venv/Scripts/python search.py --metric scl_ndsi --max-cloud 100
                                                         # rank only, no cloud cut-off

Two cloud metrics are computed for every date (both inside the bbox only):
    cloud_%        SCL classes 3, 8, 9, 10 (project rule)
    cloud_ndsi_%   same, but SCL-"cloud" pixels with NDSI > 0.4 are counted
                   as snow/ice, not cloud. Sen2Cor often labels high-altitude
                   snow as class 8/9, which inflates cloud_% in this area.

Writes:
    GlacierMonitor/raw/s2_scan.csv               every date, all metrics (cache)
    GlacierMonitor/raw/s2_candidates.csv         filtered candidates + suggested pick
    GlacierMonitor/raw/previews/YYYY-MM-DD.png   top 3 per year (true colour | SWIR)
    chosen_dates.suggested.json                  suggested picks; copy to
                                                 chosen_dates.json once confirmed
"""
import argparse
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C

STRICT = dict(max_cloud=20, min_cover=95)
RELAXED = dict(max_cloud=30, min_cover=90)
NDSI_SNOW = 0.4
SNOW_JUMP = 10          # snow-% points above the year's median = "big jump" (likely fresh snow)
N_PREVIEW = 3
SCAN_CSV = C.OUT / "s2_scan.csv"
METRIC_COL = {"scl": "cloud_%", "scl_ndsi": "cloud_ndsi_%"}


def by_date(items):
    groups = defaultdict(list)
    for it in items:
        groups[it.datetime.date().isoformat()].append(it)
    return dict(sorted(groups.items()))


def scan_year(catalog, year):
    """Coverage, cloud and snow % inside the bbox for every date in the season."""
    gbox = C.geobox(60)
    inside = C.aoi_mask(gbox)
    rows = []
    for date, items in by_date(C.search(catalog, C.S2_COLLECTION, *C.season(year))).items():
        ds, baselines = C.load_s2_date(items, ["SCL", "B03", "B11"], gbox)
        scl = ds.SCL.values
        g, s = ds.B03.values.astype(float), ds.B11.values.astype(float)
        ndsi = np.divide(g - s, g + s, out=np.zeros_like(g), where=(g + s) > 0)
        valid = (scl > 0) & inside
        nv = valid.sum()
        if nv == 0:
            continue
        cloud = np.isin(scl, C.SCL_CLOUD) & valid
        rows.append({
            "year": year, "date": date,
            "cloud_%": round(100 * cloud.sum() / nv, 1),
            "cloud_ndsi_%": round(100 * (cloud & (ndsi <= NDSI_SNOW)).sum() / nv, 1),
            "cover_%": round(100 * nv / inside.sum(), 1),
            "snow_%_hint": round(100 * ((scl == C.SCL_SNOW) & valid).sum() / nv, 1),
            "tiles": len(items), "baseline": ", ".join(baselines),
        })
        print(f"   scanned {date}: cloud {rows[-1]['cloud_%']}% / "
              f"ndsi-corrected {rows[-1]['cloud_ndsi_%']}%")
    return pd.DataFrame(rows)


def load_scan(catalog, years, rescan):
    old = pd.read_csv(SCAN_CSV) if SCAN_CSV.exists() and not rescan else pd.DataFrame(columns=["year"])
    todo = [y for y in years if y not in set(old.year)]
    for y in todo:
        print(f"Scanning {y} ...")
        old = pd.concat([old[old.year != y], scan_year(catalog, y)], ignore_index=True)
        old.sort_values(["year", "date"]).to_csv(SCAN_CSV, index=False)   # save as we go
    return old[old.year.isin(years)]


def candidates(scan, cloud_col, max_cloud, min_cover):
    df = scan[(scan["cover_%"] > min_cover) & (scan[cloud_col] < max_cloud)].copy()
    df["month"] = df.date.str[5:7]
    return df.sort_values(["month", cloud_col]).drop(columns="month").reset_index(drop=True)


def suggest(df, cloud_col):
    """October first, lowest cloud, and no big snow-% jump vs. the year's other dates."""
    if df.empty:
        return None
    ok = df[df["snow_%_hint"] <= df["snow_%_hint"].median() + SNOW_JUMP]
    pool = ok if len(ok) else df
    octo = pool[pool.date.str[5:7] == "10"]
    return (octo if len(octo) else pool).sort_values(cloud_col).iloc[0].date


def stretch(a, lo, hi):
    return np.clip((a.astype(float) - lo) / (hi - lo), 0, 1)


def preview(catalog, date, row=None):
    items = C.search(catalog, C.S2_COLLECTION, date, date)
    ds, baselines = C.load_s2_date(items, ["B02", "B03", "B04", "B08", "B11"], C.geobox(60))
    rgb = np.dstack([stretch(ds[b].values, 0, 4000) for b in ["B04", "B03", "B02"]])
    swir = np.dstack([stretch(ds[b].values, 0, 5000) for b in ["B11", "B08", "B04"]])
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    ax[0].imshow(rgb)
    ax[0].set_title(f"{date} true colour (B04-B03-B02)")
    ax[1].imshow(swir)
    ax[1].set_title(f"{date} SWIR (B11-B08-B04) - snow/ice = cyan")
    for a in ax:
        a.axis("off")
    if row is not None:
        fig.suptitle(f"cloud (SCL) {row['cloud_%']}%   cloud (NDSI-corrected) {row['cloud_ndsi_%']}%   "
                     f"cover {row['cover_%']}%   snow hint {row['snow_%_hint']}%   "
                     f"baseline {', '.join(baselines)}")
    fig.tight_layout()
    path = C.PREVIEW_DIR / f"{date}.png"
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--years", type=int, nargs="+", default=list(C.YEARS))
    p.add_argument("--metric", choices=METRIC_COL, default="scl")
    p.add_argument("--max-cloud", type=float, default=STRICT["max_cloud"],
                   help="cloud %% limit for the first pass (default 20; relaxed pass uses max(this, 30))")
    p.add_argument("--rescan", action="store_true", help="re-download instead of using s2_scan.csv")
    p.add_argument("--no-previews", action="store_true")
    args = p.parse_args()
    col = METRIC_COL[args.metric]

    C.make_dirs()
    catalog = C.open_catalog()
    scan = load_scan(catalog, args.years, args.rescan)
    frames, picks = [], {}
    for y in args.years:
        ys = scan[scan.year == y]
        strict = dict(STRICT, max_cloud=args.max_cloud)
        relax = dict(RELAXED, max_cloud=max(args.max_cloud, RELAXED["max_cloud"]))
        df, relaxed = candidates(ys, col, **strict), False
        if df.empty:
            print(f"{y}: no candidates at {col}<{strict['max_cloud']:g}% / cover>95% - "
                  f"retrying <{relax['max_cloud']:g}% / >90%")
            df, relaxed = candidates(ys, col, **relax), True
        pick = suggest(df, col)
        picks[y] = pick
        print(f"\n===== {y}: {len(df)} candidates{' (RELAXED limits)' if relaxed else ''}, "
              f"suggested pick: {pick} =====")
        if df.empty:
            print("none - needs manual handling (try Sep / early Dec)")
            continue
        df["relaxed"] = relaxed
        df["rank"] = range(1, len(df) + 1)
        df["suggested"] = df.date == pick
        print(df.drop(columns="year").to_string(index=False))
        frames.append(df)
        if not args.no_previews:
            for d in dict.fromkeys(list(df.head(N_PREVIEW).date) + [pick]):
                print("   preview ->", preview(catalog, d, df[df.date == d].iloc[0]).name)

    if frames:
        out = pd.concat(frames, ignore_index=True)
        out.insert(1, "metric", args.metric)
        if C.CANDIDATES_CSV.exists() and set(args.years) != set(C.YEARS):
            old = pd.read_csv(C.CANDIDATES_CSV)
            out = pd.concat([old[~old.year.isin(args.years)], out]).sort_values(["year", "rank"])
        out.to_csv(C.CANDIDATES_CSV, index=False)
        print("\nCandidates ->", C.CANDIDATES_CSV)
    sug = C.ROOT / "chosen_dates.suggested.json"
    existing = json.loads(sug.read_text()) if sug.exists() else {}
    existing.update({str(y): d for y, d in picks.items()})
    sug.write_text(json.dumps(dict(sorted(existing.items())), indent=2))
    print("Suggested picks ->", sug)


if __name__ == "__main__":
    main()
