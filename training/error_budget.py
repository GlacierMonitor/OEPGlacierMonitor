"""Error budget of the per-glacier area time series: how much of the year-to-year change is real.

Usage (after run_all.py --all-years):
    venv/Scripts/python error_budget.py --runs runs/matrix --glaciers glacier_ids.tif --lookup glacier_ids.csv \
        --out runs/matrix/budget --exclude-years 2025

Every run of the matrix is one way of making the maps (method, terrain on/off, cloud mask, split,
preprocessing A/B, seed). The spread between runs, for the same glacier and year, is the error the choice
of pipeline brings in. It is split into:
  - seed: same setting, different training seed (sd within settings, pooled over settings);
  - one sd per factor that varies in the matrix (method, terrain, cloud, split, preprocessing): the
    seed-averaged area changed by that factor alone, the others held fixed, pooled over the fixed settings.
    NDSI has no seed and no terrain setting; it is compared with the models at every terrain setting.
  - total: sd over all runs.
For a change between two years the same is done on the change itself, so errors that repeat every year
(a method that always misses the same debris) cancel. rho is the correlation of the run errors between the
two years: near 1 means the errors repeat and change is easier to see than the per-year spread suggests.
MDC95 (minimum detectable change) = 1.96 x the sd of the change; a change smaller than that is noise.
The decade trend is a least-squares slope per run over the years kept (--exclude-years leaves out, for
example, the snow-covered 2025 scene), then the mean and sd over runs.

Not in this budget: label error, and any bias that every run shares (all runs trained on the same label).
Label error comes from agreement.py (the blind redraw); a shared bias needs the 2016 test squares.
Off-glacier ground (further than --buffer pixels from any outline) should map as no glacier in every year;
its mapped area is written as a control. "Any outline" means every inventory glacier, not only the study
glaciers: give --all-glaciers the full glacier_ids.py raster when --glaciers holds only the study glaciers
(make_labels.py ids writes the 20 study glaciers only), or the other glaciers count as off-glacier ground.

Writes <out>_areas.csv (run, year, glacier, area), <out>_budget.csv (glacier-year sds), <out>_change.csv
(year pairs: consecutive years and first-last), <out>_trend.csv and <out>_offglacier.csv.
"""
import argparse
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

STEM = re.compile(r"^(S2GEE_(\d{4})|S2_(\d{4})\d{4})_mask$")
FACTORS = ["method", "terrain", "cloud", "split", "preprocessing"]
ALL = "ALL study glaciers"


def year_maps(run):
    """{(year, preprocessing): mask path} for one run folder (U-Net/NDSI in years/, RF in model/)."""
    out = {}
    for sub in ("years", "model"):
        for f in sorted((run / sub).glob("*_mask.tif")):
            m = STEM.match(f.stem)
            if m:
                out[(int(m.group(2) or m.group(3)), "B" if m.group(2) else "A")] = f
    return out


def read_areas(args, study):
    with rasterio.open(args.glaciers) as src:
        ids = src.read(1).astype("int64")
        grid = (src.crs, src.transform, ids.shape)
        px_km2 = abs(src.transform.a * src.transform.e) / 1e6
    n = int(ids.max()) + 1
    pixels = np.bincount(ids.ravel(), minlength=n)
    if args.all_glaciers:
        with rasterio.open(args.all_glaciers) as src:
            if not (src.crs == grid[0] and src.transform.almost_equals(grid[1]) and src.shape == grid[2]):
                raise SystemExit(f"{args.all_glaciers} is not on the glacier-ID grid")
            any_ice = src.read(1) > 0
    else:
        any_ice = ids > 0
        if len(np.unique(ids)) - 1 < 50:
            print(f"  warning: {args.glaciers} holds only {len(np.unique(ids)) - 1} glaciers; the off-glacier "
                  "control will include the other glaciers - give --all-glaciers the full inventory raster")
    far = ndimage.distance_transform_edt(~any_ice) > args.buffer          # off-glacier control ground
    rows, off = [], []
    runs = sorted(p.parent for p in Path(args.runs).rglob("scores.json"))
    for run in runs:
        s = json.loads((run / "scores.json").read_text())
        if "config" not in s:
            continue
        cfg = {k: s["config"].get(k, "na") for k in ("method", "terrain", "cloud", "split", "seed")}
        maps = year_maps(run)
        if not maps:
            print(f"  {run.name}: no year maps (run_all.py --all-years), skipped")
            continue
        for (year, prep), f in maps.items():
            with rasterio.open(f) as src:
                mask = src.read(1)
                if not (src.crs == grid[0] and src.transform.almost_equals(grid[1]) and mask.shape == grid[2]):
                    raise SystemExit(f"{f} is not on the glacier-ID grid")
            g = np.bincount(ids.ravel(), weights=(mask == 1).ravel(), minlength=n)
            nd = np.bincount(ids.ravel(), weights=(mask == 255).ravel(), minlength=n)
            base = {"run": run.name, **cfg, "preprocessing": prep, "year": year}
            for v, name in study.items():
                gap = 100 * nd[v] / pixels[v]
                rows.append({**base, "value": v, "glacier": name, "nodata_pct": round(gap, 2),
                             "area_km2": g[v] * px_km2 if gap <= args.max_nodata else np.nan})
            scored = far & (mask != 255)
            off.append({**base, "off_glacier_km2": round(float(((mask == 1) & far).sum()) * px_km2, 3),
                        "off_glacier_scored_km2": round(float(scored.sum()) * px_km2, 1)})
    if not rows:
        raise SystemExit(f"no runs with year maps under {args.runs}")
    areas = pd.DataFrame(rows)
    # the sum over study glaciers, only for run-years where every study glacier has an area
    tot = areas.groupby(["run", "year", "preprocessing"]).agg(
        area_km2=("area_km2", lambda a: a.sum() if a.notna().all() else np.nan),
        nodata_pct=("nodata_pct", "max")).reset_index()
    tot = tot.merge(areas.drop_duplicates("run")[["run", "method", "terrain", "cloud", "split", "seed"]], on="run")
    tot["value"], tot["glacier"] = 0, ALL
    return pd.concat([areas, tot], ignore_index=True), pd.DataFrame(off)


def expand_na(df, col, levels):
    """Rows with 'na' in col (NDSI has no terrain setting) are copied to every level of col."""
    na = df[df[col] == "na"]
    if na.empty or not levels:
        return df
    copies = [na.assign(**{col: lv}) for lv in levels]
    return pd.concat([df[df[col] != "na"], *copies], ignore_index=True)


def spread(df, value, factors):
    """{'seed': sd, <factor>: sd, 'total': sd, 'n': runs} for one glacier-year (or year pair)."""
    d = df.dropna(subset=[value])
    out = {"n": len(d), "mean": d[value].mean(), "total": d[value].std(ddof=1) if len(d) > 1 else np.nan}
    if not factors:                       # a single setting: all the spread is seed
        out["seed"] = out["total"]
        return out
    var = d.groupby(factors, dropna=False)[value].var(ddof=1).dropna()
    out["seed"] = float(np.sqrt(var.mean())) if len(var) else np.nan
    means = d.groupby(factors, dropna=False)[value].mean().reset_index()     # seed-averaged settings
    for f in factors:
        m = means
        for other in factors:
            if other != f:
                m = expand_na(m, other, sorted(set(m[other]) - {"na"}))
        m = m[m[f] != "na"]
        rest = [o for o in factors if o != f]
        groups = m.groupby(rest, dropna=False)[value] if rest else [(None, m[value])]
        v = [s.var(ddof=1) for _, s in groups if len(s) > 1]
        v = [x for x in v if np.isfinite(x)]
        out[f] = float(np.sqrt(np.mean(v))) if v else np.nan
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--runs", required=True, help="run_all.py --out folder (run folders with scores.json)")
    p.add_argument("--glaciers", required=True); p.add_argument("--lookup", required=True)
    p.add_argument("--values", nargs="*", type=int, help="study glaciers by value (fid) in the glacier-ID raster")
    p.add_argument("--names", nargs="*", help="study glaciers by name (default: named, >= --min-km2)")
    p.add_argument("--min-km2", type=float, default=1.0, help="smallest outline for the default study list")
    p.add_argument("--exclude-years", nargs="*", type=int, default=[], help="left out of change and trend")
    p.add_argument("--max-nodata", type=float, default=5.0, help="%% no data above which an area is left empty")
    p.add_argument("--buffer", type=int, default=30, help="off-glacier control: pixels from any outline")
    p.add_argument("--all-glaciers", help="glacier-ID raster of every inventory glacier (glacier_ids.py), for the "
                   "off-glacier control; default: --glaciers")
    p.add_argument("--out", required=True, help="output path prefix")
    args = p.parse_args()

    lk = pd.read_csv(args.lookup, keep_default_na=False)
    named = lk[lk.glacier_name.str.strip() != ""]
    if args.values:
        named = lk[lk.value.isin(args.values)].copy()
        missing = set(args.values) - set(named.value)
        if missing:
            raise SystemExit(f"not in {args.lookup}: {sorted(missing)}")
        named["glacier_name"] = [n.strip() or f"fid {v}" for v, n in zip(named.value, named.glacier_name)]
    elif args.names:
        want = {n.lower() for n in args.names}
        named = named[named.glacier_name.str.lower().isin(want)]
        missing = want - set(named.glacier_name.str.lower())
        if missing:
            raise SystemExit(f"not in {args.lookup}: {sorted(missing)}")
    elif "pixels" in named:
        named = named[named.pixels.astype(float) * 1e-4 >= args.min_km2]
    study = dict(zip(named.value.astype(int), named.glacier_name))
    print(f"{len(study)} study glaciers: {', '.join(sorted(study.values()))}")

    areas, off = read_areas(args, study)
    factors = [f for f in FACTORS if areas[f].nunique() > 1]
    print(f"{areas.run.nunique()} runs; factors that vary: {', '.join(factors) or 'none'}; "
          f"seeds per setting: up to {areas.groupby(factors or ['method']).seed.nunique().max()}")
    cols = ["seed"] + factors

    budget = []
    for (glacier, year), d in areas.groupby(["glacier", "year"]):
        s = spread(d, "area_km2", factors)
        budget.append({"glacier": glacier, "year": year, **{k: s[k] for k in ("n", "mean", "total", *cols)}})
    budget = pd.DataFrame(budget).rename(columns={"mean": "mean_km2", "total": "sd_total",
                                                  **{c: f"sd_{c}" for c in cols}})
    budget["sd_total_pct"] = 100 * budget.sd_total / budget.mean_km2

    years = sorted(set(areas.year) - set(args.exclude_years))
    pairs = list(zip(years, years[1:])) + ([(years[0], years[-1])] if len(years) > 2 else [])
    wide = areas[areas.year.isin(years)].pivot_table(index=["run", "glacier"] + FACTORS + ["seed"],
                                                     columns="year", values="area_km2").reset_index()
    change = []
    for glacier, d in wide.groupby("glacier"):
        for y1, y2 in pairs:
            if y1 not in d or y2 not in d:
                continue
            dd = d.assign(delta=d[y2] - d[y1])
            s = spread(dd, "delta", factors)
            ok = dd[[y1, y2]].dropna()
            rho = np.corrcoef(ok[y1], ok[y2])[0, 1] if len(ok) > 2 and ok.std().min() > 0 else np.nan
            change.append({"glacier": glacier, "y1": y1, "y2": y2, "n": s["n"], "mean_delta_km2": s["mean"],
                           "sd_delta_total": s["total"], **{f"sd_delta_{c}": s[c] for c in cols},
                           "rho_years": rho, "mdc95_km2": 1.96 * s["total"], "mdc95_seed_km2": 1.96 * s["seed"]})
    change = pd.DataFrame(change)
    change["detectable"] = change.mean_delta_km2.abs() > change.mdc95_km2
    mean_area = budget.groupby("glacier").mean_km2.mean()
    change["mdc95_pct"] = 100 * change.mdc95_km2 / change.glacier.map(mean_area)

    trend = []
    yr = np.array(years, float)
    for glacier, d in wide.groupby("glacier"):
        a = d.reindex(columns=years).to_numpy(float)
        keep = np.isfinite(a).sum(1) >= 3
        slopes = [np.polyfit(yr[np.isfinite(r)], r[np.isfinite(r)], 1)[0] for r in a[keep]]
        if not slopes:
            continue
        m, sd = float(np.mean(slopes)), float(np.std(slopes, ddof=1)) if len(slopes) > 1 else np.nan
        trend.append({"glacier": glacier, "runs": len(slopes), "years": f"{years[0]}-{years[-1]}",
                      "mean_km2": mean_area[glacier], "slope_km2_yr": m, "slope_sd_km2_yr": sd,
                      "slope_pct_yr": 100 * m / mean_area[glacier], "slope_sd_pct_yr": 100 * sd / mean_area[glacier],
                      "t": m / sd if sd else np.nan})
    trend = pd.DataFrame(trend)

    base = Path(args.out); base.parent.mkdir(parents=True, exist_ok=True)
    areas.to_csv(f"{base}_areas.csv", index=False)
    budget.round(5).to_csv(f"{base}_budget.csv", index=False)
    change.round(5).to_csv(f"{base}_change.csv", index=False)
    trend.round(5).to_csv(f"{base}_trend.csv", index=False)
    off.to_csv(f"{base}_offglacier.csv", index=False)

    # summary: per glacier, the median over years of each sd as % of area, and the trend
    pct = budget.assign(**{c: 100 * budget[f"sd_{c}"] / budget.mean_km2 for c in ["total"] + cols})
    summ = pct.groupby("glacier")[["mean_km2", "total"] + cols].median()
    summ.columns = ["area_km2"] + [f"sd_{c}_%" for c in ["total"] + cols]
    cons = change[change.y2 - change.y1 == 1].groupby("glacier")
    summ["mdc95_1yr_%"] = cons.mdc95_pct.median()
    summ["rho_1yr"] = cons.rho_years.median()
    if len(trend):
        summ = summ.join(trend.set_index("glacier")[["slope_pct_yr", "slope_sd_pct_yr"]])
    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print("\nmedian over years, % of glacier area (sd = spread between runs):")
        print(summ.sort_values("area_km2", ascending=False).to_string())
        o = off.groupby("year").off_glacier_km2.agg(["mean", "std"])
        print(f"\noff-glacier control (> {args.buffer} px from any outline), mapped glacier km2, mean over runs:")
        print(o.T.to_string())
    print(f"-> {base}_areas.csv, _budget.csv, _change.csv, _trend.csv, _offglacier.csv")


if __name__ == "__main__":
    main()
