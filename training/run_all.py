"""Run the experiment matrix: every method, setting and seed is trained, mapped and scored. Resumable.

Usage:
    venv/Scripts/python run_all.py --data "D:/My Drive/GlacierMonitor" --label label_2022.tif \
        --tiles tiles.csv --glaciers glacier_ids.tif --lookup glacier_ids.csv --out runs/matrix
    # see what would run without running it:   ... --dry-run
    # a smaller matrix:                         ... --methods unet --seeds 0 1 --terrain on

--data uses the team Drive layout: raw/sentinel2/S2_YYYYMMDD.tif, raw/dem/CopDEM_GLO30_UTM45N.tif and
masks/mask_YYYYMMDD.tif. A flat folder with all of them at the top (like dummy/) works too. The labelled scene is the S2 file whose year matches the label name (label_2022.tif
-> S2_2022*.tif), or --ref. Reading from a streamed Drive folder is slow (about 1.5 min per scene the first
time): set raw/ and masks/ to "Available offline" in Google Drive first.

Every combination of these runs (each takes a list):
  --methods  unet rf ndsi     ndsi is the NDSI > 0.4 baseline: no training, no seed, no terrain setting
  --seeds    0 1 2 3 4
  --terrain  on off           off = --no-terrain (no elevation and slope)
  --cloud    ocm scl none     ocm = Part 2 masks (--masks), scl = SCL cloud (--mask-cloud), none
  --split    glacier random   random = random_split.py with the run's seed (the leakage control)
  --all-years                 also map every S2 file into <run>/years (for the area time series)

Each run gets <out>/<method>_terrain-<t>_cloud-<c>_split-<s>_seed<n>/ with run.log, model/ (weights, or the
forest), the reference-year map and scores.json from score.py. A step whose output exists is skipped, so a
stopped run carries on where it left off; delete a run folder to redo it. At the end `score.py collect`
writes <out>/results_runs.csv, results_summary.csv (mean, sd and n over seeds) and results_glaciers.csv.
"""
import argparse
import itertools
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sh(cmd, log):
    """Run one step, appending its output to log; stop the matrix on failure."""
    cmd = [str(c) for c in cmd]
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n"); fh.flush()
        rc = subprocess.run(cmd, cwd=HERE, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc:
        raise SystemExit(f"step failed (exit {rc}); see {log}")


def plan(args):
    runs = []
    for method, cloud, split in itertools.product(args.methods, args.cloud, args.split):
        if method == "ndsi":
            seeds = args.seeds if split == "random" else [None]
            runs += [dict(method=method, terrain="na", cloud=cloud, split=split, seed=s) for s in seeds]
        else:
            runs += [dict(method=method, terrain=t, cloud=cloud, split=split, seed=s)
                     for t in args.terrain for s in args.seeds]
    return runs


def name(r):
    seed = "" if r["seed"] is None else f"_seed{r['seed']}"
    return f"{r['method']}_terrain-{r['terrain']}_cloud-{r['cloud']}_split-{r['split']}{seed}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", required=True, help="folder with raw/sentinel2, raw/dem and masks")
    p.add_argument("--label", required=True); p.add_argument("--tiles", required=True)
    p.add_argument("--glaciers", help="glacier_ids.tif, for test_strict and per-glacier scores")
    p.add_argument("--lookup", help="glacier_ids.csv")
    p.add_argument("--ref", help="the labelled S2 file (default: picked from the label's year)")
    p.add_argument("--out", required=True)
    p.add_argument("--methods", nargs="+", default=["unet", "rf", "ndsi"], choices=["unet", "rf", "ndsi"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--terrain", nargs="+", default=["on", "off"], choices=["on", "off"])
    p.add_argument("--cloud", nargs="+", default=["ocm"], choices=["ocm", "scl", "none"])
    p.add_argument("--split", nargs="+", default=["glacier"], choices=["glacier", "random"])
    p.add_argument("--all-years", action="store_true", help="also map every S2 file")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if bool(args.glaciers) != bool(args.lookup):
        raise SystemExit("--glaciers and --lookup go together")

    data = Path(args.data)
    pick = lambda sub: data / sub if (data / sub).exists() else data
    s2_dir, mask_dir = pick("raw/sentinel2"), data / "masks"
    dem = pick("raw/dem") / "CopDEM_GLO30_UTM45N.tif"
    scenes = sorted(s2_dir.glob("S2_*.tif"))
    if args.ref:
        ref = Path(args.ref)
    else:
        year = "".join(ch for ch in Path(args.label).stem if ch.isdigit())[:4]
        hits = [s for s in scenes if s.stem.startswith(f"S2_{year}")]
        if len(hits) != 1:
            raise SystemExit(f"found {len(hits)} S2_{year}*.tif in {s2_dir}; pass --ref")
        ref = hits[0]
    for f in [ref, dem, args.label, args.tiles] + ([args.glaciers, args.lookup] if args.glaciers else []):
        if not Path(f).exists():
            raise SystemExit(f"missing {f}")
    if "ocm" in args.cloud and not mask_dir.is_dir():
        raise SystemExit(f"--cloud ocm needs {mask_dir}")
    label, tiles = Path(args.label).resolve(), Path(args.tiles).resolve()
    glaciers = [Path(args.glaciers).resolve(), Path(args.lookup).resolve()] if args.glaciers else []

    runs = plan(args)
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    print(f"{len(runs)} runs -> {out}  (reference scene {ref.name}, {len(scenes)} scenes)")
    if args.dry_run:
        for r in runs:
            print(" ", name(r))
        return

    py = sys.executable
    t_all = time.time()
    for i, r in enumerate(runs, 1):
        d = out / name(r); d.mkdir(exist_ok=True)
        log = d / "run.log"
        if (d / "scores.json").exists() and (not args.all_years or r["method"] == "ndsi"
                                            or (d / "years" / "areas.csv").exists()
                                            or (d / "model" / "years.done").exists()):
            print(f"[{i}/{len(runs)}] {d.name}: done, skipped")
            continue
        t0 = time.time()
        seed = 0 if r["seed"] is None else r["seed"]
        cloud = {"ocm": ["--masks", mask_dir], "scl": ["--mask-cloud"], "none": []}[r["cloud"]]
        terrain = ["--no-terrain"] if r["terrain"] == "off" else []

        run_tiles = tiles
        if r["split"] == "random":
            run_tiles = d / "tiles.csv"
            if not run_tiles.exists():
                sh([py, "random_split.py", "--tiles", tiles, "--seed", seed, "--out", run_tiles], log)

        model = d / "model"
        if r["method"] == "unet":
            if not (model / "summary.json").exists():
                sh([py, "train.py", "--s2", ref, "--dem", dem, "--label", label, "--tiles", run_tiles,
                    "--out", model, "--seed", seed, "--epochs", args.epochs, "--patience", args.patience,
                    *terrain, *cloud], log)
            ref_map = d / "ref" / f"{ref.stem}_mask.tif"
            masks_arg = cloud if r["cloud"] == "ocm" else []      # scl is stored in best.pt
            if not (d / "ref" / "areas.csv").exists():
                sh([py, "predict.py", "--model", model / "best.pt", "--dem", dem, "--s2", ref,
                    "--out", d / "ref", *masks_arg], log)
            if args.all_years and not (d / "years" / "areas.csv").exists():
                sh([py, "predict.py", "--model", model / "best.pt", "--dem", dem, "--s2", *scenes,
                    "--out", d / "years", *masks_arg], log)
        elif r["method"] == "rf":
            done = model / ("years.done" if args.all_years else "scores.json")
            if not done.exists():
                extra = ["--predict", *scenes] if args.all_years else []
                sh([py, "rf_baseline.py", "--s2", ref, "--dem", dem, "--label", label, "--tiles", run_tiles,
                    "--out", model, "--seed", seed, *terrain, *cloud, *extra], log)
                if args.all_years:
                    done.write_text("all S2 files mapped\n")
            ref_map = model / f"{ref.stem}_mask.tif"
        else:
            ref_map = None

        what = ["--pred", ref_map] if ref_map else ["--ndsi", "0.4"]
        score_cloud = {"ocm": ["--masks", mask_dir], "scl": ["--mask-cloud"], "none": []}[r["cloud"]]
        g = ["--glaciers", glaciers[0], "--lookup", glaciers[1]] if glaciers else []
        tags = [f"{k}={'na' if v is None else v}" for k, v in r.items()]
        sh([py, "score.py", "run", *what, "--s2", ref, "--dem", dem, "--label", label, "--tiles", run_tiles,
            *score_cloud, *g, "--tag", *tags, "--out", d / "scores.json"], log)
        print(f"[{i}/{len(runs)}] {d.name}: {time.time() - t0:.0f}s")

    sh([py, "score.py", "collect", out, "--out", out / "results"], out / "collect.log")
    print((out / "collect.log").read_text(encoding="utf-8").split("\n", 2)[-1])
    print(f"all {len(runs)} runs in {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
