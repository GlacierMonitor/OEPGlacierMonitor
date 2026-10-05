"""Checks on exported files: CRS, resolution, identical S2 grid, band counts,
offset consistency across processing baselines, and disk use.

Usage:
    venv/Scripts/python verify.py
"""
import numpy as np
import rasterio

import config as C

EXPECT = {"sentinel2": (10, 7), "dem": (30, 1), "landsat_thermal": (30, 1)}


def check_file(path, res, bands):
    problems = []
    with rasterio.open(path) as src:
        if src.crs.to_epsg() != 32645:
            problems.append(f"CRS {src.crs}")
        if tuple(round(abs(r), 6) for r in src.res) != (res, res):
            problems.append(f"resolution {src.res}")
        if src.count != bands:
            problems.append(f"{src.count} bands (expected {bands})")
        grid = (src.width, src.height, tuple(src.transform)[:6])
        desc = src.descriptions
    return problems, grid, desc


def reflectance_stats(path):
    """Median and 5-95th percentile of valid, clear, non-snow B02-B12 pixels."""
    with rasterio.open(path) as src:
        a = src.read(out_shape=(src.count, src.height // 4, src.width // 4))   # 40 m overview is enough
    scl = a[6]
    clear = np.isin(scl, [4, 5])                         # vegetation, not-vegetated (rock/debris)
    return {b: np.percentile(a[i][clear & (a[i] > 0)], [5, 50, 95]).round().astype(int).tolist()
            for i, b in enumerate(C.S2_BANDS)}


def main():
    ok = True
    for sub, (res, bands) in EXPECT.items():
        grids = {}
        for f in sorted((C.OUT / sub).glob("*.tif")):
            problems, grid, desc = check_file(f, res, bands)
            grids.setdefault(grid, []).append(f.name)
            status = "OK" if not problems else "FAIL: " + "; ".join(problems)
            ok &= not problems
            print(f"{sub:16s} {f.name:28s} {grid[0]}x{grid[1]}  bands={desc}  {status}")
        if sub == "sentinel2" and grids:
            same = len(grids) == 1
            ok &= same
            print(f"-> all {sum(map(len, grids.values()))} S2 files on the same grid: {same}")
            if not same:
                for g, names in grids.items():
                    print("   ", g, names)

    s2 = {f.stem[3:7]: f for f in sorted(C.S2_DIR.glob("S2_*.tif"))}
    print("\nReflectance (5th / median / 95th pct, clear non-snow pixels) - pre vs post 2022 baseline:")
    for yr in ("2020", "2023"):
        if yr in s2:
            print(f"  {s2[yr].name}: {reflectance_stats(s2[yr])}")
        else:
            print(f"  {yr}: no file")

    total = sum(f.stat().st_size for f in C.OUT.rglob("*") if f.is_file())
    print(f"\nDisk use of {C.OUT}: {total / 1e6:.1f} MB")
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
