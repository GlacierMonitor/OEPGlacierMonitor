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


def read_small(path):
    with rasterio.open(path) as src:
        return src.read(out_shape=(src.count, src.height // 4, src.width // 4))   # 40 m is enough


def compare_offset(old_path, new_path):
    """Median reflectance on pixels that are clear (SCL 4/5) in BOTH images.

    Using the same pixels avoids comparing different surfaces. If the +1000
    offset was not removed from the newer image, the difference is ~+1000.
    """
    a, b = read_small(old_path), read_small(new_path)
    m = np.isin(a[6], [4, 5]) & np.isin(b[6], [4, 5]) & (a[0] > 0) & (b[0] > 0)
    print(f"  {m.sum()} pixels clear in both {old_path.name} and {new_path.name}")
    ok = True
    for i, band in enumerate(C.S2_BANDS):
        oa, nb = int(np.median(a[i][m])), int(np.median(b[i][m]))
        diff = int(np.median(b[i][m].astype(int) - a[i][m].astype(int)))
        ok &= abs(diff) < 300
        print(f"  {band}: {oa:5d} vs {nb:5d}  median diff {diff:+d}")
    return ok


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
    print("\nOffset check - 2020 (baseline < 04.00) vs 2023 (baseline >= 04.00):")
    if "2020" in s2 and "2023" in s2:
        same = compare_offset(s2["2020"], s2["2023"])
        ok &= same
        print(f"-> reflectance in a similar range (|diff| < 300 DN): {same}")
    else:
        print("  2020 or 2023 file missing - skipped")

    total = sum(f.stat().st_size for f in C.OUT.rglob("*") if f.is_file())
    print(f"\nDisk use of {C.OUT}: {total / 1e6:.1f} MB")
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
