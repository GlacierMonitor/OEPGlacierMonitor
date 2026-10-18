"""GlacierMonitor Part 3: training squares, the label drawing file, and label rasters.

  venv\\Scripts\\python make_labels.py setup                     # once: tiles.csv + labels_2022.gpkg
  venv\\Scripts\\python make_labels.py setup --year 2016 --from 2022   # copy a finished drawing to another year
  venv\\Scripts\\python make_labels.py build --year 2022         # after drawing: label_2022.tif

setup writes, in GlacierMonitor/labels/:
  tiles.csv              row, col, split: top-left pixel of each 256 x 256 square (Kavya's training format)
  labels_<year>.gpkg     the drawing file, with three layers:
                           glaciers  copy of inventory/glaciers.gpkg (or of another year's drawing): edit to match the image
                           unsure    empty: draw areas you cannot decide (they become 255)
                           squares   the 48 squares, read-only guide for where to work
build writes label_<year>.tif on the year's S2 grid: 1 glacier, 0 not glacier, 255 ignore.
255 = outside the squares, 'unsure' polygons, S2 no data, and cloud / cloud shadow from masks/mask_<date>.tif.
Terrain shadow stays labelled (shadowed ice is still glacier).
"""
import argparse
import shutil
import sqlite3
import struct
import sys

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from shapely import wkb
from shapely.geometry import MultiPolygon, box

import config as C

LABEL_DIR = C.ROOT / "GlacierMonitor" / "labels"
INVENTORY = C.ROOT / "GlacierMonitor" / "inventory" / "glaciers.gpkg"
MASK_DIR = C.ROOT / "GlacierMonitor" / "masks"
TILES_CSV = LABEL_DIR / "tiles.csv"
TILE, IGNORE, SRS = 256, 255, 32645

# Squares agreed 2026-10-18 (decisions.md 18), as (tile row, tile col); pixel offset = index * 256.
# Split by glacier: test = all of Khumbu + Imja/Lhotse Shar, val = Changri Shar/Nup + Lobuche,
# train = Ngozumpa, Ama Dablam and squares picked for off-glacier snow, lakes, shadow, debris, rock.
SQUARES = {
    "test": [(5, 9), (5, 10), (5, 11), (6, 8), (6, 9), (6, 10), (6, 11), (6, 12), (7, 8), (7, 13), (7, 14),
             (8, 13), (9, 12), (9, 13), (10, 13)],
    "val": [(4, 7), (5, 7), (5, 8), (6, 6), (6, 7), (7, 7)],
    "train": [(0, 2), (0, 5), (1, 2), (1, 3), (1, 4), (1, 5), (2, 2), (2, 3), (2, 4), (2, 5), (2, 12), (3, 3),
              (3, 4), (3, 5), (3, 13), (4, 12), (4, 13), (5, 3), (5, 12), (7, 0), (7, 10), (9, 1), (9, 10),
              (10, 3), (10, 10), (11, 10), (11, 12)],
}

LABEL_QML = """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.44" styleCategories="Symbology">
 <pipe>
  <rasterrenderer type="paletted" band="1" opacity="1" alphaBand="-1" nodataColor="">
   <rasterTransparency/>
   <colorPalette>
    <paletteEntry value="0" color="#8c5a3c" alpha="110" label="not glacier"/>
    <paletteEntry value="1" color="#00e5ff" alpha="150" label="glacier"/>
   </colorPalette>
  </rasterrenderer>
 </pipe>
</qgis>
"""


# --- GeoPackage helpers (plain sqlite, no GDAL needed) ----------------------
def gpkg_geom(blob):
    """GeoPackage geometry blob -> shapely (skips the GP header and envelope)."""
    env = (blob[3] >> 1) & 7
    return wkb.loads(bytes(blob[8 + {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[env]:]))


def gpkg_blob(geom):
    """shapely -> GeoPackage geometry blob (little endian, xy envelope)."""
    minx, miny, maxx, maxy = geom.bounds
    return b"GP" + bytes([0, 0b011]) + struct.pack("<i4d", SRS, minx, maxx, miny, maxy) + wkb.dumps(geom)


def read_layer(path, table):
    con = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        return [gpkg_geom(g) for (g,) in con.execute(f'select geom from "{table}" where geom is not null')]
    finally:
        con.close()


def add_layer(con, table, columns, rows):
    """Create a MultiPolygon feature table registered in the GeoPackage, and insert rows (geom, *values)."""
    cols = "".join(f", {n} {t}" for n, t in columns)
    con.execute(f'create table "{table}" (fid INTEGER PRIMARY KEY AUTOINCREMENT, geom MULTIPOLYGON{cols})')
    con.execute("insert into gpkg_contents (table_name, data_type, identifier, srs_id) values (?, 'features', ?, ?)",
                (table, table, SRS))
    con.execute("insert into gpkg_geometry_columns values (?, 'geom', 'MULTIPOLYGON', ?, 0, 0)", (table, SRS))
    marks = ", ".join("?" * (len(columns) + 1))
    names = ", ".join(["geom"] + [n for n, _ in columns])
    con.executemany(f'insert into "{table}" ({names}) values ({marks})',
                    [(gpkg_blob(MultiPolygon([g]) if g.geom_type == "Polygon" else g), *v) for g, *v in rows])


# --- Grid -------------------------------------------------------------------
def s2_file(year):
    files = sorted(C.S2_DIR.glob(f"S2_{year}*.tif"))
    if len(files) != 1:
        raise SystemExit(f"expected one S2 file for {year} in {C.S2_DIR}, found {len(files)}")
    return files[0]


def square_box(transform, r, c):
    x0, y0 = transform * (c * TILE, r * TILE)
    x1, y1 = transform * ((c + 1) * TILE, (r + 1) * TILE)
    return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def tiles_frame():
    return pd.DataFrame([{"row": r * TILE, "col": c * TILE, "split": s}
                         for s, sq in SQUARES.items() for r, c in sq]).sort_values(["row", "col"])


def check_tiles(tiles, shape):
    """Kavya's read_tiles rules: inside the image, and train squares do not overlap val/test squares."""
    h, w = shape
    if ((tiles.row < 0) | (tiles.col < 0) | (tiles.row + TILE > h) | (tiles.col + TILE > w)).any():
        raise SystemExit("a square falls outside the image")
    if tiles.duplicated(["row", "col"]).any():
        raise SystemExit("a square is listed twice")
    tr = tiles[tiles.split == "train"][["row", "col"]].to_numpy()
    ot = tiles[tiles.split != "train"][["row", "col"]].to_numpy()
    if (np.abs(tr[:, None, :] - ot[None, :, :]) < TILE).all(2).any():
        raise SystemExit("train squares overlap val/test squares")


# --- Commands ---------------------------------------------------------------
def setup(year, from_year):
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    out = LABEL_DIR / f"labels_{year}.gpkg"
    if out.exists():
        raise SystemExit(f"{out.name} already exists - not overwriting your drawing")
    with rasterio.open(s2_file(year)) as s:
        transform, shape = s.transform, (s.height, s.width)
    tiles = tiles_frame()
    check_tiles(tiles, shape)
    if not TILES_CSV.exists():
        tiles.to_csv(TILES_CSV, index=False)
        print(f"-> {TILES_CSV} ({tiles.split.value_counts().to_dict()})")

    src = LABEL_DIR / f"labels_{from_year}.gpkg" if from_year else INVENTORY
    if not src.exists():
        raise SystemExit(f"{src} not found")
    shutil.copyfile(src, out)
    con = sqlite3.connect(out)
    try:
        if not from_year:     # fresh drawing: add the empty 'unsure' layer and the guide squares
            add_layer(con, "unsure", [("note", "TEXT")], [])
            add_layer(con, "squares", [("row", "INTEGER"), ("col", "INTEGER"), ("split", "TEXT"), ("name", "TEXT")],
                      [(square_box(transform, r, c), r * TILE, c * TILE, s, f"{r},{c}")
                       for s, sq in SQUARES.items() for r, c in sq])
        con.commit()
        print("integrity:", con.execute("pragma integrity_check").fetchone()[0])
    finally:
        con.close()
    print(f"-> {out} (copied from {src.name})")


def build(year):
    s2 = s2_file(year)
    date = s2.stem[3:]
    drawing = LABEL_DIR / f"labels_{year}.gpkg"
    with rasterio.open(s2) as s:
        profile, valid = s.profile, s.read(2) > 0          # B03 > 0: pixel has data
    shape, transform = valid.shape, profile["transform"]
    tiles = pd.read_csv(TILES_CSV)
    check_tiles(tiles, shape)

    glacier = read_layer(drawing, "glaciers")
    unsure = read_layer(drawing, "unsure")
    y = np.zeros(shape, "uint8")
    if glacier:
        y = rasterize([(g, 1) for g in glacier], out_shape=shape, transform=transform, fill=0, dtype="uint8")
    if unsure:
        y[rasterize([(g, 1) for g in unsure], out_shape=shape, transform=transform, dtype="uint8") == 1] = IGNORE

    inside = np.zeros(shape, bool)
    for r, c in tiles[["row", "col"]].itertuples(index=False):
        inside[r:r + TILE, c:c + TILE] = True
    y[~inside | ~valid] = IGNORE
    mask_file = MASK_DIR / f"mask_{date}.tif"
    if mask_file.exists():
        with rasterio.open(mask_file) as m:
            y[np.isin(m.read(1), [1, 2])] = IGNORE              # cloud, cloud shadow
    else:
        print(f"warning: {mask_file.name} not found - cloud not set to ignore")

    out = LABEL_DIR / f"label_{year}.tif"
    p = {"driver": "GTiff", "height": shape[0], "width": shape[1], "count": 1, "dtype": "uint8",
         "crs": profile["crs"], "transform": transform, "nodata": IGNORE,
         "compress": "deflate", "tiled": True, "blockxsize": 512, "blockysize": 512}
    with rasterio.open(out, "w", **p) as d:
        d.write(y, 1)
        d.set_band_description(1, "1 glacier, 0 not glacier, 255 ignore")
    out.with_suffix(".qml").write_text(LABEL_QML)

    # checks: same grid as the image, only 0/1/255, glacier share per split
    with rasterio.open(out) as d:
        same = d.crs == profile["crs"] and d.transform.almost_equals(transform) and d.shape == shape
    vals = set(np.unique(y).tolist())
    print(f"-> {out.name}: same grid as {s2.name}: {same}; values {sorted(vals)}; "
          f"{len(glacier)} glacier and {len(unsure)} unsure polygons")
    for split, grp in tiles.groupby("split"):
        cut = np.concatenate([y[r:r + TILE, c:c + TILE].ravel() for r, c in grp[["row", "col"]].itertuples(index=False)])
        lab = cut != IGNORE
        print(f"   {split:5s} {len(grp):2d} squares: glacier {100 * (cut[lab] == 1).mean():5.1f}% of labelled, "
              f"ignored {100 * (~lab).mean():4.1f}%")
    if not same or not vals <= {0, 1, IGNORE}:
        sys.exit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("command", choices=["setup", "build"])
    p.add_argument("--year", type=int, default=2022)
    p.add_argument("--from", dest="from_year", type=int, help="setup: copy this year's drawing instead of the inventory")
    a = p.parse_args()
    setup(a.year, a.from_year) if a.command == "setup" else build(a.year)


if __name__ == "__main__":
    main()
