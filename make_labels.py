"""GlacierMonitor Part 3: training squares, the label drawing file, and label rasters.

  venv\\Scripts\\python make_labels.py setup                     # once: tiles.csv + labels_2022.gpkg
  venv\\Scripts\\python make_labels.py setup --year 2016 --from 2022   # copy a finished drawing to another year
  venv\\Scripts\\python make_labels.py split --year 2022         # one drawing file per labeller + review files
  venv\\Scripts\\python make_labels.py build --year 2022         # after drawing: label_2022.tif
  venv\\Scripts\\python make_labels.py ids --year 2022           # glacier_ids.tif + .csv for the 20 glaciers
  venv\\Scripts\\python make_labels.py redraw --year 2022        # blind-redraw files for the agreement check

setup writes, in GlacierMonitor/labels/:
  tiles.csv              row, col, split: top-left pixel of each 256 x 256 square (Kavya's training format)
  labels_<year>.gpkg     the drawing file, with three layers:
                           glaciers  copy of inventory/glaciers.gpkg (or of another year's drawing): edit to match the image
                           unsure    empty: draw areas you cannot decide (they become 255)
                           squares   the squares, read-only guide for where to work
split rewrites tiles.csv with all SQUARES and copies labels_<year>.gpkg to one file per labeller:
  labels_<year>_<labeller>.gpkg   glaciers + unsure as in labels_<year>.gpkg; squares = only that labeller's squares
  review_<year>_<labeller>.gpkg   empty 'notes' point layer for the reviewer of that labeller's squares
build writes label_<year>.tif on the year's S2 grid: 1 glacier, 0 not glacier, 255 ignore.
If the per-labeller files exist, each square is taken from its labeller's file; otherwise from labels_<year>.gpkg.
ids writes glacier_ids.tif (one value per study glacier, 0 elsewhere) + glacier_ids.csv (value, glacier_id,
  glacier_name) for training/glacier_areas.py, from the same outlines as build (labels_<year>.gpkg outside the squares).
redraw writes labels_<year>_redraw_<who>.gpkg: inventory outlines, only that person's REDRAW squares.
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

# Squares agreed 2026-10-18, tongue ends added 2026-10-27 (decisions.md 18), 20 glaciers 2026-10-31 (decisions.md 19),
# as (tile row, tile col); pixel offset = index * 256. Split by glacier: test = Khumbu, Imja/Lhotse Shar, Amphu Laptse
# and the small glaciers above Gorak Shep, Gokyo, Na, Machhermo, Lungden; val = Changri Shar/Nup, Lobuche, Cholo,
# Cholotse; train = Ngozumpa, Ama Dablam, Nuptse, Lhotse, Duwo, Ombigaichain, Dzongla glacier, and squares picked
# for off-glacier snow, lakes, shadow, debris, rock and villages / vegetation.
SQUARES = {
    "test": [(5, 9), (5, 10), (5, 11), (6, 8), (6, 9), (6, 10), (6, 11), (6, 12), (7, 8), (7, 13), (7, 14),
             (8, 13), (9, 12), (9, 13), (10, 13),
             (8, 7), (8, 8),                                      # Khumbu lower tongue + tongue end
             (7, 2), (8, 2), (9, 2), (9, 3), (10, 2), (10, 12),   # 20 glaciers: above Gokyo, Na, Machhermo, Lungden;
             (9, 4), (10, 1)],                                    #   Amphu Laptse; Machhermo, Lungden villages
    "val": [(4, 7), (5, 7), (5, 8), (6, 6), (6, 7), (7, 7),
            (9, 5), (9, 6), (9, 7), (9, 8)],                      # 20 glaciers: Cholotse, Cholo; Pheriche, Dingboche
    "train": [(0, 2), (0, 5), (1, 2), (1, 3), (1, 4), (1, 5), (2, 2), (2, 3), (2, 4), (2, 5), (2, 12), (3, 3),
              (3, 4), (3, 5), (3, 13), (4, 12), (4, 13), (5, 3), (5, 12), (7, 0), (7, 10), (9, 1), (9, 10),
              (10, 3), (10, 10), (11, 10), (11, 12),
              (6, 3), (7, 3), (7, 4), (8, 4),                     # Ngozumpa lower debris tongue + tongue end
              (8, 5), (8, 10), (8, 11), (7, 12), (8, 12), (9, 11), (10, 9), (10, 11),  # 20 glaciers: Dzongla, Nuptse,
              (11, 5), (11, 7)],                                  #   Lhotse, Duwo, Ombigaichain; Phortse, Pangboche
}
# Villages in or at the edge of a square (approximate positions), kept for later change tracking.
PLACES = {(6, 8): "Gorak Shep", (7, 3): "Gokyo", (7, 4): "Thangnak", (7, 8): "Lobuche", (8, 4): "Na",
          (9, 4): "Machhermo", (9, 8): "Pheriche, Dingboche", (9, 10): "Chukhung", (10, 1): "Lungden",
          (11, 5): "Phortse", (11, 7): "Pangboche"}
REVIEWER = {"krisha": "siddhi", "kavya": "snehi"}
# Blind redraw for the two-person agreement number (decisions.md 19): test squares redrawn from the inventory
# outlines by someone who has not seen the first drawing.
REDRAW = {"siddhi": [(5, 10), (6, 11), (7, 8), (8, 7), (8, 8), (8, 13), (9, 12), (9, 13)],
          "snehi": [(7, 2), (8, 2), (9, 2), (9, 3), (10, 2)]}
# The 20 study glaciers (decisions.md 19) by RGI id, with the names written to glacier_ids.csv / results.csv.
# The unnamed ones carry working names after the nearest village until the team agrees on names.
GLACIERS = {
    "RGI2000-v7.0-G-15-08331": "Khumbu", "RGI2000-v7.0-G-15-06763": "Imja/Lhotse Shar",
    "RGI2000-v7.0-G-15-06797": "Amphu Laptse", "RGI2000-v7.0-G-15-06748": "Gorak Shep (fid 96)",
    "RGI2000-v7.0-G-15-06700": "Gokyo (fid 48)", "RGI2000-v7.0-G-15-06701": "Na (fid 49)",
    "RGI2000-v7.0-G-15-06712": "Machhermo (fid 60)", "RGI2000-v7.0-G-15-06713": "Lungden (fid 61)",
    "RGI2000-v7.0-G-15-08295": "Changri Shar/Nup", "RGI2000-v7.0-G-15-06739": "Lobuche",
    "RGI2000-v7.0-G-15-08346": "Cholo", "RGI2000-v7.0-G-15-08342": "Cholotse",
    "RGI2000-v7.0-G-15-06741": "Lobuche village (fid 89)", "RGI2000-v7.0-G-15-08305": "Ngozumpa",
    "RGI2000-v7.0-G-15-06794": "Ama Dablam", "RGI2000-v7.0-G-15-06755": "Nuptse",
    "RGI2000-v7.0-G-15-06758": "Lhotse", "RGI2000-v7.0-G-15-06793": "Duwo",
    "RGI2000-v7.0-G-15-06795": "Ombigaichain", "RGI2000-v7.0-G-15-06724": "Dzongla (fid 72)",
}


def labeller(r, c, split):
    """Krisha: the west (Gokyo, Ngozumpa, Bhote Koshi) and all val squares; Kavya: Khumbu, Imja, Lhotse, Ama Dablam."""
    return "krisha" if c <= 6 or split == "val" else "kavya"

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


def read_glaciers(path):
    """(shape, RGI glacier_id) for every outline in the 'glaciers' layer; parts split off in QGIS keep their id."""
    con = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        return [(gpkg_geom(g), gid) for g, gid in con.execute('select geom, glacier_id from "glaciers" where geom is not null')]
    finally:
        con.close()


def add_layer(con, table, columns, rows, gtype="MULTIPOLYGON"):
    """Create a feature table registered in the GeoPackage, and insert rows (geom, *values)."""
    cols = "".join(f", {n} {t}" for n, t in columns)
    con.execute(f'create table "{table}" (fid INTEGER PRIMARY KEY AUTOINCREMENT, geom {gtype}{cols})')
    con.execute("insert into gpkg_contents (table_name, data_type, identifier, srs_id) values (?, 'features', ?, ?)",
                (table, table, SRS))
    con.execute("insert into gpkg_geometry_columns values (?, 'geom', ?, ?, 0, 0)", (table, gtype, SRS))
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


def split_files(year):
    """One drawing file per labeller (only their squares in the 'squares' layer) and an empty review file each."""
    src = LABEL_DIR / f"labels_{year}.gpkg"
    if not src.exists():
        raise SystemExit(f"{src} not found - run setup first")
    with rasterio.open(s2_file(year)) as s:
        transform, shape = s.transform, (s.height, s.width)
    tiles = tiles_frame()
    check_tiles(tiles, shape)
    outs = [LABEL_DIR / f"{kind}_{year}_{who}.gpkg" for who in REVIEWER for kind in ("labels", "review")]
    if any(o.exists() for o in outs):
        raise SystemExit("per-labeller files already exist - not overwriting anyone's drawing or notes")
    tiles.to_csv(TILES_CSV, index=False)
    print(f"-> {TILES_CSV} ({tiles.split.value_counts().to_dict()})")

    for who in REVIEWER:
        mine = [(s, r, c) for s, sq in SQUARES.items() for r, c in sq if labeller(r, c, s) == who]
        out = LABEL_DIR / f"labels_{year}_{who}.gpkg"
        shutil.copyfile(src, out)
        con = sqlite3.connect(out)
        try:
            con.execute('drop table "squares"')
            con.execute("delete from gpkg_geometry_columns where table_name = 'squares'")
            con.execute("delete from gpkg_contents where table_name = 'squares'")
            add_layer(con, "squares", [("row", "INTEGER"), ("col", "INTEGER"), ("split", "TEXT"), ("name", "TEXT"),
                                       ("labeller", "TEXT"), ("reviewer", "TEXT"), ("place", "TEXT")],
                      [(square_box(transform, r, c), r * TILE, c * TILE, s, f"{r},{c}", who, REVIEWER[who],
                        PLACES.get((r, c), "")) for s, r, c in mine])
            con.commit()
            print(f"-> {out.name}: {len(mine)} squares, integrity", con.execute("pragma integrity_check").fetchone()[0])
        finally:
            con.close()

        review = LABEL_DIR / f"review_{year}_{who}.gpkg"
        shutil.copyfile(src, review)                       # copy for the GeoPackage system tables, then empty it
        con = sqlite3.connect(review)
        try:
            for (t,) in con.execute("select table_name from gpkg_contents").fetchall():
                con.execute(f'drop table "{t}"')
                con.execute("delete from gpkg_geometry_columns where table_name = ?", (t,))
                con.execute("delete from gpkg_contents where table_name = ?", (t,))
            for (t,) in con.execute("select name from sqlite_master where sql like 'CREATE VIRTUAL TABLE%'").fetchall():
                con.execute(f'drop table "{t}"')            # spatial index (also drops its _node/_parent/_rowid)
            con.execute("delete from gpkg_extensions where table_name is not null")
            con.execute("delete from gpkg_ogr_contents")
            add_layer(con, "notes", [("square", "TEXT"), ("problem", "TEXT"), ("status", "TEXT")], [], gtype="POINT")
            con.commit()
        finally:
            con.close()
        con = sqlite3.connect(review)
        con.execute("vacuum")
        print(f"-> {review.name}: empty notes layer for {REVIEWER[who]}, integrity",
              con.execute("pragma integrity_check").fetchone()[0])
        con.close()


def redraw_files(year):
    """Fresh drawing files from the inventory outlines, holding only each blind redrawer's squares."""
    with rasterio.open(s2_file(year)) as s:
        transform = s.transform
    outs = {who: LABEL_DIR / f"labels_{year}_redraw_{who}.gpkg" for who in REDRAW}
    if any(o.exists() for o in outs.values()):
        raise SystemExit("redraw files already exist - not overwriting anyone's drawing")
    split = {(r, c): s for s, sq in SQUARES.items() for r, c in sq}
    for who, out in outs.items():
        shutil.copyfile(INVENTORY, out)
        con = sqlite3.connect(out)
        try:
            add_layer(con, "unsure", [("note", "TEXT")], [])
            add_layer(con, "squares", [("row", "INTEGER"), ("col", "INTEGER"), ("split", "TEXT"), ("name", "TEXT")],
                      [(square_box(transform, r, c), r * TILE, c * TILE, split[r, c], f"{r},{c}") for r, c in REDRAW[who]])
            con.commit()
            print(f"-> {out.name}: {len(REDRAW[who])} squares from the inventory outlines, integrity",
                  con.execute("pragma integrity_check").fetchone()[0])
        finally:
            con.close()


def merge_by_labeller(year, tiles, base, draw):
    """base, with each square replaced by draw(<its labeller's file>) when the per-labeller files exist."""
    files = {who: LABEL_DIR / f"labels_{year}_{who}.gpkg" for who in REVIEWER}
    if not any(p.exists() for p in files.values()):
        return base
    missing = [p.name for p in files.values() if not p.exists()]
    if missing:
        raise SystemExit(f"missing {missing}")
    for who, path in files.items():
        yw = draw(path)
        for r, c, s in tiles[["row", "col", "split"]].itertuples(index=False):
            if labeller(r // TILE, c // TILE, s) == who:
                base[r:r + TILE, c:c + TILE] = yw[r:r + TILE, c:c + TILE]
    return base


def glacier_ids(year):
    """glacier_ids.tif + .csv for the 20 glaciers (format of training/glacier_areas.py): the year's final
    outlines inside the squares, labels_<year>.gpkg (inventory + earlier edits) outside them."""
    with rasterio.open(s2_file(year)) as s:
        profile = s.profile
    shape, transform = (profile["height"], profile["width"]), profile["transform"]
    tiles = pd.read_csv(TILES_CSV)
    value = {gid: i for i, gid in enumerate(GLACIERS, start=1)}

    def draw(path):
        shapes = [(g, value[gid]) for g, gid in read_glaciers(path) if gid in value]
        return rasterize(shapes, out_shape=shape, transform=transform, fill=0, dtype="uint8")

    ids = merge_by_labeller(year, tiles, draw(LABEL_DIR / f"labels_{year}.gpkg"), draw)
    out = LABEL_DIR / "glacier_ids.tif"
    p = {"driver": "GTiff", "height": shape[0], "width": shape[1], "count": 1, "dtype": "uint8",
         "crs": profile["crs"], "transform": transform, "compress": "deflate", "tiled": True,
         "blockxsize": 512, "blockysize": 512}
    with rasterio.open(out, "w", **p) as d:
        d.write(ids, 1)
        d.set_band_description(1, "glacier value (glacier_ids.csv), 0 no glacier")
    lookup = pd.DataFrame({"value": list(value.values()), "glacier_id": list(value), "glacier_name": list(GLACIERS.values())})
    lookup.to_csv(out.with_suffix(".csv"), index=False)

    px = np.bincount(ids.ravel(), minlength=len(value) + 1)
    print(f"-> {out.name} + {out.with_suffix('.csv').name}: {len(value)} glaciers")
    for v, name in zip(lookup.value, lookup.glacier_name):
        print(f"   {v:2d} {name:26s} {px[v] / 1e4:7.2f} km2")
    if (px[1:] == 0).any():
        sys.exit("a glacier has no pixels - check its glacier_id in the drawing")


def rasterize_drawing(drawing, shape, transform):
    glacier = read_layer(drawing, "glaciers")
    unsure = read_layer(drawing, "unsure")
    y = np.zeros(shape, "uint8")
    if glacier:
        y = rasterize([(g, 1) for g in glacier], out_shape=shape, transform=transform, fill=0, dtype="uint8")
    if unsure:
        y[rasterize([(g, 1) for g in unsure], out_shape=shape, transform=transform, dtype="uint8") == 1] = IGNORE
    print(f"   {drawing.name}: {len(glacier)} glacier and {len(unsure)} unsure polygons")
    return y


def build(year):
    s2 = s2_file(year)
    date = s2.stem[3:]
    with rasterio.open(s2) as s:
        profile, valid = s.profile, s.read(2) > 0          # B03 > 0: pixel has data
    shape, transform = valid.shape, profile["transform"]
    tiles = pd.read_csv(TILES_CSV)
    check_tiles(tiles, shape)

    draw = lambda path: rasterize_drawing(path, shape, transform)
    if any((LABEL_DIR / f"labels_{year}_{who}.gpkg").exists() for who in REVIEWER):
        y = merge_by_labeller(year, tiles, np.full(shape, IGNORE, "uint8"), draw)
    else:
        y = draw(LABEL_DIR / f"labels_{year}.gpkg")

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
    print(f"-> {out.name}: same grid as {s2.name}: {same}; values {sorted(vals)}")
    for split, grp in tiles.groupby("split"):
        cut = np.concatenate([y[r:r + TILE, c:c + TILE].ravel() for r, c in grp[["row", "col"]].itertuples(index=False)])
        lab = cut != IGNORE
        print(f"   {split:5s} {len(grp):2d} squares: glacier {100 * (cut[lab] == 1).mean():5.1f}% of labelled, "
              f"ignored {100 * (~lab).mean():4.1f}%")
    if not same or not vals <= {0, 1, IGNORE}:
        sys.exit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("command", choices=["setup", "split", "build", "ids", "redraw"])
    p.add_argument("--year", type=int, default=2022)
    p.add_argument("--from", dest="from_year", type=int, help="setup: copy this year's drawing instead of the inventory")
    a = p.parse_args()
    {"setup": lambda: setup(a.year, a.from_year), "split": lambda: split_files(a.year), "build": lambda: build(a.year),
     "ids": lambda: glacier_ids(a.year), "redraw": lambda: redraw_files(a.year)}[a.command]()


if __name__ == "__main__":
    main()
