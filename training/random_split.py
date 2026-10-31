"""Random-split control: the same squares as tiles.csv, with splits dealt out at random.

Usage:
    venv/Scripts/python random_split.py --tiles tiles.csv --seed 0 --out tiles_random_0.csv

Our real split is by glacier: no test glacier is seen in training. Most published glacier-mapping scores
use random splits, where squares of the same glacier land in both train and test. Training on this file
with the same seeds and comparing the test scores to the glacier split shows how much a random split
flatters a model. The number of squares per split is kept, so only the assignment changes. Extra
columns (glacier, ...) are kept as they are.
"""
import argparse

import numpy as np
import pandas as pd

from data import TILE


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--tiles", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    tiles = pd.read_csv(args.tiles)
    if {"row", "col", "split"} - set(tiles.columns):
        raise SystemExit(f"{args.tiles}: needs columns row, col, split")
    on_grid = (tiles.row % TILE == 0) & (tiles.col % TILE == 0)
    if not on_grid.all():
        # off-grid squares can overlap, and a shuffle could then put overlapping squares in train and test
        raise SystemExit(f"{args.tiles}: {(~on_grid).sum()} squares are not on the {TILE}-pixel grid")
    rng = np.random.default_rng(args.seed)
    out = tiles.copy()
    out["split"] = rng.permutation(tiles.split.to_numpy())
    out.to_csv(args.out, index=False)
    moved = (out.split != tiles.split).sum()
    print(f"{dict(out.split.value_counts())}, {moved} of {len(out)} squares changed split -> {args.out}")


if __name__ == "__main__":
    main()
