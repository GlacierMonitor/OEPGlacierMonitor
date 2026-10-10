"""Train the U-Net on the reference year.

Usage:
    venv/Scripts/python train.py --s2 S2_20221024.tif --dem CopDEM_GLO30_UTM45N.tif \
        --label label_2022.tif --tiles tiles.csv --out runs/ref2022

Writes to --out: best.pt (model + settings + normalisation stats), stats.json, log.csv.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from data import CHANNELS, TileDataset, compute_stats, load_stack, normalise, read_label, read_tiles, save_stats
from model import build_model, confusion, dice_bce_loss, iou
from data import read_mask


def evaluate(model, loader, device, amp_dtype):
    model.eval()
    tp = fp = fn = 0
    loss_sum, n = 0.0, 0
    with torch.no_grad(), torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += dice_bce_loss(logits, y).item() * len(x); n += len(x)
            a, b, c = confusion(logits[:, 0] > 0, y)
            tp += a; fp += b; fn += c
    return loss_sum / max(n, 1), iou(tp, fp, fn)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--s2", required=True); p.add_argument("--dem", required=True)
    p.add_argument("--label", required=True); p.add_argument("--tiles", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=15, help="stop after this many epochs without val IoU gain")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--weights", default="imagenet", help="'imagenet' or 'none'")
    p.add_argument("--masks", help="Part 2 masks folder (mask_YYYYMMDD.tif): ignore cloud, cloud shadow, no data")
    p.add_argument("--mask-cloud", action="store_true", help="ignore SCL cloud pixels (only reliable 2022+)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = (torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None)
    print(f"device {device}{' ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}, "
          f"mixed precision {amp_dtype}")

    x, valid, profile = load_stack(args.s2, args.dem, args.mask_cloud)
    if args.masks:
        valid &= ~read_mask(args.s2, args.masks, profile)
    y = read_label(args.label, valid, profile)
    tiles = read_tiles(args.tiles, valid.shape)
    train_t, val_t = tiles[tiles.split == "train"], tiles[tiles.split == "val"]
    if len(train_t) == 0 or len(val_t) == 0:
        raise SystemExit("tiles.csv needs both 'train' and 'val' tiles")
    print(f"tiles: {len(train_t)} train, {len(val_t)} val, {(tiles.split == 'test').sum()} test "
          f"(test is not touched here)")

    stats = compute_stats(x, y, train_t)
    save_stats(stats, out / "stats.json")
    x = normalise(x, valid, stats)

    train_dl = torch.utils.data.DataLoader(TileDataset(x, y, train_t, augment=True), batch_size=args.batch,
                                           shuffle=True, drop_last=len(train_t) > args.batch,
                                           pin_memory=device.type == "cuda")
    val_dl = torch.utils.data.DataLoader(TileDataset(x, y, val_t), batch_size=args.batch,
                                         pin_memory=device.type == "cuda")

    weights = None if args.weights.lower() == "none" else args.weights
    model = build_model(args.encoder, weights).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best, since_best = -1.0, 0
    with open(out / "log.csv", "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["epoch", "train_loss", "val_loss", "val_iou", "lr", "seconds"])
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            model.train()
            total, n = 0.0, 0
            for xb, yb in train_dl:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    loss = dice_bce_loss(model(xb), yb)
                loss.backward()
                opt.step()
                total += loss.item() * len(xb); n += len(xb)
            sched.step()
            val_loss, val_iou = evaluate(model, val_dl, device, amp_dtype)
            secs = time.time() - t0
            log.writerow([epoch, round(total / n, 4), round(val_loss, 4), round(val_iou, 4),
                          f"{opt.param_groups[0]['lr']:.2e}", round(secs, 1)]); f.flush()
            mark = ""
            if val_iou > best:
                best, since_best, mark = val_iou, 0, "  * saved"
                torch.save({"model": model.state_dict(), "encoder": args.encoder, "channels": CHANNELS,
                            "stats": stats, "epoch": epoch, "val_iou": val_iou,
                            "mask_cloud": args.mask_cloud, "s2": str(args.s2)}, out / "best.pt")
            else:
                since_best += 1
            print(f"epoch {epoch:3d}  train {total / n:.4f}  val {val_loss:.4f}  IoU {val_iou:.4f}  "
                  f"{secs:.0f}s{mark}")
            if since_best >= args.patience:
                print(f"no val IoU gain for {args.patience} epochs - stopping")
                break

    if device.type == "cuda":
        print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    print(f"best val IoU {best:.4f} -> {out / 'best.pt'}")
    (out / "summary.json").write_text(json.dumps({"best_val_iou": best, **vars(args)}, indent=2))


if __name__ == "__main__":
    main()
