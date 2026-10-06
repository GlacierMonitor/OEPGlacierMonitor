"""Check the GPU set-up and find the largest batch size that fits (training step, bf16).

Usage: venv/Scripts/python gpu_check.py [--encoder resnet34] [--sizes 4 8 12 16 24 32]
"""
import argparse
import time

import torch

from data import CHANNELS, TILE
from model import build_model, dice_bce_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--sizes", type=int, nargs="+", default=[4, 8, 12, 16, 24, 32])
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(f"CUDA not available (torch {torch.__version__}) - install the CUDA build of torch")
    dev = torch.device("cuda")
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}, {total:.1f} GB, "
          f"bf16 {torch.cuda.is_bf16_supported()}")

    model = build_model(args.encoder, weights=None).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for bs in args.sizes:
        x = y = None
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            x = torch.randn(bs, len(CHANNELS), TILE, TILE, device=dev)
            y = (torch.rand(bs, TILE, TILE, device=dev) > 0.7).long()
            times = []
            for i in range(4):
                torch.cuda.synchronize(); t0 = time.time()
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = dice_bce_loss(model(x), y)
                loss.backward(); opt.step()
                torch.cuda.synchronize()
                if i: times.append(time.time() - t0)                     # skip warm-up step
            peak = torch.cuda.max_memory_allocated() / 2**30
            print(f"batch {bs:3d}: peak {peak:.2f} GB, {sum(times) / len(times) * 1000:.0f} ms/step, "
                  f"{bs * len(times) / sum(times):.0f} tiles/s")
        except torch.OutOfMemoryError:
            print(f"batch {bs:3d}: out of memory")
            break
        finally:
            del x, y
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
