#!/usr/bin/env python3
r"""
infer_sr.py
===========
Apply a trained SR3DNet (best_model.pt) to a low-res volume.

Pipeline: load LR -> normalize -> interpolate up to the HR grid (x factor) -> run the
model in overlapping, tapered tiles (keeps large volumes off the GPU all at once) ->
save the open-SR volume (and, optionally, the interpolation baseline it was built from).

This produces the "open-SR" for a held-out group (e.g. vcr298) that we then register
against the independent HR and compare.

Usage:
    uv run python infer_sr.py ^
      --lr "...\Data\ct\lr_vcr" --model runs\vcr276\best_model.pt ^
      --roi 0 400 0 790 800 1600 ^          (crop the vcr298 region out of lr_vcr)
      --factor 2.521 --tile 96 --overlap 16 ^
      --out infer\vcr298 --save-interp
"""
from __future__ import annotations
import argparse, os, time
import numpy as np
import torch

from sr_model import SR3DNet
from sr_validation_harness import load_stack, normalize, interpolate_to_hr, save_stack


def _starts(dim, tile, step):
    if dim <= tile:
        return [0]
    s = list(range(0, dim - tile + 1, step))
    if s[-1] != dim - tile:
        s.append(dim - tile)
    return s


def _taper(shape, overlap):
    """Separable linear taper so overlapping tiles blend without seams."""
    def ramp(n):
        w = np.ones(n, dtype=np.float32)
        o = min(overlap, n // 2)
        if o > 0:
            r = (np.arange(o) + 1) / (o + 1)
            w[:o] = r
            w[-o:] = r[::-1]
        return w
    wz, wy, wx = ramp(shape[0]), ramp(shape[1]), ramp(shape[2])
    return (wz[:, None, None] * wy[None, :, None] * wx[None, None, :]).astype(np.float32)


def tiled_infer(net, vol, tile, overlap, device, use_amp):
    D, H, W = vol.shape
    tile = [min(tile, D), min(tile, H), min(tile, W)]
    step = [max(1, t - overlap) for t in tile]
    out = np.zeros_like(vol, dtype=np.float32)
    acc = np.zeros_like(vol, dtype=np.float32)
    zs = _starts(D, tile[0], step[0])
    ys = _starts(H, tile[1], step[1])
    xs = _starts(W, tile[2], step[2])
    n = len(zs) * len(ys) * len(xs); done = 0
    net.eval()
    for z in zs:
        for y in ys:
            for x in xs:
                z1, y1, x1 = z + tile[0], y + tile[1], x + tile[2]
                block = vol[z:z1, y:y1, x:x1]
                t = torch.from_numpy(np.ascontiguousarray(block)[None, None]).to(device)
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                    pred = net(t)
                pred = pred[0, 0].float().cpu().numpy()
                w = _taper(pred.shape, overlap)
                out[z:z1, y:y1, x:x1] += pred * w
                acc[z:z1, y:y1, x:x1] += w
                done += 1
                print(f"\r[infer] tile {done}/{n}", end="", flush=True)
    print()
    return out / np.maximum(acc, 1e-6)


def main():
    ap = argparse.ArgumentParser(description="Apply a trained SR3DNet to an LR volume (tiled).")
    ap.add_argument("--lr", required=True, help="LR tiff stack (folder or multipage tif).")
    ap.add_argument("--model", required=True, help="Trained checkpoint, e.g. best_model.pt")
    ap.add_argument("--factor", type=float, default=2.521)
    ap.add_argument("--roi", nargs=6, type=int, default=None,
                    metavar=("z0", "z1", "y0", "y1", "x0", "x1"),
                    help="Crop the LR before upscaling (e.g. isolate the vcr298 region).")
    ap.add_argument("--tile", type=int, default=96, help="Tile size on the HR grid.")
    ap.add_argument("--overlap", type=int, default=16)
    ap.add_argument("--out", default="infer/out")
    ap.add_argument("--save-interp", action="store_true", help="Also save the interp baseline.")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    use_amp = (device == "cuda")
    os.makedirs(args.out, exist_ok=True)

    print(f"[load] LR <- {args.lr}")
    lr_raw = load_stack(args.lr)
    if args.roi:
        z0, z1, y0, y1, x0, x1 = args.roi
        lr_raw = lr_raw[z0:z1, y0:y1, x0:x1]
    lr, (lo, hi) = normalize(lr_raw)
    print(f"[norm] LR shape={lr.shape} range=({lo:.3g},{hi:.3g}) -> [0,1]")

    hr_shape = tuple(int(round(s * args.factor)) for s in lr.shape)
    print(f"[interp] LR {lr.shape} -> HR grid {hr_shape} (x{args.factor})")
    interp = interpolate_to_hr(lr, hr_shape, order=3)

    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    net = SR3DNet(cfg.get("channels", 64), cfg.get("groups", 4), cfg.get("blocks", 4)).to(device)
    net.load_state_dict(ckpt["model"])
    print(f"[model] {args.model}  params={net.param_count():,}  device={device}  "
          f"(val_psnr={ckpt.get('val_psnr', float('nan')):.2f} dB)")

    t0 = time.time()
    sr = tiled_infer(net, interp.astype(np.float32), args.tile, args.overlap, device, use_amp)
    print(f"[infer] done in {time.time()-t0:.0f}s  SR shape={sr.shape}")

    save_stack(sr, os.path.join(args.out, "open_SR.tif"))
    print(f"[out] open-SR -> {os.path.join(args.out, 'open_SR.tif')}")
    if args.save_interp:
        save_stack(interp.astype(np.float32), os.path.join(args.out, "interp_baseline.tif"))
        print(f"[out] interp  -> {os.path.join(args.out, 'interp_baseline.tif')}")


if __name__ == "__main__":
    main()
