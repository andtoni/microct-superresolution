#!/usr/bin/env python3
r"""
compare_sr.py
=============
The payoff step. Given the open-SR of the located vcr298 region, the interpolation baseline,
and the independent HR (hr_vcr298), this:

  1. (optionally) pre-orients the HR per the localizer (flip/in-plane angle);
  2. rigidly registers HR -> open-SR (both at HR resolution -> a clean, well-posed alignment);
  3. resamples HR onto the SR grid and builds the overlap mask;
  4. on the overlap, scores open-SR vs HR and interpolation vs HR (PSNR / SSIM / MS-SSIM);
  5. builds a hallucination map = |open-SR - HR| - |interp - HR|  (where the model deviates
     from truth MORE than plain interpolation does -> fabricated structure);
  6. writes a comparison figure + metrics.csv.

Each volume is normalized to its own [0,1] so the comparison is about structure/contrast,
not absolute intensity (the scans have different intensity scales).

Usage:
    uv run python compare_sr.py ^
      --sr infer\vcr298\open_SR.tif --interp infer\vcr298\interp_baseline.tif ^
      --hr "...\Data\ct\hr_vcr298" --orient id --angle 0 --out compare\vcr298
"""
from __future__ import annotations
import argparse, csv, os
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import rotate as ndrotate

from sr_validation_harness import load_stack, normalize, save_stack, psnr, ssim3d, ms_ssim3d
from register_volumes import to_sitk, register, resample_onto

FLIPS = {"id": (), "flipX": (2,), "flipY": (1,), "flipZ": (0,), "flipXZ": (0, 2),
         "rot180": (1, 2)}


def pre_orient(vol, orient, angle):
    for a in FLIPS.get(orient, ()):
        vol = np.flip(vol, a)
    if angle % 360 != 0:
        vol = ndrotate(vol, angle, axes=(1, 2), reshape=False, order=1)
    return np.ascontiguousarray(vol)


def bbox_of(mask):
    idx = np.where(mask)
    return tuple(slice(int(i.min()), int(i.max()) + 1) for i in idx)


def main():
    ap = argparse.ArgumentParser(description="Register HR to open-SR and compare (vs interp).")
    ap.add_argument("--sr", required=True, help="open_SR.tif")
    ap.add_argument("--interp", help="interp_baseline.tif (optional but recommended)")
    ap.add_argument("--hr", required=True, help="independent HR (e.g. hr_vcr298)")
    ap.add_argument("--orient", default="id", help="pre-orientation from the localizer")
    ap.add_argument("--angle", type=float, default=0.0, help="in-plane angle from the localizer")
    ap.add_argument("--spacing", type=float, default=0.9696, help="HR-grid voxel size (um)")
    ap.add_argument("--train-maxslices", type=int, default=256,
                    help="Central HR slices the model trained on (excluded from the test).")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--sampling", type=float, default=0.01,
                    help="fraction of fixed-image voxels per metric eval during "
                         "registration; 0.01 is plenty for >1e9-voxel volumes")
    ap.add_argument("--out", default="compare/out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"[load] open-SR <- {args.sr}")
    sr, _ = normalize(load_stack(args.sr))
    interp = None
    if args.interp:
        interp, _ = normalize(load_stack(args.interp))
    print(f"[load] HR <- {args.hr}  (orient={args.orient}, angle={args.angle})")
    hr_raw = pre_orient(load_stack(args.hr), args.orient, args.angle)
    hr, _ = normalize(hr_raw)

    # register HR (moving) onto open-SR (fixed); both on the HR grid -> easy alignment
    fixed = to_sitk(sr, args.spacing); moving = to_sitk(hr, args.spacing)
    print("[reg] registering HR -> open-SR ...")
    tfm, metric = register(fixed, moving, iters=args.iters, sampling=args.sampling)
    moved, mask = resample_onto(fixed, moving, tfm)
    hr_a = sitk.GetArrayFromImage(moved)
    m = sitk.GetArrayFromImage(mask) > 0.5
    print(f"[reg] final MI={metric:.4f}   overlap={100*m.mean():.1f}% of SR volume")
    sitk.WriteTransform(tfm, os.path.join(args.out, "hr_to_sr.tfm"))

    # ---- spatial holdout: exclude the central slices the model trained on -------------
    N = hr.shape[0]; tm = args.train_maxslices
    c0 = max(0, N // 2 - tm // 2)
    holdout_hr = np.ones(hr.shape, dtype=np.float32)
    holdout_hr[c0:c0 + tm] = 0.0                          # 0 = trained, 1 = held-out
    hmask = sitk.Resample(to_sitk(holdout_hr, args.spacing), fixed, tfm,
                          sitk.sitkNearestNeighbor, 0.0, sitk.sitkFloat32)
    holdout = sitk.GetArrayFromImage(hmask) > 0.5
    held = m & holdout
    print(f"[holdout] HR trained slices {c0}:{c0 + tm} of {N}; held-out test region = "
          f"{int(held.sum())} voxels ({100 * held.mean():.1f}% of SR volume)")

    def score(region, label):
        if int(region.sum()) == 0:
            return []
        sl = bbox_of(region); valid = region[sl]
        def ap(v):
            o = v[sl].copy(); o[~valid] = 0.0; return o
        hrc, src = ap(hr_a), ap(sr)
        out = [(f"{label}: open-SR vs HR", psnr(hrc, src), ssim3d(hrc, src), ms_ssim3d(hrc, src))]
        if interp is not None:
            ipc = ap(interp)
            out.append((f"{label}: interp vs HR", psnr(hrc, ipc), ssim3d(hrc, ipc), ms_ssim3d(hrc, ipc)))
        return out

    rows = score(held, "held-out (non-circular)") + score(m, "full overlap (circular ref)")
    print("\n  comparison:")
    print(f"  {'region: method':40s} {'PSNR(dB)':>9s} {'SSIM':>7s} {'MS-SSIM':>8s}")
    for name, p, s, ms in rows:
        print(f"  {name:40s} {p:9.2f} {s:7.3f} {ms:8.3f}")
    with open(os.path.join(args.out, "metrics.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["region_method", "PSNR_dB", "SSIM", "MS_SSIM"])
        for r in rows: w.writerow([r[0], f"{r[1]:.4f}", f"{r[2]:.4f}", f"{r[3]:.4f}"])

    # hallucination map over the full overlap (visual); mean reported on the held-out region
    excess = None
    if interp is not None:
        excess = np.clip(np.abs(sr - hr_a) - np.abs(interp - hr_a), 0, None)
        excess[~m] = 0.0
        save_stack(excess, os.path.join(args.out, "hallucination_map.tif"))
        reg = held if int(held.sum()) else m
        print(f"[out] hallucination map -> {os.path.join(args.out,'hallucination_map.tif')}  "
              f"(mean excess on held-out {float(excess[reg].mean()):.4f})")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        region = held if int(held.sum()) else m
        sl = bbox_of(region); valid = region[sl]
        def ap(v):
            o = v[sl].copy(); o[~valid] = 0.0; return o
        HRz, SRz = ap(hr_a), ap(sr); z = HRz.shape[0] // 2
        panels = [("HR (truth)", HRz[z], dict(cmap="gray", vmin=0, vmax=1)),
                  ("open-SR", SRz[z], dict(cmap="gray", vmin=0, vmax=1))]
        if interp is not None:
            panels.insert(1, ("interpolation", ap(interp)[z], dict(cmap="gray", vmin=0, vmax=1)))
            panels.append(("excess error vs interp", ap(excess)[z], dict(cmap="inferno", vmin=0, vmax=0.4)))
        fig, ax = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
        for a, (t, im, kw) in zip(np.atleast_1d(ax), panels):
            a.imshow(im, **kw); a.set_title(t); a.axis("off")
        plt.tight_layout(); plt.savefig(os.path.join(args.out, "compare_midslice.png"), dpi=120)
        print(f"[out] figure -> {os.path.join(args.out, 'compare_midslice.png')}")
    except Exception as e:
        print(f"[warn] no figure: {e}")


if __name__ == "__main__":
    main()
