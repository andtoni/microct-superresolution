#!/usr/bin/env python3
r"""
localize_hr.py  (rotation-augmented, thorough)
==============================================
Find where an HR window (hr_vcr298) sits inside a large LR volume (lr_vcr) when the two
scans may differ by an in-plane rotation and/or a flip. Strategy:

  1. COARSE search over {orientation x in-plane angle x translation}, downsampled by
     --coarsen, normalized cross-correlation. Orientations: identity, in-plane mirror
     (flipX), z-direction flip (flipZ), and both. In-plane rotation swept every --rot-step
     degrees over 0..360.
  2. Variance mask (precomputed ONCE for the fixed template size) so flat air can never win.
  3. FINE refine at full resolution in a window around the best coarse hit.
  4. report TOP-K candidates and the winning orientation + angle.

Location math tracks the template CENTRE (all crops are central, rotation is about the
centre), so it stays correct under rotation/flip. The reported angle/flip are handed to the
compare step so hr_vcr298 is pre-oriented before the final rigid registration.

Usage (thorough):
    uv run python localize_hr.py ^
      --hr "...\Data\ct\hr_vcr298" --lr "...\Data\ct\lr_vcr" ^
      --lr-zrange 0 1000 --coarsen 5 --rot-step 10 --out localize\vcr298
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np
from skimage.transform import resize
from skimage.feature import match_template
from scipy.ndimage import uniform_filter, rotate as ndrotate

from sr_validation_harness import load_stack, normalize, make_porous_phantom

ORIENTS = {"id": (), "flipX": (2,), "flipZ": (0,), "flipXZ": (0, 2)}


def downscale_to(vol, out_shape):
    return resize(vol, out_shape, order=1, anti_aliasing=True,
                  preserve_range=True).astype(np.float32)


def load_lr_coarse(path, z0, z1, c):
    import tifffile
    if not os.path.isdir(path):
        vol = load_stack(path)[z0:z1:c]
        return downscale_to(vol, (vol.shape[0], max(1, vol.shape[1] // c),
                                  max(1, vol.shape[2] // c)))
    files = sorted(glob.glob(os.path.join(path, "*.tif")) +
                   glob.glob(os.path.join(path, "*.tiff")))[z0:z1:c]
    out = []
    for fp in files:
        s = tifffile.imread(fp).astype(np.float32)
        out.append(downscale_to(s, (max(1, s.shape[0] // c), max(1, s.shape[1] // c))))
    return np.stack(out, 0)


def load_lr_window(path, z0, z1, y0, y1, x0, x1):
    import tifffile
    if not os.path.isdir(path):
        return load_stack(path)[z0:z1, y0:y1, x0:x1].astype(np.float32)
    files = sorted(glob.glob(os.path.join(path, "*.tif")) +
                   glob.glob(os.path.join(path, "*.tiff")))[z0:z1]
    return np.stack([tifffile.imread(fp)[y0:y1, x0:x1].astype(np.float32) for fp in files], 0)


def orient(vol, axes):
    out = vol
    for a in axes:
        out = np.flip(out, a)
    return np.ascontiguousarray(out)


def rotate_crop(t, ang, frac):
    """In-plane rotate (about z), then central-crop to drop rotation corners. Fixed output
    shape for a given input + frac, independent of angle (so the mask is reusable)."""
    if ang % 360 != 0:
        t = ndrotate(t, ang, axes=(1, 2), reshape=False, order=1)
    Z, Y, X = t.shape
    fy, fx = int(round(frac * Y)), int(round(frac * X))
    cy, cx = (Y - fy) // 2, (X - fx) // 2
    return np.ascontiguousarray(t[:, cy:cy + fy, cx:cx + fx])


def validity_mask(target, win):
    t = target.astype(np.float64)
    m1 = uniform_filter(t, win); m2 = uniform_filter(t * t, win)
    std = np.sqrt(np.clip(m2 - m1 * m1, 0, None))
    sl = tuple(slice(w // 2, w // 2 + target.shape[i] - w + 1) for i, w in enumerate(win))
    std_c = std[sl]
    thr = max(0.02, 0.1 * float(std_c.max()))
    return std_c >= thr


def masked_corr(target, template, mask):
    ncc = np.clip(match_template(target, template), -1.0, 1.0)
    return np.where(mask, ncc, -1.0).astype(np.float32)


def sweep(base, target, orient_set, angles, frac):
    win = rotate_crop(base, 0, frac).shape
    mask = validity_mask(target, win)
    best = ("id", 0, None, -2.0, None)
    for name, ax in orient_set.items():
        b0 = orient(base, ax)
        for ang in angles:
            t = rotate_crop(b0, ang, frac)
            ncc = masked_corr(target, t, mask)
            pk = float(ncc.max())
            if pk > best[3]:
                corner = np.array(np.unravel_index(int(np.argmax(ncc)), ncc.shape))
                best = (name, ang, corner, pk, ncc)
    return best, win


def topk_peaks(ncc, k, min_sep=4):
    flat = np.argsort(ncc.ravel())[::-1]; picks = []
    for idx in flat:
        p = np.array(np.unravel_index(int(idx), ncc.shape))
        if all(np.abs(p - q).max() >= min_sep for q, _ in picks):
            picks.append((p, float(ncc[tuple(p)])))
        if len(picks) >= k:
            break
    return picks


def run_selftest():
    print("[selftest] embed a +15-deg rotated window; recover location + angle ...")
    big = normalize(make_porous_phantom(shape=(110, 200, 200)))[0]
    z0, y0, x0 = 30, 70, 70; sz = (40, 70, 70)
    sub = big[z0:z0 + sz[0], y0:y0 + sz[1], x0:x0 + sz[2]].copy()
    target = big.copy()
    target[z0:z0 + sz[0], y0:y0 + sz[1], x0:x0 + sz[2]] = ndrotate(sub, 15, axes=(1, 2),
                                                                   reshape=False, order=1)
    target = (target + np.random.default_rng(0).normal(0, 0.02, target.shape)).astype(np.float32)
    (name, ang, corner, pk, _), win = sweep(sub, target, {"id": ()}, range(0, 360, 5), 0.6)
    cz = corner[0] + win[0] / 2; cy = corner[1] + win[1] / 2; cx = corner[2] + win[2] / 2
    true_c = (z0 + sz[0] / 2, y0 + sz[1] / 2, x0 + sz[2] / 2)
    print(f"[selftest] angle found={ang} (true 15)  centre={tuple(round(v) for v in (cz,cy,cx))} "
          f"true={tuple(round(v) for v in true_c)}  ncc={pk:.3f}")
    ok = abs(ang - 15) <= 5 and abs(cz - true_c[0]) <= 3 and abs(cy - true_c[1]) <= 4
    print(f"[selftest] {'OK' if ok else 'CHECK'}")


def localize(hr_norm, lr_path, lr_zrange=(0, None), coarsen=5, rot_step=10, no_flips=False,
             crop_frac=0.6, hr_incircle_frac=0.7, margin=40, refine_pad=16,
             hr_spacing=0.9696, lr_spacing=2.4442, topk=5,
             seed_orient=None, seed_angle=None, verbose=True):
    """Locate a normalized HR volume inside a large LR volume (which may hold several
    samples). Pure search; loads only LR from disk. Returns a result dict:
      orient, angle_deg, center_lr (z,y,x voxels), roi (z0,z1,y0,y1,x0,x1),
      coarse_ncc, fine_ncc, box, win, candidates, and a _fine context for QC plotting.
    This is the importable entry point used by align_pair.py."""
    def log(*a):
        if verbose:
            print(*a)
    c = coarsen
    ratio = hr_spacing / lr_spacing
    hr = hr_norm
    f = hr_incircle_frac
    Yh, Xh = hr.shape[1], hr.shape[2]
    fy, fx = int(round(f * Yh)), int(round(f * Xh))
    hr_int = hr[:, (Yh - fy) // 2:(Yh - fy) // 2 + fy, (Xh - fx) // 2:(Xh - fx) // 2 + fx]
    box = tuple(max(1, int(round(s * ratio))) for s in hr.shape)            # full HR window in LR
    int_lr = tuple(max(1, int(round(s * ratio))) for s in hr_int.shape)
    base_int = downscale_to(hr_int, int_lr)

    angles = range(0, 360, rot_step)
    orient_set = {"id": ()} if no_flips else ORIENTS
    if seed_orient is not None:
        if seed_orient not in ORIENTS:
            raise ValueError(f"seed_orient '{seed_orient}' not in {list(ORIENTS)}")
        orient_set = {seed_orient: ORIENTS[seed_orient]}
        angles = [int(seed_angle) if seed_angle is not None else 0]
        log(f"[seed] fixed pose orient={seed_orient} angle={angles[0]} deg "
            f"-> searching position only")
    z0, z1 = lr_zrange
    log(f"[coarse] LR z{z0}:{z1} every {c}; sweeping {len(orient_set)} orient x "
        f"{len(angles)} angles = {len(orient_set) * len(angles)} matches")
    target_c = normalize(load_lr_coarse(lr_path, z0, z1, c))[0]
    base_c = downscale_to(base_int, tuple(max(1, s // c) for s in int_lr))
    (name, ang, corner, peak, ncc), win = sweep(base_c, target_c, orient_set, angles, crop_frac)
    log(f"[coarse] best: orient={name} angle={ang} deg   peak NCC={peak:.3f}")
    cands = []
    for p, v in topk_peaks(ncc, topk):
        cc = (z0 + (p[0] + win[0] / 2) * c, (p[1] + win[1] / 2) * c, (p[2] + win[2] / 2) * c)
        cands.append((v, tuple(int(round(x)) for x in cc)))
        log(f"[coarse]   cand: NCC={v:.3f}  centre LR (z,y,x)="
            f"({int(round(cc[0]))},{int(round(cc[1]))},{int(round(cc[2]))})")

    cz = z0 + (corner[0] + win[0] / 2) * c
    cy = (corner[1] + win[1] / 2) * c
    cx = (corner[2] + win[2] / 2) * c
    az, ay, ax_ = cz - box[0] / 2, cy - box[1] / 2, cx - box[2] / 2

    pad = refine_pad
    wz0 = max(0, int(az - pad)); wy0 = max(0, int(ay - pad)); wx0 = max(0, int(ax_ - pad))
    wz1, wy1, wx1 = int(az + box[0] + pad), int(ay + box[1] + pad), int(ax_ + box[2] + pad)
    log(f"[fine] re-reading full-res LR window z{wz0}:{wz1} y{wy0}:{wy1} x{wx0}:{wx1}")
    win_v = normalize(load_lr_window(lr_path, wz0, wz1, wy0, wy1, wx0, wx1))[0]
    fine_t = rotate_crop(orient(base_int, ORIENTS[name]), ang, crop_frac)
    if any(t > s for t, s in zip(fine_t.shape, win_v.shape)):
        log("[fine] window smaller than template; keeping coarse centre")
        fcz, fcy, fcx, peak_f = cz, cy, cx, peak
    else:
        m = validity_mask(win_v, fine_t.shape)
        nf = masked_corr(win_v, fine_t, m)
        cf = np.array(np.unravel_index(int(np.argmax(nf)), nf.shape)); peak_f = float(nf.max())
        fcz = wz0 + cf[0] + fine_t.shape[0] / 2
        fcy = wy0 + cf[1] + fine_t.shape[1] / 2
        fcx = wx0 + cf[2] + fine_t.shape[2] / 2
    log(f"[fine] refined NCC={peak_f:.3f}  centre (z,y,x)="
        f"({int(round(fcz))},{int(round(fcy))},{int(round(fcx))})")

    oz_, oy_, ox_ = fcz - box[0] / 2, fcy - box[1] / 2, fcx - box[2] / 2
    roi = [max(0, int(oz_ - margin)), int(oz_ + box[0] + margin),
           max(0, int(oy_ - margin)), int(oy_ + box[1] + margin),
           max(0, int(ox_ - margin)), int(ox_ + box[2] + margin)]
    return dict(orient=name, angle_deg=int(ang), center_lr=(fcz, fcy, fcx), roi=roi,
                coarse_ncc=float(peak), fine_ncc=float(peak_f), box=box, win=win,
                candidates=cands,
                _fine=dict(win_v=win_v, fine_t=fine_t, oz=oz_, oy=oy_, ox=ox_,
                           wz0=wz0, wy0=wy0, wx0=wx0))


def save_match_figure(res, out_dir):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fctx = res["_fine"]; win_v = fctx["win_v"]; fine_t = fctx["fine_t"]
        box = res["box"]; zc = fine_t.shape[0] // 2
        ty = int(max(0, fctx["oy"] + (box[1] - fine_t.shape[1]) / 2 - fctx["wy0"]))
        tx = int(max(0, fctx["ox"] + (box[2] - fine_t.shape[2]) / 2 - fctx["wx0"]))
        tz = int(min(max(0, fctx["oz"] - fctx["wz0"]) + zc, win_v.shape[0] - 1))
        lr_match = win_v[tz, ty:ty + fine_t.shape[1], tx:tx + fine_t.shape[2]]
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(fine_t[zc], cmap="gray")
        ax[0].set_title(f"HR interior, rot {res['angle_deg']} ({res['orient']})")
        ax[1].imshow(lr_match, cmap="gray")
        ax[1].set_title(f"LR at match (NCC={res['fine_ncc']:.2f})")
        for a in ax:
            a.axis("off")
        plt.tight_layout()
        p = os.path.join(out_dir, "match_check.png")
        plt.savefig(p, dpi=120); plt.close(fig)
        print(f"[out] visual -> {p}")
    except Exception as e:
        print(f"[warn] no figure: {e}")


def main():
    ap = argparse.ArgumentParser(description="Rotation-augmented HR-in-LR localization.")
    ap.add_argument("--hr"); ap.add_argument("--lr")
    ap.add_argument("--lr-zrange", nargs=2, type=int, default=[0, None], metavar=("z0", "z1"))
    ap.add_argument("--coarsen", type=int, default=5)
    ap.add_argument("--rot-step", type=int, default=10, help="In-plane angle step (deg).")
    ap.add_argument("--no-flips", action="store_true", help="Only identity orientation.")
    ap.add_argument("--crop-frac", type=float, default=0.6, help="Central crop for rotation.")
    ap.add_argument("--hr-incircle-frac", type=float, default=0.7)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--margin", type=int, default=40)
    ap.add_argument("--refine-pad", type=int, default=16)
    ap.add_argument("--hr-spacing", type=float, default=0.9696)
    ap.add_argument("--lr-spacing", type=float, default=2.4442)
    ap.add_argument("--out", default="localize/out")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        run_selftest(); return
    assert args.hr and args.lr, "provide --hr and --lr (or --selftest)"
    os.makedirs(args.out, exist_ok=True)
    print(f"[load] HR <- {args.hr}")
    hr = normalize(load_stack(args.hr))[0]
    res = localize(hr, args.lr, lr_zrange=tuple(args.lr_zrange), coarsen=args.coarsen,
                   rot_step=args.rot_step, no_flips=args.no_flips, crop_frac=args.crop_frac,
                   hr_incircle_frac=args.hr_incircle_frac, margin=args.margin,
                   refine_pad=args.refine_pad, hr_spacing=args.hr_spacing,
                   lr_spacing=args.lr_spacing, topk=args.topk, verbose=True)
    r = res["roi"]
    print(f"[roi]  --roi {r[0]} {r[1]} {r[2]} {r[3]} {r[4]} {r[5]}")
    print(f"[orient] winning orientation='{res['orient']}', in-plane angle={res['angle_deg']} deg")
    save_match_figure(res, args.out)


if __name__ == "__main__":
    main()
