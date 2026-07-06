#!/usr/bin/env python3
r"""
align_pair.py  --  align ANY high-res / low-res micro-CT pair, with open tools only
===================================================================================
Given a high-resolution scan (HR) and a low-resolution scan (LR) of the same specimen
-- where the LR may be a larger field of view that contains several samples, and the two
scans may differ by an in-plane rotation and/or an axis flip -- this produces a clean,
co-registered training/validation pair on a single common voxel grid:

    lr_input.tif    real LR, localized + rigidly registered + resampled onto the HR grid
                    (this is the model INPUT)
    hr_target.tif   the HR, in the matched orientation, normalized (the model TARGET; only
                    lossless flips are ever applied to it -- never an interpolating rotation)
    overlap_mask.tif    where real LR data backs each voxel
    hr_to_lr.tfm        the recovered rigid transform (an open, reusable alignment record)
    align_meta.json     voxel sizes, the recovered LR crop box (= the DeepScout crop region),
                        orientation, angles, and every quality number
    align_qc.png        HR / registered-LR / overlay at a populated slice -- the human check

Nothing here needs licensed software: localization is masked normalized cross-correlation
(scikit-image), registration is rigid Mattes-MI (SimpleITK). The same command works for
porogen, laser, electrospun, or any future pair -- only --hr/--lr/--name change.

Usage, writing into the clean material folder:
  uv run python align_pair.py `
    --hr  "C:\...\Data\ct\hr_l" `
    --lr  "C:\...\Data\ct\lr_l" `
    --name laser50 `
    --out "/data/superresolution/Data/codeoutput/aligned"
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from scipy.ndimage import rotate as ndrotate, uniform_filter, gaussian_filter

from sr_validation_harness import load_stack, normalize, save_stack
from localize_hr import localize, ORIENTS, orient, load_lr_window
from register_volumes import register, resample_onto, to_sitk
import SimpleITK as sitk


def masked_ncc(a, b, mask):
    """Pearson NCC over masked voxels -- the post-registration fidelity check."""
    m = mask > 0.5
    if int(m.sum()) < 100:
        return -1.0
    av, bv = a[m].astype(np.float64), b[m].astype(np.float64)
    av -= av.mean(); bv -= bv.mean()
    d = np.sqrt((av * av).sum() * (bv * bv).sum())
    return float((av * bv).sum() / d) if d > 0 else -1.0


def _central(arr, frac=0.6):
    sl = tuple(slice(int(s * (1 - frac) / 2), int(s * (1 - frac) / 2) + max(1, int(s * frac)))
               for s in arr.shape)
    return arr[sl]


def alignment_quality(hr, lr, mask, blur_sigma=1.3, fg_thresh=0.05):
    """Score alignment where it is meaningful, and -- crucially -- test it *relatively*.

    A global Pearson NCC is misleading for two reasons unrelated to alignment: (1) the HR's
    circular FOV puts ~20% zeros in the corners against real LR signal; (2) the HR is
    sharp+grainy and the LR smooth, so Pearson penalises the resolution gap. So we score on
    the in-FOV foreground and at matched blur. But the *absolute* value is also morphology-
    dependent: where the LR genuinely lacks the HR's fine detail (e.g. a laser-machined
    fibrous wall, whose texture is below LR resolution), even a perfect alignment gives a low
    correlation. So the verdict is based on a PEAK-SHARPNESS test -- nudge the LR a few voxels
    in every direction and confirm the correlation drops -- which asks "is this the best fit?"
    rather than "is the correlation high?", and is therefore morphology-independent.
    """
    fg = (hr > fg_thresh) & (mask > 0.5)
    ncc_fg = masked_ncc(hr, lr, fg)
    ncc_blur = masked_ncc(gaussian_filter(hr, blur_sigma), lr, fg)
    step = max(1, hr.shape[0] // 40)
    prof = [masked_ncc(hr[z], lr[z], fg[z]) for z in range(0, hr.shape[0], step)
            if int(fg[z].sum()) > 200]
    prof = np.array(prof) if prof else np.array([ncc_fg])

    # peak-sharpness: is zero-shift a clear local maximum? (computed on a central crop)
    hc, lc, fc = _central(hr), _central(lr), _central(fg)
    base = masked_ncc(hc, lc, fc)
    drops = [base - masked_ncc(hc, np.roll(lc, s, axis=ax), fc)
             for ax in (0, 1, 2) for s in (3, -3, 6, -6)]
    min_drop, mean_drop = float(min(drops)), float(np.mean(drops))
    locked = (base > 0.08) and (min_drop > 0.01)   # every nudge degrades the fit -> aligned
    verdict = "PASS" if locked else "REVIEW" if min_drop > -0.005 else "FAIL"

    return dict(ncc_foreground=ncc_fg, ncc_blur_matched=ncc_blur,
                slice_ncc_median=float(np.median(prof)),
                slice_ncc_p10=float(np.percentile(prof, 10)),
                peak_base_ncc=base, peak_min_drop=min_drop, peak_mean_drop=mean_drop,
                locked=bool(locked), verdict=verdict)


def align(hr_norm, lr_path, hr_voxel, lr_voxel, loc_kwargs, iters, sampling, verbose=True):
    """Returns (lr_on_hr, hr_target, mask, transform, meta_dict). hr_norm is normalized HR.
    lr_path may be a folder or a multipage tif (load_lr_window handles both)."""
    res = localize(hr_norm, lr_path, hr_spacing=hr_voxel, lr_spacing=lr_voxel,
                   verbose=verbose, **loc_kwargs)
    z0, z1, y0, y1, x0, x1 = res["roi"]
    if verbose:
        print(f"[align] LR crop box (voxels) z{z0}:{z1} y{y0}:{y1} x{x0}:{x1}")

    # real LR sub-volume around the located specimen
    lr_roi = normalize(load_lr_window(lr_path, z0, z1, y0, y1, x0, x1))[0]

    # bring the two scans into the SAME frame WITHOUT interpolating the HR target:
    #   localizer found  R(ang) . O . HR  ~=  LR   =>   O.HR  ~=  R(-ang) . LR
    # so we flip the HR (lossless) and de-rotate the LR (the input; interpolation is fine).
    ang = res["angle_deg"]
    hr_target = orient(hr_norm, ORIENTS[res["orient"]])                 # lossless flips only
    if ang % 360 != 0:
        lr_roi = ndrotate(lr_roi, -ang, axes=(1, 2), reshape=True, order=1).astype(np.float32)

    fixed = to_sitk(hr_target, hr_voxel)        # HR grid (pristine target)
    moving = to_sitk(lr_roi, lr_voxel)          # real LR, to be resampled onto the HR grid
    tfm, mi = register(fixed, moving, iters=iters, sampling=sampling)
    moved, mask_img = resample_onto(fixed, moving, tfm)
    lr_on_hr = sitk.GetArrayFromImage(moved).astype(np.float32)
    mask = (sitk.GetArrayFromImage(mask_img) > 0.5).astype(np.float32)

    ncc_global = masked_ncc(hr_target, lr_on_hr, mask)
    q = alignment_quality(hr_target, lr_on_hr, mask)
    overlap = 100.0 * float(mask.mean())
    meta = dict(orient=res["orient"], angle_deg=ang,
                lr_crop_box=dict(z0=z0, z1=z1, y0=y0, y1=y1, x0=x0, x1=x1),
                center_lr=[float(v) for v in res["center_lr"]],
                coarse_ncc=res["coarse_ncc"], fine_ncc=res["fine_ncc"],
                ncc_global=ncc_global, ncc_foreground=q["ncc_foreground"],
                ncc_blur_matched=q["ncc_blur_matched"],
                slice_ncc_median=q["slice_ncc_median"], slice_ncc_p10=q["slice_ncc_p10"],
                peak_base_ncc=q["peak_base_ncc"], peak_min_drop=q["peak_min_drop"],
                peak_mean_drop=q["peak_mean_drop"], locked=q["locked"],
                postreg_MI=float(mi), overlap_pct=overlap,
                hr_voxel_um=hr_voxel, lr_voxel_um=lr_voxel,
                factor=lr_voxel / hr_voxel,
                hr_shape=list(hr_target.shape), grid_shape=list(lr_on_hr.shape),
                verdict=q["verdict"])
    if verbose:
        print(f"[align] alignment NCC: blur-matched={q['ncc_blur_matched']:.3f}  "
              f"foreground={q['ncc_foreground']:.3f}  (global={ncc_global:.3f})")
        print(f"[align] per-slice foreground NCC: median={q['slice_ncc_median']:.3f}  "
              f"p10={q['slice_ncc_p10']:.3f}")
        print(f"[align] peak-sharpness: base={q['peak_base_ncc']:.3f}  "
              f"min nudge drop={q['peak_min_drop']:+.3f}  "
              f"-> {'LOCKED' if q['locked'] else 'NOT a clean local max'}")
        print(f"[align] overlap={overlap:.1f}%  MI={mi:.3f}  -> {q['verdict']}")
    return lr_on_hr, hr_target, mask, tfm, meta


def save_qc(hr, lr, mask, out_dir):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from scipy.ndimage import sobel
        cover = (mask > 0.5).reshape(mask.shape[0], -1).sum(1)
        z = int(cover.argmax()) if cover.max() > 0 else mask.shape[0] // 2
        h, l = hr[z], lr[z]
        edge = np.hypot(sobel(h, 0), sobel(h, 1)); edge /= (edge.max() + 1e-6)
        rgb = np.dstack([np.clip(l + edge, 0, 1), l, l])
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(h, cmap="gray", vmin=0, vmax=1); ax[0].set_title("HR target")
        ax[1].imshow(l, cmap="gray", vmin=0, vmax=1); ax[1].set_title("LR registered -> HR grid")
        ax[2].imshow(rgb); ax[2].set_title("overlay: HR edges (red) on LR")
        for a in ax:
            a.axis("off")
        plt.tight_layout()
        p = os.path.join(out_dir, "align_qc.png")
        plt.savefig(p, dpi=120); plt.close(fig)
        print(f"[out] QC figure -> {p}")
    except Exception as e:
        print(f"[warn] QC figure failed: {e}")


def main():
    ap = argparse.ArgumentParser(description="Align any HR/LR micro-CT pair (open tools only).")
    ap.add_argument("--hr"); ap.add_argument("--lr")
    ap.add_argument("--name", default="pair", help="sample label (subfolder under --out)")
    ap.add_argument("--out", default="aligned", help="output root (point this at Nextcloud)")
    ap.add_argument("--hr-voxel", type=float, default=0.9696)
    ap.add_argument("--lr-voxel", type=float, default=2.4442)
    # localizer controls (search scope + thoroughness)
    ap.add_argument("--lr-zrange", nargs=2, type=int, default=None, metavar=("z0", "z1"),
                    help="restrict the LR z-search (e.g. one half of a stacked scan)")
    ap.add_argument("--hr-zrange", nargs=2, type=int, default=None, metavar=("z0", "z1"),
                    help="crop the HR template in z before alignment (drop tape/empty slices)")
    ap.add_argument("--coarsen", type=int, default=5)
    ap.add_argument("--rot-step", type=int, default=10)
    ap.add_argument("--no-flips", action="store_true")
    ap.add_argument("--hr-incircle-frac", type=float, default=0.7)
    ap.add_argument("--margin", type=int, default=40)
    # pose seeding for co-mounted samples (share orientation + XY, differ only in z)
    ap.add_argument("--seed-from", default=None,
                    help="path to a locked sibling aligned dir; fixes orient+angle from its "
                         "align_meta.json so only position is searched")
    ap.add_argument("--seed-orient", default=None, help="override: fix orientation (e.g. flipX)")
    ap.add_argument("--seed-angle", type=int, default=None, help="override: fix angle (degrees)")
    # registration controls
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--sampling", type=float, default=0.02,
                    help="fraction of HR voxels per metric eval; the pair starts "
                         "well-seeded so ~0.02 is plenty (drop to 0.01 if slow)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.hr or not args.lr:
        ap.error("--hr and --lr are required (unless --selftest)")

    out_dir = os.path.join(args.out, args.name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[load] HR <- {args.hr}")
    _hr_raw = load_stack(args.hr)
    if args.hr_zrange is not None:
        z0h, z1h = args.hr_zrange
        _hr_raw = _hr_raw[z0h:z1h]
        print(f"[hr-crop] HR z{z0h}:{z1h} -> shape {_hr_raw.shape}")
    hr_norm = normalize(_hr_raw)[0]
    loc_kwargs = dict(coarsen=args.coarsen, rot_step=args.rot_step, no_flips=args.no_flips,
                      hr_incircle_frac=args.hr_incircle_frac, margin=args.margin)
    if args.lr_zrange is not None:
        loc_kwargs["lr_zrange"] = tuple(args.lr_zrange)
    # pose seeding: fix orient+angle from a locked sibling (co-mounted sample) or directly
    seed_orient, seed_angle = args.seed_orient, args.seed_angle
    if args.seed_from is not None:
        with open(os.path.join(args.seed_from, "align_meta.json")) as f:
            sm = json.load(f)
        seed_orient, seed_angle = sm["orient"], sm["angle_deg"]
        print(f"[seed] pose from {args.seed_from}: orient={seed_orient} angle={seed_angle}")
    if seed_orient is not None:
        loc_kwargs["seed_orient"] = seed_orient
        loc_kwargs["seed_angle"] = seed_angle if seed_angle is not None else 0

    lr_on_hr, hr_target, mask, tfm, meta = align(
        hr_norm, args.lr, args.hr_voxel, args.lr_voxel, loc_kwargs, args.iters, args.sampling)

    save_stack(lr_on_hr, os.path.join(out_dir, "lr_input.tif"))
    save_stack(hr_target, os.path.join(out_dir, "hr_target.tif"))
    save_stack(mask, os.path.join(out_dir, "overlap_mask.tif"))
    sitk.WriteTransform(tfm, os.path.join(out_dir, "hr_to_lr.tfm"))
    with open(os.path.join(out_dir, "align_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    save_qc(hr_target, lr_on_hr, mask, out_dir)
    print(f"[out] aligned pair + metadata -> {out_dir}")
    if meta["verdict"] != "PASS":
        print(f"[!] verdict={meta['verdict']}: inspect align_qc.png before using this pair "
              f"for training.")


def selftest():
    """Embed a flipped + rotated HR-like volume inside a larger LR-like volume at coarse
    resolution, then check align() recovers a high-fidelity registered pair end-to-end."""
    rng = np.random.default_rng(0)
    from sr_validation_harness import make_porous_phantom
    hr = normalize(make_porous_phantom((80, 200, 200), feature_scale=5.0, seed=3))[0]
    ratio = 0.9696 / 2.4442
    # build an LR volume: degrade HR, flip + rotate it, drop it into a bigger air volume
    from skimage.transform import resize
    lr_small = resize(hr, tuple(max(1, int(round(s * ratio))) for s in hr.shape),
                      order=1, anti_aliasing=True, preserve_range=True).astype(np.float32)
    true_ang = 12
    lr_or = np.flip(lr_small, axis=2)                                   # flipX
    lr_or = ndrotate(lr_or, true_ang, axes=(1, 2), reshape=False, order=1)
    big = np.zeros((lr_small.shape[0] + 20, 260, 260), np.float32)
    oz, oy, ox = 8, 40, 50
    big[oz:oz + lr_or.shape[0], oy:oy + lr_or.shape[1], ox:ox + lr_or.shape[2]] = lr_or
    big += 0.01 * rng.standard_normal(big.shape).astype(np.float32)
    big = np.clip(big, 0, 1)
    os.makedirs("/tmp/ap_lr", exist_ok=True)
    import tifffile
    for i, s in enumerate(big):
        tifffile.imwrite(f"/tmp/ap_lr/{i:04d}.tif", s)
    loc = dict(coarsen=2, rot_step=6, hr_incircle_frac=0.8, margin=10)
    lr_on_hr, hr_t, mask, tfm, meta = align(hr, "/tmp/ap_lr", 0.9696, 2.4442, loc,
                                            iters=200, sampling=0.2, verbose=True)
    print(f"[selftest] true: flipX + {true_ang} deg | found: {meta['orient']} "
          f"{meta['angle_deg']} deg | locked={meta['locked']} base={meta['peak_base_ncc']:.3f} "
          f"({meta['verdict']})")
    print("[selftest]", "OK" if (meta["orient"] == "flipX" and meta["locked"]) else
          "CHECK orientation/rotation/lock logic")


if __name__ == "__main__":
    main()
