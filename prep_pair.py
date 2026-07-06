#!/usr/bin/env python3
r"""
prep_pair.py  --  turn an aligned (lr_input, hr_target) pair into a training-ready tile.

Reads an aligned-pair folder (lr_input.tif / hr_target.tif / overlap_mask.tif) and produces,
on the same HR grid:

  structure_mask.tif  high local-variance voxels = material/air interfaces -- the places
                      super-resolution actually matters. Drives foreground-biased patch
                      sampling so the model trains on scaffold, not air.
  scaffold_mask.tif   the connected scaffold-following region (dilated structure, largest
                      component, intersected with real-LR backing) -- the region we compute
                      masked validation metrics inside, so scores reflect the scaffold and
                      not background or the circular-FOV corners.
  hr.tif / lr.tif     the pair tight-cropped to the scaffold bounding box (+margin), trimming
                      the surrounding air -> smaller files, faster training I/O.
  prep_meta.json      crop box, foreground fractions, shapes.
  prep_qc.png         a slice with the masks overlaid -- the human check.

Everything is morphology-adaptive: the structure threshold is Otsu on the local-variance
field, so the laser wall and the porogen network both work with no per-sample tuning.
Open tools only (scipy + scikit-image).

Usage:
  uv run python prep_pair.py --dir "C:\...\codeoutput\aligned\vcr276"
"""
import argparse, json, os
import numpy as np
import tifffile
from scipy.ndimage import (uniform_filter, binary_dilation, binary_closing,
                           binary_erosion, binary_fill_holes, label)
from skimage.filters import threshold_otsu

from sr_validation_harness import load_stack, save_stack


def local_std(vol, w=5):
    m1 = uniform_filter(vol, w)
    m2 = uniform_filter(vol * vol, w)
    return np.sqrt(np.clip(m2 - m1 * m1, 0, None)).astype(np.float32)


def largest_component(mask):
    lab, n = label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    return lab == int(sizes.argmax())


def save_mask(mask, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tifffile.imwrite(path, mask.astype(np.uint8))


def bbox(mask, margin, shape):
    idx = np.where(mask)
    lo = [max(0, int(idx[i].min()) - margin) for i in range(3)]
    hi = [min(shape[i], int(idx[i].max()) + 1 + margin) for i in range(3)]
    return lo, hi


def make_masks(hr, lr, overlap, struct_window=5, dilate=4, close_iter=2, fov_erode=6):
    """Returns (structure_mask, scaffold_mask). Both bool, on the input grid.

    The circular reconstruction boundary is a sharp bright->black edge that the variance
    detector would otherwise flag as 'structure'. We erode the in-FOV region inward
    (in-plane only, so the top/bottom slices aren't trimmed) by `fov_erode` voxels so that
    rim is excluded before detection. fov_erode is kept > dilate so growing the scaffold
    cannot push it back onto the boundary.
    """
    in_fov = (overlap > 0.5) & (hr > 0.03)        # in-FOV, LR-backed
    # Define the FOV as a *filled* region so interior pores (however dark) count as in-FOV;
    # only the outer reconstruction rim is then eroded away. (Filling first is what makes
    # this robust: eroding the raw bright-voxel mask would chew away the thin scaffold.)
    fov = binary_fill_holes(in_fov)
    if fov_erode > 0:
        active = binary_erosion(fov, structure=np.ones((1, 3, 3), bool),
                                iterations=fov_erode)
    else:
        active = fov
    std = local_std(hr, struct_window)
    vals = std[active]
    thr = float(threshold_otsu(vals[vals > 0])) if vals.size else 0.02
    thr = max(thr, 0.012)
    structure = (std > thr) & active

    scaffold = binary_dilation(structure, iterations=dilate)
    if close_iter:
        scaffold = binary_closing(scaffold, iterations=close_iter)
    scaffold = largest_component(scaffold) & active
    return structure, scaffold, thr


def prep(dir_, struct_window=5, dilate=4, crop_margin=16, fov_erode=6, verbose=True):
    hr = load_stack(os.path.join(dir_, "hr_target.tif"))
    lr = load_stack(os.path.join(dir_, "lr_input.tif"))
    overlap = load_stack(os.path.join(dir_, "overlap_mask.tif"))
    if verbose:
        print(f"[prep] loaded grid {hr.shape}")

    structure, scaffold, thr = make_masks(hr, lr, overlap, struct_window, dilate,
                                          fov_erode=fov_erode)
    if not scaffold.any():
        raise RuntimeError("empty scaffold mask -- check the pair / thresholds")

    lo, hi = bbox(scaffold, crop_margin, hr.shape)
    sl = tuple(slice(lo[i], hi[i]) for i in range(3))
    hr_c, lr_c = hr[sl], lr[sl]
    struct_c, scaff_c, over_c = structure[sl], scaffold[sl], (overlap[sl] > 0.5)

    save_stack(hr_c, os.path.join(dir_, "hr.tif"))
    save_stack(lr_c, os.path.join(dir_, "lr.tif"))
    save_mask(struct_c, os.path.join(dir_, "structure_mask.tif"))
    save_mask(scaff_c, os.path.join(dir_, "scaffold_mask.tif"))
    save_mask(over_c, os.path.join(dir_, "overlap_crop.tif"))

    meta = dict(crop_lo=lo, crop_hi=hi, crop_shape=list(hr_c.shape),
                struct_threshold=thr,
                structure_frac=float(struct_c.mean()),
                scaffold_frac=float(scaff_c.mean()),
                structure_frac_in_scaffold=float(struct_c[scaff_c].mean()) if scaff_c.any() else 0.0)
    with open(os.path.join(dir_, "prep_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    if verbose:
        print(f"[prep] crop {hr.shape} -> {hr_c.shape}  (margin {crop_margin})")
        print(f"[prep] structure thr={thr:.3f}  structure frac={meta['structure_frac']:.3f}  "
              f"scaffold frac={meta['scaffold_frac']:.3f}")

    save_prep_qc(hr_c, struct_c, scaff_c, dir_)
    print(f"[out] training-ready tile + masks -> {dir_}")
    return meta


def save_prep_qc(hr, structure, scaffold, dir_):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        cover = scaffold.reshape(scaffold.shape[0], -1).sum(1)
        z = int(cover.argmax()) if cover.max() > 0 else scaffold.shape[0] // 2
        h = hr[z]
        rgb_s = np.dstack([np.clip(h + 0.6 * structure[z], 0, 1), h, h])  # structure in red
        sc = scaffold[z].astype(np.float32)
        rgb_e = np.dstack([h, np.clip(h + 0.4 * sc, 0, 1), h])            # scaffold in green
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(h, cmap="gray", vmin=0, vmax=1); ax[0].set_title("HR")
        ax[1].imshow(rgb_s); ax[1].set_title("structure mask (sampling)")
        ax[2].imshow(rgb_e); ax[2].set_title("scaffold mask (metrics / crop)")
        for a in ax:
            a.axis("off")
        plt.tight_layout()
        p = os.path.join(dir_, "prep_qc.png")
        plt.savefig(p, dpi=120); plt.close(fig)
        print(f"[out] QC figure -> {p}")
    except Exception as e:
        print(f"[warn] QC figure failed: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", help="aligned-pair folder")
    ap.add_argument("--struct-window", type=int, default=5)
    ap.add_argument("--dilate", type=int, default=4)
    ap.add_argument("--crop-margin", type=int, default=16)
    ap.add_argument("--fov-erode", type=int, default=6,
                    help="in-plane voxels to erode the FOV inward, dropping the circular "
                         "reconstruction boundary (set 0 for non-FOV-limited scans)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.dir:
        ap.error("--dir is required (unless --selftest)")
    prep(args.dir, args.struct_window, args.dilate, args.crop_margin, args.fov_erode)


def selftest():
    from sr_validation_harness import make_porous_phantom, normalize
    os.makedirs("/tmp/prep_t", exist_ok=True)
    hr = normalize(make_porous_phantom((60, 160, 160), feature_scale=5.0, seed=4))[0]
    big = np.zeros((60, 200, 200), np.float32); big[:, 20:180, 20:180] = hr
    # mid-gray pores inside the FOV (as in a real scan), pure black only outside the disc
    yy, xx = np.ogrid[:200, :200]
    disc = ((yy - 100) ** 2 + (xx - 100) ** 2) <= 95 ** 2
    pore = big < 0.1
    big[pore & disc[None]] = 0.35
    big[~np.broadcast_to(disc[None], big.shape)] = 0.0
    lr = np.clip(big + 0.03 * np.random.default_rng(0).standard_normal(big.shape).astype('float32'), 0, 1)
    over = np.ones_like(big)
    for nm, v in [("hr_target.tif", big), ("lr_input.tif", lr), ("overlap_mask.tif", over)]:
        save_stack(v, f"/tmp/prep_t/{nm}")
    m = prep("/tmp/prep_t", crop_margin=8, verbose=True)
    ok = (0.05 < m["scaffold_frac"] < 0.95) and (m["crop_shape"][1] < 200) and \
         (m["structure_frac_in_scaffold"] > m["structure_frac"])
    print(f"[selftest] scaffold_frac={m['scaffold_frac']:.3f}  crop={m['crop_shape']}  "
          f"struct-in-scaffold={m['structure_frac_in_scaffold']:.3f}")
    print("[selftest]", "OK" if ok else "CHECK")


if __name__ == "__main__":
    main()
