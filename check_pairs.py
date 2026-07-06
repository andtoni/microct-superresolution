#!/usr/bin/env python3
r"""
check_pairs.py -- integrity audit of the aligned/prepped pairs that feed SR training.

Answers, per architecture:
  * Are overlap_crop / scaffold_mask / structure_mask actually binary masks? (and how full?)
  * Is the scaffold sampling region contained in the overlap? (the 80% foreground patches
    are drawn from scaffold_mask, so scaffold-outside-overlap should be ~0).
  * (--full) Are hr.tif / lr.tif real greyscale, do their shapes match the masks
    (train_sr_real asserts this), and -- the key one -- does lr.tif carry real data
    OUTSIDE the overlap mask? If yes, the ~20% "sample-anywhere" patches are still real
    LR->HR pairs (zero contamination). If ~0, the non-overlap region has no LR backing,
    so a fraction of anywhere-patches are flat-input -> mild blur bias (the SAFE direction).

Masks are 785 MB each (fast). hr/lr are 3.14 GB float32 each, so --full is slower
(~30-60 s per architecture of I/O); start without --full.

Usage (uv):
  uv run python check_pairs.py --root "C:\...\codeoutput\aligned"
  uv run python check_pairs.py --root "C:\...\codeoutput\aligned" --full
  uv run python check_pairs.py --root "C:\...\codeoutput\aligned" --full --only pel16 laser50
"""
from __future__ import annotations
import argparse, os, time
import numpy as np
import tifffile

MASKS = ("overlap_crop.tif", "scaffold_mask.tif", "structure_mask.tif")


def is_binary(a):
    u = np.unique(a[:: max(1, a.size // 2_000_000)])  # subsample for speed
    return set(np.unique(u).tolist()) <= {0, 1}


def read(path):
    return tifffile.imread(path)


def audit_arch(root, arch, full):
    d = os.path.join(root, arch)
    line = {"arch": arch}
    # ---- masks ----
    ov = sc = st = None
    if os.path.exists(os.path.join(d, "overlap_crop.tif")):
        ov = read(os.path.join(d, "overlap_crop.tif")) > 0
        line["overlap%"] = 100 * ov.mean()
        line["ov_bin"] = is_binary(read(os.path.join(d, "overlap_crop.tif")))
    if os.path.exists(os.path.join(d, "scaffold_mask.tif")):
        sc = read(os.path.join(d, "scaffold_mask.tif")) > 0
        line["scaffold%"] = 100 * sc.mean()
    if os.path.exists(os.path.join(d, "structure_mask.tif")):
        st = read(os.path.join(d, "structure_mask.tif")) > 0
        line["structure%"] = 100 * st.mean()
    if sc is not None and ov is not None:
        line["scaf_outside_overlap%"] = 100 * (sc & ~ov).sum() / max(sc.sum(), 1)
    # ---- greyscale (slow) ----
    if full:
        hp, lp = os.path.join(d, "hr.tif"), os.path.join(d, "lr.tif")
        if os.path.exists(hp) and os.path.exists(lp):
            hr = read(hp).astype(np.float32)
            lr = read(lp).astype(np.float32)
            line["hr_dtype"] = str(read(hp).dtype)
            line["hr_shape"] = "x".join(map(str, hr.shape))
            line["hr_min"] = float(hr.min()); line["hr_max"] = float(hr.max())
            line["hr>0.03%"] = 100 * (hr > 0.03).mean()
            line["lr>0%"] = 100 * (lr > 1e-6).mean()
            # shapes must match (train_sr_real asserts hr==lr==scaffold)
            shapes = {hr.shape, lr.shape}
            if sc is not None:
                shapes.add(sc.shape)
            line["shapes_match"] = (len(shapes) == 1)
            # KEY: is there real LR data outside the overlap mask?
            if ov is not None and (~ov).sum() > 0:
                line["LR_real_outside_overlap%"] = 100 * ((lr > 1e-6) & ~ov).sum() / (~ov).sum()
    return line


def fmt(line):
    a = line.pop("arch")
    parts = []
    for k, v in line.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.1f}")
        else:
            parts.append(f"{k}={v}")
    return f"{a:9s} " + "  ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="aligned dir")
    ap.add_argument("--full", action="store_true", help="also read hr.tif/lr.tif (slow)")
    ap.add_argument("--only", nargs="*", default=None, help="subset of architectures")
    args = ap.parse_args()

    archs = []
    for name in sorted(os.listdir(args.root)):
        d = os.path.join(args.root, name)
        if not os.path.isdir(d) or name == "sweep":
            continue
        if any(os.path.exists(os.path.join(d, m)) for m in MASKS) or \
           os.path.exists(os.path.join(d, "hr.tif")):
            if args.only is None or name in args.only:
                archs.append(name)

    print(f"auditing {len(archs)} architectures under {args.root}"
          f"{'  (+greyscale)' if args.full else ''}\n")
    for arch in archs:
        t = time.time()
        try:
            line = audit_arch(args.root, arch, args.full)
            print(fmt(line) + f"   [{time.time()-t:.0f}s]")
        except Exception as e:
            print(f"{arch:9s} ERROR: {e}")

    print("\nread me:")
    print("  overlap%/scaffold%/structure% in (0,100) and *_bin=True  -> these are binary masks (expected)")
    print("  overlap%=100 (e.g. laser50) -> crop fully LR-backed; overlap% low (e.g. pel16) -> crop extends past overlap")
    print("  scaf_outside_overlap% ~0 -> the foreground (80%) training patches are all LR-backed (good)")
    print("  --full: hr_max>0 and hr>0.03% sensible -> hr.tif is real greyscale; shapes_match=True -> trainer assert passes")
    print("  --full: LR_real_outside_overlap% high -> even the 20% anywhere-patches are real LR->HR pairs (zero contamination)")
    print("          LR_real_outside_overlap% ~0   -> non-overlap has no LR; those patches are flat-input -> mild blur bias only")


if __name__ == "__main__":
    main()
