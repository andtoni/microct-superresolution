#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_SEG = "/data/superresolution/Data/codeoutput/segmentations"
DEFAULT_ALIGNED = "/data/superresolution/Data/codeoutput/aligned"


def _norm(a, lo=1, hi=99):
    a = a.astype(np.float32)
    p1, p99 = np.percentile(a, [lo, hi])
    if p99 <= p1:
        p99 = p1 + 1.0
    return np.clip((a - p1) / (p99 - p1), 0, 1)


def _zshape(path):
    with tifffile.TiffFile(str(path)) as tf:
        return tf.series[0].shape


def main(argv=None):
    ap = argparse.ArgumentParser(description="mid-slice raw+label overlay QC PNGs")
    ap.add_argument("--seg-root", default=DEFAULT_SEG)
    ap.add_argument("--aligned-root", default=DEFAULT_ALIGNED)
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--n-slices", type=int, default=3)
    ap.add_argument("--label-suffix", default="_labels.tif")
    args = ap.parse_args(argv)

    seg = Path(args.seg_root)
    out = Path(args.out) if args.out else seg / "_qc"
    out.mkdir(parents=True, exist_ok=True)
    only = {s.strip() for s in args.only.split(",")} if args.only else None

    for d in sorted(p for p in seg.iterdir() if p.is_dir() and p.name != "_qc"):
        if only and d.name not in only:
            continue
        lab_p = d / f"{d.name}{args.label_suffix}"
        hr_p = Path(args.aligned_root) / d.name / "hr.tif"
        if not lab_p.exists() or not hr_p.exists():
            print(f"[skip] {d.name}: missing label or hr"); continue

        Z = _zshape(hr_p)[0]
        zs = [int(round(f * (Z - 1))) for f in np.linspace(0.2, 0.8, args.n_slices)]
        n = len(zs)
        fig, axes = plt.subplots(2, n, figsize=(4 * n, 8.2), squeeze=False)
        for j, z in enumerate(zs):
            raw = _norm(np.asarray(tifffile.imread(str(hr_p), key=z), dtype=np.float32))
            lab = (np.asarray(tifffile.imread(str(lab_p), key=z)) > 0)
            rgb = np.dstack([raw, raw, raw])
            ov = rgb.copy()
            ov[lab] = 0.5 * rgb[lab] + 0.5 * np.array([1.0, 0.15, 0.15])
            axes[0, j].imshow(raw, cmap="gray", vmin=0, vmax=1)
            axes[0, j].set_title(f"z={z}  raw", fontsize=9)
            axes[1, j].imshow(ov)
            axes[1, j].set_title(f"solid frac {lab.mean():.3f}", fontsize=9)
            for r in (0, 1):
                axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
        fig.suptitle(f"{d.name}  -- red = label solid; should sit on material, not pores", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        op = out / f"{d.name}_overlay.png"
        fig.savefig(str(op), dpi=110); plt.close(fig)
        print(f"{d.name:9s} z={zs} -> {op}")

    print(f"\nPNGs in {out} -- red on fibres/material = correct; red in pores = inverted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
