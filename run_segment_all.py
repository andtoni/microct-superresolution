#!/usr/bin/env python3
"""
run_segment_all.py -- bootstrap segmentation labels across ALL architectures at once.

Runs `segment.bootstrap_labels` on each real-pair architecture's HR volume (best quality
-> best labels; the trained segmenter is then applied to HR and SR identically), writing
filled-fibre labels, an ambiguity map, and a QC overlay per architecture, plus a summary
table you can scan before refining anything.

Usage (uv):
  uv run python run_segment_all.py                 # uses CONFIG below
  uv run python run_segment_all.py --root "<aligned dir>"   # auto-discover <arch>/HR
  uv run python run_segment_all.py --decup-mode subtract --threshold li

Then: skim each <arch>_ambiguity.tif, fix the flagged voxels with Ilastik scribbles or
TomoSAM prompts where needed, and feed the (raw, label) pairs to train_seg.py.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

import morphometry as mp
import segment as sg
try:
    import provenance as prov
    _HAVE_PROV = True
except Exception:
    _HAVE_PROV = False

# ----------------------------------------------------------------------- CONFIG
# Value = (HR volume path, voxel_um).
_DATA = "/data/superresolution/Data/codeoutput/aligned"
CONFIG = {
    "pel16":  (f"{_DATA}/pel16/HR",  0.9696),
    "pel18":  (f"{_DATA}/pel18/HR",  0.9696),
    "pel20":  (f"{_DATA}/pel20/HR",  0.9696),
    "laser50": (f"{_DATA}/laser50/HR", 0.9696),
    "vcr276": (f"{_DATA}/vcr276/HR", 0.9696),
}


def discover(root, vol_name=None):
    root = Path(root)
    cfg = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if vol_name:                                   # explicit file per arch folder
            cand = d / vol_name
            if cand.exists():
                cfg[d.name] = (str(cand), 0.9696)
            continue
        for sub in ("HR", "hr", "high", ""):
            cand = d / sub if sub else d
            if cand.exists():
                cfg[d.name] = (str(cand), 0.9696)
                break
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description="bootstrap labels across all architectures")
    ap.add_argument("--root", default=None, help="aligned dir to auto-discover <arch>/HR")
    ap.add_argument("--vol-name", default=None,
                    help="filename inside each <arch> folder to use (e.g. hr.tif, overlap_crop.tif)")
    ap.add_argument(
        "--out",
        default="/data/superresolution/Data/codeoutput/segmentations",
        help="output root",
    )
    ap.add_argument("--decup-sigma", type=float, default=0.25)
    ap.add_argument("--decup-mode", default="divide", choices=["divide", "subtract"])
    ap.add_argument("--mode", default="local", choices=list(sg.MODES),
                    help="classical variant for the bootstrap: otsu | otsu_decup | "
                         "local (Sauvola, default) | ridge | fixed")
    ap.add_argument("--local-window", type=int, default=25)
    ap.add_argument("--sauvola-k", type=float, default=0.2)
    ap.add_argument("--no-fill", action="store_true",
                    help="disable rim-fill (use for solid-wall/porogen archs so pores aren't filled)")
    ap.add_argument("--level", type=float, default=None)
    ap.add_argument("--min-size", type=int, default=27)
    ap.add_argument("--only", default=None, help="comma list to restrict architectures")
    args = ap.parse_args(argv)

    cfg = discover(args.root, vol_name=args.vol_name) if args.root else dict(CONFIG)
    if args.only:
        keep = {s.strip() for s in args.only.split(",")}
        cfg = {k: v for k, v in cfg.items() if k in keep}

    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for arch, (path, voxel) in cfg.items():
        p = Path(path)
        if not p.exists():
            print(f"[skip] {arch}: path not found -> {path}")
            rows.append((arch, "MISSING", "", "", ""))
            continue
        print(f"\n=== {arch}  ({path}) ===")
        try:
            src = mp.resolve_volume(path)
            print(f"   using volume: {Path(src).name}")
            vol = mp.load_stack(src)
            labels, amb, vflat, info = sg.bootstrap_labels(
                vol, mode=args.mode, decup_sigma=args.decup_sigma, decup_mode=args.decup_mode,
                level=args.level, min_size=args.min_size, fill=not args.no_fill,
                local_window=args.local_window, sauvola_k=args.sauvola_k)
            od = out_root / arch; od.mkdir(parents=True, exist_ok=True)
            sg._save_volume(labels.astype(bool), od / f"{arch}_labels.tif")
            sg._save_volume(amb, od / f"{arch}_ambiguity.tif")
            # keep the de-cupped image too (useful for inspection)
            import tifffile
            tifffile.imwrite(str(od / f"{arch}_decupped.tif"),
                             (mp.normalize(vflat) * 255).astype(np.uint8), compression="zlib")
            sg._qc(vol, vflat, labels.astype(bool), amb, od, arch)
            if _HAVE_PROV:
                prov.stamp(od, params={"arch": arch, "voxel_um": voxel,
                                       "decup_mode": args.decup_mode,
                                       "mode": args.mode}, inputs=[path])
            rows.append((arch, f"{labels.mean():.3f}", f"{(~labels.astype(bool)).mean():.3f}",
                         f"{info['ambiguous_fraction']:.3f}", f"{info['threshold']:.3f}"))
            print(f"   labels solid {labels.mean():.3f} | ambiguous {info['ambiguous_fraction']:.3f}"
                  f" | thr {info['threshold']:.3f} -> {od}")
        except Exception as e:
            print(f"   [error] {arch}: {e}")
            rows.append((arch, "ERROR", "", "", str(e)[:40]))

    print("\n================ SUMMARY ================")
    print(f"{'arch':10s} {'solid':>7s} {'porosity':>9s} {'ambig':>7s} {'thr':>7s}")
    for a, s, po, am, th in rows:
        print(f"{a:10s} {s:>7s} {po:>9s} {am:>7s} {th:>7s}")
    print(f"\nOutputs under {out_root.resolve()} | refine *_ambiguity.tif then run train_seg.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
