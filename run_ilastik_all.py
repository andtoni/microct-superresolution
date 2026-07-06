#!/usr/bin/env python3
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

import import_refined as ir

DEFAULT_ILASTIK = "ilastik"
DEFAULT_SEG = "/data/superresolution/Data/codeoutput/segmentations"


def main(argv=None):
    ap = argparse.ArgumentParser(description="headless pooled-Ilastik predict + import")
    ap.add_argument("--project", required=True)
    ap.add_argument("--ilastik", default=DEFAULT_ILASTIK)
    ap.add_argument("--seg-root", default=DEFAULT_SEG)
    ap.add_argument("--root", default=None)
    ap.add_argument("--input-suffix", default="_decupped.tif")
    ap.add_argument("--input-internal", default="data")
    ap.add_argument("--export-source", default="Simple Segmentation",
                    choices=["Simple Segmentation", "Probabilities"])
    ap.add_argument("--input-axes", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--solid-value", type=int, default=None)
    ap.add_argument("--apply-roi", action="store_true")
    ap.add_argument("--no-import", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    exe = Path(args.ilastik)
    if not args.dry_run and not exe.exists():
        print(f"[error] ilastik not found: {exe}"); return 2
    proj = Path(args.project)
    if not proj.exists():
        print(f"[error] project not found: {proj}"); return 2

    seg_root = Path(args.seg_root)
    only = {s.strip() for s in args.only.split(",")} if args.only else None
    inputs, archs = [], []
    for d in sorted(p for p in seg_root.iterdir() if p.is_dir()):
        if only and d.name not in only:
            continue
        cand = d / f"{d.name}{args.input_suffix}"
        if cand.exists():
            inputs.append(cand); archs.append(d.name)
        else:
            print(f"[skip] {d.name}: no {cand.name}")
    if not inputs:
        print("[error] no inputs found."); return 2

    is_h5 = args.input_suffix.lower().endswith((".h5", ".hdf5"))
    ext = ".h5" if is_h5 else ".tif"
    if is_h5:
        out_fmt = "{dataset_dir}/{nickname}_seg.h5"
        cmd = [str(exe), "--headless", f"--project={proj}",
               f"--export_source={args.export_source}",
               "--output_format=hdf5", "--output_internal_path=exported_data",
               f"--output_filename_format={out_fmt}"]
    else:
        out_fmt = "{dataset_dir}/{nickname}_seg.tif"
        cmd = [str(exe), "--headless", f"--project={proj}",
               f"--export_source={args.export_source}",
               "--output_format=multipage tiff", f"--output_filename_format={out_fmt}"]
    axes = args.input_axes or ("zyx" if is_h5 else None)
    if axes:
        cmd.append(f"--input_axes={axes}")
    cmd += [f"{p}/{args.input_internal}" if is_h5 else str(p) for p in inputs]

    print(f"predicting {len(inputs)} volumes with {proj.name}:")
    for a in archs:
        print(f"   - {a}")
    print("\ncommand:\n  " + " ".join(f'"{c}"' if " " in c else c for c in cmd) + "\n")
    if args.dry_run:
        return 0

    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"[error] ilastik exited {r.returncode}"); return r.returncode
    if args.no_import:
        print("prediction done (import skipped)."); return 0

    print("\nimporting predictions -> <arch>_labels.tif:")
    for arch, inp in zip(archs, inputs):
        pred = inp.with_name(f"{inp.stem}_seg{ext}")
        if not pred.exists():
            cands = sorted(inp.parent.glob(f"{inp.stem}*_seg{ext}"), key=lambda p: p.stat().st_mtime)
            if cands:
                pred = cands[-1]
        if not pred.exists():
            print(f"   [warn] {arch}: no *_seg{ext} found for {inp.stem}"); continue
        iargs = ["--in", str(pred), "--arch", arch, "--seg-root", str(seg_root)]
        if args.root:
            iargs += ["--root", args.root]
        if args.solid_value is not None:
            iargs += ["--solid-value", str(args.solid_value)]
        if args.apply_roi:
            roi = seg_root / arch / f"{arch}_roi.tif"
            if roi.exists():
                iargs += ["--roi", str(roi)]
            else:
                print(f"   [warn] {arch}: --apply-roi set but {roi.name} missing")
        ir.main(iargs)
    print("\ndone. labels in segmentations/<arch>/<arch>_labels.tif")
    return 0


if __name__ == "__main__":
    sys.exit(main())
