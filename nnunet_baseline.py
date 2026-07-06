#!/usr/bin/env python3
"""
nnunet_baseline.py -- prepare an nnU-Net v2 Residual-Encoder (ResEnc) baseline.

The MICCAI-2024 "nnU-Net Revisited" benchmark found the SOTA recipe is a CNN U-Net
(ResNet/ConvNeXt) in the nnU-Net framework scaled to hardware -- so nnU-Net ResEnc is the
reviewer-standard strong baseline to put next to the artifact-aware MONAI model. This
script converts the bootstrapped (raw HR, label) pairs to the nnU-Net dataset format and
prints the exact commands (it does not train here -- nnU-Net trains in your uv env on the 5090).

Usage (uv):
  uv add nnunetv2
  uv run python nnunet_baseline.py --raw-root "<aligned dir>" --label-root seg_labels \
      --dataset-id 1 --nnunet-raw "<nnUNet_raw>"
  # leave-one-architecture-out (separate dataset per held-out route):
  uv run python nnunet_baseline.py ... --exclude pel16 --dataset-id 11
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _find_pairs(raw_root, label_root, exclude=None):
    raw_root, label_root = Path(raw_root), Path(label_root)
    pairs = []
    for ld in sorted(p for p in label_root.iterdir() if p.is_dir()):
        arch = ld.name
        if exclude and arch == exclude:
            continue
        lab = ld / f"{arch}_labels.tif"
        raw = None
        for sub in ("HR", "hr", "high", ""):
            cand = raw_root / arch / sub if sub else raw_root / arch
            if cand.exists():
                raw = cand
                break
        if lab.exists() and raw is not None:
            try:
                import morphometry as mp
                rawv = mp.resolve_volume(str(raw))   # pick the HR file, not the whole dir (HR+LR+masks differ in shape)
            except Exception as e:
                print(f"  [find] {arch}: {e}"); continue
            pairs.append({"arch": arch, "raw": rawv, "label": str(lab)})
    return pairs


def _load(path):
    import tifffile
    p = Path(path)
    if p.is_dir():
        import morphometry as mp
        return mp.load_stack(path)
    return tifffile.imread(str(p))


def _write_nii(arr, path):
    # SimpleITK only -> orientation stays (Z,Y,X) end-to-end: nnU-Net reads/writes with
    # SimpleITKIO and eval_seg reads predictions with SimpleITK; a nibabel+SimpleITK mix
    # transposes X<->Z. dtype preserved (raw float32, label uint8).
    import SimpleITK as sitk
    arr = np.ascontiguousarray(arr)
    sitk.WriteImage(sitk.GetImageFromArray(arr), str(path))


def main(argv=None):
    ap = argparse.ArgumentParser(description="prepare nnU-Net v2 ResEnc baseline dataset")
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--label-root", required=True, help="seg_labels dir from run_segment_all.py")
    ap.add_argument("--nnunet-raw", default="nnUNet_raw", help="nnU-Net raw data root")
    ap.add_argument("--dataset-id", type=int, default=1)
    ap.add_argument("--dataset-name", default="Scaffold")
    ap.add_argument("--exclude", default=None, help="architecture to leave out (LOAO)")
    ap.add_argument("--preset", default="L", choices=["M", "L", "XL"],
                    help="ResEnc preset; L/XL suit a 24-32 GB GPU like the 5090")
    args = ap.parse_args(argv)

    pairs = _find_pairs(args.raw_root, args.label_root, exclude=args.exclude)
    if not pairs:
        print("no pairs found -- run run_segment_all.py first."); return 1

    did = f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    root = Path(args.nnunet_raw) / did
    (root / "imagesTr").mkdir(parents=True, exist_ok=True)
    (root / "labelsTr").mkdir(parents=True, exist_ok=True)
    print(f"building {did} from {[p['arch'] for p in pairs]}"
          + (f" (excluding {args.exclude})" if args.exclude else ""))

    for p in pairs:
        raw = _load(p["raw"]).astype(np.float32)
        import tifffile
        lab = (tifffile.imread(p["label"]) > 0).astype(np.uint8)
        if lab.shape != raw.shape:
            print(f"  [skip] {p['arch']}: raw {raw.shape} != label {lab.shape}"); continue
        _write_nii(raw, root / "imagesTr" / f"{p['arch']}_0000.nii.gz")
        _write_nii(lab, root / "labelsTr" / f"{p['arch']}.nii.gz")
        print(f"  + {p['arch']} ({raw.shape})")

    ds = {"channel_names": {"0": "CT"},
          "labels": {"background": 0, "scaffold": 1},
          "numTraining": len(pairs), "file_ending": ".nii.gz"}
    (root / "dataset.json").write_text(json.dumps(ds, indent=2))
    print(f"wrote {root/'dataset.json'}")

    planner = {"M": "nnUNetPlannerResEncM", "L": "nnUNetPlannerResEncL", "XL": "nnUNetPlannerResEncXL"}[args.preset]
    plans = {"M": "nnUNetResEncUNetMPlans", "L": "nnUNetResEncUNetLPlans", "XL": "nnUNetResEncUNetXLPlans"}[args.preset]
    did_n = args.dataset_id
    print("\n================ nnU-Net v2 ResEnc commands ================")
    print("# set these once:")
    print('#   export nnUNet_raw="<nnUNet_raw>" nnUNet_preprocessed="<prep>" nnUNet_results="<results>"')
    print(f"uv run nnUNetv2_plan_and_preprocess -d {did_n} -pl {planner} --verify_dataset_integrity")
    print(f"uv run nnUNetv2_train {did_n} 3d_fullres all -p {plans}")
    print(f"#   (or 5-fold: replace 'all' with 0 1 2 3 4)")
    print(f"uv run nnUNetv2_predict -d {did_n} -i <input_dir> -o <output_dir> -c 3d_fullres -p {plans} -f all")
    if args.exclude:
        print(f"\n# LOAO: predict the held-out architecture '{args.exclude}' (put its HR nii in <input_dir>)")
    print("\nThen score nnU-Net vs the MONAI model with eval_seg.py (--pred-root on the predictions).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
