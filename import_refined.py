#!/usr/bin/env python3
r"""
import_refined.py -- bring Ilastik / TomoSAM / 3D-Slicer refined masks into the scaffx
label convention (uint8: 1 = solid scaffold, 0 = void) and place them under
...\codeoutput\segmentations\<arch>\.

Two roles (keep them separate -- see master plan D9):
  * training label  (full volume)        -> <arch>_labels.tif   (default; feeds train_seg.py)
  * evaluation GT   (verified sub-volume) -> <arch>_gt.tif        (--as-gt; feeds eval_seg.py)

Handles inputs:
  * Ilastik "Simple Segmentation" TIFF/HDF5  (class indices, usually 1 and 2)
  * binary masks 0/1 or 0/255               (TomoSAM / Slicer / manual)
  * .tif/.tiff, .h5/.hdf5, .nrrd, .nii/.nii.gz

Foreground (solid) selection:
  default: auto -- {0,1}|{0,255} -> >0 ; {1,2} (2-class Ilastik) -> ==1 (FIRST label = solid)
  override: --solid-value N   (e.g. --solid-value 2 if you painted void as Label 1)
            --invert          (flip after selection)

Usage (uv):
  uv run python import_refined.py --in "<arch>_ilastik.tif" --arch pel16 --root "<aligned>"
  uv run python import_refined.py --in pores.nrrd --arch vcr276 --solid-value 2
  uv run python import_refined.py --in gt_crop.tif --arch laser50 --as-gt
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np


def _load_any(path):
    p = Path(path); s = p.suffix.lower()
    if s in (".tif", ".tiff"):
        import tifffile
        return tifffile.imread(str(p))
    if s in (".h5", ".hdf5"):
        import h5py
        with h5py.File(str(p), "r") as f:
            keys = list(f.keys())
            # ilastik writes "exported_data"; our converter writes "data"
            key = next((k for k in ("exported_data", "data") if k in keys), keys[0])
            a = np.array(f[key])
        return np.squeeze(a)
    if s in (".nrrd", ".nii", ".gz", ".mha", ".mhd"):
        import SimpleITK as sitk
        return sitk.GetArrayFromImage(sitk.ReadImage(str(p)))
    raise ValueError(f"unsupported extension: {s}")


def _to_solid(arr, solid_value=None, invert=False):
    arr = np.squeeze(np.asarray(arr))
    uniq = np.unique(arr)
    if solid_value is not None:
        solid = arr == solid_value
    elif set(uniq.tolist()) <= {0, 1}:
        solid = arr > 0
    elif set(uniq.tolist()) <= {0, 255}:
        solid = arr > 0
    elif set(uniq.tolist()) <= {1, 2}:                  # 2-class Ilastik Simple Segmentation
        solid = arr == 1                                # convention: FIRST label = solid
        print("   [note] 2-class input -> treating value 1 as SOLID. "
              "If you painted void as Label 1, re-run with --solid-value 2.")
    else:
        raise ValueError(f"ambiguous label values {uniq.tolist()} -- pass --solid-value N")
    if invert:
        solid = ~solid
    return solid.astype(np.uint8)


def main(argv=None):
    ap = argparse.ArgumentParser(description="import refined Ilastik/TomoSAM masks into scaffx labels")
    ap.add_argument("--in", dest="inp", required=True, help="refined mask file")
    ap.add_argument("--arch", required=True, help="architecture name (folder under segmentations)")
    ap.add_argument("--seg-root",
                    default="/data/superresolution/Data/codeoutput/segmentations",
                    help="segmentations output root")
    ap.add_argument("--root", default=None,
                    help="aligned dir; if given, the saved label's shape is checked vs <arch>/hr.tif")
    ap.add_argument("--as-gt", action="store_true",
                    help="save as <arch>_gt.tif (evaluation ground truth) instead of <arch>_labels.tif")
    ap.add_argument("--solid-value", type=int, default=None, help="value that means SOLID")
    ap.add_argument("--invert", action="store_true", help="flip foreground/background after selection")
    ap.add_argument("--roi", default=None,
                    help="analysis ROI tif (from make_roi.py); labels outside it are set to void")
    args = ap.parse_args(argv)

    arr = _load_any(args.inp)
    solid = _to_solid(arr, args.solid_value, args.invert)
    if args.roi:
        roi = _load_any(args.roi) > 0
        if roi.shape == solid.shape:
            before = float(solid.mean())
            solid = (solid.astype(bool) & roi).astype(np.uint8)
            print(f"   ROI applied: solid {before:.3f} -> {solid.mean():.3f} (edge excluded)")
        else:
            print(f"   [WARN] ROI shape {roi.shape} != label {solid.shape}; ROI not applied")
    print(f"   loaded {Path(args.inp).name} shape {solid.shape} | solid frac {solid.mean():.3f}")

    if args.root:
        try:
            import tifffile
            hr = Path(args.root) / args.arch / "hr.tif"
            if hr.exists():
                shp = tifffile.TiffFile(str(hr)).series[0].shape
                if tuple(shp) != tuple(solid.shape):
                    print(f"   [WARN] shape {solid.shape} != hr.tif {tuple(shp)} -- "
                          "is this a sub-volume (GT crop), or a transposed export?")
                else:
                    print("   shape matches hr.tif (full-volume label).")
        except Exception as e:
            print(f"   [warn] shape check skipped: {e}")

    out_dir = Path(args.seg_root) / args.arch
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{args.arch}_gt.tif" if args.as_gt else f"{args.arch}_labels.tif"
    import tifffile
    tifffile.imwrite(str(out_dir / name), solid, compression="zlib")
    role = "evaluation GT" if args.as_gt else "training label"
    print(f"   -> {out_dir / name}  ({role})")
    if not args.as_gt:
        print("   train_seg.py / eval_seg.py read <arch>_labels.tif, so this is picked up automatically.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
