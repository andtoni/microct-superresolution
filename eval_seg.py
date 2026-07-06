#!/usr/bin/env python3
"""
eval_seg.py -- segmentation validation harness for the scaffx toolbox (SI S6).

Scores segmentation quality and the comparisons that constitute the paper's segmentation
story, against the bootstrapped labels, per architecture:

  metrics  : Dice, IoU, ASSD (um), HD95 (um), Betti-number error (b0,b1), clDice
  methods  : --model (MONAI), --classical (de-cup + rim-fill), --pred-root (nnU-Net / any)
  compare  : classical vs ML vs nnU-Net on identical volumes  (does learning help?)
  HR<->SR  : segment HR and SR with the SAME model and compare  (thesis-critical: the
             segmenter must behave identically across resolutions)
  --holdout: restrict to the held-out architecture -> leave-one-architecture-out report

Outputs: seg_eval.xlsx (per_method + consistency + meta), Dice/IoU/Betti bar figures,
and pred-vs-GT QC overlays.

Usage (uv):
  uv run python eval_seg.py --raw-root "<aligned>" --label-root seg_labels --out seg_eval \
      --model seg_model --classical
  uv run python eval_seg.py ... --pred-root nnunet_predictions      # score nnU-Net too
  uv run python eval_seg.py ... --sr-root "<sr volumes>" --model seg_model   # HR<->SR
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

import morphometry as mp
import segment as sg
try:
    import scaffold_viz as viz
    import provenance as prov
    _HAVE_FACTORY = True
except Exception:
    _HAVE_FACTORY = False


# ------------------------------------------------------------------- metrics
def betti_of(lab_bool):
    import gudhi
    cc = gudhi.CubicalComplex(top_dimensional_cells=np.where(lab_bool, 0.0, 1.0))
    cc.compute_persistence()
    b = list(cc.persistent_betti_numbers(0.5, 0.5)) + [0, 0, 0]
    return (int(b[0]), int(b[1]), int(b[2]))


def cldice_metric(pred, gt):
    from skimage.morphology import skeletonize
    sp = skeletonize(pred); sg_ = skeletonize(gt)
    tprec = (sp & gt).sum() / (sp.sum() + 1e-9)
    tsens = (sg_ & pred).sum() / (sg_.sum() + 1e-9)
    return float(2 * tprec * tsens / (tprec + tsens + 1e-9))


def surface_distances(pred, gt, voxel_um):
    pe = pred ^ ndi.binary_erosion(pred)
    ge = gt ^ ndi.binary_erosion(gt)
    if pe.sum() == 0 or ge.sum() == 0:
        return float("nan"), float("nan")
    dt_to_g = ndi.distance_transform_edt(~ge) * voxel_um
    dt_to_p = ndi.distance_transform_edt(~pe) * voxel_um
    d_pg = dt_to_g[pe]; d_gp = dt_to_p[ge]
    assd = float((d_pg.sum() + d_gp.sum()) / (pe.sum() + ge.sum()))
    hd95 = float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95)))
    return assd, hd95


def metrics_pair(pred, gt, voxel_um):
    pred = pred.astype(bool); gt = gt.astype(bool)
    inter = (pred & gt).sum(); union = (pred | gt).sum()
    dice = float(2 * inter / (pred.sum() + gt.sum() + 1e-9))
    iou = float(inter / (union + 1e-9))
    assd, hd95 = surface_distances(pred, gt, voxel_um)
    bp, bg = betti_of(pred), betti_of(gt)
    return {"dice": dice, "iou": iou, "assd_um": assd, "hd95_um": hd95,
            "betti0_err": abs(bp[0] - bg[0]), "betti1_err": abs(bp[1] - bg[1]),
            "cldice": cldice_metric(pred, gt)}


# ------------------------------------------------------------------- IO
def _find(raw_root, label_root, holdout=None):
    raw_root, label_root = Path(raw_root), Path(label_root)
    out = []
    for ld in sorted(p for p in label_root.iterdir() if p.is_dir()):
        arch = ld.name
        if holdout and arch != holdout:
            continue
        lab = ld / f"{arch}_labels.tif"
        raw = None
        for sub in ("HR", "hr", "high", ""):
            cand = raw_root / arch / sub if sub else raw_root / arch
            if cand.exists():
                raw = cand; break
        if lab.exists() and raw is not None:
            try:
                rawv = mp.resolve_volume(str(raw))
            except Exception as e:
                print(f"[find] {arch}: {e}"); continue
            out.append({"arch": arch, "raw": rawv, "label": str(lab)})
    return out


def _central_crop(a, n):
    """Central NxNxN crop (n<=0 -> unchanged). Identical slices apply to raw/gt/SR (same shape)."""
    if not n or n <= 0:
        return a
    sl = []
    for s in a.shape:
        st = max(0, (s - n) // 2)
        sl.append(slice(st, st + n) if s > n else slice(0, s))
    return a[tuple(sl)]


def _load_mask(path):
    p = Path(path)
    if p.suffix in (".tif", ".tiff"):
        import tifffile
        return tifffile.imread(str(p)) > 0
    import SimpleITK as sitk
    return sitk.GetArrayFromImage(sitk.ReadImage(str(p))) > 0


def _pred_for(arch, pred_root):
    for ext in (".nii.gz", ".tif", ".tiff"):
        c = Path(pred_root) / f"{arch}{ext}"
        if c.exists():
            return _load_mask(c)
    return None


# ------------------------------------------------------------------- driver
def analyze_eval(raw_root, label_root, out_dir, voxel_um=0.9696, model=None,
                 classical=False, classical_modes=None, pred_root=None, sr_root=None,
                 holdout=None, roi=0):
    import tifffile
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    items = _find(raw_root, label_root, holdout=holdout)
    if not items:
        print("no (raw,label) pairs found."); return
    # build the classical ladder to score (named cls_<mode>); --classical = legacy otsu_decup
    modes = list(classical_modes) if classical_modes else []
    if classical and "otsu_decup" not in modes:
        modes.append("otsu_decup")
    rows, crows = [], []
    for it in items:
        arch = it["arch"]
        gt = tifffile.imread(it["label"]) > 0
        raw = mp.normalize(mp.load_stack(it["raw"])).astype(np.float32)
        if roi and roi > 0:
            gt = _central_crop(gt, roi); raw = _central_crop(raw, roi)
            print(f"[{arch}] ROI central crop -> {raw.shape}")
        methods = {}
        if model:
            seg, note = sg.segment_ml(raw, model)
            if seg is not None:
                methods["ml"] = seg
            else:
                print(f"[{arch}] ml: {note}")
        for m in modes:
            methods[f"cls_{m}"] = sg.segment_classical(raw, mode=m)[1]
        if pred_root:
            m = _pred_for(arch, pred_root)
            if m is not None:
                if roi and roi > 0:
                    m = _central_crop(m, roi)
                methods["nnunet"] = m
        for name, pred in methods.items():
            if pred.shape != gt.shape:
                print(f"[{arch}] {name}: shape {pred.shape} != gt {gt.shape}, skipping"); continue
            r = metrics_pair(pred, gt, voxel_um)
            r.update({"arch": arch, "method": name})
            rows.append(r)
            print(f"[{arch}] {name:9s} Dice {r['dice']:.3f} IoU {r['iou']:.3f} "
                  f"ASSD {r['assd_um']:.2f}um b0err {r['betti0_err']} b1err {r['betti1_err']} clDice {r['cldice']:.3f}")
        # HR<->SR consistency (same model on HR and SR)
        if sr_root and model and "ml" in methods:
            srp = None
            for sub in ("", "HR", "hr"):
                c = Path(sr_root) / arch / sub if sub else Path(sr_root) / arch
                if c.exists():
                    srp = c; break
            if srp is not None:
                srvol = mp.normalize(mp.load_stack(str(srp))).astype(np.float32)
                if roi and roi > 0:
                    srvol = _central_crop(srvol, roi)
                sm, _ = sg.segment_ml(srvol, model)
                if sm is not None and sm.shape == methods["ml"].shape:
                    hr = methods["ml"]
                    inter = (hr & sm).sum()
                    crows.append({"arch": arch,
                                  "hr_sr_dice": float(2 * inter / (hr.sum() + sm.sum() + 1e-9)),
                                  "hr_sr_iou": float(inter / ((hr | sm).sum() + 1e-9)),
                                  "betti0_hr": betti_of(hr)[0], "betti0_sr": betti_of(sm)[0],
                                  "betti1_hr": betti_of(hr)[1], "betti1_sr": betti_of(sm)[1]})
                    print(f"[{arch}] HR<->SR Dice {crows[-1]['hr_sr_dice']:.3f}")
        _qc(raw, gt, methods, out_dir, arch)

    _write(rows, crows, out_dir, voxel_um, model, holdout)


def _qc(raw, gt, methods, out_dir, arch):
    if not _HAVE_FACTORY or not methods:
        return
    try:
        import matplotlib.pyplot as plt
        viz.set_pub_style()
        z = raw.shape[0] // 2
        cols = 1 + len(methods)
        fig, ax = plt.subplots(1, cols, figsize=(2.2 * cols, 2.4))
        ax = np.atleast_1d(ax)
        base = np.dstack([mp.normalize(raw[z])] * 3)
        ge = gt[z] ^ ndi.binary_erosion(gt[z])
        g = base.copy(); g[ge] = [0, 1, 0]
        ax[0].imshow(g); ax[0].set_title(f"{arch} GT", fontsize=7); ax[0].axis("off")
        for i, (name, pred) in enumerate(methods.items(), 1):
            ov = base.copy()
            pe = pred[z] ^ ndi.binary_erosion(pred[z])
            ov[pe] = [1, 0.3, 0]
            ax[i].imshow(ov); ax[i].set_title(name, fontsize=7); ax[i].axis("off")
        viz.save_figure(fig, f"{arch}_eval_qc", out_dir)
        plt.close(fig)
    except Exception as e:
        print(f"   [qc] {arch} skipped: {e}")


def _write(rows, crows, out_dir, voxel_um, model, holdout):
    import pandas as pd
    df = pd.DataFrame(rows)
    sheets = {"per_method": df}
    if crows:
        sheets["hr_sr_consistency"] = pd.DataFrame(crows)
    if _HAVE_FACTORY and not df.empty:
        viz.write_workbook(sheets, out_dir / "seg_eval.xlsx",
                           meta={"voxel_um": voxel_um, "model": str(model), "holdout": str(holdout)})
        try:
            import matplotlib.pyplot as plt
            viz.set_pub_style()
            for metric in ["dice", "iou", "betti1_err"]:
                piv = df.pivot_table(index="arch", columns="method", values=metric)
                fig, axx = plt.subplots(figsize=(3.4, 2.6))
                piv.plot(kind="bar", ax=axx, color=viz.PALETTE[:piv.shape[1]], legend=True)
                axx.set_ylabel(metric); axx.set_xlabel(""); axx.legend(frameon=False, fontsize=6)
                viz.save_figure(fig, f"seg_eval_{metric}", out_dir)
                plt.close(fig)
        except Exception as e:
            print(f"   [fig] skipped: {e}")
        prov.stamp(out_dir, params={"voxel_um": voxel_um, "model": str(model), "holdout": str(holdout)}, inputs=[])
    elif not df.empty:
        df.to_csv(out_dir / "seg_eval.csv", index=False)
    if not df.empty:
        print("\n== mean by method ==")
        print(df.groupby("method")[["dice", "iou", "assd_um", "betti1_err", "cldice"]].mean().round(3))


def main(argv=None):
    ap = argparse.ArgumentParser(description="segmentation validation harness (S6)")
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--label-root", required=True)
    ap.add_argument("--out", default="seg_eval")
    ap.add_argument("--voxel", type=float, default=0.9696)
    ap.add_argument("--model", default=None, help="MONAI seg_model dir/best.pt")
    ap.add_argument("--classical", action="store_true", help="score legacy otsu_decup baseline")
    ap.add_argument("--classical-modes", default=None,
                    help="comma list of classical ladder modes to score, e.g. "
                         "otsu,otsu_decup,local,ridge (ridge is slow on full volumes)")
    ap.add_argument("--pred-root", default=None, help="external predictions (e.g. nnU-Net) named <arch>.nii.gz/.tif")
    ap.add_argument("--sr-root", default=None, help="SR volumes per arch for HR<->SR consistency")
    ap.add_argument("--holdout", default=None, help="restrict to held-out arch (LOAO report)")
    ap.add_argument("--roi", type=int, default=0, help="central NxNxN crop for fast eval (0 = full volume)")
    args = ap.parse_args(argv)
    cmodes = [s.strip() for s in args.classical_modes.split(",")] if args.classical_modes else None
    analyze_eval(args.raw_root, args.label_root, args.out, voxel_um=args.voxel,
                 model=args.model, classical=args.classical, classical_modes=cmodes,
                 pred_root=args.pred_root, sr_root=args.sr_root, holdout=args.holdout, roi=args.roi)
    return 0


if __name__ == "__main__":
    sys.exit(main())
