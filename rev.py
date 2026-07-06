#!/usr/bin/env python3
"""
rev.py -- Representative Elementary Volume (REV) gate for the scaffx toolbox.

Determines the smallest cube edge length at which chosen metrics converge, measured
through the IDENTICAL morphometry pipeline (so REV and downstream morph are
consistent).  Two criteria:

  dREV (deterministic, default) -- growing CENTRED cube; REV = the edge length beyond
      which the metric stops changing (running relative change <= tolerance).
  sREV (statistical, --srev)    -- N random windows per size; REV = the edge length at
      which the across-window CV drops below tolerance (Kanit 2003).

Design (locked):
  * default metrics = porosity + permeability (the demanding pair); connectivity, ssa,
    tortuosity, corr_length also selectable (--metrics).
  * tolerances: porosity 5% (det) / 2.5% (CV); permeability 10%; others 5%.
  * OPERATIVE REV is measured on SR (the downstream modality, large FOV); pass
    --validate-against <HR_stack> to overlay the HR curve, confirm the two agree on the
    overlap (<= HR FOV), and report the global gate cube.  If a metric never plateaus
    within the volume it is reported honestly as "not reached".

Outputs (standard contract): <source>_rev.xlsx (rev_curve + rev_summary + meta), a
metric-vs-size figure per metric (pdf/svg/png) with the REV marked, and manifest.json.

CLI:
  uv run python rev.py --vol <SR_stack> --voxel 0.9696 --source SR --out <dir> \
      [--metrics porosity,permeability] [--srev --n-windows 10] \
      [--validate-against <HR_stack> --validate-voxel 0.9696] \
      [--n-sizes 8 --min-frac 0.25] [--seg-method otsu|fixed --seg-level 0.5]
"""
from __future__ import annotations
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

import morphometry as mp                      # identical-pipeline metric functions
try:
    import scaffold_viz as viz
    import provenance as prov
    _HAVE_FACTORY = True
except Exception:
    _HAVE_FACTORY = False

M2_TO_MD = 1.0 / 9.869233e-16

# deterministic (dREV running-change) and CV (sREV) tolerances per metric
TOL_DET = {"porosity": 0.05, "permeability": 0.10, "connectivity": 0.05,
           "ssa": 0.05, "tortuosity": 0.05, "corr_length": 0.05}
TOL_CV = {"porosity": 0.025, "permeability": 0.10, "connectivity": 0.05,
          "ssa": 0.05, "tortuosity": 0.05, "corr_length": 0.05}


# ------------------------------------------------------ per-cube metric evaluation
def _perm_mD(void, env, voxel_m):
    pn = mp.extract_network(void & env, voxel_m)
    if pn is None:
        return np.nan
    K = mp.permeability(pn, void.shape, voxel_m, axes=(0, 1, 2))
    vals = [K[a] for a in (0, 1, 2) if a in K and np.isfinite(K[a])]
    return float(np.mean(vals)) * M2_TO_MD if vals else np.nan


def _tort_mean(void, env, voxel_m):
    T = mp.tortuosity(void & env, axes=(0, 1, 2))
    vals = [T[a] for a in (0, 1, 2) if a in T and np.isfinite(T[a])]
    return float(np.mean(vals)) if vals else np.nan


def _metric(name, void, env, voxel_m, voxel_um):
    if name == "porosity":
        return mp.porosity_metrics(void, env)[0]
    if name == "permeability":
        return _perm_mD(void, env, voxel_m)
    if name == "connectivity":                          # intensive: Euler density
        return mp.topology_metrics(void, env, voxel_m)["euler_density_per_mm3"]
    if name == "ssa":
        return mp.surface_metrics(void, env, voxel_m)["ssa_per_um"]
    if name == "tortuosity":
        return _tort_mean(void, env, voxel_m)
    if name == "corr_length":
        return mp.correlation_metrics(void, env, voxel_m)[0]["corr_length_um"]
    raise ValueError(f"unknown REV metric {name!r}")


# ------------------------------------------------------------------ curve building
def _sizes(shape, n_sizes, min_frac):
    m = min(shape)
    lo = max(16, int(min_frac * m))
    return [int(round(s)) for s in np.linspace(lo, m, n_sizes)]


def compute_curve(vol, voxel_um, metrics, sizes, mode="drev", n_windows=10,
                  seg_method="otsu", seg_level=None, rng=0):
    """Return {metric: list of (size_um, size_vox, value, cv)} over growing cubes."""
    voxel_m = float(voxel_um) * 1e-6
    Z, Y, X = vol.shape
    cz, cy, cx = Z // 2, Y // 2, X // 2
    rs = np.random.default_rng(rng)
    curve = {m: [] for m in metrics}
    for L in sizes:
        if mode == "drev":
            h = L // 2
            cube = vol[cz - h:cz - h + L, cy - h:cy - h + L, cx - h:cx - h + L]
            void, _ = mp.segment(cube, seg_method, seg_level)
            env = np.ones(cube.shape, bool)
            for m in metrics:
                try:
                    v = float(_metric(m, void, env, voxel_m, voxel_um))
                except Exception:
                    v = np.nan
                curve[m].append((L * voxel_um, L, v, np.nan))
        else:                                            # sREV: random windows
            bucket = {m: [] for m in metrics}
            for _ in range(n_windows):
                z0 = int(rs.integers(0, max(Z - L, 1)))
                y0 = int(rs.integers(0, max(Y - L, 1)))
                x0 = int(rs.integers(0, max(X - L, 1)))
                cube = vol[z0:z0 + L, y0:y0 + L, x0:x0 + L]
                void, _ = mp.segment(cube, seg_method, seg_level)
                env = np.ones(cube.shape, bool)
                for m in metrics:
                    try:
                        bucket[m].append(float(_metric(m, void, env, voxel_m, voxel_um)))
                    except Exception:
                        bucket[m].append(np.nan)
            for m in metrics:
                a = np.array(bucket[m], float)
                a = a[np.isfinite(a)]
                mean = float(np.mean(a)) if a.size else np.nan
                cv = float(np.std(a) / np.mean(a)) if a.size and np.mean(a) != 0 else np.nan
                curve[m].append((L * voxel_um, L, mean, cv))
        print(f"   size {L:4d} vox ({L*voxel_um:7.1f} um): "
              + ", ".join(f"{m}={curve[m][-1][2]:.4g}"
                          + (f"(cv {curve[m][-1][3]:.3f})" if mode == 'srev' else "")
                          for m in metrics))
    return curve


def detect_rev(curve, mode):
    """Per-metric REV + convergence flag from a curve."""
    out = {}
    for m, pts in curve.items():
        sizes_um = np.array([p[0] for p in pts], float)
        vals = np.array([p[2] for p in pts], float)
        cvs = np.array([p[3] for p in pts], float)
        rev_um, converged = np.nan, False
        if mode == "drev":
            tol = TOL_DET.get(m, 0.05)
            ref = np.nanmean(vals[-2:]) if np.isfinite(vals[-2:]).any() else np.nan
            if np.isfinite(ref) and ref != 0 and len(vals) > 1:
                deltas = np.abs(np.diff(vals)) / abs(ref)        # change between sizes
                if np.isfinite(deltas[-1]) and deltas[-1] <= tol:
                    converged = True
                    for k in range(len(deltas)):
                        seg = deltas[k:]
                        seg = seg[np.isfinite(seg)]
                        if seg.size and np.all(seg <= tol):
                            rev_um = float(sizes_um[k])
                            break
        else:                                                    # sREV
            tol = TOL_CV.get(m, 0.05)
            ok = np.isfinite(cvs) & (cvs <= tol)
            if ok.any():
                converged = True
                rev_um = float(sizes_um[np.argmax(ok)])
        out[m] = {"rev_um": rev_um, "converged": bool(converged),
                  "plateau_value": float(np.nanmean(vals[-2:])) if vals.size else np.nan}
    return out


# --------------------------------------------------------------------- DRIVER
def analyze_rev(vol, voxel_um, source, out_dir, metrics=("porosity", "permeability"),
                mode="drev", n_windows=10, n_sizes=8, min_frac=0.25,
                seg_method="otsu", seg_level=None, validate_vol=None,
                validate_voxel=None, write=True):
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = list(metrics)

    sizes = _sizes(vol.shape, n_sizes, min_frac)
    print(f"[{source}] {mode.upper()} on cube sizes (vox): {sizes}")
    curve = compute_curve(vol, voxel_um, metrics, sizes, mode=mode,
                          n_windows=n_windows, seg_method=seg_method, seg_level=seg_level)
    rev = detect_rev(curve, mode)

    # optional HR overlay -> validate SR REV curve on the overlap, report agreement
    vcurve, agree = None, {}
    if validate_vol is not None:
        vvox = validate_voxel if validate_voxel is not None else voxel_um
        vsizes = _sizes(validate_vol.shape, n_sizes, min_frac)
        print(f"[validate] curve on {validate_vol.shape} (sizes vox): {vsizes}")
        vcurve = compute_curve(validate_vol, vvox, metrics, vsizes, mode=mode,
                               n_windows=n_windows, seg_method=seg_method,
                               seg_level=seg_level)
        fov_overlap = min(max(p[0] for p in curve[metrics[0]]),
                          max(p[0] for p in vcurve[metrics[0]]))
        for m in metrics:
            s1 = np.array([p[0] for p in curve[m]]); v1 = np.array([p[2] for p in curve[m]])
            s2 = np.array([p[0] for p in vcurve[m]]); v2 = np.array([p[2] for p in vcurve[m]])
            mask = s1 <= fov_overlap
            if mask.sum() >= 2 and np.isfinite(v2).any():
                v2i = np.interp(s1[mask], s2, v2)
                ref = np.nanmean(v1[mask]) or 1.0
                agree[m] = float(np.nanmax(np.abs(v1[mask] - v2i) / abs(ref)))
            else:
                agree[m] = np.nan

    if write:
        _write(curve, vcurve, rev, agree, metrics, mode, source, out_dir,
               voxel_um, seg_method, seg_level)

    print(f"\n[{source}] REV summary ({mode}):")
    for m in metrics:
        r = rev[m]
        tag = (f"{r['rev_um']:.1f} um" if r["converged"] else "NOT REACHED within volume")
        extra = f" | HR-overlap max rel diff {agree[m]:.3f}" if agree.get(m) == agree.get(m) and m in agree else ""
        print(f"   {m:13s} REV = {tag}{extra}")
    operative = [rev[m]["rev_um"] for m in metrics if rev[m]["converged"]]
    if operative:
        print(f"   -> global gate cube (max over converged metrics) = "
              f"{max(operative):.1f} um ({max(operative)/voxel_um:.0f} vox on this grid)")
    print(f"   ({round(time.time()-t0,1)}s)")
    return {"curve": curve, "rev": rev, "agreement": agree}


def _write(curve, vcurve, rev, agree, metrics, mode, source, out_dir,
           voxel_um, seg_method, seg_level):
    import pandas as pd
    rows = []
    for m in metrics:
        for (s_um, s_vox, val, cv) in curve[m]:
            rows.append({"source": source, "metric": m, "size_um": s_um,
                         "size_vox": s_vox, "value": val, "cv": cv})
        if vcurve is not None:
            for (s_um, s_vox, val, cv) in vcurve[m]:
                rows.append({"source": "validate", "metric": m, "size_um": s_um,
                             "size_vox": s_vox, "value": val, "cv": cv})
    curve_df = pd.DataFrame(rows)
    summ = pd.DataFrame([{"metric": m, "mode": mode, "rev_um": rev[m]["rev_um"],
                          "rev_vox": (rev[m]["rev_um"] / voxel_um
                                      if rev[m]["rev_um"] == rev[m]["rev_um"] else np.nan),
                          "converged": rev[m]["converged"],
                          "plateau_value": rev[m]["plateau_value"],
                          "hr_overlap_max_rel_diff": agree.get(m, np.nan)}
                         for m in metrics])
    sheets = {"rev_curve": curve_df, "rev_summary": summ}
    stem = f"{source}_rev"
    if _HAVE_FACTORY:
        viz.write_workbook(sheets, out_dir / f"{stem}.xlsx",
                           meta={"source": source, "mode": mode, "voxel_um": voxel_um,
                                 "seg_method": seg_method, "seg_level": seg_level})
        # one figure per metric: value vs cube size, REV marked
        try:
            import matplotlib.pyplot as plt
            viz.set_pub_style()
            for m in metrics:
                fig, ax = plt.subplots(figsize=(3.3, 2.6))
                s = [p[0] for p in curve[m]]; v = [p[2] for p in curve[m]]
                ax.plot(s, v, "o-", color=viz.PALETTE[0], label=source)
                if vcurve is not None:
                    s2 = [p[0] for p in vcurve[m]]; v2 = [p[2] for p in vcurve[m]]
                    ax.plot(s2, v2, "s--", color=viz.PALETTE[1], label="validate (HR)")
                if rev[m]["converged"]:
                    ax.axvline(rev[m]["rev_um"], color=viz.PALETTE[2], ls=":",
                               label=f"REV {rev[m]['rev_um']:.0f} um")
                ax.set_xlabel("cube edge (um)"); ax.set_ylabel(m); ax.legend(frameon=False, fontsize=5)
                viz.save_figure(fig, f"{stem}_{m}", out_dir)
                plt.close(fig)
        except Exception as e:
            print(f"   [plot] skipped: {e}")
        prov.stamp(out_dir, params={"source": source, "mode": mode, "metrics": metrics,
                                    "voxel_um": voxel_um, "seg_method": seg_method},
                   inputs=[])
    else:
        curve_df.to_csv(out_dir / f"{stem}_curve.csv", index=False)
        summ.to_csv(out_dir / f"{stem}_summary.csv", index=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description="REV gate (dREV/sREV) for the scaffx toolbox")
    ap.add_argument("--vol", required=True, help="operative volume (SR): TIFF folder/stack")
    ap.add_argument("--voxel", type=float, required=True)
    ap.add_argument("--source", required=True, help="label e.g. SR / HR")
    ap.add_argument("--out", required=True)
    ap.add_argument("--metrics", default="porosity,permeability",
                    help="comma list from: porosity,permeability,connectivity,ssa,tortuosity,corr_length")
    ap.add_argument("--srev", action="store_true", help="statistical REV (random windows) instead of growing cube")
    ap.add_argument("--n-windows", type=int, default=10)
    ap.add_argument("--n-sizes", type=int, default=8)
    ap.add_argument("--min-frac", type=float, default=0.25, help="smallest cube as fraction of min dim")
    ap.add_argument("--seg-method", default="otsu", choices=["otsu", "fixed", "percentile"])
    ap.add_argument("--seg-level", type=float, default=None)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--validate-against", default=None, help="HR stack to overlay/validate the curve")
    ap.add_argument("--validate-voxel", type=float, default=None)
    args = ap.parse_args(argv)

    vol = mp.load_stack(args.vol)
    if args.normalize:
        vol = mp.normalize(vol)
    val = None
    if args.validate_against:
        val = mp.load_stack(args.validate_against)
        if args.normalize:
            val = mp.normalize(val)
    metrics = [s.strip() for s in args.metrics.split(",") if s.strip()]
    analyze_rev(vol, args.voxel, args.source, args.out, metrics=metrics,
                mode="srev" if args.srev else "drev", n_windows=args.n_windows,
                n_sizes=args.n_sizes, min_frac=args.min_frac,
                seg_method=args.seg_method, seg_level=args.seg_level,
                validate_vol=val, validate_voxel=args.validate_voxel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
