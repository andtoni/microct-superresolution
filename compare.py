#!/usr/bin/env python3
"""
compare.py -- Act II equivalence engine for the scaffx toolbox.

Establishes the thesis-critical claim that SR recovers HR-fidelity metrics (and quantifies
how interpolation and DeepScout compare), per metric, across samples:

  TOST  -- two one-sided tests for statistical EQUIVALENCE within a per-metric margin
           (the correct test for "the same", unlike a t-test which only fails to reject
           difference). Permeability is tested in log10 space (margin in decades).
  Bland-Altman -- bias (mean SR-HR difference) + 95% limits of agreement.
  CCC   -- Lin's concordance correlation coefficient (agreement, not just correlation).

Input: morph outputs (a folder of <sample>_<modality>_morphometry.xlsx) OR a tidy table
(CSV/xlsx with columns sample, modality, metric, value). Modalities compared against the
reference (default HR): SR, interp, deepscout.

Outputs (standard contract): compare.xlsx (equivalence + meta), per-metric Bland-Altman and
identity (HR-vs-test) figures, manifest.json.

Usage (uv):
  uv run python compare.py --data <morph_out_dir> --ref HR --test SR,interp,deepscout --out compare
  uv run python compare.py --data combined.csv --ref HR --test SR --out compare
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import scaffold_viz as viz
    import provenance as prov
    _HAVE_FACTORY = True
except Exception:
    _HAVE_FACTORY = False

# per-metric equivalence margins: (margin, mode) with mode in {abs, rel, log}
MARGINS = {
    "porosity_total": (0.025, "abs"),
    "porosity_accessible": (0.025, "abs"),
    "perm_mean_mD": (0.5, "log"),          # +/- half a decade
    "tortuosity_mean": (0.1, "abs"),
    "ssa_per_um": (0.10, "rel"),
    "euler_density_per_mm3": (0.15, "rel"),
    "pore_diam_mean_um": (0.10, "rel"),
    "throat_diam_mean_um": (0.10, "rel"),
    "coordination_mean": (0.10, "rel"),
    "corr_length_um": (0.10, "rel"),
}
DEFAULT_MARGIN = (0.10, "rel")


# ----------------------------------------------------------------- statistics
def _tost(d, margin, alpha=0.05):
    """Paired-difference TOST. Returns (p, equivalent, mean_diff, se)."""
    from scipy import stats
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 2:
        return np.nan, False, float(np.mean(d)) if n else np.nan, np.nan
    m = float(np.mean(d)); se = float(np.std(d, ddof=1) / np.sqrt(n))
    if se == 0:
        eq = abs(m) < margin
        return (0.0 if eq else 1.0), eq, m, 0.0
    t_lo = (m + margin) / se                       # H0: mean <= -margin
    t_hi = (m - margin) / se                       # H0: mean >= +margin
    p_lo = 1 - stats.t.cdf(t_lo, n - 1)
    p_hi = stats.t.cdf(t_hi, n - 1)
    p = float(max(p_lo, p_hi))
    return p, bool(p < alpha), m, se


def _ccc(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 2:
        return np.nan
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = np.mean((x - mx) * (y - my))
    denom = vx + vy + (mx - my) ** 2
    return float(2 * cov / denom) if denom > 0 else np.nan


def _bland_altman(ref, test):
    ref = np.asarray(ref, float); test = np.asarray(test, float)
    d = test - ref
    d = d[np.isfinite(d)]
    bias = float(np.mean(d)); sd = float(np.std(d, ddof=1)) if d.size > 1 else 0.0
    return bias, bias - 1.96 * sd, bias + 1.96 * sd


# ----------------------------------------------------------------- aggregation
def _aggregate(data_path):
    """Build a tidy DataFrame [sample, modality, metric, value] from morph outputs or a table."""
    import pandas as pd
    p = Path(data_path)
    if p.is_file() and p.suffix.lower() in (".csv", ".tsv"):
        df = pd.read_csv(p, sep="\t" if p.suffix.lower() == ".tsv" else ",")
        return df
    if p.is_file() and p.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(p)
    rows = []
    for f in sorted(p.glob("*_morphometry.xlsx")):
        stem = f.stem.replace("_morphometry", "")
        parts = stem.split("_")
        if len(parts) < 2:
            print(f"[skip] {f.name}: name must be <sample>_<modality>_morphometry.xlsx"); continue
        modality = parts[-1]; sample = "_".join(parts[:-1])
        try:
            m = pd.read_excel(f, sheet_name="metrics")
        except Exception as e:
            print(f"[skip] {f.name}: {e}"); continue
        # metrics sheet may be wide (1 row) or key/value
        if set(c.lower() for c in m.columns) >= {"metric", "value"}:
            for _, r in m.iterrows():
                rows.append({"sample": sample, "modality": modality,
                             "metric": str(r["metric"]), "value": r["value"]})
        else:
            r0 = m.iloc[0]
            for c in m.columns:
                rows.append({"sample": sample, "modality": modality,
                             "metric": str(c), "value": r0[c]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- comparison
def compare_modalities(df, ref="HR", tests=("SR",), metrics=None, margins=None):
    import pandas as pd
    margins = margins or MARGINS
    df = df.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    wide = df.pivot_table(index=["sample", "metric"], columns="modality",
                          values="value", aggfunc="first").reset_index()
    out = []
    all_metrics = metrics or sorted(df["metric"].unique())
    for test in tests:
        for metric in all_metrics:
            sub = wide[wide["metric"] == metric]
            if ref not in sub.columns or test not in sub.columns:
                continue
            a = sub[ref].to_numpy(float); b = sub[test].to_numpy(float)
            ok = np.isfinite(a) & np.isfinite(b)
            a, b = a[ok], b[ok]
            if a.size < 2:
                continue
            margin, mode = margins.get(metric, DEFAULT_MARGIN)
            if mode == "log":
                la, lb = np.log10(np.clip(a, 1e-12, None)), np.log10(np.clip(b, 1e-12, None))
                d = lb - la; eff_margin = margin
                bias, lo, hi = _bland_altman(la, lb); ccc = _ccc(la, lb)
            elif mode == "rel":
                eff_margin = margin * float(np.mean(np.abs(a)))
                d = b - a; bias, lo, hi = _bland_altman(a, b); ccc = _ccc(a, b)
            else:                                   # abs
                eff_margin = margin; d = b - a
                bias, lo, hi = _bland_altman(a, b); ccc = _ccc(a, b)
            p, equiv, md, se = _tost(d, eff_margin)
            out.append({"metric": metric, "comparison": f"{test}_vs_{ref}", "n": int(a.size),
                        "mode": mode, "margin": margin, "mean_diff": md, "tost_p": p,
                        "equivalent": equiv, "ba_bias": bias, "ba_loa_lo": lo, "ba_loa_hi": hi,
                        "ccc": ccc})
    return pd.DataFrame(out)


def analyze_compare(data_path, out_dir, ref="HR", tests=("SR",), metrics=None):
    import pandas as pd
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = _aggregate(data_path)
    if df.empty:
        print("no data aggregated."); return
    res = compare_modalities(df, ref=ref, tests=tests, metrics=metrics)
    if res.empty:
        print(f"no paired metrics for ref={ref}, tests={tests}."); return

    print(f"\n== equivalence vs {ref} ==")
    for _, r in res.iterrows():
        tag = "EQUIVALENT" if r["equivalent"] else "not shown"
        print(f"  {r['comparison']:20s} {r['metric']:24s} n={r['n']} "
              f"TOST p={r['tost_p']:.3f} [{tag}] CCC={r['ccc']:.3f} bias={r['ba_bias']:.4g}")

    _write(df, res, out_dir, ref, tests, str(data_path))
    return res


def _write(df, res, out_dir, ref, tests, data_path=""):
    import pandas as pd
    if _HAVE_FACTORY:
        viz.write_workbook({"equivalence": res}, out_dir / "compare.xlsx",
                           meta={"ref": ref, "tests": ",".join(tests)})
        try:
            import matplotlib.pyplot as plt
            viz.set_pub_style()
            df = df.copy(); df["value"] = pd.to_numeric(df["value"], errors="coerce")
            wide = df.pivot_table(index=["sample", "metric"], columns="modality",
                                  values="value", aggfunc="first").reset_index()
            for _, r in res.iterrows():
                metric = r["metric"]; test = r["comparison"].split("_vs_")[0]
                sub = wide[wide["metric"] == metric]
                if ref not in sub or test not in sub:
                    continue
                a = sub[ref].to_numpy(float); b = sub[test].to_numpy(float)
                ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
                if a.size < 2:
                    continue
                # Bland-Altman
                fig, ax = plt.subplots(figsize=(3.2, 2.6))
                mean = (a + b) / 2; diff = b - a
                ax.scatter(mean, diff, s=14, color=viz.PALETTE[2])
                ax.axhline(r["ba_bias"], color=viz.PALETTE[0], lw=1)
                ax.axhline(r["ba_loa_lo"], color=viz.PALETTE[1], ls="--", lw=1)
                ax.axhline(r["ba_loa_hi"], color=viz.PALETTE[1], ls="--", lw=1)
                ax.set_xlabel(f"mean({ref},{test})"); ax.set_ylabel(f"{test}-{ref}")
                ax.set_title(f"{metric}", fontsize=7)
                viz.save_figure(fig, f"ba_{test}_{metric}", out_dir); plt.close(fig)
                # identity scatter
                fig, ax = plt.subplots(figsize=(2.8, 2.8))
                ax.scatter(a, b, s=14, color=viz.PALETTE[3])
                lim = [min(a.min(), b.min()), max(a.max(), b.max())]
                ax.plot(lim, lim, color="k", lw=0.8, ls=":")
                ax.set_xlabel(ref); ax.set_ylabel(test); ax.set_title(f"{metric} (CCC {r['ccc']:.2f})", fontsize=7)
                viz.save_figure(fig, f"identity_{test}_{metric}", out_dir); plt.close(fig)
        except Exception as e:
            print(f"   [fig] skipped: {e}")
        prov.stamp(out_dir, params={"ref": ref, "tests": list(tests)}, inputs=[data_path] if data_path else [])
    else:
        res.to_csv(out_dir / "compare.csv", index=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Act II equivalence engine (TOST/Bland-Altman/CCC)")
    ap.add_argument("--data", required=True, help="morph output dir or tidy CSV/xlsx")
    ap.add_argument("--ref", default="HR")
    ap.add_argument("--test", default="SR", help="comma list: SR,interp,deepscout")
    ap.add_argument("--out", default="compare")
    ap.add_argument("--metrics", default=None, help="comma list to restrict metrics")
    args = ap.parse_args(argv)
    tests = tuple(s.strip() for s in args.test.split(",") if s.strip())
    metrics = [s.strip() for s in args.metrics.split(",")] if args.metrics else None
    analyze_compare(args.data, args.out, ref=args.ref, tests=tests, metrics=metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
