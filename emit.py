#!/usr/bin/env python3
"""emit.py -- the single sanctioned exit for any quantitative result.

emit_result(...) writes, in ONE call, the 4-part artifact contract (DATA_TAXONOMY SS5,7)
and appends to the master_metrics ledger:
  * tidy-long  <fig>_<panel>_data.csv + .xlsx          (R / Python / JASP canonical)
  * Prism wide export + <fig>_<panel>_prism.layout.json (Column or Grouped table)
  * stats sidecar <fig>_<panel>_stats.csv/.json         (effect size + seeded bootstrap CI)
  * figure <fig>_<panel>.pdf/.svg (+ 600 dpi .tiff/.png) if a matplotlib fig is passed
  * manifest.json
So tool-correct shapes are guaranteed everywhere and stats never separate from numbers.
"""
from __future__ import annotations
import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

import atomicio
import provenance as prov


def _data_root():
    env = os.environ.get("SCAFFX_DATA_ROOT")
    if env:
        return Path(env)
    try:
        import paths
        return Path(paths.OUTPUT_ROOT)
    except Exception:
        return Path("codeoutput")


def _slug(s):
    return "".join(c if c.isalnum() else "_" for c in str(s)).strip("_")


# ---------------------------------------------------------------- Prism shapes
def to_prism_column(df, group_col, value_col, case_col=None):
    """COLUMN table: one column per group; rows = per-case values (paired) or stacked."""
    if case_col and case_col in df.columns:
        return df.pivot_table(index=case_col, columns=group_col, values=value_col, aggfunc="first")
    cols = {g: sub[value_col].reset_index(drop=True) for g, sub in df.groupby(group_col)}
    return pd.DataFrame({g: pd.Series(v) for g, v in cols.items()})


def to_prism_grouped(df, row_col, group_col, value_col):
    """GROUPED table: rows = row_col; columns = group with replicate sub-columns
    (two-row banner header via a MultiIndex)."""
    blocks = []
    for g, sub in df.groupby(group_col):
        rep = sub.groupby(row_col).cumcount()
        w = sub.assign(_rep=rep).pivot_table(index=row_col, columns="_rep", values=value_col)
        w.columns = pd.MultiIndex.from_tuples([(g, f"rep{c + 1}") for c in w.columns])
        blocks.append(w)
    return pd.concat(blocks, axis=1)


# ---------------------------------------------------------------- statistics
def cliffs_delta(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return np.nan
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return float((gt - lt) / (a.size * b.size))


def bootstrap_ci(a, b, paired=True, n_boot=2000, seed=42, level=0.95):
    """Bootstrap CI of the median difference (paired or unpaired). Seed recorded
    by the caller so the CI is reproducible."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = float(np.median(a) - np.median(b))
    boots = []
    if paired and a.size == b.size and a.size > 0:
        d = a - b
        for _ in range(n_boot):
            boots.append(np.median(d[rng.integers(0, d.size, d.size)]))
    elif a.size and b.size:
        for _ in range(n_boot):
            boots.append(np.median(a[rng.integers(0, a.size, a.size)]) -
                         np.median(b[rng.integers(0, b.size, b.size)]))
    else:
        return obs, np.nan, np.nan
    lo, hi = np.percentile(boots, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return obs, float(lo), float(hi)


# ---------------------------------------------------------------- ledger
def append_ledger(rows):
    """Append rows (list of dicts) to master_metrics (parquet canonical + csv export)."""
    root = _data_root() / "data"
    root.mkdir(parents=True, exist_ok=True)
    pq = root / "master_metrics.parquet"
    new = pd.DataFrame(rows)
    if pq.exists():
        try:
            new = pd.concat([pd.read_parquet(pq), new], ignore_index=True)
        except Exception:
            pass
    tmp = pq.with_name(f".master_metrics.parquet.tmp.{uuid.uuid4().hex[:8]}")
    new.to_parquet(tmp, index=False)
    os.replace(tmp, pq)
    atomicio.write_text_atomic(root / "master_metrics.csv", new.to_csv(index=False))
    return str(pq)


# ---------------------------------------------------------------- the emitter
def emit_result(fig_id, panel, long_df, *, value_col="value", group_col="model",
                case_col="case_id", metric=None, modality=None, dataset=None, factor=None,
                experiment=None, fig=None, prism="column", stats_pair=None, paired=True,
                n_boot=2000, seed=42, unit="1", out_root=None, ledger=True, params=None):
    """Write the 4-part contract for one panel + append to the ledger. Returns paths."""
    long_df = long_df.copy()
    out = Path(out_root) if out_root else _data_root() / "figures" / f"{fig_id}_{_slug(panel)}"
    out.mkdir(parents=True, exist_ok=True)
    base = f"{fig_id}_{_slug(panel)}"
    written = {"dir": str(out)}

    # 1) tidy-long (canonical for R / Python / JASP)
    atomicio.write_text_atomic(out / f"{base}_data.csv", long_df.to_csv(index=False))
    try:
        long_df.to_excel(out / f"{base}_data.xlsx", index=False)
    except Exception as e:
        written["xlsx_error"] = str(e)

    # 2) Prism wide export + layout sidecar
    if prism == "grouped":
        wide = to_prism_grouped(long_df, metric or group_col, group_col, value_col)
    else:
        wide = to_prism_column(long_df, group_col, value_col, case_col)
    atomicio.write_text_atomic(out / f"{base}_prism.csv", wide.to_csv())
    atomicio.write_json_atomic(out / f"{base}_prism.layout.json", {
        "geometry": prism, "group_col": group_col, "value_col": value_col,
        "case_col": case_col if case_col in long_df.columns else None,
        "recommended_test": "Wilcoxon (paired)" if paired else "Mann-Whitney",
    })

    # 3) stats sidecar (effect size + seeded bootstrap CI; bare p never reported alone)
    if stats_pair:
        g1, g2 = stats_pair
        a = long_df.loc[long_df[group_col] == g1, value_col].to_numpy(float)
        b = long_df.loc[long_df[group_col] == g2, value_col].to_numpy(float)
        from scipy import stats as ss
        try:
            if paired and a.size == b.size and a.size > 0:
                stat, p = ss.wilcoxon(a, b)
                test = "wilcoxon_paired"
            else:
                stat, p = ss.mannwhitneyu(a, b, alternative="two-sided")
                test = "mannwhitney"
        except Exception:
            stat, p, test = np.nan, np.nan, "undefined"
        obs, lo, hi = bootstrap_ci(a, b, paired, n_boot, seed)
        from figure_style import stars
        srow = {"group1": g1, "group2": g2, "test": test, "n1": int(a.size), "n2": int(b.size),
                "statistic": float(stat), "p": float(p), "effect_median_diff": obs,
                "ci_lo": lo, "ci_hi": hi, "ci_method": "bootstrap_percentile",
                "n_boot": n_boot, "bootstrap_seed": seed, "cliffs_delta": cliffs_delta(a, b),
                "stars": stars(p) if p == p else "ns"}
        atomicio.write_text_atomic(out / f"{base}_stats.csv", pd.DataFrame([srow]).to_csv(index=False))
        atomicio.write_json_atomic(out / f"{base}_stats.json", srow)
        written["stats"] = srow

    # 4) figure (layered, Adobe-ready)
    if fig is not None:
        import figure_style as fs
        written["figure"] = fs.layered_savefig(fig, out / base)

    # manifest (schema-conforming, atomic)
    prov.write_manifest(out, params={"fig_id": fig_id, "panel": panel, "metric": metric,
                                     "value_col": value_col, "group_col": group_col,
                                     **(params or {})},
                        outputs=[str(p) for p in sorted(out.glob(f"{base}*")) if p.is_file()],
                        seeds={"seed": seed})

    # 5) ledger append (one row per measured value)
    if ledger and metric:
        rows = []
        for _, r in long_df.iterrows():
            rows.append({
                "project": "rapid-scaffold-sr", "experiment": experiment or fig_id,
                "model": r.get(group_col), "case_id": r.get(case_col),
                "dataset": r.get("dataset", dataset), "modality": r.get("modality", modality),
                "factor": r.get("factor", factor), "metric": metric,
                "value": r.get(value_col), "unit": unit, "seed": seed,
                "manifest_ref": str(out / "manifest.json"),
            })
        written["ledger"] = append_ledger(rows)
    return written
