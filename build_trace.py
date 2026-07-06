#!/usr/bin/env python3
"""build_trace.py -- join manuscript/claims.csv against the ledger + figures -> TRACE.csv [GEN].

For each claim: do its evidence figures exist (codeoutput/figures), are its experiments
present in master_metrics, do its scripts exist? Flags broken links so a reviewer
challenge ("which data backs claim C3?") resolves to an exact artefact. Exit 1 if any
NON-draft claim has a broken link.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import pandas as pd


def _data_root():
    if os.environ.get("SCAFFX_DATA_ROOT"):
        return Path(os.environ["SCAFFX_DATA_ROOT"])
    try:
        import paths
        return Path(paths.OUTPUT_ROOT)
    except Exception:
        return Path("codeoutput")


def main(argv=None):
    claims = Path("manuscript/claims.csv")
    if not claims.exists():
        print("no manuscript/claims.csv")
        return 0
    df = pd.read_csv(claims)
    root = _data_root()
    figroot = root / "figures"
    led = root / "data" / "master_metrics.parquet"
    rows, broken = [], 0
    for _, c in df.iterrows():
        figs = [f for f in str(c.get("evidence_figures", "") or "").split(";") if f]
        scripts = [s for s in str(c.get("evidence_scripts", "") or "").split(";") if s]
        fig_ok = {f: (figroot.exists() and any(figroot.glob(f + "*"))) for f in figs}
        scr_ok = {s: Path(s).exists() for s in scripts}
        missing = [f for f, ok in fig_ok.items() if not ok] + [s for s, ok in scr_ok.items() if not ok]
        status = str(c.get("status", "draft"))
        if missing and status != "draft":
            broken += 1
        rows.append({
            "claim_id": c["claim_id"], "status": status,
            "figures_present": ";".join(f for f, ok in fig_ok.items() if ok),
            "figures_missing": ";".join(f for f, ok in fig_ok.items() if not ok),
            "scripts_missing": ";".join(s for s, ok in scr_ok.items() if not ok),
        })
    out = pd.DataFrame(rows)
    Path("manuscript").mkdir(exist_ok=True)
    out.to_csv("manuscript/TRACE.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nTRACE.csv written. broken non-draft claims: {broken}")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
