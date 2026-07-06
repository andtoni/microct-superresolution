#!/usr/bin/env python3
"""validate_artifacts.py -- enforce the artifact contract + provenance integrity.

Run on demand or in CI (NOT per-commit: it scans the data tree, not staged files;
the per-commit guard is the 5 MB size-gate). Checks:
  1. every figures/<dir>/manifest.json validates against schemas/manifest.schema.json
  2. every figure dir has the contract minimum (>=1 *_data.csv; PDF implies SVG)
  3. every master_metrics row's manifest_ref resolves to a manifest on disk
  4. every metric name is in the data_dictionary controlled vocabulary
Exit 0 = clean, 1 = violations.

    uv run python validate_artifacts.py [--data-root <codeoutput>]
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path


def _data_root(arg):
    if arg:
        return Path(arg)
    if os.environ.get("SCAFFX_DATA_ROOT"):
        return Path(os.environ["SCAFFX_DATA_ROOT"])
    try:
        import paths
        return Path(paths.OUTPUT_ROOT)
    except Exception:
        return Path("codeoutput")


def main(argv=None):
    ap = argparse.ArgumentParser(description="validate the artifact contract + provenance")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--schema", default="schemas/manifest.schema.json")
    ap.add_argument("--dict", default="schemas/data_dictionary.yaml")
    a = ap.parse_args(argv)
    root = _data_root(a.data_root)
    problems = []

    import jsonschema
    import yaml
    schema = json.loads(Path(a.schema).read_text(encoding="utf-8"))
    vocab = set(yaml.safe_load(Path(a.dict).read_text(encoding="utf-8"))["metrics"])

    figroot = root / "figures"
    fig_dirs = [d for d in figroot.glob("*") if d.is_dir()] if figroot.exists() else []
    for d in fig_dirs:
        man = d / "manifest.json"
        if not man.exists():
            problems.append(f"{d.name}: no manifest.json")
        else:
            try:
                jsonschema.validate(json.loads(man.read_text(encoding="utf-8")), schema)
            except Exception as e:
                problems.append(f"{d.name}: manifest invalid ({str(e).splitlines()[0]})")
        if not list(d.glob("*_data.csv")):
            problems.append(f"{d.name}: no *_data.csv (contract incomplete)")
        if list(d.glob("*.pdf")) and not list(d.glob("*.svg")):
            problems.append(f"{d.name}: PDF present but no SVG (not Adobe-editable)")

    led = root / "data" / "master_metrics.parquet"
    if led.exists():
        import pandas as pd
        df = pd.read_parquet(led)
        bad = sorted(set(df["metric"].dropna()) - vocab)
        if bad:
            problems.append(f"ledger: metric(s) not in controlled vocabulary: {bad}")
        refs = df.get("manifest_ref", pd.Series(dtype=str)).dropna().unique()
        missing = [r for r in refs if not Path(r).exists()]
        if missing:
            problems.append(f"ledger: {len(missing)} manifest_ref(s) missing on disk")

    if problems:
        print("ARTIFACT VALIDATION FAILED:")
        for p in problems:
            print("  -", p)
        return 1
    print(f"artifact validation OK ({len(fig_dirs)} figure dir(s), ledger checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
