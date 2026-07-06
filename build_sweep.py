#!/usr/bin/env python3
"""
build_sweep.py -- generate subset manifests for the overnight training sweep.

Reads the full manifest produced by `train_sr_real.py --scan` and writes
leave-one-architecture-out and diversity-tier subsets into <out>/, preserving
the manifest's top-level structure so the trainer reads them identically.

Pairs are grouped by name prefix: pel* -> pellethane, laser* -> laser,
vcr* -> porogen.  Prints the plan so the grouping can be verified.

Usage:
    uv run python build_sweep.py --manifest "<aligned>\\manifest.json" --out "<aligned>\\sweep"
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path


def family(name: str) -> str:
    n = str(name).lower()
    if n.startswith("pel"):
        return "pellethane"
    if n.startswith("laser"):
        return "laser"
    if n.startswith("vcr"):
        return "porogen"
    return "other"


def load(manifest_path: Path):
    """Return (raw_data, list_of_entries). Handles list / dict-with-list / dict-of-dicts."""
    data = json.loads(Path(manifest_path).read_text())
    if isinstance(data, list):
        return data, data
    if isinstance(data, dict):
        for k in ("pairs", "entries", "manifest", "items"):
            if isinstance(data.get(k), list):
                return data, data[k]
        vals = list(data.values())
        if vals and all(isinstance(v, dict) for v in vals):
            # dict keyed by name -> inject the key as 'name' if missing
            entries = []
            for k, v in data.items():
                v = dict(v)
                v.setdefault("name", k)
                entries.append(v)
            return data, entries
    raise ValueError("could not locate pair entries in manifest")


def entry_name(e) -> str:
    if "name" in e:
        return e["name"]
    raise ValueError(f"entry missing 'name': {e}")


def subset(data, entries, keep: set):
    """Rebuild the manifest keeping only entries whose name is in `keep`,
    preserving the original top-level structure."""
    if isinstance(data, list):
        return [e for e in data if entry_name(e) in keep]
    if isinstance(data, dict):
        for k in ("pairs", "entries", "manifest", "items"):
            if isinstance(data.get(k), list):
                d = dict(data)
                d[k] = [e for e in data[k] if entry_name(e) in keep]
                return d
        return {k: v for k, v in data.items()
                if isinstance(v, dict) and v.get("name", k) in keep}
    raise ValueError("unknown manifest structure")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data, entries = load(Path(args.manifest))

    names = [entry_name(e) for e in entries]
    fam = {n: family(n) for n in names}
    families = sorted(set(fam.values()))
    print(f"[build_sweep] {len(names)} pairs: " +
          ", ".join(f"{n}({fam[n]})" for n in names))
    print(f"[build_sweep] families: {families}")

    if len(names) < 2 or len([f for f in families if f != 'other']) < 2:
        print("[build_sweep] ERROR: need >=2 named families; aborting subset build "
              "(v2_production will still train on the full manifest).", file=sys.stderr)
        return 2

    pel = {n for n in names if fam[n] == "pellethane"}
    las = {n for n in names if fam[n] == "laser"}
    por = {n for n in names if fam[n] == "porogen"}
    alln = set(names)

    plan = {
        "loao_no_pellethane":  alln - pel,
        "loao_no_laser":       alln - las,
        "loao_no_porogen":     alln - por,
        "div_1fam_pellethane": pel,
    }
    for label, keep in plan.items():
        if not keep:
            print(f"[build_sweep] skip {label}: empty"); continue
        sub = subset(data, entries, keep)
        p = out / f"{label}.json"
        p.write_text(json.dumps(sub, indent=2))
        kept = sorted(keep)
        print(f"[build_sweep] wrote {label}.json  ({len(kept)} pairs: {', '.join(kept)})")
    print("[build_sweep] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
