# /// script
# requires-python = ">=3.12"
# dependencies = ["mlflow", "protobuf<5"]
# ///
"""mlflow_setup.py -- OPT-IN, non-disruptive mirror of the runlog ledger to MLflow.

ISOLATION: this runs in its OWN uv environment (PEP 723 inline deps above) so it never
touches the training repo's dependency tree. Run it with `uv run mlflow_setup.py` and uv
builds a throwaway env with mlflow + a compatible protobuf. Point it at the ledger with
`--ledger PATH` or the SCAFFX_DATA_ROOT env (same override runlog.py uses).


The append-only ``RUNS.jsonl`` ledger (see runlog.py) stays the source of truth.
This module *mirrors* completed runs to the self-hosted MLflow tracking server so
experiments are browsable/comparable at https://mlflow.example.com -- WITHOUT
changing any training-pipeline logic. It is a no-op unless MLFLOW_TRACKING_URI is set.

Usage:
    export MLFLOW_TRACKING_URI=https://mlflow.example.com
    uv run python mlflow_setup.py

Idempotent: each MLflow run is tagged ``scaffx_run_id``; already-mirrored runs are skipped.
Mapping: RUNNING.params -> MLflow params; numeric HEARTBEAT fields -> metrics (step=epoch
if present); git_commit/host/target/cmdline/final-state/seconds -> tags. One MLflow
experiment per ledger ``name`` (e.g. train_seg, train_sr).
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path


def enabled() -> bool:
    return bool(os.environ.get("MLFLOW_TRACKING_URI"))


def _ledger_path() -> Path:
    """Same resolution order as runlog.py so we read the exact same ledger."""
    env = os.environ.get("SCAFFX_DATA_ROOT")
    if env:
        root = Path(env)
    else:
        try:
            import paths
            root = Path(paths.OUTPUT_ROOT)
        except Exception:
            root = Path("logs")
    return root / "logs" / "runs" / "RUNS.jsonl"


def _read_runs(ledger: Path) -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = defaultdict(list)
    with open(ledger, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = rec.get("run_id")
            if rid:
                runs[rid].append(rec)
    return runs


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def mirror_runs(ledger: Path | None = None) -> int:
    """Mirror every completed (DONE/FAILED) ledger run to MLflow. Returns count mirrored."""
    if not enabled():
        print("MLFLOW_TRACKING_URI not set -- mirror skipped (no-op). Set it to enable.")
        return 0

    import mlflow
    from mlflow.tracking import MlflowClient

    ledger = ledger or _ledger_path()
    if not ledger.exists():
        print(f"no ledger at {ledger} -- nothing to mirror.")
        return 0

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()
    runs = _read_runs(ledger)
    mirrored = 0

    for rid, recs in runs.items():
        finals = [r for r in recs if r.get("state") in ("DONE", "FAILED")]
        if not finals:
            continue  # still running -- mirror only terminal runs
        final = finals[-1]
        name = final.get("name") or "unknown"

        exp = mlflow.set_experiment(name)
        # idempotency: skip if this run_id was already mirrored into this experiment
        existing = client.search_runs(
            [exp.experiment_id], filter_string=f"tags.scaffx_run_id = '{rid}'", max_results=1
        )
        if existing:
            continue

        running = next((r for r in recs if r.get("state") == "RUNNING"), {})
        heartbeats = [r for r in recs if r.get("state") == "HEARTBEAT"]

        with mlflow.start_run(run_name=f"{name}-{rid}"):
            mlflow.set_tags({
                "scaffx_run_id": rid,
                "git_commit": final.get("git_commit") or "",
                "host": final.get("host") or "",
                "target": final.get("target") or "",
                "cmdline": final.get("cmdline") or "",
                "final_state": final.get("state"),
                "mirrored_from": "RUNS.jsonl",
            })
            for k, v in (running.get("params") or {}).items():
                mlflow.log_param(k, v)
            if _is_number(final.get("seconds")):
                mlflow.log_metric("seconds", float(final["seconds"]))
            # heartbeat numeric fields -> metrics (stepped by epoch when present)
            for hb in heartbeats:
                step = hb.get("epoch") if _is_number(hb.get("epoch")) else None
                for k, v in hb.items():
                    if k in ("state", "run_id", "name", "target", "git_commit", "host",
                             "pid", "cmdline", "ts", "epoch"):
                        continue
                    if _is_number(v):
                        mlflow.log_metric(k, float(v), step=int(step) if step is not None else None)
            mirrored += 1

    print(f"mirrored {mirrored} new run(s) to {os.environ['MLFLOW_TRACKING_URI']}")
    return mirrored


if __name__ == "__main__":
    import argparse
    import sys
    try:  # MLflow prints a 🏃 emoji run URL; force UTF-8 so Windows cp1252 doesn't crash
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Mirror RUNS.jsonl to MLflow (opt-in, isolated).")
    ap.add_argument("--ledger", type=Path, default=None,
                    help="path to RUNS.jsonl (else SCAFFX_DATA_ROOT, else paths.OUTPUT_ROOT, else ./logs)")
    args = ap.parse_args()
    raise SystemExit(0 if mirror_runs(args.ledger) >= 0 else 1)
