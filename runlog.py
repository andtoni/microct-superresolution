"""runlog.py -- append-only run ledger for crash-safe, resumable state.

    from runlog import run
    with run("train_seg", target="runs/seg_loao_pel16", params={...}) as r:
        for epoch in range(60):
            ...
            r.note(epoch=epoch, val_dice=d)     # heartbeat

Writes one JSON line per state transition (RUNNING -> DONE | FAILED, plus
HEARTBEATs) to <data root>/logs/runs/RUNS.jsonl, fsync'd. Prose ("epoch ~40")
cannot be parsed by a fresh session; this typed, append-only ledger can --
reconcile.py reads the last line per run_id and turns it into STATE.md.

Ledger location: env SCAFFX_DATA_ROOT, else paths.OUTPUT_ROOT, else ./logs.
"""
from __future__ import annotations
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


def _ledger_path():
    env = os.environ.get("SCAFFX_DATA_ROOT")
    if env:
        root = Path(env)
    else:
        try:
            import paths
            root = Path(paths.OUTPUT_ROOT)
        except Exception:
            root = Path("logs")
    p = root / "logs" / "runs" / "RUNS.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def _append(rec):
    p = _ledger_path()
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


class _Run:
    def __init__(self, name, target, params):
        self.id = uuid.uuid4().hex[:12]
        self.name, self.target, self.params = name, target, params
        self.t0 = time.time()

    def _base(self, **extra):
        return {
            "run_id": self.id, "name": self.name, "target": self.target,
            "git_commit": _git_commit(), "host": socket.gethostname(),
            "pid": os.getpid(), "cmdline": " ".join(sys.argv),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra,
        }

    def note(self, **fields):
        """Emit a HEARTBEAT line (progress markers: epoch, metric, ...)."""
        _append(self._base(state="HEARTBEAT", **fields))


@contextmanager
def run(name, target=None, params=None):
    r = _Run(name, target, params or {})
    _append(r._base(state="RUNNING", params=r.params))
    try:
        yield r
    except BaseException as e:
        _append(r._base(state="FAILED", error=f"{type(e).__name__}: {e}",
                        seconds=round(time.time() - r.t0, 1)))
        raise
    else:
        _append(r._base(state="DONE", seconds=round(time.time() - r.t0, 1)))
