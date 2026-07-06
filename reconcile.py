#!/usr/bin/env python3
"""reconcile.py -- regenerate STATE.md from committed artifacts.

STATE.md is [GEN]: never hand-edited. It answers "the session died and I lost all
in-memory context -- where exactly was I?" It is computed from RUNS.jsonl (the run
ledger), git status, the Snakemake DAG frontier, and HEAD, so resumption needs no
memory. /session-start runs this first and flags STATE.md stale if derived_from_commit
!= HEAD.

    uv run python reconcile.py        # writes ./STATE.md
"""
from __future__ import annotations
import json
import os
import socket
import subprocess
import time
from pathlib import Path

import atomicio


def _sh(*args):
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def _data_root():
    env = os.environ.get("SCAFFX_DATA_ROOT")
    if env:
        return Path(env)
    try:
        import paths
        return Path(paths.OUTPUT_ROOT)
    except Exception:
        return Path("logs")


def _pid_alive(pid, host):
    if not pid or host != socket.gethostname():
        return None                      # different host -> can't tell
    try:
        if os.name == "nt":
            import ctypes
            h = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
            if h:
                ctypes.windll.kernel32.CloseHandle(h)
                return True
            return False
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _runs():
    led = _data_root() / "logs" / "runs" / "RUNS.jsonl"
    if not led.exists():
        return {}
    by = {}
    for line in led.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
        except Exception:
            continue
        by[d["run_id"]] = d              # last line per run_id wins
    return by


def _torch_env():
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
        return f"torch {torch.__version__} | cuda {getattr(torch.version, 'cuda', None)} | gpu {gpu}"
    except Exception:
        return "torch: not importable"


def build():
    commit = _sh("git", "rev-parse", "HEAD")
    branch = _sh("git", "rev-parse", "--abbrev-ref", "HEAD")
    dirty = _sh("git", "status", "--short")

    running, recent = [], []
    for d in _runs().values():
        if d.get("state") in ("RUNNING", "HEARTBEAT"):
            alive = _pid_alive(d.get("pid"), d.get("host"))
            tag = "RUNNING" if alive else ("ORPHANED" if alive is False else "RUNNING?")
            running.append((d, tag))
        else:
            recent.append(d)
    recent = sorted(recent, key=lambda d: d.get("ts", ""))[-5:]

    dag = "Snakefile not present yet."
    if Path("Snakefile").exists():
        dag = _sh("uv", "run", "snakemake", "-n", "--quiet") or "(dry-run: no output)"

    orphans = [d for d, t in running if t == "ORPHANED"]
    nxt = "See MASTER_PLAN_SR_microCT_v3.md execution sequence + EXPERIMENT_FRAMEWORK.md §9."
    if orphans:
        nxt = f"Relaunch orphaned run -> {orphans[0].get('cmdline', '?')}"

    L = ["# STATE.md  [GEN -- do not edit; regenerate with `uv run python reconcile.py`]",
         f"derived_from_commit: {commit}  ({branch})",
         f"generated: {time.strftime('%Y-%m-%dT%H:%M:%S')}",
         "",
         "## ENVIRONMENT",
         f"- {_torch_env()}",
         f"- git {commit[:10]} on {branch} | dirty: {'yes' if dirty else 'no'}",
         "",
         "## RUN STATE"]
    if running:
        for d, t in running:
            L.append(f"- [{t}] {d.get('name')} -> {d.get('target')} "
                     f"(pid {d.get('pid')}, since {d.get('ts')})")
    else:
        L.append("- no active runs")
    if recent:
        L.append("- recent:")
        for d in recent:
            secs = f", {d.get('seconds')}s" if d.get("seconds") else ""
            L.append(f"  - [{d.get('state')}] {d.get('name')} ({d.get('ts')}{secs})")
    L += ["", "## DAG FRONTIER", "```", dag[:2000], "```",
          "", "## DIRTY / UNTRACKED", "```", (dirty or "(clean)"), "```",
          "", "## NEXT COMMAND", nxt, ""]

    atomicio.write_text_atomic("STATE.md", "\n".join(L))
    return "\n".join(L)


if __name__ == "__main__":
    build()
    print("wrote STATE.md")
