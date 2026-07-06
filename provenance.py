"""
provenance.py — reproducibility stamping for every output folder.

Call provenance.stamp(out_dir, params=..., inputs=[...]) at the end of each
analysis step. Writes manifest.json with timestamp, git commit, python &
package versions, input file hashes, and the parameters used.
"""
from __future__ import annotations
import json
import hashlib
import subprocess
import sys
import platform
import datetime
from pathlib import Path
from importlib import metadata as imd

KEY_PACKAGES = [
    "numpy", "scipy", "scikit-image", "matplotlib", "pandas", "pyarrow", "openpyxl",
    "tifffile", "pillow", "torch", "monai", "nnunetv2", "SimpleITK", "porespy", "openpnm",
    "taufactor", "pytrax", "structure-tensor", "skan", "gudhi", "cc3d", "jsonschema",
]


def pkg_versions(names=KEY_PACKAGES) -> dict:
    out = {}
    for n in names:
        try:
            out[n] = imd.version(n)
        except Exception:
            out[n] = None
    return out


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def file_hash(path, chunk=1 << 20) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(chunk), b""):
                h.update(blk)
        return h.hexdigest()
    except Exception:
        return None


def stamp(out_dir, params: dict | None = None, inputs: list | None = None) -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": pkg_versions(),
        "inputs": {str(p): file_hash(p) for p in (inputs or [])},
        "params": params or {},
    }
    p = out_dir / "manifest.json"
    p.write_text(json.dumps(manifest, indent=2))
    return str(p)


def _git(*args):
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def build_manifest(params=None, inputs=None, outputs=None, seeds=None, device=None) -> dict:
    """Manifest dict conforming to schemas/manifest.schema.json (full SHA-256 hashes)."""
    import socket
    return {
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "cmdline": " ".join(sys.argv),
        "cwd": str(Path.cwd()),
        "host": socket.gethostname(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "device": device,
        "git": {"commit": _git("rev-parse", "HEAD"),
                "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": bool(_git("status", "--porcelain"))},
        "versions": pkg_versions(),
        "seeds": seeds or {},
        "params": params or {},
        "inputs": {str(p): file_hash(p) for p in (inputs or [])},
        "outputs": {str(p): file_hash(p) for p in (outputs or [])},
    }


def write_manifest(out_dir, **kw) -> dict:
    """Write a schema-conforming manifest.json atomically; returns the dict."""
    import atomicio
    m = build_manifest(**kw)
    atomicio.write_json_atomic(Path(out_dir) / "manifest.json", m)
    return m
