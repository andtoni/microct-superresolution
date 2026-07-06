"""atomicio.py -- crash-safe writes + resumable checkpoints.

No script writes a final-named file directly: write to a temp sibling, fsync,
then os.replace() (atomic on the same NTFS volume). A killed run never leaves a
torn parquet/nii/checkpoint, and a sync client never uploads a half-written file.
Pairs with runlog.py (run-state ledger) and reconcile.py (STATE.md) to make a
session resumable from committed artifacts alone.
"""
from __future__ import annotations
import json
import os
import uuid
from pathlib import Path


def write_atomic(path, write_fn, mode="wb"):
    """Produce `path` atomically. `write_fn(handle)` writes into a temp file
    opened in `mode`; on success it is fsync'd and os.replace()'d onto `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex[:8]}")
    try:
        with open(tmp, mode) as fh:
            write_fn(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)            # atomic rename on the same volume
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return str(path)


def write_text_atomic(path, text):
    return write_atomic(path, lambda fh: fh.write(text.encode("utf-8")))


def write_json_atomic(path, obj):
    return write_text_atomic(path, json.dumps(obj, indent=2))


def save_checkpoint(out_dir, state, *, capture_rng=True):
    """Atomically save a resumable checkpoint: payload (weights/optim/etc.) +
    progress.json (scalar fields) + RNG state (torch/cuda/numpy/python) so a
    resume after a crash is bit-faithful. `state` is any picklable dict."""
    import torch
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(state)
    if capture_rng:
        import random
        import numpy as np
        payload["_rng"] = {
            "torch": torch.get_rng_state(),
            "cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }
    tmp = out_dir / f".checkpoint.pt.tmp.{uuid.uuid4().hex[:8]}"
    torch.save(payload, tmp)
    os.replace(tmp, out_dir / "checkpoint.pt")
    scalars = {k: v for k, v in state.items() if isinstance(v, (int, float, str, bool))}
    write_json_atomic(out_dir / "progress.json", scalars)
    return str(out_dir / "checkpoint.pt")


def load_latest(out_dir):
    """Return the checkpoint payload (restoring RNG state in-process) or None."""
    import torch
    p = Path(out_dir) / "checkpoint.pt"
    if not p.exists():
        return None
    payload = torch.load(p, map_location="cpu", weights_only=False)
    rng = payload.pop("_rng", None)
    if rng:
        try:
            import random
            import numpy as np
            torch.set_rng_state(rng["torch"])
            if rng.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python"])
        except Exception:
            pass
    return payload
