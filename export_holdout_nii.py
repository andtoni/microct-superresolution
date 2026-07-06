#!/usr/bin/env python3
"""Export an architecture HR volume as a NIfTI for nnUNetv2_predict (_0000 naming)."""
import argparse
from pathlib import Path
import nnunet_baseline as nb
import morphometry as mp
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    raw = None
    for sub in ("HR", "hr", "high", ""):
        cand = Path(a.raw_root) / a.arch / sub if sub else Path(a.raw_root) / a.arch
        if cand.exists():
            raw = cand; break
    if raw is None:
        raise SystemExit(f"no raw volume for {a.arch} under {a.raw_root}")
    rv = mp.resolve_volume(str(raw))
    vol = nb._load(rv).astype("float32")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{a.arch}_0000.nii.gz"
    nb._write_nii(vol, dst)
    print(f"exported {a.arch} HR -> {dst} ({vol.shape})")
if __name__ == "__main__":
    main()
