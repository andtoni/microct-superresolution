#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import tifffile

DEFAULT_SEG = "/data/superresolution/Data/codeoutput/segmentations"


def convert(tif, h5, dset="data", chunk=64, compression="lzf"):
    import h5py
    vol = tifffile.imread(str(tif))
    if vol.ndim != 3:
        vol = np.squeeze(vol)
    ch = tuple(int(min(chunk, s)) for s in vol.shape)
    with h5py.File(str(h5), "w") as f:
        f.create_dataset(dset, data=vol, chunks=ch, compression=compression)
    return vol.shape, vol.dtype


def main(argv=None):
    ap = argparse.ArgumentParser(description="convert de-cupped TIFFs to chunked HDF5 for ilastik")
    ap.add_argument("--seg-root", default=DEFAULT_SEG)
    ap.add_argument("--suffix", default="_decupped.tif")
    ap.add_argument("--in", dest="inp", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dset", default="data")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--compression", default="lzf", choices=["lzf", "gzip", "none"])
    ap.add_argument("--only", default=None)
    args = ap.parse_args(argv)
    comp = None if args.compression == "none" else args.compression

    if args.inp:
        out = args.out or str(Path(args.inp).with_suffix(".h5"))
        shp, dt = convert(args.inp, out, args.dset, args.chunk, comp)
        print(f"   {Path(args.inp).name} {shp} {dt} -> {out}")
        return 0

    seg_root = Path(args.seg_root)
    only = {s.strip() for s in args.only.split(",")} if args.only else None
    n = 0
    for d in sorted(p for p in seg_root.iterdir() if p.is_dir()):
        if only and d.name not in only:
            continue
        tif = d / f"{d.name}{args.suffix}"
        if not tif.exists():
            print(f"[skip] {d.name}: no {tif.name}"); continue
        h5 = tif.with_suffix(".h5")
        t = time.time()
        shp, dt = convert(tif, h5, args.dset, args.chunk, comp)
        print(f"{d.name:9s} {shp} {dt} -> {h5.name}  [{time.time()-t:.0f}s]")
        n += 1
    if n == 0:
        print("[error] no inputs found."); return 2
    print(f"\nconverted {n}. now: run_ilastik_all.py --input-suffix {args.suffix.replace('.tif', '.h5')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
