#!/usr/bin/env python3
r"""
txm_to_slices.py  --  Convert a Zeiss Xradia .txm (reconstructed volume) into a
folder of zero-padded single-slice TIFFs, matching the existing ct/ data layout
(e.g. ct/lr_pel/0000.tif, 0001.tif, ...).

A .txm is a reconstructed tomographic volume stored in a Zeiss OLE2 container.
txrm2tiff only handles .txrm/.xrm (projection) files, so this reads the OLE
streams directly with olefile (already installed). The native reconstructed
dtype is preserved verbatim -- do NOT rescale here; the SR pipeline
percentile-normalises on load.

Xradia OLE layout used here:
    ImageInfo/ImageWidth   (int32)
    ImageInfo/ImageHeight  (int32)
    ImageInfo/NoOfImages   (int32)
    ImageInfo/DataType     (int32 Xradia code; 5=uint16, 10=float32, ...)
    ImageData{g}/Image{n}  pixel data for slice n (width*height of DataType)

Usage (uv):
    uv add olefile tifffile numpy
    # inspect structure first (no files written):
    uv run python txm_to_slices.py --txm "<path>\lr_pel.txm" --inspect
    # then convert:
    uv run python txm_to_slices.py --txm "<path>\lr_pel.txm" --out "<path>\ct\lr_pel"

Sanity-check before aligning:
    uv run python -c "from sr_validation_harness import load_stack; v=load_stack(r'<path>\ct\lr_pel'); print(v.shape, v.dtype, v.min(), v.max())"
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import olefile
import tifffile

# Xradia DataType code -> numpy dtype (from Zeiss XrmDataTypes enum)
XRADIA_DTYPE = {
    2: np.int8, 3: np.uint8, 4: np.int16, 5: np.uint16,
    6: np.int32, 7: np.uint32, 8: np.int64, 9: np.uint64,
    10: np.float32, 11: np.float64,
}


def _read_int(ole, stream_path):
    """Read the first little-endian int32 from an OLE stream."""
    data = ole.openstream(stream_path).read()
    return struct.unpack("<i", data[:4])[0]


def _image_streams(ole):
    """Map slice-number -> stream-path for every ImageData*/Image* stream present."""
    streams = {}
    for entry in ole.listdir():
        if len(entry) == 2 and entry[0].startswith("ImageData") and entry[1].startswith("Image"):
            try:
                num = int(entry[1][len("Image"):])
            except ValueError:
                continue
            streams[num] = entry
    return streams


def inspect_txm(txm_path):
    ole = olefile.OleFileIO(str(txm_path))
    try:
        width = _read_int(ole, "ImageInfo/ImageWidth")
        height = _read_int(ole, "ImageInfo/ImageHeight")
        n_imgs = _read_int(ole, "ImageInfo/NoOfImages")
        dcode = _read_int(ole, "ImageInfo/DataType")
        streams = _image_streams(ole)
        print(f"  ImageWidth  : {width}")
        print(f"  ImageHeight : {height}")
        print(f"  NoOfImages  : {n_imgs}")
        print(f"  DataType    : {dcode} -> {np.dtype(XRADIA_DTYPE.get(dcode, np.uint16)).name}"
              + ("" if dcode in XRADIA_DTYPE else "  (UNKNOWN code, defaulting uint16 -- check!)"))
        print(f"  Image*/streams found : {len(streams)}"
              + ("" if len(streams) == n_imgs else f"  (MISMATCH vs NoOfImages={n_imgs})"))
        # show a couple of example top-level entries for orientation
        tops = sorted({e[0] for e in ole.listdir() if e})
        print(f"  top-level entries    : {tops[:12]}{' ...' if len(tops) > 12 else ''}")
        return dict(width=width, height=height, n_imgs=n_imgs, dcode=dcode, streams=streams)
    finally:
        ole.close()


def read_xradia_volume(txm_path):
    """Read a .txm reconstructed volume as (Z, Y, X) in its native dtype."""
    ole = olefile.OleFileIO(str(txm_path))
    try:
        width = _read_int(ole, "ImageInfo/ImageWidth")
        height = _read_int(ole, "ImageInfo/ImageHeight")
        dcode = _read_int(ole, "ImageInfo/DataType")
        dtype = np.dtype(XRADIA_DTYPE.get(dcode, np.uint16))
        streams = _image_streams(ole)
        if not streams:
            raise ValueError("No ImageData*/Image* streams found -- not a standard Xradia volume?")
        nums = sorted(streams)
        vol = np.empty((len(nums), height, width), dtype=dtype)
        for z, n in enumerate(nums):
            buf = ole.openstream(streams[n]).read()
            sl = np.frombuffer(buf, dtype=dtype, count=width * height)
            vol[z] = sl.reshape(height, width)
        return vol
    finally:
        ole.close()


def write_slice_folder(vol, out_dir, prefix, pad):
    out_dir.mkdir(parents=True, exist_ok=True)
    n = vol.shape[0]
    width = max(pad, len(str(n - 1)))
    for i in range(n):
        tifffile.imwrite(str(out_dir / f"{prefix}{i:0{width}d}.tif"), vol[i], dtype=vol.dtype)
    print(f"  wrote {n} slices -> {out_dir}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Convert a Zeiss .txm reconstructed volume to a TIFF slice folder.")
    ap.add_argument("--txm", required=True, help="Path to the reconstructed .txm file")
    ap.add_argument("--out", help="Output folder for the slice stack (omit with --inspect)")
    ap.add_argument("--inspect", action="store_true", help="Print OLE structure and exit (no files written)")
    ap.add_argument("--prefix", default="", help="Slice filename prefix (default: none)")
    ap.add_argument("--pad", type=int, default=4, help="Min zero-padding width (default 4)")
    args = ap.parse_args(argv)

    txm_path = Path(args.txm)
    if not txm_path.exists():
        sys.exit(f"ERROR: .txm not found: {txm_path}")
    if not olefile.isOleFile(str(txm_path)):
        sys.exit(f"ERROR: not a valid OLE2/Xradia file: {txm_path}")

    print(f"Inspecting {txm_path.name} ...")
    inspect_txm(txm_path)
    if args.inspect:
        print("Inspect-only mode; no files written.")
        return
    if not args.out:
        sys.exit("ERROR: --out is required unless --inspect is given.")

    print("Reading volume ...")
    vol = read_xradia_volume(txm_path)
    print(f"Loaded volume: shape={vol.shape} dtype={vol.dtype} "
          f"min={vol.min()} max={vol.max()} mean={vol.mean():.4g}")
    write_slice_folder(vol, Path(args.out), args.prefix, args.pad)
    print("Done. Inspect a mid-stack slice (axial cross-section, polymer bright),")
    print("then run align_pair.py with --lr pointing at this folder.")


if __name__ == "__main__":
    main()
