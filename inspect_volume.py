r"""
inspect_volume.py — quick, reusable QC look at any scan (HR / LR / SR).

Emits orthoviews, a slice montage, and a per-slice mean-intensity profile
(the profile reveals axial group boundaries / tape gaps in combined scans),
plus an Excel of the profile and a provenance manifest.

Usage:
    uv run python inspect_volume.py --stack "<...\ct\lr_pel>" --name lr_pel --voxel 2.4442 --scalebar 500
    uv run python inspect_volume.py --stack "<...\ct\hr_pel18>" --name hr_pel18 --voxel 0.9696 --scalebar 100
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sr_validation_harness import load_stack
import scaffold_viz as sv
import provenance as prov
import paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect a scan volume (QC).")
    ap.add_argument("--stack", required=True, help="folder of slices or a multipage .tif")
    ap.add_argument("--name", required=True)
    ap.add_argument("--voxel", type=float, default=None, help="voxel size in µm (for scalebar)")
    ap.add_argument("--scalebar", type=float, default=200.0, help="scalebar length in µm")
    ap.add_argument("--montage-n", type=int, default=16)
    ap.add_argument("--out", default=None, help="output folder (default figures/QC_inspect/<name>)")
    args = ap.parse_args()

    sv.set_pub_style()
    paths.make_tree()
    out = Path(args.out) if args.out else (paths.FIGURES / "QC_inspect" / args.name)
    out.mkdir(parents=True, exist_ok=True)

    vol = load_stack(args.stack)            # raw float32; viz helpers percentile-window
    sb = args.scalebar if args.voxel else None
    print(f"{args.name}: shape={vol.shape} dtype={vol.dtype} "
          f"min={vol.min():.4g} max={vol.max():.4g} "
          f"p1={np.percentile(vol,1):.4g} p50={np.percentile(vol,50):.4g} "
          f"p99={np.percentile(vol,99):.4g}")

    files = []
    files += sv.orthoviews(vol, f"{args.name}_orthoviews", out,
                           voxel_um=args.voxel, scalebar_um=sb)
    files += sv.montage(vol, f"{args.name}_zmontage", out, n=args.montage_n,
                        axis=0, voxel_um=args.voxel, scalebar_um=sb)

    # per-slice mean intensity along Z — steps/dips flag group boundaries
    prof = vol.reshape(vol.shape[0], -1).mean(axis=1)
    df = pd.DataFrame({"z_index": np.arange(vol.shape[0]),
                       "mean_intensity": prof.astype(np.float32)})
    fig, ax = sv.figure(120, 50)
    ax.plot(df["z_index"], df["mean_intensity"], color=sv.PALETTE[5], lw=0.8)
    ax.set_xlabel("z slice index")
    ax.set_ylabel("mean intensity (a.u.)")
    ax.set_title(f"{args.name}: axial intensity profile", fontsize=7)
    files += sv.save_figure(fig, f"{args.name}_zprofile", out)

    sv.write_workbook({"z_profile": df}, out / f"{args.name}_inspect_data.xlsx",
                      meta={"stack": args.stack, "shape": list(vol.shape),
                            "voxel_um": args.voxel,
                            "intensity_p1_p50_p99": [float(np.percentile(vol, p))
                                                     for p in (1, 50, 99)]})
    prov.stamp(out, params={"stack": args.stack, "voxel_um": args.voxel,
                            "shape": list(vol.shape)}, inputs=[args.stack])

    print("Wrote QC to:", out)
    for f in sorted(out.iterdir()):
        print("   ", f.name)


if __name__ == "__main__":
    main()
