#!/usr/bin/env python3
r"""
qc_alignment.py  --  re-score an already-saved aligned pair without re-running alignment.

Loads lr_input.tif / hr_target.tif / overlap_mask.tif from an aligned-pair folder, recomputes
the foreground + blur-matched + per-slice alignment metrics, redraws align_qc.png, and updates
align_meta.json in place. Use it after changing the QC logic, or to inspect any cached pair.

Usage:
  uv run python qc_alignment.py --dir "C:\...\codeoutput\aligned\vcr276"
"""
import argparse, json, os
from sr_validation_harness import load_stack
from align_pair import alignment_quality, masked_ncc, save_qc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="aligned-pair folder")
    args = ap.parse_args()

    hr = load_stack(os.path.join(args.dir, "hr_target.tif"))
    lr = load_stack(os.path.join(args.dir, "lr_input.tif"))
    mask = load_stack(os.path.join(args.dir, "overlap_mask.tif"))

    q = alignment_quality(hr, lr, mask)
    g = masked_ncc(hr, lr, mask)
    print(f"[qc] blur-matched NCC={q['ncc_blur_matched']:.3f}  foreground={q['ncc_foreground']:.3f}"
          f"  (global={g:.3f})")
    print(f"[qc] per-slice foreground NCC: median={q['slice_ncc_median']:.3f}  "
          f"p10={q['slice_ncc_p10']:.3f}")
    print(f"[qc] peak-sharpness: base={q['peak_base_ncc']:.3f}  "
          f"min nudge drop={q['peak_min_drop']:+.3f}  "
          f"-> {'LOCKED' if q['locked'] else 'NOT a clean local max'}")
    print(f"[qc] verdict -> {q['verdict']}")

    save_qc(hr, lr, mask, args.dir)
    mp = os.path.join(args.dir, "align_meta.json")
    if os.path.exists(mp):
        meta = json.load(open(mp))
        meta.update(ncc_global=g, ncc_foreground=q["ncc_foreground"],
                    ncc_blur_matched=q["ncc_blur_matched"],
                    slice_ncc_median=q["slice_ncc_median"], slice_ncc_p10=q["slice_ncc_p10"],
                    peak_base_ncc=q["peak_base_ncc"], peak_min_drop=q["peak_min_drop"],
                    peak_mean_drop=q["peak_mean_drop"], locked=q["locked"],
                    verdict=q["verdict"])
        json.dump(meta, open(mp, "w"), indent=2)
        print(f"[qc] updated {mp}")


if __name__ == "__main__":
    main()
