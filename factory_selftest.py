"""
factory_selftest.py — verify the output contract end-to-end.

Run after setting up the environment:
    uv run python factory_selftest.py
It builds a synthetic phantom and emits a vector plot, image panels, an error
map, an Excel workbook (tidy + pivot + meta) with CSV mirrors, and a provenance
manifest into  <SCAFFOLD_OUT>/figures/S0_factory_selftest/ .
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

import scaffold_viz as sv
import provenance as prov
import paths


def porous_phantom(shape=(64, 160, 160), seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = gaussian_filter(rng.normal(size=shape), 2.2)
    return (v > 0.0).astype(np.float32)   # bright solid, dark pores


def main() -> None:
    sv.set_pub_style()
    root = paths.make_tree()
    out = paths.fig_dir("S0_factory_selftest")
    voxel, sb = 0.9696, 20.0

    hr = porous_phantom(seed=1)
    sr = gaussian_filter(hr, 0.6)                                   # faithful-ish
    halluc = sr + 0.18 * np.random.default_rng(3).normal(size=hr.shape)  # fabricated texture

    files = []
    files += sv.orthoviews(hr, "S0_orthoviews", out, voxel_um=voxel, scalebar_um=sb)
    files += sv.save_image_panel(hr[hr.shape[0] // 2], "S0_image_slice", out,
                                 voxel_um=voxel, scalebar_um=sb,
                                 cbar_label="intensity (a.u.)")
    files += sv.error_map(halluc, hr, "S0_error_excess", out, mode="excess",
                          voxel_um=voxel, scalebar_um=sb)

    # a vector line-plot panel
    fig, ax = sv.figure(89, 60)
    x = np.linspace(0, 1, 50)
    ax.plot(x, np.sqrt(x), color=sv.PALETTE[5], label="SR")
    ax.plot(x, x, color=sv.PALETTE[6], ls="--", label="interp")
    ax.set_xlabel("upscaling factor (norm.)")
    ax.set_ylabel("fidelity (norm.)")
    ax.legend(frameon=False)
    files += sv.save_figure(fig, "S0_plot", out)

    # data workbook: tidy (long) + prism pivot (wide) + meta
    tidy = pd.DataFrame({
        "sample":   ["demo"] * 6,
        "family":   ["phantom"] * 6,
        "modality": ["HR", "SR", "interp", "HR", "SR", "interp"],
        "metric":   ["porosity_frac"] * 3 + ["psnr_dB"] * 3,
        "value":    [float(1 - hr.mean()), float(1 - sr.mean()), float(1 - sr.mean()),
                     np.nan, 34.2, 28.1],
    })
    pivot = (tidy.pivot_table(index=["sample", "metric"], columns="modality",
                              values="value").reset_index())
    sv.write_workbook({"tidy": tidy, "prism_pivot": pivot},
                      out / "S0_factory_selftest_data.xlsx",
                      meta={"voxel_um": voxel, "scalebar_um": sb,
                            "phantom_shape": list(hr.shape), "seed": 1,
                            "note": "factory self-test"})

    prov.stamp(out, params={"phantom_shape": list(hr.shape), "voxel_um": voxel,
                            "seed": 1}, inputs=[])

    print("OUTPUT ROOT :", root)
    print("PANEL FOLDER:", out)
    for f in sorted(out.iterdir()):
        print("   ", f.name, f"({f.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
