"""
scaffold_viz.py — publication-grade figure & data output factory.

Import in EVERY analysis script so all outputs share one style and contract.
Per the master plan, each panel is emitted as:
  - vector draft .pdf + .svg   (Illustrator-editable: fonts kept as text)
  - raster .tiff (600 dpi, for images/renders) + .png preview
  - data .xlsx (tidy + pivot + meta) + .csv mirror   (Prism-ready)
Colour: perceptually-uniform / colourblind-safe by default.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd

# ---- palettes / colormaps (colourblind-safe) ----
PALETTE = ["#000000", "#E69F00", "#56B4E9", "#009E73",
           "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]   # Okabe-Ito
CMAP_SEQ = "cividis"     # sequential quantitative fields (thickness, permeability, etc.)
CMAP_ERR = "inferno"     # |error| / hallucination
CMAP_DIV = "RdBu_r"      # signed difference
CMAP_IMG = "gray"        # raw structural CT/SEM slices (publication convention)

SINGLE_COL_MM = 89.0
DOUBLE_COL_MM = 183.0


def mm(x: float) -> float:
    """millimetres -> inches (for figsize)."""
    return x / 25.4


def set_pub_style() -> None:
    """Apply publication rcParams. Call once at the top of each script."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,     # embed fonts as editable TrueType (Illustrator)
        "ps.fonttype": 42,
        "svg.fonttype": "none",  # keep SVG text as <text> (editable)
        "image.cmap": CMAP_SEQ,
    })


def figure(width_mm: float = SINGLE_COL_MM, height_mm: float | None = None, **kw):
    """A single-axes figure sized to a journal column width. Returns (fig, ax)."""
    if height_mm is None:
        height_mm = width_mm * 0.72
    return plt.subplots(figsize=(mm(width_mm), mm(height_mm)),
                        constrained_layout=True, **kw)


def save_figure(fig, stem: str, out_dir, formats=("pdf", "svg", "png"),
                dpi: int = 600, close: bool = True) -> list[str]:
    """Save a matplotlib figure to several formats; returns the file paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in formats:
        p = out_dir / f"{stem}.{f}"
        if f == "tiff":
            fig.savefig(p, dpi=dpi, pil_kwargs={"compression": "tiff_lzw"})
        else:
            fig.savefig(p, dpi=dpi)
        paths.append(str(p))
    if close:
        plt.close(fig)
    return paths


def add_scalebar(ax, length_um: float, voxel_um: float, color: str = "white",
                 pad_frac: float = 0.06, thickness_frac: float = 0.02,
                 label: bool = True, fontsize: int = 6) -> None:
    """Draw a scalebar on an imshow axis (origin='upper'). Stays vector in SVG."""
    px = length_um / voxel_um
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    xmin, xmax = min(xlim), max(xlim)
    ymin, ymax = min(ylim), max(ylim)           # ymax = visual bottom (origin upper)
    W, H = xmax - xmin, ymax - ymin
    x0 = xmax - pad_frac * W - px
    y0 = ymax - pad_frac * H
    ax.add_patch(Rectangle((x0, y0), px, thickness_frac * H,
                           color=color, ec="none", zorder=5))
    if label:
        ax.text(x0 + px / 2, y0 - 0.012 * H, f"{length_um:g} \u00b5m",
                color=color, ha="center", va="bottom", fontsize=fontsize, zorder=5)


def save_image_panel(arr, stem: str, out_dir, cmap: str = CMAP_IMG,
                     voxel_um: float | None = None, scalebar_um: float | None = None,
                     cbar_label: str | None = None, vmin=None, vmax=None,
                     width_mm: float = SINGLE_COL_MM, close: bool = True) -> list[str]:
    """Render a 2D array to a publication image panel (tiff+png+svg)."""
    arr = np.asarray(arr)
    if vmin is None:
        vmin = np.percentile(arr, 1)
    if vmax is None:
        vmax = np.percentile(arr, 99)
    fig, ax = figure(width_mm=width_mm, height_mm=width_mm)
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if scalebar_um and voxel_um:
        add_scalebar(ax, scalebar_um, voxel_um)
    if cbar_label is not None:
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(cbar_label); cb.ax.tick_params(labelsize=6)
    return save_figure(fig, stem, out_dir, formats=("tiff", "png", "svg"), close=close)


def orthoviews(vol, stem: str, out_dir, voxel_um: float | None = None,
               scalebar_um: float | None = None, cmap: str = CMAP_IMG,
               vmin=None, vmax=None, width_mm: float = DOUBLE_COL_MM,
               close: bool = True) -> list[str]:
    """Central XY/XZ/YZ slices of a 3D volume in one row (tiff+png+svg)."""
    vol = np.asarray(vol)
    z, y, x = vol.shape
    slices = [vol[z // 2], vol[:, y // 2, :], vol[:, :, x // 2]]
    titles = ["XY (axial)", "XZ (coronal)", "YZ (sagittal)"]
    if vmin is None:
        vmin = np.percentile(vol, 1)
    if vmax is None:
        vmax = np.percentile(vol, 99)
    fig, axs = plt.subplots(1, 3, figsize=(mm(width_mm), mm(width_mm / 3 * 1.08)),
                            constrained_layout=True)
    for ax, s, t in zip(axs, slices, titles):
        ax.imshow(s, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(t); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    if scalebar_um and voxel_um:
        add_scalebar(axs[0], scalebar_um, voxel_um)
    return save_figure(fig, stem, out_dir, formats=("tiff", "png", "svg"), close=close)


def error_map(pred, target, stem: str, out_dir, mode: str = "signed",
              voxel_um: float | None = None, scalebar_um: float | None = None,
              width_mm: float = SINGLE_COL_MM, close: bool = True) -> list[str]:
    """|error| ('abs') or signed difference ('signed'/'excess') map of a central slice."""
    pred = np.asarray(pred, float); target = np.asarray(target, float)
    d = pred - target
    if d.ndim == 3:
        d = d[d.shape[0] // 2]
    if mode == "abs":
        d = np.abs(d); cmap = CMAP_ERR; vmin, vmax = 0, np.percentile(d, 99)
        label = "|error|"
    else:  # 'signed' / 'excess'  (positive = pred has more structure than truth)
        m = np.percentile(np.abs(d), 99) or 1.0
        cmap, vmin, vmax = CMAP_DIV, -m, m
        label = "pred \u2212 target"
    fig, ax = figure(width_mm=width_mm, height_mm=width_mm)
    im = ax.imshow(d, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(label); cb.ax.tick_params(labelsize=6)
    if scalebar_um and voxel_um:
        add_scalebar(ax, scalebar_um, voxel_um, color="black")
    return save_figure(fig, stem, out_dir, formats=("tiff", "png", "svg"), close=close)


def write_workbook(sheets: dict, path, meta: dict | None = None,
                   also_csv: bool = True, index: bool = False) -> str:
    """Write an .xlsx (one sheet per dataframe + a 'meta' sheet) and CSV mirrors."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=index)
        if meta:
            pd.DataFrame({"key": list(meta), "value": [str(v) for v in meta.values()]}
                         ).to_excel(xw, sheet_name="meta", index=False)
    if also_csv:
        for name, df in sheets.items():
            df.to_csv(path.with_name(f"{path.stem}__{name}.csv"), index=index)
    return str(path)


def montage(vol, stem: str, out_dir, n: int = 16, axis: int = 0,
            cmap: str = CMAP_IMG, voxel_um: float | None = None,
            scalebar_um: float | None = None, vmin=None, vmax=None,
            ncols: int | None = None, width_mm: float = DOUBLE_COL_MM,
            close: bool = True) -> list[str]:
    """Tile n evenly-spaced slices along an axis into a grid (tiff+png+svg).
    Consistent windowing across tiles so group/contrast changes are visible."""
    vol = np.moveaxis(np.asarray(vol), axis, 0)
    Z = vol.shape[0]
    idx = np.unique(np.linspace(0, Z - 1, min(n, Z)).round().astype(int))
    if vmin is None:
        vmin = np.percentile(vol, 1)
    if vmax is None:
        vmax = np.percentile(vol, 99)
    n_show = len(idx)
    if ncols is None:
        ncols = int(np.ceil(np.sqrt(n_show)))
    nrows = int(np.ceil(n_show / ncols))
    fig, axs = plt.subplots(nrows, ncols,
                            figsize=(mm(width_mm), mm(width_mm * nrows / ncols)),
                            constrained_layout=True)
    axs = np.atleast_1d(axs).ravel()
    for a in axs:
        a.axis("off")
    for k, i in enumerate(idx):
        ax = axs[k]
        ax.imshow(vol[i], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"slice {int(i)}", fontsize=5, pad=1)
        if k == 0 and scalebar_um and voxel_um:
            add_scalebar(ax, scalebar_um, voxel_um)
    return save_figure(fig, stem, out_dir, formats=("tiff", "png", "svg"), close=close)
