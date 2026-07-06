"""figure_style.py -- publication styling + Adobe-ready layered export.

Loads assets/figure_style.json. set_pub_style() applies Arial / sizes / line weights;
palette_for(token) returns the semantic colour; layered_savefig(fig, stem) writes an
editable vector PDF (fonttype 42 = embedded TrueType) + SVG (live text) with per-panel
named gids (-> Illustrator layers), plus a 600 dpi TIFF (LZW) + PNG. Satisfies the
Adobe import contract in DATA_TAXONOMY.md SS7.
"""
from __future__ import annotations
import json
import os
import uuid
from pathlib import Path

_STYLE = None


def style():
    global _STYLE
    if _STYLE is None:
        p = Path(__file__).with_name("assets") / "figure_style.json"
        _STYLE = json.loads(p.read_text(encoding="utf-8"))
    return _STYLE


def set_pub_style():
    import matplotlib as mpl
    s = style()
    mpl.rcParams.update({
        "font.family": s["font"]["family"],
        "font.size": s["font"]["size_pt"],
        "axes.linewidth": s["line_pt"],
        "lines.linewidth": s["line_pt"] * 2,
        "savefig.dpi": s["dpi"],
        "pdf.fonttype": s["save"]["pdf_fonttype"],
        "svg.fonttype": s["save"]["svg_fonttype"],
        "figure.dpi": 150,
    })


def palette_for(token, default="#333333"):
    return style()["palette"].get(token, default)


def stars(p):
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"


def layered_savefig(fig, stem, *, raster=True):
    """Save fig as editable PDF + SVG (per-panel named gids) and 600 dpi TIFF + PNG.
    `stem` is a path without extension. Each write is atomic (tmp -> os.replace).
    Returns {ext: path}."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for i, ax in enumerate(fig.get_axes()):
        ax.set_gid(f"panel-{i}")
    dpi = style()["dpi"]
    out = {}

    def _atomic(ext, **kw):
        final = stem.with_suffix("." + ext)
        tmp = final.with_name(f".{final.name}.tmp.{uuid.uuid4().hex[:8]}")
        fig.savefig(tmp, format=ext, bbox_inches="tight", **kw)
        os.replace(tmp, final)
        out[ext] = str(final)

    _atomic("pdf")
    _atomic("svg")
    if raster:
        _atomic("tiff", dpi=dpi, pil_kwargs={"compression": "tiff_lzw"})
        _atomic("png", dpi=dpi)
    return out
