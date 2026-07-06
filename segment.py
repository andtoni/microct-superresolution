#!/usr/bin/env python3
"""
segment.py -- segmentation for the scaffx toolbox (Phase 3.0 foundation).

The acquisition (50 kVp, unfiltered soft spectrum, low-Z polymer) produces three
reconstructed-domain problems that have defeated naive thresholding for years:
  (a) beam-hardening CUPPING (smooth low-frequency intensity field),
  (b) intrinsically low polymer<->air contrast,
  (c) inline Fresnel PHASE CONTRAST -> a bright "shell" rim with a dark, hollow-looking
      core on solid fibres/struts.

This module ships the classical, artifact-aware default and bootstraps training labels:

  decup()            -- de-cupping by large-scale background flattening (divide/subtract);
                        removes the low-frequency field while preserving fibre-scale detail.
  fill_shells()      -- multi-axis 2D hole-fill that turns phase-contrast rims into filled
                        solids (the rim is a precise boundary cue); fixes the hollow core.
  segment_classical()-- decup -> threshold -> rim-fill -> clean.  Default `--method classical`,
                        identical for HR and SR (S6 requirement).
  bootstrap_labels() -- filled-fibre labels + an AMBIGUITY map (rim band / low-gradient
                        interior) flagging voxels to refine with Ilastik scribbles or
                        TomoSAM prompts before training.  Spans every architecture.
  segment_ml()       -- graceful nnU-Net / MONAI inference hook for once labels exist.

QC overlays (de-cupping before/after, boundary on slices) are written for every run so
each architecture's segmentation is visually auditable.

CLI:
  uv run python segment.py --vol <stack> --out <dir>                # classical default
  uv run python segment.py --vol <stack> --out <dir> --bootstrap-labels
  uv run python segment.py --vol <stack> --out <dir> --method ml --model <nnunet_dir>
"""
from __future__ import annotations
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

warnings.filterwarnings("ignore")

import morphometry as mp                       # load_stack, normalize, save_volume helpers
try:
    import scaffold_viz as viz
    import provenance as prov
    _HAVE_FACTORY = True
except Exception:
    _HAVE_FACTORY = False

try:
    from skimage.filters import threshold_otsu, threshold_li
    from skimage.morphology import remove_small_objects, remove_small_holes
except Exception:
    threshold_otsu = threshold_li = None


# ----------------------------------------------------------------- de-cupping
def decup(vol, sigma_frac=0.25, mode="divide", max_bg_dim=128):
    """Flatten the low-frequency (beam-hardening) intensity field.

    The background is estimated on a DOWNSAMPLED copy so a large Gaussian sigma stays cheap
    on full-size (~1000^3) volumes, then upsampled back. Divide (the physics of beam
    hardening) or subtract. Cupping is low-frequency, so downsampling preserves it."""
    v = vol.astype(np.float32)
    factors = [max(1, int(np.ceil(s / max_bg_dim))) for s in v.shape]
    small = v[::factors[0], ::factors[1], ::factors[2]]
    sig = max(1.0, float(sigma_frac) * min(small.shape))
    bg_small = ndi.gaussian_filter(small, sigma=sig)
    bg = ndi.zoom(bg_small, [v.shape[i] / bg_small.shape[i] for i in range(3)], order=1)
    bg = bg[tuple(slice(0, v.shape[i]) for i in range(3))]      # zoom can be off by a voxel
    if bg.shape != v.shape:
        bg = np.pad(bg, [(0, v.shape[i] - bg.shape[i]) for i in range(3)], mode="edge")
    if mode == "divide":
        bg = np.clip(bg, 1e-6, None)
        out = v / bg * float(np.median(bg))
    else:                                       # subtract
        out = v - bg + float(np.median(bg))
    return out


# ------------------------------------------------------- phase-contrast rim fill
def fill_shells(binary):
    """Turn bright-rim/hollow-core shells into filled solids by filling holes in each
    orthogonal 2D cross-section (catches cores enclosed in any fibre orientation)."""
    filled = binary.copy()
    for ax in range(3):
        moved = np.moveaxis(filled, ax, 0)
        for i in range(moved.shape[0]):
            moved[i] = ndi.binary_fill_holes(moved[i])
        filled = np.moveaxis(moved, 0, ax)
    return filled


def _global_threshold(v, method, level):
    if method == "fixed":
        return float(level if level is not None else 0.5)
    if method == "li" and threshold_li is not None:
        return float(threshold_li(v))
    if threshold_otsu is not None:
        return float(threshold_otsu(v))
    return float(v.mean())


def _sauvola_3d(v, window=25, k=0.2, R=0.5):
    """Per-voxel Sauvola adaptive threshold SURFACE in 3D (separable box filters, fast).

    T(x) = m(x)*(1 + k*(s(x)/R - 1)); m,s = local mean/std in a `window` cube, R = dynamic
    range (0.5 for a [0,1] volume). Adapts to local contrast so faint phase-contrast fibre
    cores beside bright rims are still split from void -- the global-Otsu failure mode here.
    """
    v = v.astype(np.float32)
    w = int(window)
    mean = ndi.uniform_filter(v, w)
    mean2 = ndi.uniform_filter(v * v, w)
    std = np.sqrt(np.clip(mean2 - mean * mean, 0.0, None))
    return mean * (1.0 + k * (std / float(R) - 1.0))


def _ridge_response(v, scales=(1.0, 2.0, 3.0)):
    """Multiscale Sato tubular ridge filter (bright fibres) -> [0,1] response, or None.

    Responds to tube-like geometry across `scales` (~fibre radii in voxels), which the
    phase-contrast bright rim makes salient; thresholding the response then rim-filling
    recovers solid fibres. Slow on full ~1000^3 volumes -> intended for sub-volumes / the
    eval comparison; use `local` for full-volume bootstrap.
    """
    try:
        from skimage.filters import sato
    except Exception:
        return None
    r = sato(v.astype(np.float32), sigmas=tuple(scales), black_ridges=False)
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    mx = float(r.max())
    return (r / mx) if mx > 0 else r


# --------------------------------------------------------------- mode dispatch
# Each mode = (apply de-cup?, how foreground is decided). Post-processing (rim-fill +
# cleanup) is IDENTICAL across modes, so a comparison isolates the threshold/enhancement
# step. 'otsu' (no de-cup) is the naive strawman; otsu_decup -> local -> ridge is the ladder.
MODES = ("otsu", "otsu_decup", "local", "ridge", "fixed")


def _foreground(vol, mode, decup_sigma, decup_mode, level,
                local_window, sauvola_k, ridge_scales):
    """Return (scaffold_bool_pre_postproc, threshold_value_for_report, vflat[0,1])."""
    use_decup = mode != "otsu"                       # 'otsu' = naive no-de-cup strawman
    if use_decup:
        vflat = mp.normalize(decup(vol, decup_sigma, decup_mode))
    else:
        vflat = mp.normalize(vol.astype(np.float32))

    if mode == "local":
        tsurf = _sauvola_3d(vflat, local_window, sauvola_k)
        scaffold = vflat >= tsurf
        thr = float(np.median(tsurf))
    elif mode == "ridge":
        resp = _ridge_response(vflat, ridge_scales)
        if resp is None:                             # skimage missing -> fall back to otsu
            thr = _global_threshold(vflat, "otsu", None); scaffold = vflat >= thr
        else:
            thr = _global_threshold(resp, "otsu", None); scaffold = resp >= thr
    else:                                            # otsu / otsu_decup / fixed
        meth = "fixed" if mode == "fixed" else "otsu"
        thr = _global_threshold(vflat, meth, level)
        scaffold = vflat >= thr
    return scaffold, float(thr), vflat


# ----------------------------------------------------------- classical pipeline
def segment_classical(vol, mode="otsu_decup", decup_sigma=0.25, decup_mode="divide",
                      level=None, fill=True, min_size=27, close_holes=True,
                      local_window=25, sauvola_k=0.2, ridge_scales=(1.0, 2.0, 3.0)):
    """De-cup (mode-dependent) -> threshold/enhance (mode) -> rim-fill -> clean.

    mode: 'otsu' (naive, no de-cup) | 'otsu_decup' (de-cup+Otsu) | 'local' (de-cup+Sauvola)
          | 'ridge' (de-cup+Sato vesselness) | 'fixed' (de-cup+--level). Post-processing is
    identical across modes; identical for HR and SR (S6 requirement).
    """
    scaffold, thr, vflat = _foreground(vol, mode, decup_sigma, decup_mode, level,
                                       local_window, sauvola_k, ridge_scales)
    if fill:
        scaffold = fill_shells(scaffold)
    if close_holes:
        try:
            scaffold = remove_small_holes(scaffold, area_threshold=max(8, min_size))
        except Exception:
            pass
    if min_size > 0:
        try:
            scaffold = remove_small_objects(scaffold, min_size)
        except Exception:
            pass
    void = ~scaffold
    return void, scaffold, thr, vflat


# --------------------------------------------------------- label bootstrapping
def bootstrap_labels(vol, mode="local", decup_sigma=0.25, decup_mode="divide",
                     level=None, min_size=27, rim_band=1, grad_lo_pct=40, fill=True,
                     local_window=25, sauvola_k=0.2, ridge_scales=(1.0, 2.0, 3.0)):
    """Filled-fibre labels + an ambiguity map for human/SAM refinement before training.

    Ambiguity = the rim band (dilated boundary, where the phase fringe sits) UNION the
    low-gradient solid interior (where filling is an inference, not a measurement).  These
    are exactly the voxels worth a scribble/prompt; the rest are confident labels.

    Default mode='local' (Sauvola adaptive) -- the practical best on the full volume; global
    Otsu (mode='otsu_decup') under-separates faint phase-contrast fibre cores from void.
    fill=True (rim-fill) recovers hollow fibre cores (electrospun); set fill=False for
    solid-wall/porogen architectures so genuine pores are NOT filled."""
    void, scaffold, thr, vflat = segment_classical(
        vol, mode=mode, decup_sigma=decup_sigma, decup_mode=decup_mode, level=level,
        fill=fill, min_size=min_size, close_holes=True,
        local_window=local_window, sauvola_k=sauvola_k, ridge_scales=ridge_scales)

    boundary = scaffold ^ ndi.binary_erosion(scaffold)
    rim = ndi.binary_dilation(boundary, iterations=int(rim_band))

    grad = ndi.gaussian_gradient_magnitude(vflat.astype(np.float32), sigma=1.0)
    interior = ndi.binary_erosion(scaffold, iterations=2)
    if interior.any():
        lo = np.percentile(grad[interior], grad_lo_pct)
        low_grad_interior = interior & (grad <= lo)
    else:
        low_grad_interior = np.zeros_like(scaffold)

    ambiguity = rim | low_grad_interior
    frac = float(ambiguity.mean())
    labels = scaffold.astype(np.uint8)          # 1 = solid, 0 = void
    return labels, ambiguity, vflat, {"threshold": thr, "ambiguous_fraction": frac}


# ------------------------------------------------------------- nnU-Net hook (ML)
def segment_ml(vol, model_dir, overlap=0.25):
    """Load the MONAI 3D U-Net trained by train_seg.py and run sliding-window inference.
    Returns (mask, note); falls back gracefully (note) until a model exists."""
    try:
        import torch
        from monai.networks.nets import UNet
        from monai.inferers import sliding_window_inference
    except Exception:
        return None, "torch/monai not available -- uv add monai (uses your uv torch cu128)."
    mdl = Path(model_dir) if model_dir else None
    ckpt = None
    if mdl:
        if mdl.is_file():
            ckpt = mdl
        elif (mdl / "best.pt").exists():
            ckpt = mdl / "best.pt"
    if ckpt is None or not ckpt.exists():
        return None, ("no trained model yet -- bootstrap labels (--bootstrap-labels), refine the "
                      "ambiguity map (Ilastik/TomoSAM), train with train_seg.py, then pass "
                      "--model <seg_model dir or best.pt>.")
    try:
        st = torch.load(str(ckpt), map_location="cpu")
        in_ch = int(st.get("in_ch", 1)); patch = int(st.get("patch", 96)); use_grad = bool(st.get("grad", False))
        net = UNet(spatial_dims=3, in_channels=in_ch, out_channels=2,
                   channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2)
        net.load_state_dict(st["model"]); net.eval()
        dev = "cuda" if torch.cuda.is_available() else "cpu"; net.to(dev)
        v = mp.normalize(vol).astype(np.float32)
        x = v[None, None]
        if use_grad:
            g = ndi.gaussian_gradient_magnitude(v, 1.0).astype(np.float32)
            x = np.concatenate([v[None, None], g[None, None]], 1)
        xt = torch.from_numpy(x).float().to(dev)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(dev == "cuda")):
                pr = sliding_window_inference(xt, (patch, patch, patch), 1, net, overlap=overlap)
        seg = torch.argmax(pr, 1)[0].cpu().numpy().astype(bool)
        return seg, f"MONAI 3D U-Net inference OK (model {ckpt.name}, in_ch {in_ch}, patch {patch})"
    except Exception as e:
        return None, f"model present but inference failed: {e}"


# --------------------------------------------------------------------- QC + IO
def _save_volume(arr, path):
    import tifffile
    tifffile.imwrite(str(path), arr.astype(np.uint8) * (255 if arr.dtype == bool else 1),
                     compression="zlib")


def _qc(vol, vflat, scaffold, ambiguity, out_dir, stem):
    if not _HAVE_FACTORY:
        return
    try:
        import matplotlib.pyplot as plt
        viz.set_pub_style()
        z = vol.shape[0] // 2
        n = 3 if ambiguity is not None else 2
        fig, ax = plt.subplots(1, n + 1, figsize=(2.2 * (n + 1), 2.4))
        ax[0].imshow(vol[z], cmap="gray"); ax[0].set_title("raw", fontsize=7)
        ax[1].imshow(vflat[z], cmap="gray"); ax[1].set_title("de-cupped", fontsize=7)
        ov = np.dstack([mp.normalize(vflat[z])] * 3)
        edge = scaffold[z] ^ ndi.binary_erosion(scaffold[z])
        ov[edge] = [1, 0.3, 0]
        ax[2].imshow(ov); ax[2].set_title("boundary", fontsize=7)
        if ambiguity is not None:
            am = np.dstack([mp.normalize(vflat[z])] * 3)
            am[ambiguity[z]] = [0.1, 0.8, 1.0]
            ax[3].imshow(am); ax[3].set_title("ambiguity", fontsize=7)
        for a in ax:
            a.axis("off")
        viz.save_figure(fig, f"{stem}_qc", out_dir)
        plt.close(fig)
    except Exception as e:
        print(f"   [qc] skipped: {e}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="scaffx segmentation (classical default; ML hook)")
    ap.add_argument("--vol", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", default="classical", choices=["classical", "ml"])
    ap.add_argument("--mode", default="local", choices=list(MODES),
                    help="classical variant: otsu (naive, no de-cup) | otsu_decup | "
                         "local (Sauvola adaptive, default) | ridge (Sato vesselness) | fixed")
    ap.add_argument("--bootstrap-labels", action="store_true",
                    help="emit filled-fibre training labels + ambiguity map")
    ap.add_argument("--decup-sigma", type=float, default=0.25, help="bg sigma as fraction of min dim")
    ap.add_argument("--decup-mode", default="divide", choices=["divide", "subtract"])
    ap.add_argument("--local-window", type=int, default=25, help="Sauvola cube window (mode=local)")
    ap.add_argument("--sauvola-k", type=float, default=0.2, help="Sauvola k (mode=local)")
    ap.add_argument("--ridge-scales", default="1,2,3", help="comma scales in voxels (mode=ridge)")
    ap.add_argument("--level", type=float, default=None, help="threshold for mode=fixed")
    ap.add_argument("--no-fill", action="store_true", help="disable phase-contrast rim fill")
    ap.add_argument("--min-size", type=int, default=27)
    ap.add_argument("--model", default=None, help="trained nnU-Net dir for --method ml")
    args = ap.parse_args(argv)
    ridge_scales = tuple(float(s) for s in str(args.ridge_scales).split(","))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    vol = mp.load_stack(args.vol)
    stem = Path(args.vol.rstrip("/")).name or "volume"
    t0 = time.time()

    if args.method == "ml":
        seg, note = segment_ml(vol, args.model)
        print(f"[ml] {note}")
        if seg is None:
            print("[ml] falling back to classical."); args.method = "classical"
        else:
            _save_volume(seg, out / f"{stem}_mask_ml.tif")
            print(f"[ml] wrote {stem}_mask_ml.tif ({round(time.time()-t0,1)}s)")
            return 0

    if args.bootstrap_labels:
        labels, amb, vflat, info = bootstrap_labels(
            vol, mode=args.mode, decup_sigma=args.decup_sigma, decup_mode=args.decup_mode,
            level=args.level, min_size=args.min_size,
            local_window=args.local_window, sauvola_k=args.sauvola_k, ridge_scales=ridge_scales)
        _save_volume(labels.astype(bool), out / f"{stem}_labels.tif")
        _save_volume(amb, out / f"{stem}_ambiguity.tif")
        _qc(vol, vflat, labels.astype(bool), amb, out, stem)
        print(f"[bootstrap] {stem} (mode={args.mode}): solid frac {labels.mean():.3f}, "
              f"ambiguous {info['ambiguous_fraction']:.3f}, thr {info['threshold']:.3f} "
              f"({round(time.time()-t0,1)}s)")
        print(f"[bootstrap] refine {stem}_ambiguity.tif in Ilastik/TomoSAM, then train.")
    else:
        void, scaffold, thr, vflat = segment_classical(
            vol, mode=args.mode, decup_sigma=args.decup_sigma, decup_mode=args.decup_mode,
            level=args.level, fill=not args.no_fill, min_size=args.min_size,
            local_window=args.local_window, sauvola_k=args.sauvola_k, ridge_scales=ridge_scales)
        _save_volume(scaffold, out / f"{stem}_mask.tif")
        _qc(vol, vflat, scaffold, None, out, stem)
        print(f"[classical] {stem} (mode={args.mode}): solid frac {scaffold.mean():.3f}, porosity "
              f"{(~scaffold).mean():.3f}, thr {thr:.3f} ({round(time.time()-t0,1)}s)")

    if _HAVE_FACTORY:
        prov.stamp(out, params=vars(args), inputs=[args.vol])
    return 0


if __name__ == "__main__":
    sys.exit(main())
