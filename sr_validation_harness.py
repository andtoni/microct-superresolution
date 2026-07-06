#!/usr/bin/env python3
"""
sr_validation_harness.py
========================
Foundational tooling for the non-circular validation of an open super-resolution
(SR) micro-CT pipeline for soft, low-contrast polymeric scaffolds.

Covers playbook tasks:
  - P1.2  Degradation model (HR -> simulated-LR), registration-free / perfectly paired
  - P2.1  Interpolation baselines (trilinear / bicubic-spline)
  - P2.2  Full-reference image-fidelity metrics (PSNR, SSIM, multiscale-SSIM)
  - (P2.3 FRC resolution: basic two-image FRC included; refine next iteration)

Design intent
-------------
* Regime (b): generate a *perfectly co-registered* (HR, simulated-LR) pair by applying a
  realistic forward model to the real HR volume:  PSF blur  ->  area-average downsample
  ->  matched noise.  This sidesteps registration entirely and gives exact ground truth
  for image-fidelity metrics and (later) clean hallucination maps.
* The simulated-LR MUST be tuned to resemble the *real* LR (PSF sigma + noise sigma).
  Use `compare_to_real_lr()` to match histogram + noise before trusting any metric.
* SR slots in later: once the open model (3D EDSR/RCAN) is trained, point `--sr` at the
  SR volume and the same metrics are computed SR-vs-HR alongside the interpolation floor.

Usage (real data, current stage = before SR is trained)
-------------------------------------------------------
    uv run python sr_validation_harness.py \
        --hr  /path/to/HR_20X_tiff_stack \
        --factor 2.556 --psf-sigma 1.0 --noise-sigma auto \
        --roi 0 512 0 512 0 512 \
        --out ./val_out

This writes: simulated-LR stack, the two interpolation baselines, an interp-vs-HR
metrics CSV, and a mid-slice comparison PNG.  When you have a trained SR volume:

    uv run python sr_validation_harness.py --hr ... --sr /path/to/SR_stack --out ./val_out

Author: drafted with Claude for A. Tonelli, June 2026.  MIT-style, release with the paper.
"""

from __future__ import annotations
import argparse, csv, glob, os, sys
import numpy as np
from scipy import ndimage as ndi
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

# ----------------------------------------------------------------------------- I/O
def load_stack(path, max_slices=None):
    """Load a TIFF stack: either a multipage .tif file or a folder of slice .tif(f)s.
    If max_slices is set, load only that many central slices (keeps huge real-LR volumes
    out of memory when we only need representative noise/contrast stats)."""
    import tifffile
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.tif")) +
                       glob.glob(os.path.join(path, "*.tiff")))
        if not files:
            raise FileNotFoundError(f"No .tif/.tiff in {path}")
        if max_slices and len(files) > max_slices:
            c = len(files) // 2
            h = max_slices // 2
            files = files[max(0, c - h): c - h + max_slices]
        vol = np.stack([tifffile.imread(f) for f in files], axis=0)
    else:
        if max_slices:
            with tifffile.TiffFile(path) as tf:
                n = len(tf.pages)
                c = n // 2; h = max_slices // 2
                keys = list(range(max(0, c - h), min(n, c - h + max_slices)))
            vol = tifffile.imread(path, key=keys)
        else:
            vol = tifffile.imread(path)
    return vol.astype(np.float32)

def save_stack(vol, path):
    import tifffile
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tifffile.imwrite(path, vol.astype(np.float32))

def normalize(vol, lo=None, hi=None):
    """Scale to [0,1] using given or 0.5/99.5 percentile bounds (robust to outliers)."""
    if lo is None: lo = np.percentile(vol, 0.5)
    if hi is None: hi = np.percentile(vol, 99.5)
    out = (vol - lo) / max(hi - lo, 1e-8)
    return np.clip(out, 0, 1).astype(np.float32), (lo, hi)

# --------------------------------------------------------------------- degradation
def estimate_noise_sigma(vol, flat_frac=0.1):
    """Crude noise estimate: std within the most-uniform sub-region (a 'flat' patch).
    Replace with a measurement from a true background ROI in the real LR scan."""
    # use a robust high-pass: vol - median-filtered, std of the quietest decile of slices
    hp = vol - ndi.median_filter(vol, size=(1, 3, 3))
    return float(np.std(hp))

def degrade_hr_to_lr(hr, factor=2.556, psf_sigma_lr=1.0, noise_sigma=0.0,
                     in_plane_only=False, seed=0):
    """
    Forward model: HR -> simulated LR.
      1. Gaussian PSF blur (the LR system's resolution loss), sigma given in *LR-voxel*
         units and converted to HR voxels (sigma_hr = psf_sigma_lr * factor).
      2. Area-average downsample by `factor` (mimics detector binning / larger voxel).
      3. Additive Gaussian noise (sigma matched to the real LR background).

    Parameters
    ----------
    hr : (Z,Y,X) float array, ideally normalized to [0,1]
    factor : LR_voxel / HR_voxel  (e.g. 1.3852 / 0.5473 ~= 2.531; use your measured ratio)
    psf_sigma_lr : Gaussian sigma in LR voxels. Start ~0.8-1.2, then TUNE to match real LR FRC.
    noise_sigma : std of additive Gaussian noise (in normalized units). Match to real LR.
    in_plane_only : if True, do not blur/downsample along Z (use if LR/HR differ only in-plane)
    """
    sigma_hr = psf_sigma_lr * factor
    if in_plane_only:
        sigma_vec = (0.0, sigma_hr, sigma_hr)
    else:
        sigma_vec = (sigma_hr, sigma_hr, sigma_hr)
    blurred = ndi.gaussian_filter(hr, sigma=sigma_vec)

    if in_plane_only:
        out_shape = (hr.shape[0],
                     int(round(hr.shape[1] / factor)),
                     int(round(hr.shape[2] / factor)))
    else:
        out_shape = tuple(int(round(s / factor)) for s in hr.shape)
    lr = resize(blurred, out_shape, order=1, anti_aliasing=False, preserve_range=True)

    if noise_sigma and noise_sigma > 0:
        rng = np.random.default_rng(seed)
        lr = lr + rng.normal(0.0, noise_sigma, size=lr.shape)
    return lr.astype(np.float32)

def interpolate_to_hr(lr, hr_shape, order=3):
    """Upsample LR back onto the HR grid. order=1 trilinear, order=3 cubic-spline (~bicubic)."""
    return resize(lr, hr_shape, order=order, anti_aliasing=False,
                  preserve_range=True).astype(np.float32)

def compare_to_real_lr(sim_lr, real_lr, out=None):
    """
    Tune the degradation model: compare simulated-LR to a REAL-LR crop.
    Reports intensity-histogram distance (Wasserstein) and a noise-sigma estimate for each.
    Both inputs should be normalized to [0,1] on the SAME intensity bounds.
    Shapes need NOT match (we compare distributions/noise, not pixels).

    Read-out -> action:
      * sim noise < real noise  -> raise --noise-sigma ;  sim noise > real -> lower it
      * sim sharper than real    -> raise --psf-sigma   ;  sim blurrier    -> lower it
    Iterate psf-sigma/noise-sigma until Wasserstein is small and noise sigmas match.
    """
    from scipy.stats import wasserstein_distance
    wd = float(wasserstein_distance(sim_lr.ravel(), real_lr.ravel()))
    ns_sim = estimate_noise_sigma(sim_lr if sim_lr.ndim == 3 else sim_lr[None])
    ns_real = estimate_noise_sigma(real_lr if real_lr.ndim == 3 else real_lr[None])
    print(f"[match] histogram Wasserstein(sim,real) = {wd:.4f}   (lower = better match)")
    print(f"[match] noise sigma  sim={ns_sim:.4g}   real={ns_real:.4g}")
    if ns_sim < ns_real * 0.9:
        print("        -> raise --noise-sigma toward the real value")
    elif ns_sim > ns_real * 1.1:
        print("        -> lower --noise-sigma")
    else:
        print("        -> noise levels match well")
    if out:
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 4))
            plt.hist(real_lr.ravel(), bins=128, density=True, alpha=0.5, label="real LR")
            plt.hist(sim_lr.ravel(), bins=128, density=True, alpha=0.5, label="simulated LR")
            plt.xlabel("normalized intensity"); plt.ylabel("density"); plt.legend()
            plt.title(f"sim vs real LR  (Wasserstein={wd:.3f})"); plt.tight_layout()
            plt.savefig(out, dpi=130); plt.close()
            print(f"[match] histogram overlay -> {out}")
        except Exception as e:
            print(f"[match] plot skipped: {e}")
    return dict(wasserstein=wd, noise_sim=ns_sim, noise_real=ns_real)

# ------------------------------------------------------------------------- metrics
def psnr(ref, test, data_range=1.0):
    return float(sk_psnr(ref, test, data_range=data_range))

def ssim3d(ref, test, data_range=1.0):
    """Volumetric SSIM (skimage handles N-D with an isotropic window)."""
    return float(sk_ssim(ref, test, data_range=data_range))

def ms_ssim3d(ref, test, data_range=1.0, n_scales=4,
              weights=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333)):
    """
    Multiscale SSIM proxy: weighted mean of SSIM across Gaussian-pyramid scales.
    NOTE: this is a pragmatic multiscale-SSIM average, NOT the exact Wang et al.
    contrast/structure formulation. For the final paper figure use `pytorch-msssim`
    (canonical MS-SSIM). This proxy is fine for relative SR-vs-interpolation comparison.
    """
    w = np.array(weights[:n_scales], dtype=float); w /= w.sum()
    a, b = ref.copy(), test.copy()
    vals = []
    for i in range(n_scales):
        # smallest spatial dim must stay >= ~7 for the SSIM window
        if min(a.shape) < 8:
            w = w[:len(vals)]; w /= w.sum(); break
        vals.append(sk_ssim(a, b, data_range=data_range))
        if i < n_scales - 1:
            a = ndi.zoom(a, 0.5, order=1)
            b = ndi.zoom(b, 0.5, order=1)
    vals = np.array(vals)
    return float(np.sum(w[:len(vals)] * vals))

def frc_2d(img1, img2, threshold="half-bit"):
    """
    Fourier Ring Correlation between two 2D images of the SAME scene (two independent
    noise realizations, e.g. odd/even split). Returns (freqs, frc, resolution_pix).
    resolution_pix is in pixels (multiply by voxel size for microns).
    For single-image resolution use `single_image_frc_split`.
    """
    img1 = img1 - img1.mean(); img2 = img2 - img2.mean()
    F1 = np.fft.fftshift(np.fft.fft2(img1))
    F2 = np.fft.fftshift(np.fft.fft2(img2))
    ny, nx = img1.shape
    cy, cx = ny // 2, nx // 2
    Y, X = np.ogrid[:ny, :nx]
    r = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2).astype(int)
    rmax = min(cy, cx)
    num = np.zeros(rmax); d1 = np.zeros(rmax); d2 = np.zeros(rmax); npx = np.zeros(rmax)
    for ri in range(rmax):
        m = r == ri
        num[ri] = np.real(np.sum(F1[m] * np.conj(F2[m])))
        d1[ri] = np.sum(np.abs(F1[m]) ** 2)
        d2[ri] = np.sum(np.abs(F2[m]) ** 2)
        npx[ri] = np.count_nonzero(m)
    frc = num / np.sqrt(d1 * d2 + 1e-12)
    freqs = np.arange(rmax) / (2.0 * rmax)  # cycles/pixel, 0..0.5
    if threshold == "half-bit":
        thr = (0.2071 + 1.9567 / np.sqrt(npx)) / (1.2071 + 0.9567 / np.sqrt(npx))
    else:  # fixed 1/7 threshold
        thr = np.full(rmax, 1.0 / 7.0)
    # ignore the DC bin (ring 0, ==0 after mean subtraction) and lightly smooth the
    # curve so a single noisy ring dip doesn't register as the resolution crossing.
    frc_s = ndi.uniform_filter1d(frc, size=3)
    valid = np.arange(rmax) >= 1
    cross = np.where((frc_s < thr) & valid)[0]
    res_pix = (1.0 / (2.0 * freqs[cross[0]])) if len(cross) and freqs[cross[0]] > 0 else np.nan
    return freqs, frc, thr, float(res_pix)

def single_image_frc_split(img):
    """Checkerboard split of one image into two sub-images for single-image FRC (Koho-style)."""
    a = img[0::2, 0::2]; b = img[1::2, 1::2]
    n = (min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1]))
    return a[:n[0], :n[1]], b[:n[0], :n[1]]

# ------------------------------------------------------------------- phantom (test)
def make_porous_phantom(shape=(48, 160, 160), feature_scale=5.0, porosity=0.55, seed=0):
    """Grayscale bicontinuous porous phantom (thresholded Gaussian random field), for testing."""
    rng = np.random.default_rng(seed)
    field = ndi.gaussian_filter(rng.normal(size=shape), sigma=feature_scale)
    field = (field - field.min()) / (np.ptp(field) + 1e-8)
    # soft 'solid' phase with smooth edges, mimicking grayscale uCT of a porous polymer
    thr = np.quantile(field, porosity)
    solid = ndi.gaussian_filter((field > thr).astype(np.float32), sigma=0.8)
    return solid.astype(np.float32)

# ----------------------------------------------------------------------- pipeline
def run_metrics(hr, lr, dr=1.0):
    """Compute interpolation baselines and metrics vs HR. Returns dict of rows."""
    interp_lin = interpolate_to_hr(lr, hr.shape, order=1)
    interp_cub = interpolate_to_hr(lr, hr.shape, order=3)
    rows = {}
    for name, vol in [("trilinear", interp_lin), ("cubicspline", interp_cub)]:
        rows[name] = dict(
            method=name,
            PSNR=psnr(hr, vol, dr),
            SSIM=ssim3d(hr, vol, dr),
            MS_SSIM=ms_ssim3d(hr, vol, dr),
        )
    return rows, dict(trilinear=interp_lin, cubicspline=interp_cub)

def main():
    ap = argparse.ArgumentParser(description="SR validation harness (degradation + metrics).")
    ap.add_argument("--hr", help="HR tiff stack (file or folder). If omitted, runs --demo.")
    ap.add_argument("--sr", help="Optional SR tiff stack (file or folder) to score vs HR.")
    ap.add_argument("--real-lr", help="Optional real-LR stack to tune the degradation model against.")
    ap.add_argument("--real-lr-maxslices", type=int, default=120,
                    help="Load only this many central slices of the real-LR (memory).")
    ap.add_argument("--factor", type=float, default=2.556, help="LR_voxel/HR_voxel ratio.")
    ap.add_argument("--psf-sigma", type=float, default=1.0, help="PSF sigma in LR voxels.")
    ap.add_argument("--noise-sigma", default="0.0",
                    help="Additive noise sigma in normalized units, or 'auto' to estimate.")
    ap.add_argument("--in-plane-only", action="store_true")
    ap.add_argument("--roi", nargs=6, type=int, default=None,
                    metavar=("z0", "z1", "y0", "y1", "x0", "x1"),
                    help="Optional ROI to keep memory sane on full stacks.")
    ap.add_argument("--out", default="./val_out")
    ap.add_argument("--demo", action="store_true", help="Run on a synthetic phantom.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.demo or not args.hr:
        print("[demo] generating synthetic porous phantom as HR ground truth ...")
        hr_raw = make_porous_phantom()
    else:
        print(f"[load] HR <- {args.hr}")
        hr_raw = load_stack(args.hr)
        if args.roi:
            z0, z1, y0, y1, x0, x1 = args.roi
            hr_raw = hr_raw[z0:z1, y0:y1, x0:x1]

    hr, (lo, hi) = normalize(hr_raw)
    print(f"[norm] HR shape={hr.shape} range=({lo:.3g},{hi:.3g}) -> [0,1]")

    ns = args.noise_sigma
    noise_sigma = estimate_noise_sigma(hr) if str(ns).lower() == "auto" else float(ns)
    print(f"[degrade] factor={args.factor} psf_sigma_lr={args.psf_sigma} noise_sigma={noise_sigma:.4g}")
    lr = degrade_hr_to_lr(hr, factor=args.factor, psf_sigma_lr=args.psf_sigma,
                          noise_sigma=noise_sigma, in_plane_only=args.in_plane_only)
    print(f"[degrade] simulated-LR shape={lr.shape}")
    save_stack(lr, os.path.join(args.out, "simulated_LR.tif"))

    if args.real_lr:
        print(f"[load] real-LR <- {args.real_lr} (central {args.real_lr_maxslices} slices)")
        real_lr = load_stack(args.real_lr, max_slices=args.real_lr_maxslices)
        real_lr, _ = normalize(real_lr)   # normalize on its OWN percentiles, so we compare
                                          # distribution shape + relative noise, not absolute scale
        compare_to_real_lr(lr, real_lr, out=os.path.join(args.out, "sim_vs_real_LR_hist.png"))

    rows, interps = run_metrics(hr, lr, dr=1.0)
    save_stack(interps["cubicspline"], os.path.join(args.out, "interp_cubicspline_to_HR.tif"))

    if args.sr:
        print(f"[load] SR <- {args.sr}")
        sr = load_stack(args.sr)
        if args.roi:
            z0, z1, y0, y1, x0, x1 = args.roi
            sr = sr[z0:z1, y0:y1, x0:x1]
        sr, _ = normalize(sr, lo, hi)
        if sr.shape != hr.shape:
            sr = resize(sr, hr.shape, order=1, anti_aliasing=False, preserve_range=True)
        rows["SR"] = dict(method="SR", PSNR=psnr(hr, sr, 1.0),
                          SSIM=ssim3d(hr, sr, 1.0), MS_SSIM=ms_ssim3d(hr, sr, 1.0))

    # write CSV
    csv_path = os.path.join(args.out, "image_fidelity_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["method", "PSNR", "SSIM", "MS_SSIM"])
        wtr.writeheader()
        for k in rows: wtr.writerow(rows[k])
    print(f"[out] metrics -> {csv_path}")
    for k, v in rows.items():
        print(f"      {v['method']:>12}:  PSNR={v['PSNR']:6.2f} dB   SSIM={v['SSIM']:.4f}   MS-SSIM={v['MS_SSIM']:.4f}")

    # FRC demo on the mid-slice (single-image split) for HR vs interpolation
    mid = hr.shape[0] // 2
    a, b = single_image_frc_split(hr[mid])
    _, _, _, res_hr = frc_2d(a, b)
    a, b = single_image_frc_split(interps["cubicspline"][mid])
    _, _, _, res_in = frc_2d(a, b)
    def _fmt(x): return f"{x:.2f} px" if np.isfinite(x) else "no crossing (needs real noisy data)"
    print(f"[frc]  HR self-resolution ~{_fmt(res_hr)}   |  cubic-interp ~{_fmt(res_in)}  (smaller=finer)")

    # comparison figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 4, figsize=(14, 4))
        lrshow = resize(lr, hr.shape, order=0, anti_aliasing=False, preserve_range=True)
        for a_, im, t in zip(ax,
                             [lrshow[mid], interps["cubicspline"][mid], hr[mid],
                              np.abs(interps["cubicspline"][mid] - hr[mid])],
                             ["sim-LR (nn-up)", "cubic interp", "HR (truth)", "|interp - HR|"]):
            a_.imshow(im, cmap="gray"); a_.set_title(t); a_.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "comparison_midslice.png"), dpi=130)
        print(f"[out] figure -> {os.path.join(args.out, 'comparison_midslice.png')}")
    except Exception as e:
        print(f"[warn] figure skipped: {e}")

    print("[done]")

if __name__ == "__main__":
    main()
