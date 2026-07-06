#!/usr/bin/env python3
"""
morphometry.py -- Act II transport-morphometry engine.

Turns ONE volume (HR / SR / interp / DeepScout) into the transport-relevant
metric suite, run IDENTICALLY across sources so any differences reflect SR
fidelity rather than analysis choices (the S6 methodology guarantee).

Pipeline (all params explicit + logged):
  segment -> filled sample envelope -> SNOW2 network -> {porosity, PSD, throat
  sizes, coordination, permeability (OpenPNM Stokes), tortuosity (TauFactor2 or
  PoreSpy FD fallback)}.

Convention: scaffold is BRIGHT, pore space is DARK -> void = below threshold.

Outputs (standard contract):
  <out>/<source>_morphometry.xlsx (+ csv mirrors): scalar metrics, pore-size,
  throat-size, coordination distributions, and a meta sheet.
  <out>/manifest.json provenance stamp.

Critical: to compare HR vs SR vs interp vs DeepScout, call analyze_volume on each
with the SAME seg_method / seg_level / envelope settings.  Use seg_method='fixed'
on matched-normalised volumes for the strictest like-for-like segmentation.

CLI:
  uv run python morphometry.py --vol <stack_dir_or_tif> --voxel 0.9696 \
      --source HR --out <dir> [--mask <stack>] [--seg-method otsu|fixed|percentile]
      [--seg-level 0.5] [--max-size 400] [--axes xyz] [--no-perm] [--no-tort]
"""
from __future__ import annotations
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# --- factory + provenance (repo-local); degrade gracefully if run standalone ---
try:
    import scaffold_viz as viz
    import provenance as prov
    _HAVE_FACTORY = True
except Exception:
    _HAVE_FACTORY = False

AXIS_LABELS = {0: ("xmin", "xmax"), 1: ("ymin", "ymax"), 2: ("zmin", "zmax")}
MU_WATER = 1.0e-3          # Pa.s
M2_TO_MD = 1.0 / 9.869233e-16


# --------------------------------------------------------------------------- IO
def load_stack(path, max_slices=None):
    """Load a folder of TIFFs or a multipage TIFF as float32 (Z,Y,X)."""
    import tifffile
    p = Path(path)
    if p.is_dir():
        files = sorted([f for f in p.iterdir()
                        if f.suffix.lower() in (".tif", ".tiff")])
        if max_slices:
            files = files[:max_slices]
        if not files:
            raise FileNotFoundError(f"no TIFFs in {p}")
        arr = np.stack([tifffile.imread(str(f)) for f in files]).astype(np.float32)
    else:
        arr = tifffile.imread(str(p)).astype(np.float32)
        if max_slices:
            arr = arr[:max_slices]
    return arr


def resolve_volume(path, prefer=("hr", "high")):
    """Resolve an architecture path to ONE loadable HR volume.
    Handles three layouts: (a) a file -> return it; (b) a folder with an HR/hr/high
    slice subfolder -> return that subfolder; (c) a folder holding one multipage TIFF
    per modality (hr/lr/sr/...) -> return the HR-named file (so we don't stack HR+LR,
    which have different XY shapes)."""
    p = Path(path)
    if p.is_file():
        return str(p)
    for sub in ("HR", "hr", "high"):
        d = p / sub
        if d.is_dir() and (any(d.glob("*.tif")) or any(d.glob("*.tiff"))):
            return str(d)
    tifs = sorted(list(p.glob("*.tif")) + list(p.glob("*.tiff")))
    if not tifs:
        raise FileNotFoundError(f"no TIFFs in {p}")

    def _hrish(name):
        n = name.lower()
        return any(k in n for k in prefer) and not any(
            b in n for b in ("lr", "low", "sr", "super", "interp", "deepscout", "scout", "_ds"))

    hr = [f for f in tifs if _hrish(f.name)]
    if len(hr) > 5:
        return str(p)                       # a folder of HR slices -> load the folder
    if len(hr) >= 1:
        return str(hr[0])                   # one multipage HR volume among modalities
    if len(tifs) == 1:
        return str(tifs[0])
    if len(tifs) > 5:
        return str(p)                       # single-volume slice folder (no modality in names)
    raise ValueError(f"{p.name}: can't identify the HR volume among {[f.name for f in tifs]}; "
                     f"point at the HR file directly or put HR slices in an 'HR' subfolder")


def normalize(vol, lo=1.0, hi=99.0):
    """Percentile min-max to [0,1] (robust to outliers)."""
    a, b = np.percentile(vol, [lo, hi])
    if b <= a:
        b = a + 1e-6
    return np.clip((vol - a) / (b - a), 0, 1).astype(np.float32)


# ------------------------------------------------------------------- SEGMENTATION
def segment(vol, method="otsu", level=None):
    """Return boolean VOID mask (True = pore). Scaffold is bright -> void is dark.

    method 'otsu'       : per-volume Otsu threshold (level ignored)
    method 'fixed'      : void = vol < level   (level in same units as vol;
                          use on matched-normalised [0,1] volumes for strict
                          like-for-like segmentation across sources)
    method 'percentile' : threshold at the given intensity percentile (level=pct)
    Returns (void_bool, threshold_used).
    """
    from skimage.filters import threshold_otsu
    if method == "otsu":
        thr = float(threshold_otsu(vol))
    elif method == "fixed":
        if level is None:
            raise ValueError("seg method 'fixed' requires --seg-level")
        thr = float(level)
    elif method == "percentile":
        thr = float(np.percentile(vol, level if level is not None else 50.0))
    else:
        raise ValueError(f"unknown seg method {method!r}")
    void = vol < thr
    return void, thr


def sample_envelope(scaffold_bool):
    """Specimen silhouette via per-slice convex hull of the scaffold.

    Open (percolating) porosity means hole-filling cannot recover the silhouette,
    so we take the per-slice convex hull of the scaffold points -- this fills the
    space *between* fibres/struts and defines the bulk region in which porosity
    and transport are measured. Within it, void-in-envelope = envelope & ~scaffold.

    NOTE: convex hull suits compact specimens (cylinders, discs, fibre mats).
    For concave geometries (e.g. tubular grafts, where the lumen is not scaffold)
    pass an explicit --mask envelope instead.
    """
    from scipy import ndimage as ndi
    from skimage.morphology import convex_hull_image
    env = np.zeros_like(scaffold_bool, dtype=bool)
    for z in range(scaffold_bool.shape[0]):
        sl = scaffold_bool[z]
        if sl.sum() < 16:                        # too sparse this slice
            continue
        try:
            env[z] = convex_hull_image(sl)
        except Exception:
            env[z] = ndi.binary_fill_holes(sl)
    lbl, n = ndi.label(env)
    if n > 1:                                     # keep largest 3D component
        sizes = np.bincount(lbl.ravel())
        sizes[0] = 0
        env = lbl == sizes.argmax()
    return env


def porosity_metrics(void, env):
    """Total and accessible (boundary-connected) porosity within the envelope."""
    from scipy import ndimage as ndi
    void_in = void & env
    total = float(void_in.sum()) / float(max(env.sum(), 1))
    # accessible = void clusters touching the envelope's z-faces (flow boundaries)
    lbl, n = ndi.label(void_in)
    if n == 0:
        return total, 0.0
    face_labels = set(np.unique(lbl[0])) | set(np.unique(lbl[-1]))
    face_labels.discard(0)
    if not face_labels:
        return total, 0.0
    acc = np.isin(lbl, list(face_labels))
    accessible = float(acc.sum()) / float(max(env.sum(), 1))
    return total, accessible


# --------------------------------------------------------------------- NETWORK
def extract_network(void, voxel_m):
    """SNOW2 -> OpenPNM, trimmed to the connected cluster. Returns pn or None."""
    import porespy as ps
    import openpnm as op
    snow = ps.networks.snow2(void, voxel_size=voxel_m, boundary_width=3,
                             accuracy="standard")
    if snow.network["pore.coords"].shape[0] < 4:
        return None
    pn = op.io.network_from_porespy(snow.network)
    h = op.utils.check_network_health(pn)
    bad = []
    for key in ("disconnected_pores", "isolated_pores"):
        v = h[key]
        if v is not None and len(v):
            bad.append(np.asarray(v, dtype=int))
    if bad:
        op.topotools.trim(network=pn, pores=np.unique(np.concatenate(bad)))
    return pn


def network_stats(pn):
    """Distributions + scalar summaries. Diameters returned in micrometres."""
    import numpy as np
    pore_d = pn["pore.equivalent_diameter"] * 1e6      # m -> um
    thr_d = pn["throat.inscribed_diameter"] * 1e6
    Z = np.bincount(pn["throat.conns"].ravel(), minlength=pn.Np).astype(float)
    stats = {
        "n_pores": int(pn.Np),
        "n_throats": int(pn.Nt),
        "pore_diam_um_mean": float(np.mean(pore_d)),
        "pore_diam_um_median": float(np.median(pore_d)),
        "throat_diam_um_mean": float(np.mean(thr_d)),
        "throat_diam_um_median": float(np.median(thr_d)),
        "coordination_mean": float(np.mean(Z)),
        "coordination_median": float(np.median(Z)),
    }
    dists = {"pore_diameter_um": pore_d, "throat_diameter_um": thr_d,
             "coordination": Z}
    return stats, dists


def permeability(pn, im_shape, voxel_m, axes=(0, 1, 2)):
    """Absolute permeability per axis (m^2) via OpenPNM Stokes with direct
    Hagen-Poiseuille throat conductance g = A^2 / (8 pi mu L)."""
    import numpy as np
    import openpnm as op
    out = {}
    A = pn["throat.cross_sectional_area"]
    L = pn["throat.total_length"]
    L = np.where(L <= 0, voxel_m, L)
    g = (A ** 2) / (8.0 * np.pi * MU_WATER * L)
    for ax in axes:
        lab = AXIS_LABELS[ax]
        try:
            pin, pout = pn.pores(lab[0]), pn.pores(lab[1])
            if pin.size == 0 or pout.size == 0:
                out[ax] = np.nan
                continue
            ph = op.phase.Phase(network=pn)
            ph["throat.hydraulic_conductance"] = g
            sf = op.algorithms.StokesFlow(network=pn, phase=ph)
            sf.set_value_BC(pores=pin, values=1001.0)
            sf.set_value_BC(pores=pout, values=1.0)
            sf.run()
            Q = abs(float(sf.rate(pores=pin)[0]))
            c = pn["pore.coords"][:, ax]
            span = float(c.max() - c.min())
            face = np.prod([im_shape[i] for i in range(3) if i != ax]) * voxel_m ** 2
            out[ax] = Q * MU_WATER * span / (face * 1000.0)
        except Exception as e:
            print(f"   [perm] axis {ax} failed: {e}")
            out[ax] = np.nan
    return out


def tortuosity(void, axes=(0, 1, 2)):
    """Tortuosity per axis. TauFactor2 (GPU) if installed, else PoreSpy FD.
    Each axis is attempted independently: a specimen that does not span the
    bounding box in a given direction simply yields NaN for that axis."""
    import numpy as np
    res = {}
    engine = "none"
    # 1) TauFactor2 (preferred; uses GPU)
    try:
        import taufactor as tau
        import torch
        for ax in axes:
            try:
                vol = np.ascontiguousarray(np.moveaxis(void, ax, 0)).astype(np.float32)
                s = tau.Solver(torch.tensor(vol[None]))
                s.solve(verbose=False)
                eps = float(vol.mean())
                deff = float(s.D_eff) if hasattr(s, "D_eff") else float(s.tau)
                res[ax] = eps / deff if deff > 0 else np.nan
                engine = "taufactor2"
            except Exception:
                res[ax] = np.nan
        if engine == "taufactor2":
            res["_engine"] = engine
            return res
    except Exception:
        pass
    # 2) PoreSpy finite-difference fallback (CPU)
    try:
        import porespy as ps
        for ax in axes:
            try:
                vol = np.ascontiguousarray(np.moveaxis(void, ax, 0))
                r = ps.simulations.tortuosity_fd(vol, axis=0)
                res[ax] = float(getattr(r, "tortuosity", r))
                engine = "porespy_fd"
            except Exception:
                res[ax] = np.nan
    except Exception as e:
        print(f"   [tort] unavailable: {e}")
        for ax in axes:
            res.setdefault(ax, np.nan)
    res["_engine"] = engine
    return res


# ----------------------------------------------------------- EXTENDED METRICS
def surface_metrics(void, env, voxel_m):
    """Interfacial surface area (marching cubes) + specific surface area."""
    from skimage.measure import marching_cubes, mesh_surface_area
    out = {"surface_area_m2": np.nan, "ssa_per_m": np.nan, "ssa_per_um": np.nan}
    scaffold_in = env & ~void
    try:
        v, f, _, _ = marching_cubes(scaffold_in.astype(np.uint8), level=0.5)
        area_m2 = float(mesh_surface_area(v, f)) * voxel_m ** 2
        vol_m3 = float(env.sum()) * voxel_m ** 3
        ssa = area_m2 / vol_m3 if vol_m3 > 0 else np.nan
        out = {"surface_area_m2": area_m2, "ssa_per_m": float(ssa),
               "ssa_per_um": float(ssa * 1e-6)}
    except Exception as e:
        print(f"   [surface] failed: {e}")
    return out


def topology_metrics(void, env, voxel_m):
    """Euler characteristic of the void phase + density + component count."""
    from skimage.measure import euler_number
    out = {"euler_number": np.nan, "euler_density_per_mm3": np.nan,
           "n_void_components": np.nan}
    void_in = void & env
    try:
        chi = int(euler_number(void_in, connectivity=3))
        vol_mm3 = float(env.sum()) * (voxel_m * 1e3) ** 3
        try:
            import cc3d
            ncomp = int(cc3d.connected_components(void_in.astype(np.uint8),
                                                  connectivity=26).max())
        except Exception:
            from scipy import ndimage as ndi
            _, ncomp = ndi.label(void_in)
        out = {"euler_number": chi,
               "euler_density_per_mm3": float(chi / vol_mm3) if vol_mm3 > 0 else np.nan,
               "n_void_components": int(ncomp)}
    except Exception as e:
        print(f"   [topology] failed: {e}")
    return out


def fibre_orientation(scaffold, env, sigma=1.0, rho=3.0):
    """Structure-tensor anisotropy (FA) + principal orientation on the solid phase.
    Replaces Quanfima (Py2-only). Returns (scalars, distributions)."""
    out = {"anisotropy_fa_mean": np.nan, "anisotropy_fa_median": np.nan,
           "orient_theta_deg_mean": np.nan}
    dists = {}
    mask = scaffold & env
    if mask.sum() < 100:
        return out, dists
    try:
        import structure_tensor as st
        S = st.structure_tensor_3d(mask.astype("f4"), sigma=sigma, rho=rho)
        val, vec = st.eig_special_3d(S)
        sel = mask.ravel()
        l = val.reshape(3, -1)[:, sel] + 1e-12
        fa = np.sqrt(0.5 * ((l[0] - l[1]) ** 2 + (l[1] - l[2]) ** 2 +
                            (l[0] - l[2]) ** 2) / (l ** 2).sum(0))
        vz = np.abs(vec[0].ravel()[sel])              # |z| of fibre-direction vector
        theta = np.degrees(np.arccos(np.clip(vz, 0, 1)))
        out = {"anisotropy_fa_mean": float(np.nanmean(fa)),
               "anisotropy_fa_median": float(np.nanmedian(fa)),
               "orient_theta_deg_mean": float(np.nanmean(theta))}
        dists = {"anisotropy_fa": fa, "orient_theta_deg": theta}
    except Exception as e:
        print(f"   [orient] failed: {e}")
    return out, dists


def skeleton_metrics(scaffold, env, voxel_um):
    """skan skeleton: branch count, branch-length stats, junction density."""
    out = {"n_branches": np.nan, "branch_length_um_mean": np.nan,
           "branch_length_um_median": np.nan, "junction_density_per_mm3": np.nan}
    dists = {}
    mask = scaffold & env
    if mask.sum() < 100:
        return out, dists
    try:
        from skimage.morphology import skeletonize
        from skan import Skeleton, summarize
        sk = skeletonize(mask)
        if sk.sum() < 10:
            return out, dists
        S = Skeleton(sk, spacing=float(voxel_um))
        df = summarize(S, separator="_")
        bl = df["branch_distance"].to_numpy(dtype=float)
        njunc = int((df["branch_type"] == 2).sum())
        vol_mm3 = float(env.sum()) * (voxel_um * 1e-3) ** 3
        out = {"n_branches": int(len(df)),
               "branch_length_um_mean": float(np.mean(bl)),
               "branch_length_um_median": float(np.median(bl)),
               "junction_density_per_mm3": float(njunc / vol_mm3) if vol_mm3 > 0 else np.nan}
        dists = {"branch_length_um": bl}
    except Exception as e:
        print(f"   [skeleton] failed: {e}")
    return out, dists


def correlation_metrics(void, env, voxel_m):
    """Two-point correlation + chord-length distributions of the void phase."""
    out = {"corr_length_um": np.nan, "chord_length_um_mean": np.nan}
    dists = {}
    void_in = void & env
    try:
        import porespy as ps
        tpc = ps.metrics.two_point_correlation(void_in)
        d = np.asarray(tpc.distance) * voxel_m * 1e6
        p = np.asarray(tpc.probability)
        dists["tpc_distance_um"] = d
        dists["tpc_probability"] = p
        phi = float(void_in.sum()) / max(float(env.sum()), 1)
        target = phi + (p[0] - phi) / np.e
        below = np.where(p <= target)[0]
        out["corr_length_um"] = float(d[below[0]]) if below.size else np.nan
    except Exception as e:
        print(f"   [tpc] failed: {e}")
    try:
        import porespy as ps
        cr = ps.filters.apply_chords(void_in, axis=0)
        cld = ps.metrics.chord_length_distribution(cr, bins=25)
        L = np.asarray(cld.bin_centers) * voxel_m * 1e6
        pdf = np.asarray(cld.pdf)
        dists["chord_length_um"] = L
        dists["chord_pdf"] = pdf
        out["chord_length_um_mean"] = float(np.sum(L * pdf) / np.sum(pdf)) \
            if pdf.sum() > 0 else np.nan
    except Exception as e:
        print(f"   [chord] failed: {e}")
    return out, dists


# --------------------------------------------------------------------- DRIVER
def analyze_volume(vol, voxel_um, source, out_dir, mask=None,
                   seg_method="otsu", seg_level=None, axes=(0, 1, 2),
                   do_perm=True, do_tort=True, do_surface=True, do_topology=True,
                   do_orientation=True, do_skeleton=True, do_correlations=True,
                   with_pytrax=False, with_gudhi=False, with_puma=False,
                   with_lbm=False, with_bonej=False, with_diameterj=False,
                   write=True):
    """Run the full identical pipeline on one volume. Returns a metrics dict."""
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    voxel_m = float(voxel_um) * 1e-6

    void, thr = segment(vol, method=seg_method, level=seg_level)
    scaffold = ~void
    env = mask.astype(bool) if mask is not None else sample_envelope(scaffold)
    void_in = void & env
    total_phi, acc_phi = porosity_metrics(void, env)

    metrics = {
        "source": source, "voxel_um": float(voxel_um),
        "seg_method": seg_method, "seg_threshold": thr,
        "porosity_total": total_phi, "porosity_accessible": acc_phi,
    }
    dists = {}

    pn = extract_network(void_in, voxel_m)
    if pn is not None:
        nstats, ndist = network_stats(pn)
        metrics.update(nstats)
        dists.update(ndist)
        if do_perm:
            K = permeability(pn, vol.shape, voxel_m, axes=axes)
            for ax in axes:
                metrics[f"perm_{'xyz'[ax]}_m2"] = K.get(ax, np.nan)
            valid = [K[a] for a in axes if a in K and np.isfinite(K[a])]
            metrics["perm_mean_m2"] = float(np.mean(valid)) if valid else np.nan
            metrics["perm_mean_mD"] = metrics["perm_mean_m2"] * M2_TO_MD \
                if np.isfinite(metrics["perm_mean_m2"]) else np.nan
    else:
        print("   [net] too few pores; network metrics skipped")

    if do_tort:
        T = tortuosity(void_in, axes=axes)
        for ax in axes:
            metrics[f"tortuosity_{'xyz'[ax]}"] = T.get(ax, np.nan)
        metrics["tortuosity_engine"] = T.get("_engine", "none")

    if do_surface:
        metrics.update(surface_metrics(void, env, voxel_m))
    if do_topology:
        metrics.update(topology_metrics(void, env, voxel_m))
    if do_orientation:
        m, d = fibre_orientation(scaffold, env)
        metrics.update(m)
        dists.update(d)
    if do_skeleton:
        m, d = skeleton_metrics(scaffold, env, voxel_um)
        metrics.update(m)
        dists.update(d)
    if do_correlations:
        m, d = correlation_metrics(void, env, voxel_m)
        metrics.update(m)
        dists.update(d)

    # ---- optional independent cross-checks (graceful; never block the spine) ----
    cc_flags = {"pytrax": with_pytrax, "gudhi": with_gudhi, "puma": with_puma,
                "lbm": with_lbm, "bonej": with_bonej, "diameterj": with_diameterj}
    if any(cc_flags.values()):
        import crosschecks as _cc
        scaffold_in = scaffold & env
        notes = []
        for name, on in cc_flags.items():
            if not on:
                continue
            fn = _cc.REGISTRY[name][0]
            try:
                if name == "pytrax":
                    cm, cd, note = fn(void_in, axes=axes)
                elif name == "gudhi":
                    cm, cd, note = fn(void_in)
                elif name in ("puma", "lbm"):
                    cm, cd, note = fn(void_in, voxel_m, axes=axes)
                else:                                # bonej, diameterj
                    cm, cd, note = fn(scaffold_in, voxel_um)
                metrics.update(cm)
                dists.update(cd)
                notes.append(f"{name}: {note}")
                print(f"   [xcheck:{name}] {note}")
            except Exception as e:
                notes.append(f"{name}: error {e}")
                print(f"   [xcheck:{name}] error {e}")
        if notes:
            metrics["crosscheck_notes"] = " | ".join(notes)

    metrics["runtime_s"] = round(time.time() - t0, 1)

    if write:
        _write_outputs(metrics, dists, source, out_dir, voxel_um, seg_method,
                       seg_level)
    print(f"[{source}] phi={total_phi:.3f} (acc {acc_phi:.3f}) "
          f"K={metrics.get('perm_mean_mD', float('nan')):.1f} mD "
          f"Z={metrics.get('coordination_mean', float('nan')):.2f} "
          f"({metrics['runtime_s']}s)")
    return metrics


def _write_outputs(metrics, dists, source, out_dir, voxel_um, seg_method, seg_level):
    import pandas as pd
    scalar_df = pd.DataFrame([metrics])
    sheets = {"metrics": scalar_df}
    maxlen = max((len(v) for v in dists.values()), default=0)
    if maxlen:
        dd = {k: np.concatenate([np.asarray(v, float),
                                 np.full(maxlen - len(v), np.nan)])
              for k, v in dists.items()}
        sheets["distributions"] = pd.DataFrame(dd)
    stem = f"{source}_morphometry"
    if _HAVE_FACTORY:
        viz.write_workbook(sheets, out_dir / f"{stem}.xlsx",
                           meta={"source": source, "voxel_um": voxel_um,
                                 "seg_method": seg_method, "seg_level": seg_level})
        prov.stamp(out_dir, params={"source": source, "voxel_um": voxel_um,
                                    "seg_method": seg_method, "seg_level": seg_level},
                   inputs=[])
    else:
        scalar_df.to_csv(out_dir / f"{stem}_metrics.csv", index=False)
        if "distributions" in sheets:
            sheets["distributions"].to_csv(out_dir / f"{stem}_distributions.csv",
                                           index=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Act II transport-morphometry engine")
    ap.add_argument("--vol", required=True, help="volume: TIFF folder or multipage TIFF")
    ap.add_argument("--voxel", type=float, required=True, help="voxel size in micrometres")
    ap.add_argument("--source", required=True, help="label: HR / SR / interp / DeepScout")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mask", default=None, help="optional envelope mask (TIFF stack)")
    ap.add_argument("--seg-method", default="otsu", choices=["otsu", "fixed", "percentile"])
    ap.add_argument("--seg-level", type=float, default=None)
    ap.add_argument("--normalize", action="store_true",
                    help="percentile-normalise to [0,1] first (use with --seg-method fixed)")
    ap.add_argument("--max-size", type=int, default=None,
                    help="crop a central cube of this many voxels per side (quick tests)")
    ap.add_argument("--axes", default="xyz")
    ap.add_argument("--no-perm", action="store_true")
    ap.add_argument("--no-tort", action="store_true")
    ap.add_argument("--no-surface", action="store_true")
    ap.add_argument("--no-topology", action="store_true")
    ap.add_argument("--no-orientation", action="store_true")
    ap.add_argument("--no-skeleton", action="store_true")
    ap.add_argument("--no-correlations", action="store_true")
    ap.add_argument("--with-pytrax", action="store_true", help="random-walk tortuosity cross-check")
    ap.add_argument("--with-gudhi", action="store_true", help="persistent-homology Betti cross-check")
    ap.add_argument("--with-puma", action="store_true", help="PuMA Stokes perm/tort/SSA cross-check (conda)")
    ap.add_argument("--with-lbm", action="store_true", help="LBM DNS permeability cross-check (MPLBM-UT/WSL)")
    ap.add_argument("--with-bonej", action="store_true", help="BoneJ2 morphometry cross-check (pyimagej/Java)")
    ap.add_argument("--with-diameterj", action="store_true", help="DiameterJ 2D fibre diameter (pyimagej/Java)")
    args = ap.parse_args(argv)

    vol = load_stack(args.vol)
    if args.normalize:
        vol = normalize(vol)
    if args.max_size:
        s = args.max_size
        c = [dim // 2 for dim in vol.shape]
        sl = tuple(slice(max(0, ci - s // 2), max(0, ci - s // 2) + s) for ci in c)
        vol = vol[sl]
        print(f"[crop] central cube -> {vol.shape}")
    mask = load_stack(args.mask) > 0 if args.mask else None
    if mask is not None and args.max_size:
        mask = mask[sl]
    axes = tuple({"x": 0, "y": 1, "z": 2}[a] for a in args.axes)

    analyze_volume(vol, args.voxel, args.source, args.out, mask=mask,
                   seg_method=args.seg_method, seg_level=args.seg_level, axes=axes,
                   do_perm=not args.no_perm, do_tort=not args.no_tort,
                   do_surface=not args.no_surface, do_topology=not args.no_topology,
                   do_orientation=not args.no_orientation,
                   do_skeleton=not args.no_skeleton,
                   do_correlations=not args.no_correlations,
                   with_pytrax=args.with_pytrax, with_gudhi=args.with_gudhi,
                   with_puma=args.with_puma, with_lbm=args.with_lbm,
                   with_bonej=args.with_bonej, with_diameterj=args.with_diameterj)
    return 0


if __name__ == "__main__":
    sys.exit(main())
