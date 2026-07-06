#!/usr/bin/env python3
"""
train_seg.py -- cross-architecture 3D segmentation trainer for the scaffx toolbox.

Trains ONE generalist 3D U-Net that segments every fabrication route (electrospun fibres,
laser-machined and porogen solids, 2PP lattices) from the artifact-laden reconstructions,
and -- critically -- works on HR and SR identically.  The artifact-specific design:

  * SIMULATE-AND-INVERT augmentation: take a filled label and synthesise the
    phase-contrast shell (bright rim + dark core) + beam-hardening cupping on the *input*,
    so the network is explicitly trained input=fringed -> label=filled.  This is the
    segmentation analogue of the SR degradation-simulation.
  * Connectivity-preserving loss: Dice + soft-clDice (Shit 2021) so thin throats and the
    pore network's topology survive -- the property the downstream transport metrics depend on.
  * Optional gradient input channel (the rim is a gradient cue).
  * --holdout <arch>: train on all architectures but one, validate on the held-out one
    -> the leave-one-architecture-out SEGMENTATION generalisation experiment (-> SI S6).

Inputs: raw HR volumes (from the aligned dir) paired with bootstrapped labels
(seg_labels/<arch>/<arch>_labels.tif from run_segment_all.py).

Usage (uv; RTX 5090):
  uv add monai
  uv run python train_seg.py --raw-root "<aligned dir>" --label-root seg_labels \
      --out seg_model --epochs 300 --patch 96 --grad-channel
  uv run python train_seg.py ... --holdout pel16        # leave-one-architecture-out
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

import torch
import torch.nn.functional as F

import morphometry as mp

try:
    from monai.networks.nets import UNet
    from monai.losses import DiceCELoss
    from monai.metrics import DiceMetric
    from monai.inferers import sliding_window_inference
    _HAVE_MONAI = True
except Exception:
    _HAVE_MONAI = False


# ============================================================ connectivity loss
def _soft_erode(x):
    p1 = -F.max_pool3d(-x, (3, 1, 1), 1, (1, 0, 0))
    p2 = -F.max_pool3d(-x, (1, 3, 1), 1, (0, 1, 0))
    p3 = -F.max_pool3d(-x, (1, 1, 3), 1, (0, 0, 1))
    return torch.min(torch.min(p1, p2), p3)


def _soft_open(x):
    return F.max_pool3d(_soft_erode(x), 3, 1, 1)


def soft_skel(x, iters=3):
    x1 = _soft_open(x)
    skel = F.relu(x - x1)
    for _ in range(iters):
        x = _soft_erode(x)
        x1 = _soft_open(x)
        delta = F.relu(x - x1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice(pred_fg, true_fg, iters=3, smooth=1.0):
    """soft centreline-Dice on foreground probabilities (connectivity preservation)."""
    sp = soft_skel(pred_fg, iters)
    st = soft_skel(true_fg, iters)
    tprec = (torch.sum(sp * true_fg) + smooth) / (torch.sum(sp) + smooth)
    tsens = (torch.sum(st * pred_fg) + smooth) / (torch.sum(st) + smooth)
    return 1.0 - 2.0 * tprec * tsens / (tprec + tsens)


def betti_of(lab_bool):
    """Betti numbers (b0,b1,b2) of a binary volume via gudhi."""
    import gudhi
    cc = gudhi.CubicalComplex(top_dimensional_cells=np.where(lab_bool, 0.0, 1.0))
    cc.compute_persistence()
    b = list(cc.persistent_betti_numbers(0.5, 0.5)) + [0, 0, 0]
    return {0: int(b[0]), 1: int(b[1]), 2: int(b[2])}


def cubical_topo_loss(pred_fg, betti_target, dims=(0, 1)):
    """Differentiable persistence-matching topology loss (Clough/Hu family): push the
    prediction's top-Betti_k persistence features to full lifetime and suppress the rest,
    matching the target's Betti numbers. Birth/death pixel values are gathered from the
    prediction tensor (gudhi gives the critical voxel indices), so it is differentiable."""
    import gudhi
    flat = (1.0 - pred_fg).reshape(-1)                       # sublevel filtration
    arr = flat.detach().cpu().numpy().astype(np.float64).reshape(tuple(pred_fg.shape))
    cc = gudhi.CubicalComplex(top_dimensional_cells=arr)
    cc.compute_persistence()
    reg, _ = cc.cofaces_of_persistence_pairs()
    N = flat.numel()
    terms = []
    for k in dims:
        if k >= len(reg) or len(reg[k]) == 0:
            continue
        pr = np.asarray(reg[k])
        if pr.size == 0 or pr.max() >= N:
            continue
        b = torch.as_tensor(pr[:, 0].astype(np.int64), device=pred_fg.device)
        d = torch.as_tensor(pr[:, 1].astype(np.int64), device=pred_fg.device)
        life = flat[d] - flat[b]
        life = life[torch.argsort(life, descending=True)]
        nt = max(int(betti_target.get(k, 0)) - (1 if k == 0 else 0), 0)  # 1 component is essential
        if nt > 0:
            terms.append((1.0 - life[:nt]) ** 2)
        if life[nt:].numel() > 0:
            terms.append(life[nt:] ** 2)
    if not terms:
        return pred_fg.new_zeros(())
    return torch.cat(terms).mean()


class TopoLoss(torch.nn.Module):
    """Dice+CE plus a topology regulariser:
       mode='cldice' -- fast soft-clDice (default workhorse for dev + LOAO sweeps),
       mode='betti'  -- persistence-matching to the target's Betti numbers (recommended
                        for the FINAL model + the loss-ablation figure; uses the canonical
                        Betti-Matching-3D backend if installed, else the in-repo gudhi loss),
       mode='dice'   -- Dice+CE only (the vanilla baseline arm of the ablation)."""
    def __init__(self, mode="cldice", alpha=0.5, iters=3, betti_dims=(0, 1)):
        super().__init__()
        self.mode = mode
        self.alpha = alpha
        self.iters = iters
        self.betti_dims = betti_dims
        self.dicece = DiceCELoss(to_onehot_y=True, softmax=True) if _HAVE_MONAI else None
        self._bm_note = False

    def _betti(self, pred_fg, target):
        try:
            import betti_matching  # efficient 3D Betti matching (Stucki/Bauer)
            if hasattr(betti_matching, "BettiMatchingLoss"):
                if not self._bm_note:
                    print("[loss] using Betti-Matching-3D backend"); self._bm_note = True
                return betti_matching.BettiMatchingLoss()(pred_fg, (target == 1).float())
        except Exception:
            if not self._bm_note:
                print("[loss] Betti-Matching-3D not found -> in-repo gudhi persistence loss "
                      "(install the optional spatial backend through uv if required)")
                self._bm_note = True
        losses = [cubical_topo_loss(pred_fg[b, 0],
                                    betti_of((target[b, 0] > 0).detach().cpu().numpy()),
                                    self.betti_dims)
                  for b in range(pred_fg.shape[0])]
        return torch.stack(losses).mean()

    def forward(self, logits, target):
        base = self.dicece(logits, target)
        if self.alpha <= 0 or self.mode == "dice":
            return base
        pred_fg = torch.softmax(logits, 1)[:, 1:2].clamp(1e-3, 1 - 1e-3)  # keep clDice skeleton soft -> stable gradient
        if self.mode == "cldice":
            reg = soft_cldice(pred_fg, (target == 1).float(), self.iters)
        else:
            reg = self._betti(pred_fg, target)
        return base + self.alpha * reg


# =================================================== simulate-and-invert artifact
def render_artifact(solid, rim_i=0.9, core_i=0.45, void_i=0.1,
                    cup=0.4, rim_dilate=1, noise=0.03, rng=None):
    """Synthesize a CT-like input from a filled solid label: bright phase-contrast rim,
    dark hollow-looking core, radial beam-hardening cupping, noise."""
    from scipy import ndimage as ndi
    rng = rng or np.random.default_rng()
    solid = solid.astype(bool)
    rim = solid ^ ndi.binary_erosion(solid, iterations=1)
    if rim_dilate:
        rim = ndi.binary_dilation(rim, iterations=int(rim_dilate))
    img = np.full(solid.shape, void_i, np.float32)
    img[solid] = core_i
    img[rim] = rim_i
    zz, yy, xx = np.indices(solid.shape)
    c = [s / 2 for s in solid.shape]
    r = np.sqrt((zz - c[0])**2 + (yy - c[1])**2 + (xx - c[2])**2)
    r /= (r.max() + 1e-6)
    img *= (1.0 - cup) + cup * r                     # darker toward centre
    img += noise * rng.standard_normal(solid.shape).astype(np.float32)
    return img


# ================================================================== data + patches
def build_pairs(raw_root, label_root, archs=None):
    """Pair raw HR volume folders with bootstrapped label TIFFs by architecture name."""
    raw_root, label_root = Path(raw_root), Path(label_root)
    pairs = []
    for ld in sorted(label_root.iterdir()):
        if not ld.is_dir():
            continue
        arch = ld.name
        if archs and arch not in archs:
            continue
        lab = ld / f"{arch}_labels.tif"
        raw = None
        archdir = raw_root / arch
        if archdir.exists():
            try:
                raw = Path(mp.resolve_volume(str(archdir)))   # picks hr.tif; excludes lr/masks (avoids stacking mismatched shapes)
            except Exception as e:
                print(f"[pairs] {arch}: could not resolve HR volume ({e})")
        if lab.exists() and raw is not None:
            pairs.append({"arch": arch, "raw": str(raw), "label": str(lab)})
        else:
            print(f"[pairs] {arch}: missing raw or label (raw={raw}, label={lab.exists()})")
    return pairs


def load_volume_pair(p):
    raw = mp.normalize(mp.load_stack(p["raw"])).astype(np.float32)
    import tifffile
    lab = (tifffile.imread(p["label"]) > 0).astype(np.uint8)
    if lab.shape != raw.shape:
        raise ValueError(f"{p['arch']}: raw {raw.shape} != label {lab.shape}")
    return raw, lab


def rand_patch(raw, lab, size, rng):
    D, H, W = raw.shape
    s = min(size, D, H, W)
    z = rng.integers(0, max(D - s, 1)); y = rng.integers(0, max(H - s, 1)); x = rng.integers(0, max(W - s, 1))
    return (raw[z:z + s, y:y + s, x:x + s].copy(),
            lab[z:z + s, y:y + s, x:x + s].copy())


def augment(rawp, labp, rng, sim_prob=0.4, grad=False):
    # simulate-and-invert: with prob, replace input by a synthetic-artifact render of the label
    if rng.random() < sim_prob and labp.any():
        rawp = render_artifact(labp, rng=rng,
                               rim_i=rng.uniform(0.75, 1.0), core_i=rng.uniform(0.35, 0.55),
                               cup=rng.uniform(0.2, 0.5), rim_dilate=int(rng.integers(1, 3)),
                               noise=rng.uniform(0.01, 0.05))
        rawp = mp.normalize(rawp).astype(np.float32)
    # flips / 90-rotations
    for ax in range(3):
        if rng.random() < 0.5:
            rawp = np.flip(rawp, ax); labp = np.flip(labp, ax)
    k = int(rng.integers(0, 4))
    if k:
        rawp = np.rot90(rawp, k, axes=(1, 2)); labp = np.rot90(labp, k, axes=(1, 2))
    rawp = np.ascontiguousarray(rawp); labp = np.ascontiguousarray(labp)
    chans = [rawp]
    if grad:
        from scipy import ndimage as ndi
        chans.append(ndi.gaussian_gradient_magnitude(rawp, 1.0).astype(np.float32))
    img = np.stack(chans, 0)                          # (C,D,H,W)
    return img, labp[None]                             # label (1,D,H,W)


# ====================================================================== training
def train(pairs, val_pairs, out_dir, epochs=300, patch=96, batch=2, lr=1e-3,
          sim_prob=0.4, grad=False, cldice_alpha=0.5, iters_per_epoch=40, seed=0,
          loss_mode="cldice", val_every=1, sw_batch=8):
    if not _HAVE_MONAI:
        print("MONAI not available -- uv add monai (needs your uv torch cu128).")
        return
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    in_ch = 2 if grad else 1

    model = UNet(spatial_dims=3, in_channels=in_ch, out_channels=2,
                 channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
                 num_res_units=2).to(dev)
    loss_fn = TopoLoss(mode=loss_mode, alpha=cldice_alpha)
    print(f"[loss] {loss_mode} (alpha {cldice_alpha})")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # pre-load volumes (128 GB RAM); for very large sets, swap to lazy loading
    vols = {p["arch"]: load_volume_pair(p) for p in pairs}
    vvols = {p["arch"]: load_volume_pair(p) for p in val_pairs} if val_pairs else {}

    # fixed validation patches (matches the SR trainer: fast, stable per-epoch Dice;
    # the full-volume Dice -- the reported number -- is computed once at the end via eval_seg.py)
    VAL_K, VAL_SEED = 8, 12345
    val_patches = []
    if vvols:
        _vrng = np.random.default_rng(VAL_SEED)
        _varchs = list(vvols.keys())
        for _ in range(VAL_K):
            _a = _varchs[int(_vrng.integers(0, len(_varchs)))]
            _raw, _lab = vvols[_a]
            _rp, _lp = rand_patch(_raw, _lab, patch, _vrng)
            _chans = [_rp.astype(np.float32)]
            if grad:
                from scipy import ndimage as ndi
                _chans.append(ndi.gaussian_gradient_magnitude(_rp, 1.0).astype(np.float32))
            val_patches.append((np.stack(_chans, 0), _lp.astype(np.int64)))
    archs = list(vols.keys())
    log = []
    best = -float("inf")

    # ---- provenance + kill-safe metrics (repeatability is the overarching principle) ----
    import json, csv, subprocess, platform, datetime, sys, hashlib
    def _ver(m):
        try: return __import__(m).__version__
        except Exception: return None
    def _git(*a):
        try:
            return subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parent), *a],
                                           text=True, stderr=subprocess.DEVNULL).strip()
        except Exception: return None
    def _sha256(fp):
        p = Path(fp)
        if not p.exists(): return None
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    def _fp(fp):
        p = Path(fp); st = p.stat() if p.exists() else None
        return {"path": str(p), "bytes": (st.st_size if st else None),
                "mtime": (datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds") if st else None),
                "sha256": _sha256(fp)}
    print("[provenance] hashing inputs (sha256) ...")
    manifest = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "cmdline": " ".join(sys.argv), "cwd": str(Path.cwd()),
        "host": platform.node(), "os": platform.platform(), "python": platform.python_version(),
        "device": dev,
        "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "cuda": getattr(torch.version, "cuda", None),
        "git": {"commit": _git("rev-parse", "HEAD"),
                "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": bool(_git("status", "--porcelain"))},
        "versions": {m: _ver(m) for m in ("torch", "monai", "numpy", "scipy", "tifffile", "skimage", "cc3d")},
        "reproducibility": {"seed": seed, "torch_seeded": True, "cudnn_benchmark": False,
                            "fully_deterministic": False},
        "params": {"epochs": epochs, "patch": patch, "batch": batch, "lr": lr,
                   "sim_prob": sim_prob, "grad": grad, "cldice_alpha": cldice_alpha,
                   "iters_per_epoch": iters_per_epoch, "seed": seed, "loss_mode": loss_mode,
                   "val_every": val_every, "sw_batch": sw_batch,
                   "val_kind": "fixed_patches", "val_k": VAL_K, "val_seed": VAL_SEED},
        "train": [{"arch": p["arch"], "raw": _fp(p["raw"]), "label": _fp(p["label"])} for p in pairs],
        "val": [{"arch": p["arch"], "raw": _fp(p["raw"]), "label": _fp(p["label"])} for p in (val_pairs or [])],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[provenance] manifest.json written "
          f"({len(manifest['train'])} train + {len(manifest['val'])} val inputs hashed); "
          f"git dirty={manifest['git']['dirty']}")
    _mlog = open(out_dir / "training_log.csv", "w", newline="")
    _mw = csv.writer(_mlog); _mw.writerow(["epoch", "train_loss", "val_dice"]); _mlog.flush()

    # ---- background prefetch (overlap CPU augmentation with GPU compute) + progress bar ----
    import threading, queue as _queue
    try:
        from tqdm import tqdm as _tqdm
        _HAVE_TQDM = True
    except Exception:
        _HAVE_TQDM = False
        def _tqdm(x, **k): return x

    def _producer(q, n):
        # single thread => rng consumed in the exact same order as the serial loop
        # (reproducibility preserved: identical batches for a given seed)
        for _ in range(n):
            imgs, labs = [], []
            for _b in range(batch):
                a = archs[int(rng.integers(0, len(archs)))]
                raw, lab = vols[a]
                rp, lp = rand_patch(raw, lab, patch, rng)
                img, l = augment(rp, lp, rng, sim_prob=sim_prob, grad=grad)
                imgs.append(img); labs.append(l)
            q.put((np.stack(imgs), np.stack(labs)))
        q.put(None)

    _q = _queue.Queue(maxsize=4)
    _pth = threading.Thread(target=_producer, args=(_q, epochs * iters_per_epoch), daemon=True)
    _pth.start()

    for ep in range(epochs):
        model.train()
        running = 0.0
        _bar = _tqdm(range(iters_per_epoch), desc=f"epoch {ep:3d}/{epochs}",
                     leave=False, dynamic_ncols=True) if _HAVE_TQDM else range(iters_per_epoch)
        for _i in _bar:
            item = _q.get()
            if item is None:
                break
            xb, yb = item
            x = torch.from_numpy(xb).float().to(dev)
            x = torch.nan_to_num(x)                       # neutralise degenerate (constant) sim patches
            y = torch.from_numpy(yb).long().to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(dev == "cuda")):
                logits = model(x)
            logits = torch.nan_to_num(logits.float()).clamp(-30, 30)   # sanitise (clamp is NOT NaN-safe) + bound, fp32
            loss = loss_fn(logits, y)
            if torch.isfinite(loss):
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
                running += float(loss.detach())
            else:
                (_tqdm.write if _HAVE_TQDM else print)(f"  [warn] non-finite loss at epoch {ep}; batch skipped")
            if _HAVE_TQDM:
                _bar.set_postfix(loss=f"{running / (_i + 1):.4f}")
        if _HAVE_TQDM:
            _bar.close()
        sched.step()
        tr = running / iters_per_epoch

        # validation on a fixed set of small patches (fast, stable Dice; no full-volume sweep)
        vdice = float("nan")
        if val_patches and (ep % val_every == 0 or ep == epochs - 1):
            model.eval(); dice_metric.reset()
            with torch.no_grad():
                xb = torch.from_numpy(np.stack([p[0] for p in val_patches])).float().to(dev)
                yb = torch.from_numpy(np.stack([p[1] for p in val_patches])).long().to(dev)
                preds = []
                for i in range(0, xb.shape[0], sw_batch):
                    with torch.cuda.amp.autocast(enabled=(dev == "cuda")):
                        lo = model(xb[i:i + sw_batch])
                    preds.append(torch.argmax(lo, 1))
                pred = torch.cat(preds, 0)
                dice_metric(y_pred=F.one_hot(pred, 2).permute(0, 4, 1, 2, 3),
                            y=F.one_hot(yb, 2).permute(0, 4, 1, 2, 3))
            vdice = float(dice_metric.aggregate().item())

        log.append({"epoch": ep, "train_loss": tr, "val_dice": vdice})
        _mw.writerow([ep, f"{tr:.6f}", f"{vdice:.6f}"]); _mlog.flush()
        _summ = f"epoch {ep:3d} | loss {tr:.4f}" + (f" | val Dice {vdice:.4f}" if vdice == vdice else "")
        _tqdm.write(_summ) if _HAVE_TQDM else print(_summ)
        score = vdice if vdice == vdice else -tr
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "in_ch": in_ch, "patch": patch,
                        "grad": grad, "epoch": ep, "val_dice": vdice},
                       out_dir / "best.pt")

    _mlog.close()
    torch.save({"model": model.state_dict(), "in_ch": in_ch, "patch": patch, "grad": grad},
               out_dir / "last.pt")
    print(f"done -> {out_dir/'best.pt'} (best score {best:.4f}; val Dice if held out, else -loss)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="cross-architecture 3D segmentation trainer")
    ap.add_argument("--raw-root", required=True, help="aligned dir with <arch>/HR volumes")
    ap.add_argument("--label-root", required=True, help="seg_labels dir from run_segment_all.py")
    ap.add_argument("--out", default="seg_model")
    ap.add_argument("--holdout", default=None, help="architecture to leave out (LOAO val)")
    ap.add_argument("--val-frac", type=float, default=0.0, help="if no holdout, fraction held for val")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patch", type=int, default=96)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sim-prob", type=float, default=0.4, help="simulate-and-invert probability")
    ap.add_argument("--loss", default="dice_cldice",
                    choices=["dice", "dice_cldice", "dice_betti"],
                    help="topology loss: dice_cldice (fast default) | dice_betti "
                         "(persistence-faithful, for final model + ablation) | dice (vanilla baseline)")
    ap.add_argument("--cldice-alpha", type=float, default=0.5)
    ap.add_argument("--iters-per-epoch", type=int, default=40)
    ap.add_argument("--grad-channel", action="store_true", help="append gradient-magnitude channel")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-every", type=int, default=1, help="validate every N epochs (calibration: 5)")
    ap.add_argument("--sw-batch", type=int, default=8, help="sliding-window batch size for validation")
    args = ap.parse_args(argv)

    pairs = build_pairs(args.raw_root, args.label_root)
    if not pairs:
        print("no (raw,label) pairs found -- run run_segment_all.py first."); return 1
    print(f"pairs: {[p['arch'] for p in pairs]}")

    if args.holdout:
        val = [p for p in pairs if p["arch"] == args.holdout]
        train_set = [p for p in pairs if p["arch"] != args.holdout]
        print(f"LOAO: train {[p['arch'] for p in train_set]} | holdout {args.holdout}")
    elif args.val_frac > 0 and len(pairs) > 1:
        n = max(1, int(round(args.val_frac * len(pairs))))
        val, train_set = pairs[:n], pairs[n:]
    else:
        val, train_set = [], pairs

    train(train_set, val, args.out, epochs=args.epochs, patch=args.patch, batch=args.batch,
          lr=args.lr, sim_prob=args.sim_prob, grad=args.grad_channel,
          cldice_alpha=args.cldice_alpha, iters_per_epoch=args.iters_per_epoch, seed=args.seed,
          loss_mode={"dice": "dice", "dice_cldice": "cldice", "dice_betti": "betti"}[args.loss],
          val_every=args.val_every, sw_batch=args.sw_batch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
