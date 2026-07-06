#!/usr/bin/env python3
r"""
train_sr_real.py  --  unified super-resolution training on REAL aligned pairs.

Trains SR3DNet to map a real LR scan (already resampled onto the HR grid by align_pair +
prep_pair: the `lr.tif`) to the real HR (`hr.tif`), pooled across every aligned sample in a
manifest. This is "Arm A" -- real-paired supervision, the anti-hallucination model.

Design (see also the conversation notes):
  * MANIFEST-DRIVEN. A JSON lists each prepped pair {name, dir, architecture, role}. Adding
    pellethane (or any future pair) = align_pair -> prep_pair -> it appears in the manifest ->
    retrain. No code changes. role="train" pairs are trained on; role="test" pairs are held
    out and only validated (leave-one-architecture-out).
  * SPATIAL HOLDOUT per pair: first (1-val_frac) of z trains, last val_frac validates, with a
    one-patch margin gap between them so no training patch reaches into the val slab.
  * FOREGROUND-BIASED SAMPLER: ~struct_prob of patches must contain >= min_frac scaffold
    voxels (rejection-sampled from the scaffold mask) so capacity goes to walls/edges; the
    rest are drawn anywhere in-FOV so the model also learns to leave smooth regions smooth
    (anti-hallucination). Loss is over the WHOLE patch -- masks steer sampling, never the loss.
  * VERSIONED OUTPUT: runs/model_<version>/ holds the checkpoint, a snapshot of the exact
    manifest used, the config+seed, the log, and per-architecture val metrics -- so every
    model version is pinned to a known dataset version. A scaling_log.csv accumulates
    (version, #pairs, architectures, val PSNR) across versions -> the data-scaling curve.
  * WARM-START: --init-from <ckpt> initialises weights from a previous version but still
    trains on the full current manifest (fast turnaround, still reproducible).

Open tools only. Pairs are loaded as float16 in RAM (halved memory) and cast per-patch.

Usage:
  uv run python train_sr_real.py --scan "C:\...\codeoutput\aligned"      # build/update manifest
  uv run python train_sr_real.py --manifest "C:\...\codeoutput\aligned\manifest.json" `
        --out "C:\...\codeoutput\runs" --version v1 --epochs 25
"""
from __future__ import annotations
import argparse, glob, json, os, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
try:
    from tqdm import tqdm
except ImportError:                       # graceful fallback if tqdm isn't installed
    def tqdm(x, **k):
        return x

from sr_validation_harness import load_stack
from sr_model import SR3DNet, SRLoss, psnr


# ------------------------------------------------------------------- manifest
def guess_arch(name):
    n = name.lower()
    if n.startswith("vcr") or "porogen" in n:
        return "porogen"
    if "pel" in n:
        return "electrospun"
    if n.startswith("laser") or n.startswith("l"):
        return "laser"
    return "unknown"


def scan_manifest(root, manifest_path):
    """Add any prepped pair folder under `root` to the manifest as role=train, preserving
    existing entries/roles. A prepped folder has hr.tif + lr.tif + scaffold_mask.tif."""
    existing = {}
    if os.path.exists(manifest_path):
        for e in json.load(open(manifest_path)).get("pairs", []):
            existing[e["name"]] = e
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(d):
            continue
        if all(os.path.exists(os.path.join(d, f)) for f in
               ("hr.tif", "lr.tif", "scaffold_mask.tif")):
            name = os.path.basename(d)
            if name not in existing:
                existing[name] = dict(name=name, dir=os.path.abspath(d),
                                      architecture=guess_arch(name), role="train")
                print(f"[scan] + {name}  (arch={existing[name]['architecture']}, role=train)")
            else:
                existing[name]["dir"] = os.path.abspath(d)  # refresh path
    pairs = sorted(existing.values(), key=lambda e: e["name"])
    json.dump(dict(pairs=pairs), open(manifest_path, "w"), indent=2)
    print(f"[scan] manifest -> {manifest_path}  ({len(pairs)} pairs)")
    for e in pairs:
        print(f"   {e['name']:12s} arch={e['architecture']:11s} role={e['role']}")
    return pairs


def load_manifest(path):
    return json.load(open(path))["pairs"]


# ------------------------------------------------------------------- dataset
class Pair:
    """One aligned pair held in RAM (float16) with its spatial-holdout z-ranges."""
    def __init__(self, entry, val_frac, patch):
        self.name = entry["name"]; self.arch = entry["architecture"]; self.role = entry["role"]
        d = entry["dir"]
        self.hr = load_stack(os.path.join(d, "hr.tif")).astype(np.float16)
        self.lr = load_stack(os.path.join(d, "lr.tif")).astype(np.float16)
        self.scaf = (load_stack(os.path.join(d, "scaffold_mask.tif")) > 0.5)
        assert self.hr.shape == self.lr.shape == self.scaf.shape, f"shape mismatch in {d}"
        D = self.hr.shape[0]
        n_val = max(patch, int(round(D * val_frac)))
        n_val = min(n_val, D // 2)
        self.val_z = (D - n_val, D)
        self.train_z = (0, max(patch, D - n_val - patch))     # one-patch margin gap before val
        self.shape = self.hr.shape

    def patch_at(self, z0, y0, x0, p):
        sl = (slice(z0, z0 + p), slice(y0, y0 + p), slice(x0, x0 + p))
        return (self.lr[sl].astype(np.float32), self.hr[sl].astype(np.float32),
                self.scaf[sl])


class RealPairDataset(Dataset):
    def __init__(self, pairs, patch=64, n_patches=4000, region="train",
                 struct_prob=0.8, min_frac=0.1, max_tries=30, augment=True, seed=0):
        self.pairs = pairs; self.p = patch; self.n = n_patches; self.region = region
        self.struct_prob = struct_prob; self.min_frac = min_frac; self.max_tries = max_tries
        self.augment = augment; self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n

    def _z_range(self, pair):
        return pair.train_z if self.region == "train" else pair.val_z

    def _sample_start(self, pair):
        p = self.p
        z0r, z1r = self._z_range(pair)
        zmax = max(z0r, z1r - p); _, Y, X = pair.shape
        want_struct = self.rng.random() < self.struct_prob
        tries = self.max_tries if want_struct else 1
        best = None
        for _ in range(tries):
            z = int(self.rng.integers(z0r, zmax + 1))
            y = int(self.rng.integers(0, max(1, Y - p + 1)))
            x = int(self.rng.integers(0, max(1, X - p + 1)))
            if not want_struct:
                return z, y, x
            frac = pair.scaf[z:z + p, y:y + p, x:x + p].mean()
            if frac >= self.min_frac:
                return z, y, x
            if best is None or frac > best[0]:
                best = (frac, (z, y, x))
        return best[1]

    def __getitem__(self, idx):
        pair = self.pairs[int(self.rng.integers(0, len(self.pairs)))]   # equal weight per pair
        z, y, x = self._sample_start(pair)
        lr, hr, _ = pair.patch_at(z, y, x, self.p)
        if self.augment:
            for ax in range(3):
                if self.rng.random() < 0.5:
                    lr = np.flip(lr, ax); hr = np.flip(hr, ax)
            if self.rng.random() < 0.5:           # in-plane 90-deg rotation
                k = int(self.rng.integers(1, 4))
                lr = np.rot90(lr, k, axes=(1, 2)); hr = np.rot90(hr, k, axes=(1, 2))
        return (torch.from_numpy(np.ascontiguousarray(lr)[None]),
                torch.from_numpy(np.ascontiguousarray(hr)[None]))


def fixed_val_patches(pairs, patch, k_per_pair=8, seed=123):
    """A stable set of scaffold-centred val patches per pair, grouped by architecture."""
    rng = np.random.default_rng(seed)
    ds = RealPairDataset(pairs, patch=patch, n_patches=1, region="val",
                         struct_prob=1.0, min_frac=0.1, augment=False, seed=seed)
    out = []
    for pair in pairs:
        for _ in range(k_per_pair):
            ds.rng = rng
            z, y, x = ds._sample_start(pair)
            lr, hr, _ = pair.patch_at(z, y, x, patch)
            out.append((torch.from_numpy(lr[None]), torch.from_numpy(hr[None]),
                        pair.arch, pair.name))
    return out


# ------------------------------------------------------------------- train
def evaluate(net, val_pairs, device):
    net.eval(); per_arch = {}
    with torch.no_grad():
        for l, h, arch, _ in val_pairs:
            v = psnr(net(l[None].to(device)), h[None].to(device))
            per_arch.setdefault(arch, []).append(v)
    arch_mean = {a: float(np.mean(v)) for a, v in per_arch.items()}
    overall = float(np.mean([v for vs in per_arch.values() for v in vs]))
    return overall, arch_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", help="scan this aligned-root and (re)build the manifest, then exit")
    ap.add_argument("--manifest", help="manifest.json listing aligned pairs")
    ap.add_argument("--out", default="runs", help="root for versioned run folders")
    ap.add_argument("--version", default="v1", help="model version label")
    ap.add_argument("--init-from", help="checkpoint to warm-start weights from (still trains "
                    "on the full manifest)")
    ap.add_argument("--factor", type=float, default=2.521)
    ap.add_argument("--val-frac", type=float, default=0.5)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--iters-per-epoch", type=int, default=300)
    ap.add_argument("--struct-prob", type=float, default=0.8)
    ap.add_argument("--min-frac", type=float, default=0.1)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.scan:
        scan_manifest(args.scan, os.path.join(args.scan, "manifest.json"))
        return

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    chlast = args.channels_last and device == "cuda"

    if args.selftest:
        return run_selftest(device)

    assert args.manifest, "provide --manifest (or --scan / --selftest)"
    entries = load_manifest(args.manifest)
    train_entries = [e for e in entries if e.get("role", "train") == "train"]
    test_entries = [e for e in entries if e.get("role") == "test"]
    assert train_entries, "no role=train entries in manifest"

    print(f"[data] loading {len(train_entries)} train pairs"
          + (f" + {len(test_entries)} held-out test pairs" if test_entries else ""))
    train_pairs = [Pair(e, args.val_frac, args.patch) for e in train_entries]
    test_pairs = [Pair(e, args.val_frac, args.patch) for e in test_entries]
    for p in train_pairs:
        print(f"   {p.name:12s} arch={p.arch:11s} shape={p.shape} "
              f"train_z={p.train_z} val_z={p.val_z}")

    out_dir = os.path.join(args.out, f"model_{args.version}")
    os.makedirs(out_dir, exist_ok=True)

    train_ds = RealPairDataset(train_pairs, args.patch, args.iters_per_epoch * args.batch,
                               region="train", struct_prob=args.struct_prob,
                               min_frac=args.min_frac, augment=True, seed=args.seed)
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=False, num_workers=0,
                        pin_memory=(device == "cuda"))
    val_pairs = fixed_val_patches(train_pairs, args.patch)
    heldout_val = fixed_val_patches(test_pairs, args.patch) if test_pairs else []

    net = SR3DNet(args.channels, args.groups, args.blocks).to(device)
    if args.init_from:
        ck = torch.load(args.init_from, map_location=device)
        net.load_state_dict(ck["model"]); print(f"[warm-start] weights <- {args.init_from}")
    if chlast:
        net = net.to(memory_format=torch.channels_last_3d)
    crit = SRLoss(); opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    use_amp = (device == "cuda"); scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    cfg = vars(args).copy()
    cfg.update(train_pairs=[e["name"] for e in train_entries],
               architectures=sorted({p.arch for p in train_pairs}))
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=2)
    json.dump(dict(pairs=entries), open(os.path.join(out_dir, "manifest_snapshot.json"), "w"),
              indent=2)
    log = open(os.path.join(out_dir, "training_log.csv"), "w")
    log.write("epoch,train_loss,val_psnr,lr,sec\n")
    print(f"[model] params={net.param_count():,}  device={device}  amp={use_amp}  -> {out_dir}")

    best = -1.0; best_arch = {}
    for ep in range(args.epochs):
        t0 = time.time(); net.train(); running = 0.0; nb = 0
        for lr, hr in tqdm(loader, desc=f"epoch {ep+1}/{args.epochs}"):
            lr, hr = lr.to(device), hr.to(device)
            if chlast:
                lr = lr.contiguous(memory_format=torch.channels_last_3d)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss, _ = crit(net(lr), hr)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            running += float(loss.detach()); nb += 1
        sched.step()
        overall, arch_mean = evaluate(net, val_pairs, device)
        dt = time.time() - t0
        amse = "  ".join(f"{a}={v:.2f}" for a, v in sorted(arch_mean.items()))
        print(f"  epoch {ep+1}: loss={running/nb:.4f}  val_psnr={overall:.2f} dB  [{amse}]  ({dt:.0f}s)")
        log.write(f"{ep+1},{running/nb:.5f},{overall:.4f},{sched.get_last_lr()[0]:.2e},{dt:.1f}\n")
        log.flush()
        if overall > best:
            best = overall; best_arch = arch_mean
            torch.save({"model": net.state_dict(), "config": cfg, "val_psnr": best},
                       os.path.join(out_dir, "best_model.pt"))
    log.close()

    heldout = {}
    if heldout_val:
        _, heldout = evaluate(net, heldout_val, device)
        print(f"[held-out] {heldout}")
    json.dump(dict(version=args.version, overall_val_psnr=best, per_architecture=best_arch,
                   heldout_architectures=heldout,
                   n_train_pairs=len(train_pairs),
                   architectures=sorted({p.arch for p in train_pairs})),
              open(os.path.join(out_dir, "val_metrics.json"), "w"), indent=2)

    scaling = os.path.join(args.out, "scaling_log.csv")
    head = not os.path.exists(scaling)
    with open(scaling, "a") as f:
        if head:
            f.write("version,n_train_pairs,architectures,val_psnr\n")
        f.write(f"{args.version},{len(train_pairs)},"
                f"{'+'.join(sorted({p.arch for p in train_pairs}))},{best:.4f}\n")
    print(f"[done] best val PSNR={best:.2f} dB  per-arch={best_arch}  -> {out_dir}/best_model.pt")


def run_selftest(device):
    import tempfile, tifffile
    from sr_validation_harness import make_porous_phantom, normalize, degrade_hr_to_lr, interpolate_to_hr
    root = tempfile.mkdtemp()
    names = ["vcr_a", "laser_b"]
    for nm in names:
        d = os.path.join(root, nm); os.makedirs(d)
        hr = normalize(make_porous_phantom((48, 96, 96), feature_scale=5.0,
                                           seed=hash(nm) % 100))[0].astype(np.float32)
        lr = interpolate_to_hr(degrade_hr_to_lr(hr, 2.521, 1.0, 0.01), hr.shape).astype(np.float32)
        scaf = (hr > 0.4).astype(np.uint8)
        tifffile.imwrite(os.path.join(d, "hr.tif"), hr)
        tifffile.imwrite(os.path.join(d, "lr.tif"), lr)
        tifffile.imwrite(os.path.join(d, "scaffold_mask.tif"), scaf)
    mpath = os.path.join(root, "manifest.json")
    scan_manifest(root, mpath)
    pairs = [Pair(e, 0.5, 32) for e in load_manifest(mpath)]
    ds = RealPairDataset(pairs, patch=32, n_patches=16, region="train", seed=0)
    loader = DataLoader(ds, batch_size=2)
    net = SR3DNet(16, 2, 2).to(device); crit = SRLoss()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for lr, hr in loader:
        loss, _ = crit(net(lr.to(device)), hr.to(device))
        opt.zero_grad(); loss.backward(); opt.step()
    vp = fixed_val_patches(pairs, 32, k_per_pair=2)
    overall, arch = evaluate(net, vp, device)
    print(f"[selftest] trained a step on 2 pooled pairs; val PSNR={overall:.2f} dB  per-arch={arch}")
    print("[selftest]", "OK" if np.isfinite(overall) and len(arch) == 2 else "CHECK")


if __name__ == "__main__":
    main()
