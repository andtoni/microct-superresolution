#!/usr/bin/env python3
r"""
train_sr.py
===========
Train the open 3D SR model (sr_model.SR3DNet) on degradation-simulated pairs generated
from an HR volume. Pre-upsampling residual learning: input = interpolated-LR on the HR
grid, target = HR. Degradation parameters are the ones we locked by tuning to the real LR.

Non-circular by construction: the HR volume is split spatially in z into a train range and
a held-out val range; the model never sees the val slices during training. (The independent
group — e.g. vcr298 — is a separate volume and stays completely untouched for final
validation.)

Usage (porogen pilot):
    uv run python train_sr.py ^
      --hr "...\Data\ct\hr_vcr276" --out runs\vcr276 ^
      --factor 2.521 --psf-sigma 1.0 --noise-sigma 0.016 ^
      --max-slices 256 --patch 64 --batch 4 --epochs 40

Quick verification before touching real data:
    uv run python train_sr.py --selftest
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from sr_model import SR3DNet, SRLoss, psnr
from sr_validation_harness import (load_stack, normalize, degrade_hr_to_lr,
                                    interpolate_to_hr, make_porous_phantom)

try:
    from tqdm import tqdm
except Exception:                                   # tqdm optional
    def tqdm(x, **k): return x


# --------------------------------------------------------------------------- dataset
class DegradationPatchDataset(Dataset):
    """Random HR patches paired with their interpolated-LR (degraded) counterpart.

    The interp-LR base volume is precomputed once (blur+downsample+noise+interp), so
    patches are cheap aligned crops. Augmentation = random axis flips.
    """
    def __init__(self, hr_vol, factor, psf, noise, patch, n_patches,
                 augment=True, seed=0):
        self.hr = np.ascontiguousarray(hr_vol, dtype=np.float32)
        lr = degrade_hr_to_lr(self.hr, factor=factor, psf_sigma_lr=psf,
                              noise_sigma=noise, seed=seed)
        self.lrint = interpolate_to_hr(lr, self.hr.shape, order=3).astype(np.float32)
        self.p = [min(patch, s) for s in self.hr.shape]   # clamp patch to volume
        self.n = n_patches
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n

    def _crop(self):
        starts = [int(self.rng.integers(0, max(1, s - p + 1)))
                  for s, p in zip(self.hr.shape, self.p)]
        sl = tuple(slice(st, st + p) for st, p in zip(starts, self.p))
        return sl

    def __getitem__(self, idx):
        sl = self._crop()
        hr = self.hr[sl]; lr = self.lrint[sl]
        if self.augment:
            for ax in range(3):
                if self.rng.random() < 0.5:
                    hr = np.flip(hr, ax); lr = np.flip(lr, ax)
        hr = np.ascontiguousarray(hr)[None]
        lr = np.ascontiguousarray(lr)[None]
        return torch.from_numpy(lr), torch.from_numpy(hr)


def fixed_val_patches(ds, k=8, seed=123):
    """A fixed set of (lr, hr) val patches for a stable PSNR each epoch."""
    rng = np.random.default_rng(seed)
    saved = ds.rng
    ds.rng = rng
    pairs = [ds[i] for i in range(k)]
    ds.rng = saved
    return pairs


# ----------------------------------------------------------------------------- train
def build_volumes(args):
    print(f"[load] HR <- {args.hr}")
    hr_raw = load_stack(args.hr, max_slices=args.max_slices)
    hr, (lo, hi) = normalize(hr_raw)
    D = hr.shape[0]
    n_val = max(args.patch, int(round(D * args.val_frac)))
    n_val = min(n_val, D // 2)
    train_vol = hr[:D - n_val]
    val_vol = hr[D - n_val:]
    print(f"[split] HR {hr.shape} -> train z=0:{D-n_val}  val z={D-n_val}:{D}")
    return train_vol, val_vol


def run_selftest(device):
    print("[selftest] synthetic volume, tiny model, 2 epochs ...")
    hr = make_porous_phantom(shape=(40, 96, 96))
    hr, _ = normalize(hr)
    tr = DegradationPatchDataset(hr[:30], 2.521, 1.0, 0.016, patch=24, n_patches=8)
    va = DegradationPatchDataset(hr[30:], 2.521, 1.0, 0.016, patch=16, n_patches=4,
                                 augment=False)
    net = SR3DNet(channels=8, n_groups=1, n_blocks=2).to(device)
    crit = SRLoss()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loader = DataLoader(tr, batch_size=2, shuffle=True, num_workers=0)
    vpatch = fixed_val_patches(va, k=2)
    for ep in range(2):
        net.train()
        for lr, hrp in loader:
            lr, hrp = lr.to(device), hrp.to(device)
            pred = net(lr); loss, parts = crit(pred, hrp)
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            ps = np.mean([psnr(net(l.to(device)[None] if l.dim() == 4 else l.to(device)),
                               h.to(device)) for l, h in
                          [(p[0][None], p[1][None]) for p in vpatch]])
        print(f"[selftest] epoch {ep}: loss={loss.item():.4f} "
              f"(l1={parts['l1']:.4f} edge={parts['edge']:.4f}) val_psnr={ps:.2f} dB")
    print(f"[selftest] params={net.param_count():,}  device={device}  -> OK")


def main():
    ap = argparse.ArgumentParser(description="Train open 3D SR model on degradation pairs.")
    ap.add_argument("--hr", help="HR tiff stack (folder or multipage tif).")
    ap.add_argument("--out", default="runs/sr")
    ap.add_argument("--factor", type=float, default=2.521)
    ap.add_argument("--psf-sigma", type=float, default=1.0)
    ap.add_argument("--noise-sigma", type=float, default=0.016)
    ap.add_argument("--max-slices", type=int, default=256, help="HR slices to load (z).")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--iters-per-epoch", type=int, default=200)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True,
                    help="Use channels_last_3d layout (faster tensor-core convs).")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the model (can be faster; may error on Blackwell).")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True            # autotune conv algos for fixed patch size
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    chlast = args.channels_last and device == "cuda"

    if args.selftest:
        run_selftest(device)
        return

    assert args.hr, "provide --hr (or use --selftest)"
    os.makedirs(args.out, exist_ok=True)
    train_vol, val_vol = build_volumes(args)

    train_ds = DegradationPatchDataset(train_vol, args.factor, args.psf_sigma,
                                       args.noise_sigma, args.patch, args.iters_per_epoch * args.batch,
                                       augment=True, seed=args.seed)
    val_ds = DegradationPatchDataset(val_vol, args.factor, args.psf_sigma,
                                     args.noise_sigma, min(args.patch, val_vol.shape[0]),
                                     n_patches=8, augment=False, seed=args.seed + 1)
    val_pairs = fixed_val_patches(val_ds, k=8)
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0,
                        pin_memory=(device == "cuda"))

    net = SR3DNet(args.channels, args.groups, args.blocks).to(device)
    if chlast:
        net = net.to(memory_format=torch.channels_last_3d)
    if args.compile:
        net = torch.compile(net)
    crit = SRLoss()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    json.dump(vars(args), open(os.path.join(args.out, "config.json"), "w"), indent=2)
    log = open(os.path.join(args.out, "training_log.csv"), "w")
    log.write("epoch,train_loss,l1,edge,val_psnr,lr,sec\n")
    print(f"[model] params={net.param_count():,}  device={device}  amp={use_amp}")

    best = -1.0
    for ep in range(args.epochs):
        t0 = time.time(); net.train(); running = 0.0; parts = {"l1": 0, "edge": 0}; nb = 0
        for lr, hrp in tqdm(loader, desc=f"epoch {ep+1}/{args.epochs}"):
            lr, hrp = lr.to(device), hrp.to(device)
            if chlast:
                lr = lr.contiguous(memory_format=torch.channels_last_3d)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = net(lr); loss, p = crit(pred, hrp)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            running += float(loss.detach()); parts["l1"] += p["l1"]; parts["edge"] += p["edge"]; nb += 1
        sched.step()

        net.eval(); ps = []
        with torch.no_grad():
            for l, h in val_pairs:
                l, h = l[None].to(device), h[None].to(device)
                ps.append(psnr(net(l), h))
        vpsnr = float(np.mean(ps))
        dt = time.time() - t0
        print(f"  epoch {ep+1}: loss={running/nb:.4f}  val_psnr={vpsnr:.2f} dB  ({dt:.0f}s)")
        log.write(f"{ep+1},{running/nb:.5f},{parts['l1']/nb:.5f},{parts['edge']/nb:.5f},"
                  f"{vpsnr:.4f},{sched.get_last_lr()[0]:.2e},{dt:.1f}\n"); log.flush()
        if vpsnr > best:
            best = vpsnr
            torch.save({"model": net.state_dict(), "config": vars(args), "val_psnr": best},
                       os.path.join(args.out, "best_model.pt"))
    log.close()
    print(f"[done] best val PSNR={best:.2f} dB -> {os.path.join(args.out, 'best_model.pt')}")


if __name__ == "__main__":
    main()
