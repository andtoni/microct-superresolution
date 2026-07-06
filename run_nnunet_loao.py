#!/usr/bin/env python3
"""run_nnunet_loao.py -- overnight orchestrator for the nnU-Net ResEnc-L LOAO baseline.

This is the reviewer-standard *learned* baseline (nnU-Net "Revisited" ResEnc) placed
opposite our artifact-aware MONAI residual 3D U-Net. Five leave-one-architecture-out
folds: each trains on four architectures and predicts the held-out one, so the held-out
prediction is genuinely out-of-distribution. nnU-Net and MONAI are later scored through
the *same* eval_seg.py against the same Tier-1 (pooled-Ilastik) reference -> apples to
apples (F4/F-bench panel).

Every phase (prep / preprocess / train / predict) is wrapped in the runlog ledger
(RUNS.jsonl) so a dead session can be reconciled, and STATE.md is regenerated at the end.
The script is idempotent: completed phases are skipped, a half-trained fold resumes with
nnU-Net's `--c`, so re-running after a crash continues rather than restarts.

Usage (from repo root, inside the uv env):
    uv run python run_nnunet_loao.py                 # all 5 folds, prep->predict
    uv run python run_nnunet_loao.py --dry-run       # print the plan, do nothing
    uv run python run_nnunet_loao.py --folds pel16 vcr276
    uv run python run_nnunet_loao.py --no-predict    # stop after training
"""
from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import paths
import runlog
import nnunet_baseline as nb

# Held-out architecture -> nnU-Net dataset id (011..015), in plan order.
LOAO = [("pel16", 11), ("pel18", 12), ("pel20", 13), ("laser50", 14), ("vcr276", 15)]
DATASET_NAME = "Scaffold"
CONFIG = "3d_fullres"
PRESET = "L"
PLANNER = "nnUNetPlannerResEncL"
PLANS = "nnUNetResEncUNetLPlans"
DEFAULT_TRAINER = "nnUNetTrainer_250epochs"   # plan-locked 250-epoch budget (not nnU-Net's 1000 default)

# nnU-Net working roots (local scratch, DATA_TAXONOMY D11 exception). Overridable via env.
NNUNET_RAW = Path(os.environ.get("nnUNet_raw", r"C:\nnunet\raw"))
NNUNET_PRE = Path(os.environ.get("nnUNet_preprocessed", r"C:\nnunet\preprocessed"))
NNUNET_RES = Path(os.environ.get("nnUNet_results", r"C:\nnunet\results"))

PRED_ROOT = paths.RUNS / "nnunet_loao"        # held-out predictions land here for scoring


def _did(dataset_id: int) -> str:
    return f"Dataset{dataset_id:03d}_{DATASET_NAME}"


def _check_trainer(trainer: str) -> None:
    """Abort early if the requested epoch-budget trainer is not importable -- bare
    nnUNetv2_train would silently default to 1000 epochs (days, not an overnight run)."""
    import importlib
    mod = "nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs"
    try:
        m = importlib.import_module(mod)
    except Exception as e:
        sys.exit(f"[guard] cannot import nnU-Net trainer variants ({mod}): {e}\n"
                 f"        nnunetv2 may not be installed in this env -- run `uv sync`.")
    if not hasattr(m, trainer):
        avail = sorted(a for a in dir(m) if a.startswith("nnUNetTrainer_") and a.endswith("epochs"))
        sys.exit(f"[guard] trainer '{trainer}' not found. Available epoch trainers: {avail}\n"
                 f"        pass one with --trainer, e.g. --trainer {avail[0] if avail else 'nnUNetTrainer_250epochs'}")
    print(f"[guard] trainer '{trainer}' OK; plans '{PLANS}', config '{CONFIG}'.")


def _exe(name: str) -> str:
    p = shutil.which(name)
    if not p:
        sys.exit(f"[guard] console script '{name}' not on PATH -- is the nnU-Net uv env active?")
    return p


def _env() -> dict:
    e = dict(os.environ)
    e["nnUNet_raw"], e["nnUNet_preprocessed"], e["nnUNet_results"] = (
        str(NNUNET_RAW), str(NNUNET_PRE), str(NNUNET_RES))
    return e


def _cmd(name: str, target: str, params: dict, argv: list[str], *, dry: bool) -> None:
    """Run one subprocess phase wrapped in the runlog ledger."""
    pretty = " ".join(argv)
    print(f"\n>>> [{name}] {target}\n    {pretty}")
    if dry:
        return
    with runlog.run(name, target=target, params={**params, "cmd": pretty}) as r:
        r.note(phase=name, dataset=target)
        subprocess.run(argv, check=True, env=_env())


def _write_test_image(arch: str, dataset_id: int, *, dry: bool) -> None:
    """Write the held-out architecture's HR volume to imagesTs/ for prediction."""
    dst = NNUNET_RAW / _did(dataset_id) / "imagesTs" / f"{arch}_0000.nii.gz"
    if dst.exists():
        print(f"    [skip] test image exists: {dst}")
        return
    import morphometry as mp
    src = None
    for sub in ("HR", "hr", "high", ""):
        cand = paths.ALIGNED / arch / sub if sub else paths.ALIGNED / arch
        if cand.exists():
            src = mp.resolve_volume(str(cand))
            break
    if src is None:
        sys.exit(f"[guard] could not resolve held-out HR volume for '{arch}' under {paths.ALIGNED/arch}")
    print(f">>> [test-image] {arch} <- {src}")
    if dry:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np
    nb._write_nii(nb._load(src).astype(np.float32), dst)


def _trained(dataset_id: int, trainer: str) -> Path:
    return (NNUNET_RES / _did(dataset_id) /
            f"{trainer}__{PLANS}__{CONFIG}" / "fold_all" / "checkpoint_final.pth")


def do_fold(arch: str, dataset_id: int, trainer: str, *, predict: bool, dry: bool) -> None:
    did = _did(dataset_id)
    print(f"\n========== FOLD  hold-out={arch}  -> {did} ==========")

    # 1) prep training dataset (4 architectures, held-out one excluded)
    if (NNUNET_RAW / did / "dataset.json").exists():
        print(f"    [skip] prep: {NNUNET_RAW/did/'dataset.json'} exists")
    else:
        print(f">>> [prep] building {did} (exclude {arch})")
        if not dry:
            with runlog.run("nnunet_prep", target=did, params={"exclude": arch}) as r:
                rc = nb.main(["--raw-root", str(paths.ALIGNED),
                              "--label-root", str(paths.SEGMENTATIONS),
                              "--exclude", arch, "--dataset-id", str(dataset_id),
                              "--dataset-name", DATASET_NAME,
                              "--nnunet-raw", str(NNUNET_RAW), "--preset", PRESET])
                if rc:
                    raise RuntimeError(f"nnunet_baseline prep returned {rc}")
                r.note(phase="prep", dataset=did)

    # 1b) held-out HR -> imagesTs for later prediction
    _write_test_image(arch, dataset_id, dry=dry)

    # 2) plan + preprocess (ResEnc-L plans)
    if (NNUNET_PRE / did / f"{PLANS}.json").exists():
        print(f"    [skip] preprocess: {PLANS}.json exists for {did}")
    else:
        _cmd("nnunet_preprocess", did, {"planner": PLANNER},
             [_exe("nnUNetv2_plan_and_preprocess"), "-d", str(dataset_id),
              "-pl", PLANNER, "--verify_dataset_integrity"], dry=dry)

    # 3) train fold_all (resume with --c if a checkpoint exists)
    final = _trained(dataset_id, trainer)
    if final.exists():
        print(f"    [skip] train: {final} exists")
    else:
        train_argv = [_exe("nnUNetv2_train"), str(dataset_id), CONFIG, "all",
                      "-p", PLANS, "-tr", trainer]
        latest = final.with_name("checkpoint_latest.pth")
        if latest.exists():
            train_argv.append("--c")
            print(f"    [resume] checkpoint_latest found -> continuing training")
        _cmd("nnunet_train", did, {"trainer": trainer, "plans": PLANS, "config": CONFIG},
             train_argv, dry=dry)

    # 4) predict the held-out architecture (out-of-distribution test)
    if predict:
        out = PRED_ROOT / arch
        if (out / f"{arch}.nii.gz").exists():
            print(f"    [skip] predict: {out/(arch+'.nii.gz')} exists")
        else:
            if not dry:
                out.mkdir(parents=True, exist_ok=True)
            _cmd("nnunet_predict", did, {"holdout": arch, "out": str(out)},
                 [_exe("nnUNetv2_predict"), "-d", str(dataset_id),
                  "-i", str(NNUNET_RAW / did / "imagesTs"), "-o", str(out),
                  "-c", CONFIG, "-p", PLANS, "-tr", trainer, "-f", "all"], dry=dry)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="overnight nnU-Net ResEnc-L LOAO baseline")
    ap.add_argument("--folds", nargs="*", default=None,
                    help="subset of held-out architectures (default: all 5)")
    ap.add_argument("--trainer", default=DEFAULT_TRAINER,
                    help="nnU-Net epoch-budget trainer (default: 250-epoch, plan-locked)")
    ap.add_argument("--no-predict", action="store_true", help="stop after training")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; execute nothing")
    args = ap.parse_args(argv)

    folds = LOAO if not args.folds else [(a, d) for a, d in LOAO if a in set(args.folds)]
    if not folds:
        sys.exit(f"[guard] no matching folds in {args.folds}; valid: {[a for a, _ in LOAO]}")

    print(f"nnU-Net ResEnc-{PRESET} LOAO baseline | folds={[a for a, _ in folds]} | "
          f"trainer={args.trainer} | predict={not args.no_predict} | dry={args.dry_run}")
    print(f"  nnUNet_raw={NNUNET_RAW}\n  nnUNet_preprocessed={NNUNET_PRE}\n  nnUNet_results={NNUNET_RES}")
    print(f"  predictions -> {PRED_ROOT}")

    if not args.dry_run:
        _check_trainer(args.trainer)
        for p in (NNUNET_RAW, NNUNET_PRE, NNUNET_RES, PRED_ROOT):
            p.mkdir(parents=True, exist_ok=True)

    for arch, dataset_id in folds:
        do_fold(arch, dataset_id, args.trainer, predict=not args.no_predict, dry=args.dry_run)

    # regenerate STATE.md from the ledger so a fresh session can resume with no memory
    if not args.dry_run:
        try:
            import reconcile
            reconcile.build()
            print("\n[state] STATE.md regenerated from RUNS.jsonl")
        except Exception as e:
            print(f"\n[state] reconcile skipped: {e}")

    print("\nDONE. Score nnU-Net vs MONAI in the morning, e.g.:")
    print(f'  uv run python eval_seg.py --raw-root "{paths.ALIGNED}" '
          f'--label-root "{paths.SEGMENTATIONS}" --pred-root "{PRED_ROOT}" '
          f'--out "{paths.OUTPUT_ROOT / "figures" / "F4_loao"}"   # confirm eval_seg flags')
    return 0


if __name__ == "__main__":
    sys.exit(main())
