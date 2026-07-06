#!/usr/bin/env python3
r"""
register_volumes.py
===================
Rigidly register a 'moving' volume onto a 'fixed' volume with SimpleITK, resample the
moving volume onto the fixed grid, and emit the overlap mask. This is the alignment step
that lets us compare an independent HR scan (hr_vcr298) against an SR output on the *same*
physical voxels.

Method: multi-resolution rigid (Euler3D) registration, Mattes mutual information (robust to
intensity differences between two separate scans), regular-step gradient descent, with
optimizer scales from physical shift. Spacing is set per volume so registration happens in
physical space — so it also works across resolutions (HR vs LR) if needed.

Convention: register MOVING -> FIXED. For the comparison, fixed = open-SR (HR grid),
moving = hr_vcr298. The resampled HR then lives on the SR grid; the mask marks where they
overlap, and metrics are computed there.

Usage:
    uv run python register_volumes.py ^
      --fixed infer\vcr298\open_SR.tif --moving "...\Data\ct\hr_vcr298" ^
      --fixed-spacing 0.9696 --moving-spacing 0.9696 ^
      --out reg\vcr298

Verify the engine on synthetic data (known transform):
    uv run python register_volumes.py --selftest
"""
from __future__ import annotations
import argparse, os
import numpy as np
import SimpleITK as sitk

from sr_validation_harness import load_stack, normalize, save_stack, make_porous_phantom


def to_sitk(arr, spacing):
    img = sitk.GetImageFromArray(np.ascontiguousarray(arr, dtype=np.float32))
    img.SetSpacing((float(spacing),) * 3)
    return img


def register(fixed_img, moving_img, exhaustive=False, iters=200, sampling=0.05,
             init_transform=None):
    """Rigid multi-resolution registration; returns (transform, final_metric).

    `sampling` is the fraction of fixed-image voxels used per metric evaluation.
    For very large volumes (>1e9 voxels) use a small value (e.g. 0.01); the
    registration only needs to refine a small residual when the inputs already
    start nearly aligned, so a few million samples is plenty.

    `init_transform` optionally seeds the optimizer (e.g. with a known in-plane
    rotation from the localizer). If None, centres are aligned geometrically.
    """
    if init_transform is None:
        initializer = sitk.CenteredTransformInitializer(
            fixed_img, moving_img, sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY)
    else:
        initializer = init_transform

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(sampling, sitk.sitkWallClock)
    reg.SetInterpolator(sitk.sitkLinear)

    if exhaustive:
        # coarse translation/rotation search to seed the optimizer (use when the moving
        # volume may sit far from the fixed centre, e.g. a window inside a big volume)
        reg.SetOptimizerAsExhaustive([2, 2, 2, 4, 4, 4])
        reg.SetOptimizerScales([0.3, 0.3, 0.3, 10.0, 10.0, 10.0])
        reg.SetInitialTransform(initializer, inPlace=True)
        initializer = reg.Execute(fixed_img, moving_img)
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(sampling, sitk.sitkWallClock)
        reg.SetInterpolator(sitk.sitkLinear)

    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-4, numberOfIterations=iters,
        gradientMagnitudeTolerance=1e-6)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(initializer, inPlace=False)

    final = reg.Execute(fixed_img, moving_img)
    return final, reg.GetMetricValue()


def resample_onto(fixed_img, moving_img, transform):
    """Resample moving into the fixed grid; also return the valid-overlap mask."""
    moved = sitk.Resample(moving_img, fixed_img, transform, sitk.sitkLinear, 0.0,
                          moving_img.GetPixelID())
    ones = sitk.Image(moving_img.GetSize(), sitk.sitkFloat32)
    ones.CopyInformation(moving_img)
    ones = ones + 1.0
    mask = sitk.Resample(ones, fixed_img, transform, sitk.sitkNearestNeighbor, 0.0,
                        sitk.sitkFloat32)
    return moved, mask


def run_selftest():
    print("[selftest] phantom + known rigid transform ...")
    vol, _ = normalize(make_porous_phantom(shape=(80, 80, 80)))
    fixed = to_sitk(vol, 1.0)

    true_t = sitk.Euler3DTransform()
    true_t.SetCenter(fixed.TransformContinuousIndexToPhysicalPoint([40, 40, 40]))
    true_t.SetRotation(0.05, -0.04, 0.06)            # ~3 deg each
    true_t.SetTranslation((3.0, -2.0, 2.5))          # voxels (spacing=1)
    moving = sitk.Resample(fixed, fixed, true_t, sitk.sitkLinear, 0.0)

    def corr(a, b):
        a = sitk.GetArrayFromImage(a).ravel(); b = sitk.GetArrayFromImage(b).ravel()
        return float(np.corrcoef(a, b)[0, 1])

    before = corr(fixed, moving)
    tfm, metric = register(fixed, moving, iters=300)
    moved, mask = resample_onto(fixed, moving, tfm)
    after = corr(fixed, moved)
    print(f"[selftest] correlation  before={before:.3f}  after={after:.3f}  "
          f"(MI metric={metric:.4f})")
    print(f"[selftest] {'OK - alignment recovered' if after > before + 0.1 else 'WEAK - check'}")


def main():
    ap = argparse.ArgumentParser(description="Rigid registration of moving -> fixed (SimpleITK).")
    ap.add_argument("--fixed", help="Fixed volume (e.g. open_SR.tif).")
    ap.add_argument("--moving", help="Moving volume (e.g. hr_vcr298).")
    ap.add_argument("--fixed-spacing", type=float, default=0.9696)
    ap.add_argument("--moving-spacing", type=float, default=0.9696)
    ap.add_argument("--exhaustive", action="store_true",
                    help="Coarse search init (moving may be far from fixed centre).")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--out", default="reg/out")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        run_selftest()
        return

    assert args.fixed and args.moving, "provide --fixed and --moving (or --selftest)"
    os.makedirs(args.out, exist_ok=True)

    print(f"[load] fixed  <- {args.fixed}")
    fx, _ = normalize(load_stack(args.fixed))
    print(f"[load] moving <- {args.moving}")
    mv, _ = normalize(load_stack(args.moving))
    fixed = to_sitk(fx, args.fixed_spacing)
    moving = to_sitk(mv, args.moving_spacing)
    print(f"[reg] fixed {fx.shape}@{args.fixed_spacing}um  moving {mv.shape}@{args.moving_spacing}um")

    tfm, metric = register(fixed, moving, exhaustive=args.exhaustive, iters=args.iters)
    moved, mask = resample_onto(fixed, moving, tfm)
    print(f"[reg] final MI metric = {metric:.4f}")
    print(f"[reg] transform params = {[round(p,4) for p in tfm.GetParameters()]}")

    sitk.WriteTransform(tfm, os.path.join(args.out, "moving_to_fixed.tfm"))
    save_stack(sitk.GetArrayFromImage(moved), os.path.join(args.out, "moving_resampled.tif"))
    save_stack(sitk.GetArrayFromImage(mask), os.path.join(args.out, "overlap_mask.tif"))
    frac = float(sitk.GetArrayFromImage(mask).mean()) * 100
    print(f"[out] resampled moving + mask -> {args.out}  (overlap {frac:.1f}% of fixed)")


if __name__ == "__main__":
    main()
