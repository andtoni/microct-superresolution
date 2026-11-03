# Super-resolution and segmentation for micro-CT of porous scaffolds

Research code for a manuscript in preparation. It tests whether a fast, low-resolution micro-CT scan of
a porous biomaterial scaffold can be computationally restored to the quality of a slow, high-resolution
scan, well enough that the transport-relevant measurements made on it still hold.

> **Public mirror.** This is a sanitised copy of a private research repository. A scripted export
> regenerates it from the private source and removes secrets, personal data and infrastructure details.
> Every update must pass a leak check (a private-string denylist and gitleaks over the full history)
> before it is published. Development happens in the private repository, so this copy takes no pull
> requests. Last synced 3 November 2026.

The manuscript, planning notes and data are not included, and commit messages are reduced to their
subject lines. The code has no automated test suite yet and has only been run on my own scans.

Built with AI coding agents (Claude Code, Codex and Pi) that I direct.

## What is here

| Stage | Files | What it does |
|---|---|---|
| Ingest | `txm_to_slices.py`, `tif_to_h5.py`, `inspect_volume.py` | Read Zeiss TXM reconstructions and convert volumes |
| Registration | `register_volumes.py`, `align_pair.py`, `prep_pair.py`, `localize_hr.py`, `qc_alignment.py` | Rigidly align a low-resolution scan to its high-resolution pair (SimpleITK) and check the overlap |
| Super-resolution | `sr_model.py`, `train_sr.py`, `train_sr_real.py`, `infer_sr.py` | A 3D residual network trained on simulated degradations or on real aligned pairs (PyTorch) |
| Segmentation | `train_seg.py`, `segment.py`, `nnunet_baseline.py`, `run_nnunet_loao.py`, `eval_seg.py`, `seg_qc.py` | A 3D U-Net across fabrication routes, an nnU-Net ResEnc baseline with leave-one-architecture-out runs, and evaluation (Dice, IoU, surface distances, topology error) |
| Morphometry | `morphometry.py`, `compare.py`, `compare_sr.py` | The same pore and transport metrics computed on high-resolution, super-resolved and interpolated volumes, so differences reflect the restoration and not the analysis |
| Reproducibility | `Snakefile`, `provenance.py`, `runlog.py`, `mlflow_setup.py`, `validate_artifacts.py`, `schemas/` | A regeneration workflow, run logging, experiment tracking and schema-checked outputs |

## Setup

```bash
uv sync
uv run python check_gpu.py
```

Training needs a CUDA GPU. Paths to data are passed on the command line or through the environment;
see `paths.py`.

## Related

The pore-network method these measurements build on is published and archived:
[growth-tunnel-analysis](https://github.com/andtoni/growth-tunnel-analysis).
