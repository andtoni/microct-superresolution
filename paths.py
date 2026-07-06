"""Single source of truth for superresolution output locations.

Override the root with ``SCAFFOLD_OUT`` if needed. By default, derived data lives in the
clean material folder under Nextcloud, while code lives in ``~/coding/papers/superresolution``.
"""
from __future__ import annotations
import os
from pathlib import Path

PROJECT_MATERIAL = Path(os.environ.get(
    "SUPERRESOLUTION_PROJECT_DIR",
    "/data/superresolution",
))
OUTPUT_ROOT = Path(os.environ.get(
    "SCAFFOLD_OUT",
    str(PROJECT_MATERIAL / "Data" / "codeoutput"),
))
CT_ROOT = Path(os.environ.get("SCAFFOLD_CT_ROOT", str(PROJECT_MATERIAL / "Data" / "ct")))

ALIGNED       = OUTPUT_ROOT / "aligned"
RUNS          = OUTPUT_ROOT / "runs"
SEGMENTATIONS = OUTPUT_ROOT / "segmentations"
NETWORKS      = OUTPUT_ROOT / "networks"
DATA          = OUTPUT_ROOT / "data"
FIGURES       = OUTPUT_ROOT / "figures"
LOGS          = OUTPUT_ROOT / "logs"

FIGURE_DIRS = [
    "F1_concept", "F2_degradation", "F3_hallucination", "F4_loao",
    "F5_equivalence", "F6_breakdown", "F7_morphometry", "F8_resolution_map",
    "F9_structure_transport", "F10_whole_device", "F11_scaling",
    "F12_objective_sweep",
    "S1_frc", "S2_training", "S3_interp_error", "S4_ablations",
    "S5_registration_qc", "S6_segmentation", "S7_pore_size_cv", "S8_uncertainty",
]


def fig_dir(name: str) -> Path:
    """Return (creating if needed) a figure subfolder by name."""
    d = FIGURES / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_tree() -> str:
    """Create the full output directory layout. Idempotent."""
    for d in (ALIGNED, RUNS, SEGMENTATIONS, NETWORKS, DATA, FIGURES, LOGS):
        d.mkdir(parents=True, exist_ok=True)
    for f in FIGURE_DIRS:
        (FIGURES / f).mkdir(parents=True, exist_ok=True)
    return str(OUTPUT_ROOT)
