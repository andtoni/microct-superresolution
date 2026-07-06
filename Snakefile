# Snakefile -- regeneration DAG for rapid-scaffold-sr (Manuscript OS).
# `uv run snakemake -n` = the dry-run frontier (consumed by reconcile.py -> STATE.md,
# and usable as a no-GPU CI smoke check that every target resolves with no cycle).
# Add one rule per figure script as it is built; rules carry params(seed=...) so a
# parameter change retriggers regeneration. Heavy GPU steps (training/inference) run
# from the main session, not the DAG -- the DAG wires the deterministic post-steps.
import os

DATA = os.environ.get(
    "SCAFFX_DATA_ROOT",
    os.environ.get(
        "SCAFFOLD_OUT",
        "/data/superresolution/Data/codeoutput",
    ),
)


rule all:
    input:
        f"{DATA}/data/.validated"


# Provenance/contract gate: re-validate whenever results change.
rule validate:
    output:
        touch(f"{DATA}/data/.validated")
    shell:
        f"uv run python validate_artifacts.py --data-root '{DATA}'"


# --- Figure rule template (copy + adapt as each figure script lands) -----------
# rule figure_F7_morphometry:
#     input:
#         seg   = f"{DATA}/segmentations",
#         model = f"{DATA}/runs/seg_generalist_v2_all/best.pt",
#     output:
#         directory(f"{DATA}/figures/F7_morphometry"),
#     params:
#         seed = 42,
#     shell:
#         "uv run python make_F7.py --seed {params.seed}"
