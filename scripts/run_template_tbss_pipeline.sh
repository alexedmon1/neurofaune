#!/usr/bin/env bash
#
# Template-Space TBSS Pipeline
#
# Runs TBSS entirely in per-cohort template space (avoids template→SIGMA
# z-alignment errors).  Each cohort is processed independently:
#   1. Prepare template-space data (warp atlas + metrics)
#   2. Generate design matrices (per_pnd only, no pooled)
#   3. Run randomise with cluster reporting
#
# Usage:
#   bash scripts/run_template_tbss_pipeline.sh
#
#   # Single cohort
#   COHORTS="p90" bash scripts/run_template_tbss_pipeline.sh
#
#   # Custom output
#   OUTPUT=/mnt/arborea/bpa-rat/analysis/tbss_tpl bash scripts/run_template_tbss_pipeline.sh
#
set -euo pipefail

STUDY_ROOT="${STUDY_ROOT:-/mnt/arborea/bpa-rat}"
OUTPUT="${OUTPUT:-${STUDY_ROOT}/analysis/tbss_template}"
CONFIG="${CONFIG:-${STUDY_ROOT}/configs/bpa_rat_example.yaml}"
TRACKER="${TRACKER:-${STUDY_ROOT}/study_tracker_combined.csv}"
COHORTS="${COHORTS:-p30 p60 p90}"
METRICS="${METRICS:-FA MD AD RD}"
N_PERM="${N_PERM:-5000}"
MIN_COVERAGE="${MIN_COVERAGE:-0.75}"

echo "================================================================"
echo "Template-Space TBSS Pipeline"
echo "================================================================"
echo "Study root:   ${STUDY_ROOT}"
echo "Output:       ${OUTPUT}"
echo "Cohorts:      ${COHORTS}"
echo "Metrics:      ${METRICS}"
echo "Permutations: ${N_PERM}"
echo "Min coverage: ${MIN_COVERAGE}"
echo ""

for cohort in ${COHORTS}; do
    COHORT_DIR="${OUTPUT}/${cohort}"

    echo ""
    echo "================================================================"
    echo "  COHORT: ${cohort}"
    echo "================================================================"

    # ── Step 1: Prepare template-space data ──────────────────────────
    echo ""
    echo "--- Step 1: Prepare template-space TBSS data ---"
    PYTHONUNBUFFERED=1 uv run python scripts/run_template_tbss_prepare.py \
        --study-root "${STUDY_ROOT}" \
        --output-dir "${COHORT_DIR}" \
        --cohort "${cohort}" \
        --metrics ${METRICS} \
        --min-coverage "${MIN_COVERAGE}" \
        2>&1 | tee "${COHORT_DIR}/logs/prepare_$(date +%Y%m%d_%H%M%S).log"

    # ── Step 2: Generate design matrices ─────────────────────────────
    echo ""
    echo "--- Step 2: Generate design matrices ---"
    PYTHONUNBUFFERED=1 uv run python scripts/prepare_tbss_designs.py \
        --study-tracker "${TRACKER}" \
        --tbss-dir "${COHORT_DIR}" \
        --output-dir "${COHORT_DIR}/designs" \
        2>&1 | tee "${COHORT_DIR}/logs/designs_$(date +%Y%m%d_%H%M%S).log"

    # ── Step 3: Run randomise ────────────────────────────────────────
    echo ""
    echo "--- Step 3: Run randomise ---"
    ANALYSIS="per_pnd_${cohort}"
    PARCELLATION="${COHORT_DIR}/atlas/parcellation.nii.gz"

    PYTHONUNBUFFERED=1 uv run python scripts/run_tbss_analysis.py \
        --tbss-dir "${COHORT_DIR}" \
        --config "${CONFIG}" \
        --analyses "${ANALYSIS}" \
        --metrics ${METRICS} \
        --n-permutations "${N_PERM}" \
        --parcellation "${PARCELLATION}" \
        2>&1 | tee "${COHORT_DIR}/logs/randomise_$(date +%Y%m%d_%H%M%S).log"

    echo ""
    echo "  ${cohort} complete: ${COHORT_DIR}"
done

echo ""
echo "================================================================"
echo "ALL COHORTS COMPLETE"
echo "================================================================"
echo "Results: ${OUTPUT}/{p30,p60,p90}/randomise/"
