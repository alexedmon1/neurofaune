#!/bin/bash
# Helper script to run skull stripping parameter sweep on test subject

# Configuration
SUBJECT="sub-Rat108"
SESSION="ses-p30"
BOLD_FILE="/mnt/arborea/bpa-rat/raw/bids/${SUBJECT}/${SESSION}/func/${SUBJECT}_${SESSION}_run-13_bold.nii.gz"
OUTPUT_DIR="/mnt/arborea/bpa-rat/test/skull_strip_optimization/${SUBJECT}_${SESSION}"
REF_METHOD="middle"  # Options: "mean" (temporal average), "middle" (middle volume), or volume index

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "========================================="
echo "Functional Skull Stripping Optimization"
echo "========================================="
echo "Subject: ${SUBJECT}"
echo "Session: ${SESSION}"
echo "BOLD file: ${BOLD_FILE}"
echo "Output: ${OUTPUT_DIR}"
echo ""

# Check if BOLD file exists
if [ ! -f "${BOLD_FILE}" ]; then
    echo "ERROR: BOLD file not found: ${BOLD_FILE}"
    echo ""
    echo "Available BOLD files for ${SUBJECT}:"
    find /mnt/arborea/bpa-rat/raw/bids/${SUBJECT} -name "*bold.nii.gz" 2>/dev/null
    exit 1
fi

# Run parameter sweep
echo "Starting parameter sweep..."
echo ""

uv run python /home/edm9fd/sandbox/neurofaune/scripts/optimize_func_skull_strip.py \
    --subject "${SUBJECT}" \
    --session "${SESSION}" \
    --bold-file "${BOLD_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --ref-method "${REF_METHOD}"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "SUCCESS!"
    echo "========================================="
    echo "Results saved to: ${OUTPUT_DIR}"
    echo ""
    echo "Review the comparison figure:"
    echo "  ${OUTPUT_DIR}/${SUBJECT}_${SESSION}_skull_strip_comparison.png"
    echo ""
    echo "View results JSON:"
    echo "  ${OUTPUT_DIR}/results.json"
else
    echo ""
    echo "========================================="
    echo "FAILED!"
    echo "========================================="
    echo "Check the output above for errors."
fi
