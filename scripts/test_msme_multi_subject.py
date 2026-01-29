#!/usr/bin/env python3
"""
Test MSME adaptive skull stripping on multiple subjects to verify generalization.

Tests across different age cohorts (p30, p60, p90) and different subjects.
"""

import sys
from pathlib import Path

# Add neurofaune to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neurofaune.config import load_config
from neurofaune.preprocess.workflows.msme_preprocess import run_msme_preprocessing
from neurofaune.utils.transforms import create_transform_registry


def main():
    """Test MSME preprocessing on multiple subjects."""

    # Test subjects across different cohorts
    test_subjects = [
        ("sub-Rat111", "ses-p90"),  # Different p90 subject
        ("sub-Rat102", "ses-p60"),  # p60 cohort
        ("sub-Rat141", "ses-p30"),  # p30 cohort (younger rats)
        ("sub-Rat050", "ses-unknown"),  # Unknown session (may have different acquisition)
    ]

    # Load config
    config_path = Path("/home/edm9fd/sandbox/neurofaune/configs/bpa_rat_example.yaml")
    config = load_config(config_path)

    # Base paths
    bids_root = Path("/mnt/arborea/bpa-rat/raw/bids")
    derivatives = Path("/mnt/arborea/bpa-rat/derivatives")

    results = []

    for subject, session in test_subjects:
        print("\n" + "="*80)
        print(f"Testing: {subject} {session}")
        print("="*80)

        # Find MSME file
        msme_dir = bids_root / subject / session / "msme"
        msme_files = list(msme_dir.glob("*MSME.nii.gz"))

        if not msme_files:
            print(f"  ✗ No MSME file found")
            results.append((subject, session, "NO_MSME", {}))
            continue

        msme_file = msme_files[0]

        # Find T2w file
        t2w_file = derivatives / subject / session / "anat" / f"{subject}_{session}_desc-skullstrip_T2w.nii.gz"

        if not t2w_file.exists():
            print(f"  ✗ No T2w file found")
            results.append((subject, session, "NO_T2W", {}))
            continue

        print(f"  MSME: {msme_file.name}")
        print(f"  T2w: {t2w_file.name}")

        # Use separate work directory per subject
        work_dir = Path(f"/mnt/arborea/bpa-rat/work/{subject}/{session}/msme_multi_test")

        # Determine cohort from session
        cohort = session.replace("ses-", "") if session.startswith("ses-") else "unknown"

        # Create transform registry for this subject
        transform_registry = create_transform_registry(
            config=config,
            subject=subject,
            cohort=cohort
        )

        try:
            # Run preprocessing
            outputs = run_msme_preprocessing(
                config=config,
                subject=subject,
                session=session,
                msme_file=msme_file,
                t2w_file=t2w_file,
                output_dir=Path("/mnt/arborea/bpa-rat"),
                work_dir=work_dir,
                transform_registry=transform_registry
            )

            results.append((subject, session, "SUCCESS", outputs))
            print(f"\n  ✓ {subject} {session} completed successfully")

        except Exception as e:
            print(f"\n  ✗ {subject} {session} FAILED: {e}")
            results.append((subject, session, "FAILED", {"error": str(e)}))

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    success_count = 0
    for subject, session, status, outputs in results:
        if status == "SUCCESS":
            success_count += 1
            reg_info = outputs.get("registration", {})
            if isinstance(reg_info, dict) and "affine_transform" in reg_info:
                print(f"  ✓ {subject} {session}: Complete with registration")
            else:
                print(f"  ✓ {subject} {session}: Complete (no registration info)")
        else:
            print(f"  ✗ {subject} {session}: {status}")

    print(f"\n{success_count}/{len(test_subjects)} subjects completed successfully")


if __name__ == "__main__":
    main()
