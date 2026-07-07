#!/usr/bin/env python3
"""Cross-sectional & longitudinal registration-consistency QC.

Warps every registered subject brain into the common atlas (SIGMA) space and reports:
  * cross-sectional Dice — within a timepoint, agreement of subjects (pairwise + vs atlas)
  * longitudinal Dice     — within an animal, agreement of its brain across timepoints

These are group/longitudinal measures that per-registration QC cannot capture; they
answer "do the brains land in the same place on the atlas, within and across timepoints?"

Usage:
    python scripts/registration_consistency_qc.py --config config.yaml
    python scripts/registration_consistency_qc.py <study_root> --atlas-mask <SIGMA_brain_mask.nii>

Requires antsApplyTransforms on PATH.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from neurofaune.templates.consistency_qc import run_consistency_qc


def _resolve_atlas_mask(config: dict) -> Path:
    """Prefer the study-space brain mask; fall back to the raw SIGMA brain mask."""
    atlas = config.get("atlas", {})
    ss = atlas.get("study_space", {})
    if ss.get("brain_mask") and Path(ss["brain_mask"]).exists():
        return Path(ss["brain_mask"])
    base = Path(atlas.get("base_path", ""))
    cand = base / "SIGMA_Rat_Anatomical_Imaging" / "SIGMA_Rat_Anatomical_InVivo_Template" / "SIGMA_InVivo_Brain_Mask.nii"
    if cand.exists():
        return cand
    raise FileNotFoundError("Could not resolve a SIGMA brain mask; pass --atlas-mask explicitly.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("study_root", type=Path, nargs="?", help="study root (contains derivatives/, transforms/, templates/)")
    p.add_argument("--config", type=Path, default=None, help="study config YAML (resolves paths + atlas mask)")
    p.add_argument("--atlas-mask", type=Path, default=None, help="SIGMA brain mask (registration target space)")
    p.add_argument("--output-dir", type=Path, default=None, help="output dir (default: <study_root>/qc/reports/consistency)")
    p.add_argument("--brain-mask-suffix", default="desc-brain_mask", help="brain-mask filename suffix")
    args = p.parse_args()

    config = {}
    if args.config:
        from neurofaune.config import load_config
        config = load_config(args.config)

    study_root = args.study_root or (Path(config["paths"]["study_root"]) if config else None)
    if study_root is None:
        p.error("provide study_root or --config")

    atlas_mask = args.atlas_mask or _resolve_atlas_mask(config)
    output_dir = args.output_dir or (study_root / "qc" / "reports" / "consistency")

    res = run_consistency_qc(
        derivatives_dir=study_root / "derivatives",
        transforms_dir=study_root / "transforms",
        templates_dir=study_root / "templates",
        atlas_mask=atlas_mask,
        output_dir=output_dir,
        brain_mask_suffix=args.brain_mask_suffix,
    )

    print(f"\nWarped {res['n_warped']} subject-sessions into atlas space.\n")
    print("=== Cross-sectional Dice (within timepoint) ===")
    print(res["cross_sectional"].to_string(index=False))
    if res["longitudinal_by_pair"] is not None:
        print("\n=== Longitudinal Dice (within animal, between timepoints) ===")
        print(res["longitudinal_by_pair"].round(3).to_string())
    print(f"\nCSVs written under: {output_dir}")


if __name__ == "__main__":
    main()
