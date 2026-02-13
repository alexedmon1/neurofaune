#!/usr/bin/env python3
"""
Generate skull strip omnibus QC reports.

Produces one scrollable HTML report per modality showing ALL subjects' skull
strip mosaics on a single page for rapid screening.

Output: qc/reports/skull_strip_{modality}.html

Usage:
    uv run python scripts/generate_omnibus_qc.py /mnt/arborea/bpa-rat
    uv run python scripts/generate_omnibus_qc.py /mnt/arborea/bpa-rat --modality anat --modality dwi
"""

import argparse
import sys
from pathlib import Path

from neurofaune.preprocess.qc import generate_skull_strip_omnibus

ALL_MODALITIES = ['anat', 'dwi', 'func', 'msme']


def main():
    parser = argparse.ArgumentParser(
        description='Generate skull strip omnibus QC reports',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('study_root', type=Path, help='Study root directory')
    parser.add_argument('--modality', action='append', dest='modalities',
                        choices=ALL_MODALITIES,
                        help='Modality to generate (can specify multiple; default: all)')
    args = parser.parse_args()

    study_root = args.study_root.resolve()
    modalities = args.modalities or ALL_MODALITIES

    if not study_root.exists():
        print(f"Error: study root does not exist: {study_root}")
        sys.exit(1)

    print(f"Study root: {study_root}")
    print(f"Modalities: {', '.join(modalities)}")
    print()

    generated = []
    for modality in modalities:
        print(f"--- {modality.upper()} ---")
        result = generate_skull_strip_omnibus(study_root, modality)
        if result:
            generated.append(result)
        print()

    print(f"Done. Generated {len(generated)} omnibus report(s).")
    for p in generated:
        print(f"  {p}")


if __name__ == '__main__':
    main()
