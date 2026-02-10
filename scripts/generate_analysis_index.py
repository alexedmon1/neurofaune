#!/usr/bin/env python3
"""
Generate (or regenerate) the unified analysis index dashboard.

Optionally backfills the registry by scanning for existing summary JSONs
before generating the HTML.

Usage:
    # Backfill existing results and generate index
    uv run python scripts/generate_analysis_index.py \
        --analysis-root /mnt/arborea/bpa-rat/analysis \
        --study-name "BPA Rat Study" \
        --backfill

    # Regenerate index from existing registry only
    uv run python scripts/generate_analysis_index.py \
        --analysis-root /mnt/arborea/bpa-rat/analysis

    # List current registry entries
    uv run python scripts/generate_analysis_index.py \
        --analysis-root /mnt/arborea/bpa-rat/analysis \
        --list
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurofaune.analysis.reporting import (
    backfill_registry,
    generate_index_html,
    list_entries,
    load_registry,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate unified analysis index dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--analysis-root",
        type=Path,
        required=True,
        help="Root directory of analysis outputs",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="",
        help="Study name for the dashboard header",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Scan for existing summary JSONs and add to registry",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_entries",
        help="List current registry entries and exit",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override output path for index.html",
    )

    args = parser.parse_args()
    analysis_root = args.analysis_root.resolve()

    if not analysis_root.exists():
        logger.error("Analysis root does not exist: %s", analysis_root)
        sys.exit(1)

    # List mode
    if args.list_entries:
        entries = list_entries(analysis_root)
        if not entries:
            print("No entries in registry.")
        else:
            print(f"{'ID':<35} {'Type':<18} {'Status':<12} {'Timestamp'}")
            print("-" * 90)
            for e in entries:
                print(
                    f"{e['entry_id']:<35} {e['analysis_type']:<18} "
                    f"{e['status']:<12} {e.get('timestamp', '')}"
                )
        return

    # Backfill
    if args.backfill:
        logger.info("Scanning for existing analysis summaries...")
        n = backfill_registry(
            analysis_root,
            study_name=args.study_name,
            auto_generate_index=False,
        )
        logger.info("Backfill: %d new entries discovered", n)

    # Update study name if provided
    if args.study_name:
        from neurofaune.analysis.reporting.registry import save_registry

        reg = load_registry(analysis_root)
        reg["study_name"] = args.study_name
        save_registry(analysis_root, reg)

    # Generate index
    path = generate_index_html(analysis_root, output_path=args.output)
    logger.info("Dashboard written to: %s", path)


if __name__ == "__main__":
    main()
