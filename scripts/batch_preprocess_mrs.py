#!/usr/bin/env python3
"""
Batch single-voxel spectroscopy processing (Bruker PRESS -> FSL-MRS).

Unlike the other modalities this script reads the raw Bruker tree directly
rather than a BIDS directory: spectroscopy is not converted during BIDS-ification
because spec2nii cannot read ParaVision 360.3 SVS data (see
neurofaune.preprocess.utils.mrs.bruker_mrs).

Usage
-----
    # Inventory only -- what PRESS scans exist, and which would be used
    uv run python scripts/batch_preprocess_mrs.py /mnt/arborea/bruker/cpz \\
        /path/to/study --config config.yaml --dry-run

    # Process everything, measuring tissue fractions where anat is available
    uv run python scripts/batch_preprocess_mrs.py /mnt/arborea/bruker/cpz \\
        /path/to/study --config config.yaml \\
        --basis /path/to/basis/gamma_press_te20_7t_v1 --n-jobs 4
"""

import argparse
import json
import logging
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from neurofaune.config import load_config
from neurofaune.preprocess.utils.mrs.bruker_mrs import find_press_scans
from neurofaune.preprocess.utils.mrs.bruker_params import read_scan_params
from neurofaune.preprocess.workflows.mrs_preprocess import run_mrs_preprocessing

logger = logging.getLogger('batch_mrs')

# Bruker session directory names look like
#   IRC1200_Cuprizone_CageCPZ1_Rat1Y_2__Cage_CPZ1__Rat_1Y_1_2_20260408_140507
# from which the subject is 'Rat1Y' and the session '2'. The optional trailing
# letter marks a repeat scan of the same timepoint (e.g. '1a').
SESSION_NAME_RE = re.compile(r'_(Rat[0-9]+[A-Za-z])_([0-9]+[a-z]?)__')


def parse_session_name(name: str) -> Optional[Dict[str, str]]:
    """Extract BIDS-style subject and session labels from a Bruker directory name."""
    match = SESSION_NAME_RE.search(name)
    if match is None:
        return None
    # Rat IDs are inconsistently cased in the scanner logs ('Rat4y' vs 'Rat4Y').
    subject = 'Rat' + match.group(1)[3:].upper()
    return {'subject': f'sub-{subject}', 'session': f'ses-{match.group(2)}'}


def find_anat_scan(session_dir: Path) -> Optional[str]:
    """Find the highest-resolution multi-slice RARE scan in a session.

    That is the anatomical the SVS voxel is localised against. Scans with only
    a handful of slices are scout/localiser acquisitions and are skipped.
    """
    best: Optional[tuple] = None
    for scan_dir in sorted(session_dir.iterdir()):
        if not scan_dir.is_dir() or not scan_dir.name.isdigit():
            continue
        if not (scan_dir / 'method').exists():
            continue
        try:
            params = read_scan_params(scan_dir)
        except Exception:
            continue
        if params.get('Method') != 'Bruker:RARE':
            continue
        try:
            n_slices = int(params['PVM_SPackArrNSlices'][0])
            matrix = int(params['PVM_Matrix'][0])
        except (KeyError, IndexError, TypeError):
            continue
        if n_slices < 10:
            continue
        score = (n_slices, matrix)
        if best is None or score > best[0]:
            best = (score, scan_dir.name)
    return best[1] if best else None


def find_tissue_maps(study_root: Path, subject: str, session: str) -> Optional[Dict[str, Path]]:
    """Locate the T2w tissue probability maps written by anatomical preprocessing."""
    anat_dir = study_root / 'derivatives' / subject / session / 'anat'
    maps = {
        label: anat_dir / f'{subject}_{session}_label-{label}_probseg.nii.gz'
        for label in ('GM', 'WM', 'CSF')
    }
    return maps if all(path.exists() for path in maps.values()) else None


def find_anat_image(study_root: Path, subject: str, session: str) -> Optional[Path]:
    """Locate the converted T2w the tissue maps are defined on."""
    anat_dir = study_root / 'derivatives' / subject / session / 'anat'
    for pattern in (f'{subject}_{session}*_T2w.nii.gz', f'{subject}_{session}*T2w*.nii.gz'):
        matches = sorted(anat_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def discover_sessions(bruker_root: Path, study_root: Path) -> List[Dict[str, Any]]:
    """Build the work list: one entry per Bruker session holding a PRESS scan."""
    sessions: List[Dict[str, Any]] = []

    for session_dir in sorted(p for p in bruker_root.iterdir() if p.is_dir()):
        labels = parse_session_name(session_dir.name)
        if labels is None:
            logger.warning("Could not parse subject/session from %s", session_dir.name)
            continue

        press = find_press_scans(session_dir)
        acquisitions = [scan for scan in press if scan['water_suppressed']]
        if not acquisitions:
            logger.warning(
                "%s %s: %d PRESS scans but none water-suppressed; skipping",
                labels['subject'], labels['session'], len(press),
            )
            continue
        chosen = max(acquisitions, key=lambda r: (r['n_averages'], r['scan_number']))

        anat_scan = find_anat_scan(session_dir)
        tissue_maps = find_tissue_maps(study_root, labels['subject'], labels['session'])
        anat_image = find_anat_image(study_root, labels['subject'], labels['session'])

        sessions.append({
            **labels,
            'session_dir': session_dir,
            'svs_scan': chosen['scan_number'],
            'protocol': chosen['protocol'],
            'n_averages': chosen['n_averages'],
            'echo_time_ms': chosen['echo_time'],
            'voxel_size_mm': chosen['voxel_size'],
            'anat_scan': anat_scan,
            'anat_image': anat_image,
            'tissue_maps': tissue_maps,
        })

    return sessions


def process_one(entry: Dict[str, Any], config_path: Path, study_root: Path,
                basis: Optional[Path]) -> Dict[str, Any]:
    """Process a single session. Exceptions are captured, not raised."""
    config = load_config(config_path)
    try:
        result = run_mrs_preprocessing(
            config=config,
            subject=entry['subject'],
            session=entry['session'],
            session_dir=entry['session_dir'],
            output_dir=study_root,
            basis=basis,
            anat_scan=entry['anat_scan'],
            anat_image=entry['anat_image'],
            tissue_maps=entry['tissue_maps'],
        )
        if result is None:
            return {**_key(entry), 'status': 'skipped', 'reason': 'no SVS scan'}
        return {
            **_key(entry),
            'status': 'ok',
            'summary': str(result['summary']),
            'internal_ref': '+'.join(result['metadata']['internal_ref']),
            'snr': result.get('qc', {}).get('metrics', {}).get('snr'),
            'fwhm_hz': result.get('qc', {}).get('metrics', {}).get('fwhm_hz'),
            'qc_pass': result.get('qc', {}).get('metrics', {}).get('overall_pass'),
        }
    except Exception as exc:
        logger.error("%s %s failed: %s", entry['subject'], entry['session'], exc)
        # Full tracebacks go to a per-session file: embedding them in the CSV
        # puts newlines and quotes in a cell and makes the table unreadable.
        log_dir = study_root / 'derivatives' / 'batch_logs' / 'mrs_failures'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{entry['subject']}_{entry['session']}.log"
        log_file.write_text(traceback.format_exc())
        return {
            **_key(entry),
            'status': 'failed',
            'reason': ' '.join(str(exc).split())[:300],
            'log': str(log_file),
        }


def _key(entry: Dict[str, Any]) -> Dict[str, str]:
    return {'subject': entry['subject'], 'session': entry['session']}


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Batch Bruker PRESS spectroscopy processing with FSL-MRS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('bruker_root', type=Path,
                        help='Directory of raw Bruker session directories')
    parser.add_argument('study_root', type=Path,
                        help='Study root; outputs go to derivatives/ and qc/')
    parser.add_argument('--config', type=Path, required=True, help='Config YAML')
    parser.add_argument('--basis', type=Path, default=None,
                        help='FSL-MRS basis set (overrides spectroscopy.basis)')
    parser.add_argument('--subjects', nargs='+', default=None,
                        help='Limit to these subjects (e.g. sub-Rat1Y)')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Parallel sessions (each fit uses several cores)')
    parser.add_argument('--dry-run', action='store_true',
                        help='List what would be processed and exit')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)-7s %(message)s',
    )

    if not args.bruker_root.is_dir():
        parser.error(f"Bruker root not found: {args.bruker_root}")

    sessions = discover_sessions(args.bruker_root, args.study_root)
    if args.subjects:
        wanted = set(args.subjects)
        sessions = [s for s in sessions if s['subject'] in wanted]

    print(f"\nFound {len(sessions)} sessions with a water-suppressed PRESS scan")
    inventory = pd.DataFrame([
        {
            'subject': s['subject'], 'session': s['session'],
            'svs_scan': s['svs_scan'], 'protocol': s['protocol'],
            'n_averages': s['n_averages'], 'TE_ms': s['echo_time_ms'],
            'voxel_mm': ' x '.join(f'{v:g}' for v in s['voxel_size_mm']),
            'anat_scan': s['anat_scan'] or '-',
            'tissue_maps': 'yes' if s['tissue_maps'] else 'no',
        }
        for s in sessions
    ])
    if not inventory.empty:
        print(inventory.to_string(index=False))
        n_measured = int((inventory['tissue_maps'] == 'yes').sum())
        print(f"\n{n_measured}/{len(inventory)} sessions have a T2w segmentation; "
              f"the rest will use default tissue fractions.")

    if args.dry_run:
        return 0
    if not sessions:
        return 1

    args.study_root.mkdir(parents=True, exist_ok=True)
    started = datetime.now()
    results: List[Dict[str, Any]] = []

    if args.n_jobs > 1:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as pool:
            futures = {
                pool.submit(process_one, entry, args.config, args.study_root, args.basis): entry
                for entry in sessions
            }
            for future in as_completed(futures):
                results.append(future.result())
    else:
        for entry in sessions:
            results.append(process_one(entry, args.config, args.study_root, args.basis))

    summary = pd.DataFrame(results).sort_values(['subject', 'session'])
    out_dir = args.study_root / 'derivatives' / 'batch_logs'
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = started.strftime('%Y%m%d_%H%M%S')
    summary.to_csv(out_dir / f'mrs_batch_{stamp}.csv', index=False)

    counts = summary['status'].value_counts().to_dict()
    print(f"\n{'=' * 60}")
    print(f"MRS batch finished in {datetime.now() - started}")
    for status in ('ok', 'skipped', 'failed'):
        print(f"  {status:8s}: {counts.get(status, 0)}")
    if counts.get('failed'):
        print("\nFailures:")
        for _, row in summary[summary['status'] == 'failed'].iterrows():
            print(f"  {row['subject']} {row['session']}: {row['reason']}")
    print(f"\nSummary: {out_dir / f'mrs_batch_{stamp}.csv'}")

    # A single tidy table across sessions is what group analysis actually needs.
    frames = [
        pd.read_csv(row['summary'])
        for _, row in summary[summary['status'] == 'ok'].iterrows()
    ]
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined_file = args.study_root / 'network' / 'mrs' / 'mrs_metabolites_long.csv'
        combined_file.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(combined_file, index=False)
        print(f"Combined metabolite table: {combined_file}")

    with open(out_dir / f'mrs_batch_{stamp}.json', 'w') as handle:
        json.dump({
            'started': started.isoformat(),
            'finished': datetime.now().isoformat(),
            'bruker_root': str(args.bruker_root),
            'study_root': str(args.study_root),
            'config': str(args.config),
            'basis': str(args.basis) if args.basis else None,
            'counts': counts,
        }, handle, indent=2)

    return 0 if not counts.get('failed') else 1


if __name__ == '__main__':
    sys.exit(main())
