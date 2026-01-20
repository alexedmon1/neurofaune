"""
Study initialization module.

This module provides functions to initialize a new neuroimaging study,
including BIDS data discovery, config generation, directory structure
setup, and atlas configuration.
"""

import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import yaml

from neurofaune.config import load_config, validate_config


@dataclass
class ScanInfo:
    """Information about a single scan."""
    file_path: Path
    modality: str
    suffix: str  # e.g., 'T2w', 'dwi', 'bold'
    run: Optional[str] = None
    acquisition: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            'file_path': str(self.file_path),
            'modality': self.modality,
            'suffix': self.suffix,
            'run': self.run,
            'acquisition': self.acquisition
        }


@dataclass
class SessionInfo:
    """Information about a session."""
    session: str
    cohort: str  # Derived from session name (ses-p60 -> p60)
    modalities: Dict[str, List[ScanInfo]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'session': self.session,
            'cohort': self.cohort,
            'modalities': {
                mod: [s.to_dict() for s in scans]
                for mod, scans in self.modalities.items()
            }
        }


@dataclass
class SubjectInfo:
    """Information about a subject."""
    subject: str
    sessions: Dict[str, SessionInfo] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'subject': self.subject,
            'sessions': {
                ses: info.to_dict()
                for ses, info in self.sessions.items()
            }
        }


@dataclass
class BIDSManifest:
    """Complete manifest of BIDS data."""
    bids_root: Path
    subjects: Dict[str, SubjectInfo] = field(default_factory=dict)
    issues: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def n_subjects(self) -> int:
        return len(self.subjects)

    @property
    def n_sessions(self) -> int:
        return sum(len(s.sessions) for s in self.subjects.values())

    def get_cohort_breakdown(self) -> Dict[str, int]:
        """Count subjects per cohort."""
        cohorts = {}
        for subj in self.subjects.values():
            for ses_info in subj.sessions.values():
                cohort = ses_info.cohort
                cohorts[cohort] = cohorts.get(cohort, 0) + 1
        return cohorts

    def get_modality_breakdown(self) -> Dict[str, int]:
        """Count sessions with each modality."""
        modalities = {}
        for subj in self.subjects.values():
            for ses_info in subj.sessions.values():
                for mod in ses_info.modalities.keys():
                    modalities[mod] = modalities.get(mod, 0) + 1
        return modalities

    def to_dict(self) -> dict:
        return {
            'bids_root': str(self.bids_root),
            'summary': {
                'n_subjects': self.n_subjects,
                'n_sessions': self.n_sessions,
                'cohort_breakdown': self.get_cohort_breakdown(),
                'modality_breakdown': self.get_modality_breakdown()
            },
            'subjects': {
                subj: info.to_dict()
                for subj, info in self.subjects.items()
            },
            'issues': self.issues
        }


def discover_bids_data(bids_root: Path) -> BIDSManifest:
    """
    Scan BIDS directory and return complete inventory.

    Parameters
    ----------
    bids_root : Path
        Root of BIDS directory

    Returns
    -------
    BIDSManifest
        Complete inventory of subjects, sessions, and scans
    """
    manifest = BIDSManifest(bids_root=bids_root)

    # Define modality patterns
    modality_patterns = {
        'anat': ['*_T2w.nii.gz', '*_T2w.nii', '*_T1w.nii.gz', '*_T1w.nii'],
        'dwi': ['*_dwi.nii.gz', '*_dwi.nii'],
        'func': ['*_bold.nii.gz', '*_bold.nii'],
        'fmap': ['*_epi.nii.gz', '*_magnitude*.nii.gz', '*_phasediff.nii.gz'],
    }

    # Find all subject directories
    subject_dirs = sorted(bids_root.glob('sub-*'))

    for subject_dir in subject_dirs:
        if not subject_dir.is_dir():
            continue

        subject = subject_dir.name
        subject_info = SubjectInfo(subject=subject)

        # Find all session directories
        session_dirs = sorted(subject_dir.glob('ses-*'))

        if not session_dirs:
            # No session structure - treat as single session
            manifest.issues.append({
                'subject': subject,
                'issue': 'No session directories found',
                'severity': 'warning'
            })
            continue

        for session_dir in session_dirs:
            if not session_dir.is_dir():
                continue

            session = session_dir.name

            # Extract cohort from session name (ses-p60 -> p60)
            cohort = session.replace('ses-', '')

            session_info = SessionInfo(session=session, cohort=cohort)

            # Check each modality
            for modality, patterns in modality_patterns.items():
                modality_dir = session_dir / modality

                if not modality_dir.exists():
                    continue

                scans = []
                for pattern in patterns:
                    for scan_file in modality_dir.glob(pattern):
                        # Parse BIDS filename
                        scan_info = _parse_bids_filename(scan_file, modality)
                        if scan_info:
                            scans.append(scan_info)

                if scans:
                    session_info.modalities[modality] = scans

            if session_info.modalities:
                subject_info.sessions[session] = session_info
            else:
                manifest.issues.append({
                    'subject': subject,
                    'session': session,
                    'issue': 'No valid scans found',
                    'severity': 'warning'
                })

        if subject_info.sessions:
            manifest.subjects[subject] = subject_info

    return manifest


def _parse_bids_filename(file_path: Path, modality: str) -> Optional[ScanInfo]:
    """Parse BIDS filename to extract metadata."""
    name = file_path.name

    # Remove extension
    if name.endswith('.nii.gz'):
        name = name[:-7]
    elif name.endswith('.nii'):
        name = name[:-4]
    else:
        return None

    parts = name.split('_')

    # Last part is suffix (T2w, dwi, bold, etc.)
    suffix = parts[-1]

    # Parse optional fields
    run = None
    acquisition = None

    for part in parts:
        if part.startswith('run-'):
            run = part.replace('run-', '')
        elif part.startswith('acq-'):
            acquisition = part.replace('acq-', '')

    return ScanInfo(
        file_path=file_path,
        modality=modality,
        suffix=suffix,
        run=run,
        acquisition=acquisition
    )


@dataclass
class BrukerScanInfo:
    """Information about a Bruker scan."""
    scan_dir: Path
    scan_number: str
    scan_name: str
    method: Optional[str] = None
    protocol: Optional[str] = None
    n_slices: Optional[int] = None
    n_repetitions: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            'scan_dir': str(self.scan_dir),
            'scan_number': self.scan_number,
            'scan_name': self.scan_name,
            'method': self.method,
            'protocol': self.protocol,
            'n_slices': self.n_slices,
            'n_repetitions': self.n_repetitions
        }


@dataclass
class BrukerSessionInfo:
    """Information about a Bruker session/experiment."""
    session_dir: Path
    subject: Optional[str] = None
    session: Optional[str] = None
    date: Optional[str] = None
    scans: List[BrukerScanInfo] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'session_dir': str(self.session_dir),
            'subject': self.subject,
            'session': self.session,
            'date': self.date,
            'scans': [s.to_dict() for s in self.scans]
        }


@dataclass
class BrukerManifest:
    """Complete manifest of Bruker raw data."""
    bruker_root: Path
    sessions: List[BrukerSessionInfo] = field(default_factory=list)
    issues: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def n_sessions(self) -> int:
        return len(self.sessions)

    @property
    def n_scans(self) -> int:
        return sum(len(s.scans) for s in self.sessions)

    def get_scan_types(self) -> Dict[str, int]:
        """Count scans by method/type."""
        types = {}
        for session in self.sessions:
            for scan in session.scans:
                method = scan.method or 'unknown'
                types[method] = types.get(method, 0) + 1
        return types

    def to_dict(self) -> dict:
        return {
            'bruker_root': str(self.bruker_root),
            'summary': {
                'n_sessions': self.n_sessions,
                'n_scans': self.n_scans,
                'scan_types': self.get_scan_types()
            },
            'sessions': [s.to_dict() for s in self.sessions],
            'issues': self.issues
        }


def discover_bruker_data(bruker_root: Path) -> BrukerManifest:
    """
    Scan Bruker directory and return inventory of raw data.

    Bruker data is organized as:
    bruker_root/
        CohortX/
            session_folder/  (contains subject, date info in name)
                1/  (scan number)
                    method, acqp, pdata/1/2dseq
                2/
                ...

    Parameters
    ----------
    bruker_root : Path
        Root of Bruker data directory

    Returns
    -------
    BrukerManifest
        Inventory of all Bruker sessions and scans
    """
    manifest = BrukerManifest(bruker_root=bruker_root)

    if not bruker_root.exists():
        manifest.issues.append({
            'issue': f'Bruker root not found: {bruker_root}',
            'severity': 'error'
        })
        return manifest

    # Look for session directories (contain numbered scan folders with 'method' files)
    # Try multiple organizational patterns
    session_candidates = []

    # Pattern 1: CohortX/session_folder/
    for cohort_dir in bruker_root.glob('Cohort*'):
        if cohort_dir.is_dir():
            for session_dir in cohort_dir.iterdir():
                if session_dir.is_dir():
                    session_candidates.append(session_dir)

    # Pattern 2: Direct session folders
    for item in bruker_root.iterdir():
        if item.is_dir() and not item.name.startswith('Cohort'):
            # Check if this looks like a session (has numbered scan folders)
            numbered_dirs = [d for d in item.iterdir() if d.is_dir() and d.name.isdigit()]
            if numbered_dirs:
                session_candidates.append(item)

    # Process each session
    for session_dir in session_candidates:
        session_info = _parse_bruker_session(session_dir)
        if session_info and session_info.scans:
            manifest.sessions.append(session_info)

    return manifest


def _parse_bruker_session(session_dir: Path) -> Optional[BrukerSessionInfo]:
    """Parse a Bruker session directory."""
    import re

    session_info = BrukerSessionInfo(session_dir=session_dir)

    # Try to parse subject/session from directory name
    dir_name = session_dir.name

    # Common patterns:
    # 20230712_..._Rat110_1_...
    # IRC1200_Cohort7_Rat102_1__Rat_102__p60_1_1_20250407
    # p60_July2023/20230712_...

    # Extract subject (RatXXX pattern)
    rat_match = re.search(r'Rat(\d+)', dir_name)
    if rat_match:
        session_info.subject = f"sub-Rat{rat_match.group(1)}"

    # Extract session/cohort (p30, p60, p90 pattern)
    cohort_match = re.search(r'[_/](p\d+)[_/]', dir_name) or re.search(r'^(p\d+)', dir_name)
    if cohort_match:
        session_info.session = f"ses-{cohort_match.group(1)}"

    # Extract date (YYYYMMDD pattern)
    date_match = re.search(r'(\d{8})', dir_name)
    if date_match:
        session_info.date = date_match.group(1)

    # Find all scan directories (numbered folders with method file)
    for item in sorted(session_dir.iterdir()):
        if item.is_dir() and item.name.isdigit():
            method_file = item / 'method'
            if method_file.exists():
                scan_info = _parse_bruker_scan(item)
                if scan_info:
                    session_info.scans.append(scan_info)

    return session_info if session_info.scans else None


def _parse_bruker_scan(scan_dir: Path) -> Optional[BrukerScanInfo]:
    """Parse a Bruker scan directory."""
    scan_info = BrukerScanInfo(
        scan_dir=scan_dir,
        scan_number=scan_dir.name,
        scan_name=scan_dir.name
    )

    # Read method file for scan parameters
    method_file = scan_dir / 'method'
    if method_file.exists():
        try:
            with open(method_file, 'r', errors='ignore') as f:
                content = f.read()

            # Extract method name
            method_match = re.search(r'##\$Method=<([^>]+)>', content)
            if method_match:
                scan_info.method = method_match.group(1)

            # Extract protocol name
            protocol_match = re.search(r'##\$PVM_ScanTimeStr=\s*<([^>]+)>', content)
            if protocol_match:
                scan_info.protocol = protocol_match.group(1)

            # Extract number of slices
            slices_match = re.search(r'##\$PVM_SPackArrNSlices=\s*\(\s*\d+\s*\)\s*(\d+)', content)
            if slices_match:
                scan_info.n_slices = int(slices_match.group(1))

            # Extract repetitions (for fMRI)
            reps_match = re.search(r'##\$PVM_NRepetitions=\s*(\d+)', content)
            if reps_match:
                scan_info.n_repetitions = int(reps_match.group(1))

        except Exception:
            pass

    # Try to get scan name from visu_pars
    visu_pars = scan_dir / 'pdata' / '1' / 'visu_pars'
    if visu_pars.exists():
        try:
            with open(visu_pars, 'r', errors='ignore') as f:
                content = f.read()

            # Extract scan name
            name_match = re.search(r'##\$VisuAcquisitionProtocol=\s*\(\s*\d+\s*\)\s*<([^>]+)>', content)
            if name_match:
                scan_info.scan_name = name_match.group(1)
        except Exception:
            pass

    return scan_info


def create_study_directories(study_root: Path, create_all: bool = False) -> Dict[str, Path]:
    """
    Create the study directory structure.

    Parameters
    ----------
    study_root : Path
        Root directory for the study
    create_all : bool
        If True, create all directories. If False, only create essential ones.

    Returns
    -------
    Dict[str, Path]
        Dictionary mapping directory names to paths
    """
    directories = {
        'study_root': study_root,
        'raw': study_root / 'raw',
        'bids': study_root / 'raw' / 'bids',
        'derivatives': study_root / 'derivatives',
        'templates': study_root / 'templates',
        'atlas': study_root / 'atlas',
        'transforms': study_root / 'transforms',
        'qc': study_root / 'qc',
        'work': study_root / 'work',
    }

    # Template subdirectories
    template_modalities = ['anat', 'dwi', 'func']
    cohorts = ['p30', 'p60', 'p90']

    for modality in template_modalities:
        for cohort in cohorts:
            key = f'templates_{modality}_{cohort}'
            directories[key] = study_root / 'templates' / modality / cohort

    # Create directories
    essential_dirs = ['study_root', 'raw', 'derivatives', 'templates', 'atlas',
                      'transforms', 'qc', 'work']

    for name, path in directories.items():
        if create_all or name in essential_dirs:
            path.mkdir(parents=True, exist_ok=True)

    return directories


def generate_config(
    study_root: Path,
    study_name: str,
    study_code: str,
    bids_root: Optional[Path] = None,
    sigma_atlas_path: Optional[Path] = None,
    species: str = 'rat',
    strain: str = 'Wistar',
    cohorts: Optional[Dict[str, Dict]] = None,
    output_path: Optional[Path] = None
) -> Path:
    """
    Generate a study configuration file.

    Parameters
    ----------
    study_root : Path
        Root directory for the study
    study_name : str
        Human-readable study name
    study_code : str
        Short study code (e.g., 'bpa-rat')
    bids_root : Path, optional
        Path to BIDS data (defaults to study_root/raw/bids)
    sigma_atlas_path : Path, optional
        Path to SIGMA atlas
    species : str
        Species (default: 'rat')
    strain : str
        Strain (default: 'Wistar')
    cohorts : dict, optional
        Cohort definitions (default: p30, p60, p90)
    output_path : Path, optional
        Where to save config (defaults to study_root/config.yaml)

    Returns
    -------
    Path
        Path to generated config file
    """
    if bids_root is None:
        bids_root = study_root / 'raw' / 'bids'

    if cohorts is None:
        cohorts = {
            'p30': {'age_days': 30, 'description': 'Postnatal day 30'},
            'p60': {'age_days': 60, 'description': 'Postnatal day 60'},
            'p90': {'age_days': 90, 'description': 'Postnatal day 90'},
        }

    config = {
        'study': {
            'name': study_name,
            'code': study_code,
            'species': species,
            'strain': strain,
            'created': datetime.now().isoformat(),
        },
        'paths': {
            'study_root': str(study_root),
            'raw': str(study_root / 'raw'),
            'bids': str(bids_root),
            'derivatives': str(study_root / 'derivatives'),
            'templates': str(study_root / 'templates'),
            'atlas': str(study_root / 'atlas'),
            'transforms': str(study_root / 'transforms'),
            'qc': str(study_root / 'qc'),
            'work': str(study_root / 'work'),
        },
        'cohorts': cohorts,
        'atlas': {
            'name': 'SIGMA',
            'base_path': str(sigma_atlas_path) if sigma_atlas_path else '${HOME}/atlases/SIGMA',
            'study_space': str(study_root / 'atlas' / 'SIGMA_study_space'),
        },
        'anatomical': {
            'modality': 'T2w',
            'bet': {'frac': 0.3, 'robust': True},
            'n4': {'iterations': [50, 50, 30, 20]},
        },
        'diffusion': {
            'bet': {'frac': 0.3},
            'eddy': {
                'slice_padding': 2,
                'repol': True,
            },
        },
        'functional': {
            'bet': {'frac': 0.3},
            'smoothing_fwhm': 0.8,
        },
        'execution': {
            'plugin': 'Linear',
            'n_procs': 4,
        },
    }

    if output_path is None:
        output_path = study_root / 'config.yaml'

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Generated config: {output_path}")

    return output_path


def setup_study(
    study_root: Path,
    study_name: str,
    study_code: str,
    bids_root: Optional[Path] = None,
    bruker_root: Optional[Path] = None,
    sigma_atlas_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
    link_bids: bool = True,
    link_bruker: bool = True,
    setup_atlas: bool = True,
    force: bool = False
) -> Dict[str, Any]:
    """
    Initialize a complete study with directory structure, config, and atlas.

    This is the main entry point for setting up a new study. It:
    1. Creates the directory structure
    2. Discovers raw data (Bruker) and/or BIDS data
    3. Generates or validates configuration
    4. Sets up the study-space atlas
    5. Returns a comprehensive report

    Parameters
    ----------
    study_root : Path
        Root directory for the study
    study_name : str
        Human-readable study name
    study_code : str
        Short study code (e.g., 'bpa-rat')
    bids_root : Path, optional
        Path to BIDS data. If None, looks for study_root/raw/bids
    bruker_root : Path, optional
        Path to raw Bruker data. If None, looks for study_root/raw/bruker
    sigma_atlas_path : Path, optional
        Path to SIGMA atlas base directory
    config_path : Path, optional
        Path to existing config file. If provided, will validate instead of generate.
    link_bids : bool
        If True and bids_root is outside study_root, create a symlink
    link_bruker : bool
        If True and bruker_root is outside study_root, create a symlink
    setup_atlas : bool
        If True, set up the study-space atlas from SIGMA
    force : bool
        If True, overwrite existing files/directories

    Returns
    -------
    Dict[str, Any]
        Initialization report with status of each step
    """
    study_root = Path(study_root).resolve()

    report = {
        'status': 'success',
        'study_root': str(study_root),
        'timestamp': datetime.now().isoformat(),
        'steps': {},
        'bids_manifest': None,
        'bruker_manifest': None,
        'next_steps': [],
    }

    print("=" * 70)
    print(f"Initializing Study: {study_name}")
    print(f"Study Root: {study_root}")
    print("=" * 70)

    # Step 1: Create directory structure
    print("\n[1/6] Creating directory structure...")
    try:
        directories = create_study_directories(study_root, create_all=False)
        report['steps']['directories'] = {
            'status': 'success',
            'created': [str(p) for p in directories.values() if p.exists()]
        }
        print(f"  Created {len(directories)} directories")
    except Exception as e:
        report['steps']['directories'] = {'status': 'failed', 'error': str(e)}
        report['status'] = 'failed'
        return report

    # Step 2: Discover raw Bruker data
    print("\n[2/6] Discovering raw Bruker data...")
    if bruker_root is None:
        bruker_root = study_root / 'raw' / 'bruker'
    else:
        bruker_root = Path(bruker_root).resolve()

    # Create symlink if needed
    target_bruker = study_root / 'raw' / 'bruker'
    if bruker_root != target_bruker and link_bruker:
        if target_bruker.exists():
            if target_bruker.is_symlink():
                if force:
                    target_bruker.unlink()
                else:
                    print(f"  Bruker symlink already exists: {target_bruker}")

        if not target_bruker.exists() and bruker_root.exists():
            target_bruker.symlink_to(bruker_root)
            print(f"  Created symlink: {target_bruker} -> {bruker_root}")

    # Discover Bruker data
    if bruker_root.exists():
        bruker_manifest = discover_bruker_data(bruker_root)
        report['bruker_manifest'] = bruker_manifest.to_dict()
        report['steps']['bruker_discovery'] = {
            'status': 'success',
            'n_sessions': bruker_manifest.n_sessions,
            'n_scans': bruker_manifest.n_scans,
            'scan_types': bruker_manifest.get_scan_types(),
            'issues': len(bruker_manifest.issues)
        }
        print(f"  Found {bruker_manifest.n_sessions} sessions, {bruker_manifest.n_scans} scans")
        if bruker_manifest.get_scan_types():
            print(f"  Scan types: {bruker_manifest.get_scan_types()}")
    else:
        report['steps']['bruker_discovery'] = {
            'status': 'skipped',
            'reason': f'Bruker directory not found: {bruker_root}'
        }
        print(f"  Bruker directory not found: {bruker_root}")

    # Step 3: Handle BIDS data
    print("\n[3/6] Discovering BIDS data...")
    if bids_root is None:
        bids_root = study_root / 'raw' / 'bids'
    else:
        bids_root = Path(bids_root).resolve()

    # Create symlink if needed
    target_bids = study_root / 'raw' / 'bids'
    if bids_root != target_bids and link_bids:
        if target_bids.exists():
            if target_bids.is_symlink():
                if force:
                    target_bids.unlink()
                else:
                    print(f"  BIDS symlink already exists: {target_bids}")

        if not target_bids.exists() and bids_root.exists():
            target_bids.symlink_to(bids_root)
            print(f"  Created symlink: {target_bids} -> {bids_root}")

    # Discover BIDS data
    if bids_root.exists():
        manifest = discover_bids_data(bids_root)
        report['bids_manifest'] = manifest.to_dict()
        report['steps']['bids_discovery'] = {
            'status': 'success',
            'n_subjects': manifest.n_subjects,
            'n_sessions': manifest.n_sessions,
            'cohorts': manifest.get_cohort_breakdown(),
            'modalities': manifest.get_modality_breakdown(),
            'issues': len(manifest.issues)
        }
        print(f"  Found {manifest.n_subjects} subjects, {manifest.n_sessions} sessions")
        print(f"  Cohorts: {manifest.get_cohort_breakdown()}")
        print(f"  Modalities: {manifest.get_modality_breakdown()}")
        if manifest.issues:
            print(f"  Warnings: {len(manifest.issues)}")
    else:
        report['steps']['bids_discovery'] = {
            'status': 'skipped',
            'reason': f'BIDS directory not found: {bids_root}'
        }
        print(f"  BIDS directory not found: {bids_root}")
        manifest = None

    # Step 4: Generate or validate config
    print("\n[4/6] Setting up configuration...")
    if config_path and Path(config_path).exists():
        # Validate existing config
        try:
            config = load_config(Path(config_path))
            report['steps']['config'] = {
                'status': 'success',
                'action': 'validated',
                'path': str(config_path)
            }
            print(f"  Validated existing config: {config_path}")
        except Exception as e:
            report['steps']['config'] = {
                'status': 'failed',
                'action': 'validation',
                'error': str(e)
            }
            report['status'] = 'partial'
            print(f"  Config validation failed: {e}")
    else:
        # Generate new config
        config_output = study_root / 'config.yaml'
        if config_output.exists() and not force:
            report['steps']['config'] = {
                'status': 'skipped',
                'reason': 'Config already exists (use force=True to overwrite)',
                'path': str(config_output)
            }
            print(f"  Config already exists: {config_output}")
        else:
            try:
                config_path = generate_config(
                    study_root=study_root,
                    study_name=study_name,
                    study_code=study_code,
                    bids_root=bids_root,
                    sigma_atlas_path=sigma_atlas_path,
                    output_path=config_output
                )
                report['steps']['config'] = {
                    'status': 'success',
                    'action': 'generated',
                    'path': str(config_path)
                }
            except Exception as e:
                report['steps']['config'] = {
                    'status': 'failed',
                    'error': str(e)
                }
                report['status'] = 'partial'

    # Step 5: Set up atlas
    print("\n[5/6] Setting up study-space atlas...")
    if setup_atlas and sigma_atlas_path:
        sigma_atlas_path = Path(sigma_atlas_path)
        if sigma_atlas_path.exists():
            try:
                from neurofaune.templates.slice_registration import setup_study_atlas

                atlas_output = study_root / 'atlas' / 'SIGMA_study_space'

                if atlas_output.exists() and not force:
                    report['steps']['atlas'] = {
                        'status': 'skipped',
                        'reason': 'Atlas already exists (use force=True to overwrite)',
                        'path': str(atlas_output)
                    }
                    print(f"  Atlas already exists: {atlas_output}")
                else:
                    setup_study_atlas(
                        sigma_base_path=sigma_atlas_path,
                        study_atlas_dir=atlas_output,
                        config_path=study_root / 'config.yaml'
                    )
                    report['steps']['atlas'] = {
                        'status': 'success',
                        'path': str(atlas_output)
                    }
                    print(f"  Created study-space atlas: {atlas_output}")
            except ImportError:
                report['steps']['atlas'] = {
                    'status': 'skipped',
                    'reason': 'Atlas setup module not available'
                }
                print("  Atlas setup module not available")
            except Exception as e:
                report['steps']['atlas'] = {
                    'status': 'failed',
                    'error': str(e)
                }
                report['status'] = 'partial'
                print(f"  Atlas setup failed: {e}")
        else:
            report['steps']['atlas'] = {
                'status': 'skipped',
                'reason': f'SIGMA atlas not found: {sigma_atlas_path}'
            }
            print(f"  SIGMA atlas not found: {sigma_atlas_path}")
    else:
        report['steps']['atlas'] = {
            'status': 'skipped',
            'reason': 'Atlas setup disabled or no path provided'
        }
        print("  Atlas setup skipped")

    # Step 5: Save manifest and report
    print("\n[6/6] Saving study manifest...")
    try:
        manifest_path = study_root / 'study_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(report, f, indent=2)
        report['steps']['manifest'] = {
            'status': 'success',
            'path': str(manifest_path)
        }
        print(f"  Saved manifest: {manifest_path}")
    except Exception as e:
        report['steps']['manifest'] = {
            'status': 'failed',
            'error': str(e)
        }

    # Generate next steps
    report['next_steps'] = _generate_next_steps(report, study_root)

    # Print summary
    print("\n" + "=" * 70)
    print("Study Initialization Complete!")
    print("=" * 70)
    print(f"\nStatus: {report['status'].upper()}")

    if report['next_steps']:
        print("\nNext Steps:")
        for i, step in enumerate(report['next_steps'], 1):
            print(f"  {i}. {step}")

    return report


def _generate_next_steps(report: dict, study_root: Path) -> List[str]:
    """Generate recommended next steps based on initialization results."""
    steps = []

    # Check if Bruker data was found but not BIDS
    bruker_info = report['steps'].get('bruker_discovery', {})
    bids_info = report['steps'].get('bids_discovery', {})

    has_bruker = bruker_info.get('status') == 'success' and bruker_info.get('n_scans', 0) > 0
    has_bids = bids_info.get('status') == 'success' and bids_info.get('n_subjects', 0) > 0

    if has_bruker and not has_bids:
        # Need to convert Bruker to BIDS first
        steps.append(
            f"Convert Bruker data to BIDS format ({bruker_info.get('n_scans', 0)} scans found). "
            f"Use brkraw or similar tool."
        )
        steps.append(f"  Example: brkraw tonii {study_root}/raw/bruker -o {study_root}/raw/bids")

    if has_bids:
        modalities = bids_info.get('modalities', {})

        if modalities.get('anat', 0) > 0:
            steps.append(
                f"Run anatomical preprocessing ({modalities['anat']} sessions): "
                f"uv run python scripts/batch_preprocess_anat.py --config {study_root}/config.yaml"
            )

        if modalities.get('dwi', 0) > 0:
            steps.append(
                f"Run DTI preprocessing ({modalities['dwi']} sessions): "
                f"uv run python scripts/batch_preprocess_dwi.py --config {study_root}/config.yaml"
            )

        if modalities.get('func', 0) > 0:
            steps.append(
                f"Run functional preprocessing ({modalities['func']} sessions): "
                f"uv run python scripts/batch_preprocess_func.py --config {study_root}/config.yaml"
            )

        steps.append(
            "Build age-specific templates: "
            f"uv run python scripts/build_templates.py --config {study_root}/config.yaml"
        )
    elif not has_bruker:
        steps.append(f"Add raw data (Bruker) to: {study_root}/raw/bruker/")
        steps.append(f"Or add BIDS data to: {study_root}/raw/bids/")

    # Check atlas status
    atlas_info = report['steps'].get('atlas', {})
    if atlas_info.get('status') != 'success':
        steps.append("Set up SIGMA atlas (required for registration)")

    return steps


def get_study_subjects(
    study_root: Path,
    cohort: Optional[str] = None,
    modality: Optional[str] = None,
    exclude_failed: bool = True
) -> List[Dict[str, Any]]:
    """
    Get list of subjects ready for processing.

    Parameters
    ----------
    study_root : Path
        Study root directory
    cohort : str, optional
        Filter by cohort (e.g., 'p60')
    modality : str, optional
        Filter by modality (e.g., 'anat', 'dwi')
    exclude_failed : bool
        If True, exclude subjects with exclusion markers

    Returns
    -------
    List[Dict]
        List of subject info dicts with subject, session, available modalities
    """
    study_root = Path(study_root)
    bids_root = study_root / 'raw' / 'bids'

    if not bids_root.exists():
        return []

    manifest = discover_bids_data(bids_root)
    subjects = []

    for subj_id, subj_info in manifest.subjects.items():
        for ses_id, ses_info in subj_info.sessions.items():
            # Filter by cohort
            if cohort and ses_info.cohort != cohort:
                continue

            # Filter by modality
            if modality and modality not in ses_info.modalities:
                continue

            # Check for exclusion marker
            if exclude_failed:
                exclusion_file = (study_root / 'derivatives' / subj_id / ses_id /
                                 'anat' / '.preprocessing_failed')
                if exclusion_file.exists():
                    continue

            subjects.append({
                'subject': subj_id,
                'session': ses_id,
                'cohort': ses_info.cohort,
                'modalities': list(ses_info.modalities.keys())
            })

    return subjects


def print_study_summary(study_root: Path) -> None:
    """Print a summary of study status."""
    study_root = Path(study_root)

    print("=" * 70)
    print(f"Study Summary: {study_root.name}")
    print("=" * 70)

    # Check for manifest
    manifest_path = study_root / 'study_manifest.json'
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

        print(f"\nInitialized: {manifest.get('timestamp', 'Unknown')}")

        bids_info = manifest.get('steps', {}).get('bids_discovery', {})
        if bids_info.get('status') == 'success':
            print(f"\nBIDS Data:")
            print(f"  Subjects: {bids_info.get('n_subjects', 0)}")
            print(f"  Sessions: {bids_info.get('n_sessions', 0)}")
            print(f"  Cohorts: {bids_info.get('cohorts', {})}")
            print(f"  Modalities: {bids_info.get('modalities', {})}")
    else:
        print("\nStudy not initialized. Run setup_study() first.")
        return

    # Check preprocessing status
    derivatives_dir = study_root / 'derivatives'
    if derivatives_dir.exists():
        processed = list(derivatives_dir.glob('sub-*/ses-*'))
        print(f"\nPreprocessing Status:")
        print(f"  Processed sessions: {len(processed)}")

        # Count by modality
        for modality in ['anat', 'dwi', 'func']:
            count = len(list(derivatives_dir.glob(f'sub-*/ses-*/{modality}')))
            if count > 0:
                print(f"    {modality}: {count}")

    # Check templates
    templates_dir = study_root / 'templates'
    if templates_dir.exists():
        templates = list(templates_dir.glob('*/*/template.nii.gz'))
        if templates:
            print(f"\nTemplates Built: {len(templates)}")
            for t in templates:
                print(f"    {t.parent.parent.name}/{t.parent.name}")

    print("=" * 70)
